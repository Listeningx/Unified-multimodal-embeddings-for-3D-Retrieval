import torch
import logging
import re
import json
import os

from .distributed import is_master

# 检查环境变量是否禁用了 FusedAdam 和 FusedLAMB
_DS_BUILD_FUSED_ADAM = os.environ.get('DS_BUILD_FUSED_ADAM', '1')
_DS_BUILD_FUSED_LAMB = os.environ.get('DS_BUILD_FUSED_LAMB', '1')
_DISABLE_FUSED_OPTIMIZERS = (_DS_BUILD_FUSED_ADAM == '0' or _DS_BUILD_FUSED_LAMB == '0')

FusedAdam = None
FusedLAMB = None

if not _DISABLE_FUSED_OPTIMIZERS:
    # 仅在未禁用时尝试导入
    try:
        from apex.optimizers import FusedAdam
    except ImportError:
        # 静默失败，不打印警告（因为用户可能故意不使用 apex）
        pass

# 如果需要使用 FusedAdam 但未安装，可以在创建优化器时检查并给出明确提示

def get_num_layer_for_transformer(param_name, num_max_layer):
    layer_0 = {
        "patch_embed", 
        "pos_embed", 
        "cls_token", 
        "mask_token", 
        "conv1",
        "positional_embedding",
        "token_embedding",
        "transformer.embeddings.word_embeddings",
        "transformer.embeddings.position_embeddings",
        "transformer.embeddings.token_type_embeddings",
    }

    if any(l in param_name for l in layer_0):
        return 0

    block_regex = re.compile(r"blocks\.([0-9]+)\.")
    match_block = block_regex.search(param_name)

    #huggingface->text.transformer.encoder.layer
    layer_regex = re.compile(r"layer\.([0-9]+)\.") 
    match_layer = layer_regex.search(param_name)
    if match_block is not None:
        return int(match_block.group(1)) + 1
    elif match_layer is not None:
        return int(match_layer.group(1)) + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_transformer(var_name, len(self.values))

def get_parameters(args, model, assigner, tower):
    filter_parameters = []
    skip = set()
    if tower == 'visual':
        lr = args.visual_lr if args.visual_lr is not None else args.lr
        weight_decay = args.visual_wd if args.visual_wd is not None else args.wd
        filter_parameters = [[name, param] for name, param in model.named_parameters() if 'visual.' in name and 'point_encoder.' not in name]
        if hasattr(model, 'visual'):
            if hasattr(model.visual, 'no_weight_decay'):
                skip = set.union(skip, model.visual.no_weight_decay())
        skip = ['visual.' + n for n in skip]
    elif tower == 'text':
        lr = args.text_lr if args.text_lr is not None else args.lr
        weight_decay = args.text_wd if args.text_wd is not None else args.wd
        filter_parameters = [[name, param] for name, param in model.named_parameters() if 'text.' in name]
        if hasattr(model, 'text'):
            if hasattr(model.text, 'no_weight_decay'):
                skip = set.union(skip, model.text.no_weight_decay())
        skip = ['text.' + n for n in skip]
    elif tower == 'point':
        lr = args.point_lr if args.point_lr is not None else args.lr
        weight_decay = args.point_wd if args.point_wd is not None else args.wd
        filter_parameters = [[name, param] for name, param in model.named_parameters() if 'point_encoder.visual' in name]
        if hasattr(model, 'point_encoder'):
            if hasattr(model.point_encoder.visual, 'no_weight_decay'):
                # skip = set.union(skip, model.point_encoder.visual.no_weight_decay())
                skit =  set.union(skip, {'pos_embed', 'cls_token'})
        skip = ['point_encoder.visual.' + n for n in skip]
    else:
        lr = args.lr
        weight_decay = args.wd
        exclude = lambda n: 'visual.' not in n and 'text.' not in n and 'point_encoder.visual.' not in n
        filter_parameters = [[n, p] for n, p in model.named_parameters() if exclude(n)]
        if hasattr(model, 'no_weight_decay'):
            skip = set.union(skip, model.no_weight_decay())

    get_num_layer  = assigner.get_layer_id if assigner is not None else None
    get_layer_scale = assigner.get_scale if assigner is not None else None


    parameter_group_names = {}
    parameter_group_vars = {}
    for name, param in filter_parameters:
        if not param.requires_grad:
            continue

        # if param.ndim < 2 or "bn" in name or "ln" in name or "bias" in name or 'logit_scale' in name or name in skip:
        if param.ndim <= 1 or name.endswith(".bias") or name in skip:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = tower + "_" + "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "group": tower,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                "lr": lr
            }
            parameter_group_vars[group_name] = {
                "group": tower,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                "lr": lr,
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    if is_master(args, local=getattr(args, 'log_local', False)):
        logging.info(f"Tower = {tower}")
        logging.info(f"Skip weight decay name marked in tower-{tower}: {skip}")
        logging.info(f"Num of parameters group in tower-{tower}: {len(parameter_group_vars.values())}")
        logging.info(f"Param groups = {json.dumps(parameter_group_names, indent=2)}")
    return list(parameter_group_vars.values())


def get_assigner(args, model):
    visual_ld = args.visual_ld if args.visual_ld else args.ld
    text_ld = args.text_ld if args.text_ld else args.ld
    point_ld = args.point_ld if args.point_ld else args.ld

    
    if visual_ld < 1.0:
        visual_num_layers = model.visual.get_num_layers()
        assigner_visual = LayerDecayValueAssigner(list(visual_ld ** (visual_num_layers + 1 - i) for i in range(visual_num_layers + 2)))
    else:
        assigner_visual = None

    if text_ld < 1.0:
        text_num_layers = model.text.get_num_layers()
        assigner_text = LayerDecayValueAssigner(list(text_ld ** (text_num_layers + 1 - i) for i in range(text_num_layers + 2)))
    else:
        assigner_text = None

    if point_ld < 1.0:
        visual_num_layers =  len(model.point_encoder.visual.blocks)
        assigner_point = LayerDecayValueAssigner(list(point_ld ** (visual_num_layers + 1 - i) for i in range(visual_num_layers + 2)))
    else:
        visual_num_layers = len(model.point_encoder.visual.blocks)
        assigner_point = LayerDecayValueAssigner(list(point_ld ** (visual_num_layers + 1 - i) for i in range(visual_num_layers + 2)))

    if assigner_visual is not None:
        logging.info("Assigned visual values = %s" % str(assigner_visual.values))
    if assigner_text is not None:
        logging.info("Assigned text values = %s" % str(assigner_text.values))
    if assigner_point is not None:
        logging.info("Assigned point values = %s" % str(assigner_point.values))
    return assigner_visual, assigner_text, assigner_point

def get_all_parameters(args, model):
    assigner_visual, assigner_text, assiner_point = get_assigner(args, model)
        
    parameters = []
    visual_parameters = get_parameters(args, model, assigner_visual, 'visual')
    text_parameters = get_parameters(args, model, assigner_text, 'text')
    point_parameters = get_parameters(args, model, assiner_point, 'point')
    other_parameters = get_parameters(args, model, None, 'other')

    parameters.extend(visual_parameters)
    parameters.extend(text_parameters)
    parameters.extend(point_parameters)
    parameters.extend(other_parameters)

    if len(parameters) == 0:
        parameters = model.parameters()
    return parameters

def create_optimizer(args, model, return_params=False):
    optimizer_args = dict(
            betas=(args.beta1, args.beta2),
        )
    if args.optimizer != 'lion':
        optimizer_args['eps'] = args.eps
        
    if args.optimizer == 'fused_adam':
        base_optimizer = FusedAdam
    else:
        base_optimizer = torch.optim.AdamW

    parameters = get_all_parameters(args, model)

    optimizer = base_optimizer(parameters, **optimizer_args)

    if is_master(args, local=getattr(args, 'log_local', False)):
        logging.info(f'Optimizer: {args.optimizer}')
        logging.info(f'Optimizer config: {optimizer_args}')

    if return_params:
        return optimizer, parameters
    return optimizer

def get_loss_scale_for_deepspeed(model, compute_grad_norm=True):
    """
    获取 DeepSpeed 的 loss scale 和 global grad norm
    
    注意：应该在 model.step() 之后调用此函数，因为 grad_norm 是在 step() 内部计算的
    
    Args:
        model: DeepSpeed 包装的模型
        compute_grad_norm: 如果内置方式获取失败，是否手动计算 grad_norm
    
    Returns:
        tuple: (loss_scale, grad_norm)
    """
    optimizer = model.optimizer
    loss_scale = None
    grad_norm = 0.0
    
    # 获取 loss_scale
    if hasattr(optimizer, 'loss_scale'):
        loss_scale = optimizer.loss_scale
    elif hasattr(optimizer, 'cur_scale'):
        loss_scale = optimizer.cur_scale
    elif hasattr(optimizer, 'loss_scaler') and hasattr(optimizer.loss_scaler, 'loss_scale'):
        loss_scale = optimizer.loss_scaler.loss_scale
    # DeepSpeed FP16 优化器的 loss_scaler 可能在不同位置
    elif hasattr(model, 'loss_scale'):
        loss_scale = model.loss_scale()
    
    # 获取 grad_norm（按优先级尝试多种方式）
    grad_norm_found = False
    
    # 方式1: 从 optimizer 的 _global_grad_norm 属性获取（普通优化器）
    if hasattr(optimizer, '_global_grad_norm') and optimizer._global_grad_norm is not None:
        grad_norm = optimizer._global_grad_norm
        grad_norm_found = True
    
    # 方式2: 从 model 的 get_global_grad_norm 方法获取
    if not grad_norm_found and hasattr(model, 'get_global_grad_norm'):
        try:
            gn = model.get_global_grad_norm()
            if gn is not None and gn > 0:
                grad_norm = gn
                grad_norm_found = True
        except:
            pass
    
    # 方式3: ZeRO 优化器 - 检查 optimizer.optimizer（ZeRO 包装的内部优化器）
    if not grad_norm_found and hasattr(optimizer, 'optimizer'):
        inner_optimizer = optimizer.optimizer
        if hasattr(inner_optimizer, '_global_grad_norm') and inner_optimizer._global_grad_norm is not None:
            grad_norm = inner_optimizer._global_grad_norm
            grad_norm_found = True
    
    # 方式4: 从 DeepSpeed engine 的内部状态获取
    if not grad_norm_found:
        # DeepSpeed 可能将 grad_norm 存储在 engine 的不同位置
        for attr_name in ['_global_grad_norm', 'grad_norm', '_grad_norm']:
            if hasattr(model, attr_name):
                gn = getattr(model, attr_name)
                if gn is not None and (isinstance(gn, (int, float)) or hasattr(gn, 'item')):
                    grad_norm = gn
                    grad_norm_found = True
                    break
    
    # 方式5: 从 optimizer 的梯度裁剪状态获取
    if not grad_norm_found and hasattr(optimizer, 'grad_norm'):
        grad_norm = optimizer.grad_norm
        grad_norm_found = True
    
    # 方式6: 手动计算 grad_norm（作为最后手段）
    if not grad_norm_found and compute_grad_norm:
        try:
            # 获取模型参数
            if hasattr(model, 'module'):
                params = model.module.parameters()
            else:
                params = model.parameters()
            
            # 计算梯度范数
            total_norm_sq = 0.0
            for p in params:
                if p.grad is not None:
                    param_norm = p.grad.data.float().norm(2)
                    total_norm_sq += param_norm.item() ** 2
            
            if total_norm_sq > 0:
                grad_norm = total_norm_sq ** 0.5
                grad_norm_found = True
        except Exception as e:
            # 手动计算也失败了，保持 grad_norm = 0
            pass
    
    # 方式7: 检查 overflow 状态
    if not grad_norm_found and hasattr(optimizer, 'overflow') and optimizer.overflow:
        grad_norm = float('inf')  # 表示梯度溢出
    
    # 确保 grad_norm 是一个数值
    if grad_norm is None:
        grad_norm = 0.0
    elif hasattr(grad_norm, 'item'):
        grad_norm = grad_norm.item()
    
    # 确保是 float 类型
    try:
        grad_norm = float(grad_norm)
    except:
        grad_norm = 0.0
    
    return loss_scale, grad_norm

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == torch.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm.to(dtype=torch.float32)


# ============ 多模态训练专用函数 ============

def get_multimodal_parameters(args, model, model_type='multimodal'):
    """
    获取多模态训练的参数分组
    
    适用于 train_multimodal.py 中的 MultimodalTrainingWrapper 模型
    
    参数分组策略：
    1. point_encoder: 预训练模块，使用较小学习率 (point_lr * 0.1)
    2. 其他 uni3d 模块（融合块、投影层等）: 新初始化，使用较大学习率 (lr)
    3. logit_scale: 温度参数，使用最小学习率 (lr * 0.1)
    
    Args:
        args: 参数配置，需包含 lr, point_lr, wd, point_wd 等
        model: MultimodalTrainingWrapper 模型
        model_type: 模型类型，'multimodal' 或 'standard'
    
    Returns:
        list: 参数组列表，每个元素是 {'params': [...], 'lr': ..., 'weight_decay': ...}
    """
    from .distributed import is_master
    
    # 获取基础学习率和权重衰减
    lr = args.lr
    point_lr = getattr(args, 'point_lr', args.lr)
    wd = getattr(args, 'wd', 0.1)
    point_wd = getattr(args, 'point_wd', wd)
    
    # 获取模型（处理 DDP 包装）
    if hasattr(model, 'module'):
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    
    # 参数分组
    point_encoder_params = []
    other_uni3d_params = []
    logit_scale_params = []
    
    # 检查模型结构
    if hasattr(model_without_ddp, 'uni3d'):
        # MultimodalTrainingWrapper 结构
        uni3d = model_without_ddp.uni3d
        
        # 点云编码器参数
        if hasattr(uni3d, 'point_encoder'):
            for p in uni3d.point_encoder.parameters():
                if p.requires_grad:
                    point_encoder_params.append(p)
        
        # 其他 uni3d 模块参数（融合块、投影层、路由器等）
        for name, p in uni3d.named_parameters():
            if 'point_encoder' not in name and p.requires_grad:
                other_uni3d_params.append(p)
        
        # logit_scale 参数
        if hasattr(model_without_ddp, 'logit_scale'):
            logit_scale_params.append(model_without_ddp.logit_scale)
    else:
        # 直接是 Uni3DMultimodal 模型
        if hasattr(model_without_ddp, 'point_encoder'):
            for p in model_without_ddp.point_encoder.parameters():
                if p.requires_grad:
                    point_encoder_params.append(p)
        
        for name, p in model_without_ddp.named_parameters():
            if 'point_encoder' not in name and p.requires_grad:
                if 'logit_scale' in name:
                    logit_scale_params.append(p)
                else:
                    other_uni3d_params.append(p)
    
    # 构建参数组
    param_groups = []
    
    if point_encoder_params:
        param_groups.append({
            'params': point_encoder_params,
            'lr': point_lr * 0.1,  # 预训练模块使用较小学习率
            'weight_decay': point_wd,
            'group': 'point_encoder',
            'base_lr': point_lr * 0.1,
        })
    
    if other_uni3d_params:
        param_groups.append({
            'params': other_uni3d_params,
            'lr': lr,  # 新初始化模块使用较大学习率
            'weight_decay': wd,
            'group': 'uni3d_other',
            'base_lr': lr,
        })
    
    if logit_scale_params:
        param_groups.append({
            'params': logit_scale_params,
            'lr': lr * 0.1,  # 温度参数使用最小学习率
            'weight_decay': 0.0,  # 温度参数不使用权重衰减
            'group': 'logit_scale',
            'base_lr': lr * 0.1,
        })
    
    # 打印参数统计
    if is_master(args, local=getattr(args, 'log_local', False)):
        logging.info("=" * 50)
        logging.info("Multimodal Parameter Groups:")
        logging.info(f"  Point Encoder: {len(point_encoder_params)} params, lr={point_lr * 0.1}, wd={point_wd}")
        logging.info(f"  Uni3D Other: {len(other_uni3d_params)} params, lr={lr}, wd={wd}")
        logging.info(f"  Logit Scale: {len(logit_scale_params)} params, lr={lr * 0.1}, wd=0.0")
        
        total_params = sum(p.numel() for group in param_groups for p in group['params'])
        logging.info(f"  Total trainable parameters: {total_params / 1e6:.2f}M")
        logging.info("=" * 50)
    
    return param_groups


def create_multimodal_optimizer(args, model, return_params=False):
    """
    为多模态训练创建优化器
    
    Args:
        args: 参数配置
        model: MultimodalTrainingWrapper 模型
        return_params: 是否返回参数组
    
    Returns:
        optimizer: AdamW 优化器
        param_groups: (可选) 参数组列表
    """
    optimizer_args = dict(
        betas=(getattr(args, 'beta1', 0.9), getattr(args, 'beta2', 0.95)),
    )
    if getattr(args, 'optimizer', 'adamw') != 'lion':
        optimizer_args['eps'] = getattr(args, 'eps', 1e-8)
    
    # 获取参数组
    param_groups = get_multimodal_parameters(args, model)
    
    # 选择优化器
    if getattr(args, 'optimizer', 'adamw') == 'fused_adam' and FusedAdam is not None:
        base_optimizer = FusedAdam
    else:
        base_optimizer = torch.optim.AdamW
    
    optimizer = base_optimizer(param_groups, **optimizer_args)
    
    if is_master(args, local=getattr(args, 'log_local', False)):
        logging.info(f'Multimodal Optimizer: {type(optimizer).__name__}')
        logging.info(f'Optimizer config: {optimizer_args}')
        logging.info(f'Number of parameter groups: {len(param_groups)}')
    
    if return_params:
        return optimizer, param_groups
    return optimizer


def get_multimodal_assigner(args, model):
    """
    获取多模态训练的层级衰减分配器
    
    对于多模态训练，主要对 point_encoder 应用层级衰减
    
    Args:
        args: 参数配置，需包含 point_ld
        model: 模型
    
    Returns:
        assigner: LayerDecayValueAssigner 或 None
    """
    point_ld = getattr(args, 'point_ld', 1.0)
    
    if point_ld >= 1.0:
        return None
    
    # 获取模型
    if hasattr(model, 'module'):
        model = model.module
    
    # 获取点云编码器的层数
    if hasattr(model, 'uni3d') and hasattr(model.uni3d, 'point_encoder'):
        point_encoder = model.uni3d.point_encoder
    elif hasattr(model, 'point_encoder'):
        point_encoder = model.point_encoder
    else:
        logging.warning("Could not find point_encoder in model, skipping layer decay")
        return None
    
    # 获取 Transformer 块的数量
    if hasattr(point_encoder, 'visual') and hasattr(point_encoder.visual, 'blocks'):
        num_layers = len(point_encoder.visual.blocks)
    elif hasattr(point_encoder, 'blocks'):
        num_layers = len(point_encoder.blocks)
    else:
        logging.warning("Could not determine number of layers in point_encoder, using default 24")
        num_layers = 24
    
    # 创建层级衰减值分配器
    values = [point_ld ** (num_layers + 1 - i) for i in range(num_layers + 2)]
    assigner = LayerDecayValueAssigner(values)
    
    logging.info(f"Point encoder layer decay: ld={point_ld}, num_layers={num_layers}")
    logging.info(f"Layer decay values: {assigner.values}")
    
    return assigner