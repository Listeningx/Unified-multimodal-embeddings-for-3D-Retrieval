import numpy as np


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def warmup_cosine_lr(optimizer, args, steps):
    """
    支持多种参数组结构的学习率调度器：
    
    1. 旧版结构（main.py）：使用 param_group['group'] = 'text'/'visual'/'point'/'other'
    2. 新版结构（train_multimodal.py）：使用 param_group['lr'] 直接指定学习率
    
    新版结构支持以下参数组：
    - point_encoder: 使用 point_lr * 0.1
    - 其他 uni3d 模块: 使用 lr
    - logit_scale: 使用 lr * 0.1
    """
    def _lr_adjuster(step):
        for i, param_group in enumerate(optimizer.param_groups):
            # 获取该参数组的基础学习率
            # 优先使用参数组自身设定的 lr，这样可以兼容新旧两种结构
            
            if 'group' in param_group:
                # 旧版结构：根据 group 字段确定基础学习率
                group_name = param_group['group']
                if group_name == 'text':
                    base_lr = args.text_lr if hasattr(args, 'text_lr') and args.text_lr is not None else args.lr
                elif group_name == 'visual':
                    base_lr = args.visual_lr if hasattr(args, 'visual_lr') and args.visual_lr is not None else args.lr
                elif group_name == 'point':
                    base_lr = args.point_lr if hasattr(args, 'point_lr') and args.point_lr is not None else args.lr
                else:
                    base_lr = args.lr
            elif 'base_lr' in param_group:
                # 新版结构：使用预设的 base_lr
                base_lr = param_group['base_lr']
            else:
                # 使用参数组初始化时设定的 lr 作为基础学习率
                # 这是最通用的方式，适用于 train_multimodal.py 的参数组设置
                base_lr = param_group.get('initial_lr', param_group.get('lr', args.lr))
            
            # 计算当前步的学习率
            if step < args.warmup:
                lr = _warmup_lr(base_lr, args.warmup, step)
            else:
                e = step - args.warmup
                es = steps - args.warmup
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            
            # 应用层级衰减缩放（如果存在）
            scale = param_group.get("lr_scale", 1.0)
            param_group["lr"] = scale * lr
            
        return optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 0 else args.lr
    
    # 保存每个参数组的初始学习率，用于后续计算
    for param_group in optimizer.param_groups:
        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = param_group['lr']
    
    return _lr_adjuster


def warmup_step_lr(optimizer, args, decay_t=500, decay_rate=0.8):
    """
    支持多种参数组结构的 step 学习率调度器
    """
    def _lr_adjuster(step):
        for param_group in optimizer.param_groups:
            # 获取该参数组的基础学习率
            if 'group' in param_group:
                group_name = param_group['group']
                if group_name == 'text':
                    base_lr = args.text_lr if hasattr(args, 'text_lr') and args.text_lr else args.lr
                elif group_name == 'visual':
                    base_lr = args.visual_lr if hasattr(args, 'visual_lr') and args.visual_lr else args.lr
                elif group_name == 'point':
                    base_lr = args.point_lr if hasattr(args, 'point_lr') and args.point_lr else args.lr
                else:
                    base_lr = args.lr
            else:
                base_lr = param_group.get('initial_lr', param_group.get('lr', args.lr))

            if step < args.warmup:
                lr = _warmup_lr(base_lr, args.warmup, step)
            else:
                e = step - args.warmup
                lr = base_lr * (decay_rate ** (e // decay_t))
            scale = param_group.get("lr_scale", 1.0)
            param_group["lr"] = scale * lr
        return optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 0 else args.lr
    
    # 保存每个参数组的初始学习率
    for param_group in optimizer.param_groups:
        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = param_group['lr']
    
    return _lr_adjuster