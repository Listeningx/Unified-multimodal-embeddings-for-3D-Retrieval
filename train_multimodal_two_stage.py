"""
多模态 Uni3D 两阶段训练脚本

Stage 1: 
- 三流融合模块 (TripleStreamBlock) 同时输入点云、图片、文本特征
- 单专家 MOE (num_experts=1)，只有 'v' 模态的 router/resample_tokens
- 融合后只取点云部分的特征进入 MOE
- point_encoder 解冻参与训练

Stage 2: 
- 使用 modality-dropout，7 种模态组合都有机会出现
- 3 专家 MOE，从 Stage 1 单专家复制初始化
- 其他模态模块随机初始化
- point_encoder 冻结

训练目标:
- uni3d_multimodal(modality) <-> clip_text(t)
- uni3d_multimodal(modality) <-> clip_image(i)
"""

import os
import sys
import argparse
import time
import math
import json
import logging
import random
from pathlib import Path
from collections import OrderedDict
from datetime import datetime
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

import open_clip
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms

# TensorBoard 支持
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import customized_collate_fn
from utils.utils import get_dataset
from utils.tokenizer import SimpleTokenizer
from utils.distributed import is_master, init_distributed_device, world_info_from_env, create_deepspeed_config
from utils.optim import get_all_parameters, get_loss_scale_for_deepspeed, get_grad_norm_

from models.uni3d_multimodal_two_stage import create_uni3d_multimodal_two_stage
from models.losses_multimodal import Uni3dMultimodalLoss
import collections


# ============ 支持的模态组合 ============
SUPPORTED_MODALS = ['i', 'v', 't', 'iv', 'it', 'vt', 'ivt']
STAGE1_MODALS = ['v']  # Stage 1 只支持 'v'


def setup_logging(output_dir, log_level=logging.INFO, rank=0):
    """设置日志"""
    os.makedirs(output_dir, exist_ok=True)
    handlers = [logging.StreamHandler()]
    if rank == 0:
        log_file = os.path.join(output_dir, 'train.log')
        file_handler = logging.FileHandler(log_file, mode='a')
        handlers.append(file_handler)
    logging.basicConfig(
        level=log_level if rank == 0 else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )
    return logging.getLogger(__name__)


def random_seed(seed=42, rank=0):
    """设置随机种子"""
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def sample_modality_dropout_local(modality_dropout_prob: float = 0.5):
    """
    本地模态 Dropout 采样（不包含分布式同步）
    
    Args:
        modality_dropout_prob: 丢弃某个模态的概率
    
    Returns:
        modal: 采样到的模态组合
    """
    # 使用 modality-dropout
    # 方案：随机决定每个模态是否使用，确保至少有一个模态
    has_i = random.random() > modality_dropout_prob
    has_v = random.random() > modality_dropout_prob
    has_t = random.random() > modality_dropout_prob
    
    # 确保至少有一个模态
    if not (has_i or has_v or has_t):
        # 随机选择一个模态
        choice = random.choice(['i', 'v', 't'])
        if choice == 'i':
            has_i = True
        elif choice == 'v':
            has_v = True
        else:
            has_t = True
    
    # 构建模态字符串
    modal = ''
    if has_i:
        modal += 'i'
    if has_v:
        modal += 'v'
    if has_t:
        modal += 't'
    
    return modal


def sample_modality_dropout(stage: int, modality_dropout_prob: float = 0.5, 
                            distributed: bool = False, rank: int = 0, 
                            world_size: int = 1, device: torch.device = None):
    """
    模态 Dropout 采样（支持分布式同步）
    
    Stage 1: 只返回 'v' 模态
    Stage 2: 随机采样 7 种模态组合之一
    
    **重要**：多卡训练时，必须保证所有 GPU 使用相同的模态组合
    否则 all_gather/all_reduce 时会因为输出维度不一致而卡住
    
    Args:
        stage: 训练阶段
        modality_dropout_prob: 丢弃某个模态的概率（Stage 2）
        distributed: 是否分布式训练
        rank: 当前进程的 rank
        world_size: 总进程数
        device: 当前设备
    
    Returns:
        modal: 采样到的模态组合
    """
    if stage == 1:
        return 'v'
    
    # Stage 2: 使用 modality-dropout
    if distributed and world_size > 1:
        # 分布式训练：rank 0 采样，然后广播给其他 GPU
        modal_idx = torch.tensor([0], device=device, dtype=torch.long)
        
        if rank == 0:
            # 只有 rank 0 进行采样
            current_modal = sample_modality_dropout_local(modality_dropout_prob)
            # 将模态组合转换为索引
            modal_to_idx = {m: i for i, m in enumerate(SUPPORTED_MODALS)}
            modal_idx[0] = modal_to_idx[current_modal]
        
        # 广播模态索引给所有 GPU
        dist.broadcast(modal_idx, src=0)
        
        # 将索引转换回模态字符串
        modal = SUPPORTED_MODALS[modal_idx.item()]
    else:
        # 单卡训练：直接采样
        modal = sample_modality_dropout_local(modality_dropout_prob)
    
    return modal


class TwoStageTrainingWrapper(nn.Module):
    """
    两阶段训练包装器
    """
    
    def __init__(self, uni3d_model, clip_model, args):
        super().__init__()
        self.uni3d = uni3d_model
        self.clip = clip_model
        self.args = args
        self.use_embed = getattr(args, 'use_embed', False)
        self.stage = getattr(args, 'stage', 1)
        
        if self.clip is not None:
            for param in self.clip.parameters():
                param.requires_grad = False
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, pc, image, text, modal='v'):
        """
        前向传播
        
        Args:
            pc: [B, N, 6] 点云
            image: [B, embed_dim] 预提取图像特征
            text: [B, embed_dim] 预提取文本特征
            modal: 模态组合
        
        Stage 1 特殊处理：
        - 传入所有特征（点云、图片、文本）用于三流融合
        - modal 固定为 'v'，MOE 只使用 'v' 模态的 router/resample_tokens
        
        Stage 2:
        - 根据 modal 决定使用哪些模态进行融合
        - 但始终传入全部图文特征，用于后续损失计算
        """
        if self.stage == 1:
            # Stage 1: 传入所有特征用于三流融合
            # 模型内部会使用全部模态进行三流融合，但 MOE 只用 'v'
            pc_input = pc
            image_input = image
            text_input = text
            effective_modal = 'v'  # Stage 1 的 MOE modal 固定为 'v'
        else:
            # Stage 2: 始终传入全部特征，模型内部根据 modal 决定融合哪些模态
            # 但原始图文特征会保留用于损失计算
            pc_input = pc
            image_input = image
            text_input = text
            effective_modal = modal
        
        if self.use_embed:
            uni3d_output = self.uni3d.forward(
                pc=pc_input,
                image_embed=image_input,
                text_embed=text_input,
                modal=effective_modal
            )
        else:
            uni3d_output = self.uni3d.forward(
                pc=pc_input,
                image=image_input,
                text=text_input,
                modal=effective_modal
            )
        
        fused_feats = uni3d_output['fused_feats']
        clip_text_embed = uni3d_output['txt_feats']
        clip_image_embed = uni3d_output['image_feats']

        return {
            'fused_feats': fused_feats,
            'clip_text_embed': clip_text_embed,
            'clip_image_embed': clip_image_embed,
            'logit_scale': self.logit_scale.exp(),
            'modal': effective_modal
        }


class AverageMeter(object):
    """计算并存储平均值和当前值"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        from utils.utils import is_dist_avail_and_initialized
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def train_one_epoch(model, dataloader, clip_model, optimizer, scheduler, scaler, criterion,
                    epoch, device, args, logger, writer=None):
    """训练一个 epoch"""
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    
    metric_names = ['loss', 'loss_text', 'loss_image', 'fused_text_acc', 'fused_image_acc']
    metrics = OrderedDict([(name, AverageMeter(name, ':.4f')) for name in metric_names])
    
    # 模态统计
    modal_counts = {m: 0 for m in SUPPORTED_MODALS}
    
    accum_steps = getattr(args, 'grad_accumulation_steps', 1)
    effective_batch_size = args.batch_size * accum_steps * args.world_size
    
    iters_per_epoch = len(dataloader)
    optim_steps_per_epoch = iters_per_epoch // accum_steps
    
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))
    
    if is_master(args):
        logger.info(f"Stage {args.stage} Training | Modality-Dropout: {'Enabled (Stage 2)' if args.stage == 2 else 'Disabled (Stage 1, v only)'}")
        if accum_steps > 1:
            logger.info(f"Gradient Accumulation: {accum_steps} steps, Effective Batch Size: {effective_batch_size}")

    model.train()

    end = time.time()
    accumulated_loss = 0.0
    accumulated_loss_dict = {k: 0.0 for k in metric_names}
    nan_count = 0
    max_nan_tolerance = getattr(args, 'max_nan_tolerance', 10)
    
    if is_master(args):
        pbar = tqdm(total=iters_per_epoch, desc=f"Epoch {epoch}", 
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    else:
        pbar = None
    
    for data_iter, inputs in enumerate(dataloader):
        if inputs is None:
            continue
        
        optim_step = epoch * optim_steps_per_epoch + data_iter // accum_steps
        is_accumulating = (data_iter + 1) % accum_steps != 0
        is_first_micro_batch = data_iter % accum_steps == 0
        
        if scheduler is not None and not is_accumulating:
            if args.enable_deepspeed:
                scheduler(optim_step)
            else:
                scheduler(optim_step)

        data_time.update(time.time() - end)

        texts = inputs[3]
        pc = inputs[4]
        image = inputs[5]
        rgb = inputs[6]
        
        if pc.size(0) == 0:
            continue
        
        use_image = inputs[2].reshape(-1)
        loss_masks = use_image.float()

        feature = torch.cat((pc, rgb), dim=-1)

        feature = feature.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        image = image.to(device=device, non_blocking=True)
        
        del pc, rgb

        # ============ 模态采样 ============
        # **重要**：多卡训练时，使用 broadcast 确保所有 GPU 使用相同的模态组合
        # 否则 all_gather/all_reduce 时会因为输出维度不一致而卡住
        modal = sample_modality_dropout(
            stage=args.stage, 
            modality_dropout_prob=getattr(args, 'modality_dropout_prob', 0.3),
            distributed=args.distributed,
            rank=args.rank,
            world_size=args.world_size,
            device=device
        )
        modal_counts[modal] += 1

        if args.enable_deepspeed:
            with amp.autocast(enabled=not args.disable_amp):
                outputs = model(pc=feature, image=image, text=texts, modal=modal)
                loss_dict = criterion(outputs)
                loss = loss_dict['loss']

            if not math.isfinite(loss.item()):
                nan_count += 1
                logging.warning(f"NaN loss at step {data_iter}, modal={modal}. NaN count: {nan_count}")
                model.optimizer.zero_grad()
                if nan_count >= max_nan_tolerance:
                    logging.error(f"Too many NaN losses, stopping")
                    sys.exit(1)
                continue
            else:
                nan_count = 0

            accumulated_loss += loss.item()
            for k in metric_names:
                if k in loss_dict:
                    accumulated_loss_dict[k] += loss_dict[k].item()

            model.backward(loss)
            model.step()
            
            loss_scale_value, grad_norm_value = get_loss_scale_for_deepspeed(model)
            
        else:
            if is_first_micro_batch:
                optimizer.zero_grad()
            
            context_manager = model.no_sync() if (args.distributed and is_accumulating and hasattr(model, 'no_sync')) else nullcontext()
            
            with context_manager:
                with amp.autocast(enabled=not args.disable_amp):
                    outputs = model(pc=feature, image=image, text=texts, modal=modal)
                    loss_dict = criterion(outputs)
                    loss = loss_dict['loss'] / accum_steps

                if not math.isfinite(loss.item() * accum_steps):
                    nan_count += 1
                    optimizer.zero_grad()
                    if scaler is not None:
                        scaler.update()
                    if nan_count >= max_nan_tolerance:
                        sys.exit(1)
                    continue
                else:
                    nan_count = 0

                accumulated_loss += loss.item() * accum_steps
                for k in metric_names:
                    if k in loss_dict:
                        accumulated_loss_dict[k] += loss_dict[k].item()

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

        if not args.enable_deepspeed:
            grad_norm_value = 0.0
            loss_scale_value = 0.0
            if not is_accumulating:
                if scaler is not None:
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        grad_norm_value = get_grad_norm_(model.parameters()).item()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    else:
                        scaler.unscale_(optimizer)
                        grad_norm_value = get_grad_norm_(model.parameters()).item()
                    
                    scaler.step(optimizer)
                    scaler.update()
                    loss_scale_value = scaler.get_scale()
                else:
                    grad_norm_value = get_grad_norm_(model.parameters()).item()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
        
        should_update_logit_scale = args.enable_deepspeed or (not is_accumulating)
        if should_update_logit_scale:
            from utils.utils import get_model
            get_model(model).logit_scale.data.clamp_(0, 4.6052)
            logit_scale = get_model(model).logit_scale.exp().item()

        if not is_accumulating:
            for k in metrics:
                if k in accumulated_loss_dict:
                    metrics[k].update(accumulated_loss_dict[k] / accum_steps, args.batch_size * accum_steps)
            
            accumulated_loss = 0.0
            accumulated_loss_dict = {k: 0.0 for k in metric_names}

        batch_time.update(time.time() - end)
        end = time.time()
        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if not is_accumulating and (data_iter // accum_steps) % args.log_interval == 0:
            if is_master(args):
                current_lr = model.optimizer.param_groups[0]['lr'] if args.enable_deepspeed else optimizer.param_groups[0]['lr']
                
                print(f"\n{'='*60}")
                print(f"📊 Epoch [{epoch}/{args.epochs}] | Step [{data_iter // accum_steps}/{optim_steps_per_epoch}] | Stage {args.stage}")
                print(f"   🎲 Current Modal: {modal}")
                print(f"   📉 Total Loss: {metrics['loss'].val:.4f} (avg: {metrics['loss'].avg:.4f})")
                print(f"   📝 Text Loss:  {metrics.get('loss_text', AverageMeter('', '')).val:.4f}")
                print(f"   🖼️  Image Loss: {metrics.get('loss_image', AverageMeter('', '')).val:.4f}")
                print(f"   🎯 Text Acc:   {metrics.get('fused_text_acc', AverageMeter('', '')).val:.2f}%")
                print(f"   🎯 Image Acc:  {metrics.get('fused_image_acc', AverageMeter('', '')).val:.2f}%")
                print(f"   📈 LR: {current_lr:.2e} | Grad Norm: {grad_norm_value:.4f}")
                print(f"{'='*60}")
                
                if writer is not None:
                    global_step = epoch * optim_steps_per_epoch + data_iter // accum_steps
                    # 记录损失
                    writer.add_scalar('train_step/loss', metrics['loss'].val, global_step)
                    writer.add_scalar('train_step/loss_text', metrics['loss_text'].val, global_step)
                    writer.add_scalar('train_step/loss_image', metrics['loss_image'].val, global_step)
                    # 记录准确率
                    writer.add_scalar('train_step/fused_text_acc', metrics['fused_text_acc'].val, global_step)
                    writer.add_scalar('train_step/fused_image_acc', metrics['fused_image_acc'].val, global_step)
                    # 记录学习率和梯度
                    writer.add_scalar('train_step/lr', current_lr, global_step)
                    writer.add_scalar('train_step/grad_norm', grad_norm_value, global_step)
                    # 记录温度参数和 loss scale
                    writer.add_scalar('train_step/logit_scale', logit_scale, global_step)
                    writer.add_scalar('train_step/loss_scale', loss_scale_value, global_step)
                    writer.flush()
                    
            progress.display(data_iter)
        
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({
                'loss': f"{metrics['loss'].avg:.4f}",
                'modal': modal
            })

    if pbar is not None:
        pbar.close()
    
    # 打印模态统计
    if is_master(args):
        print(f"\n📊 Modality Distribution in Epoch {epoch}:")
        for m, cnt in sorted(modal_counts.items(), key=lambda x: -x[1]):
            if cnt > 0:
                print(f"   {m}: {cnt} ({cnt/sum(modal_counts.values())*100:.1f}%)")
    
    progress.synchronize()
    
    if 'logit_scale' not in dir():
        from utils.utils import get_model
        logit_scale = get_model(model).logit_scale.exp().item()
    
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[-1]['lr'] if not args.enable_deepspeed else model.optimizer.param_groups[-1]['lr'],
            'logit_scale': logit_scale,
            'effective_batch_size': args.batch_size * accum_steps * args.world_size,
            'modal_counts': modal_counts}


def accuracy(output, target, topk=(1,)):
    """计算 top-k 准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, correct


def test_zeroshot_3d_core(test_loader, validate_dataset_name, model, clip_model, tokenizer, args, test_data=None):
    """零样本分类测试"""
    from utils import utils
    
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f') 
    top3 = AverageMeter('Acc@3', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, top1, top3, top5],
        prefix='Test: ')

    model.eval()

    with open(os.path.join("./data", 'templates.json')) as f:
        templates = json.load(f)[args.validate_dataset_prompt]

    with open(os.path.join("./data", 'labels.json')) as f:
        labels = json.load(f)[validate_dataset_name]

    with torch.no_grad():
        logging.info('=> encoding captions')               
        text_features = []
        for l in labels:
            texts = [t.format(l) for t in templates]
            texts = tokenizer(texts).to(device=args.device, non_blocking=True)
            if len(texts.shape) < 2:
                texts = texts[None, ...]
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)

        end = time.time()
        per_class_stats = collections.defaultdict(int)
        per_class_correct_top1 = collections.defaultdict(int)

        for i, (pc, target, target_name, rgb) in enumerate(test_loader):
            for name in target_name:
                per_class_stats[name] += 1

            pc = pc.to(device=args.device, non_blocking=True)
            rgb = rgb.to(device=args.device, non_blocking=True)
            feature = torch.cat((pc, rgb), dim=-1)
            target = target.to(device=args.device, non_blocking=True)

            uni3d_model = utils.get_model(model).uni3d
            pc_features, _, _ = uni3d_model.encode_multimodal(point=feature, modal='v')
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)

            logits_per_pc = pc_features.float() @ text_features.float().t()

            (acc1, acc3, acc5), correct = accuracy(logits_per_pc, target, topk=(1, 3, 5))
            acc1, acc3, acc5 = utils.scaled_all_reduce([acc1, acc3, acc5])
            top1.update(acc1.item(), pc.size(0))
            top3.update(acc3.item(), pc.size(0))
            top5.update(acc5.item(), pc.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
    
    progress.synchronize()
    logging.info(f'0-shot * Acc@1 {top1.avg:.3f} Acc@3 {top3.avg:.3f} Acc@5 {top5.avg:.3f}')
    return {'acc1': top1.avg, 'acc3': top3.avg, 'acc5': top5.avg}


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, args, is_best=False, best_acc1=0.0):
    """保存检查点"""
    from utils.utils import get_model
    
    if args.enable_deepspeed:
        deepspeed_checkpoint_path = os.path.join(args.output_dir, "checkpoints")
        os.makedirs(deepspeed_checkpoint_path, exist_ok=True)
        client_state = {
            'epoch': epoch,
            'best_acc1': best_acc1,
            'stage': args.stage,
            'args': vars(args),
        }
        model.save_checkpoint(
            save_dir=deepspeed_checkpoint_path, 
            tag=f"epoch_{epoch}", 
            client_state=client_state
        )
        if is_best:
            model.save_checkpoint(
                save_dir=deepspeed_checkpoint_path, 
                tag="best", 
                client_state=client_state
            )
    else:
        model_state = get_model(model).state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'best_acc1': best_acc1,
            'stage': args.stage,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler and hasattr(scheduler, 'state_dict') else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'args': args
        }
        
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_stage{args.stage}_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_stage{args.stage}_epoch_{epoch}.pth')
            torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(args.output_dir, f'checkpoint_stage{args.stage}_best.pth')
            torch.save(checkpoint, best_path)


def get_args():
    """获取命令行参数"""
    parser = argparse.ArgumentParser(description='Two-Stage Multimodal Uni3D Training')
    
    # 阶段参数
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2], help='训练阶段 (1 或 2)')
    parser.add_argument('--stage1_checkpoint', type=str, default=None, help='Stage 1 检查点路径（Stage 2 时使用）')
    parser.add_argument('--modality_dropout_prob', type=float, default=0.3, help='模态 dropout 概率（Stage 2）')
    
    # 数据参数
    parser.add_argument('--pretrain_dataset_name', type=str, default='ensembled_embedding')
    parser.add_argument('--validate_dataset_name', type=str, default='modelnet40_openshape')
    parser.add_argument('--validate_dataset_name_lvis', type=str, default='objaverse_lvis_openshape')
    parser.add_argument('--validate_dataset_name_scanobjnn', type=str, default='scanobjnn_openshape')
    parser.add_argument('--validate_dataset_prompt', type=str, default='modelnet40_64')
    parser.add_argument('--pretrain_dataset_prompt', type=str, default='modelnet40_64')
    parser.add_argument('--npoints', type=int, default=10000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--use_height', action='store_true')
    parser.add_argument('--openshape_setting', action='store_true')
    parser.add_argument('--use_lvis', action='store_true')
    parser.add_argument('--use_embed', action='store_true')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='create_uni3d')
    parser.add_argument('--pc_model', type=str, default='eva_giant_patch14_560.m30m_ft_in22k_in1k')
    parser.add_argument('--pretrained_pc', type=str, default='')
    parser.add_argument('--clip_model', type=str, default='EVA02-E-14-plus')
    parser.add_argument('--clip_model_path', type=str, default='')
    parser.add_argument('--pretrained', type=str, default='openai')
    parser.add_argument('--embed_dim', type=int, default=1280)
    parser.add_argument('--pc_feat_dim', type=int, default=1408)
    parser.add_argument('--pc_encoder_dim', type=int, default=512)
    parser.add_argument('--group_size', type=int, default=64)
    parser.add_argument('--num_group', type=int, default=512)
    parser.add_argument('--drop_path_rate', type=float, default=0.2)
    parser.add_argument('--patch_dropout', type=float, default=0.5)
    parser.add_argument('--use_fusion_blocks', action='store_true', default=True)
    parser.add_argument('--no_fusion_blocks', action='store_true')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--point_lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=0.1)
    parser.add_argument('--point_wd', type=float, default=0.1)
    parser.add_argument('--ld', type=float, default=1.0)
    parser.add_argument('--point_ld', type=float, default=0.95)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--grad_clip_norm', type=float, default=5.0)
    parser.add_argument('--smoothing', type=float, default=0.0)
    parser.add_argument('--grad_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_nan_tolerance', type=int, default=10)
    
    # 损失参数
    parser.add_argument('--text_weight', type=float, default=1.0)
    parser.add_argument('--image_weight', type=float, default=1.0)
    
    # DeepSpeed 参数
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--zero_stage', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--grad_checkpointing', action='store_true', default=False)
    
    # 其他参数
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./output_two_stage')
    parser.add_argument('--logs', type=str, default='./logs')
    parser.add_argument('--log_local', action='store_true', default=False)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--save_frequency', type=int, default=1)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--disable_amp', action='store_true')
    parser.add_argument('--precision', type=str, default='amp')
    parser.add_argument('--seed', type=int, default=4096)
    parser.add_argument('--use_distributed', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--tensorboard', action='store_true', default=True)
    parser.add_argument('--tensorboard_dir', type=str, default=None)
    
    # 分布式参数
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--dist_url', type=str, default='env://')
    parser.add_argument('--dist_backend', type=str, default='nccl')
    parser.add_argument('--no_set_device_rank', action='store_true', default=False)
    
    args = parser.parse_args()
    
    if args.no_fusion_blocks:
        args.use_fusion_blocks = False
    if args.grad_clip_norm is None:
        args.grad_clip_norm = args.grad_clip
    
    return args


def init_deepspeed(args):
    """初始化 DeepSpeed"""
    ds_init = None
    if args.enable_deepspeed:
        try:
            os.environ['DS_BUILD_FUSED_ADAM'] = '0'
            os.environ['DS_BUILD_FUSED_LAMB'] = '0'
            import deepspeed
            os.environ['ENV_TYPE'] = "deepspeed"
            ds_init = deepspeed.initialize
            logging.info("DeepSpeed enabled")
        except ImportError:
            logging.error("Please 'pip install deepspeed>=0.9.4'")
            sys.exit(1)
    else:
        os.environ['ENV_TYPE'] = "pytorch"
    return ds_init


def get_scheduler(optimizer, args, total_steps):
    """获取学习率调度器"""
    from utils.scheduler import warmup_cosine_lr
    return warmup_cosine_lr(optimizer, args, total_steps)


def main():
    """主函数"""
    args = get_args()
    
    ds_init = init_deepspeed(args)
    
    if args.name is None:
        args.name = f"stage{args.stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if ds_init is not None:
        dsconfig_path = os.path.join(os.getcwd(), "dsconfig", args.name)
        os.makedirs(dsconfig_path, exist_ok=True)
        create_deepspeed_config(args)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    
    if args.use_distributed or args.enable_deepspeed:
        device = init_distributed_device(args)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    args.device = device
    
    logger = setup_logging(args.output_dir, rank=args.rank)
    logger.info(f"Arguments: {args}")
    logger.info(f"Training Stage: {args.stage}")
    
    random_seed(args.seed, args.rank)
    
    # CLIP 模型
    if args.use_embed:
        logger.info("=> using precomputed embeddings")
        clip_model = None
    else:
        logger.info("=> loading CLIP model...")
        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name=args.clip_model, pretrained=args.pretrained
        )
        clip_model.to(device)
        clip_model.eval()
        for param in clip_model.parameters():
            param.requires_grad = False
    
    # ============ 创建模型 ============
    logger.info(f"=> creating model for Stage {args.stage}")
    
    if args.stage == 1:
        # Stage 1: 直接创建 stage=1 的模型
        uni3d_model = create_uni3d_multimodal_two_stage(args, stage=1)
        
        # Stage 1: 解冻 Module A（包括 point_encoder），训练 Module B
        uni3d_model.unfreeze_module_a(unfreeze_point_encoder=True)
        uni3d_model.unfreeze_module_b()
        
    else:
        # Stage 2: 先创建 stage=1 的模型，加载权重，再扩展到 stage=2
        uni3d_model = create_uni3d_multimodal_two_stage(args, stage=1)
        
        # 加载 Stage 1 检查点
        if args.stage1_checkpoint:
            logger.info(f"=> loading Stage 1 checkpoint: {args.stage1_checkpoint}")
            checkpoint = torch.load(args.stage1_checkpoint, map_location='cpu')
            
            # 处理 DDP 包装后的 state_dict
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', '').replace('uni3d.', ''): v 
                              for k, v in state_dict.items()}
            
            # 只加载模型权重（不包括 wrapper 的 logit_scale）
            uni3d_state = {}
            for k, v in state_dict.items():
                if k.startswith('uni3d.'):
                    uni3d_state[k.replace('uni3d.', '')] = v
                elif not k.startswith('logit_scale'):
                    uni3d_state[k] = v
            
            missing, unexpected = uni3d_model.load_state_dict(uni3d_state, strict=False)
            logger.info(f"Loaded Stage 1 checkpoint. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            
            del checkpoint
            import gc
            gc.collect()
        else:
            logger.warning("No Stage 1 checkpoint provided for Stage 2 training!")
        
        # 扩展到 Stage 2（专家复制 3 份，其他模态随机初始化）
        uni3d_model.expand_to_stage2()
        
        # Stage 2: 冻结 Module A（包括 point_encoder），只训练 Module B
        uni3d_model.freeze_module_a(freeze_point_encoder=True)
        uni3d_model.unfreeze_module_b()
    
    import gc
    gc.collect()
    
    uni3d_model.to(device)
    
    # 包装模型
    model = TwoStageTrainingWrapper(uni3d_model, clip_model, args)
    model = model.to(device)
    model_without_ddp = model
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # 打印参数统计
    uni3d_model.get_trainable_params_info()
    
    # 梯度检查点
    if args.grad_checkpointing:
        logger.info("=> Enabling gradient checkpointing")
        point_visual = model_without_ddp.uni3d.point_encoder.visual
        if hasattr(point_visual, 'set_grad_checkpointing'):
            point_visual.set_grad_checkpointing(enable=True)
    
    # ============ 数据集 ============
    logger.info("=> creating dataset")
    tokenizer = SimpleTokenizer()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.ToTensor(),
        normalize
    ])
    
    train_dataset = get_dataset(train_transform, tokenizer, args, 'train')
    val_dataset = get_dataset(None, tokenizer, args, 'val')
    val_dataset_scanobjnn = get_dataset(None, tokenizer, args, 'val_scanobjnn')
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        val_scanobjnn_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset_scanobjnn)
    else:
        train_sampler = None
        val_sampler = None
        val_scanobjnn_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,
        drop_last=True, collate_fn=customized_collate_fn,
        persistent_workers=True if args.workers > 0 else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler
    )
    
    val_scanobjnn_loader = torch.utils.data.DataLoader(
        val_dataset_scanobjnn, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_scanobjnn_sampler
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples (ModelNet40): {len(val_dataset)}")
    logger.info(f"Val samples (ScanObjNN): {len(val_dataset_scanobjnn)}")
    
    # ============ 优化器 ============
    criterion = Uni3dMultimodalLoss(
        text_weight=args.text_weight, image_weight=args.image_weight,
        use_distributed=args.distributed
    ).to(device)
    
    # 分层学习率
    param_groups = [
        {'params': [p for p in model_without_ddp.uni3d.point_encoder.parameters() if p.requires_grad], 
         'lr': args.point_lr, 'weight_decay': args.point_wd},
        {'params': [p for n, p in model_without_ddp.uni3d.named_parameters() 
                    if 'point_encoder' not in n and p.requires_grad], 
         'lr': args.lr, 'weight_decay': args.wd},
        {'params': [model_without_ddp.logit_scale], 
         'lr': args.lr, 'weight_decay': 0.0}
    ]
    
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.wd, betas=(args.beta1, args.beta2))
    scaler = None
    scheduler = None
    
    if args.enable_deepspeed:
        model, optimizer, _, _ = ds_init(
            args=args, model=model, optimizer=optimizer,
            model_parameters=param_groups, dist_init_required=not args.distributed,
        )
        model_without_ddp = model.module if hasattr(model, 'module') else model
        
        iters_per_epoch = len(train_loader)
        optim_steps_per_epoch = iters_per_epoch // args.grad_accumulation_steps
        total_optim_steps = optim_steps_per_epoch * args.epochs
        warmup_steps = args.warmup
        initial_lrs = [args.point_lr, args.lr, args.lr]
        
        def deepspeed_lr_scheduler(current_step):
            if current_step < warmup_steps:
                lr_scale = float(current_step) / float(max(1, warmup_steps))
            else:
                progress = float(current_step - warmup_steps) / float(max(1, total_optim_steps - warmup_steps))
                lr_scale = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            for i, param_group in enumerate(model.optimizer.param_groups):
                group_base_lr = initial_lrs[i] if i < len(initial_lrs) else args.lr
                param_group['lr'] = group_base_lr * lr_scale
            return args.lr * lr_scale
        
        scheduler = deepspeed_lr_scheduler
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
            model_without_ddp = model.module
        scaler = amp.GradScaler() if args.use_amp else None
    
    total_steps = len(train_loader) * args.epochs
    if not args.enable_deepspeed:
        scheduler = get_scheduler(optimizer, args, total_steps)
    
    # ============ 恢复训练 ============
    start_epoch = 0
    best_acc1 = 0.0
    
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_acc1 = checkpoint.get('best_acc1', 0.0)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"=> resuming from '{args.resume}' (epoch {start_epoch})")
    
    # TensorBoard
    writer = None
    if args.tensorboard and TENSORBOARD_AVAILABLE and is_master(args):
        tensorboard_dir = args.tensorboard_dir or os.path.join(args.output_dir, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(tensorboard_dir)
    
    # ============ 训练循环 ============
    logger.info(f"Starting Stage {args.stage} training...")
    best_epoch = -1
    
    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            print(f"\n{'#'*70}")
            print(f"#{'':^68}#")
            print(f"#{'🚀 STAGE ' + str(args.stage) + ' | EPOCH ' + str(epoch) + '/' + str(args.epochs):^68}#")
            print(f"#{'':^68}#")
            print(f"{'#'*70}\n")
        
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, train_loader, clip_model, optimizer, scheduler, scaler, criterion,
            epoch, device, args, logger, writer=writer
        )
        
        # 保存检查点
        if is_master(args):
            logger.info(f"=> Saving checkpoint for epoch {epoch}...")
        
        if args.enable_deepspeed:
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, args)
        elif is_master(args):
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, args)
        
        # 验证
        if args.use_embed:
            logger.info("=> loading CLIP model for validation...")
            clip_model_for_val, _, _ = open_clip.create_model_and_transforms(
                model_name=args.clip_model, pretrained=args.pretrained
            )
            clip_model_for_val.to(device)
            clip_model_for_val.eval()
            for param in clip_model_for_val.parameters():
                param.requires_grad = False
        else:
            clip_model_for_val = clip_model
        
        with amp.autocast(enabled=not args.disable_amp):
            val_stats = test_zeroshot_3d_core(
                val_loader, args.validate_dataset_name, model, clip_model_for_val, tokenizer, args
            )
            val_scanobjnn_stats = test_zeroshot_3d_core(
                val_scanobjnn_loader, args.validate_dataset_name_scanobjnn, model, clip_model_for_val, tokenizer, args
            )
            acc1 = val_scanobjnn_stats["acc1"]
        
        if args.use_embed:
            del clip_model_for_val
            torch.cuda.empty_cache()
        
        if is_master(args):
            print(f"\n{'~'*70}")
            print(f"🎯 STAGE {args.stage} | EPOCH {epoch} VALIDATION")
            print(f"   📊 ModelNet40 - Acc@1: {val_stats['acc1']:.2f}%")
            print(f"   📊 ScanObjNN  - Acc@1: {val_scanobjnn_stats['acc1']:.2f}%")
            print(f"{'~'*70}\n")
        
        is_best = acc1 > best_acc1
        if is_best:
            best_acc1 = acc1
            best_epoch = epoch
            if is_master(args):
                logger.info(f"=> New best! Acc@1: {best_acc1:.2f}% at epoch {best_epoch}")
            if args.enable_deepspeed:
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, args, is_best=True, best_acc1=best_acc1)
            elif is_master(args):
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, args, is_best=True, best_acc1=best_acc1)
        
        # TensorBoard 记录：训练 epoch 平均指标
        if writer is not None:
            # 训练指标（epoch 平均）
            writer.add_scalar('train_epoch/loss', train_stats['loss'], epoch)
            writer.add_scalar('train_epoch/loss_text', train_stats['loss_text'], epoch)
            writer.add_scalar('train_epoch/loss_image', train_stats['loss_image'], epoch)
            writer.add_scalar('train_epoch/fused_text_acc', train_stats['fused_text_acc'], epoch)
            writer.add_scalar('train_epoch/fused_image_acc', train_stats['fused_image_acc'], epoch)
            writer.add_scalar('train_epoch/logit_scale', train_stats['logit_scale'], epoch)
            writer.add_scalar('train_epoch/lr', train_stats['lr'], epoch)
            
            # 模态分布统计（Stage 2 使用 modality-dropout）
            modal_counts = train_stats.get('modal_counts', {})
            total_modal_samples = sum(modal_counts.values()) if modal_counts else 1
            for modal_name, count in modal_counts.items():
                if total_modal_samples > 0:
                    ratio = count / total_modal_samples * 100
                    writer.add_scalar(f'modal_dist/{modal_name}_count', count, epoch)
                    writer.add_scalar(f'modal_dist/{modal_name}_ratio', ratio, epoch)
            
            # 验证指标 - ModelNet40
            writer.add_scalar('val/modelnet40_acc1', val_stats['acc1'], epoch)
            writer.add_scalar('val/modelnet40_acc3', val_stats['acc3'], epoch)
            writer.add_scalar('val/modelnet40_acc5', val_stats['acc5'], epoch)
            
            # 验证指标 - ScanObjNN
            writer.add_scalar('val/scanobjnn_acc1', val_scanobjnn_stats['acc1'], epoch)
            writer.add_scalar('val/scanobjnn_acc3', val_scanobjnn_stats['acc3'], epoch)
            writer.add_scalar('val/scanobjnn_acc5', val_scanobjnn_stats['acc5'], epoch)
            
            # 最佳准确率
            writer.add_scalar('val/best_acc1', best_acc1, epoch)
            
            writer.flush()
    
    if writer is not None:
        writer.close()
    
    if is_master(args):
        logger.info(f"Stage {args.stage} Training completed!")
        logger.info(f"Best Acc@1: {best_acc1:.2f}% at epoch {best_epoch}")
        if args.stage == 1:
            logger.info(f"Next: Run Stage 2 with --stage 2 --stage1_checkpoint {args.output_dir}/checkpoint_stage1_best.pth")


if __name__ == '__main__':
    main()