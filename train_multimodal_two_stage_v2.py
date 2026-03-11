"""
多模态 Uni3D 两阶段训练脚本 V2

Stage 1: 
- 学习率较小
- 仅点云编码器参与微调
- 三模态特征直接 concat 后池化

Stage 2: 
- 学习率较大
- 点云编码器冻结
- TSB + MOE 参与训练
- 使用 modality-dropout

学习率调度：余弦退火 (Cosine Annealing)
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

from models.uni3d_multimodal_two_stage_v2 import create_uni3d_multimodal_two_stage_v2
from models.losses_multimodal import Uni3dMultimodalLoss
import collections


# ============ 支持的模态组合 ============
SUPPORTED_MODALS = ['i', 'v', 't', 'iv', 'it', 'vt', 'ivt']


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


# ============ 模态采样（支持分布式同步） ============

def sample_modality_dropout_local(modality_dropout_prob: float = 0.5):
    """本地模态 Dropout 采样"""
    has_i = random.random() > modality_dropout_prob
    has_v = random.random() > modality_dropout_prob
    has_t = random.random() > modality_dropout_prob
    
    if not (has_i or has_v or has_t):
        choice = random.choice(['i', 'v', 't'])
        if choice == 'i':
            has_i = True
        elif choice == 'v':
            has_v = True
        else:
            has_t = True
    
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
    模态采样（支持分布式同步）
    
    Stage 1: 返回 'ivt'（三模态concat）
    Stage 2: 随机采样
    """
    if stage == 1:
        return 'ivt'  # Stage 1 使用所有模态
    
    # Stage 2: 使用 modality-dropout
    if distributed and world_size > 1:
        modal_idx = torch.tensor([0], device=device, dtype=torch.long)
        
        if rank == 0:
            current_modal = sample_modality_dropout_local(modality_dropout_prob)
            modal_to_idx = {m: i for i, m in enumerate(SUPPORTED_MODALS)}
            modal_idx[0] = modal_to_idx[current_modal]
        
        dist.broadcast(modal_idx, src=0)
        modal = SUPPORTED_MODALS[modal_idx.item()]
    else:
        modal = sample_modality_dropout_local(modality_dropout_prob)
    
    return modal


# ============ 余弦退火学习率调度器 ============

class CosineAnnealingScheduler:
    """
    余弦退火学习率调度器
    
    支持：
    - 预热阶段 (warmup)
    - 余弦退火衰减
    - 最小学习率
    - 多参数组（不同基础学习率）
    """
    
    def __init__(self, optimizer, base_lrs, total_steps, warmup_steps, min_lr_ratio=0.01):
        """
        Args:
            optimizer: 优化器
            base_lrs: 各参数组的基础学习率列表
            total_steps: 总训练步数
            warmup_steps: 预热步数
            min_lr_ratio: 最小学习率比例（相对于基础学习率）
        """
        self.optimizer = optimizer
        self.base_lrs = base_lrs
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.current_step = 0
    
    def get_lr_scale(self, step):
        """计算学习率缩放比例"""
        if step < self.warmup_steps:
            # 线性预热
            return float(step) / float(max(1, self.warmup_steps))
        else:
            # 余弦退火
            progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay
    
    def step(self, step=None):
        """更新学习率"""
        if step is not None:
            self.current_step = step
        
        lr_scale = self.get_lr_scale(self.current_step)
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i] if i < len(self.base_lrs) else self.base_lrs[-1]
            param_group['lr'] = base_lr * lr_scale
        
        self.current_step += 1
        return self.optimizer.param_groups[0]['lr']
    
    def __call__(self, step):
        """DeepSpeed 兼容接口"""
        return self.step(step)


def create_cosine_scheduler(optimizer, args, total_steps, stage):
    """
    创建余弦退火调度器
    
    Args:
        optimizer: 优化器
        args: 配置参数
        total_steps: 总训练步数
        stage: 训练阶段
    """
    if stage == 1:
        # Stage 1: 学习率较小
        base_lrs = [args.stage1_point_lr, args.stage1_lr, args.stage1_lr]
    else:
        # Stage 2: 学习率较大
        base_lrs = [args.stage2_point_lr, args.stage2_lr, args.stage2_lr]
    
    scheduler = CosineAnnealingScheduler(
        optimizer=optimizer,
        base_lrs=base_lrs,
        total_steps=total_steps,
        warmup_steps=args.warmup,
        min_lr_ratio=args.min_lr_ratio
    )
    
    return scheduler


# ============ 训练包装器 ============

class TwoStageTrainingWrapper(nn.Module):
    """两阶段训练包装器"""
    
    def __init__(self, uni3d_model, args):
        super().__init__()
        self.uni3d = uni3d_model
        self.args = args
        self.stage = getattr(args, 'stage', 1)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, pc, image, text, modal='ivt'):
        """前向传播"""
        outputs = self.uni3d(
            pc=pc,
            image_embed=image,
            text_embed=text,
            modal=modal
        )
        
        return {
            'fused_feats': outputs['fused_feats'],
            'clip_text_embed': outputs['txt_feats'],
            'clip_image_embed': outputs['image_feats'],
            'logit_scale': self.logit_scale.exp(),
            'modal': outputs['modal']
        }


# ============ 辅助类 ============

class AverageMeter(object):
    """计算并存储平均值"""
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


# ============ 训练函数 ============

def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, criterion,
                    epoch, device, args, logger, writer=None):
    """训练一个 epoch"""
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    
    metric_names = ['loss', 'loss_text', 'loss_image', 'fused_text_acc', 'fused_image_acc']
    metrics = OrderedDict([(name, AverageMeter(name, ':.4f')) for name in metric_names])
    
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
        stage_info = "Stage 1 (concat+pool)" if args.stage == 1 else "Stage 2 (TSB+MOE)"
        logger.info(f"Training {stage_info} | Epoch {epoch}")

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
        
        # 更新学习率
        if not is_accumulating:
            if args.enable_deepspeed:
                scheduler(optim_step)
            else:
                scheduler.step(optim_step)

        data_time.update(time.time() - end)

        texts = inputs[3]
        pc = inputs[4]
        image = inputs[5]
        rgb = inputs[6]
        
        if pc.size(0) == 0:
            continue

        feature = torch.cat((pc, rgb), dim=-1)
        feature = feature.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        image = image.to(device=device, non_blocking=True)
        
        del pc, rgb

        # 模态采样
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
                logging.warning(f"NaN loss at step {data_iter}")
                model.optimizer.zero_grad()
                if nan_count >= max_nan_tolerance:
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
        
        # 更新 logit_scale
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

        # 日志记录
        if not is_accumulating and (data_iter // accum_steps) % args.log_interval == 0:
            if is_master(args):
                current_lr = model.optimizer.param_groups[0]['lr'] if args.enable_deepspeed else optimizer.param_groups[0]['lr']
                
                print(f"\n{'='*60}")
                print(f"📊 Stage {args.stage} | Epoch [{epoch}/{args.epochs}] | Step [{data_iter // accum_steps}/{optim_steps_per_epoch}]")
                print(f"   🎲 Modal: {modal}")
                print(f"   📉 Loss: {metrics['loss'].val:.4f} (avg: {metrics['loss'].avg:.4f})")
                print(f"   📝 Text Loss:  {metrics.get('loss_text', AverageMeter('', '')).val:.4f}")
                print(f"   🖼️  Image Loss: {metrics.get('loss_image', AverageMeter('', '')).val:.4f}")
                print(f"   🎯 Text Acc:   {metrics.get('fused_text_acc', AverageMeter('', '')).val:.2f}%")
                print(f"   🎯 Image Acc:  {metrics.get('fused_image_acc', AverageMeter('', '')).val:.2f}%")
                print(f"   📈 LR: {current_lr:.2e} | Grad Norm: {grad_norm_value:.4f}")
                print(f"{'='*60}")
                
                # TensorBoard 记录
                if writer is not None:
                    global_step = epoch * optim_steps_per_epoch + data_iter // accum_steps
                    writer.add_scalar('train_step/loss', metrics['loss'].val, global_step)
                    writer.add_scalar('train_step/loss_text', metrics['loss_text'].val, global_step)
                    writer.add_scalar('train_step/loss_image', metrics['loss_image'].val, global_step)
                    writer.add_scalar('train_step/fused_text_acc', metrics['fused_text_acc'].val, global_step)
                    writer.add_scalar('train_step/fused_image_acc', metrics['fused_image_acc'].val, global_step)
                    writer.add_scalar('train_step/lr', current_lr, global_step)
                    writer.add_scalar('train_step/grad_norm', grad_norm_value, global_step)
                    writer.add_scalar('train_step/logit_scale', logit_scale, global_step)
                    writer.add_scalar('train_step/loss_scale', loss_scale_value, global_step)
                    writer.flush()
                    
            progress.display(data_iter)
        
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({'loss': f"{metrics['loss'].avg:.4f}", 'modal': modal})

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

            # 获取点云特征
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
                client_state=client_state,
                save_latest=False  # 防止 best 覆盖 latest，确保 latest 始终指向最新 epoch
            )
    else:
        model_state = get_model(model).state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'best_acc1': best_acc1,
            'stage': args.stage,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
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


# ============ 参数解析 ============

def get_args():
    """获取命令行参数"""
    parser = argparse.ArgumentParser(description='Two-Stage Multimodal Uni3D Training V2')
    
    # 阶段参数
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2], help='训练阶段')
    parser.add_argument('--stage1_checkpoint', type=str, default=None, help='Stage 1 检查点路径')
    parser.add_argument('--modality_dropout_prob', type=float, default=0.3, help='模态 dropout 概率')
    
    # 学习率参数 - 分阶段设置
    parser.add_argument('--stage1_lr', type=float, default=1e-4, help='Stage 1 学习率（较小）')
    parser.add_argument('--stage1_point_lr', type=float, default=1e-5, help='Stage 1 点云编码器学习率')
    parser.add_argument('--stage2_lr', type=float, default=1e-3, help='Stage 2 学习率（较大）')
    parser.add_argument('--stage2_point_lr', type=float, default=0.0, help='Stage 2 点云编码器学习率（冻结为0）')
    parser.add_argument('--min_lr_ratio', type=float, default=0.01, help='最小学习率比例')
    
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
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--wd', type=float, default=0.1)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--grad_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_nan_tolerance', type=int, default=10)
    
    # 损失参数
    parser.add_argument('--text_weight', type=float, default=1.0)
    parser.add_argument('--image_weight', type=float, default=1.0)
    
    # DeepSpeed 参数
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--zero_stage', type=int, default=2, help='DeepSpeed Zero 优化阶段 (1, 2, 或 3)')
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--grad_checkpointing', action='store_true', default=False)
    parser.add_argument('--grad_clip_norm', type=float, default=None, help='DeepSpeed 梯度裁剪范数')
    parser.add_argument('--lr', type=float, default=1e-4, help='基础学习率 (用于 DeepSpeed 配置)')
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp16', 'bf16', 'fp32'], help='训练精度')
    
    # 其他参数
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./output_two_stage_v2')
    parser.add_argument('--logs', type=str, default='./logs')
    parser.add_argument('--log_local', action='store_true', default=False)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--save_frequency', type=int, default=1)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--disable_amp', action='store_true')
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


# ============ 主函数 ============

def main():
    """主函数"""
    args = get_args()
    
    ds_init = init_deepspeed(args)
    
    if args.name is None:
        args.name = f"stage{args.stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if ds_init is not None:
        dsconfig_path = os.path.join(os.getcwd(), "dsconfig", args.name)
        os.makedirs(dsconfig_path, exist_ok=True)
        
        # 设置 DeepSpeed 配置所需的参数
        if args.stage == 1:
            args.lr = args.stage1_lr
        else:
            args.lr = args.stage2_lr
        
        # 设置梯度裁剪范数 (如果没有显式设置，使用 grad_clip)
        if args.grad_clip_norm is None and args.grad_clip > 0:
            args.grad_clip_norm = args.grad_clip
        
        # 计算总训练步数供 DeepSpeed scheduler 使用
        # 注意: 此时 dataloader 还未创建，使用估算值
        # 实际训练时会通过自定义 scheduler 覆盖
        args.total_steps = 100000  # 占位值
        
        create_deepspeed_config(args)
        print(f"[DeepSpeed] Config created at: {args.deepspeed_config}")
        print(f"[DeepSpeed] Zero Stage: {args.zero_stage}, Precision: {args.precision}")
    
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
    
    # ============ 创建模型 ============
    logger.info(f"=> Creating model for Stage {args.stage}")
    
    if args.stage == 1:
        # Stage 1: 直接创建
        uni3d_model = create_uni3d_multimodal_two_stage_v2(args, stage=1)
        uni3d_model.setup_stage1_training()
    else:
        # Stage 2: 加载 Stage 1 检查点并扩展
        uni3d_model = create_uni3d_multimodal_two_stage_v2(args, stage=1)
        
        if args.stage1_checkpoint:
            logger.info(f"=> Loading Stage 1 checkpoint: {args.stage1_checkpoint}")
            checkpoint = torch.load(args.stage1_checkpoint, map_location='cpu')
            
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', '').replace('uni3d.', ''): v 
                              for k, v in state_dict.items()}
            
            uni3d_state = {}
            for k, v in state_dict.items():
                if k.startswith('uni3d.'):
                    uni3d_state[k.replace('uni3d.', '')] = v
                elif not k.startswith('logit_scale'):
                    uni3d_state[k] = v
            
            missing, unexpected = uni3d_model.load_state_dict(uni3d_state, strict=False)
            logger.info(f"Loaded Stage 1. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            
            del checkpoint
        else:
            logger.warning("No Stage 1 checkpoint provided!")
        
        # 扩展到 Stage 2
        uni3d_model.expand_to_stage2()
        uni3d_model.setup_stage2_training()
    
    uni3d_model.to(device)
    
    # 包装模型
    model = TwoStageTrainingWrapper(uni3d_model, args)
    model = model.to(device)
    model_without_ddp = model
    
    # 打印参数统计
    uni3d_model.get_trainable_params_info()
    
    # ============ 数据集 ============
    logger.info("=> Creating dataset")
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
    
    # ============ 优化器和调度器 ============
    criterion = Uni3dMultimodalLoss(
        text_weight=args.text_weight, image_weight=args.image_weight,
        use_distributed=args.distributed
    ).to(device)
    
    # 根据阶段设置学习率
    if args.stage == 1:
        point_lr = args.stage1_point_lr
        other_lr = args.stage1_lr
    else:
        point_lr = args.stage2_point_lr  # 通常为 0（冻结）
        other_lr = args.stage2_lr
    
    param_groups = [
        {'params': [p for p in model_without_ddp.uni3d.point_encoder.parameters() if p.requires_grad], 
         'lr': point_lr, 'weight_decay': args.wd},
        {'params': [p for n, p in model_without_ddp.uni3d.named_parameters() 
                    if 'point_encoder' not in n and p.requires_grad], 
         'lr': other_lr, 'weight_decay': args.wd},
        {'params': [model_without_ddp.logit_scale], 
         'lr': other_lr, 'weight_decay': 0.0}
    ]
    
    optimizer = torch.optim.AdamW(param_groups, lr=other_lr, weight_decay=args.wd, betas=(args.beta1, args.beta2))
    scaler = None
    
    # 计算总步数
    iters_per_epoch = len(train_loader)
    optim_steps_per_epoch = iters_per_epoch // args.grad_accumulation_steps
    total_optim_steps = optim_steps_per_epoch * args.epochs
    
    if args.enable_deepspeed:
        # DeepSpeed 会自动从 args.deepspeed_config 读取配置
        # 不需要再通过 config 参数传入，否则会报冲突错误
        model, optimizer, _, _ = ds_init(
            args=args, model=model, optimizer=optimizer,
            model_parameters=param_groups, dist_init_required=not args.distributed,
        )
        model_without_ddp = model.module if hasattr(model, 'module') else model
        
        # DeepSpeed 的余弦退火调度器
        base_lrs = [point_lr, other_lr, other_lr]
        
        def deepspeed_cosine_scheduler(current_step):
            if current_step < args.warmup:
                lr_scale = float(current_step) / float(max(1, args.warmup))
            else:
                progress = float(current_step - args.warmup) / float(max(1, total_optim_steps - args.warmup))
                lr_scale = args.min_lr_ratio + (1.0 - args.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
            
            for i, param_group in enumerate(model.optimizer.param_groups):
                group_base_lr = base_lrs[i] if i < len(base_lrs) else other_lr
                param_group['lr'] = group_base_lr * lr_scale
            return other_lr * lr_scale
        
        scheduler = deepspeed_cosine_scheduler
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
            model_without_ddp = model.module
        scaler = amp.GradScaler() if args.use_amp else None
        
        # 余弦退火调度器
        scheduler = create_cosine_scheduler(optimizer, args, total_optim_steps, args.stage)
    
    # ============ 恢复训练 ============
    start_epoch = 0
    best_acc1 = 0.0
    
    if args.resume:
        if args.enable_deepspeed:
            # DeepSpeed 检查点恢复：resume 应指向 checkpoints 目录路径
            # 例如: output_two_stage_v2/20260205_214814_stage1/checkpoints
            resume_dir = args.resume
            if os.path.isdir(resume_dir):
                # tag=None 时会自动读取 latest 文件确定最新检查点
                _, client_state = model.load_checkpoint(resume_dir, tag=None)
                if client_state:
                    start_epoch = client_state.get('epoch', 0) + 1
                    best_acc1 = client_state.get('best_acc1', 0.0)
                    logger.info(f"=> DeepSpeed resumed from '{resume_dir}' (next epoch: {start_epoch}, best_acc1: {best_acc1:.2f})")
                else:
                    logger.warning(f"=> DeepSpeed checkpoint loaded but no client_state found")
            else:
                logger.warning(f"=> DeepSpeed resume dir not found: {resume_dir}")
        elif os.path.isfile(args.resume):
            # 普通 PyTorch 检查点恢复
            checkpoint = torch.load(args.resume, map_location='cpu')
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_acc1 = checkpoint.get('best_acc1', 0.0)
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scaler is not None and checkpoint.get('scaler_state_dict'):
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info(f"=> Resuming from '{args.resume}' (epoch {start_epoch})")
            del checkpoint
        else:
            logger.warning(f"=> Resume path not found: {args.resume}")
    
    # TensorBoard
    writer = None
    if args.tensorboard and TENSORBOARD_AVAILABLE and is_master(args):
        tensorboard_dir = args.tensorboard_dir or os.path.join(args.output_dir, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(tensorboard_dir)
    
    # CLIP 模型用于验证
    logger.info("=> Loading CLIP model for validation...")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_name=args.clip_model, pretrained=args.pretrained
    )
    clip_model.to(device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    
    # ============ 训练循环 ============
    logger.info(f"Starting Stage {args.stage} training...")
    logger.info(f"  - Learning Rate: {'Small (Stage 1)' if args.stage == 1 else 'Large (Stage 2)'}")
    logger.info(f"  - Point Encoder: {'Trainable' if args.stage == 1 else 'Frozen'}")
    logger.info(f"  - TSB/MOE: {'Not Used' if args.stage == 1 else 'Trainable'}")
    logger.info(f"  - LR Schedule: Cosine Annealing (warmup={args.warmup})")
    
    best_epoch = -1
    
    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            print(f"\n{'#'*70}")
            print(f"#{'':^68}#")
            stage_desc = "concat+pool" if args.stage == 1 else "TSB+MOE"
            print(f"#{'🚀 STAGE ' + str(args.stage) + ' (' + stage_desc + ') | EPOCH ' + str(epoch) + '/' + str(args.epochs):^68}#")
            print(f"#{'':^68}#")
            print(f"{'#'*70}\n")
        
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, criterion,
            epoch, device, args, logger, writer=writer
        )
        
        # 保存检查点
        if is_master(args):
            logger.info(f"=> Saving checkpoint for epoch {epoch}...")
        
        if args.enable_deepspeed:
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, args, best_acc1=best_acc1)
        elif is_master(args):
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, args, best_acc1=best_acc1)
        
        # 验证
        with amp.autocast(enabled=not args.disable_amp):
            val_stats = test_zeroshot_3d_core(
                val_loader, args.validate_dataset_name, model, clip_model, tokenizer, args
            )
            val_scanobjnn_stats = test_zeroshot_3d_core(
                val_scanobjnn_loader, args.validate_dataset_name_scanobjnn, model, clip_model, tokenizer, args
            )
            acc1 = val_scanobjnn_stats["acc1"]
        
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
        
        # TensorBoard 记录 epoch 级别指标
        if writer is not None:
            writer.add_scalar('train_epoch/loss', train_stats['loss'], epoch)
            writer.add_scalar('train_epoch/loss_text', train_stats['loss_text'], epoch)
            writer.add_scalar('train_epoch/loss_image', train_stats['loss_image'], epoch)
            writer.add_scalar('train_epoch/fused_text_acc', train_stats['fused_text_acc'], epoch)
            writer.add_scalar('train_epoch/fused_image_acc', train_stats['fused_image_acc'], epoch)
            writer.add_scalar('train_epoch/logit_scale', train_stats['logit_scale'], epoch)
            writer.add_scalar('train_epoch/lr', train_stats['lr'], epoch)
            
            # 模态分布
            modal_counts = train_stats.get('modal_counts', {})
            total_modal_samples = sum(modal_counts.values()) if modal_counts else 1
            for modal_name, count in modal_counts.items():
                if total_modal_samples > 0:
                    ratio = count / total_modal_samples * 100
                    writer.add_scalar(f'modal_dist/{modal_name}_count', count, epoch)
                    writer.add_scalar(f'modal_dist/{modal_name}_ratio', ratio, epoch)
            
            # 验证指标
            writer.add_scalar('val/modelnet40_acc1', val_stats['acc1'], epoch)
            writer.add_scalar('val/modelnet40_acc3', val_stats['acc3'], epoch)
            writer.add_scalar('val/modelnet40_acc5', val_stats['acc5'], epoch)
            writer.add_scalar('val/scanobjnn_acc1', val_scanobjnn_stats['acc1'], epoch)
            writer.add_scalar('val/scanobjnn_acc3', val_scanobjnn_stats['acc3'], epoch)
            writer.add_scalar('val/scanobjnn_acc5', val_scanobjnn_stats['acc5'], epoch)
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