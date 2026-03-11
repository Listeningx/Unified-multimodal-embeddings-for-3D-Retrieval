"""
多模态 Uni3D 对比学习训练脚本 (带 Modality Dropout + MSE 对齐损失)

与 train_multimodal_dropout.py 的区别：
- 在前几轮添加 MSE 损失，让 fused_emb 和图文特征之间对齐
- MSE 权重从初始值逐渐降低到 0（支持可配置的衰减策略）
- 支持配置 MSE 生效的 epoch 数量和衰减方式

MSE 损失说明:
- 目的：在训练初期帮助 fused_emb 快速对齐到 CLIP 图文特征空间
- loss_mse = MSE(fused_emb, text_embed) + MSE(fused_emb, image_embed)
- 前 mse_warmup_epochs 轮使用完整 MSE 权重
- 之后在 mse_decay_epochs 轮内线性衰减到 0

训练目标:
- uni3d_multimodal(任意模态组合) <-> clip_text(t)  [仅当有 t 时]
- uni3d_multimodal(任意模态组合) <-> clip_image(i) [仅当有 i 时]
- fused_emb -> text_embed (MSE, 前期使用)
- fused_emb -> image_embed (MSE, 前期使用)

模态组合概率（可配置）：
- ivt: 50%  (完整三模态)
- iv:  15%  (图像+点云)
- vt:  15%  (点云+文本)
- v:   10%  (仅点云)
- it:  5%   (图像+文本，无点云，比例极小)
- i:   2.5% (仅图像)
- t:   2.5% (仅文本)

支持:
- 单卡/多卡 DDP 训练
- DeepSpeed 分布式训练（ZeRO-1/2/3）
- 可配置的模态 dropout 概率
- 可配置的 MSE 对齐损失
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
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import customized_collate_fn
from utils.utils import get_dataset
from utils.tokenizer import SimpleTokenizer
from utils.distributed import is_master, init_distributed_device, world_info_from_env, create_deepspeed_config
from utils.optim import get_all_parameters, get_loss_scale_for_deepspeed, get_grad_norm_

from models.uni3d_multimodal import create_uni3d_multimodal
from models.losses_multimodal import Uni3dMultimodalLoss, get_multimodal_loss, ModalityDropoutLoss, get_modality_dropout_loss
import collections


# ============ 配置日志 ============

def setup_logging(output_dir, log_level=logging.INFO, rank=0):
    """
    设置日志
    
    Args:
        output_dir: 日志输出目录
        log_level: 日志级别
        rank: 当前进程的 rank，只有 rank 0 才写入文件日志
    """
    os.makedirs(output_dir, exist_ok=True)
    
    handlers = [logging.StreamHandler()]
    
    # 只在主进程（rank 0）添加文件日志处理器
    # 避免分布式训练时多进程同时写入同一文件导致内容丢失
    if rank == 0:
        log_file = os.path.join(output_dir, 'train.log')
        file_handler = logging.FileHandler(log_file, mode='a')  # 追加模式
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=log_level if rank == 0 else logging.WARNING,  # 非主进程只输出警告
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True  # 强制重新配置（覆盖之前的配置）
    )
    return logging.getLogger(__name__)


# ============ 辅助函数 ============

def random_seed(seed=42, rank=0):
    """设置随机种子"""
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    import random
    random.seed(seed + rank)


def compute_embedding(clip_model, texts, image):
    """计算 CLIP embedding（当不使用预计算 embedding 时）"""
    text_embed_all = []
    for i in range(texts.shape[0]):
        text_for_one_sample = texts[i]
        text_embed = clip_model.encode_text(text_for_one_sample)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        text_embed = text_embed.mean(dim=0)
        text_embed_all.append(text_embed)

    texts = torch.stack(text_embed_all)
    image = clip_model.encode_image(image)
    image = image / image.norm(dim=-1, keepdim=True)
    texts = texts.clone().detach()
    image = image.clone().detach()
    return texts, image


# ============ MSE 对齐损失配置 ============

def compute_mse_weight(epoch: int, args) -> float:
    """
    计算当前 epoch 的 MSE 损失权重
    
    权重调度策略：
    1. epoch < mse_warmup_epochs: 使用完整权重 mse_weight
    2. mse_warmup_epochs <= epoch < mse_warmup_epochs + mse_decay_epochs: 逐渐衰减
    3. epoch >= mse_warmup_epochs + mse_decay_epochs: 权重为 0
    
    Args:
        epoch: 当前 epoch（从 0 开始）
        args: 包含 MSE 配置的参数
            - mse_weight: 初始 MSE 权重
            - mse_warmup_epochs: 保持完整权重的 epoch 数
            - mse_decay_epochs: 衰减到 0 所需的 epoch 数
            - mse_decay_type: 衰减方式 ('linear', 'cosine', 'exponential')
    
    Returns:
        当前 epoch 的 MSE 损失权重
    """
    if not getattr(args, 'enable_mse_loss', False):
        return 0.0
    
    mse_weight = getattr(args, 'mse_weight', 1.0)
    warmup_epochs = getattr(args, 'mse_warmup_epochs', 2)
    decay_epochs = getattr(args, 'mse_decay_epochs', 3)
    decay_type = getattr(args, 'mse_decay_type', 'linear')
    
    # 阶段 1: warmup 阶段，使用完整权重
    if epoch < warmup_epochs:
        return mse_weight
    
    # 阶段 2: 衰减阶段
    decay_start = warmup_epochs
    decay_end = warmup_epochs + decay_epochs
    
    if epoch >= decay_end:
        # 阶段 3: 衰减完成，权重为 0
        return 0.0
    
    # 计算衰减进度 (0 -> 1)
    progress = (epoch - decay_start) / max(1, decay_epochs)
    
    if decay_type == 'linear':
        # 线性衰减: 1 -> 0
        scale = 1.0 - progress
    elif decay_type == 'cosine':
        # 余弦衰减: 平滑过渡
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    elif decay_type == 'exponential':
        # 指数衰减: 快速下降
        scale = math.exp(-3.0 * progress)  # exp(-3) ≈ 0.05
    else:
        # 默认线性
        scale = 1.0 - progress
    
    return mse_weight * scale


def compute_mse_alignment_loss(fused_embed, text_embed, image_embed, args):
    """
    计算 MSE 对齐损失
    
    目的：让 fused_embed 在训练初期快速对齐到 CLIP 图文特征空间
    
    Args:
        fused_embed: [B, D] 融合特征
        text_embed: [B, D] 文本特征
        image_embed: [B, D] 图像特征
        args: 包含 MSE 配置的参数
    
    Returns:
        dict: 包含 mse_loss, mse_text_loss, mse_image_loss
    """
    mse_text_weight = getattr(args, 'mse_text_weight', 1.0)
    mse_image_weight = getattr(args, 'mse_image_weight', 1.0)
    
    device = fused_embed.device
    target_dtype = fused_embed.dtype
    
    mse_text_loss = torch.tensor(0.0, device=device)
    mse_image_loss = torch.tensor(0.0, device=device)
    
    # 对特征进行归一化后计算 MSE（在归一化空间中对齐）
    fused_norm = F.normalize(fused_embed, dim=-1, p=2)
    
    # MSE 与文本对齐
    if text_embed is not None:
        text_embed = text_embed.to(target_dtype)
        if text_embed.dim() == 3 and text_embed.size(1) == 1:
            text_embed = text_embed.squeeze(1)
        text_norm = F.normalize(text_embed, dim=-1, p=2)
        mse_text_loss = F.mse_loss(fused_norm, text_norm)
    
    # MSE 与图像对齐
    if image_embed is not None:
        image_embed = image_embed.to(target_dtype)
        if image_embed.dim() == 3 and image_embed.size(1) == 1:
            image_embed = image_embed.squeeze(1)
        image_norm = F.normalize(image_embed, dim=-1, p=2)
        mse_image_loss = F.mse_loss(fused_norm, image_norm)
    
    # 加权求和
    mse_loss = mse_text_weight * mse_text_loss + mse_image_weight * mse_image_loss
    
    return {
        'mse_loss': mse_loss,
        'mse_text_loss': mse_text_loss,
        'mse_image_loss': mse_image_loss
    }


# ============ Modality Dropout 配置 ============

class ModalityDropoutConfig:
    """
    Modality Dropout 配置类
    
    控制训练时各模态组合的出现概率
    - i: image (图像)
    - v: point cloud (点云)
    - t: text (文本)
    
    默认概率分布：
    - ivt: 50%  (完整三模态，最重要)
    - iv:  15%  (图像+点云)
    - vt:  15%  (点云+文本)
    - v:   10%  (仅点云，这是核心模态)
    - it:  5%   (图像+文本，无点云，比例极小)
    - i:   2.5% (仅图像)
    - t:   2.5% (仅文本)
    """
    
    # 所有支持的模态组合
    SUPPORTED_MODALS = ['ivt', 'iv', 'vt', 'v', 'it', 'i', 't']
    
    def __init__(self, 
                 prob_ivt: float = 0.50,
                 prob_iv: float = 0.15,
                 prob_vt: float = 0.15,
                 prob_v: float = 0.10,
                 prob_it: float = 0.05,
                 prob_i: float = 0.025,
                 prob_t: float = 0.025):
        """
        Args:
            prob_ivt: ivt 模态组合出现的概率
            prob_iv: iv 模态组合出现的概率
            prob_vt: vt 模态组合出现的概率
            prob_v: 仅 v 模态出现的概率
            prob_it: it 模态组合出现的概率（应该非常小）
            prob_i: 仅 i 模态出现的概率
            prob_t: 仅 t 模态出现的概率
        """
        self.probabilities = {
            'ivt': prob_ivt,
            'iv': prob_iv,
            'vt': prob_vt,
            'v': prob_v,
            'it': prob_it,
            'i': prob_i,
            't': prob_t,
        }
        
        # 归一化概率（确保总和为 1）
        total = sum(self.probabilities.values())
        if abs(total - 1.0) > 1e-6:
            logging.warning(f"Modality dropout probabilities sum to {total}, normalizing...")
            for k in self.probabilities:
                self.probabilities[k] /= total
        
        # 预计算累积概率用于采样
        self._compute_cumulative_probs()
    
    def _compute_cumulative_probs(self):
        """计算累积概率分布"""
        self.modals = list(self.probabilities.keys())
        self.cumulative_probs = []
        cumsum = 0.0
        for modal in self.modals:
            cumsum += self.probabilities[modal]
            self.cumulative_probs.append(cumsum)
    
    def sample(self) -> str:
        """随机采样一个模态组合"""
        r = random.random()
        for i, cum_prob in enumerate(self.cumulative_probs):
            if r < cum_prob:
                return self.modals[i]
        return self.modals[-1]  # 兜底
    
    def sample_batch(self, batch_size: int) -> list:
        """为一个 batch 采样模态组合（每个样本独立采样）"""
        return [self.sample() for _ in range(batch_size)]
    
    def __repr__(self):
        prob_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in self.probabilities.items()])
        return f"ModalityDropoutConfig({prob_str})"


def get_modality_dropout_config(args) -> ModalityDropoutConfig:
    """从命令行参数创建 ModalityDropoutConfig"""
    return ModalityDropoutConfig(
        prob_ivt=getattr(args, 'modal_prob_ivt', 0.50),
        prob_iv=getattr(args, 'modal_prob_iv', 0.15),
        prob_vt=getattr(args, 'modal_prob_vt', 0.15),
        prob_v=getattr(args, 'modal_prob_v', 0.10),
        prob_it=getattr(args, 'modal_prob_it', 0.05),
        prob_i=getattr(args, 'modal_prob_i', 0.025),
        prob_t=getattr(args, 'modal_prob_t', 0.025),
    )

class MultimodalTrainingWrapper(nn.Module):
    """
    多模态训练包装器
    
    封装 Uni3DMultimodal 和 CLIP，提供训练所需的接口
    
    支持两种模式:
    1. use_embed=True: 使用预提取的图文特征，跳过CLIP编码
    2. use_embed=False: 实时编码图文特征
    
    支持两种融合方式:
    1. use_fusion_blocks=True: 通过三流融合模块交互
    2. use_fusion_blocks=False: 直接拼接三个模态特征
    """
    
    def __init__(self, uni3d_model, clip_model, args):
        super().__init__()
        self.uni3d = uni3d_model
        self.clip = clip_model
        self.args = args
        self.use_embed = getattr(args, 'use_embed', False)
        self.use_fusion_blocks = getattr(args, 'use_fusion_blocks', True)
        
        # 冻结 CLIP（仅在不使用预提取特征时有用）
        if self.clip is not None:
            for param in self.clip.parameters():
                param.requires_grad = False
        
        # 温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    @torch.no_grad()
    def encode_clip_text(self, text: torch.Tensor) -> torch.Tensor:
        """
        使用 CLIP 编码文本
        Args:
            text: [B, embed_dim] 预计算的文本 embedding 或 [B, 77] tokenized text
        Returns:
            text_embed: [B, embed_dim]
        """
        if self.use_embed:
            # 使用预计算的 embedding，直接返回归一化后的结果
            text_features = text / text.norm(dim=-1, keepdim=True)
        else:
            # 实时编码
            text_features = self.clip.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    @torch.no_grad()
    def encode_clip_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        使用 CLIP 编码图像
        Args:
            image: [B, embed_dim] 预计算的图像 embedding 或 [B, 3, H, W] 原始图像
        Returns:
            image_embed: [B, embed_dim]
        """
        if self.use_embed:
            # 使用预计算的 embedding，直接返回归一化后的结果
            image_features = image / image.norm(dim=-1, keepdim=True)
        else:
            # 实时编码
            image_features = self.clip.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def forward(self, pc, image, text, modal='ivt'):
        """
        前向传播（支持 Modality Dropout）
        
        Args:
            pc: [B, N, 6] 点云 (xyz + rgb)，当 modal 不包含 'v' 时可为 None
            image: [B, embed_dim] 预计算的图像 embedding 或 [B, 3, H, W] 原始图像，当 modal 不包含 'i' 时可为 None
            text: [B, embed_dim] 预计算的文本 embedding 或 [B, 77] tokenized text，当 modal 不包含 't' 时可为 None
            modal: 模态组合，可选值为 'ivt', 'iv', 'it', 'vt', 'i', 'v', 't'
        
        Returns:
            dict with:
                - 'fused_feats': [B, clip_embed_dim] 投影后的融合特征（用于对比学习）
                - 'clip_text_embed': [B, clip_embed_dim] CLIP 文本特征（始终提供，作为监督信号）
                - 'clip_image_embed': [B, clip_embed_dim] CLIP 图像特征（始终提供，作为监督信号）
                - 'logit_scale': 温度参数
                - 'modal': 当前使用的模态组合
        """
        # 根据 modal 参数选择传递给模型用于融合的输入
        # 注意：pc_input 根据 modal 选择性传递，但 image/text 始终传递（用于保存原始特征作为监督信号）
        pc_input = pc if 'v' in modal else None
        
        # 1. 使用 Uni3D 编码融合特征
        if self.use_embed:
            # 使用预提取特征
            # 注意：image 和 text 始终传递给模型（用于保存原始特征），但模型内部会根据 modal 决定是否用于融合
            uni3d_output = self.uni3d.forward(
                pc=pc_input, 
                image_embed=image,  # 始终传递，模型内部会根据 modal 决定是否用于融合
                text_embed=text,    # 始终传递，模型内部会根据 modal 决定是否用于融合
                modal=modal
            )
        else:
            # 实时编码
            uni3d_output = self.uni3d.forward(
                pc=pc_input, 
                image=image,  # 始终传递
                text=text,    # 始终传递
                modal=modal
            )
        
        fused_feats = uni3d_output['fused_feats']  # [B, clip_embed_dim] 投影后的融合特征

        # 2. 获取原始 CLIP 特征（用于对比学习目标）
        # 注意：无论使用什么模态组合，始终返回完整的 CLIP 文本和图像特征作为监督信号
        # 损失函数会根据 modal 决定使用哪些对比目标
        clip_text_embed = uni3d_output.get('txt_feats', None)
        clip_image_embed = uni3d_output.get('image_feats', None)

        return {
            'fused_feats': fused_feats,           # 融合特征（用于对比学习）
            'clip_text_embed': clip_text_embed,   # 原始 CLIP 文本特征
            'clip_image_embed': clip_image_embed, # 原始 CLIP 图像特征
            'logit_scale': self.logit_scale.exp(),
            'modal': modal                        # 当前使用的模态组合
        }


# ============ AverageMeter ============

class AverageMeter(object):
    """Computes and stores the average and current value"""
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

def train_one_epoch(model, dataloader, clip_model, optimizer, scheduler, scaler, criterion, 
                    epoch, device, args, logger, writer=None, modality_dropout_config=None):
    """训练一个 epoch（支持 Modality Dropout + MSE 对齐损失）
    
    与 train_multimodal_dropout.py 的区别：
    - 支持 MSE 对齐损失，在训练初期帮助 fused_emb 对齐到 CLIP 图文特征空间
    - MSE 权重从初始值逐渐降低到 0
    
    MSE 损失说明：
    - 目的：在训练初期帮助 fused_emb 快速对齐到 CLIP 图文特征空间
    - loss_mse = MSE(fused_emb, text_embed) + MSE(fused_emb, image_embed)
    - 前 mse_warmup_epochs 轮使用完整 MSE 权重
    - 之后在 mse_decay_epochs 轮内线性衰减到 0
    
    梯度累积说明（重要：跨累积步骤收集特征以增加负样本）：
    - 实际 batch size = args.batch_size * args.grad_accumulation_steps * world_size
    - 例如：batch_size=32, grad_accumulation_steps=8, 8卡 -> 实际 batch size = 32*8*8 = 2048
    - **负样本数量**：单卡 = batch_size * grad_accumulation_steps - 1
    - 通过在累积期间收集特征，累积完成后统一计算对比损失，实现真正的大 batch 对比学习
    
    Args:
        modality_dropout_config: ModalityDropoutConfig 实例，控制模态组合的采样概率
        writer: TensorBoard SummaryWriter，用于实时记录训练指标
    """
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    
    # 添加 MSE 相关指标
    metric_names = ['loss', 'loss_text', 'loss_image', 'loss_mse', 'loss_mse_text', 'loss_mse_image', 
                    'fused_text_acc', 'fused_image_acc']
    metrics = OrderedDict([(name, AverageMeter(name, ':.4f')) for name in metric_names])
    
    # 计算当前 epoch 的 MSE 权重
    current_mse_weight = compute_mse_weight(epoch, args)
    
    # Modality Dropout 统计
    modal_counts = {m: 0 for m in ['ivt', 'iv', 'vt', 'v', 'it', 'i', 't']}
    
    # 梯度累积配置
    accum_steps = getattr(args, 'grad_accumulation_steps', 1)
    effective_batch_size = args.batch_size * accum_steps * args.world_size
    
    iters_per_epoch = len(dataloader)
    # 实际的优化步数 = 数据迭代次数 / 累积步数
    optim_steps_per_epoch = iters_per_epoch // accum_steps
    
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))
    
    if is_master(args) and accum_steps > 1:
        logger.info(f"Gradient Accumulation: {accum_steps} steps, Effective Batch Size: {effective_batch_size}")
    
    # 输出 Modality Dropout 配置
    if is_master(args) and modality_dropout_config is not None:
        logger.info(f"Modality Dropout enabled: {modality_dropout_config}")
    
    # 输出 MSE 对齐损失配置
    if is_master(args) and getattr(args, 'enable_mse_loss', False):
        logger.info(f"MSE Alignment Loss enabled: weight={current_mse_weight:.4f} "
                   f"(warmup={args.mse_warmup_epochs}, decay={args.mse_decay_epochs}, "
                   f"type={args.mse_decay_type})")

    # 切换到训练模式
    model.train()

    end = time.time()
    # 累积损失（用于日志记录）
    accumulated_loss = 0.0
    accumulated_loss_dict = {k: 0.0 for k in metric_names}
    
    # NaN 处理配置
    # 允许一定数量的连续 NaN batch，超过阈值才停止训练
    # 这样可以容忍偶发的数值不稳定，避免训练意外中断
    nan_count = 0
    max_nan_tolerance = getattr(args, 'max_nan_tolerance', 10)  # 默认允许连续 10 次 NaN
    
    # 使用 tqdm 显示进度条（仅主进程）
    if is_master(args):
        pbar = tqdm(total=iters_per_epoch, desc=f"Epoch {epoch}", 
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    else:
        pbar = None
    
    for data_iter, inputs in enumerate(dataloader):
        # 检查是否是空 batch（所有数据都损坏时会发生）
        if inputs is None:
            logging.warning(f"Skipping batch {data_iter} due to all data being corrupted.")
            continue
        
        # 计算当前是第几个优化步骤（考虑梯度累积）
        optim_step = epoch * optim_steps_per_epoch + data_iter // accum_steps
        is_accumulating = (data_iter + 1) % accum_steps != 0
        is_first_micro_batch = data_iter % accum_steps == 0
        
        # 更新学习率（每个优化步骤更新一次）
        # 非 DeepSpeed 模式：使用 warmup_cosine_lr 返回的函数
        # DeepSpeed 模式：直接调用 deepspeed_lr_scheduler 函数
        if scheduler is not None and not is_accumulating:
            if args.enable_deepspeed:
                # DeepSpeed 模式：调用自定义的学习率调度函数
                # 这个函数会直接修改 optimizer 的学习率
                scheduler(optim_step)
            else:
                # 非 DeepSpeed 模式：调用 warmup_cosine_lr 返回的函数
                scheduler(optim_step)

        # 记录数据加载时间
        data_time.update(time.time() - end)

        # 解析输入数据（与 main.py 格式一致）
        # inputs: (name, name, use_image, texts, pc, image, rgb)
        texts = inputs[3]
        pc = inputs[4]
        image = inputs[5]
        rgb = inputs[6]
        
        # 检查 batch size 是否为 0（部分数据损坏后可能出现）
        if pc.size(0) == 0:
            logging.warning(f"Skipping batch {data_iter} due to empty batch after filtering corrupted data.")
            continue
        
        use_image = inputs[2].reshape(-1)
        loss_masks = use_image.float()

        # 拼接点云坐标和颜色（与 main.py 一致）
        feature = torch.cat((pc, rgb), dim=-1)

        # 如果不使用预计算的 embedding，则实时计算
        if not args.use_embed:
            logging.info('=> encoding captions')
            texts, image = compute_embedding(clip_model, texts, image)

        # 移动到设备（使用 non_blocking 异步传输）
        feature = feature.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        image = image.to(device=device, non_blocking=True)
        
        # 释放 CPU 内存中不再需要的中间变量
        del pc, rgb
        if not args.use_embed:
            del inputs  # 释放原始输入

        # ============ Modality Dropout 采样 ============
        # 为当前 batch 随机采样一个模态组合
        # **重要**：多卡训练时，必须保证所有 GPU 使用相同的模态组合
        # 否则 all_gather 时会因为输出不一致而卡住
        if modality_dropout_config is not None:
            if args.use_distributed and args.world_size > 1:
                # 分布式训练：rank 0 采样，然后广播给其他 GPU
                modal_idx = torch.tensor([0], device=device, dtype=torch.long)
                if args.rank == 0:
                    current_modal = modality_dropout_config.sample()
                    # 将模态组合转换为索引
                    modal_to_idx = {m: i for i, m in enumerate(modality_dropout_config.modals)}
                    modal_idx[0] = modal_to_idx[current_modal]
                # 广播模态索引给所有 GPU
                dist.broadcast(modal_idx, src=0)
                # 将索引转换回模态字符串
                current_modal = modality_dropout_config.modals[modal_idx.item()]
            else:
                # 单卡训练：直接采样
                current_modal = modality_dropout_config.sample()
        else:
            current_modal = 'ivt'  # 默认使用完整三模态
        
        # 更新模态统计
        modal_counts[current_modal] += 1

        # ============ 前向传播和反向传播 ============
        # 注意：DeepSpeed 自动管理梯度累积，不需要手动 no_sync
        # 对于普通 DDP，使用 no_sync 上下文管理器避免在累积期间进行梯度同步
        
        if args.enable_deepspeed:
            # DeepSpeed 模式：DeepSpeed 自动处理梯度累积
            # 每次 backward() + step() 是一个 micro step，DeepSpeed 自动累积
            with amp.autocast(enabled=not args.disable_amp):
                # 前向传播（传入采样的模态组合）
                outputs = model(pc=feature, image=image, text=texts, modal=current_modal)
                
                # 计算对比损失（DeepSpeed 自动处理累积，不需要手动除以 accum_steps）
                loss_dict = criterion(outputs)
                loss = loss_dict['loss']
                
                # ============ 计算 MSE 对齐损失 ============
                if current_mse_weight > 0:
                    fused_embed = outputs['fused_feats']
                    text_embed = outputs.get('clip_text_embed', None)
                    image_embed = outputs.get('clip_image_embed', None)
                    
                    mse_loss_dict = compute_mse_alignment_loss(
                        fused_embed, text_embed, image_embed, args
                    )
                    
                    # 加权添加 MSE 损失
                    mse_total = mse_loss_dict['mse_loss']
                    loss = loss + current_mse_weight * mse_total
                    
                    # 更新 loss_dict 以便日志记录
                    loss_dict['loss_mse'] = mse_loss_dict['mse_loss']
                    loss_dict['loss_mse_text'] = mse_loss_dict['mse_text_loss']
                    loss_dict['loss_mse_image'] = mse_loss_dict['mse_image_loss']
                    loss_dict['loss'] = loss  # 更新总损失
                else:
                    # MSE 权重为 0，添加零值以保持日志格式一致
                    loss_dict['loss_mse'] = torch.tensor(0.0, device=device)
                    loss_dict['loss_mse_text'] = torch.tensor(0.0, device=device)
                    loss_dict['loss_mse_image'] = torch.tensor(0.0, device=device)

            if not math.isfinite(loss.item()):
                nan_count += 1
                logging.warning(f"⚠️ NaN/Inf loss detected at step {step}, batch {data_iter}. Loss={loss.item()}. NaN count: {nan_count}/{max_nan_tolerance}")
                
                # 跳过当前 batch，重置优化器状态
                model.optimizer.zero_grad()
                
                if nan_count >= max_nan_tolerance:
                    logging.error(f"❌ Too many NaN losses ({nan_count}), stopping training")
                    if is_master(args):
                        print(f"\n{'!'*70}")
                        print(f"❌ TRAINING STOPPED: {nan_count} consecutive NaN losses detected!")
                        print(f"Suggestions:")
                        print(f"  1. Lower learning rate (current: {args.lr})")
                        print(f"  2. Increase warmup steps (current: {args.warmup})")
                        print(f"  3. Enable gradient clipping (current: {args.grad_clip})")
                        print(f"  4. Check for data corruption")
                        print(f"{'!'*70}\n")
                    sys.exit(1)
                continue  # 跳过当前 batch
            else:
                nan_count = 0  # 重置 NaN 计数

            # 累积损失（用于日志）
            accumulated_loss += loss.item()
            for k in metric_names:
                if k in loss_dict:
                    accumulated_loss_dict[k] += loss_dict[k].item()

            # DeepSpeed 反向传播 + 优化器更新（DeepSpeed 自动处理累积）
            model.backward(loss)
            model.step()  # DeepSpeed 内部自动处理梯度累积
            
            # 获取 grad_norm 和 loss_scale
            loss_scale_value, grad_norm_value = get_loss_scale_for_deepspeed(model)
            
        else:
            # 普通 DDP / 单卡模式
            # 只在累积周期开始时清零梯度
            if is_first_micro_batch:
                optimizer.zero_grad()
            
            # 使用 no_sync 上下文管理器避免在累积期间进行梯度同步（提高效率）
            context_manager = model.no_sync() if (args.distributed and is_accumulating and hasattr(model, 'no_sync')) else nullcontext()
            
            with context_manager:
                with amp.autocast(enabled=not args.disable_amp):
                    # 前向传播（传入采样的模态组合）
                    outputs = model(pc=feature, image=image, text=texts, modal=current_modal)
                    
                    # 计算对比损失（除以累积步数以保持梯度尺度一致）
                    loss_dict = criterion(outputs)
                    contrastive_loss = loss_dict['loss']
                    
                    # ============ 计算 MSE 对齐损失 ============
                    if current_mse_weight > 0:
                        fused_embed = outputs['fused_feats']
                        text_embed = outputs.get('clip_text_embed', None)
                        image_embed = outputs.get('clip_image_embed', None)
                        
                        mse_loss_dict = compute_mse_alignment_loss(
                            fused_embed, text_embed, image_embed, args
                        )
                        
                        # 加权添加 MSE 损失
                        mse_total = mse_loss_dict['mse_loss']
                        total_loss = contrastive_loss + current_mse_weight * mse_total
                        
                        # 更新 loss_dict 以便日志记录
                        loss_dict['loss_mse'] = mse_loss_dict['mse_loss']
                        loss_dict['loss_mse_text'] = mse_loss_dict['mse_text_loss']
                        loss_dict['loss_mse_image'] = mse_loss_dict['mse_image_loss']
                        loss_dict['loss'] = total_loss  # 更新总损失
                    else:
                        # MSE 权重为 0，添加零值以保持日志格式一致
                        total_loss = contrastive_loss
                        loss_dict['loss_mse'] = torch.tensor(0.0, device=device)
                        loss_dict['loss_mse_text'] = torch.tensor(0.0, device=device)
                        loss_dict['loss_mse_image'] = torch.tensor(0.0, device=device)
                    
                    # 除以累积步数
                    loss = total_loss / accum_steps

                if not math.isfinite(loss.item() * accum_steps):
                    nan_count += 1
                    logging.warning(f"⚠️ NaN/Inf loss detected at step {step}, batch {data_iter}. Loss={loss.item() * accum_steps}. NaN count: {nan_count}/{max_nan_tolerance}")
                    
                    # 跳过当前 batch，重置优化器状态
                    optimizer.zero_grad()
                    if scaler is not None:
                        # 重置 scaler 状态
                        scaler.update()
                    
                    if nan_count >= max_nan_tolerance:
                        logging.error(f"❌ Too many NaN losses ({nan_count}), stopping training")
                        if is_master(args):
                            print(f"\n{'!'*70}")
                            print(f"❌ TRAINING STOPPED: {nan_count} consecutive NaN losses detected!")
                            print(f"Suggestions:")
                            print(f"  1. Lower learning rate (current: {args.lr})")
                            print(f"  2. Increase warmup steps (current: {args.warmup})")
                            print(f"  3. Enable gradient clipping (current: {args.grad_clip})")
                            print(f"  4. Check for data corruption")
                            print(f"{'!'*70}\n")
                        sys.exit(1)
                    continue  # 跳过当前 batch
                else:
                    nan_count = 0  # 重置 NaN 计数

                # 累积损失（用于日志）
                accumulated_loss += loss.item() * accum_steps
                for k in metric_names:
                    if k in loss_dict:
                        accumulated_loss_dict[k] += loss_dict[k].item()

                # 反向传播（每个 micro-batch 都进行）
                if scaler is not None:
                    # 普通 AMP 模式
                    scaler.scale(loss).backward()
                else:
                    # 普通 FP32 模式
                    loss.backward()

        # 只在累积完成后进行优化器更新（仅非 DeepSpeed 模式）
        # DeepSpeed 模式下，step() 已经在上面调用了
        if not args.enable_deepspeed:
            grad_norm_value = 0.0
            loss_scale_value = 0.0
            if not is_accumulating:
                if scaler is not None:
                    # 普通 AMP 模式 - 梯度裁剪和优化器更新
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        # 在 clip 之前计算 grad_norm（反映真实梯度大小）
                        grad_norm_value = get_grad_norm_(model.parameters()).item()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    else:
                        scaler.unscale_(optimizer)
                        grad_norm_value = get_grad_norm_(model.parameters()).item()
                    
                    scaler.step(optimizer)
                    scaler.update()
                    loss_scale_value = scaler.get_scale()
                else:
                    # 普通 FP32 模式
                    # 在 clip 之前计算 grad_norm
                    grad_norm_value = get_grad_norm_(model.parameters()).item()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    
                    optimizer.step()
                    loss_scale_value = 0.0
        
        # clamp logit scale to [0, 100]（与 main.py 一致）
        # DeepSpeed 模式：每个 micro step 都执行（因为每个 step 都更新参数）
        # 非 DeepSpeed 模式：只在累积完成后执行
        should_update_logit_scale = args.enable_deepspeed or (not is_accumulating)
        if should_update_logit_scale:
            from utils.utils import get_model
            get_model(model).logit_scale.data.clamp_(0, 4.6052)
            logit_scale = get_model(model).logit_scale.exp().item()

        # 更新统计（每个 accum_steps 周期汇总一次）
        # 注意：即使 DeepSpeed 每个 micro step 都 step()，我们仍然按 accum_steps 周期统计
        if not is_accumulating:
            for k in metrics:
                if k in accumulated_loss_dict:
                    metrics[k].update(accumulated_loss_dict[k] / accum_steps, args.batch_size * accum_steps)
            
            # 重置累积损失
            accumulated_loss = 0.0
            accumulated_loss_dict = {k: 0.0 for k in metric_names}

        # 记录时间
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        # 打印日志（每 log_interval 个优化步骤打印一次）
        if not is_accumulating and (data_iter // accum_steps) % args.log_interval == 0:
            # grad_norm 和 loss_scale 已经在优化器更新前计算好了
            loss_scale = loss_scale_value
            grad_norm = grad_norm_value
            
            if is_master(args):
                # 获取当前学习率
                if args.enable_deepspeed:
                    current_lr = model.optimizer.param_groups[0]['lr']
                else:
                    current_lr = optimizer.param_groups[0]['lr']
                
                # ========== 终端直接输出（更醒目）==========
                print(f"\n{'='*60}")
                print(f"📊 Epoch [{epoch}/{args.epochs}] | Step [{data_iter // accum_steps}/{optim_steps_per_epoch}]")
                print(f"   📉 Total Loss: {metrics['loss'].val:.4f} (avg: {metrics['loss'].avg:.4f})")
                print(f"   📝 Text Loss:  {metrics.get('loss_text', AverageMeter('', '')).val:.4f}")
                print(f"   🖼️  Image Loss: {metrics.get('loss_image', AverageMeter('', '')).val:.4f}")
                # MSE 对齐损失输出
                if current_mse_weight > 0:
                    print(f"   🔗 MSE Loss:   {metrics.get('loss_mse', AverageMeter('', '')).val:.4f} (weight: {current_mse_weight:.4f})")
                    print(f"      └─ Text:   {metrics.get('loss_mse_text', AverageMeter('', '')).val:.4f}")
                    print(f"      └─ Image:  {metrics.get('loss_mse_image', AverageMeter('', '')).val:.4f}")
                print(f"   🎯 Text Acc:   {metrics.get('fused_text_acc', AverageMeter('', '')).val:.2f}% (avg: {metrics.get('fused_text_acc', AverageMeter('', '')).avg:.2f}%)")
                print(f"   🎯 Image Acc:  {metrics.get('fused_image_acc', AverageMeter('', '')).val:.2f}% (avg: {metrics.get('fused_image_acc', AverageMeter('', '')).avg:.2f}%)")
                print(f"   📈 LR: {current_lr:.2e} | Grad Norm: {grad_norm:.4f} | Loss Scale: {loss_scale}")
                print(f"   ⏱️  Time: {batch_time.val:.2f}s/batch | Data: {data_time.val:.2f}s")
                print(f"   💾 GPU Memory: {mem.val:.1f} GB")
                print(f"{'='*60}")
                
                # 日志记录
                logging.info(f"OptimStep {data_iter // accum_steps}/{optim_steps_per_epoch} - "
                           f"DataIter {data_iter}/{iters_per_epoch} - "
                           f"Loss: {metrics['loss'].val:.4f} - "
                           f"MSE: {metrics.get('loss_mse', AverageMeter('', '')).val:.4f} (w={current_mse_weight:.4f}) - "
                           f"Loss Scale: {loss_scale} - Grad Norm: {grad_norm:.4f}")
                
                # 实时写入 TensorBoard（每个 log_interval 写入一次）
                if writer is not None:
                    global_step = epoch * optim_steps_per_epoch + data_iter // accum_steps
                    writer.add_scalar('train_step/loss', metrics['loss'].val, global_step)
                    writer.add_scalar('train_step/loss_text', metrics.get('loss_text', AverageMeter('', '')).val, global_step)
                    writer.add_scalar('train_step/loss_image', metrics.get('loss_image', AverageMeter('', '')).val, global_step)
                    # MSE 相关指标
                    writer.add_scalar('train_step/loss_mse', metrics.get('loss_mse', AverageMeter('', '')).val, global_step)
                    writer.add_scalar('train_step/loss_mse_text', metrics.get('loss_mse_text', AverageMeter('', '')).val, global_step)
                    writer.add_scalar('train_step/loss_mse_image', metrics.get('loss_mse_image', AverageMeter('', '')).val, global_step)
                    writer.add_scalar('train_step/mse_weight', current_mse_weight, global_step)
                    writer.add_scalar('train_step/fused_text_acc', metrics.get('fused_text_acc', AverageMeter('', '')).val, global_step)
                    writer.add_scalar('train_step/fused_image_acc', metrics.get('fused_image_acc', AverageMeter('', '')).val, global_step)
                    writer.add_scalar('train_step/grad_norm', grad_norm, global_step)
                    writer.add_scalar('train_step/loss_scale', loss_scale if loss_scale else 0, global_step)
                    writer.add_scalar('train_step/logit_scale', logit_scale, global_step)  # 温度参数
                    if scheduler is not None and not args.enable_deepspeed:
                        writer.add_scalar('train_step/lr', optimizer.param_groups[0]['lr'], global_step)
                    elif args.enable_deepspeed:
                        writer.add_scalar('train_step/lr', model.optimizer.param_groups[0]['lr'], global_step)
                    writer.flush()  # 立即写入磁盘
                    
            progress.display(data_iter)
        
        # 更新进度条
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({
                'loss': f"{metrics['loss'].avg:.4f}",
                'step': f"{data_iter // accum_steps}/{optim_steps_per_epoch}"
            })

    # 关闭进度条
    if pbar is not None:
        pbar.close()
    
    # ============ 检查是否处理了完整的 epoch ============
    # 如果 DataLoader 提前终止（如 worker 崩溃），data_iter 会小于预期
    actual_iters = data_iter + 1  # data_iter 是从 0 开始的索引
    expected_iters = iters_per_epoch
    completion_ratio = actual_iters / expected_iters
    
    if completion_ratio < 0.99:  # 允许 1% 的误差（由于 drop_last 等原因）
        warning_msg = (
            f"⚠️ WARNING: Epoch only processed {actual_iters}/{expected_iters} batches "
            f"({completion_ratio*100:.1f}%)! "
            f"This may indicate a DataLoader worker crash or data corruption."
        )
        logging.warning(warning_msg)
        if is_master(args):
            print(f"\n{'!'*70}")
            print(warning_msg)
            print(f"{'!'*70}\n")
    else:
        if is_master(args):
            logging.info(f"Epoch completed: {actual_iters}/{expected_iters} batches processed ({completion_ratio*100:.1f}%)")
    
    # ============ 输出 Modality Dropout 统计信息 ============
    if is_master(args) and modality_dropout_config is not None:
        total_samples = sum(modal_counts.values())
        if total_samples > 0:
            print(f"\n{'─'*70}")
            print(f"📊 Modality Dropout Statistics for Epoch {epoch}:")
            for modal, count in sorted(modal_counts.items(), key=lambda x: -x[1]):
                if count > 0:
                    pct = count / total_samples * 100
                    bar_len = int(pct / 2)  # 最长 50 个字符
                    bar = '█' * bar_len
                    print(f"   {modal:>4}: {count:>6} ({pct:>5.1f}%) {bar}")
            print(f"{'─'*70}\n")
            logger.info(f"Modal counts: {modal_counts}")
    
    progress.synchronize()
    
    # 确保 logit_scale 在最后一次累积未完成时也有有效值
    if 'logit_scale' not in dir():
        from utils.utils import get_model
        logit_scale = get_model(model).logit_scale.exp().item()
    
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[-1]['lr'] if not args.enable_deepspeed else model.optimizer.param_groups[-1]['lr'],
            'logit_scale': logit_scale,
            'effective_batch_size': args.batch_size * accum_steps * args.world_size,
            'completion_ratio': completion_ratio,
            'modal_counts': modal_counts,  # 添加模态统计
            'mse_weight': current_mse_weight}  # 添加当前 MSE 权重


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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
    """
    零样本分类测试核心函数（与 main.py 完全一致）
    
    Args:
        test_loader: 测试数据加载器
        validate_dataset_name: 验证数据集名称
        model: 多模态训练包装器模型
        clip_model: CLIP 模型（用于编码文本标签）
        tokenizer: 分词器
        args: 参数
        test_data: 测试数据类型标识
    
    Returns:
        dict: 包含 acc1, acc3, acc5 的字典
    """
    from utils import utils
    
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f') 
    top3 = AverageMeter('Acc@3', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, top1, top3, top5],
        prefix='Test: ')

    # 切换到评估模式
    model.eval()

    # 加载模板和标签（与 main.py 一致）
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
        per_class_correct_top3 = collections.defaultdict(int)
        per_class_correct_top5 = collections.defaultdict(int)

        for i, (pc, target, target_name, rgb) in enumerate(test_loader):
            for name in target_name:
                per_class_stats[name] += 1

            pc = pc.to(device=args.device, non_blocking=True)
            rgb = rgb.to(device=args.device, non_blocking=True)
            feature = torch.cat((pc, rgb), dim=-1)
            target = target.to(device=args.device, non_blocking=True)

            # 使用多模态模型的点云编码器
            # 获取底层的 uni3d 模型
            uni3d_model = utils.get_model(model).uni3d
            
            # 编码点云特征
            # pc_features = uni3d_model.encode_pc(feature)
            pc_features, _, _ = uni3d_model.encode_multimodal(point = feature)
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)

            # 计算余弦相似度作为 logits
            logits_per_pc = pc_features.float() @ text_features.float().t()

            # 计算准确率
            (acc1, acc3, acc5), correct = accuracy(logits_per_pc, target, topk=(1, 3, 5))
            acc1, acc3, acc5 = utils.scaled_all_reduce([acc1, acc3, acc5])
            top1.update(acc1.item(), pc.size(0))
            top3.update(acc3.item(), pc.size(0))
            top5.update(acc5.item(), pc.size(0))

            # 记录时间
            batch_time.update(time.time() - end)
            end = time.time()

            top1_accurate = correct[:1].squeeze()
            top3_accurate = correct[:3].float().sum(0, keepdim=True).squeeze()
            top5_accurate = correct[:5].float().sum(0, keepdim=True).squeeze()
            for idx, name in enumerate(target_name):
                if top1_accurate[idx].item():
                    per_class_correct_top1[name] += 1
                if top3_accurate[idx].item():
                    per_class_correct_top3[name] += 1
                if top5_accurate[idx].item():
                    per_class_correct_top5[name] += 1

            if i % args.print_freq == 0:
                progress.display(i)

        # 计算每个类别的准确率
        top1_accuracy_per_class = {}
        top3_accuracy_per_class = {}
        top5_accuracy_per_class = {}
        for name in per_class_stats.keys():
            top1_accuracy_per_class[name] = per_class_correct_top1[name] / per_class_stats[name]
            top3_accuracy_per_class[name] = per_class_correct_top3[name] / per_class_stats[name]
            top5_accuracy_per_class[name] = per_class_correct_top5[name] / per_class_stats[name]

        top1_accuracy_per_class = collections.OrderedDict(top1_accuracy_per_class)
        top3_accuracy_per_class = collections.OrderedDict(top3_accuracy_per_class)
        top5_accuracy_per_class = collections.OrderedDict(top5_accuracy_per_class)
        logging.info(','.join(top1_accuracy_per_class.keys()))
        logging.info(','.join([str(value) for value in top1_accuracy_per_class.values()]))
        logging.info(','.join([str(value) for value in top3_accuracy_per_class.values()]))        
        logging.info(','.join([str(value) for value in top5_accuracy_per_class.values()]))
    
    progress.synchronize()
    logging.info(f'0-shot * Acc@1 {top1.avg:.3f} Acc@3 {top3.avg:.3f} Acc@5 {top5.avg:.3f}')
    return {'acc1': top1.avg, 'acc3': top3.avg, 'acc5': top5.avg}


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, args, is_best=True, best_acc1=0.0):
    """保存检查点"""
    from utils.utils import get_model
    
    if args.enable_deepspeed:
        # DeepSpeed 检查点保存
        deepspeed_checkpoint_path = os.path.join(args.output_dir, "checkpoints")
        os.makedirs(deepspeed_checkpoint_path, exist_ok=True)
        client_state = {
            'epoch': epoch,
            'best_acc1': best_acc1,
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
        # 普通检查点保存
        # 获取底层模型状态
        model_state = get_model(model).state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'best_acc1': best_acc1,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler and hasattr(scheduler, 'state_dict') else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'args': args
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存周期性检查点
        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)


# ============ 主函数 ============

def get_args():
    """获取命令行参数（参考 main.py 的参数设置，添加 DeepSpeed 支持）"""
    parser = argparse.ArgumentParser(description='Multimodal Uni3D Training')
    
    # 数据参数（与 main.py 一致）
    parser.add_argument('--pretrain_dataset_name', type=str, default='ensembled_embedding', help='预训练数据集名称')
    parser.add_argument('--validate_dataset_name', type=str, default='modelnet40_openshape', help='验证数据集名称')
    parser.add_argument('--validate_dataset_name_lvis', type=str, default='objaverse_lvis_openshape', help='LVIS验证数据集名称')
    parser.add_argument('--validate_dataset_name_scanobjnn', type=str, default='scanobjnn_openshape', help='ScanObjNN验证数据集名称')
    parser.add_argument('--validate_dataset_prompt', type=str, default='modelnet40_64', help='验证提示模板')
    parser.add_argument('--pretrain_dataset_prompt', type=str, default='modelnet40_64', help='验证提示模板')#??

    parser.add_argument('--npoints', type=int, default=10000, help='点云采样点数')
    parser.add_argument('--workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--use_height', action='store_true', help='是否使用高度特征')
    parser.add_argument('--openshape_setting', action='store_true', help='使用OpenShape设置')
    parser.add_argument('--use_lvis', action='store_true', help='使用LVIS数据集')
    parser.add_argument('--use_embed', action='store_true', help='使用预计算的embedding')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='create_uni3d', help='模型名称')
    parser.add_argument('--pc_model', type=str, default='eva_giant_patch14_560.m30m_ft_in22k_in1k', help='点云 Transformer 模型')
    parser.add_argument('--pretrained_pc', type=str, default='', help='预训练点云模型路径')
    parser.add_argument('--clip_model', type=str, default='EVA02-E-14-plus', help='CLIP 模型')
    parser.add_argument('--clip_model_path', type=str, default='', help='CLIP 模型权重路径')
    parser.add_argument('--pretrained', type=str, default='openai', help='CLIP 预训练权重')
    parser.add_argument('--embed_dim', type=int, default=1280, help='输出嵌入维度')
    parser.add_argument('--pc_feat_dim', type=int, default=1408, help='点云特征维度')
    parser.add_argument('--pc_encoder_dim', type=int, default=512, help='点云编码器维度')
    parser.add_argument('--group_size', type=int, default=64, help='点云分组大小')
    parser.add_argument('--num_group', type=int, default=512, help='点云分组数量')
    parser.add_argument('--drop_path_rate', type=float, default=0.2, help='DropPath 率')
    parser.add_argument('--patch_dropout', type=float, default=0.5, help='Patch Dropout 率')
    
    # 新增：融合和特征选项
    parser.add_argument('--use_fusion_blocks', action='store_true', default=True, 
                        help='是否启用三流融合模块（默认启用）')
    parser.add_argument('--no_fusion_blocks', action='store_true', 
                        help='禁用三流融合模块，直接拼接特征')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=48, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--point_lr', type=float, default=1e-3, help='点云模型学习率')
    parser.add_argument('--wd', type=float, default=0.1, help='权重衰减')
    parser.add_argument('--point_wd', type=float, default=0.1, help='点云模型权重衰减')
    parser.add_argument('--ld', type=float, default=1.0, help='层级衰减')
    parser.add_argument('--point_ld', type=float, default=0.95, help='点云模型层级衰减')
    parser.add_argument('--warmup', type=int, default=10000, help='预热步数')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='梯度裁剪')
    parser.add_argument('--grad_clip_norm', type=float, default=5.0, help='梯度裁剪（兼容 DeepSpeed）')
    parser.add_argument('--smoothing', type=float, default=0.0, help='标签平滑')
    parser.add_argument('--grad_accumulation_steps', type=int, default=1, help='梯度累积步数')
    parser.add_argument('--max_nan_tolerance', type=int, default=10, 
                        help='允许的最大连续 NaN loss 次数，超过则停止训练（默认 10）')
    
    # Modality Dropout 参数
    parser.add_argument('--enable_modality_dropout', action='store_true', default=True,
                        help='启用 Modality Dropout（默认启用）')
    parser.add_argument('--no_modality_dropout', action='store_true',
                        help='禁用 Modality Dropout，使用完整 ivt 模态')
    parser.add_argument('--modal_prob_ivt', type=float, default=0.50, 
                        help='ivt 模态组合出现的概率（默认 50%%）')
    parser.add_argument('--modal_prob_iv', type=float, default=0.15, 
                        help='iv 模态组合出现的概率（默认 15%%）')
    parser.add_argument('--modal_prob_vt', type=float, default=0.15, 
                        help='vt 模态组合出现的概率（默认 15%%）')
    parser.add_argument('--modal_prob_v', type=float, default=0.10, 
                        help='仅 v 模态出现的概率（默认 10%%）')
    parser.add_argument('--modal_prob_it', type=float, default=0.05, 
                        help='it 模态组合出现的概率（默认 5%%，应该非常小）')
    parser.add_argument('--modal_prob_i', type=float, default=0.025, 
                        help='仅 i 模态出现的概率（默认 2.5%%）')
    parser.add_argument('--modal_prob_t', type=float, default=0.025, 
                        help='仅 t 模态出现的概率（默认 2.5%%）')
    
    # 损失参数
    parser.add_argument('--text_weight', type=float, default=1.0, help='文本损失权重')
    parser.add_argument('--image_weight', type=float, default=1.0, help='图像损失权重')
    
    # MSE 对齐损失参数
    parser.add_argument('--enable_mse_loss', action='store_true', default=True,
                        help='启用 MSE 对齐损失（默认启用）')
    parser.add_argument('--no_mse_loss', action='store_true',
                        help='禁用 MSE 对齐损失')
    parser.add_argument('--mse_weight', type=float, default=1.0, 
                        help='MSE 损失的初始权重（默认 1.0）')
    parser.add_argument('--mse_text_weight', type=float, default=1.0, 
                        help='MSE 文本对齐损失权重（默认 1.0）')
    parser.add_argument('--mse_image_weight', type=float, default=1.0, 
                        help='MSE 图像对齐损失权重（默认 1.0）')
    parser.add_argument('--mse_warmup_epochs', type=int, default=2, 
                        help='MSE 损失保持完整权重的 epoch 数（默认 2）')
    parser.add_argument('--mse_decay_epochs', type=int, default=3, 
                        help='MSE 损失权重衰减到 0 所需的 epoch 数（默认 3）')
    parser.add_argument('--mse_decay_type', type=str, default='linear', 
                        choices=['linear', 'cosine', 'exponential'],
                        help='MSE 损失权重衰减方式（默认 linear）')
    
    # DeepSpeed 参数
    parser.add_argument('--enable_deepspeed', action='store_true', default=False, help='启用 DeepSpeed')
    parser.add_argument('--zero_stage', type=int, default=1, help='ZeRO 优化阶段 (1, 2, 或 3)')
    parser.add_argument('--optimizer', type=str, default='adamw', help='优化器类型')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.95, help='Adam beta2')
    parser.add_argument('--eps', type=float, default=1e-8, help='Adam epsilon')
    parser.add_argument('--grad_checkpointing', action='store_true', default=False, help='启用梯度检查点')
    
    # 其他参数
    parser.add_argument('--name', type=str, default=None, help='实验名称')
    parser.add_argument('--output_dir', type=str, default='./output_multimodal', help='输出目录')
    parser.add_argument('--logs', type=str, default='./logs', help='日志目录')
    parser.add_argument('--log_local', action='store_true', default=False, help='本地日志')
    parser.add_argument('--log_interval', type=int, default=10, help='日志打印间隔')
    parser.add_argument('--print_freq', type=int, default=10, help='打印频率')
    parser.add_argument('--save_interval', type=int, default=10, help='保存间隔')
    parser.add_argument('--save_frequency', type=int, default=1, help='保存频率')
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度')
    parser.add_argument('--disable_amp', action='store_true', help='禁用AMP')
    parser.add_argument('--precision', type=str, default='amp', help='精度设置')
    parser.add_argument('--seed', type=int, default=4096, help='随机种子')
    parser.add_argument('--use_distributed', action='store_true', help='使用分布式训练')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    
    # TensorBoard 参数
    parser.add_argument('--tensorboard', action='store_true', default=True, help='启用 TensorBoard 日志')
    parser.add_argument('--tensorboard_dir', type=str, default=None, help='TensorBoard 日志目录（默认为 output_dir/tensorboard）')
    
    # 分布式参数
    parser.add_argument('--local_rank', type=int, default=0, help='本地进程rank')
    parser.add_argument('--rank', type=int, default=0, help='全局进程rank')
    parser.add_argument('--world_size', type=int, default=1, help='世界大小')
    parser.add_argument('--dist_url', type=str, default='env://', help='分布式URL')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='分布式后端')
    parser.add_argument('--no_set_device_rank', action='store_true', default=False, help='不设置设备rank')
    
    args = parser.parse_args()
    
    # 处理 fusion_blocks 参数
    if args.no_fusion_blocks:
        args.use_fusion_blocks = False
    
    # 处理 modality_dropout 参数
    if args.no_modality_dropout:
        args.enable_modality_dropout = False
    
    # 处理 mse_loss 参数
    if args.no_mse_loss:
        args.enable_mse_loss = False
    
    # 同步 grad_clip 参数
    if args.grad_clip_norm is None:
        args.grad_clip_norm = args.grad_clip
    
    return args


def init_deepspeed(args):
    """初始化 DeepSpeed
    
    只在实际启用 DeepSpeed 时才导入，避免 apex/FusedAdam 相关警告
    """
    ds_init = None
    if args.enable_deepspeed:
        try:
            # 在导入 deepspeed 之前设置环境变量，禁用 FusedAdam/FusedLAMB
            # 这样即使没有安装 apex，也不会报错
            os.environ['DS_BUILD_FUSED_ADAM'] = '0'
            os.environ['DS_BUILD_FUSED_LAMB'] = '0'
            
            import deepspeed
            from deepspeed import DeepSpeedConfig
            os.environ['ENV_TYPE'] = "deepspeed"
            ds_init = deepspeed.initialize
            logging.info("DeepSpeed enabled successfully (using PyTorch native optimizers)")
        except ImportError:
            logging.error("Please 'pip install deepspeed>=0.9.4'")
            sys.exit(1)
    else:
        os.environ['ENV_TYPE'] = "pytorch"
    return ds_init


def get_scheduler(optimizer, args, total_steps):
    """获取学习率调度器（与 main.py 的 warmup_cosine_lr 类似）"""
    from utils.scheduler import warmup_cosine_lr
    return warmup_cosine_lr(optimizer, args, total_steps)


def main():
    """主函数（参考 main.py 的结构，支持 DeepSpeed）"""
    args = get_args()
    
    # 初始化 DeepSpeed
    ds_init = init_deepspeed(args)
    
    # 生成实验名称
    if args.name is None:
        args.name = '-'.join([
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            f"model_multimodal",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
        ])
    else:
        args.name = '-'.join([
            args.name,
            datetime.now().strftime("%Y_%m_%d-%H")
        ])
    
    # 创建 DeepSpeed 配置文件目录
    if ds_init is not None:
        dsconfig_path = os.path.join(os.getcwd(), "dsconfig", args.name)
        os.makedirs(dsconfig_path, exist_ok=True)
        create_deepspeed_config(args)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # CUDA 设置（性能优化）
    if torch.cuda.is_available():
        # TF32 优化：在 Ampere+ GPU 上启用 TensorFloat-32，提升矩阵乘法速度
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # cuDNN 自动调优：自动选择最优卷积算法（第一次运行稍慢，后续更快）
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Flash Attention 优化（PyTorch 2.0+）
        # 启用更高效的注意力实现
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    # 初始化分布式环境
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    
    if args.use_distributed or args.enable_deepspeed:
        device = init_distributed_device(args)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    args.device = device
    
    # 设置日志（在分布式初始化之后，确保能正确获取 rank）
    # 只有 rank 0 才写入文件日志，避免多进程同时写入导致内容丢失
    logger = setup_logging(args.output_dir, rank=args.rank)
    logger.info(f"Arguments: {args}")
    logger.info(f"Using device: {device}")
    
    if args.distributed:
        logger.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}. '
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.'
        )
    else:
        logger.info(f'Running with a single process. Device {args.device}.')
    
    if args.enable_deepspeed:
        logger.info(f'DeepSpeed enabled with ZeRO stage {args.zero_stage}')
    
    # 设置随机种子
    random_seed(args.seed, args.rank)
    
    # ============ CLIP 模型处理 ============
    # 当 use_embed=True 时，训练只需要预提取的特征，不需要 CLIP 模型
    # CLIP 模型仅在验证时用于编码类别文本标签，进行零样本分类评估
    # 为了节省训练时的显存，延迟加载 CLIP 模型到验证阶段
    
    if args.use_embed:
        logger.info("=> using precomputed embeddings for training")
        logger.info("=> CLIP model will be loaded on-demand during validation to save GPU memory")
        clip_model = None  # 训练时不加载 CLIP 模型
    else:
        # 不使用预提取特征时，需要实时编码，必须加载 CLIP 模型
        logger.info("=> create clip model for real-time encoding...")
        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name=args.clip_model, 
            pretrained=args.pretrained
        )
        clip_model.to(device)
        clip_model.eval()  # CLIP 始终在 eval 模式
        for param in clip_model.parameters():
            param.requires_grad = False  # 冻结 CLIP 参数
    
    # ============ 创建 Uni3D Multimodal 模型 ============
    logger.info("=> creating model: uni3d_multimodal")
    logger.info(f"   use_embed: {args.use_embed}")
    logger.info(f"   use_fusion_blocks: {args.use_fusion_blocks}")
    uni3d_model = create_uni3d_multimodal(args)
    
    # 清理 checkpoint 加载后的 CPU 内存
    import gc
    gc.collect()
    
    uni3d_model.to(device)
    
    # 包装模型
    model = MultimodalTrainingWrapper(uni3d_model, clip_model, args)
    model = model.to(device)
    model_without_ddp = model
    
    # 清理 CUDA 缓存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params / 1e6:.2f}M")
    logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # ============ 启用梯度检查点（节省显存）============
    if args.grad_checkpointing:
        logger.info("=> Enabling gradient checkpointing to save memory")
        # 对 timm 模型（点云 transformer）启用梯度检查点
        point_visual = model_without_ddp.uni3d.point_encoder.visual
        if hasattr(point_visual, 'set_grad_checkpointing'):
            point_visual.set_grad_checkpointing(enable=True)
            logger.info("   Point transformer: gradient checkpointing enabled via set_grad_checkpointing()")
        elif hasattr(point_visual, 'grad_checkpointing'):
            point_visual.grad_checkpointing = True
            logger.info("   Point transformer: gradient checkpointing enabled via attribute")
        else:
            # 手动为 transformer blocks 启用梯度检查点
            try:
                from torch.utils.checkpoint import checkpoint
                logger.info("   Point transformer: will use manual gradient checkpointing")
            except ImportError:
                logger.warning("   Gradient checkpointing not available for point transformer")
    
    # 打印显存使用情况（帮助调试 OOM）
    if torch.cuda.is_available():
        logger.info(f"GPU memory allocated after model creation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # ============ 创建数据集（与 main.py 完全一致）============
    logger.info("=> creating dataset")
    tokenizer = SimpleTokenizer()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.ToTensor(),
        normalize
    ])
    
    train_dataset = get_dataset(train_transform, tokenizer, args, 'train')
    val_dataset = get_dataset(None, tokenizer, args, 'val')
    # val_dataset_lvis = get_dataset(None, tokenizer, args, 'val_lvis')
    val_dataset_scanobjnn = get_dataset(None, tokenizer, args, 'val_scanobjnn')
    
    # 创建采样器（与 main.py 一致）
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        # val_lvis_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset_lvis)
        val_scanobjnn_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset_scanobjnn)
    else:
        train_sampler = None
        val_sampler = None
        val_lvis_sampler = None
        val_scanobjnn_sampler = None
    
    # 创建 DataLoader（优化版本）
    # 性能优化：
    # - persistent_workers=True: 避免每个 epoch 重新创建 worker 进程
    # - prefetch_factor=2: 每个 worker 预取的 batch 数量
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=customized_collate_fn,
        persistent_workers=True if args.workers > 0 else False,
        prefetch_factor=2 if args.workers > 0 else None
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=(val_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False,
        persistent_workers=True if args.workers > 0 else False,
        prefetch_factor=2 if args.workers > 0 else None
    )
    
    # val_lvis_loader = torch.utils.data.DataLoader(
    #     val_dataset_lvis,
    #     batch_size=args.batch_size,
    #     shuffle=(val_lvis_sampler is None),
    #     num_workers=args.workers,
    #     pin_memory=True,
    #     sampler=val_lvis_sampler,
    #     drop_last=False
    # )
    
    val_scanobjnn_loader = torch.utils.data.DataLoader(
        val_dataset_scanobjnn,
        batch_size=args.batch_size,
        shuffle=(val_scanobjnn_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_scanobjnn_sampler,
        drop_last=False,
        persistent_workers=True if args.workers > 0 else False,
        prefetch_factor=2 if args.workers > 0 else None
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples (ModelNet40): {len(val_dataset)}")
    # logger.info(f"Val samples (LVIS): {len(val_dataset_lvis)}")
    logger.info(f"Val samples (ScanObjNN): {len(val_dataset_scanobjnn)}")
    
    # ============ 创建优化器（支持 DeepSpeed）============
    logger.info("Creating optimizer and scheduler...")
    
    optimizer = None
    scaler = None
    scheduler = None
    
    # ============ 创建 Modality Dropout 配置 ============
    if args.enable_modality_dropout:
        modality_dropout_config = get_modality_dropout_config(args)
        logger.info(f"Modality Dropout enabled: {modality_dropout_config}")
    else:
        modality_dropout_config = None
        logger.info("Modality Dropout disabled, using full ivt modality")
    
    # ============ 创建损失函数 ============
    # 使用支持 Modality Dropout 的损失函数
    # 注意：投影层已移至模型定义中（方案 B），损失函数不再包含可训练参数
    if args.enable_modality_dropout:
        criterion = ModalityDropoutLoss(
            text_weight=args.text_weight,
            image_weight=args.image_weight,
            use_distributed=args.distributed
        )
        logger.info("Using ModalityDropoutLoss for dynamic modality combinations")
    else:
        criterion = Uni3dMultimodalLoss(
            text_weight=args.text_weight,
            image_weight=args.image_weight,
            use_distributed=args.distributed
        )
        logger.info("Using Uni3dMultimodalLoss for full ivt modality")
    
    criterion = criterion.to(device)
    logger.info("Criterion created (projection layer is now in model)")
    
    # 创建优化器（分层学习率）
    # 1. point_encoder：预训练模块，使用较小学习率
    # 2. 其他模块（融合块、投影层、fused_to_clip_proj 等）：新初始化，使用较大学习率
    # 3. 温度参数：使用最小学习率
    # 注意：fused_to_clip_proj 投影层现在在模型中，会被包含在 "其他模块" 参数组中
    param_groups = [
        # 点云编码器：使用较小学习率（已预训练，微调即可）
        {'params': [p for p in model_without_ddp.uni3d.point_encoder.parameters() if p.requires_grad], 
            'lr': args.point_lr, 'weight_decay': args.point_wd},
        # 其他模块（融合块、投影层、路由器、fused_to_clip_proj 等）：使用较大学习率（新初始化）
        {'params': [p for n, p in model_without_ddp.uni3d.named_parameters() 
                    if 'point_encoder' not in n and p.requires_grad], 
            'lr': args.lr, 'weight_decay': args.wd},
        # 温度参数：使用较大学习率（对比学习中温度参数很重要）
        {'params': [model_without_ddp.logit_scale], 
            'lr': args.lr, 'weight_decay': 0.0}
    ]
    
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        weight_decay=args.wd,
        betas=(args.beta1, args.beta2)
    )
    if args.enable_deepspeed:
        # DeepSpeed 模式
        
        # 使用 DeepSpeed 初始化
        model, optimizer, _, _ = ds_init(
            args=args,
            model=model,
            optimizer = optimizer,
            model_parameters=param_groups,
            dist_init_required=not args.distributed,
        )
        model_without_ddp = model.module if hasattr(model, 'module') else model
        scaler = None  # DeepSpeed 管理自己的混合精度
        
        logger.info(f"DeepSpeed initialized with ZeRO stage {args.zero_stage}")
        
        # DeepSpeed 学习率调度：
        # DeepSpeedZeroOptimizer 不是标准 PyTorch Optimizer，不能直接用 LambdaLR
        # 解决方案：创建一个自定义的 scheduler 函数，在训练循环中手动调整学习率
        
        # 计算总步数
        iters_per_epoch = len(train_loader)
        optim_steps_per_epoch = iters_per_epoch // args.grad_accumulation_steps
        total_optim_steps = optim_steps_per_epoch * args.epochs
        warmup_steps = args.warmup
        base_lr = args.lr
        
        logger.info(f"DeepSpeed LR Scheduler: warmup_steps={warmup_steps}, total_optim_steps={total_optim_steps}")
        
        # 保存每个参数组的初始学习率，用于后续按比例调整
        # 参数组学习率：
        # - 组0 (point_encoder): point_lr
        # - 组1 (其他 uni3d 模块): lr
        # - 组2 (logit_scale): lr
        initial_lrs = [args.point_lr, args.lr, args.lr]
        logger.info(f"DeepSpeed initial LRs for param groups: {initial_lrs}")
        
        def deepspeed_lr_scheduler(current_step):
            """计算 DeepSpeed 的学习率（Cosine decay with warmup）并更新
            
            关键：保持每个参数组的初始学习率比例，实现分层学习率
            - point_encoder: point_lr * lr_scale
            - 其他模块: lr * lr_scale
            """
            if current_step < warmup_steps:
                # Warmup 阶段：线性增加
                lr_scale = float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine decay 阶段
                progress = float(current_step - warmup_steps) / float(max(1, total_optim_steps - warmup_steps))
                lr_scale = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
            # 按参数组的初始学习率比例更新
            for i, param_group in enumerate(model.optimizer.param_groups):
                # 使用该参数组的初始学习率乘以衰减因子
                group_base_lr = initial_lrs[i] if i < len(initial_lrs) else base_lr
                param_group['lr'] = group_base_lr * lr_scale
            
            return base_lr * lr_scale  # 返回主学习率用于日志记录
        
        # 将函数作为 scheduler 传递（在训练循环中会被调用）
        scheduler = deepspeed_lr_scheduler
    else:
        # 普通 DDP 模式
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, 
                device_ids=[device]
            )
            model_without_ddp = model.module
        
 
        # 混合精度
        scaler = amp.GradScaler() if args.use_amp or args.precision == "amp" else None
    
    # 学习率调度器（非 DeepSpeed 模式）
    total_steps = len(train_loader) * args.epochs
    logger.info(f"total_steps: {total_steps}")
    if not args.enable_deepspeed:
        scheduler = get_scheduler(optimizer, args, total_steps)
    
    # ============ 恢复训练 ============
    start_epoch = 0
    best_acc1 = 0.0  # 使用 acc1 作为最佳标准（与 main.py 一致）
    
    if args.resume is not None:
        if args.enable_deepspeed:
            # DeepSpeed 恢复
            if os.path.exists(args.resume):
                import glob
                all_checkpoints = glob.glob(os.path.join(args.resume, 'epoch_*'))
                latest_ckpt = -1
                for ckpt in all_checkpoints:
                    t = ckpt.split('/')[-1].split('_')[1]
                    if t.isdigit():
                        latest_ckpt = max(int(t), latest_ckpt)
                if latest_ckpt >= 0:
                    start_epoch = latest_ckpt
                    _, client_states = model.load_checkpoint(args.resume, tag=f'epoch_{latest_ckpt}')
                    if client_states and 'best_acc1' in client_states:
                        best_acc1 = client_states['best_acc1']
                    logger.info(f"=> resuming DeepSpeed checkpoint '{args.resume}' (epoch {latest_ckpt})")
                else:
                    logger.info(f"=> no checkpoint found at '{args.resume}'")
        else:
            # 普通恢复
            if os.path.isfile(args.resume):
                checkpoint = torch.load(args.resume, map_location='cpu')
                start_epoch = checkpoint.get('epoch', 0) + 1  # 从下一个 epoch 开始
                best_acc1 = checkpoint.get('best_acc1', 0.0)
                model.load_state_dict(checkpoint['model_state_dict'])
                if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if scaler is not None and 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                    # 注意：warmup_cosine_lr 返回的是函数，不是对象，无法恢复状态
                    # 如果需要精确恢复学习率，需要重新计算到当前 step
                    pass
                logger.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch}, best_acc1 {best_acc1:.2f}%)")
            else:
                logger.warning(f"=> no checkpoint found at '{args.resume}'")
    
    # ============ 训练前先执行一次测试（用于和后续训练结果比较）============
    if is_master(args):
        print(f"\n{'#'*70}")
        print(f"#{'':^68}#")
        print(f"#{'🔍 INITIAL VALIDATION (Before Training)':^68}#")
        print(f"#{'':^68}#")
        print(f"{'#'*70}\n")
        logger.info("=> Running initial validation before training starts...")
    
    # 临时加载 CLIP 模型进行初始验证
    if args.use_embed:
        logger.info("=> loading CLIP model for initial validation...")
        clip_model_for_init_val, _, _ = open_clip.create_model_and_transforms(
            model_name=args.clip_model, 
            pretrained=args.pretrained
        )
        clip_model_for_init_val.to(device)
        clip_model_for_init_val.eval()
        for param in clip_model_for_init_val.parameters():
            param.requires_grad = False
    else:
        clip_model_for_init_val = clip_model
    
    with amp.autocast(enabled=not args.disable_amp):
        # ModelNet40 初始验证
        init_val_stats = test_zeroshot_3d_core(
            val_loader, args.validate_dataset_name, model, clip_model_for_init_val, tokenizer, args, "modelnet"
        )
        logging.info(f"Initial ModelNet40: {init_val_stats}")
        
        # ScanObjNN 初始验证
        init_val_scanobjnn_stats = test_zeroshot_3d_core(
            val_scanobjnn_loader, args.validate_dataset_name_scanobjnn, model, clip_model_for_init_val, tokenizer, args, "scanobjnn"
        )
        logging.info(f"Initial ScanObjNN: {init_val_scanobjnn_stats}")
    
    # 释放临时加载的 CLIP 模型
    if args.use_embed:
        del clip_model_for_init_val
        torch.cuda.empty_cache()
        logger.info("=> CLIP model released after initial validation")
    
    if is_master(args):
        print(f"\n{'─'*70}")
        print(f"📊 INITIAL VALIDATION RESULTS (Epoch -1, Before Training)")
        print(f"   📊 ModelNet40 - Acc@1: {init_val_stats['acc1']:.2f}% | Acc@3: {init_val_stats['acc3']:.2f}% | Acc@5: {init_val_stats['acc5']:.2f}%")
        print(f"   📊 ScanObjNN  - Acc@1: {init_val_scanobjnn_stats['acc1']:.2f}% | Acc@3: {init_val_scanobjnn_stats['acc3']:.2f}% | Acc@5: {init_val_scanobjnn_stats['acc5']:.2f}%")
        print(f"{'─'*70}\n")
        logger.info(f"Initial validation completed. ModelNet40 Acc@1: {init_val_stats['acc1']:.2f}%, ScanObjNN Acc@1: {init_val_scanobjnn_stats['acc1']:.2f}%")
        
     # 初始化 TensorBoard（仅主进程）
    writer = None
    if args.tensorboard and TENSORBOARD_AVAILABLE and is_master(args):
        tensorboard_dir = args.tensorboard_dir or os.path.join(args.output_dir, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(tensorboard_dir)
        logger.info(f"TensorBoard logging enabled at: {tensorboard_dir}")
        logger.info(f"View with: tensorboard --logdir={tensorboard_dir} --port=6006")

    # 记录初始验证结果到 TensorBoard
    if writer is not None:
        writer.add_scalar('val_modelnet40/acc1', init_val_stats['acc1'], -1)  # epoch = -1 表示初始状态
        writer.add_scalar('val_modelnet40/acc3', init_val_stats['acc3'], -1)
        writer.add_scalar('val_modelnet40/acc5', init_val_stats['acc5'], -1)
        writer.add_scalar('val_scanobjnn/acc1', init_val_scanobjnn_stats['acc1'], -1)
        writer.add_scalar('val_scanobjnn/acc3', init_val_scanobjnn_stats['acc3'], -1)
        writer.add_scalar('val_scanobjnn/acc5', init_val_scanobjnn_stats['acc5'], -1)
        writer.flush()
    
    # ============ 训练循环 ============
    logger.info("Starting training...")
    best_epoch = -1
    

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            # Epoch 开始的醒目输出
            print(f"\n{'#'*70}")
            print(f"#{'':^68}#")
            print(f"#{'🚀 EPOCH ' + str(epoch) + '/' + str(args.epochs) + ' STARTED':^68}#")
            print(f"#{'':^68}#")
            print(f"{'#'*70}\n")
            logger.info(f'Start epoch {epoch}')
        
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        completed_epoch = epoch + 1
        
        # 计时开始
        epoch_start_time = time.time()
        
        # 训练
        train_stats = train_one_epoch(
            model, train_loader, clip_model, optimizer, scheduler, scaler, criterion,
            epoch, device, args, logger, writer=writer, 
            modality_dropout_config=modality_dropout_config
        )
        
        # ============ 检查 epoch 是否完整 ============
        completion_ratio = train_stats.get('completion_ratio', 1.0)
        if completion_ratio < 0.99:
            if is_master(args):
                print(f"\n{'❌'*35}")
                print(f"❌ CRITICAL: Epoch {epoch} terminated early! Only {completion_ratio*100:.1f}% completed.")
                print(f"❌ This is likely due to a DataLoader worker crash or CUDA OOM.")
                print(f"❌ Possible solutions:")
                print(f"❌   1. Reduce num_workers (current: {args.workers})")
                print(f"❌   2. Reduce batch_size (current: {args.batch_size})")
                print(f"❌   3. Check for corrupted data in your dataset")
                print(f"❌   4. Increase shared memory: --shm-size in docker")
                print(f"{'❌'*35}\n")
                logger.error(f"Epoch {epoch} terminated early! Only {completion_ratio*100:.1f}% completed.")
            # 不直接退出，让训练继续，但用户会看到明显的警告
        
        # 计算 epoch 耗时
        epoch_time = time.time() - epoch_start_time
        
        if is_master(args):
            effective_bs = train_stats.get('effective_batch_size', args.batch_size * args.world_size)
            mse_weight = train_stats.get('mse_weight', 0.0)
            
            # Epoch 训练完成的醒目输出
            print(f"\n{'*'*70}")
            print(f"✅ EPOCH {epoch} TRAINING COMPLETED")
            print(f"   📉 Final Loss: {train_stats['loss']:.4f}")
            # MSE 损失信息
            if mse_weight > 0:
                print(f"   🔗 MSE Loss: {train_stats.get('loss_mse', 0):.4f} (weight: {mse_weight:.4f})")
            print(f"   🎯 Text Acc: {train_stats.get('fused_text_acc', 0):.2f}% | Image Acc: {train_stats.get('fused_image_acc', 0):.2f}%")
            print(f"   📈 Learning Rate: {train_stats['lr']:.6f}")
            print(f"   🌡️  Logit Scale: {train_stats['logit_scale']:.4f}")
            print(f"   📦 Effective Batch Size: {effective_bs}")
            print(f"   ⏱️  Epoch Time: {epoch_time/60:.2f} min")
            print(f"{'*'*70}\n")
            
            logger.info(
                f"Epoch {epoch} Training - "
                f"Loss: {train_stats['loss']:.4f} "
                f"MSE: {train_stats.get('loss_mse', 0):.4f} (w={mse_weight:.4f}) "
                f"LR: {train_stats['lr']:.6f} "
                f"Logit Scale: {train_stats['logit_scale']:.4f} "
                f"Effective BS: {effective_bs}"
            )

         # 先保存再验证，防止验证时程序崩溃保存失败
         # DeepSpeed 检查点需要所有进程参与
        if is_master(args):
            logger.info(f"=> Saving checkpoint for epoch {epoch} BEFORE validation...")
        
        if args.enable_deepspeed:
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, args)
        elif is_master(args):
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, args)
        
        if is_master(args):
            logger.info(f"=> Checkpoint saved successfully. Now starting validation...")

        # 验证（与 main.py 保持一致：对三个验证集进行零样本分类测试）
        # ============ 验证时临时加载 CLIP 模型（节省训练时显存）============
        if args.use_embed:
            # 训练时未加载 CLIP，验证时临时加载
            logger.info("=> loading CLIP model for validation...")
            clip_model_for_val, _, _ = open_clip.create_model_and_transforms(
                model_name=args.clip_model, 
                pretrained=args.pretrained
            )
            clip_model_for_val.to(device)
            clip_model_for_val.eval()
            for param in clip_model_for_val.parameters():
                param.requires_grad = False
        else:
            clip_model_for_val = clip_model
        
        with amp.autocast(enabled=not args.disable_amp):
            # ModelNet40 验证
            val_stats = test_zeroshot_3d_core(
                val_loader, args.validate_dataset_name, model, clip_model_for_val, tokenizer, args, "modelnet"
            )
            logging.info(f"ModelNet40: {val_stats}")
            
            # LVIS 验证
            # val_lvis_stats = test_zeroshot_3d_core(
            #     val_lvis_loader, args.validate_dataset_name_lvis, model, clip_model_for_val, tokenizer, args, "lvis"
            # )
            # logging.info(f"LVIS: {val_lvis_stats}")
            
            # ScanObjNN 验证
            val_scanobjnn_stats = test_zeroshot_3d_core(
                val_scanobjnn_loader, args.validate_dataset_name_scanobjnn, model, clip_model_for_val, tokenizer, args, "scanobjnn"
            )
            logging.info(f"ScanObjNN: {val_scanobjnn_stats}")
            
            # 使用 LVIS 的 acc1 作为最佳模型选择标准（与 main.py 一致）
            acc1 = val_scanobjnn_stats["acc1"]
        
        # ============ 验证完成后释放 CLIP 模型显存 ============
        if args.use_embed:
            del clip_model_for_val
            torch.cuda.empty_cache()
            logger.info("=> CLIP model released to free GPU memory")
        
        if is_master(args):
            # 验证结果的醒目输出
            print(f"\n{'~'*70}")
            print(f"🎯 EPOCH {epoch} VALIDATION RESULTS")
            print(f"   📊 ModelNet40 - Acc@1: {val_stats['acc1']:.2f}% | Acc@3: {val_stats['acc3']:.2f}% | Acc@5: {val_stats['acc5']:.2f}%")
            print(f"   📊 ScanObjNN  - Acc@1: {val_scanobjnn_stats['acc1']:.2f}% | Acc@3: {val_scanobjnn_stats['acc3']:.2f}% | Acc@5: {val_scanobjnn_stats['acc5']:.2f}%")
            print(f"   🏆 Best Acc@1: {max(best_acc1, acc1):.2f}%")
            print(f"{'~'*70}\n")
            
            logger.info(
                f"Epoch {epoch} Validation - "
                f"ModelNet40 Acc@1: {val_stats['acc1']:.2f}% "
                # f"LVIS Acc@1: {val_lvis_stats['acc1']:.2f}% "
                f"ScanObjNN Acc@1: {val_scanobjnn_stats['acc1']:.2f}%"
            )
        
        # 保存最佳模型（验证后单独保存最佳检查点）
        is_best = acc1 > best_acc1
        if is_best:
            best_acc1 = acc1
            best_epoch = epoch
            # 单独保存最佳模型
            if is_master(args):
                logger.info(f"=> New best model! Acc@1: {best_acc1:.2f}% at epoch {best_epoch}")
            if args.enable_deepspeed:
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, args, is_best=True, best_acc1=best_acc1)
            elif is_master(args):
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, args, is_best=True, best_acc1=best_acc1)
        

    
        # 记录日志（与 main.py 格式一致）
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in val_stats.items()},
            # **{f'test_lvis_{k}': v for k, v in val_lvis_stats.items()},
            **{f'test_scanobjnn_{k}': v for k, v in val_scanobjnn_stats.items()},
            'epoch': epoch,
            'best_acc1': best_acc1,
            'best_epoch': best_epoch
        }
        
        if is_master(args):
            logger.info(f"Epoch {epoch} Stats: {log_stats}")
        
        # 记录 TensorBoard 日志（仅主进程）
        if writer is not None:
            # 训练指标
            writer.add_scalar('train/loss', train_stats['loss'], epoch)
            writer.add_scalar('train/loss_text', train_stats.get('loss_text', 0), epoch)
            writer.add_scalar('train/loss_image', train_stats.get('loss_image', 0), epoch)
            # MSE 对齐损失指标
            writer.add_scalar('train/loss_mse', train_stats.get('loss_mse', 0), epoch)
            writer.add_scalar('train/loss_mse_text', train_stats.get('loss_mse_text', 0), epoch)
            writer.add_scalar('train/loss_mse_image', train_stats.get('loss_mse_image', 0), epoch)
            writer.add_scalar('train/mse_weight', train_stats.get('mse_weight', 0), epoch)
            writer.add_scalar('train/fused_text_acc', train_stats.get('fused_text_acc', 0), epoch)
            writer.add_scalar('train/fused_image_acc', train_stats.get('fused_image_acc', 0), epoch)
            writer.add_scalar('train/lr', train_stats['lr'], epoch)
            writer.add_scalar('train/logit_scale', train_stats['logit_scale'], epoch)
            
            # ModelNet40 验证指标
            writer.add_scalar('val_modelnet40/acc1', val_stats['acc1'], epoch)
            writer.add_scalar('val_modelnet40/acc3', val_stats['acc3'], epoch)
            writer.add_scalar('val_modelnet40/acc5', val_stats['acc5'], epoch)
            
            # LVIS 验证指标
            # writer.add_scalar('val_lvis/acc1', val_lvis_stats['acc1'], epoch)
            # writer.add_scalar('val_lvis/acc3', val_lvis_stats['acc3'], epoch)
            # writer.add_scalar('val_lvis/acc5', val_lvis_stats['acc5'], epoch)
            
            # ScanObjNN 验证指标
            writer.add_scalar('val_scanobjnn/acc1', val_scanobjnn_stats['acc1'], epoch)
            writer.add_scalar('val_scanobjnn/acc3', val_scanobjnn_stats['acc3'], epoch)
            writer.add_scalar('val_scanobjnn/acc5', val_scanobjnn_stats['acc5'], epoch)
            
            # 最佳准确率
            writer.add_scalar('best/acc1', best_acc1, epoch)
            writer.add_scalar('best/epoch', best_epoch, epoch)
            
            writer.flush()  # 确保数据写入磁盘
    
    # 关闭 TensorBoard writer
    if writer is not None:
        writer.close()
        logger.info("TensorBoard writer closed.")
    
    if is_master(args):
        logger.info("Training completed!")
        logger.info(f"Best LVIS Acc@1: {best_acc1:.2f}% at epoch {best_epoch}")


if __name__ == '__main__':
    main()
