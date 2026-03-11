"""
多模态 Uni3D 对比学习损失函数

训练目标：
- uni3d_multimodal(ivt) <-> clip_text(t): 融合特征与文本特征对齐
- uni3d_multimodal(ivt) <-> clip_image(i): 融合特征与图像特征对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

try:
    from utils import utils
except ImportError:
    utils = None


class Uni3dMultimodalLoss(nn.Module):
    """
    多模态 Uni3D 对比学习损失
    
    对比学习目标：
    1. fused_projected (ivt 投影后) <-> clip_text_embed (t)
    2. fused_projected (ivt 投影后) <-> clip_image_embed (i)
    
    重要：
    - 融合特征的投影层已经移到模型定义中（方案 B）
    - 损失函数直接接收投影后的融合特征和原始 CLIP 特征
    - clip_text_embed 和 clip_image_embed 是原始的 CLIP 预提取特征，不经过模型投影
    """
    
    def __init__(self, 
                 text_weight: float = 1.0,
                 image_weight: float = 1.0,
                 use_distributed: bool = True):
        """
        Args:
            text_weight: 文本对比损失权重
            image_weight: 图像对比损失权重
            use_distributed: 是否使用分布式训练 (all_gather)
        """
        super().__init__()
        self.text_weight = text_weight
        self.image_weight = image_weight
        self.use_distributed = use_distributed
        
        # 注意：投影层已移至模型定义中（Uni3DMultimodal.fused_to_clip_proj）
        # 损失函数不再包含可训练参数
        
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, outputs: Dict[str, torch.Tensor], masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算对比学习损失
        
        Args:
            outputs: 包含以下键的字典:
                - 'fused_projected_txt': [B, clip_embed_dim] 用于文本对比的投影融合特征
                - 'fused_projected_img': [B, clip_embed_dim] 用于图像对比的投影融合特征
                - 'clip_text_embed': [B, clip_embed_dim] CLIP 文本特征
                - 'clip_image_embed': [B, clip_embed_dim] CLIP 图像特征
                - 'logit_scale': 温度参数
        
            masks: [B] 可选的 mask，用于忽略某些样本
        
        Returns:
            包含各项损失和准确率的字典
        """
        # 获取特征
        # 两个不同模态的投影融合特征
        fused_embed = outputs['fused_feats']  # [B, D_clip]

        clip_text_embed = outputs['clip_text_embed']  # [B, D_clip]
        clip_image_embed = outputs['clip_image_embed']  # [B, D_clip]
        
        # 如果第二维是 1，则去掉第二维，只保留第一维和第三维
        # [B, 1, D] -> [B, D]
        if clip_text_embed.dim() == 3 and clip_text_embed.size(1) == 1:
            clip_text_embed = clip_text_embed.squeeze(1)
        if clip_image_embed.dim() == 3 and clip_image_embed.size(1) == 1:
            clip_image_embed = clip_image_embed.squeeze(1)
        
        logit_scale = outputs['logit_scale']
        
        # 使用文本投影特征获取 batch size 和 device
        local_batch_size = fused_embed.size(0)
        device = fused_embed.device
        
        # 更新标签
        if local_batch_size != self.last_local_batch_size:
            if self.use_distributed and utils is not None:
                self.labels = local_batch_size * utils.get_rank() + torch.arange(
                    local_batch_size, device=device
                )
            else:
                self.labels = torch.arange(local_batch_size, device=device)
            self.last_local_batch_size = local_batch_size

        
        # 统一数据类型，确保所有张量都是相同的 dtype（防止 Half/Double 混合导致的错误）
        target_dtype = fused_embed.dtype
        clip_text_embed = clip_text_embed.to(target_dtype)
        clip_image_embed = clip_image_embed.to(target_dtype)
        
        # 归一化特征
        fused_embed = F.normalize(fused_embed, dim=-1, p=2)
        clip_text_embed = F.normalize(clip_text_embed, dim=-1, p=2)
        clip_image_embed = F.normalize(clip_image_embed, dim=-1, p=2)
        
        # 分布式训练时 gather 所有特征（使用支持梯度的版本）
        if self.use_distributed and utils is not None:
            fused_embed_all,  clip_text_embed_all, clip_image_embed_all = \
                utils.all_gather_batch_with_grad([fused_embed, 
                                        clip_text_embed, clip_image_embed])
        else:
            fused_embed_all = fused_embed
            clip_text_embed_all = clip_text_embed
            clip_image_embed_all = clip_image_embed
        
        # ============ 计算对比损失 ============
        
        # 1. fused_txt <-> text 对比损失
        logits_fused_text = logit_scale * fused_embed @ clip_text_embed_all.t()
        logits_text_fused = logit_scale * clip_text_embed @ fused_embed_all.t()
        
        loss_text = (F.cross_entropy(logits_fused_text, self.labels) + 
                     F.cross_entropy(logits_text_fused, self.labels)) / 2
        
        # 2. fused_img <-> image 对比损失
        logits_fused_image = logit_scale * fused_embed @ clip_image_embed_all.t()
        logits_image_fused = logit_scale * clip_image_embed @ fused_embed_all.t()
        
        # 处理 mask (如果有些样本没有图像)
        if masks is not None:
            masks = masks.to(device).bool()
            labels_masked = self.labels.clone()
            labels_masked[masks] = -100
            
            loss_image = (F.cross_entropy(logits_fused_image, labels_masked, ignore_index=-100) +
                          F.cross_entropy(logits_image_fused, labels_masked, ignore_index=-100)) / 2
        else:
            loss_image = (F.cross_entropy(logits_fused_image, self.labels) + 
                          F.cross_entropy(logits_image_fused, self.labels)) / 2
        
        # 总损失
        loss = self.text_weight * loss_text + self.image_weight * loss_image
        
        # ============ 计算准确率 ============
        with torch.no_grad():
            # fused -> text 准确率
            pred_text = torch.argmax(logits_fused_text, dim=-1)
            correct_text = pred_text.eq(self.labels).sum()
            fused_text_acc = 100 * correct_text / local_batch_size
            
            # fused -> image 准确率
            pred_image = torch.argmax(logits_fused_image, dim=-1)
            if masks is not None:
                valid_samples = (~masks).sum()
                correct_image = pred_image[~masks].eq(self.labels[~masks]).sum()
                fused_image_acc = 100 * correct_image / (valid_samples + 1e-8)
            else:
                correct_image = pred_image.eq(self.labels).sum()
                fused_image_acc = 100 * correct_image / local_batch_size
        
        return {
            'loss': loss,
            'loss_text': loss_text,
            'loss_image': loss_image,
            'fused_text_acc': fused_text_acc,
            'fused_image_acc': fused_image_acc
        }


class Uni3dMultimodalAllPairsLoss(nn.Module):
    """
    多模态 Uni3D 全配对对比学习损失
    
    支持更多对比配对：
    1. fused_projected (ivt) <-> clip_text (t)
    2. fused_projected (ivt) <-> clip_image (i)
    3. 可选：fused (ivt) <-> uni3d_pc (v)
    
    注意：投影层已移至模型定义中（方案 B）
    """
    
    def __init__(self, 
                 text_weight: float = 1.0,
                 image_weight: float = 1.0,
                 pc_weight: float = 0.5,
                 use_pc_alignment: bool = False,
                 use_distributed: bool = True):
        super().__init__()
        self.text_weight = text_weight
        self.image_weight = image_weight
        self.pc_weight = pc_weight
        self.use_pc_alignment = use_pc_alignment
        self.use_distributed = use_distributed
        
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, outputs: Dict[str, torch.Tensor], masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算全配对对比学习损失
        """
        # 使用已投影的融合特征
        fused_embed_projected = outputs['fused_feats']  # [B, D_clip]
        clip_text_embed = outputs['clip_text_embed']  # [B, D_clip] 或 [B, 1, D_clip]
        clip_image_embed = outputs['clip_image_embed']  # [B, D_clip] 或 [B, 1, D_clip]
        logit_scale = outputs['logit_scale']
        
        # 如果第二维是 1，则去掉第二维，只保留第一维和第三维
        # [B, 1, D] -> [B, D]
        if clip_text_embed.dim() == 3 and clip_text_embed.size(1) == 1:
            clip_text_embed = clip_text_embed.squeeze(1)
        if clip_image_embed.dim() == 3 and clip_image_embed.size(1) == 1:
            clip_image_embed = clip_image_embed.squeeze(1)
        
        local_batch_size = fused_embed_projected.size(0)
        device = fused_embed_projected.device
        
        # 更新标签
        if local_batch_size != self.last_local_batch_size:
            if self.use_distributed and utils is not None:
                self.labels = local_batch_size * utils.get_rank() + torch.arange(
                    local_batch_size, device=device
                )
            else:
                self.labels = torch.arange(local_batch_size, device=device)
            self.last_local_batch_size = local_batch_size
        
        # 归一化
        fused_embed_projected = F.normalize(fused_embed_projected, dim=-1, p=2)
        clip_text_embed = F.normalize(clip_text_embed, dim=-1, p=2)
        clip_image_embed = F.normalize(clip_image_embed, dim=-1, p=2)
        
        # Gather（使用支持梯度的版本）
        if self.use_distributed and utils is not None:
            fused_embed_all, clip_text_embed_all, clip_image_embed_all = \
                utils.all_gather_batch_with_grad([fused_embed_projected, clip_text_embed, clip_image_embed])
        else:
            fused_embed_all = fused_embed_projected
            clip_text_embed_all = clip_text_embed
            clip_image_embed_all = clip_image_embed
        
        # 统一数据类型，确保所有张量都是相同的 dtype（防止 Half/Double 混合导致的错误）
        target_dtype = fused_embed_projected.dtype
        clip_text_embed = clip_text_embed.to(target_dtype)
        clip_image_embed = clip_image_embed.to(target_dtype)
        clip_text_embed_all = clip_text_embed_all.to(target_dtype)
        clip_image_embed_all = clip_image_embed_all.to(target_dtype)
        fused_embed_all = fused_embed_all.to(target_dtype)
        
        # 计算损失
        # 1. fused <-> text
        logits_fused_text = logit_scale * fused_embed_projected @ clip_text_embed_all.t()
        logits_text_fused = logit_scale * clip_text_embed @ fused_embed_all.t()
        
        # 数值稳定性：限制 logits 范围，防止 softmax 溢出
        # logit_scale 最大为 exp(4.6052) ≈ 100，特征归一化后点积在 [-1, 1]
        # 所以 logits 最大约为 100，但为安全起见限制在 [-100, 100]
        logits_fused_text = torch.clamp(logits_fused_text, -100, 100)
        logits_text_fused = torch.clamp(logits_text_fused, -100, 100)
        
        loss_text = (F.cross_entropy(logits_fused_text, self.labels) + 
                     F.cross_entropy(logits_text_fused, self.labels)) / 2
        
        # 2. fused <-> image
        logits_fused_image = logit_scale * fused_embed_projected @ clip_image_embed_all.t()
        logits_image_fused = logit_scale * clip_image_embed @ fused_embed_all.t()
        
        # 数值稳定性：限制 logits 范围
        logits_fused_image = torch.clamp(logits_fused_image, -100, 100)
        logits_image_fused = torch.clamp(logits_image_fused, -100, 100)
        
        loss_image = (F.cross_entropy(logits_fused_image, self.labels) + 
                      F.cross_entropy(logits_image_fused, self.labels)) / 2
        
        # 3. 可选：fused <-> pc（使用原始融合特征池化）
        loss_pc = torch.tensor(0.0, device=device)
        fused_pc_acc = torch.tensor(0.0, device=device)
        
        if self.use_pc_alignment and 'pc_embed' in outputs and outputs['pc_embed'] is not None:
            pc_embed = outputs['pc_embed']  # [B, 30, D]
            pc_embed_pooled = F.normalize(pc_embed.mean(dim=1), dim=-1, p=2)
            
            # 对于 pc 对齐，使用原始融合特征的池化结果（与 pc 维度匹配）
            fused_embed = outputs['fused_embed']  # [B, 30, D]
            fused_embed_pooled = F.normalize(fused_embed.mean(dim=1), dim=-1, p=2)
            
            if self.use_distributed and utils is not None:
                pc_embed_all = utils.all_gather_batch_with_grad([pc_embed_pooled])[0]
                fused_embed_pooled_all = utils.all_gather_batch_with_grad([fused_embed_pooled])[0]
            else:
                pc_embed_all = pc_embed_pooled
                fused_embed_pooled_all = fused_embed_pooled
            
            logits_fused_pc = logit_scale * fused_embed_pooled @ pc_embed_all.t()
            logits_pc_fused = logit_scale * pc_embed_pooled @ fused_embed_pooled_all.t()
            loss_pc = (F.cross_entropy(logits_fused_pc, self.labels) + 
                       F.cross_entropy(logits_pc_fused, self.labels)) / 2
            
            with torch.no_grad():
                pred_pc = torch.argmax(logits_fused_pc, dim=-1)
                correct_pc = pred_pc.eq(self.labels).sum()
                fused_pc_acc = 100 * correct_pc / local_batch_size
        
        # 总损失
        loss = (self.text_weight * loss_text + 
                self.image_weight * loss_image + 
                self.pc_weight * loss_pc)
        
        # 计算准确率
        with torch.no_grad():
            pred_text = torch.argmax(logits_fused_text, dim=-1)
            correct_text = pred_text.eq(self.labels).sum()
            fused_text_acc = 100 * correct_text / local_batch_size
            
            pred_image = torch.argmax(logits_fused_image, dim=-1)
            correct_image = pred_image.eq(self.labels).sum()
            fused_image_acc = 100 * correct_image / local_batch_size
        
        return {
            'loss': loss,
            'loss_text': loss_text,
            'loss_image': loss_image,
            'loss_pc': loss_pc,
            'fused_text_acc': fused_text_acc,
            'fused_image_acc': fused_image_acc,
            'fused_pc_acc': fused_pc_acc
        }


class ModalityDropoutLoss(nn.Module):
    """
    支持 Modality Dropout 的多模态对比学习损失函数
    
    核心思想：
    - 融合特征应该与图片和文本做对比学习
    - 例如：输入 iv（图像+点云）时，融合特征应该和【文本】对比（预测文本）
    - 例如：输入 vt（点云+文本）时，融合特征应该和【图像】对比（预测图像）
    
    支持的模态组合及对比目标：
    - ivt: 完整三模态 → fused ↔ text, fused ↔ image（标准多模态对比）
    - iv:  图像+点云 → fused ↔ text（用图像和点云预测文本）
    - vt:  点云+文本 → fused ↔ image（用点云和文本预测图像）
    - it:  图像+文本 → fused ↔ image, fused ↔ text（无点云，直接图文对比）
    - v:   仅点云 → fused ↔ text, fused ↔ image（用点云同时预测图像和文本）
    - i:   仅图像 → fused ↔ text（用图像预测文本，类似 CLIP）
    - t:   仅文本 → fused ↔ image（用文本预测图像，类似 CLIP）
    """
    
    def __init__(self, 
                 text_weight: float = 1.0,
                 image_weight: float = 1.0,
                 use_distributed: bool = True):
        super().__init__()
        self.text_weight = text_weight
        self.image_weight = image_weight
        self.use_distributed = use_distributed
        
        self.labels = None
        self.last_local_batch_size = None
    
    def _get_contrastive_targets(self, modal: str) -> tuple:
        """
        根据输入模态组合，决定对比学习的目标模态
        
        核心逻辑：
        - 无论使用哪种模态组合，融合特征始终与图文特征做对比学习
        - 这样可以保持融合特征空间与 CLIP 图文特征空间对齐
        - 即使只有单一模态输入，也要学习预测完整的图文特征
        
        所有模态组合的对比目标：
        - ivt: 完整三模态 → fused ↔ text, fused ↔ image
        - iv:  图像+点云 → fused ↔ text, fused ↔ image
        - vt:  点云+文本 → fused ↔ text, fused ↔ image
        - it:  图像+文本 → fused ↔ text, fused ↔ image
        - v:   仅点云 → fused ↔ text, fused ↔ image
        - i:   仅图像 → fused ↔ text, fused ↔ image
        - t:   仅文本 → fused ↔ text, fused ↔ image
        
        Returns:
            (use_text_contrast, use_image_contrast): 始终返回 (True, True)
        """
        # 无论什么模态组合，始终与图文特征做对比学习
        return True, True

    def forward(self, outputs: Dict[str, torch.Tensor], masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算对比学习损失（支持 Modality Dropout）
        
        Args:
            outputs: 模型输出，包含：
                - 'fused_feats': [B, clip_embed_dim] 融合特征（根据输入模态编码）
                - 'clip_text_embed': [B, clip_embed_dim] 原始文本特征（始终提供，作为监督信号）
                - 'clip_image_embed': [B, clip_embed_dim] 原始图像特征（始终提供，作为监督信号）
                - 'logit_scale': 温度参数
                - 'modal': 当前使用的模态组合
        
        Returns:
            损失字典，包含各项损失和准确率
        """
        fused_embed_projected = outputs['fused_feats']
        clip_text_embed = outputs.get('clip_text_embed', None)
        clip_image_embed = outputs.get('clip_image_embed', None)
        logit_scale = outputs['logit_scale']
        modal = outputs.get('modal', 'ivt')
        
        local_batch_size = fused_embed_projected.size(0)
        device = fused_embed_projected.device
        
        # 处理 text embed 的维度
        if clip_text_embed is not None:
            if clip_text_embed.dim() == 3 and clip_text_embed.size(1) == 1:
                clip_text_embed = clip_text_embed.squeeze(1)
        
        # 处理 image embed 的维度
        if clip_image_embed is not None:
            if clip_image_embed.dim() == 3 and clip_image_embed.size(1) == 1:
                clip_image_embed = clip_image_embed.squeeze(1)
        
        # 更新标签
        if local_batch_size != self.last_local_batch_size:
            if self.use_distributed and utils is not None:
                self.labels = local_batch_size * utils.get_rank() + torch.arange(
                    local_batch_size, device=device
                )
            else:
                self.labels = torch.arange(local_batch_size, device=device)
            self.last_local_batch_size = local_batch_size
        
        # 归一化融合特征
        fused_embed_projected = F.normalize(fused_embed_projected, dim=-1, p=2)
        
        # 初始化损失和准确率
        # 使用 None 初始化，避免创建无梯度的张量
        loss_text = None
        loss_image = None
        fused_text_acc = torch.tensor(0.0, device=device)
        fused_image_acc = torch.tensor(0.0, device=device)
        
        # Gather 融合特征（用于分布式训练）
        if self.use_distributed and utils is not None:
            fused_embed_all = utils.all_gather_batch_with_grad([fused_embed_projected])[0]
        else:
            fused_embed_all = fused_embed_projected
        
        # 统一数据类型
        target_dtype = fused_embed_projected.dtype
        fused_embed_all = fused_embed_all.to(target_dtype)
        
        # 根据输入模态决定对比目标
        use_text_contrast, use_image_contrast = self._get_contrastive_targets(modal)
        
        # ============ 计算文本对比损失 ============
        if use_text_contrast and clip_text_embed is not None:
            clip_text_embed = clip_text_embed.to(target_dtype)
            clip_text_embed = F.normalize(clip_text_embed, dim=-1, p=2)
            
            if self.use_distributed and utils is not None:
                clip_text_embed_all = utils.all_gather_batch_with_grad([clip_text_embed])[0]
            else:
                clip_text_embed_all = clip_text_embed
            clip_text_embed_all = clip_text_embed_all.to(target_dtype)
            
            # 计算 logits
            logits_fused_text = logit_scale * fused_embed_projected @ clip_text_embed_all.t()
            logits_text_fused = logit_scale * clip_text_embed @ fused_embed_all.t()
            
            # 数值稳定性
            logits_fused_text = torch.clamp(logits_fused_text, -100, 100)
            logits_text_fused = torch.clamp(logits_text_fused, -100, 100)
            
            loss_text = (F.cross_entropy(logits_fused_text, self.labels) + 
                         F.cross_entropy(logits_text_fused, self.labels)) / 2
            
            # 计算准确率
            with torch.no_grad():
                pred_text = torch.argmax(logits_fused_text, dim=-1)
                correct_text = pred_text.eq(self.labels).sum()
                fused_text_acc = 100 * correct_text / local_batch_size
        
        # ============ 计算图像对比损失 ============
        if use_image_contrast and clip_image_embed is not None:
            clip_image_embed = clip_image_embed.to(target_dtype)
            clip_image_embed = F.normalize(clip_image_embed, dim=-1, p=2)
            
            if self.use_distributed and utils is not None:
                clip_image_embed_all = utils.all_gather_batch_with_grad([clip_image_embed])[0]
            else:
                clip_image_embed_all = clip_image_embed
            clip_image_embed_all = clip_image_embed_all.to(target_dtype)
            
            # 计算 logits
            logits_fused_image = logit_scale * fused_embed_projected @ clip_image_embed_all.t()
            logits_image_fused = logit_scale * clip_image_embed @ fused_embed_all.t()
            
            # 数值稳定性
            logits_fused_image = torch.clamp(logits_fused_image, -100, 100)
            logits_image_fused = torch.clamp(logits_image_fused, -100, 100)
            
            loss_image = (F.cross_entropy(logits_fused_image, self.labels) + 
                          F.cross_entropy(logits_image_fused, self.labels)) / 2
            
            # 计算准确率
            with torch.no_grad():
                pred_image = torch.argmax(logits_fused_image, dim=-1)
                correct_image = pred_image.eq(self.labels).sum()
                fused_image_acc = 100 * correct_image / local_batch_size
        
        # ============ 计算总损失 ============
        if loss_text is not None and loss_image is not None:
            # 两种对比都使用
            loss = self.text_weight * loss_text + self.image_weight * loss_image
        elif loss_text is not None:
            # 仅文本对比
            loss = loss_text
        elif loss_image is not None:
            # 仅图像对比
            loss = loss_image
        else:
            # 无对比目标（理论上不应该发生，但需要返回一个有梯度的 dummy loss）
            # 使用融合特征的 L2 范数作为 dummy loss，确保在计算图中
            loss = (fused_embed_projected ** 2).sum() * 0.0
        
        # 为返回值准备默认的 loss 值（用于日志记录）
        loss_text_val = loss_text if loss_text is not None else torch.tensor(0.0, device=device)
        loss_image_val = loss_image if loss_image is not None else torch.tensor(0.0, device=device)
        
        return {
            'loss': loss,
            'loss_text': loss_text_val,
            'loss_image': loss_image_val,
            'loss_pc': torch.tensor(0.0, device=device),  # 保持接口一致
            'fused_text_acc': fused_text_acc,
            'fused_image_acc': fused_image_acc,
            'fused_pc_acc': torch.tensor(0.0, device=device),
            'modal': modal,  # 返回当前模态组合，方便调试
            'use_text_contrast': use_text_contrast,
            'use_image_contrast': use_image_contrast
        }


def get_multimodal_loss(args=None):
    """获取多模态损失函数"""
    use_pc = getattr(args, 'use_pc_alignment', False) if args else False
    return Uni3dMultimodalAllPairsLoss(
        text_weight=getattr(args, 'text_weight', 1.0) if args else 1.0,
        image_weight=getattr(args, 'image_weight', 1.0) if args else 1.0,
        pc_weight=getattr(args, 'pc_weight', 0.5) if args else 0.5,
        use_pc_alignment=use_pc,
        use_distributed=getattr(args, 'use_distributed', True) if args else True
    )


def get_modality_dropout_loss(args=None):
    """获取支持 Modality Dropout 的损失函数"""
    return ModalityDropoutLoss(
        text_weight=getattr(args, 'text_weight', 1.0) if args else 1.0,
        image_weight=getattr(args, 'image_weight', 1.0) if args else 1.0,
        use_distributed=getattr(args, 'use_distributed', True) if args else True
    )


def get_multimodal_metric_names():
    """获取指标名称列表"""
    return ['loss', 'loss_text', 'loss_image', 'loss_pc', 
            'fused_text_acc', 'fused_image_acc', 'fused_pc_acc']
