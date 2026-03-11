"""
Uni3D Multimodal Model
参考 OneLLM 的设计，支持 i, v, t, iv, it, vt, ivt 七种模态组合输入
- i: image (图像)
- v: point cloud (点云，在3D领域通常称为volumetric)
- t: text (文本)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import open_clip
from typing import Optional, Dict, Tuple, List, Union
from einops import rearrange

from .point_encoder import Group, Encoder, PatchDropout, PointcloudEncoder


# ============ 辅助模块 ============

class Mlp(nn.Module):
    """MLP用于Router"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        x_float = x.float()
        rms = torch.mean(x_float * x_float, dim=-1, keepdim=True)
        rms = torch.rsqrt(rms + self.eps)
        x_norm = (x_float * rms).to(dtype=x.dtype)
        return x_norm * self.scale


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q, k, v):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class EmbedND(nn.Module):
    """N维位置编码"""
    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [self._rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)

    def _rope(self, pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
        assert dim % 2 == 0
        scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
        omega = 1.0 / (theta ** scale + 1e-10)
        out = torch.einsum("...n,d->...nd", pos.float(), omega)
        out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
        out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
        return out.to(pos.dtype)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, min=-1e4, max=1e4)
        
        qkv = self.qkv(x.float())
        qkv = torch.nan_to_num(qkv, nan=0.0, posinf=0.0, neginf=0.0)
        
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        
        attn_out = self._attention(q.float(), k.float(), v.float(), pe=pe)
        attn_out = torch.clamp(attn_out, min=-1e4, max=1e4)
        
        x_out = self.proj(attn_out)
        return x_out.to(dtype=x.dtype)

    def _attention(self, q, k, v, pe):
        q, k = self._apply_rope(q, k, pe)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "B H L D -> B L (H D)")
        return x

    def _apply_rope(self, xq, xk, freqs_cis):
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class TripleStreamBlock(nn.Module):
    """三流融合模块，支持1-3个模态的任意组合"""
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        # Stream 1 (Image)
        self.norm1_1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn_1 = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2_1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp_1 = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # Stream 2 (Point Cloud)
        self.norm1_2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn_2 = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2_2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp_2 = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # Stream 3 (Text)
        self.norm1_3 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn_3 = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2_3 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp_3 = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, x1: Optional[torch.Tensor], x2: Optional[torch.Tensor], 
                x3: Optional[torch.Tensor], pe: torch.Tensor):
        """
        Args:
            x1: Image features [B, L1, D] or None
            x2: Point cloud features [B, L2, D] or None
            x3: Text features [B, L3, D] or None
            pe: Position encoding
        """
        def process_stream(x, norm, attn):
            if x is None:
                return None, None, None
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x_norm = norm(x.float()).to(dtype=x.dtype)
            qkv = attn.qkv(x_norm.float())
            qkv = torch.clamp(qkv, min=-1e4, max=1e4)
            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
            q, k = attn.norm(q, k, v)
            return q, k, v

        q1, k1, v1 = process_stream(x1, self.norm1_1, self.attn_1)
        q2, k2, v2 = process_stream(x2, self.norm1_2, self.attn_2)
        q3, k3, v3 = process_stream(x3, self.norm1_3, self.attn_3)

        qs = [q for q in [q1, q2, q3] if q is not None]
        ks = [k for k in [k1, k2, k3] if k is not None]
        vs = [v for v in [v1, v2, v3] if v is not None]
        
        if not qs:
            return x1, x2, x3

        q = torch.cat(qs, dim=2)
        k = torch.cat(ks, dim=2)
        v = torch.cat(vs, dim=2)

        # Attention
        attn = self._attention(q.float(), k.float(), v.float(), pe)
        attn = torch.clamp(attn, min=-1e4, max=1e4)

        # Split and apply residual + MLP
        curr = 0
        
        if x1 is not None:
            length = x1.shape[1]
            x1_attn = attn[:, curr:curr + length]
            curr += length
            x1 = x1 + self.attn_1.proj(x1_attn)
            x1 = x1 + self.mlp_1(self.norm2_1(x1))
            x1 = torch.clamp(x1, min=-1e4, max=1e4)
        
        if x2 is not None:
            length = x2.shape[1]
            x2_attn = attn[:, curr:curr + length]
            curr += length
            x2 = x2 + self.attn_2.proj(x2_attn)
            x2 = x2 + self.mlp_2(self.norm2_2(x2))
            x2 = torch.clamp(x2, min=-1e4, max=1e4)

        if x3 is not None:
            length = x3.shape[1]
            x3_attn = attn[:, curr:curr + length]
            curr += length
            x3 = x3 + self.attn_3.proj(x3_attn)
            x3 = x3 + self.mlp_3(self.norm2_3(x3))
            x3 = torch.clamp(x3, min=-1e4, max=1e4)

        return x1, x2, x3

    def _attention(self, q, k, v, pe):
        q, k = self._apply_rope(q, k, pe)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "B H L D -> B L (H D)")
        return x

    def _apply_rope(self, xq, xk, freqs_cis):
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]


        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class ResamplerBlock(nn.Module):
    """重采样Transformer块"""
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, start_pos=0, freqs_cis=None, mask=None):
        # 保存原始 dtype 用于最后转换
        original_dtype = x.dtype
        
        # Self-attention with residual
        # 注意：nn.MultiheadAttention 的权重是 float32，需要将输入转换为 float32
        x_float = x.float()
        x_norm = self.norm1(x_float)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x_float = x_float + attn_out
        
        # FFN with residual
        x_float = x_float + self.ffn(self.norm2(x_float))
        
        # 转换回原始 dtype
        return x_float.to(original_dtype)


# ============ 主模型 ============

class Uni3DMultimodal(nn.Module):
    """
    多模态 Uni3D 模型
    支持 7 种模态组合: i, v, t, iv, it, vt, ivt
    - i: image (图像)
    - v: point cloud (点云)
    - t: text (文本)
    
    新增功能:
    - use_fusion_blocks: 是否启用三流融合模块，关闭时直接拼接特征
    - use_embed: 是否使用预提取的特征，启用时跳过编码器
    - load_pretrained: 是否加载原始预训练权重（测试时应设为 False，因为 checkpoint 已包含完整权重）
    """
    
    SUPPORTED_MODALS = ['i', 'v', 't', 'iv', 'it', 'vt', 'ivt']
    
    def __init__(self, args, load_pretrained=True):
        super().__init__()
        self.args = args
        
        # 基础维度配置
        self.embed_dim = args.embed_dim  # 最终输出维度 (512)
        self.trans_dim = args.pc_feat_dim  # Transformer 隐藏维度 (768)
        self.clip_width = self.trans_dim  # CLIP 特征维度
        
        # 新增配置选项
        self.use_fusion_blocks = getattr(args, 'use_fusion_blocks', True)  # 默认启用三流融合
        self.use_embed = getattr(args, 'use_embed', False)  # 默认不使用预提取特征
        
        # 对比学习温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.04))
        
        # ============ 模态编码器 ============
        
        # 1. 点云编码器 (保留原始 Uni3D 的点云处理)
        # 无论是否使用预提取特征，点云都需要实时编码，因此必须创建点云编码器
        point_transformer = timm.create_model(args.pc_model, drop_path_rate=args.drop_path_rate)
        self.point_encoder = PointcloudEncoder(point_transformer, args)
        
        # 仅在训练时加载原始预训练权重
        # 测试时应设置 load_pretrained=False，因为 checkpoint 已包含完整的模型权重
        self.load_pretrained = load_pretrained
        if load_pretrained and hasattr(args, 'pretrained_pc') and args.pretrained_pc:
            self._load_point_transformer_weights(args)
        
        # 点云投影层 (point encoder 输出) - 始终需要
        self.point_proj = nn.Linear(1408, self.trans_dim)
        
        if not self.use_embed:
            # 不使用预提取特征时，需要创建 CLIP 编码器
            # 2. 图像编码器 (使用 CLIP ViT-g)
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                model_name=args.clip_model, pretrained=args.clip_model_path
            )
            for param in self.clip_model.parameters():
                param.requires_grad = False
            
            # 3. 文本编码器投影层 (CLIP text encoder 输出)
            # EVA02-E-14-plus 文本编码器输出维度是 1280
            self.text_proj = nn.Linear(1280, self.trans_dim)  # CLIP text dim (1280) -> trans_dim
            
            # 4. 图像投影层 (CLIP vision encoder 输出)
            # EVA02-E-14-plus (eva02_enormous) 视觉编码器输出维度是 1792
            self.image_proj = nn.Linear(1792, self.trans_dim)  # CLIP vision dim (1792) -> trans_dim
            
            # 占位符，避免权重加载时报错
            self.embed_image_proj = None
            self.embed_text_proj = None
        else:
            # 使用预提取特征时的投影层
            # OpenShape 提供的文本预提取特征是 1280 维（embed_dim）
            self.embed_image_proj = nn.Linear(args.embed_dim, self.trans_dim)  # [B, embed_dim] -> [B, trans_dim]
            self.embed_text_proj = nn.Linear(args.embed_dim, self.trans_dim)   # [B, embed_dim] -> [B, trans_dim]
            
            # 不需要 CLIP 编码器相关的投影层，节省显存
            self.clip_model = None
            self.clip_preprocess = None
            self.text_proj = None
            self.image_proj = None
        
        # ============ 多模态融合模块 ============
        
        # 三流融合块（可选）
        self.num_fusion_blocks = 4
        self.num_fusion_heads = 16
        
        if self.use_fusion_blocks:
            self.fusion_blocks = nn.ModuleList([
                TripleStreamBlock(
                    hidden_size=self.trans_dim,
                    num_heads=self.num_fusion_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True
                ) for _ in range(self.num_fusion_blocks)
            ])
        
        # 位置编码
        # head_dim = trans_dim // num_heads，RoPE 编码应用于每个 head
        self.pe_head_dim = self.trans_dim // self.num_fusion_heads
        self.pe_theta = 10000  # RoPE 基础频率
        self.pe_embedder = EmbedND(dim=self.pe_head_dim, theta=self.pe_theta, axes_dim=[self.pe_head_dim])
        
        # ============ 融合特征到 CLIP 空间的投影层（方案 B）============
        # 将融合特征（trans_dim）投影到 CLIP 特征空间（clip_embed_dim）
        # 这样融合特征可以与原始 CLIP 预提取特征进行对比学习

        self.clip_dim = 1280
        self.fused_to_clip_proj = nn.Sequential(
            nn.Linear(self.trans_dim, self.trans_dim),
            nn.GELU(),
            nn.Linear(self.trans_dim, self.clip_dim),
            nn.LayerNorm(self.clip_dim)
        )
        
        # ============ 路由控制 (MoE) ============
        
        self.num_experts = 3
        self.num_resample_layers = 8
        
        # 重采样层 (每个专家)
        self.resample_layers = nn.ModuleDict()
        for expert_id in range(self.num_experts):
            self.resample_layers[str(expert_id)] = nn.ModuleList([
                ResamplerBlock(self.trans_dim, num_heads=16)
                for _ in range(self.num_resample_layers)
            ])
        
        # 每种模态组合的专用模块
        self.routers = nn.ModuleDict()
        self.resample_tokens = nn.ParameterDict()
        self.clip_proj1 = nn.ModuleDict()
        self.clip_proj2 = nn.ModuleDict()
        
        for modal in self.SUPPORTED_MODALS:
            # Router: 决定每个专家的权重
            self.routers[modal] = Mlp(self.trans_dim, self.trans_dim * 4, self.num_experts)
            
            # Resample tokens: 可学习的查询 tokens
            self.resample_tokens[modal] = nn.Parameter(torch.empty([1, 30, self.trans_dim]))
            nn.init.normal_(self.resample_tokens[modal], std=0.02)
            
            # 投影层
            self.clip_proj1[modal] = nn.Sequential(
                nn.Linear(self.trans_dim, self.trans_dim),
                nn.LayerNorm(self.trans_dim)
            )
            self.clip_proj2[modal] = nn.Sequential(
                nn.Linear(self.trans_dim, self.trans_dim),
                nn.LayerNorm(self.trans_dim)
            )
        
        # 初始化
        self._init_weights()

    def _load_point_transformer_weights(self, args):
        """
        手动加载点云 Transformer 预训练权重（支持部分匹配）
        
        Args:
            pretrained_path: 预训练权重路径 (.safetensors 或 .pth/.pt)
        """
        pretrained_path = args.pretrained_pc
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        print('loaded checkpoint {}'.format(pretrained_path))
        sd = checkpoint['module']
        if not args.use_distributed and next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        
        # 使用 strict=False 允许部分加载（因为 use_embed 模式下有些层可能不存在）
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        
        if missing_keys:
            print(f"Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
        
        # 显式释放 checkpoint 内存
        del checkpoint
        del sd
        import gc
        gc.collect()

    def _init_weights(self):
        """
        初始化权重 - 保持原有编码能力的初始化策略
        
        核心思想：让新增模块在初始化时近似恒等变换，使得模型初始状态
        接近于原始预训练的点云编码器，从而保持原有的对比学习能力。
        
        策略：
        1. 融合模块的 MLP 输出层初始化为 0 -> 初始时只有残差连接生效
        2. 投影层使用小值初始化 -> 减少对原始特征的扰动
        3. fused_to_clip_proj 特殊初始化 -> 保持点云特征的对比学习能力
        """
        # 基础初始化：对所有 Linear 和 LayerNorm 进行标准初始化
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # ============ 特殊初始化：保持原有编码能力 ============
        
        # 1. 融合模块 (TripleStreamBlock) 的 MLP 输出层初始化为 0
        #    这样初始时 x = x + 0 = x，即恒等映射
        if self.use_fusion_blocks and hasattr(self, 'fusion_blocks'):
            for block in self.fusion_blocks:
                # 每个 stream 的 MLP 最后一层初始化为 0
                for mlp in [block.mlp_1, block.mlp_2, block.mlp_3]:
                    if isinstance(mlp, nn.Sequential) and len(mlp) > 0:
                        last_layer = mlp[-1]
                        if isinstance(last_layer, nn.Linear):
                            nn.init.zeros_(last_layer.weight)
                            if last_layer.bias is not None:
                                nn.init.zeros_(last_layer.bias)
                
                # Attention 投影层也初始化为小值
                for attn in [block.attn_1, block.attn_2, block.attn_3]:
                    if hasattr(attn, 'proj'):
                        nn.init.xavier_uniform_(attn.proj.weight, gain=0.01)
                        if attn.proj.bias is not None:
                            nn.init.zeros_(attn.proj.bias)
        
        # 2. 预提取特征投影层使用小值初始化
        if hasattr(self, 'embed_image_proj') and self.embed_image_proj is not None:
            nn.init.xavier_uniform_(self.embed_image_proj.weight, gain=0.01)
            if self.embed_image_proj.bias is not None:
                nn.init.zeros_(self.embed_image_proj.bias)
        
        if hasattr(self, 'embed_text_proj') and self.embed_text_proj is not None:
            nn.init.xavier_uniform_(self.embed_text_proj.weight, gain=0.01)
            if self.embed_text_proj.bias is not None:
                nn.init.zeros_(self.embed_text_proj.bias)
        
        # 3. 点云投影层保持小值初始化（连接预训练点云编码器和融合模块）
        if hasattr(self, 'point_proj'):
            nn.init.xavier_uniform_(self.point_proj.weight, gain=0.1)
            if self.point_proj.bias is not None:
                nn.init.zeros_(self.point_proj.bias)
        
        # 4. fused_to_clip_proj 特殊初始化
        #    目标：让点云特征在融合后仍能保持与原始 CLIP 空间的对齐
        #    策略：最后一层初始化为近似恒等（或小值），中间层正常初始化
        if hasattr(self, 'fused_to_clip_proj'):
            for i, layer in enumerate(self.fused_to_clip_proj):
                if isinstance(layer, nn.Linear):
                    if i == len(self.fused_to_clip_proj) - 2:  # 最后一个 Linear（LayerNorm 之前）
                        # 最后的投影层使用较小的初始化
                        nn.init.xavier_uniform_(layer.weight, gain=0.1)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                    else:
                        # 中间层正常初始化
                        nn.init.xavier_uniform_(layer.weight, gain=0.5)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
        
        # 5. clip_proj1 和 clip_proj2 使用小值初始化
        for modal in self.SUPPORTED_MODALS:
            if modal in self.clip_proj1:
                for layer in self.clip_proj1[modal]:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=0.1)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
            if modal in self.clip_proj2:
                for layer in self.clip_proj2[modal]:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=0.1)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
        
        # 6. ResamplerBlock 的 FFN 输出层初始化为 0（保持残差连接的恒等性）
        if hasattr(self, 'resample_layers'):
            for expert_id in range(self.num_experts):
                if str(expert_id) in self.resample_layers:
                    for block in self.resample_layers[str(expert_id)]:
                        if hasattr(block, 'ffn') and isinstance(block.ffn, nn.Sequential):
                            last_layer = block.ffn[-1]
                            if isinstance(last_layer, nn.Linear):
                                nn.init.zeros_(last_layer.weight)
                                if last_layer.bias is not None:
                                    nn.init.zeros_(last_layer.bias)
        
        print("[Uni3DMultimodal] Weights initialized with identity-preserving strategy")

    # ============ 单模态编码 ============
    
    @torch.no_grad()
    def encode_image_raw(self, image: torch.Tensor) -> torch.Tensor:
        """
        使用 CLIP 编码图像 (兼容 EVA02-E-14-plus)
        
        EVA02-E-14-plus 的视觉编码器结构:
        - visual: TimmModel wrapper
        - visual.trunk: timm EVA ViT 模型 (eva02_enormous_patch14_clip_224)
        - trunk.patch_embed: PatchEmbed
        - trunk.cls_token: [1, 1, 1792]
        - trunk.pos_embed: [1, 257, 1792]  (256 patches + 1 CLS)
        - trunk.blocks: 64 个 Block
        - trunk.norm: LayerNorm
        - trunk.fc_norm: LayerNorm (for CLS token)
        
        Args:
            image: [B, 3, 224, 224]
        Returns:
            image_feats: [B, 257, 1792] 完整序列特征 (1 CLS + 256 patches)
        """
        visual = self.clip_model.visual
        # visual.reset_classifier(0, '')
        if hasattr(visual, 'trunk'):
            # EVA-CLIP 使用 TimmModel
            trunk = visual.trunk
            
            # 手动执行 forward 以获取完整序列 (不使用 forward_features，因为它可能只返回 CLS)
            # Step 1: Patch embedding
            x = trunk.patch_embed(image)  # [B, 256, embed_dim]
            
            # Step 2: Add CLS token
            cls_token = trunk.cls_token.expand(x.shape[0], -1, -1)  # [B, 1, embed_dim]
            x = torch.cat([cls_token, x], dim=1)  # [B, 257, embed_dim]
            
            # Step 3: Add position embedding
            if hasattr(trunk, 'pos_embed') and trunk.pos_embed is not None:
                x = x + trunk.pos_embed
            
            # Step 4: Position dropout (如果有)
            if hasattr(trunk, 'pos_drop'):
                x = trunk.pos_drop(x)
            
            # Step 5: Transformer blocks
            for blk in trunk.blocks:
                x = blk(x)
            
            # Step 6: Final normalization
            x = trunk.norm(x)  # [B, 257, embed_dim]
            print(f'image_feat.shape = {x.shape}')
        
        return x

    def encode_point_raw(self, pts: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
        """
        编码点云 (与原始 PointcloudEncoder 逻辑保持一致，但输出完整序列)
        Args:
            pts: [B, N, 3] 点坐标
            colors: [B, N, 3] 点颜色
        Returns:
            point_feats: [B, 513, trans_dim] (1 CLS + 512 groups)
        """
        # 分组 (与原始代码一致)
        model = self.point_encoder
        _, center, features = model.group_divider(pts, colors)
        
        # 局部编码 (与原始代码一致)
        group_input_tokens = model.encoder(features)  # [B, num_group, encoder_dim]
        group_input_tokens = model.encoder2trans(group_input_tokens)  # [B, num_group, trans_dim]
        
        # 添加 CLS token (与原始代码一致：使用 cls_token 和 cls_pos)
        bsz = group_input_tokens.size(0)
        cls_tokens = model.cls_token.expand(bsz, -1, -1)
        cls_pos = model.cls_pos.expand(bsz, -1, -1)
        
        # 位置编码 (与原始代码一致)
        pos = model.pos_embed(center)  # [B, num_group, trans_dim]
        
        # 组合 (与原始代码一致)
        x = torch.cat([cls_tokens, group_input_tokens], dim=1)  # [B, 513, trans_dim]
        pos = torch.cat([cls_pos, pos], dim=1)  # [B, 513, trans_dim]
        x = x + pos
        
        # Patch dropout (与原始代码一致)
        x = model.patch_dropout(x)
        
        # 通过 point_transformer (与原始代码一致)
        x = model.visual.pos_drop(x)
        for blk in model.visual.blocks:
            x = blk(x)
        x = model.visual.norm(x)
        # print(f'point.shape = {x.shape}')
        # 注意：原始代码这里只取 CLS token 并经过 fc_norm 和 trans2embed
        # 但为了融合，我们保留完整序列，在后续融合模块中处理
        
        return x

    def encode_pc(self, pc: torch.Tensor) -> torch.Tensor:
        """
        纯点云编码（与原始 Uni3D 的 encode_pc 保持一致）
        
        这个方法直接使用原始点云编码器的完整流程，不经过融合模块。
        用于零样本分类测试，确保初始状态与原始 Uni3D 输出一致。
        
        Args:
            pc: [B, N, 6] 点云 (xyz + rgb)
        
        Returns:
            pc_embed: [B, embed_dim] 点云特征（与 CLIP 空间对齐）
        """
        xyz = pc[:, :, :3].contiguous()
        color = pc[:, :, 3:].contiguous()
        # 直接调用 point_encoder 的 forward 方法
        # 这会执行完整的：group → encoder → transformer → fc_norm → trans2embed
        pc_embed = self.point_encoder(xyz, color)
        return pc_embed

    @torch.no_grad()
    def encode_text_raw(self, text: torch.Tensor) -> torch.Tensor:
        """
        使用 CLIP 编码文本 (兼容 EVA02-E-14-plus)
        
        EVA02-E-14-plus 的文本编码器结构:
        - text: HFTextEncoder wrapper
        - text.transformer: HuggingFace model (通常是 roberta-large 变体)
        - text.transformer.embeddings: token + position embeddings
        - text.transformer.encoder: transformer layers
        - text.proj: projection to clip space [text_width, embed_dim]
        
        Args:
            text: [B, 77] tokenized text
        Returns:
            text_feats: [B, 77, trans_dim] 完整序列特征
        """
        if hasattr(self.clip_model, 'text'):
            # EVA-CLIP 使用 HFTextEncoder
            model = self.clip_model.text

             # 1. token embedding
            x = model.token_embedding(text)  # [B, L, D]

            # 2. 加 positional embedding
            x = x + model.positional_embedding  # [L, D] broadcast 到 [B, L, D]

            # 3. transformer 编码
            x = x.permute(1, 0, 2)               # [L, B, D]
            x = model.transformer(x)             # [L, B, D]
            x = x.permute(1, 0, 2)               # [B, L, D]

            # 4. LayerNorm
            x = model.ln_final(x)                # [B, L, D]

            print(f'text_feats.shape = {x.shape}')
            
        
        return x

    # ============ 预提取特征处理 ============
    
    def process_precomputed_embed(
        self,
        image_embed: Optional[torch.Tensor] = None,
        text_embed: Optional[torch.Tensor] = None,
        point_embed: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        处理预提取的特征，将其投影到 trans_dim 并扩展为序列
        
        Args:
            image_embed: [B, embed_dim] 预提取的图像 embedding
            text_embed: [B, embed_dim] 预提取的文本 embedding
            point_embed: [B, N, point_feat_dim] 点云特征（如果有）
        
        Returns:
            img_feats: [B, 1, trans_dim] 或 None
            text_feats: [B, 1, trans_dim] 或 None
            point_feats: [B, L, trans_dim] 或 None
        """
        img_feats = None
        text_feats = None
        point_feats = None

        if image_embed is not None:
            # [B, embed_dim] -> [B, 1, trans_dim]
            img_feats = self.embed_image_proj(image_embed)
            if img_feats.dim() == 2:
                img_feats = img_feats.unsqueeze(1)  # [B, 1, trans_dim]
            img_feats = img_feats.to(torch.bfloat16)
        
        if text_embed is not None:
            # [B, embed_dim] -> [B, 1, trans_dim]
            text_feats = self.embed_text_proj(
                text_embed.to(dtype=self.embed_text_proj.weight.dtype)
            )
            if text_feats.dim() == 2:
                text_feats = text_feats.unsqueeze(1)  # [B, 1, trans_dim]
            text_feats = text_feats.to(torch.bfloat16)
        
        if point_embed is not None:
            # [B, N, point_feat_dim] -> [B, N, trans_dim]
            point_feats = self.embed_point_proj(point_embed)
            point_feats = point_feats.to(torch.bfloat16)
        
        return img_feats, text_feats, point_feats

    # ============ 多模态融合编码 ============
    
    def encode_multimodal(
        self, 
        image: Optional[torch.Tensor] = None,
        point: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        modal: str = None,
        # 预提取特征（当 use_embed=True 时使用）
        image_embed: Optional[torch.Tensor] = None,
        text_embed: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        多模态融合编码
        
        Args:
            image: [B, 3, 224, 224] 原始图像 或 None（use_embed=False 时使用）
            point: [B, N, 6] (xyz + rgb) 点云 或 None
            text: [B, 77] tokenized text 或 None（use_embed=False 时使用）
            modal: 模态类型，可选 'i', 'v', 't', 'iv', 'it', 'vt', 'ivt'
            image_embed: [B, embed_dim] 预提取的图像特征（use_embed=True 时使用）
            text_embed: [B, embed_dim] 预提取的文本特征（use_embed=True 时使用）
        
        Returns:
            fused_feats: [B, 30, embed_dim] 融合后的特征
            original_text_embed: [B, embed_dim] 原始文本特征（用于对比学习目标）
            original_image_embed: [B, embed_dim] 原始图像特征（用于对比学习目标）
        """
        # 自动推断模态
        if modal is None:
            if self.use_embed:
                modal = self._infer_modal_from_embed(image_embed, point, text_embed)
            else:
                modal = self._infer_modal(image, point, text)
        
        assert modal in self.SUPPORTED_MODALS, f"Unsupported modal: {modal}"
        
        # 获取设备和 batch size
        if self.use_embed:
            bsz, device = self._get_batch_info_embed(image_embed, point, text_embed)
        else:
            bsz, device = self._get_batch_info(image, point, text)
        
        # ============ Step 1: 单模态编码/处理预提取特征 ============
        img_feats = None
        point_feats = None
        text_feats = None
        
        # 保存原始特征用于对比学习目标（不经过模型投影）
        original_text_embed = None
        original_image_embed = None
        
        if self.use_embed:
            # 保存原始预提取特征（用于对比学习目标）
            # 这些特征不经过模型投影，保持原始的 CLIP 特征空间
            # 注意：无论 modal 是什么，只要有输入特征就保存，作为监督信号
            if image_embed is not None:
                original_image_embed = image_embed.clone()  # [B, embed_dim]   
            if text_embed is not None:
                original_text_embed = text_embed.clone()           
            # 使用预提取特征（投影后用于融合）
            img_feats, text_feats, point_feats = self.process_precomputed_embed(
                image_embed=image_embed if 'i' in modal else None,
                text_embed=text_embed if 't' in modal else None,
                point_embed=None  # 点云通常需要实时编码
            )
            
            # 点云仍需实时编码（即使使用预提取特征模式）
            if 'v' in modal and point is not None:
                pts = point[:, :, :3].contiguous()
                colors = point[:, :, 3:].contiguous()
                point_feats = self.encode_point_raw(pts, colors)
                point_feats = self.point_proj(point_feats).to(torch.bfloat16)
        else:
            # 使用原始编码器
            if 'i' in modal and image is not None:
                img_feats = self.encode_image_raw(image)  # [B, 257, 1792]
                img_feats = self.image_proj(img_feats)  # [B, 257, trans_dim]
                img_feats = img_feats.to(torch.bfloat16)
                
            if 'v' in modal and point is not None:
                pts = point[:, :, :3].contiguous()
                colors = point[:, :, 3:].contiguous()
                point_feats = self.encode_point_raw(pts, colors)
                point_feats = self.point_proj(point_feats).to(torch.bfloat16)  # [B, 513, trans_dim]
                
            if 't' in modal and text is not None:
                text_feats = self.encode_text_raw(text)  # [B, 77, 1280]
                text_feats = self.text_proj(text_feats)  # [B, 77, trans_dim]      
                text_feats = text_feats.to(torch.bfloat16)
        
        # ============ Step 2: 准备位置编码 ============
        img_len = img_feats.shape[1] if img_feats is not None else 0
        point_len = point_feats.shape[1] if point_feats is not None else 0
        text_len = text_feats.shape[1] if text_feats is not None else 0
        total_len = img_len + point_len + text_len
        
        ids = torch.arange(total_len, device=device, dtype=torch.long)
        ids = ids.unsqueeze(0).expand(bsz, -1).unsqueeze(-1)  # [B, L, 1]
        pe = self.pe_embedder(ids).to(torch.bfloat16)  # Position encoding
        
        # ============ Step 3: 三流融合（可选）============
        curr_img = img_feats
        curr_point = point_feats
        curr_text = text_feats
        
        if self.use_fusion_blocks:
            # 启用三流融合
            for block in self.fusion_blocks:
                curr_img, curr_point, curr_text = block(curr_img, curr_point, curr_text, pe)
        # else: 不启用时，直接使用原始特征进行拼接
        
        # ============ Step 4: 合并特征 ============
        feats_list = []
        if curr_img is not None:
            feats_list.append(curr_img)
        if curr_point is not None:
            feats_list.append(curr_point)
        if curr_text is not None:
            feats_list.append(curr_text)
        
        fused_feats = torch.cat(feats_list, dim=1)  # [B, total_len, D]
        
        # ============ Step 5: 投影和路由 ============
        fused_feats = self.clip_proj1[modal](fused_feats)
        
        # 添加 resample tokens
        tokens = self.resample_tokens[modal].repeat(bsz, 1, 1)
        fused_feats = torch.cat([tokens, fused_feats], dim=1)
        
        # 路由权重
        routing_weights = self.routers[modal](fused_feats).sigmoid()
        routing_sum = routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights / (routing_sum + 1e-8)
        
        # MoE 专家处理
        fused_feats = fused_feats.to(torch.bfloat16)
        expert_outputs = []
        
        for expert_id in range(self.num_experts):
            expert_feat = fused_feats.to(torch.bfloat16)
            for layer in self.resample_layers[str(expert_id)]:
                expert_feat = layer(expert_feat).to(torch.bfloat16)
            
            # 只取 resample tokens 部分
            expert_feat = expert_feat[:, :tokens.size(1)]
            weight = routing_weights[:, :tokens.size(1), expert_id]
            expert_feat = expert_feat * weight[:, :, None]
            expert_outputs.append(expert_feat)
        
        fused_feats = sum(expert_outputs)
        
        # 最终投影
        fused_feats = self.clip_proj2[modal](fused_feats).to(torch.bfloat16)
        
        # ============ Step 6: 将融合特征投影到 CLIP 空间（方案 B）============
        # 池化融合特征: [B, 30, trans_dim] -> [B, trans_dim]
        fused_embed_pooled = fused_feats.mean(dim=1)
        # 投影到 CLIP 空间: [B, trans_dim] -> [B, clip_embed_dim]
        fused_embed_pooled = self.fused_to_clip_proj(fused_embed_pooled.float())
        
        # 返回融合特征和原始特征（用于对比学习）
        # 注意：
        # - fused_feats: [B, 30, trans_dim] 原始融合特征序列（可用于其他任务）
        # - fused_embed_projected: [B, clip_embed_dim] 投影后的融合特征（用于对比学习）
        # - original_text_embed 和 original_image_embed 是未经模型投影的原始 CLIP 特征
        return fused_embed_pooled, original_text_embed, original_image_embed

    def _infer_modal(self, image, point, text) -> str:
        """根据输入推断模态类型"""
        has_i = image is not None
        has_v = point is not None
        has_t = text is not None
        
        if has_i and has_v and has_t:
            return 'ivt'
        elif has_i and has_v:
            return 'iv'
        elif has_i and has_t:
            return 'it'
        elif has_v and has_t:
            return 'vt'
        elif has_i:
            return 'i'
        elif has_v:
            return 'v'
        elif has_t:
            return 't'
        else:
            raise ValueError("At least one modality must be provided")

    def _infer_modal_from_embed(self, image_embed, point, text_embed) -> str:
        """根据预提取特征推断模态类型"""
        has_i = image_embed is not None
        has_v = point is not None
        has_t = text_embed is not None
        
        if has_i and has_v and has_t:
            return 'ivt'
        elif has_i and has_v:
            return 'iv'
        elif has_i and has_t:
            return 'it'
        elif has_v and has_t:
            return 'vt'
        elif has_i:
            return 'i'
        elif has_v:
            return 'v'
        elif has_t:
            return 't'
        else:
            raise ValueError("At least one modality must be provided")

    def _get_batch_info(self, image, point, text):
        """获取 batch size 和 device"""
        if image is not None:
            return image.size(0), image.device
        elif point is not None:
            return point.size(0), point.device
        elif text is not None:
            return text.size(0), text.device
        else:
            raise ValueError("At least one modality must be provided")

    def _get_batch_info_embed(self, image_embed, point, text_embed):
        """从预提取特征获取 batch size 和 device"""
        if image_embed is not None:
            return image_embed.size(0), image_embed.device
        elif point is not None:
            return point.size(0), point.device
        elif text_embed is not None:
            return text_embed.size(0), text_embed.device
        else:
            raise ValueError("At least one modality must be provided")

    def forward(
        self, 
        pc: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        modal: str = None,
        # 预提取特征（当 use_embed=True 时使用）
        image_embed: Optional[torch.Tensor] = None,
        text_embed: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        根据输入的模态组合，只调用一次 encode_multimodal 进行编码。
        
        Args:
            pc: [B, N, 6] 点云 (xyz + rgb)
            image: [B, 3, 224, 224] 图像（use_embed=False 时使用）
            text: [B, 77] tokenized 文本（use_embed=False 时使用）
            modal: 模态类型，可选 'i', 'v', 't', 'iv', 'it', 'vt', 'ivt'
                   如果为 None，则根据输入自动推断
            image_embed: [B, embed_dim] 预提取的图像特征（use_embed=True 时使用）
            text_embed: [B, embed_dim] 预提取的文本特征（use_embed=True 时使用）
        Returns:
            dict with:
                - 'embed': 编码后的特征 [B, 30, embed_dim]
                - 'modal': 实际使用的模态类型
                - 'logit_scale': 对比学习温度参数
        """
        # 自动推断模态类型
        if modal is None:
            if self.use_embed:
                modal = self._infer_modal_from_embed(image_embed, pc, text_embed)
            else:
                modal = self._infer_modal(image, pc, text)
        
        assert modal in self.SUPPORTED_MODALS, f"Unsupported modal: {modal}"
        
        # 只调用一次 encode_multimodal
        fused_feats, txt_feats, image_feats = self.encode_multimodal(
            image=image,
            point=pc,
            text=text,
            modal=modal,
            image_embed=image_embed,
            text_embed=text_embed
        )
        
        return {
            'fused_feats': fused_feats,                   
            'modal': modal,
            'logit_scale': self.logit_scale.exp(),
            'txt_feats': txt_feats,              # [B, clip_embed_dim] 原始 CLIP 文本特征
            'image_feats': image_feats           # [B, clip_embed_dim] 原始 CLIP 图像特征
        }
    
    def forward_separate(
        self, 
        pc: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        image_embed: Optional[torch.Tensor] = None,
        text_embed: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        分别编码各模态（用于对比学习等需要分开特征的场景）
        
        Args:
            pc: [B, N, 6] 点云 (xyz + rgb)
            image: [B, 3, 224, 224] 图像
            text: [B, 77] tokenized 文本
            image_embed: [B, embed_dim] 预提取的图像特征
            text_embed: [B, embed_dim] 预提取的文本特征
        Returns:
            dict with:
                - 'pc_embed': 点云特征 [B, 30, embed_dim] 或 None
                - 'image_embed': 图像特征 [B, 30, embed_dim] 或 None
                - 'text_embed': 文本特征 [B, 30, embed_dim] 或 None
                - 'logit_scale': 对比学习温度参数
        """
        pc_embed_out = None
        image_embed_out = None
        text_embed_out = None
        
        if pc is not None:
            pc_embed_out = self.encode_multimodal(point=pc, modal='v')
        
        if self.use_embed:
            if image_embed is not None:
                image_embed_out = self.encode_multimodal(image_embed=image_embed, modal='i')
            if text_embed is not None:
                text_embed_out = self.encode_multimodal(text_embed=text_embed, modal='t')
        else:
            if image is not None:
                image_embed_out = self.encode_multimodal(image=image, modal='i')
            if text is not None:
                text_embed_out = self.encode_multimodal(text=text, modal='t')
        
        return {
            'pc_embed': pc_embed_out,
            'image_embed': image_embed_out,
            'text_embed': text_embed_out,
            'logit_scale': self.logit_scale.exp()
        }
    


# ============ 工厂函数 ============

def create_uni3d_multimodal(args, load_pretrained=True):
    """
    创建多模态 Uni3D 模型
    
    Args:
        args: 模型配置参数
        load_pretrained: 是否加载原始预训练权重
                        - 训练时设为 True，会加载 args.pretrained_pc 指定的预训练点云编码器
                        - 测试时设为 False，因为会从 checkpoint 加载完整模型权重
    
    Returns:
        model: Uni3DMultimodal 模型实例
    """
    model = Uni3DMultimodal(args, load_pretrained=load_pretrained)
    return model
