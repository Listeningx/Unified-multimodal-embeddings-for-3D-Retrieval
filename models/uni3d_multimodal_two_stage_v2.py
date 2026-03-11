"""
Uni3D Multimodal Two-Stage Model V2
两阶段训练设计:
- Stage 1: 仅使用点云编码器，三模态特征直接concat后池化
- Stage 2: 加入TSB融合和MOE，点云编码器冻结

关键修改:
- Stage 1: 无TSB无MOE，三模态特征直接concat+池化
- Stage 2: TSB融合 + MOE路由控制
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
    """三流融合模块"""
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

        attn = self._attention(q.float(), k.float(), v.float(), pe)
        attn = torch.clamp(attn, min=-1e4, max=1e4)

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
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


# ============ 两阶段模型 V2 ============

class Uni3DMultimodalTwoStageV2(nn.Module):
    """
    两阶段多模态 Uni3D 模型 V2
    
    Stage 1: 
        - 仅点云编码器参与训练（微调）
        - 三模态特征直接concat后池化，不使用TSB和MOE
        - 学习率较小
    
    Stage 2: 
        - 点云编码器冻结
        - TSB融合模块参与训练
        - MOE模块参与训练（使用modality-dropout）
        - 学习率较大
    
    支持 7 种模态组合: i, v, t, iv, it, vt, ivt
    """
    
    SUPPORTED_MODALS = ['i', 'v', 't', 'iv', 'it', 'vt', 'ivt']
    
    def __init__(self, args, stage: int = 1, load_pretrained: bool = True):
        """
        Args:
            args: 配置参数
            stage: 训练阶段 (1 或 2)
            load_pretrained: 是否加载预训练权重
        """
        super().__init__()
        self.args = args
        self.stage = stage
        
        # 基础维度配置
        self.embed_dim = args.embed_dim
        self.trans_dim = args.pc_feat_dim
        self.clip_width = self.trans_dim
        self.clip_dim = 1280
        
        # 配置选项
        self.use_embed = getattr(args, 'use_embed', False)
        
        # 对比学习温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.04))
        
        # ============ 点云编码器 ============
        point_transformer = timm.create_model(args.pc_model, drop_path_rate=args.drop_path_rate)
        self.point_encoder = PointcloudEncoder(point_transformer, args)
        
        if load_pretrained and hasattr(args, 'pretrained_pc') and args.pretrained_pc:
            self._load_point_transformer_weights(args)
        
        # 点云投影层
        self.point_proj = nn.Linear(1408, self.trans_dim)
        
        # ============ 图文特征处理 ============
        if not self.use_embed:
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                model_name=args.clip_model, pretrained=args.clip_model_path
            )
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.text_proj = nn.Linear(1280, self.trans_dim)
            self.image_proj = nn.Linear(1792, self.trans_dim)
            self.embed_image_proj = None
            self.embed_text_proj = None
        else:#投影层：升维
            self.embed_image_proj = nn.Linear(args.embed_dim, self.trans_dim)
            self.embed_text_proj = nn.Linear(args.embed_dim, self.trans_dim)
            self.clip_model = None
            self.clip_preprocess = None
            self.text_proj = None
            self.image_proj = None
        
        # ============ Stage 1: 简单concat池化层 ============
        # 三模态特征concat后的投影层
        self.stage1_proj = nn.Sequential(
            nn.Linear(self.trans_dim, self.trans_dim),
            nn.GELU(),
            nn.Linear(self.trans_dim, self.clip_dim),
            nn.LayerNorm(self.clip_dim)
        )
        
        # ============ Stage 2: TSB + MOE (延迟初始化) ============
        self.fusion_blocks = None
        self.pe_embedder = None
        self.routers = None
        self.resample_tokens = None
        self.resample_layers = None
        self.clip_proj1 = None
        self.clip_proj2 = None
        self.fused_to_clip_proj = None
        
        # 如果是Stage 2，初始化TSB和MOE
        if stage == 2:
            self._init_stage2_modules()
        
        # 初始化权重
        self._init_weights()
        
        print(f"\n[Uni3DMultimodalTwoStageV2] Initialized for Stage {stage}")
        if stage == 1:
            print("  - Point Encoder: ✓ Trainable (fine-tuning)")
            print("  - Feature Fusion: concat + pooling (no TSB/MOE)")
            print("  - Learning Rate: Small")
        else:
            print("  - Point Encoder: ✗ Frozen")
            print("  - TSB Fusion: ✓ Trainable")
            print("  - MOE: ✓ Trainable (3 experts, all modalities)")
            print("  - Learning Rate: Large")
    
    def _init_stage2_modules(self):
        """初始化Stage 2所需的TSB和MOE模块"""
        print("[Uni3DMultimodalTwoStageV2] Initializing Stage 2 modules (TSB + MOE)...")
        
        # ============ TSB融合模块 ============
        self.num_fusion_blocks = 4
        self.num_fusion_heads = 16
        
        self.fusion_blocks = nn.ModuleList([
            TripleStreamBlock(
                hidden_size=self.trans_dim,
                num_heads=self.num_fusion_heads,
                mlp_ratio=4.0,
                qkv_bias=True
            ) for _ in range(self.num_fusion_blocks)
        ])
        
        # 位置编码
        self.pe_head_dim = self.trans_dim // self.num_fusion_heads
        self.pe_theta = 10000
        self.pe_embedder = EmbedND(dim=self.pe_head_dim, theta=self.pe_theta, axes_dim=[self.pe_head_dim])
        
        # ============ MOE模块 ============
        self.num_experts = 3
        self.num_resample_layers = 8
        
        # 专家网络
        self.resample_layers = nn.ModuleDict()
        for expert_id in range(self.num_experts):
            self.resample_layers[str(expert_id)] = nn.ModuleList([
                ResamplerBlock(self.trans_dim, num_heads=16)
                for _ in range(self.num_resample_layers)
            ])
        
        # 所有7种模态的Router和Tokens
        self.routers = nn.ModuleDict()
        self.resample_tokens = nn.ParameterDict()
        self.clip_proj1 = nn.ModuleDict()
        self.clip_proj2 = nn.ModuleDict()
        
        for modal in self.SUPPORTED_MODALS:
            self.routers[modal] = Mlp(self.trans_dim, self.trans_dim * 4, self.num_experts)
            self.resample_tokens[modal] = nn.Parameter(torch.empty([1, 30, self.trans_dim]))
            nn.init.normal_(self.resample_tokens[modal], std=0.02)
            self.clip_proj1[modal] = nn.Sequential(
                nn.Linear(self.trans_dim, self.trans_dim),
                nn.LayerNorm(self.trans_dim)
            )
            self.clip_proj2[modal] = nn.Sequential(
                nn.Linear(self.trans_dim, self.trans_dim),
                nn.LayerNorm(self.trans_dim)
            )
        
        # 融合特征投影到CLIP空间
        self.fused_to_clip_proj = nn.Sequential(
            nn.Linear(self.trans_dim, self.trans_dim),
            nn.GELU(),
            nn.Linear(self.trans_dim, self.clip_dim),
            nn.LayerNorm(self.clip_dim)
        )
        
        print("  - TSB Fusion Blocks: 4 layers")
        print("  - MOE: 3 experts, 7 modality routers")
    
    def expand_to_stage2(self):
        """从Stage 1扩展到Stage 2，初始化TSB和MOE模块"""
        if self.stage == 2:
            print("[Warning] Already in Stage 2, skipping expansion...")
            return
        
        print("[Uni3DMultimodalTwoStageV2] Expanding to Stage 2...")
        
        # 获取设备
        device = next(self.parameters()).device
        
        # 初始化Stage 2模块
        self._init_stage2_modules()
        
        # 移动到正确的设备
        if self.fusion_blocks is not None:
            self.fusion_blocks = self.fusion_blocks.to(device)
        if self.pe_embedder is not None:
            self.pe_embedder = self.pe_embedder.to(device)
        if self.routers is not None:
            self.routers = self.routers.to(device)
        if self.resample_layers is not None:
            self.resample_layers = self.resample_layers.to(device)
        if self.clip_proj1 is not None:
            self.clip_proj1 = self.clip_proj1.to(device)
        if self.clip_proj2 is not None:
            self.clip_proj2 = self.clip_proj2.to(device)
        if self.fused_to_clip_proj is not None:
            self.fused_to_clip_proj = self.fused_to_clip_proj.to(device)
        
        # 更新阶段
        self.stage = 2
        
        # 初始化新模块权重
        self._init_stage2_weights()
        
        print("[Uni3DMultimodalTwoStageV2] Expansion to Stage 2 completed!")
    
    def _init_stage2_weights(self):
        """初始化Stage 2新增模块的权重"""
        # TSB模块
        if self.fusion_blocks is not None:
            for block in self.fusion_blocks:
                for mlp in [block.mlp_1, block.mlp_2, block.mlp_3]:
                    if isinstance(mlp, nn.Sequential) and len(mlp) > 0:
                        last_layer = mlp[-1]
                        if isinstance(last_layer, nn.Linear):
                            nn.init.zeros_(last_layer.weight)
                            if last_layer.bias is not None:
                                nn.init.zeros_(last_layer.bias)
        
        # MOE模块
        if self.resample_layers is not None:
            for expert_id in range(self.num_experts):
                for block in self.resample_layers[str(expert_id)]:
                    if hasattr(block, 'ffn') and isinstance(block.ffn, nn.Sequential):
                        last_layer = block.ffn[-1]
                        if isinstance(last_layer, nn.Linear):
                            nn.init.zeros_(last_layer.weight)
                            if last_layer.bias is not None:
                                nn.init.zeros_(last_layer.bias)
        
        # clip_proj
        for modal in self.SUPPORTED_MODALS:
            if self.clip_proj1 is not None and modal in self.clip_proj1:
                for layer in self.clip_proj1[modal]:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=0.1)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
            if self.clip_proj2 is not None and modal in self.clip_proj2:
                for layer in self.clip_proj2[modal]:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=0.1)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)

    def _load_point_transformer_weights(self, args):
        """加载点云Transformer预训练权重"""
        pretrained_path = args.pretrained_pc
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        print(f'Loaded checkpoint from {pretrained_path}')
        sd = checkpoint['module']
        if not args.use_distributed and next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")
        
        del checkpoint, sd
        import gc
        gc.collect()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # 投影层小值初始化
        if hasattr(self, 'embed_image_proj') and self.embed_image_proj is not None:
            nn.init.xavier_uniform_(self.embed_image_proj.weight, gain=0.01)
        if hasattr(self, 'embed_text_proj') and self.embed_text_proj is not None:
            nn.init.xavier_uniform_(self.embed_text_proj.weight, gain=0.01)
        if hasattr(self, 'point_proj'):
            nn.init.xavier_uniform_(self.point_proj.weight, gain=0.1)

    # ============ 编码方法 ============
    
    def encode_point_raw(self, pts: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
        """编码点云"""
        model = self.point_encoder
        _, center, features = model.group_divider(pts, colors)
        group_input_tokens = model.encoder(features)
        group_input_tokens = model.encoder2trans(group_input_tokens)
        
        bsz = group_input_tokens.size(0)
        cls_tokens = model.cls_token.expand(bsz, -1, -1)
        cls_pos = model.cls_pos.expand(bsz, -1, -1)
        pos = model.pos_embed(center)
        
        x = torch.cat([cls_tokens, group_input_tokens], dim=1)
        pos = torch.cat([cls_pos, pos], dim=1)
        x = x + pos
        x = model.patch_dropout(x)
        x = model.visual.pos_drop(x)
        
        for blk in model.visual.blocks:
            x = blk(x)
        x = model.visual.norm(x)
        return x

    def encode_pc(self, pc: torch.Tensor) -> torch.Tensor:
        """纯点云编码（用于零样本测试）"""
        xyz = pc[:, :, :3].contiguous()
        color = pc[:, :, 3:].contiguous()
        pc_embed = self.point_encoder(xyz, color)
        return pc_embed

    def process_precomputed_embed(
        self,
        image_embed: Optional[torch.Tensor] = None,
        text_embed: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """处理预提取的特征"""
        img_feats = None
        text_feats = None
        
        if image_embed is not None:
            img_feats = self.embed_image_proj(image_embed)
            if img_feats.dim() == 2:
                img_feats = img_feats.unsqueeze(1)
            img_feats = img_feats.to(torch.bfloat16)
        
        if text_embed is not None:
            text_feats = self.embed_text_proj(
                text_embed.to(dtype=self.embed_text_proj.weight.dtype)
            )
            if text_feats.dim() == 2:
                text_feats = text_feats.unsqueeze(1)
            text_feats = text_feats.to(torch.bfloat16)
        
        return img_feats, text_feats

    def encode_stage1(
        self,
        point: torch.Tensor,
        image_embed: Optional[torch.Tensor] = None,
        text_embed: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Stage 1 编码：三模态特征concat后池化
        
        不使用TSB和MOE，直接concat + 平均池化
        """
        # 编码点云
        pts = point[:, :, :3].contiguous()
        colors = point[:, :, 3:].contiguous()
        point_feats = self.encode_point_raw(pts, colors)
        point_feats = self.point_proj(point_feats).to(torch.bfloat16)  # [B, N_pc, D]
        
        # 处理图文特征
        # img_feats, text_feats = self.process_precomputed_embed(image_embed, text_embed)
        
        # Concat所有特征
        feats_list = [point_feats]
        # if img_feats is not None:
        #     feats_list.append(img_feats)
        # if text_feats is not None:
        #     feats_list.append(text_feats)
        
        # [B, N_total, D]
        concat_feats = torch.cat(feats_list, dim=1)
        
        # 平均池化
        pooled_feats = concat_feats.mean(dim=1)  # [B, D]
        
        # 投影到CLIP空间
        output_feats = self.stage1_proj(pooled_feats.float())  # [B, clip_dim]
        
        return output_feats, text_embed, image_embed

    def encode_stage2(
        self,
        point: torch.Tensor,
        image_embed: Optional[torch.Tensor] = None,
        text_embed: Optional[torch.Tensor] = None,
        modal: str = 'ivt'
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Stage 2 编码：TSB融合 + MOE
        
        Args:
            modal: 模态组合，决定使用哪些模态
        """
        bsz = point.size(0)
        device = point.device
        
        # 编码点云
        pts = point[:, :, :3].contiguous()
        colors = point[:, :, 3:].contiguous()
        point_feats = self.encode_point_raw(pts, colors)
        point_feats = self.point_proj(point_feats).to(torch.bfloat16)
        
        # 处理图文特征
        img_feats = None
        text_feats = None
        
        if 'i' in modal and image_embed is not None:
            img_feats, _ = self.process_precomputed_embed(image_embed, None)
        if 't' in modal and text_embed is not None:
            _, text_feats = self.process_precomputed_embed(None, text_embed)
        if 'v' not in modal:
            point_feats = None
        
        # 计算位置编码
        img_len = img_feats.shape[1] if img_feats is not None else 0
        point_len = point_feats.shape[1] if point_feats is not None else 0
        text_len = text_feats.shape[1] if text_feats is not None else 0
        total_len = img_len + point_len + text_len
        
        ids = torch.arange(total_len, device=device, dtype=torch.long)
        ids = ids.unsqueeze(0).expand(bsz, -1).unsqueeze(-1)
        pe = self.pe_embedder(ids).to(torch.bfloat16)
        
        # TSB融合
        curr_img = img_feats
        curr_point = point_feats
        curr_text = text_feats
        
        for block in self.fusion_blocks:
            curr_img, curr_point, curr_text = block(curr_img, curr_point, curr_text, pe)
        
        # 合并融合后的特征
        feats_list = []
        if curr_img is not None:
            feats_list.append(curr_img)
        if curr_point is not None:
            feats_list.append(curr_point)
        if curr_text is not None:
            feats_list.append(curr_text)
        fused_feats = torch.cat(feats_list, dim=1)
        
        # MOE处理
        fused_feats = self.clip_proj1[modal](fused_feats)
        
        tokens = self.resample_tokens[modal].repeat(bsz, 1, 1)
        fused_feats = torch.cat([tokens, fused_feats], dim=1)
        
        # Router计算权重
        routing_weights = self.routers[modal](fused_feats).sigmoid()
        routing_sum = routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights / (routing_sum + 1e-8)
        
        fused_feats = fused_feats.to(torch.bfloat16)
        expert_outputs = []
        
        for expert_id in range(self.num_experts):
            expert_feat = fused_feats
            for layer in self.resample_layers[str(expert_id)]:
                expert_feat = layer(expert_feat)
            expert_feat = expert_feat[:, :tokens.size(1)]
            weight = routing_weights[:, :tokens.size(1), expert_id]
            expert_feat = expert_feat * weight[:, :, None]
            expert_outputs.append(expert_feat)
        
        fused_feats = sum(expert_outputs)
        fused_feats = self.clip_proj2[modal](fused_feats).to(torch.bfloat16)
        
        # 投影到CLIP空间
        fused_embed_pooled = fused_feats.mean(dim=1)
        fused_embed_pooled = self.fused_to_clip_proj(fused_embed_pooled.float())
        
        return fused_embed_pooled, text_embed, image_embed

    def encode_multimodal(
        self, 
        point: torch.Tensor,
        image_embed: Optional[torch.Tensor] = None,
        text_embed: Optional[torch.Tensor] = None,
        modal: str = 'v'
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        多模态编码的统一接口
        
        根据stage自动选择编码方式:
        - Stage 1: concat + pooling
        - Stage 2: TSB + MOE
        """
        if self.stage == 1:
            return self.encode_stage1(point, image_embed, text_embed)
        else:
            return self.encode_stage2(point, image_embed, text_embed, modal)

    def forward(
        self, 
        pc: torch.Tensor,
        image_embed: Optional[torch.Tensor] = None,
        text_embed: Optional[torch.Tensor] = None,
        modal: str = 'v'
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Stage 1: 三模态concat + 池化，忽略modal参数
        Stage 2: TSB + MOE，使用modal参数
        """
        fused_feats, txt_feats, image_feats = self.encode_multimodal(
            point=pc,
            image_embed=image_embed,
            text_embed=text_embed,
            modal=modal
        )
        
        return {
            'fused_feats': fused_feats,
            'modal': modal if self.stage == 2 else 'concat',
            'logit_scale': self.logit_scale.exp(),
            'txt_feats': txt_feats,
            'image_feats': image_feats
        }

    # ============ 阶段管理方法 ============
    
    def setup_stage1_training(self):
        """设置Stage 1训练：仅点云编码器可训练"""
        print("\n[Uni3DMultimodalTwoStageV2] Setting up Stage 1 training...")
        
        # 1. 点云编码器：可训练
        for param in self.point_encoder.parameters():
            param.requires_grad = True
        print("  - Point Encoder: Trainable")
        
        # 2. 投影层：可训练
        for param in self.point_proj.parameters():
            param.requires_grad = True
        if self.embed_image_proj is not None:
            for param in self.embed_image_proj.parameters():
                param.requires_grad = True
        if self.embed_text_proj is not None:
            for param in self.embed_text_proj.parameters():
                param.requires_grad = True
        print("  - Projection layers: Trainable")
        
        # 3. Stage1投影层：可训练
        for param in self.stage1_proj.parameters():
            param.requires_grad = True
        print("  - Stage1 proj: Trainable")
        
        # 4. logit_scale: 可训练
        self.logit_scale.requires_grad = True
        print("  - Logit scale: Trainable")
        
        self.stage = 1
    
    def setup_stage2_training(self):
        """设置Stage 2训练：冻结点云编码器，TSB和MOE可训练"""
        print("\n[Uni3DMultimodalTwoStageV2] Setting up Stage 2 training...")
        
        # 确保Stage 2模块已初始化
        if self.fusion_blocks is None:
            self.expand_to_stage2()
        
        # 1. 点云编码器：冻结
        for param in self.point_encoder.parameters():
            param.requires_grad = False
        print("  - Point Encoder: Frozen")
        
        # 2. 投影层：冻结
        for param in self.point_proj.parameters():
            param.requires_grad = False
        if self.embed_image_proj is not None:
            for param in self.embed_image_proj.parameters():
                param.requires_grad = False
        if self.embed_text_proj is not None:
            for param in self.embed_text_proj.parameters():
                param.requires_grad = False
        print("  - Projection layers: Frozen")
        
        # 3. Stage1投影层：冻结（不再使用）
        for param in self.stage1_proj.parameters():
            param.requires_grad = False
        
        # 4. TSB融合模块：可训练
        if self.fusion_blocks is not None:
            for param in self.fusion_blocks.parameters():
                param.requires_grad = True
        print("  - TSB Fusion: Trainable")
        
        # 5. MOE模块：可训练
        if self.routers is not None:
            for param in self.routers.parameters():
                param.requires_grad = True
        if self.resample_layers is not None:
            for param in self.resample_layers.parameters():
                param.requires_grad = True
        if self.clip_proj1 is not None:
            for param in self.clip_proj1.parameters():
                param.requires_grad = True
        if self.clip_proj2 is not None:
            for param in self.clip_proj2.parameters():
                param.requires_grad = True
        for modal in self.SUPPORTED_MODALS:
            if modal in self.resample_tokens:
                self.resample_tokens[modal].requires_grad = True
        print("  - MOE: Trainable")
        
        # 6. 融合投影层：可训练
        if self.fused_to_clip_proj is not None:
            for param in self.fused_to_clip_proj.parameters():
                param.requires_grad = True
        print("  - Fused proj: Trainable")
        
        # 7. logit_scale: 可训练
        self.logit_scale.requires_grad = True
        print("  - Logit scale: Trainable")
        
        self.stage = 2

    def get_trainable_params_info(self):
        """获取可训练参数统计"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        info = {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0
        }
        
        print(f"\n📊 Parameter Statistics (Stage {self.stage}):")
        print(f"   Total: {total_params/1e6:.2f}M")
        print(f"   Trainable: {trainable_params/1e6:.2f}M ({info['trainable_ratio']*100:.1f}%)")
        print(f"   Frozen: {frozen_params/1e6:.2f}M ({(1-info['trainable_ratio'])*100:.1f}%)")
        
        return info


# ============ 工厂函数 ============

def create_uni3d_multimodal_two_stage_v2(args, stage: int = 1, load_pretrained: bool = True):
    """
    创建两阶段多模态Uni3D模型V2
    
    Args:
        args: 配置参数
        stage: 训练阶段 (1 或 2)
        load_pretrained: 是否加载预训练权重
    
    Returns:
        model: Uni3DMultimodalTwoStageV2
    """
    model = Uni3DMultimodalTwoStageV2(args, stage=stage, load_pretrained=load_pretrained)
    return model
