"""
Uni3D Multimodal Two-Stage Model
支持两阶段训练:
- Stage 1: 只有融合模块 A (Triple Stream Block) + 单专家 MOE（只有 'v' 模态）
- Stage 2: 添加完整 MOE 模块 B (3个专家网络，从 Stage 1 单专家复制初始化)

基于 uni3d_multimodal.py 修改
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


# ============ 辅助模块（与 uni3d_multimodal.py 相同）============

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


# ============ 两阶段模型 ============

class Uni3DMultimodalTwoStage(nn.Module):
    """
    两阶段多模态 Uni3D 模型
    
    Stage 1: 
        - 融合模块 A (Triple Stream Block) - 保留完整三流结构，但只使用点云输入
        - 单专家 MOE（num_experts=1）
        - 只有 'v' 模态的 router/resample_tokens
        - point_encoder 解冻参与训练
    
    Stage 2: 
        - 从 Stage 1 的单专家复制 3 份初始化
        - 添加完整的 7 种模态的 router/resample_tokens（随机初始化）
        - 使用 modality-dropout，所有模态组合都有机会出现
        - point_encoder 冻结
    
    支持 7 种模态组合: i, v, t, iv, it, vt, ivt
    
    Args:
        load_pretrained: 是否加载原始预训练权重（测试时应设为 False，因为 checkpoint 已包含完整权重）
    """
    
    SUPPORTED_MODALS = ['i', 'v', 't', 'iv', 'it', 'vt', 'ivt']
    # Stage 1 只支持 'v' 模态
    STAGE1_MODALS = ['v']
    
    def __init__(self, args, stage: int = 1, load_pretrained: bool = True):
        """
        Args:
            args: 配置参数
            stage: 训练阶段 (1 或 2)
                   Stage 1: 单专家 MOE，只有 'v' 模态
                   Stage 2: 3 专家 MOE，支持所有 7 种模态
            load_pretrained: 是否加载原始预训练权重
                           - 训练时设为 True，会加载 args.pretrained_pc 指定的预训练点云编码器
                           - 测试时设为 False，因为会从 checkpoint 加载完整模型权重
        """
        super().__init__()
        self.args = args
        self.stage = stage
        
        # 基础维度配置
        self.embed_dim = args.embed_dim
        self.trans_dim = args.pc_feat_dim
        self.clip_width = self.trans_dim
        
        # 配置选项
        self.use_fusion_blocks = getattr(args, 'use_fusion_blocks', True)
        self.use_embed = getattr(args, 'use_embed', False)
        
        # 对比学习温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.04))
        
        # ============ 点云编码器（始终需要）============
        point_transformer = timm.create_model(args.pc_model, drop_path_rate=args.drop_path_rate)
        self.point_encoder = PointcloudEncoder(point_transformer, args)
        
        # 仅在训练时加载原始预训练权重
        # 测试时应设置 load_pretrained=False，因为 checkpoint 已包含完整的模型权重
        self.load_pretrained = load_pretrained
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
        else:
            self.embed_image_proj = nn.Linear(args.embed_dim, self.trans_dim)
            self.embed_text_proj = nn.Linear(args.embed_dim, self.trans_dim)
            self.clip_model = None
            self.clip_preprocess = None
            self.text_proj = None
            self.image_proj = None
        
        # ============ 模块 A: 融合模块（始终包含，保留完整三流结构）============
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
        self.pe_head_dim = self.trans_dim // self.num_fusion_heads
        self.pe_theta = 10000
        self.pe_embedder = EmbedND(dim=self.pe_head_dim, theta=self.pe_theta, axes_dim=[self.pe_head_dim])
        
        # 融合特征投影到 CLIP 空间
        self.clip_dim = 1280
        self.fused_to_clip_proj = nn.Sequential(
            nn.Linear(self.trans_dim, self.trans_dim),
            nn.GELU(),
            nn.Linear(self.trans_dim, self.clip_dim),
            nn.LayerNorm(self.clip_dim)
        )
        
        # ============ MOE 配置 ============
        self.num_resample_layers = 8
        
        if stage == 1:
            # Stage 1: 单专家 MOE，只有 'v' 模态
            self._init_stage1_moe()
        else:
            # Stage 2: 完整 MOE
            self._init_stage2_moe()
        
        # 初始化权重
        self._init_weights()
        
        print(f"[Uni3DMultimodalTwoStage] Initialized for Stage {stage}")
        if stage == 1:
            print("  - Module A (Fusion Blocks): ✓ Full structure, ALL modalities (i,v,t) input")
            print("  - Module B (MOE): ✓ Single expert, only 'v' modality router/tokens")
            print("  - Point Encoder: ✓ Unfrozen (will be trained)")
            print("  - Note: Fusion uses i+v+t, but only point cloud features enter MOE")
        else:
            print("  - Module A (Fusion Blocks): ✓ Full structure, all modalities")
            print("  - Module B (MOE): ✓ 3 experts, all 7 modalities")
            print("  - Point Encoder: ✓ Frozen")
    
    def _init_stage1_moe(self):
        """初始化 Stage 1 的单专家 MOE 模块"""
        print("[Uni3DMultimodalTwoStage] Initializing Stage 1 MOE (single expert, 'v' only)...")
        
        self.num_experts = 1  # Stage 1: 单专家
        
        # 单专家重采样层
        self.resample_layers = nn.ModuleDict()
        self.resample_layers['0'] = nn.ModuleList([
            ResamplerBlock(self.trans_dim, num_heads=16)
            for _ in range(self.num_resample_layers)
        ])
        
        # 只有 'v' 模态的专用模块
        self.routers = nn.ModuleDict()
        self.resample_tokens = nn.ParameterDict()
        self.clip_proj1 = nn.ModuleDict()
        self.clip_proj2 = nn.ModuleDict()
        
        # 只创建 'v' 模态
        modal = 'v'
        # Router: 单专家，输出维度为 1
        self.routers[modal] = Mlp(self.trans_dim, self.trans_dim * 4, self.num_experts)
        
        # Resample tokens
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
        
        print("  - Single expert initialized")
        print("  - Only 'v' modality modules created")
    
    def _init_stage2_moe(self):
        """初始化 Stage 2 的完整 MOE 模块"""
        print("[Uni3DMultimodalTwoStage] Initializing Stage 2 MOE (3 experts, all modalities)...")
        
        self.num_experts = 3  # Stage 2: 3 专家
        
        # 3 专家重采样层
        self.resample_layers = nn.ModuleDict()
        for expert_id in range(self.num_experts):
            self.resample_layers[str(expert_id)] = nn.ModuleList([
                ResamplerBlock(self.trans_dim, num_heads=16)
                for _ in range(self.num_resample_layers)
            ])
        
        # 所有 7 种模态的专用模块
        self.routers = nn.ModuleDict()
        self.resample_tokens = nn.ParameterDict()
        self.clip_proj1 = nn.ModuleDict()
        self.clip_proj2 = nn.ModuleDict()
        
        for modal in self.SUPPORTED_MODALS:
            # Router: 3 专家，输出维度为 3
            self.routers[modal] = Mlp(self.trans_dim, self.trans_dim * 4, self.num_experts)
            
            # Resample tokens
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
        
        print("  - 3 experts initialized")
        print("  - All 7 modality modules created")
    
    def expand_to_stage2(self, stage1_checkpoint_path: str = None):
        """
        从 Stage 1 扩展到 Stage 2
        
        关键操作：
        1. 将单专家复制 3 份
        2. 保留 'v' 模态的 router/resample_tokens
        3. 随机初始化其他 6 种模态的 router/resample_tokens/clip_proj
        
        Args:
            stage1_checkpoint_path: Stage 1 检查点路径（可选）
        """
        if self.stage == 2 and self.num_experts == 3:
            print("[Warning] Already in Stage 2 with 3 experts, skipping expansion...")
            return
        
        print("[Uni3DMultimodalTwoStage] Expanding from Stage 1 to Stage 2...")
        
        # 保存 Stage 1 的单专家权重
        stage1_expert_state = copy.deepcopy(self.resample_layers['0'].state_dict())
        stage1_v_router_state = copy.deepcopy(self.routers['v'].state_dict())
        stage1_v_resample_tokens = self.resample_tokens['v'].data.clone()
        stage1_v_clip_proj1_state = copy.deepcopy(self.clip_proj1['v'].state_dict())
        stage1_v_clip_proj2_state = copy.deepcopy(self.clip_proj2['v'].state_dict())
        
        # 获取设备
        device = next(self.parameters()).device
        
        # ============ 扩展专家网络：从 1 个复制到 3 个 ============
        self.num_experts = 3
        
        # 删除旧的单专家
        del self.resample_layers['0']
        
        # 创建 3 个专家，都从 Stage 1 的单专家复制
        for expert_id in range(self.num_experts):
            self.resample_layers[str(expert_id)] = nn.ModuleList([
                ResamplerBlock(self.trans_dim, num_heads=16)
                for _ in range(self.num_resample_layers)
            ]).to(device)
            # 加载 Stage 1 的专家权重
            self.resample_layers[str(expert_id)].load_state_dict(stage1_expert_state)
        
        print("  - Experts expanded: 1 -> 3 (copied from Stage 1)")
        
        # ============ 扩展 Router：从输出 1 到输出 3，并添加其他模态 ============
        # 保留 'v' 模态，但需要扩展输出维度
        # 删除旧的 'v' router
        del self.routers['v']
        
        # 为所有 7 种模态创建新的 router
        for modal in self.SUPPORTED_MODALS:
            self.routers[modal] = Mlp(self.trans_dim, self.trans_dim * 4, self.num_experts).to(device)
            
            # 'v' 模态：尝试复用 Stage 1 的权重（fc1 可以复用，fc2 需要扩展）
            if modal == 'v':
                # fc1 权重可以直接复用
                self.routers[modal].fc1.load_state_dict({
                    'weight': stage1_v_router_state['fc1.weight'],
                    'bias': stage1_v_router_state['fc1.bias']
                })
                # fc2 扩展：将单输出复制 3 份
                old_fc2_weight = stage1_v_router_state['fc2.weight']  # [1, hidden]
                old_fc2_bias = stage1_v_router_state['fc2.bias']      # [1]
                # 复制 3 份
                new_fc2_weight = old_fc2_weight.repeat(3, 1)  # [3, hidden]
                new_fc2_bias = old_fc2_bias.repeat(3)          # [3]
                self.routers[modal].fc2.weight.data = new_fc2_weight.to(device)
                self.routers[modal].fc2.bias.data = new_fc2_bias.to(device)
                print(f"  - Router 'v': fc1 reused, fc2 expanded 1->3")
            else:
                # 其他模态：随机初始化（已在创建时初始化）
                print(f"  - Router '{modal}': randomly initialized")
        
        # ============ 扩展 resample_tokens：添加其他模态 ============
        # 'v' 模态已存在，其他模态新建
        for modal in self.SUPPORTED_MODALS:
            if modal == 'v':
                # 保留 Stage 1 的 'v' resample_tokens
                continue
            else:
                # 新建其他模态的 resample_tokens
                self.resample_tokens[modal] = nn.Parameter(
                    torch.empty([1, 30, self.trans_dim], device=device)
                )
                nn.init.normal_(self.resample_tokens[modal], std=0.02)
                print(f"  - resample_tokens '{modal}': randomly initialized")
        
        # ============ 扩展 clip_proj：添加其他模态 ============
        for modal in self.SUPPORTED_MODALS:
            if modal == 'v':
                # 保留 Stage 1 的 'v' clip_proj
                continue
            else:
                # 新建其他模态的 clip_proj
                self.clip_proj1[modal] = nn.Sequential(
                    nn.Linear(self.trans_dim, self.trans_dim),
                    nn.LayerNorm(self.trans_dim)
                ).to(device)
                self.clip_proj2[modal] = nn.Sequential(
                    nn.Linear(self.trans_dim, self.trans_dim),
                    nn.LayerNorm(self.trans_dim)
                ).to(device)
                print(f"  - clip_proj '{modal}': randomly initialized")
        
        # 更新阶段标记
        self.stage = 2
        
        # 初始化新模块的权重
        self._init_new_modal_weights()
        
        print("[Uni3DMultimodalTwoStage] Expansion to Stage 2 completed!")
        print(f"  - num_experts: {self.num_experts}")
        print(f"  - Supported modals: {self.SUPPORTED_MODALS}")
    
    def _init_new_modal_weights(self):
        """初始化新添加的模态模块权重（Stage 2 扩展时调用）"""
        for modal in self.SUPPORTED_MODALS:
            if modal == 'v':
                continue  # 'v' 模态保留 Stage 1 的权重
            
            # clip_proj 使用小值初始化
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

    def _load_point_transformer_weights(self, args):
        """加载点云 Transformer 预训练权重"""
        pretrained_path = args.pretrained_pc
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        print('loaded checkpoint {}'.format(pretrained_path))
        sd = checkpoint['module']
        if not args.use_distributed and next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        
        if missing_keys:
            print(f"Missing keys when loading checkpoint: {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"Unexpected keys when loading checkpoint: {len(unexpected_keys)} keys")
        
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
        
        # 融合模块的 MLP 输出层初始化为 0
        if self.use_fusion_blocks and hasattr(self, 'fusion_blocks'):
            for block in self.fusion_blocks:
                for mlp in [block.mlp_1, block.mlp_2, block.mlp_3]:
                    if isinstance(mlp, nn.Sequential) and len(mlp) > 0:
                        last_layer = mlp[-1]
                        if isinstance(last_layer, nn.Linear):
                            nn.init.zeros_(last_layer.weight)
                            if last_layer.bias is not None:
                                nn.init.zeros_(last_layer.bias)
        
        # 投影层小值初始化
        if hasattr(self, 'embed_image_proj') and self.embed_image_proj is not None:
            nn.init.xavier_uniform_(self.embed_image_proj.weight, gain=0.01)
        if hasattr(self, 'embed_text_proj') and self.embed_text_proj is not None:
            nn.init.xavier_uniform_(self.embed_text_proj.weight, gain=0.01)
        if hasattr(self, 'point_proj'):
            nn.init.xavier_uniform_(self.point_proj.weight, gain=0.1)
        
        # MOE 模块权重初始化
        self._init_moe_weights()
    
    def _init_moe_weights(self):
        """初始化 MOE 模块权重"""
        # clip_proj 使用小值初始化
        for modal in (self.STAGE1_MODALS if self.stage == 1 else self.SUPPORTED_MODALS):
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
        
        # ResamplerBlock 的 FFN 输出层初始化为 0
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
        point_embed: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """处理预提取的特征"""
        img_feats = None
        text_feats = None
        point_feats = None
        
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
        
        return img_feats, text_feats, point_feats

    def encode_multimodal(
        self, 
        image: Optional[torch.Tensor] = None,
        point: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        modal: str = None,
        image_embed: Optional[torch.Tensor] = None,
        text_embed: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        多模态融合编码
        
        Stage 1 特殊处理：
        - 三流融合模块 (TripleStreamBlock) 同时输入点云、图片、文本特征
        - MOE 部分只使用 'v' 模态的 router/resample_tokens
        - 最终只取点云部分的融合特征进入 MOE
        
        Stage 2:
        - 根据 modal 参数决定使用哪些模态
        """
        if modal is None:
            if self.use_embed:
                modal = self._infer_modal_from_embed(image_embed, point, text_embed)
            else:
                modal = self._infer_modal(image, point, text)
        
        # Stage 1 的 MOE 只支持 'v' 模态，但三流融合使用全部模态
        if self.stage == 1:
            # Stage 1: MOE modal 固定为 'v'，但三流融合使用 'ivt'
            moe_modal = 'v'
            fusion_modal = 'ivt'  # 三流融合使用全部模态
        else:
            # Stage 2: 使用传入的 modal
            assert modal in self.SUPPORTED_MODALS, f"Unsupported modal: {modal}"
            moe_modal = modal
            fusion_modal = modal
        
        if self.use_embed:
            bsz, device = self._get_batch_info_embed(image_embed, point, text_embed)
        else:
            bsz, device = self._get_batch_info(image, point, text)
        
        # Step 1: 单模态编码
        # Stage 1: 三流融合使用全部模态 (ivt)，MOE 只使用 'v'
        # Stage 2: 根据 fusion_modal 决定使用哪些模态
        img_feats = None
        point_feats = None
        text_feats = None
        # 保留原始的图文特征（无论使用什么模态都传进来，用于后续计算损失）
        # 注意：需要检查 None，因为某些模态可能不提供图文特征
        original_text_embed = text_embed.clone() if text_embed is not None else None
        original_image_embed = image_embed.clone() if image_embed is not None else None
        
        if self.use_embed:
            
            # 根据 fusion_modal 决定处理哪些模态的特征
            img_feats, text_feats, point_feats = self.process_precomputed_embed(
                image_embed=image_embed if 'i' in fusion_modal else None,
                text_embed=text_embed if 't' in fusion_modal else None,
                point_embed=None
            )
            
            if 'v' in fusion_modal and point is not None:
                pts = point[:, :, :3].contiguous()
                colors = point[:, :, 3:].contiguous()
                point_feats = self.encode_point_raw(pts, colors)
                point_feats = self.point_proj(point_feats).to(torch.bfloat16)
        else:
            if 'v' in fusion_modal and point is not None:
                pts = point[:, :, :3].contiguous()
                colors = point[:, :, 3:].contiguous()
                point_feats = self.encode_point_raw(pts, colors)
                point_feats = self.point_proj(point_feats).to(torch.bfloat16)
        
        # Step 2: 位置编码
        img_len = img_feats.shape[1] if img_feats is not None else 0
        point_len = point_feats.shape[1] if point_feats is not None else 0
        text_len = text_feats.shape[1] if text_feats is not None else 0
        total_len = img_len + point_len + text_len
        
        ids = torch.arange(total_len, device=device, dtype=torch.long)
        ids = ids.unsqueeze(0).expand(bsz, -1).unsqueeze(-1)
        pe = self.pe_embedder(ids).to(torch.bfloat16)
        
        # Step 3: 三流融合（模块 A）
        # 注意：保留完整三流结构，但根据 modal 只传入对应的特征
        curr_img = img_feats
        curr_point = point_feats
        curr_text = text_feats
        
        if self.use_fusion_blocks:
            for block in self.fusion_blocks:
                curr_img, curr_point, curr_text = block(curr_img, curr_point, curr_text, pe)
        
        # Step 4: 合并特征
        # Stage 1 特殊处理：三流融合后只取点云部分进入 MOE
        if self.stage == 1:
            # Stage 1: 只使用点云融合特征进入 MOE
            if curr_point is not None:
                fused_feats = curr_point  # [B, point_len, D]
            else:
                raise ValueError("Stage 1 requires point cloud input")
        else:
            # Stage 2: 根据 modal 合并所有模态的融合特征
            feats_list = []
            if curr_img is not None:
                feats_list.append(curr_img)
            if curr_point is not None:
                feats_list.append(curr_point)
            if curr_text is not None:
                feats_list.append(curr_text)
            fused_feats = torch.cat(feats_list, dim=1)
        
        # Step 5: MOE 处理
        # 使用 moe_modal 对应的 router/resample_tokens
        fused_feats = self.clip_proj1[moe_modal](fused_feats)
        
        tokens = self.resample_tokens[moe_modal].repeat(bsz, 1, 1)
        fused_feats = torch.cat([tokens, fused_feats], dim=1)
        
        routing_weights = self.routers[moe_modal](fused_feats).sigmoid()
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
        fused_feats = self.clip_proj2[moe_modal](fused_feats).to(torch.bfloat16)
        
        # Step 6: 投影到 CLIP 空间
        fused_embed_pooled = fused_feats.mean(dim=1)
        fused_embed_pooled = self.fused_to_clip_proj(fused_embed_pooled.float())
        
        return fused_embed_pooled, original_text_embed, original_image_embed

    def _infer_modal(self, image, point, text) -> str:
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
        if image is not None:
            return image.size(0), image.device
        elif point is not None:
            return point.size(0), point.device
        elif text is not None:
            return text.size(0), text.device
        else:
            raise ValueError("At least one modality must be provided")

    def _get_batch_info_embed(self, image_embed, point, text_embed):
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
        image_embed: Optional[torch.Tensor] = None,
        text_embed: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Stage 1 特殊处理：
        - 无论传入什么 modal，三流融合都使用全部模态 (ivt)
        - MOE 部分只使用 'v' 模态的 router/resample_tokens
        - 返回的 modal 仍为 'v'（用于记录）
        
        Stage 2:
        - 根据 modal 参数决定使用哪些模态
        """
        # Stage 1: modal 固定为 'v'，但需要传入全部特征用于三流融合
        if self.stage == 1:
            # 检查点云是否存在
            if pc is None:
                raise ValueError("Stage 1 requires point cloud input")
            # Stage 1 的 modal 固定为 'v'（用于 MOE）
            effective_modal = 'v'
        else:
            # Stage 2: 自动推断或使用传入的 modal
            if modal is None:
                if self.use_embed:
                    effective_modal = self._infer_modal_from_embed(image_embed, pc, text_embed)
                else:
                    effective_modal = self._infer_modal(image, pc, text)
            else:
                effective_modal = modal
            assert effective_modal in self.SUPPORTED_MODALS, f"Unsupported modal: {effective_modal}"
        
        # 调用 encode_multimodal
        # Stage 1: 会在内部使用全部模态进行三流融合，但 MOE 只用 'v'
        # Stage 2: 根据 effective_modal 决定
        fused_feats, txt_feats, image_feats = self.encode_multimodal(
            image=image,
            point=pc,
            text=text,
            modal=effective_modal,
            image_embed=image_embed,
            text_embed=text_embed
        )
        
        return {
            'fused_feats': fused_feats,
            'modal': effective_modal,
            'logit_scale': self.logit_scale.exp(),
            'txt_feats': txt_feats,
            'image_feats': image_feats
        }

    # ============ 阶段管理方法 ============
    
    def freeze_module_a(self, freeze_point_encoder=True):
        """冻结模块 A (融合模块)
        
        Args:
            freeze_point_encoder: 是否冻结点云编码器（Stage 2 时需要冻结）
        """
        print("[Uni3DMultimodalTwoStage] Freezing Module A (Fusion Blocks)...")
        
        # 冻结点云编码器（Stage 2 需要冻结）
        if freeze_point_encoder:
            for param in self.point_encoder.parameters():
                param.requires_grad = False
            print("  - Point Encoder: Frozen")
        
        # 冻结融合模块
        if self.use_fusion_blocks and hasattr(self, 'fusion_blocks'):
            for param in self.fusion_blocks.parameters():
                param.requires_grad = False
        
        # 冻结投影层
        if self.point_proj is not None:
            for param in self.point_proj.parameters():
                param.requires_grad = False
        
        if self.embed_image_proj is not None:
            for param in self.embed_image_proj.parameters():
                param.requires_grad = False
        
        if self.embed_text_proj is not None:
            for param in self.embed_text_proj.parameters():
                param.requires_grad = False
        
        if hasattr(self, 'pe_embedder'):
            for param in self.pe_embedder.parameters():
                param.requires_grad = False
        
        print("  Module A frozen")

    def unfreeze_module_a(self, unfreeze_point_encoder=True):
        """解冻模块 A (融合模块)
        
        Args:
            unfreeze_point_encoder: 是否解冻点云编码器（Stage 1 需要解冻参与训练）
        """
        print("[Uni3DMultimodalTwoStage] Unfreezing Module A (Fusion Blocks)...")
        
        # 解冻点云编码器（Stage 1 需要解冻参与训练）
        if unfreeze_point_encoder:
            for param in self.point_encoder.parameters():
                param.requires_grad = True
            print("  - Point Encoder: Unfrozen (will be trained)")
        
        # 解冻融合模块
        if self.use_fusion_blocks and hasattr(self, 'fusion_blocks'):
            for param in self.fusion_blocks.parameters():
                param.requires_grad = True
        
        # 解冻投影层
        if self.point_proj is not None:
            for param in self.point_proj.parameters():
                param.requires_grad = True
        
        if self.embed_image_proj is not None:
            for param in self.embed_image_proj.parameters():
                param.requires_grad = True
        
        if self.embed_text_proj is not None:
            for param in self.embed_text_proj.parameters():
                param.requires_grad = True
        
        # fused_to_clip_proj 也解冻
        if hasattr(self, 'fused_to_clip_proj'):
            for param in self.fused_to_clip_proj.parameters():
                param.requires_grad = True
        
        print("  Module A unfrozen")

    def freeze_module_b(self):
        """冻结模块 B (MOE 模块)"""
        print("[Uni3DMultimodalTwoStage] Freezing Module B (MOE)...")
        
        for param in self.routers.parameters():
            param.requires_grad = False
        
        for param in self.resample_layers.parameters():
            param.requires_grad = False
        
        for param in self.clip_proj1.parameters():
            param.requires_grad = False
        
        for param in self.clip_proj2.parameters():
            param.requires_grad = False
        
        # 根据当前阶段冻结对应的 resample_tokens
        modals_to_freeze = self.STAGE1_MODALS if self.stage == 1 else self.SUPPORTED_MODALS
        for modal in modals_to_freeze:
            if modal in self.resample_tokens:
                self.resample_tokens[modal].requires_grad = False
        
        print("  Module B frozen")

    def unfreeze_module_b(self):
        """解冻模块 B (MOE 模块)"""
        print("[Uni3DMultimodalTwoStage] Unfreezing Module B (MOE)...")
        
        for param in self.routers.parameters():
            param.requires_grad = True
        
        for param in self.resample_layers.parameters():
            param.requires_grad = True
        
        for param in self.clip_proj1.parameters():
            param.requires_grad = True
        
        for param in self.clip_proj2.parameters():
            param.requires_grad = True
        
        # 根据当前阶段解冻对应的 resample_tokens
        modals_to_unfreeze = self.STAGE1_MODALS if self.stage == 1 else self.SUPPORTED_MODALS
        for modal in modals_to_unfreeze:
            if modal in self.resample_tokens:
                self.resample_tokens[modal].requires_grad = True
        
        # fused_to_clip_proj 在 stage 2 也需要训练
        if hasattr(self, 'fused_to_clip_proj'):
            for param in self.fused_to_clip_proj.parameters():
                param.requires_grad = True
        
        print("  Module B unfrozen")

    def get_trainable_params_info(self):
        """获取可训练参数统计信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        info = {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0
        }
        
        print(f"\n📊 Parameter Statistics:")
        print(f"   Total: {total_params/1e6:.2f}M")
        print(f"   Trainable: {trainable_params/1e6:.2f}M ({info['trainable_ratio']*100:.1f}%)")
        print(f"   Frozen: {frozen_params/1e6:.2f}M ({(1-info['trainable_ratio'])*100:.1f}%)")
        
        return info


# ============ 工厂函数 ============

def create_uni3d_multimodal_two_stage(args, stage: int = 1, load_pretrained: bool = True):
    """
    创建两阶段多模态 Uni3D 模型
    
    Args:
        args: 配置参数
        stage: 训练阶段 (1 或 2)
        load_pretrained: 是否加载原始预训练权重
                        - 训练时设为 True，会加载 args.pretrained_pc 指定的预训练点云编码器
                        - 测试时设为 False，因为会从 checkpoint 加载完整模型权重
    
    Returns:
        model: Uni3DMultimodalTwoStage 模型
    """
    model = Uni3DMultimodalTwoStage(args, stage=stage, load_pretrained=load_pretrained)
    return model
