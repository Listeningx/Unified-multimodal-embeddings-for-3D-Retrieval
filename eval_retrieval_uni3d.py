#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSO数据集跨模态检索评估脚本 - Uni3D Multimodal版本

支持的模型:
- Uni3D Multimodal: 基础多模态融合模型
- Uni3D Multimodal Two-Stage: 两阶段训练的多模态融合模型

支持的模态组合:
- i: 图像 (使用预提取的CLIP特征)
- v: 点云/3D (使用Uni3D编码)  
- t: 文本 (使用预提取的CLIP特征)
- iv: 图像 + 点云
- it: 图像 + 文本
- vt: 点云 + 文本
- ivt: 图像 + 点云 + 文本

注意: 
- 此脚本使用预提取的图文特征 (use_embed=True)
- 点云特征由 Uni3D 实时编码
"""

import sys
import os
import argparse
import json
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from PIL import Image
from collections import defaultdict
from datetime import datetime
from scipy import stats
import open3d as o3d
import trimesh

# 添加Uni3D代码路径
sys.path.insert(0, '/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D')

# 用于点云归一化
def normalize_pc(pc):
    # normalize pc to [-1, 1]
    pc = pc - np.mean(pc, axis=0)
    if np.max(np.linalg.norm(pc, axis=1)) < 1e-6:
        pc = np.zeros_like(pc)
    else:
        pc = pc / np.max(np.linalg.norm(pc, axis=1))
    return pc

# Uni3D Multimodal 相关
from easydict import EasyDict

# 可选: 加载 OpenCLIP 用于提取图文特征（如果没有预提取的特征）
try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    OPENCLIP_AVAILABLE = False
    print("Warning: open_clip not available. Must use pre-extracted embeddings.")

# Uni3D Multimodal 支持的模态组合
UNI3D_VALID_MODALS = ['i', 'v', 't', 'iv', 'it', 'vt', 'ivt']


def get_modal_cache_path(output_dir, modal_str):
    """获取单模态 embedding 缓存文件路径"""
    return os.path.join(output_dir, f'embeddings_cache_modal_{modal_str}.npz')


def load_embedding_cache(output_dir, modal_str):
    """
    从缓存文件加载某个模态的 embedding
    
    Args:
        output_dir: 输出目录
        modal_str: 模态字符串
    
    Returns:
        embedding_dict: {unique_key: torch.Tensor} 或 None（缓存不存在）
    """
    cache_path = get_modal_cache_path(output_dir, modal_str)
    if not os.path.exists(cache_path):
        return None
    
    try:
        print(f"  📦 发现模态 '{modal_str}' 的 embedding 缓存: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        unique_keys = list(data['unique_keys'])
        embeddings = data['embeddings']  # [N, embed_dim]
        
        embedding_dict = {}
        for i, key in enumerate(unique_keys):
            embedding_dict[key] = torch.from_numpy(embeddings[i])
        
        print(f"  ✅ 从缓存加载了 {len(embedding_dict)} 个 embedding (模态: {modal_str})")
        return embedding_dict
    except Exception as e:
        print(f"  ⚠️ 加载 embedding 缓存失败: {e}")
        return None


def save_embedding_cache(output_dir, modal_str, embedding_dict):
    """
    将某个模态的 embedding 保存为缓存文件
    
    Args:
        output_dir: 输出目录
        modal_str: 模态字符串
        embedding_dict: {unique_key: torch.Tensor}
    """
    cache_path = get_modal_cache_path(output_dir, modal_str)
    
    # 如果缓存已存在则跳过
    if os.path.exists(cache_path):
        print(f"  📦 模态 '{modal_str}' 的 embedding 缓存已存在，跳过保存")
        return
    
    try:
        unique_keys = list(embedding_dict.keys())
        embeddings = torch.stack([embedding_dict[key] for key in unique_keys]).numpy()
        
        np.savez(
            cache_path,
            modal=modal_str,
            unique_keys=np.array(unique_keys, dtype=object),
            embeddings=embeddings
        )
        print(f"  💾 已保存模态 '{modal_str}' 的 embedding 缓存 ({len(unique_keys)} 个) -> {cache_path}")
    except Exception as e:
        print(f"  ⚠️ 保存 embedding 缓存失败: {e}")


# ==================== GLB Mesh → 点云转换 (Objaverse专用) ====================

def extract_pointcloud_from_glb(glb_path: str, num_points: int = 10000):
    """
    从GLB文件提取点云（包含颜色）
    
    Args:
        glb_path: GLB文件路径
        num_points: 采样点数
        
    Returns:
        点云数组 [num_points, 6]，包含 xyz + rgb（float32）
        如果加载失败则返回 None
    """
    if not os.path.exists(glb_path):
        print(f"  ⚠️  GLB文件不存在: {glb_path}")
        return None
    
    try:
        mesh = trimesh.load(glb_path, force='mesh')
        
        if isinstance(mesh, trimesh.Scene):
            meshes = []
            for name, geom in mesh.geometry.items():
                if isinstance(geom, trimesh.Trimesh):
                    meshes.append(geom)
            if not meshes:
                print(f"  ⚠️  GLB场景中没有有效的Trimesh几何体: {glb_path}")
                return None
            mesh = trimesh.util.concatenate(meshes)
        
        if not isinstance(mesh, trimesh.Trimesh):
            print(f"  ⚠️  无法将GLB转换为Trimesh对象: {glb_path}")
            return None
        
        points, face_indices = mesh.sample(num_points, return_index=True)
        
        if mesh.visual.kind == 'vertex':
            colors = mesh.visual.vertex_colors[mesh.faces[face_indices]].mean(axis=1)[:, :3] / 255.0
        elif mesh.visual.kind == 'texture':
            try:
                colors = mesh.visual.to_color().vertex_colors[mesh.faces[face_indices]].mean(axis=1)[:, :3] / 255.0
            except:
                colors = np.ones((num_points, 3)) * 0.5
        else:
            colors = np.ones((num_points, 3)) * 0.5
        
        pointcloud = np.concatenate([points, colors], axis=1)
        return pointcloud.astype(np.float32)
        
    except Exception as e:
        print(f"  ⚠️  提取点云失败 {glb_path}: {e}")
        return None


def batch_convert_glb_to_pointcloud(glb_paths, cache_dir=None, num_points=10000):
    """
    批量将GLB文件转换为点云，支持缓存
    
    缓存策略：
    - 使用物体ID（UUID，从unique_key中提取）作为缓存文件名
    - 同一个GLB文件被多个unique_key引用时，只转换一次
    
    Args:
        glb_paths: {unique_key: glb_path} GLB文件路径字典
        cache_dir: 点云缓存目录
        num_points: 采样点数
    
    Returns:
        pointcloud_dict: {unique_key: np.ndarray [N, 6]} 点云字典
    """
    pointcloud_dict = {}
    
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    
    cached_count = 0
    converted_count = 0
    failed_count = 0
    skipped_dedup_count = 0
    
    glb_result_cache = {}
    
    for unique_key, glb_path in tqdm(glb_paths.items(), desc="转换GLB→点云"):
        object_id = unique_key.split('||')[0]
        
        # 1. 尝试从磁盘缓存加载
        if cache_dir:
            cache_file = os.path.join(cache_dir, f"{object_id}.npy")
            
            if os.path.exists(cache_file):
                try:
                    pc = np.load(cache_file)
                    pointcloud_dict[unique_key] = pc
                    glb_result_cache[glb_path] = pc
                    cached_count += 1
                    print('从缓存文件加载')
                    continue
                except:
                    pass
        
        # 2. 尝试从内存去重缓存获取
        if glb_path in glb_result_cache:
            pc = glb_result_cache[glb_path]
            if pc is not None:
                pointcloud_dict[unique_key] = pc
                skipped_dedup_count += 1
                print('从本轮实时缓存加载')
                continue
            else:
                failed_count += 1
                continue
        
        # 3. 真正执行GLB→点云转换
        pc = extract_pointcloud_from_glb(glb_path, num_points=num_points)
        print('执行转换')

        glb_result_cache[glb_path] = pc
        
        if pc is not None:
            pointcloud_dict[unique_key] = pc
            converted_count += 1
            
            if cache_dir:
                cache_file = os.path.join(cache_dir, f"{object_id}.npy")
                try:
                    np.save(cache_file, pc)
                except:
                    pass
        else:
            failed_count += 1
    
    print(f"\n点云转换统计: 磁盘缓存加载={cached_count}, 内存去重复用={skipped_dedup_count}, "
          f"新转换={converted_count}, 失败={failed_count}")
    return pointcloud_dict


def setup_seed(seed=2022):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_uni3d_multimodal_model(args):
    """
    加载Uni3D Multimodal模型
    
    Args:
        args: 包含模型配置的参数
    
    Returns:
        model: Uni3D Multimodal模型
    """
    print("\n" + "="*70)
    print("加载Uni3D Multimodal模型...")
    print("="*70)
    
    # 构建模型配置
    model_args = EasyDict()
    model_args.embed_dim = args.embed_dim
    model_args.pc_feat_dim = args.pc_feat_dim
    model_args.pc_encoder_dim = args.pc_encoder_dim
    model_args.group_size = args.group_size
    model_args.num_group = args.num_group
    model_args.drop_path_rate = args.drop_path_rate
    model_args.patch_dropout = args.patch_dropout
    model_args.pc_model = args.pc_model
    model_args.use_embed = args.use_embed
    model_args.use_fusion_blocks = args.use_fusion_blocks
    model_args.clip_model = args.clip_model
    model_args.clip_model_path = args.pretrained
    model_args.use_distributed = False  # 评测时不使用分布式
    
    # 根据模型类型创建模型
    if args.model_type == 'two_stage':
        from models.uni3d_multimodal_two_stage import create_uni3d_multimodal_two_stage
        # 测试时设置 load_pretrained=False，因为 checkpoint 已包含完整权重
        model = create_uni3d_multimodal_two_stage(model_args, stage=args.stage, load_pretrained=False)
        print(f"  模型类型: Two-Stage (stage={args.stage})")
    else:
        from models.uni3d_multimodal import create_uni3d_multimodal
        model = create_uni3d_multimodal(model_args, load_pretrained=False)
        print(f"  模型类型: Multimodal")
    
    # 加载检查点
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"  检查点: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        
        # 尝试不同的 key 来获取模型权重
        state_dict = None
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("  使用 key: model_state_dict")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("  使用 key: state_dict")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("  使用 key: model")
        elif 'module' in checkpoint:
            state_dict = checkpoint['module']
            print("  使用 key: module")
        else:
            state_dict = checkpoint
            print("  使用整个 checkpoint 作为 state_dict")
        
        # 清理 state_dict 中的前缀
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                cleaned_state_dict[k[7:]] = v
            elif k.startswith('uni3d.'):
                cleaned_state_dict[k[6:]] = v
            else:
                cleaned_state_dict[k] = v
        
        # 加载权重
        missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
        print(f"  加载完成!")
        print(f"    Missing keys: {len(missing)}")
        print(f"    Unexpected keys: {len(unexpected)}")
        if missing and len(missing) <= 10:
            print(f"    Missing: {missing}")
        if unexpected and len(unexpected) <= 10:
            print(f"    Unexpected: {unexpected}")
            
        # 显示 checkpoint 中的其他信息
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'step' in checkpoint:
            print(f"  Step: {checkpoint['step']}")
    else:
        print(f"  ⚠️ 警告: 未找到检查点文件 {args.checkpoint}")
        print("  将使用随机初始化的模型（可能不是预期的行为）")
    
    model.cuda()
    model.eval()
    
    print("  ✅ Uni3D Multimodal模型加载完成！")
    print("="*70 + "\n")
    
    return model


def load_openclip_model(args):
    """
    加载OpenCLIP模型（用于提取图像和文本特征）
    """
    if not OPENCLIP_AVAILABLE:
        raise RuntimeError("open_clip not available")
    
    print("\n加载OpenCLIP模型...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        model_name=args.clip_model, pretrained=args.pretrained
    )
    clip_model.cuda().eval()
    print("  OpenCLIP模型加载完成！")
    return clip_model, clip_preprocess


def load_ply_uni3d(file_name, num_points=10000, y_up=True):
    """
    加载点云文件并返回 Uni3D 格式的输入
    
    Args:
        file_name: 点云文件路径
        num_points: 采样点数
        y_up: 是否将 Y 轴作为向上方向
    
    Returns:
        pc_tensor: [1, num_points, 6] 的张量 (xyz + rgb)
    """
    pcd = o3d.io.read_point_cloud(file_name)
    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors)
    n = xyz.shape[0]
    
    # 采样或填充
    if n != num_points:
        if n > num_points:
            idx = random.sample(range(n), num_points)
        else:
            idx = np.random.choice(n, num_points, replace=True)
        xyz = xyz[idx]
        rgb = rgb[idx] if rgb is not None and rgb.size > 0 else None
    
    # Y-up 转换
    if y_up:
        xyz[:, [1, 2]] = xyz[:, [2, 1]]
    
    # 归一化
    xyz = normalize_pc(xyz)
    
    # 处理缺失的颜色
    if rgb is None or rgb.size == 0:
        rgb = np.ones_like(xyz) * 0.4
    
    # 合并为 [num_points, 6] 的数组
    pc_data = np.concatenate([xyz, rgb], axis=1)
    
    # 转换为张量并添加 batch 维度
    pc_tensor = torch.from_numpy(pc_data).float().unsqueeze(0)  # [1, num_points, 6]
    
    return pc_tensor


def load_image(image_path):
    """加载图片"""
    img = Image.open(image_path).convert('RGB')
    return img


@torch.no_grad()
def extract_text_feat_clip(texts, clip_model):
    """使用OpenCLIP提取文本特征"""
    if isinstance(texts, str):
        texts = [texts]
    text_tokens = open_clip.tokenizer.tokenize(texts).cuda()
    text_features = clip_model.encode_text(text_tokens)
    text_features = F.normalize(text_features, p=2, dim=1)
    return text_features


@torch.no_grad()
def extract_image_feat_clip(images, clip_model, clip_preprocess):
    """使用OpenCLIP提取图像特征"""
    if not isinstance(images, list):
        images = [images]
    image_tensors = [clip_preprocess(image) for image in images]
    image_tensors = torch.stack(image_tensors, dim=0).float().cuda()
    image_features = clip_model.encode_image(image_tensors)
    image_features = image_features.reshape((-1, image_features.shape[-1]))
    image_features = F.normalize(image_features, p=2, dim=1)
    return image_features


@torch.no_grad()
def extract_multimodal_feat_uni3d(
    pc_tensor, 
    uni3d_model, 
    modal='v',
    image_embed=None, 
    text_embed=None
):
    """
    使用Uni3D Multimodal提取多模态融合特征
    
    Args:
        pc_tensor: [B, num_points, 6] 点云数据 (xyz + rgb)，如果 modal 不包含 'v' 可以为 None
        uni3d_model: Uni3D Multimodal模型
        modal: 使用的模态组合，例如 'v', 'iv', 'ivt' 等
        image_embed: [B, embed_dim] 预提取的图像特征（可选）
        text_embed: [B, embed_dim] 预提取的文本特征（可选）
    
    Returns:
        fused_features: [B, embed_dim] 融合特征
    """
    # 移动到 GPU
    if pc_tensor is not None:
        pc_tensor = pc_tensor.cuda()
    if image_embed is not None:
        image_embed = image_embed.cuda()
    if text_embed is not None:
        text_embed = text_embed.cuda()
    
    # 调用 encode_multimodal
    fused_features, _, _ = uni3d_model.encode_multimodal(
        point=pc_tensor,
        modal=modal,
        image_embed=image_embed,
        text_embed=text_embed
    )
    
    # 归一化
    fused_features = F.normalize(fused_features, p=2, dim=-1)
    return fused_features


def collect_all_objects(json_files, args):
    """
    收集所有JSON文件中的物体信息
    
    Returns:
        object_dict: {unique_key: object_info}
        json_object_mapping: {json_file: [unique_keys]}
    """
    print(f"\n{'='*70}")
    print(f"收集物体信息 (共 {len(json_files)} 个 JSON 文件)")
    print(f"{'='*70}\n")
    
    object_dict = {}
    json_object_mapping = {}
    
    for json_file in tqdm(json_files, desc="扫描 JSON 文件"):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        items = data.get('data', [])
        json_object_mapping[str(json_file)] = []
        
        for item in items:
            obj_id = item['id']
            image_path = item['image']
            unique_key = f"{obj_id}||{image_path}"
            
            if unique_key not in object_dict:
                if args.dataset == 'esb':
                    pc_path = '/cfs_160T/serenasnliu/ranking/OS-ESB-core/'+ item['pointcloud']
                    image_path = '/cfs_160T/serenasnliu/ranking/OS-ESB-core/'+ image_path
                elif args.dataset == 'ntu':
                    pc_path = '/cfs_160T/serenasnliu/ranking/OS-NTU-core/' + item['pointcloud']
                    image_path = '/cfs_160T/serenasnliu/ranking/OS-NTU-core/'+ image_path
                elif args.dataset == 'mn40':
                    pc_path = '/cfs_160T/serenasnliu/ranking/OS-MN40-core/'+ item['pointcloud']
                    image_path = '/cfs_160T/serenasnliu/ranking/OS-MN40-core/'+ image_path
                elif args.dataset == 'gso':
                    pc_path = item['pointcloud']
                    image_path = item['image']
                elif args.dataset == 'objaverse':
                    # objaverse的pointcloud字段实际指向.glb文件，路径直接使用
                    pc_path = item['pointcloud']
                    image_path = item['image']

                object_dict[unique_key] = {
                    'id': obj_id,
                    'image': image_path,
                    'pointcloud': pc_path,
                    'text': item.get('text', ''),
                    'category': item.get('category', ''),
                    'score': item.get('score', 0)
                }
            
            json_object_mapping[str(json_file)].append(unique_key)
    
    print(f"\n{'='*70}")
    print(f"收集完成:")
    print(f"  唯一 (物体, 图像) 对数: {len(object_dict)}")
    print(f"{'='*70}\n")
    
    return object_dict, json_object_mapping


@torch.no_grad()
def extract_all_embeddings_uni3d(
    object_dict, 
    args, 
    modal_str,
    uni3d_model,
    clip_model=None, 
    clip_preprocess=None,
    pointcloud_cache_dict=None
):
    """
    使用 Uni3D Multimodal 批量提取所有物体的 embedding
    
    Args:
        object_dict: 物体信息字典
        args: 参数
        modal_str: 模态字符串，例如 'v', 'iv', 'ivt'
        uni3d_model: Uni3D Multimodal模型
        clip_model: OpenCLIP模型（用于提取图像和文本特征）
        clip_preprocess: 图像预处理函数
        pointcloud_cache_dict: {unique_key: np.ndarray} 预转换的点云缓存（Objaverse专用）
    
    Returns:
        embedding_dict: {unique_key: embedding}
    """
    if pointcloud_cache_dict is None:
        pointcloud_cache_dict = {}
    
    is_objaverse = (args.dataset == 'objaverse')
    modal_str = modal_str.lower()
    
    # 验证模态
    if modal_str not in UNI3D_VALID_MODALS:
        raise ValueError(f"Invalid modal: {modal_str}. Valid options: {UNI3D_VALID_MODALS}")
    
    modal_names = {
        'i': '图像 (Image)',
        'v': '点云 (Point Cloud)',
        't': '文本 (Text)',
        'iv': '图像 + 点云',
        'it': '图像 + 文本',
        'vt': '点云 + 文本',
        'ivt': '图像 + 点云 + 文本'
    }
    
    print(f"\n{'='*70}")
    print(f"提取 Uni3D Multimodal 特征")
    print(f"{'='*70}")
    print(f"模态: {modal_str} - {modal_names.get(modal_str, modal_str)}")
    print(f"物体数量: {len(object_dict)}")
    print(f"批次大小: {args.batch_size}")
    print(f"{'='*70}\n")
    
    embedding_dict = {}
    failed_objects = []
    
    unique_keys = list(object_dict.keys())
    batch_size = args.batch_size
    
    # 判断需要哪些模态
    need_image = 'i' in modal_str
    need_text = 't' in modal_str
    need_point = 'v' in modal_str
    
    # ============ Step 1: 预提取图像和文本特征 ============
    image_embed_dict = {}
    text_embed_dict = {}
    
    if need_image:
        if clip_model is None:
            raise ValueError("需要 CLIP 模型来提取图像特征，但 clip_model 为 None")
        
        print("Step 1/3: 预提取图像特征...")
        num_batches = (len(unique_keys) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="提取图像特征"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(unique_keys))
            batch_keys = unique_keys[start_idx:end_idx]
            
            batch_images = []
            valid_keys = []
            
            for unique_key in batch_keys:
                obj_info = object_dict[unique_key]
                try:
                    img_path = obj_info['image']
                    img = load_image(img_path)
                    batch_images.append(img)
                    valid_keys.append(unique_key)
                except Exception as e:
                    print(f"  ⚠️ 加载图像失败 {obj_info['id']}: {e}")
                    continue
            
            if len(batch_images) > 0:
                try:
                    image_feats = extract_image_feat_clip(batch_images, clip_model, clip_preprocess)
                    for i, unique_key in enumerate(valid_keys):
                        image_embed_dict[unique_key] = image_feats[i].cpu()
                except Exception as e:
                    print(f"  ⚠️ 提取图像特征失败: {e}")
        
        print(f"  ✅ 图像特征提取完成: {len(image_embed_dict)} 个")
    
    if need_text:
        if clip_model is None:
            raise ValueError("需要 CLIP 模型来提取文本特征，但 clip_model 为 None")
        
        print("Step 2/3: 预提取文本特征...")
        num_batches = (len(unique_keys) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="提取文本特征"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(unique_keys))
            batch_keys = unique_keys[start_idx:end_idx]
            
            batch_texts = []
            valid_keys = []
            
            for unique_key in batch_keys:
                obj_info = object_dict[unique_key]
                text = obj_info.get('text', '')
                if not text:
                    text = f"a {obj_info.get('category', 'object')}"
                batch_texts.append(text)
                valid_keys.append(unique_key)
            
            if len(batch_texts) > 0:
                try:
                    text_feats = extract_text_feat_clip(batch_texts, clip_model)
                    for i, unique_key in enumerate(valid_keys):
                        text_embed_dict[unique_key] = text_feats[i].cpu()
                except Exception as e:
                    print(f"  ⚠️ 提取文本特征失败: {e}")
        
        print(f"  ✅ 文本特征提取完成: {len(text_embed_dict)} 个")
    
    # ============ Step 2: 使用 Uni3D 提取融合特征 ============
    print("Step 3/3: 使用 Uni3D Multimodal 提取融合特征...")
    
    for unique_key in tqdm(unique_keys, desc=f"提取 {modal_str} 融合特征"):
        obj_info = object_dict[unique_key]
        
        try:
            # 准备输入
            pc_tensor = None
            image_embed = None
            text_embed = None
            
            # 加载点云
            if need_point:
                pc_array = None
                
                # 优先级1: 从内存中的pointcloud_cache_dict加载
                if unique_key in pointcloud_cache_dict:
                    pc_array = pointcloud_cache_dict[unique_key]
                
                # 优先级2: 对Objaverse数据，尝试从磁盘缓存目录直接读取npy文件
                if pc_array is None and is_objaverse and args.pc_cache_dir:
                    object_id = unique_key.split('||')[0]
                    cache_file = os.path.join(args.pc_cache_dir, f"{object_id}.npy")
                    if os.path.exists(cache_file):
                        try:
                            pc_array = np.load(cache_file)
                            # 同时放入内存缓存，避免后续重复IO
                            pointcloud_cache_dict[unique_key] = pc_array
                        except Exception as e:
                            print(f"  ⚠️ 从缓存加载点云失败 {cache_file}: {e}")
                
                # 优先级3: 对Objaverse数据，从GLB文件实时转换
                if pc_array is None and is_objaverse:
                    glb_path = obj_info['pointcloud']
                    if glb_path and os.path.exists(glb_path):
                        pc_array = extract_pointcloud_from_glb(glb_path, num_points=args.num_points)
                        if pc_array is not None:
                            # 保存到磁盘缓存
                            if args.pc_cache_dir:
                                object_id = unique_key.split('||')[0]
                                cache_file = os.path.join(args.pc_cache_dir, f"{object_id}.npy")
                                os.makedirs(args.pc_cache_dir, exist_ok=True)
                                try:
                                    np.save(cache_file, pc_array)
                                except:
                                    pass
                            # 放入内存缓存
                            pointcloud_cache_dict[unique_key] = pc_array
                
                if pc_array is not None:
                    # 从缓存的numpy数组构建点云tensor（Objaverse数据）
                    xyz = pc_array[:, :3]
                    rgb = pc_array[:, 3:6]
                    xyz = normalize_pc(xyz)
                    pc_data = np.concatenate([xyz, rgb], axis=1)
                    pc_tensor = torch.from_numpy(pc_data).float().unsqueeze(0)  # [1, N, 6]
                elif not is_objaverse:
                    # 从.ply文件直接加载（gso/esb/ntu/mn40等数据集）
                    pc_path = obj_info['pointcloud']
                    if not pc_path:
                        print(f"  ⚠️ 缺少点云路径: {obj_info['id']}")
                        failed_objects.append(unique_key)
                        continue
                    pc_tensor = load_ply_uni3d(pc_path, num_points=args.num_points, y_up=True)
                else:
                    print(f"  ⚠️ 无法获取点云: {obj_info['id']}")
                    failed_objects.append(unique_key)
                    continue
            
            # 获取图像特征
            if need_image and unique_key in image_embed_dict:
                image_embed = image_embed_dict[unique_key].unsqueeze(0)
            elif need_image:
                print(f"  ⚠️ 缺少图像特征: {obj_info['id']}")
                failed_objects.append(unique_key)
                continue
            
            # 获取文本特征
            if need_text and unique_key in text_embed_dict:
                text_embed = text_embed_dict[unique_key].unsqueeze(0)
            elif need_text:
                print(f"  ⚠️ 缺少文本特征: {obj_info['id']}")
                failed_objects.append(unique_key)
                continue
            
            # 提取融合特征
            fused_feat = extract_multimodal_feat_uni3d(
                pc_tensor, uni3d_model, modal=modal_str,
                image_embed=image_embed, text_embed=text_embed
            )
            
            embedding_dict[unique_key] = fused_feat.cpu().squeeze()
            
        except Exception as e:
            print(f"\n  ⚠️ 提取 {obj_info['id']} 特征失败: {e}")
            import traceback
            traceback.print_exc()
            failed_objects.append(unique_key)
            continue
    
    print(f"\n{'='*70}")
    print(f"特征提取完成 ({modal_str}):")
    print(f"  成功: {len(embedding_dict)}")
    print(f"  失败: {len(failed_objects)}")
    print(f"{'='*70}\n")
    
    return embedding_dict


def compute_cosine_similarity(query_emb, candidate_embs):
    """计算余弦相似度"""
    query_emb = query_emb.numpy() if isinstance(query_emb, torch.Tensor) else query_emb
    candidate_embs = candidate_embs.numpy() if isinstance(candidate_embs, torch.Tensor) else candidate_embs
    
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    candidate_norms = candidate_embs / (np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-8)
    
    similarities = np.dot(candidate_norms, query_norm)
    return similarities


def count_inversions(scores, predicted_order):
    """计算逆序对数量"""
    n = len(scores)
    inversions = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            idx_i = predicted_order[i]
            idx_j = predicted_order[j]
            
            if scores[idx_i] < scores[idx_j]:
                inversions += 1
    
    return inversions


def compute_kendall_tau(scores, predicted_order, query_category, categories):
    """计算Kendall Tau（仅同类）"""
    same_category_indices = [i for i, cat in enumerate(categories) if cat == query_category]
    
    if len(same_category_indices) < 2:
        return None
    
    same_category_scores = [scores[i] for i in same_category_indices]
    
    predicted_ranks = []
    for idx in same_category_indices:
        rank = np.where(predicted_order == idx)[0][0]
        predicted_ranks.append(rank)
    
    tau, _ = stats.kendalltau(same_category_scores, [-r for r in predicted_ranks])
    return tau


def compute_spearman_rho(scores, predicted_order, query_category, categories):
    """计算Spearman Rho（仅同类）"""
    same_category_indices = [i for i, cat in enumerate(categories) if cat == query_category]
    
    if len(same_category_indices) < 2:
        return None
    
    same_category_scores = [scores[i] for i in same_category_indices]
    
    predicted_ranks = []
    for idx in same_category_indices:
        rank = np.where(predicted_order == idx)[0][0]
        predicted_ranks.append(rank)
    
    rho, _ = stats.spearmanr(same_category_scores, [-r for r in predicted_ranks])
    return rho


def compute_ndcg(scores, predicted_order, k=None):
    """计算NDCG"""
    n = len(scores)
    if k is None:
        k = n
    k = min(k, n)
    
    relevances = np.array(scores)
    
    dcg = 0.0
    for i in range(k):
        idx = predicted_order[i]
        rel = relevances[idx]
        dcg += rel / np.log2(i + 2)
    
    ideal_order = np.argsort(relevances)[::-1]
    idcg = 0.0
    for i in range(k):
        idx = ideal_order[i]
        rel = relevances[idx]
        idcg += rel / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def evaluate_json_file(json_path, object_dict, query_embedding_dict, gallery_embedding_dict, args):
    """
    评估单个JSON文件
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    items = data['data']
    
    if len(items) < 2:
        print(f"⚠️  Warning: {os.path.basename(json_path)} 物体数量少于2，跳过...")
        return None
    
    query_item = items[0]
    candidate_items = items[1:]
    
    query_id = query_item['id']
    query_image = query_item['image']
    query_unique_key = f"{query_id}||{query_image}"
    
    if query_unique_key not in query_embedding_dict:
        print(f"❌ Error: Query {query_id} embedding not found!")
        return None
    
    query_emb = query_embedding_dict[query_unique_key]
    
    candidate_embs = []
    candidate_scores = []
    candidate_categories = []
    candidate_ids = []
    candidate_images = []
    
    for item in candidate_items:
        cand_id = item['id']
        cand_image = item['image']
        cand_unique_key = f"{cand_id}||{cand_image}"
        
        if cand_unique_key not in gallery_embedding_dict:
            continue
        
        candidate_embs.append(gallery_embedding_dict[cand_unique_key])
        candidate_scores.append(item['score'])
        candidate_categories.append(item['category'])
        candidate_ids.append(cand_id)
        candidate_images.append(cand_image)
    
    if len(candidate_embs) == 0:
        print(f"❌ Error: {os.path.basename(json_path)} 没有有效的候选物体!")
        return None
    
    candidate_embs = torch.stack(candidate_embs)
    
    similarities = compute_cosine_similarity(query_emb, candidate_embs)
    predicted_order = np.argsort(similarities)[::-1]
    
    inversions = count_inversions(candidate_scores, predicted_order)
    kendall_tau = compute_kendall_tau(candidate_scores, predicted_order, 
                                       query_item['category'], candidate_categories)
    spearman_rho = compute_spearman_rho(candidate_scores, predicted_order,
                                         query_item['category'], candidate_categories)
    ndcg_full = compute_ndcg(candidate_scores, predicted_order)
    ndcg_10 = compute_ndcg(candidate_scores, predicted_order, k=10)
    ndcg_5 = compute_ndcg(candidate_scores, predicted_order, k=5)
    
    results = {
        'json_file': os.path.basename(json_path),
        'query_id': query_id,
        'query_image': query_image,
        'query_category': query_item['category'],
        'query_modal': args.query_modal,
        'gallery_modal': args.gallery_modal,
        'num_candidates': len(candidate_items),
        'inversions': inversions,
        'kendall_tau': kendall_tau,
        'spearman_rho': spearman_rho,
        'ndcg': ndcg_full,
        'ndcg@10': ndcg_10,
        'ndcg@5': ndcg_5,
        'predicted_order': [{'id': candidate_ids[i], 'image': candidate_images[i]} for i in predicted_order],
        'similarities': similarities.tolist(),
        'candidate_scores': candidate_scores,
        'candidate_categories': candidate_categories
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='GSO Cross-Modal Retrieval with Uni3D Multimodal')
    
    # ========== 数据参数 ==========
    parser.add_argument('--json_dir', type=str, required=True,
                        help='JSON文件目录')
    parser.add_argument('--json_pattern', type=str, default='*.json',
                        help='JSON文件匹配模式')
    parser.add_argument('--dataset', type=str, default='gso',
                        choices=['gso', 'esb', 'ntu', 'mn40', 'objaverse'])
    
    # ========== 模态参数 ==========
    parser.add_argument('--query_modal', type=str, default='v',
                        help='Query 模态: i, v, t, iv, it, vt, ivt')
    parser.add_argument('--gallery_modal', type=str, default='v',
                        help='Gallery 模态: i, v, t, iv, it, vt, ivt')
    
    # ========== 输出参数 ==========
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    
    # ========== Uni3D Multimodal 模型参数 ==========
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Uni3D Multimodal 模型检查点路径')
    parser.add_argument('--model_type', type=str, default='two_stage',
                        choices=['two_stage', 'multimodal'],
                        help='模型类型')
    parser.add_argument('--stage', type=int, default=2,
                        help='Two-stage 模型的阶段 (1 或 2)')
    
    # 模型架构参数
    parser.add_argument('--embed_dim', type=int, default=1280,
                        help='CLIP 嵌入维度')
    parser.add_argument('--pc_feat_dim', type=int, default=1408,
                        help='点云特征维度')
    parser.add_argument('--pc_encoder_dim', type=int, default=512,
                        help='点云编码器维度')
    parser.add_argument('--group_size', type=int, default=64,
                        help='点云分组大小')
    parser.add_argument('--num_group', type=int, default=512,
                        help='点云分组数量')
    parser.add_argument('--drop_path_rate', type=float, default=0.2,
                        help='DropPath 率')
    parser.add_argument('--patch_dropout', type=float, default=0.5,
                        help='Patch Dropout 率')
    parser.add_argument('--pc_model', type=str, 
                        default='eva_giant_patch14_560.m30m_ft_in22k_in1k',
                        help='点云 Transformer 模型')
    parser.add_argument('--use_embed', action='store_true', default=True,
                        help='使用预计算的embedding')
    parser.add_argument('--use_fusion_blocks', action='store_true', default=True,
                        help='使用融合块')
    

   # CLIP 参数
    parser.add_argument('--clip_model', type=str, default='ViT-bigG-14', help='CLIP 模型')
    parser.add_argument('--pretrained', type=str, default='/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/clip_model/open_clip_pytorch_model.bin', help='CLIP 预训练权重')
    

    
    # ========== 通用参数 ==========
    parser.add_argument('--num_points', type=int, default=10000,
                        help='点云采样点数')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--seed', default=2022, type=int,
                        help='随机种子')
    
    # ========== Objaverse 专用参数 ==========
    parser.add_argument('--pc_cache_dir', type=str, default='',
                        help='点云缓存目录（Objaverse GLB→点云转换缓存）')
    parser.add_argument('--categories', type=str, nargs='*', default=None,
                        help='指定评估的类别（仅Objaverse，如 animal building character）')
    
    args = parser.parse_args()
    
    # ========== 打印配置 ==========
    print(f"\n{'='*70}")
    print(f"GSO 跨模态检索评估 - Uni3D Multimodal")
    print(f"{'='*70}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"JSON 目录: {args.json_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"Query 模态: {args.query_modal}")
    print(f"Gallery 模态: {args.gallery_modal}")
    print(f"检查点: {args.checkpoint}")
    print(f"模型类型: {args.model_type}")
    print(f"阶段: {args.stage}")
    print(f"数据集: {args.dataset}")
    if args.dataset == 'objaverse':
        print(f"点云缓存目录: {args.pc_cache_dir if args.pc_cache_dir else 'disabled'}")
        print(f"类别: {args.categories if args.categories else 'all'}")
    print(f"{'='*70}\n")
    
    # 设置随机种子
    setup_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ========== 检查是否已完成该组合 ==========
    summary_path_check = os.path.join(
        args.output_dir,
        f'summary_uni3d_q{args.query_modal}_g{args.gallery_modal}.json'
    )
    if os.path.exists(summary_path_check):
        print(f"\n⏭️  该组合 (query={args.query_modal}, gallery={args.gallery_modal}) 的结果已存在:")
        print(f"    {summary_path_check}")
        print(f"    跳过计算。如需重新计算，请先删除该文件。\n")
        return
    
    # 查找所有JSON文件
    json_dir = Path(args.json_dir)
    json_files = []
    
    if args.dataset == 'objaverse':
        # Objaverse 数据按类别子目录组织
        subdirs = [d for d in json_dir.iterdir() if d.is_dir() and d.name != 'visualizations']
        
        if subdirs:
            print(f"发现类别子目录: {[d.name for d in sorted(subdirs)]}")
            for subdir in sorted(subdirs):
                category_name = subdir.name
                if args.categories and category_name not in args.categories:
                    continue
                category_files = sorted(subdir.glob(args.json_pattern))
                category_files = [f for f in category_files if 'conversion_summary' not in f.name]
                if category_files:
                    print(f"  {category_name}: {len(category_files)} files")
                    json_files.extend(category_files)
        else:
            json_files = sorted(json_dir.glob(args.json_pattern))
            json_files = [f for f in json_files if 'conversion_summary' not in f.name]
    else:
        json_files = sorted(json_dir.glob(args.json_pattern))
    
    print(f"\n找到 {len(json_files)} 个 JSON 文件\n")
    
    if len(json_files) == 0:
        print("❌ 未找到 JSON 文件!")
        return
    
    # 收集所有物体信息
    object_dict, json_object_mapping = collect_all_objects(json_files, args)
    
    if len(object_dict) == 0:
        print("❌ 未找到物体!")
        return
    
    # ========== Objaverse: GLB→点云转换 ==========
    pointcloud_cache_dict = {}  # {unique_key: np.ndarray} 缓存的点云数据
    
    if args.dataset == 'objaverse':
        need_point = 'v' in args.query_modal or 'v' in args.gallery_modal
        if need_point:
            print(f"\n{'='*70}")
            print(f"Objaverse: 将GLB mesh转换为点云...")
            print(f"{'='*70}\n")
            
            glb_paths = {}
            for unique_key, obj_info in object_dict.items():
                glb_paths[unique_key] = obj_info['pointcloud']  # .glb路径
            
            pc_cache = args.pc_cache_dir if args.pc_cache_dir else None
            pointcloud_cache_dict = batch_convert_glb_to_pointcloud(
                glb_paths, cache_dir=pc_cache, num_points=args.num_points
            )
            print(f"成功转换 {len(pointcloud_cache_dict)}/{len(glb_paths)} 个物体的点云\n")
    
    # ========== 尝试从 embedding 缓存加载 ==========
    print(f"\n{'='*70}")
    print(f"检查 embedding 缓存...")
    print(f"{'='*70}")
    
    # 确定需要提取的模态列表（去重）
    modals_to_extract = list(set([args.query_modal, args.gallery_modal]))
    
    # 尝试从缓存加载各模态的 embedding
    modal_embedding_cache = {}  # {modal_str: embedding_dict}
    modals_need_extract = []  # 需要重新提取的模态
    
    for modal_str in modals_to_extract:
        cached = load_embedding_cache(args.output_dir, modal_str)
        if cached is not None:
            modal_embedding_cache[modal_str] = cached
        else:
            modals_need_extract.append(modal_str)
            print(f"  ❌ 模态 '{modal_str}' 无缓存，需要提取")
    
    # ========== 按需加载模型并提取特征 ==========
    if modals_need_extract:
        print(f"\n需要提取的模态: {modals_need_extract}")
        
        # 加载 Uni3D Multimodal 模型
        uni3d_model = load_uni3d_multimodal_model(args)
        
        # 判断是否需要 CLIP 模型（用于提取图文特征）
        need_clip = any(
            'i' in m or 't' in m for m in modals_need_extract
        )
        
        clip_model = None
        clip_preprocess = None
        if need_clip:
            clip_model, clip_preprocess = load_openclip_model(args)
        
        # 逐个提取需要的模态
        for modal_str in modals_need_extract:
            print(f"\n提取模态 '{modal_str}' 的特征...")
            embedding_dict = extract_all_embeddings_uni3d(
                object_dict, args, modal_str,
                uni3d_model, clip_model, clip_preprocess,
                pointcloud_cache_dict=pointcloud_cache_dict
            )
            modal_embedding_cache[modal_str] = embedding_dict
            
            # 保存该模态的 embedding 缓存
            if len(embedding_dict) > 0:
                save_embedding_cache(args.output_dir, modal_str, embedding_dict)
        
        # 释放模型显存
        del uni3d_model
        if clip_model is not None:
            del clip_model
        torch.cuda.empty_cache()
    else:
        print(f"\n✅ 所有模态的 embedding 均从缓存加载，无需加载模型！")
    
    # 获取 query 和 gallery 的 embedding
    query_embedding_dict = modal_embedding_cache.get(args.query_modal, {})
    gallery_embedding_dict = modal_embedding_cache.get(args.gallery_modal, {})
    
    if len(query_embedding_dict) == 0 or len(gallery_embedding_dict) == 0:
        print("❌ 未提取到特征!")
        return
    
    # ========== 保存特征（按 query-gallery 组合保存，保持兼容） ==========
    embedding_save_path = os.path.join(
        args.output_dir, 
        f'embeddings_uni3d_q{args.query_modal}_g{args.gallery_modal}.npz'
    )
    print(f"保存特征到: {embedding_save_path}")
    
    query_unique_keys = list(query_embedding_dict.keys())
    query_embeddings_array = torch.stack([query_embedding_dict[key] for key in query_unique_keys]).numpy()
    
    save_data = {
        'query_modal': args.query_modal,
        'gallery_modal': args.gallery_modal,
        'model': 'uni3d_multimodal',
        'model_type': args.model_type,
        'stage': args.stage,
        'checkpoint': args.checkpoint,
        'query_unique_keys': query_unique_keys,
        'query_embeddings': query_embeddings_array,
    }
    
    if args.query_modal != args.gallery_modal:
        gallery_unique_keys = list(gallery_embedding_dict.keys())
        gallery_embeddings_array = torch.stack([gallery_embedding_dict[key] for key in gallery_unique_keys]).numpy()
        save_data['gallery_unique_keys'] = gallery_unique_keys
        save_data['gallery_embeddings'] = gallery_embeddings_array
    
    np.savez(embedding_save_path, **save_data)
    print(f"✅ 特征已保存\n")
    
    # ========== 评估 ==========
    print(f"\n{'='*70}")
    print(f"计算检索指标...")
    print(f"{'='*70}\n")
    
    all_results = []
    category_results = defaultdict(list)
    
    for json_file in tqdm(json_files, desc="评估进度"):
        try:
            results = evaluate_json_file(
                str(json_file), object_dict,
                query_embedding_dict, gallery_embedding_dict,
                args
            )
            if results is not None:
                all_results.append(results)
                cat = results['query_category']
                category_results[cat].append(results)
        except Exception as e:
            print(f"❌ Error processing {json_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ========== 保存结果 ==========
    output_path = os.path.join(
        args.output_dir,
        f'retrieval_results_uni3d_q{args.query_modal}_g{args.gallery_modal}.json'
    )
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ 评估完成!")
    print(f"{'='*70}")
    print(f"处理文件数: {len(all_results)}")
    print(f"结果保存到: {output_path}")
    
    # ========== 计算平均指标 ==========
    if all_results:
        avg_inversions = np.mean([r['inversions'] for r in all_results])
        avg_ndcg = np.mean([r['ndcg'] for r in all_results])
        avg_ndcg_10 = np.mean([r['ndcg@10'] for r in all_results])
        avg_ndcg_5 = np.mean([r['ndcg@5'] for r in all_results])
        
        kendall_taus = [r['kendall_tau'] for r in all_results if r['kendall_tau'] is not None]
        spearman_rhos = [r['spearman_rho'] for r in all_results if r['spearman_rho'] is not None]
        
        print(f"\n{'='*70}")
        print(f"平均指标 (Model: Uni3D Multimodal, Query: {args.query_modal}, Gallery: {args.gallery_modal}):")
        print(f"{'='*70}")
        print(f"Average Inversions: {avg_inversions:.2f}")
        if kendall_taus:
            print(f"Average Kendall Tau: {np.mean(kendall_taus):.4f}")
        if spearman_rhos:
            print(f"Average Spearman Rho: {np.mean(spearman_rhos):.4f}")
        print(f"Average NDCG: {avg_ndcg:.4f}")
        print(f"Average NDCG@10: {avg_ndcg_10:.4f}")
        print(f"Average NDCG@5: {avg_ndcg_5:.4f}")
        print(f"{'='*70}\n")
        
        # 按类别统计
        category_summary = {}
        if len(category_results) > 1:
            print(f"{'='*70}")
            print(f"按类别统计:")
            print(f"{'='*70}")
            
            for cat, cat_results in sorted(category_results.items()):
                cat_inversions = np.mean([r['inversions'] for r in cat_results])
                cat_ndcg = np.mean([r['ndcg'] for r in cat_results])
                cat_ndcg_10 = np.mean([r['ndcg@10'] for r in cat_results])
                cat_ndcg_5 = np.mean([r['ndcg@5'] for r in cat_results])
                cat_kendalls = [r['kendall_tau'] for r in cat_results if r['kendall_tau'] is not None]
                cat_spearmans = [r['spearman_rho'] for r in cat_results if r['spearman_rho'] is not None]
                
                print(f"\n  [{cat}] ({len(cat_results)} cases)")
                print(f"    Inversions: {cat_inversions:.2f}")
                if cat_kendalls:
                    print(f"    Kendall Tau: {np.mean(cat_kendalls):.4f}")
                if cat_spearmans:
                    print(f"    Spearman Rho: {np.mean(cat_spearmans):.4f}")
                print(f"    NDCG: {cat_ndcg:.4f}")
                print(f"    NDCG@10: {cat_ndcg_10:.4f}")
                print(f"    NDCG@5: {cat_ndcg_5:.4f}")
                
                category_summary[cat] = {
                    'num_cases': len(cat_results),
                    'avg_inversions': float(cat_inversions),
                    'avg_kendall_tau': float(np.mean(cat_kendalls)) if cat_kendalls else None,
                    'avg_spearman_rho': float(np.mean(cat_spearmans)) if cat_spearmans else None,
                    'avg_ndcg': float(cat_ndcg),
                    'avg_ndcg@10': float(cat_ndcg_10),
                    'avg_ndcg@5': float(cat_ndcg_5),
                }
            
            print(f"\n{'='*70}\n")
        
        # 保存汇总统计
        summary = {
            'model': 'Uni3D Multimodal',
            'model_type': args.model_type,
            'stage': args.stage,
            'checkpoint': args.checkpoint,
            'query_modal': args.query_modal,
            'gallery_modal': args.gallery_modal,
            'num_files': len(all_results),
            'num_unique_objects': len(object_dict),
            'avg_inversions': float(avg_inversions),
            'avg_kendall_tau': float(np.mean(kendall_taus)) if kendall_taus else None,
            'avg_spearman_rho': float(np.mean(spearman_rhos)) if spearman_rhos else None,
            'avg_ndcg': float(avg_ndcg),
            'avg_ndcg@10': float(avg_ndcg_10),
            'avg_ndcg@5': float(avg_ndcg_5),
        }
        
        if category_summary:
            summary['per_category'] = category_summary
        
        summary_path = os.path.join(
            args.output_dir,
            f'summary_uni3d_q{args.query_modal}_g{args.gallery_modal}.json'
        )
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"汇总统计保存到: {summary_path}\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
