"""
多模态 Uni3D 模型零样本测试脚本

支持 7 种模态组合测试:
- i:   仅图像
- v:   仅点云
- t:   仅文本（标签嵌入对齐）
- iv:  图像 + 点云
- it:  图像 + 文本
- vt:  点云 + 文本
- ivt: 图像 + 点云 + 文本

支持的数据集:
- objaverse_lvis: Objaverse-LVIS 数据集 (1156 类)
- modelnet40: ModelNet40 数据集 (40 类)
- scanobjnn: ScanObjectNN 数据集 (15 类)

用法示例:
    # 1. Objaverse-LVIS 测试（默认数据集）
    python test_multimodal.py --checkpoint /path/to/checkpoint.pth --modal v
    
    # 2. ModelNet40 测试
    python test_multimodal.py --checkpoint /path/to/checkpoint.pth --dataset modelnet40 \
        --data_path /path/to/modelnet40_openshape/
    
    # 3. ScanObjectNN 测试
    python test_multimodal.py --checkpoint /path/to/checkpoint.pth --dataset scanobjnn \
        --data_path /path/to/scanobjnn/
    
    # 4. 多模态融合测试
    python test_multimodal.py --checkpoint /path/to/checkpoint.pth --modal ivt
    
    # 5. 测试所有模态组合
    python test_multimodal.py --checkpoint /path/to/checkpoint.pth --modal all --dataset modelnet40
    
    # 6. 指定 Objaverse-LVIS 数据路径
    python test_multimodal.py --checkpoint /path/to/checkpoint.pth --modal iv \
        --pc_path /path/to/lvis_test.txt --pc_path_root /path/to/pointclouds/
"""

import os
import sys
import argparse
import time
import json
import logging
import collections
from collections import OrderedDict
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

import open_clip
from easydict import EasyDict

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.datasets import Objaverse_lvis_openshape, ModelNet40_openshape, ScanObjNN_openshape, customized_collate_fn
from utils.tokenizer import SimpleTokenizer


# ============ 配置日志 ============

def setup_logging(log_level=logging.INFO):
    """设置日志"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()],
        force=True
    )
    return logging.getLogger(__name__)


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

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# ============ 辅助函数 ============

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


def get_model(model):
    """获取底层模型（去除 DDP 包装）"""
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module
    return model


# ============ 特征保存和加载 ============

def get_feature_cache_path(checkpoint_path: str, dataset: str, modal: str) -> str:
    """
    获取特征缓存文件路径
    
    Args:
        checkpoint_path: 模型检查点路径
        dataset: 数据集名称
        modal: 模态组合
    
    Returns:
        str: 特征缓存文件路径
    """
    output_dir = os.path.dirname(checkpoint_path) or '.'
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    cache_file = os.path.join(output_dir, f'features_cache_{checkpoint_name}_{dataset}_{modal}.pt')
    return cache_file


def save_features(features_dict: Dict, cache_path: str):
    """
    保存提取的特征到文件
    
    Args:
        features_dict: 特征字典，包含:
            - 'fused_features': 融合特征列表
            - 'targets': 目标标签列表
            - 'target_names': 目标名称列表
            - 'text_features': 类别文本特征
        cache_path: 缓存文件路径
    """
    logging.info(f"=> Saving extracted features to: {cache_path}")
    torch.save(features_dict, cache_path)
    logging.info(f"=> Features saved successfully!")


def load_features(cache_path: str) -> Optional[Dict]:
    """
    从文件加载已保存的特征
    
    Args:
        cache_path: 缓存文件路径
    
    Returns:
        Optional[Dict]: 特征字典，如果文件不存在则返回 None
    """
    if os.path.exists(cache_path):
        logging.info(f"=> Loading cached features from: {cache_path}")
        features_dict = torch.load(cache_path, map_location='cpu')
        logging.info(f"=> Loaded {len(features_dict.get('fused_features', []))} samples from cache")
        return features_dict
    return None


# ============ 数据集配置构建 ============

# 数据集默认配置
DATASET_CONFIGS = {
    'objaverse_lvis': {
        'label_key': 'objaverse_lvis_openshape',
        'prompt_template': 'modelnet40_64',
        'default_pc_path': './data/test_datasets/lvis/lvis_test.txt',
        'default_pc_path_root': '/cfs_160T/serenasnliu/3d-datasets/openshape/Objaverse/000-xxx.tar/objaverse-processed/merged_for_training_final/Objaverse/',
    },
    'modelnet40': {
        'label_key': 'modelnet40_openshape',
        'prompt_template': 'modelnet40_64',
        'default_data_path': './data/test_datasets/modelnet40/',
    },
    'scanobjnn': {
        'label_key': 'scanobjnn_openshape',
        'prompt_template': 'modelnet40_64',
        'default_data_path': './data/test_datasets/scanobjectnn/',
    },
}

def build_test_dataset_config(args) -> EasyDict:
    """
    构建测试数据集配置
    
    Args:
        args: 命令行参数
    
    Returns:
        EasyDict: 数据集配置
    """
    config = EasyDict()
    
    config.subset = 'test'
    config.npoints = args.npoints
    config.tokenizer = None  # 数据集不需要 tokenizer
    config.train_transform = None
    config.openshape_setting = args.openshape_setting
    config.use_height = args.use_height
    config.pretrain_dataset_prompt = args.prompt_template  # 用于 templates.json
    
    # 数据路径
    config.PC_PATH = args.pc_path
    config.PC_PATH_ROOT = args.pc_path_root
    
    return config


def build_modelnet40_config(args) -> EasyDict:
    """
    构建 ModelNet40 测试数据集配置
    
    Args:
        args: 命令行参数
    
    Returns:
        EasyDict: 数据集配置
    """
    config = EasyDict()
    
    config.subset = 'test'
    config.npoints = args.npoints
    config.openshape_setting = args.openshape_setting
    
    # 数据路径
    if args.data_path:
        config.DATA_PATH = args.data_path
    else:
        config.DATA_PATH = DATASET_CONFIGS['modelnet40']['default_data_path']
    
    return config


def build_scanobjnn_config(args) -> EasyDict:
    """
    构建 ScanObjectNN 测试数据集配置
    
    Args:
        args: 命令行参数
    
    Returns:
        EasyDict: 数据集配置
    """
    config = EasyDict()
    
    config.subset = 'test'
    config.npoints = args.npoints
    config.openshape_setting = args.openshape_setting
    
    # 数据路径
    if args.data_path:
        config.DATA_PATH = args.data_path
    else:
        config.DATA_PATH = DATASET_CONFIGS['scanobjnn']['default_data_path']
    
    return config


def load_test_dataset(args) -> DataLoader:
    """
    加载测试数据集
    
    支持的数据集:
    - objaverse_lvis: Objaverse-LVIS 数据集 (OpenShape 格式)
    - modelnet40: ModelNet40 数据集 (OpenShape 格式)
    - scanobjnn: ScanObjectNN 数据集 (OpenShape 格式)
    
    Args:
        args: 命令行参数
    
    Returns:
        DataLoader: 测试数据加载器
    """
    logging.info(f"=> Loading test dataset: {args.dataset}")
    
    if args.dataset == 'objaverse_lvis':
        logging.info(f"   Data list file: {args.pc_path}")
        logging.info(f"   Point cloud root: {args.pc_path_root}")
        config = build_test_dataset_config(args)
        dataset = Objaverse_lvis_openshape(config)

    
    elif args.dataset == 'modelnet40':
        config = build_modelnet40_config(args)
        logging.info(f"   Data path: {config.DATA_PATH}")
        dataset = ModelNet40_openshape(config)
    
    elif args.dataset == 'scanobjnn':
        config = build_scanobjnn_config(args)
        logging.info(f"   Data path: {config.DATA_PATH}")
        dataset = ScanObjNN_openshape(config)
    
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}. "
                        f"Supported datasets: {list(DATASET_CONFIGS.keys())}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=customized_collate_fn
    )
    
    logging.info(f"=> Loaded {len(dataset)} test samples")
    return dataloader


# ============ 模型加载 ============

def load_multimodal_model(args, device):
    """
    加载多模态 Uni3D 模型
    
    Args:
        args: 命令行参数
        device: 设备
    
    Returns:
        model: 多模态模型
    """
    logging.info(f"=> Loading multimodal model from: {args.checkpoint}")
    
    # 根据模型类型选择创建函数
    # 测试时设置 load_pretrained=False，因为会从 checkpoint 加载完整模型权重
    # 不需要再次加载原始预训练点云编码器
    if args.model_type == 'two_stage':
        from models.uni3d_multimodal_two_stage import create_uni3d_multimodal_two_stage
        model = create_uni3d_multimodal_two_stage(args, stage=args.stage, load_pretrained=False)
    else:
        from models.uni3d_multimodal import create_uni3d_multimodal
        model = create_uni3d_multimodal(args, load_pretrained=False)
    
    # 加载检查点
    if os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        
        # 打印 checkpoint 结构，帮助调试
        if isinstance(checkpoint, dict):
            logging.info(f"=> Checkpoint keys: {list(checkpoint.keys())}")
        else:
            logging.info(f"=> Checkpoint is not a dict, type: {type(checkpoint)}")
        
        # 尝试不同的 key 来获取模型权重
        # DeepSpeed 格式可能有多种变体
        state_dict = None
        state_dict_key_used = None
        
        if isinstance(checkpoint, dict):
            # 常见的 checkpoint 格式
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                state_dict_key_used = 'model_state_dict'
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                state_dict_key_used = 'state_dict'
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                state_dict_key_used = 'model'
            elif 'module' in checkpoint:
                state_dict = checkpoint['module']
                state_dict_key_used = 'module'
            # DeepSpeed ZeRO-3 格式可能有不同的结构
            elif 'dp_world_size' in checkpoint and len(checkpoint) > 0:
                # 这可能是 DeepSpeed 的 checkpoint，尝试找到模型权重
                for key in checkpoint.keys():
                    if isinstance(checkpoint[key], dict) and any(k.startswith(('point_encoder', 'fusion', 'embed')) for k in checkpoint[key].keys()):
                        state_dict = checkpoint[key]
                        state_dict_key_used = key
                        break
                if state_dict is None:
                    # 尝试把整个 checkpoint 作为 state_dict
                    state_dict = checkpoint
                    state_dict_key_used = 'checkpoint_itself (DeepSpeed format)'
            else:
                # 假设 checkpoint 本身就是 state_dict
                state_dict = checkpoint
                state_dict_key_used = 'checkpoint_itself'
        else:
            state_dict = checkpoint
            state_dict_key_used = 'checkpoint_itself'
        
        logging.info(f"=> Using state_dict from key: {state_dict_key_used}")
        logging.info(f"=> State dict has {len(state_dict)} keys")
        
        # 打印前几个 key，帮助调试
        sample_keys = list(state_dict.keys())[:10]
        logging.info(f"=> Sample keys: {sample_keys}")
        
        # 清理 state_dict 中的 module. 前缀
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
        logging.info(f"=> Loaded checkpoint (missing: {len(missing)}, unexpected: {len(unexpected)})")
        if missing:
            logging.warning(f"Missing keys ({len(missing)}): {missing[:10]}...")
        if unexpected:
            logging.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}...")
        
        # 关键检查：如果 missing 太多，说明权重可能没有正确加载
        total_params = sum(1 for _ in model.state_dict().keys())
        if len(missing) > total_params * 0.5:
            logging.error(f"⚠️ WARNING: More than 50% of model parameters are missing! Check checkpoint format.")
            logging.error(f"   Total params in model: {total_params}")
            logging.error(f"   Missing params: {len(missing)}")
        
        # 打印多个关键权重的统计信息，验证权重是否正确加载
        logging.info(f"=> Verifying loaded weights:")
        with torch.no_grad():
            verified_count = 0
            for name, param in model.named_parameters():
                # 验证几个关键层的权重
                if any(key in name for key in ['point_encoder.encoder.first_conv', 'fusion_blocks.0.mlp_1', 'fused_to_clip_proj.0.weight']):
                    logging.info(f"   {name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}, shape={tuple(param.shape)}")
                    verified_count += 1
                    if verified_count >= 5:
                        break
            
            # 额外检查：打印 logit_scale 的值
            if hasattr(model, 'logit_scale'):
                logging.info(f"   logit_scale: {model.logit_scale.exp().item():.4f}")
            
            # 检查是否有 NaN 或 Inf
            has_nan = False
            has_inf = False
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    logging.error(f"   ⚠️ NaN detected in {name}")
                    has_nan = True
                if torch.isinf(param).any():
                    logging.error(f"   ⚠️ Inf detected in {name}")
                    has_inf = True
            
            if not has_nan and not has_inf:
                logging.info(f"   ✓ No NaN or Inf detected in model weights")
    else:
        logging.error(f"=> ⚠️ No checkpoint found at '{args.checkpoint}'")
    
    model.to(device)
    model.eval()
    
    return model


def load_clip_model(args, device):
    """
    加载 CLIP 模型（用于编码类别文本）
    
    Args:
        args: 命令行参数
        device: 设备
    
    Returns:
        clip_model: CLIP 模型
        tokenizer: CLIP 分词器
    """
    logging.info(f"=> Loading CLIP model: {args.clip_model}")
    
    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_name=args.clip_model, 
        pretrained=args.pretrained
    )
    clip_model.to(device)
    clip_model.eval()
    
    # 冻结参数
    for param in clip_model.parameters():
        param.requires_grad = False
    
    # 获取 tokenizer
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    
    return clip_model, tokenizer


# ============ 核心测试函数 ============

def test_zeroshot_multimodal(
    test_loader: DataLoader,
    model: nn.Module,
    clip_model: nn.Module,
    tokenizer,
    args,
    modal: str = 'v'
) -> Dict[str, float]:
    """
    多模态零样本分类测试
    
    根据指定的模态组合进行测试:
    - i:   仅使用图像模态
    - v:   仅使用点云模态
    - t:   仅使用文本模态（用于验证文本对齐）
    - iv:  图像 + 点云
    - it:  图像 + 文本
    - vt:  点云 + 文本
    - ivt: 图像 + 点云 + 文本
    
    Args:
        test_loader: 测试数据加载器
        model: 多模态 Uni3D 模型
        clip_model: CLIP 模型（用于编码类别文本）
        tokenizer: CLIP 分词器
        args: 命令行参数
        modal: 模态组合，可选 'i', 'v', 't', 'iv', 'it', 'vt', 'ivt'
    
    Returns:
        dict: 包含 acc1, acc3, acc5 的字典
    """
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
    
    device = args.device

    # 加载模板和标签
    logging.info(f"=> Loading templates from: ./data/templates.json (key: {args.prompt_template})")
    with open(os.path.join("./data", 'templates.json')) as f:
        templates = json.load(f)[args.prompt_template]

    logging.info(f"=> Loading labels from: ./data/labels.json (key: {args.label_key})")
    with open(os.path.join("./data", 'labels.json')) as f:
        labels = json.load(f)[args.label_key]

    logging.info(f"=> Number of classes: {len(labels)}")
    logging.info(f"=> Testing with modal: {modal}")

    # 检查是否有缓存的特征
    cache_path = get_feature_cache_path(args.checkpoint, args.dataset, modal)
    cached_features = load_features(cache_path) if args.use_cache else None

    with torch.no_grad():
        # Step 1: 编码所有类别的文本特征
        if cached_features is not None and 'text_features' in cached_features:
            text_features = cached_features['text_features'].to(device)
            logging.info(f"=> Using cached text features, shape: {text_features.shape}")
        else:
            logging.info('=> Encoding class text features...')               
            text_features = []
            for l in labels:
                texts = [t.format(l) for t in templates]
                texts = tokenizer(texts).to(device=device, non_blocking=True)
                if len(texts.shape) < 2:
                    texts = texts[None, ...]
                class_embeddings = clip_model.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)
            logging.info(f"=> Text features shape: {text_features.shape}")

        # Step 2: 提取或加载样本特征
        if cached_features is not None and 'fused_features' in cached_features:
            # 使用缓存的特征
            all_fused_features = cached_features['fused_features']
            all_targets = cached_features['targets']
            all_target_names = cached_features['target_names']
            logging.info(f"=> Using cached sample features: {len(all_fused_features)} samples")
        else:
            # 提取特征
            logging.info('=> Extracting sample features...')
            all_fused_features = []
            all_targets = []
            all_target_names = []
            
            # 添加调试：打印第一个 batch 的信息
            first_batch_logged = False
            
            for i, batch in enumerate(tqdm(test_loader, desc=f"Extracting features ({modal})")):
                if batch is None:
                    continue
                
                # 根据数据集返回格式解析 batch
                # ModelNet40/ScanObjNN 返回 4 个元素: (pc, target, target_name, rgb)
                # Objaverse_lvis 返回 8 个元素: (pc, target, target_name, rgb, model_id, use_image, text, image)
                if len(batch) == 4:
                    pc, target, target_name, rgb = batch
                    text = None
                    image = None
                elif len(batch) == 8:
                    pc, target, target_name, rgb, _, _, text, image = batch
                else:
                    logging.warning(f"Unexpected batch length: {len(batch)}, skipping...")
                    continue
                
                pc = pc.to(device=device, non_blocking=True)
                rgb = rgb.to(device=device, non_blocking=True)
                target = target.to(device=device, non_blocking=True)
                
                # 拼接点云特征 (xyz + rgb)
                feature = torch.cat((pc, rgb), dim=-1)
                
                # 调试：打印第一个 batch 的信息
                if not first_batch_logged:
                    logging.info(f"=> First batch info:")
                    logging.info(f"   pc shape: {pc.shape}, dtype: {pc.dtype}")
                    logging.info(f"   rgb shape: {rgb.shape}, dtype: {rgb.dtype}")
                    logging.info(f"   feature shape: {feature.shape}")
                    logging.info(f"   target: {target}")
                    logging.info(f"   target_name: {target_name[:3]}...")
                    if text is not None:
                        logging.info(f"   text shape: {text.shape}, dtype: {text.dtype}")
                        logging.info(f"   text mean: {text.mean().item():.6f}, std: {text.std().item():.6f}")
                    else:
                        logging.info(f"   text: None")
                    if image is not None:
                        logging.info(f"   image shape: {image.shape}, dtype: {image.dtype}")
                        logging.info(f"   image mean: {image.mean().item():.6f}, std: {image.std().item():.6f}")
                    else:
                        logging.info(f"   image: None")
                    first_batch_logged = True

                # 根据模态进行特征编码
                uni3d_model = get_model(model)
                
                # ============ 关键修复：正确处理图文特征 ============
                # 注意：数据集返回的 text/image 可能是预提取的特征或随机生成的特征
                # 检查它们是否是真实的特征还是随机的 placeholder
                
                # 将 text 和 image 移动到设备
                # 注意：这里每个 batch 应该生成不同的随机特征，而非整个数据集共用一个
                if text is not None:
                    text = text.to(device=device, non_blocking=True)
                    # 检查是否是随机生成的特征（可以通过检查数值范围或标志）
                else:
                    print('no text!!!')
                    # 每个样本生成不同的随机特征
                    text = torch.randn(feature.shape[0], args.embed_dim, device=device)
                    text = F.normalize(text, p=2, dim=-1)  # 归一化
                
                if image is not None:
                    image = image.to(device=device, non_blocking=True)
                else:
                    print('no image!!!')

                    # 每个样本生成不同的随机特征
                    image = torch.randn(feature.shape[0], args.embed_dim, device=device)
                    image = F.normalize(image, p=2, dim=-1)  # 归一化
                
                if modal == 'v':
                    # 仅点云模态（但仍传入图文特征供后续使用）
                    fused_features, _, _ = uni3d_model.encode_multimodal(
                        point=feature, image_embed=image, text_embed=text, modal='v'
                    )
                elif modal == 'iv':
                    fused_features, _, _ = uni3d_model.encode_multimodal(
                        point=feature, image_embed=image, text_embed=text, modal='iv'
                    )
                elif modal == 'vt':
                    fused_features, _, _ = uni3d_model.encode_multimodal(
                        point=feature, image_embed=image, text_embed=text, modal='vt'
                    )
                elif modal == 'ivt':
                    fused_features, _, _ = uni3d_model.encode_multimodal(
                        point=feature, image_embed=image, text_embed=text, modal='ivt'
                    )
                elif modal == 'it':
                    fused_features, _, _ = uni3d_model.encode_multimodal(
                        image_embed=image, text_embed=text, modal='it'
                    )
                elif modal == 'i':
                    # 仅图像（但仍传入文本特征供后续使用）
                    fused_features, _, _ = uni3d_model.encode_multimodal(
                        image_embed=image, text_embed=text, modal='i'
                    )
                elif modal == 't':
                    # 仅文本（但仍传入图像特征供后续使用）
                    fused_features, _, _ = uni3d_model.encode_multimodal(
                        image_embed=image, text_embed=text, modal='t'
                    )
                else:
                    raise ValueError(f"Unsupported modal: {modal}")
                
                # 归一化特征
                fused_features = fused_features / fused_features.norm(dim=-1, keepdim=True)
                
                # 调试：打印第一个 batch 的输出特征统计
                if i == 0:
                    logging.info(f"=> First batch output:")
                    logging.info(f"   fused_features shape: {fused_features.shape}")
                    logging.info(f"   fused_features mean: {fused_features.mean().item():.6f}, std: {fused_features.std().item():.6f}")
                    logging.info(f"   fused_features[0][:5]: {fused_features[0][:5].tolist()}")
                
                # 保存到列表（移到CPU以节省GPU内存）
                all_fused_features.append(fused_features.cpu())
                all_targets.append(target.cpu())
                all_target_names.extend(target_name)
            
            # 保存提取的特征
            if args.save_cache:
                features_to_save = {
                    'fused_features': all_fused_features,
                    'targets': all_targets,
                    'target_names': all_target_names,
                    'text_features': text_features.cpu(),
                    'modal': modal,
                    'dataset': args.dataset,
                }
                save_features(features_to_save, cache_path)

        # Step 3: 计算准确率
        logging.info('=> Computing accuracy...')
        end = time.time()
        per_class_stats = collections.defaultdict(int)
        per_class_correct_top1 = collections.defaultdict(int)
        per_class_correct_top3 = collections.defaultdict(int)
        per_class_correct_top5 = collections.defaultdict(int)
        
        # 记录当前处理的样本名称索引
        name_idx = 0
        
        for i, (fused_features, targets) in enumerate(zip(all_fused_features, all_targets)):
            # 获取当前 batch 的 target_names
            batch_size = targets.size(0)
            target_names_batch = all_target_names[name_idx:name_idx + batch_size]
            name_idx += batch_size
            
            for name in target_names_batch:
                per_class_stats[name] += 1
            
            # 移动到设备
            fused_features = fused_features.to(device)
            targets = targets.to(device)
            
            # 计算余弦相似度作为 logits
            logits = fused_features.float() @ text_features.float().t()

            # 计算准确率
            (acc1, acc3, acc5), correct = accuracy(logits, targets, topk=(1, 3, 5))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)
            top5.update(acc5.item(), batch_size)

            # 记录时间
            batch_time.update(time.time() - end)
            end = time.time()

            # 统计每个类别的准确率
            # 修复0维张量索引问题：确保张量至少是1维的
            top1_accurate = correct[:1].reshape(-1)
            top3_accurate = correct[:3].float().sum(0).reshape(-1)
            top5_accurate = correct[:5].float().sum(0).reshape(-1)
            
            for idx, name in enumerate(target_names_batch):
                if top1_accurate[idx].item():
                    per_class_correct_top1[name] += 1
                if top3_accurate[idx].item() > 0:
                    per_class_correct_top3[name] += 1
                if top5_accurate[idx].item() > 0:
                    per_class_correct_top5[name] += 1

            if i % args.print_freq == 0:
                progress.display(i)

        # Step 4: 计算每个类别的准确率
        top1_accuracy_per_class = {}
        top3_accuracy_per_class = {}
        top5_accuracy_per_class = {}
        for name in per_class_stats.keys():
            top1_accuracy_per_class[name] = per_class_correct_top1[name] / per_class_stats[name]
            top3_accuracy_per_class[name] = per_class_correct_top3[name] / per_class_stats[name]
            top5_accuracy_per_class[name] = per_class_correct_top5[name] / per_class_stats[name]

        top1_accuracy_per_class = collections.OrderedDict(sorted(top1_accuracy_per_class.items()))
    
    logging.info(f"\n{'='*70}")
    logging.info(f"Modal: {modal} | Acc@1 {top1.avg:.3f}% | Acc@3 {top3.avg:.3f}% | Acc@5 {top5.avg:.3f}%")
    logging.info(f"{'='*70}")
    
    return {
        'acc1': top1.avg, 
        'acc3': top3.avg, 
        'acc5': top5.avg,
        'per_class_acc1': top1_accuracy_per_class
    }


def test_zeroshot_pointcloud_only(
    test_loader: DataLoader,
    model: nn.Module,
    clip_model: nn.Module,
    tokenizer,
    args
) -> Dict[str, float]:
    """
    纯点云零样本分类测试（与原始 Uni3D 一致）
    
    Args:
        test_loader: 测试数据加载器
        model: 多模态 Uni3D 模型
        clip_model: CLIP 模型（用于编码类别文本）
        tokenizer: CLIP 分词器
        args: 命令行参数
    
    Returns:
        dict: 包含 acc1, acc3, acc5 的字典
    """
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f') 
    top3 = AverageMeter('Acc@3', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, top1, top3, top5],
        prefix='Test (PC only): ')

    # 切换到评估模式
    model.eval()
    
    device = args.device

    # 加载模板和标签
    with open(os.path.join("./data", 'templates.json')) as f:
        templates = json.load(f)[args.prompt_template]

    with open(os.path.join("./data", 'labels.json')) as f:
        labels = json.load(f)[args.label_key]

    logging.info(f"=> Number of classes: {len(labels)}")
    logging.info(f"=> Testing with point cloud only (encode_pc)")

    with torch.no_grad():
        # 编码所有类别的文本特征
        logging.info('=> Encoding class text features...')               
        text_features = []
        for l in labels:
            texts = [t.format(l) for t in templates]
            texts = tokenizer(texts).to(device=device, non_blocking=True)
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

        for i, batch in enumerate(tqdm(test_loader, desc="Testing (PC only)")):
            if batch is None:
                continue
            
            # 根据数据集返回格式解析 batch
            # ModelNet40/ScanObjNN 返回 4 个元素: (pc, target, target_name, rgb)
            # Objaverse_lvis 返回 8 个元素: (pc, target, target_name, rgb, model_id, use_image, text, image)
            if len(batch) == 4:
                pc, target, target_name, rgb = batch
            elif len(batch) == 8:
                pc, target, target_name, rgb, _, _, _, _ = batch
            else:
                logging.warning(f"Unexpected batch length: {len(batch)}, skipping...")
                continue
            
            for name in target_name:
                per_class_stats[name] += 1

            pc = pc.to(device=device, non_blocking=True)
            rgb = rgb.to(device=device, non_blocking=True)
            target = target.to(device=device, non_blocking=True)
            
            feature = torch.cat((pc, rgb), dim=-1)

            # 使用纯点云编码（与原始 test_zeroshot_3d_core 一致）
            uni3d_model = get_model(model)
            
            # 尝试使用 encode_multimodal 或 encode_pc
            try:
                pc_features, _, _ = uni3d_model.encode_multimodal(point=feature)
            except:
                pc_features = uni3d_model.encode_pc(feature)
            
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)

            logits = pc_features.float() @ text_features.float().t()

            (acc1, acc3, acc5), correct = accuracy(logits, target, topk=(1, 3, 5))
            top1.update(acc1.item(), pc.size(0))
            top3.update(acc3.item(), pc.size(0))
            top5.update(acc5.item(), pc.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            # 修复0维张量索引问题
            top1_accurate = correct[:1].reshape(-1)
            for idx, name in enumerate(target_name):
                if top1_accurate[idx].item():
                    per_class_correct_top1[name] += 1

            if i % args.print_freq == 0:
                progress.display(i)

    logging.info(f"\n{'='*70}")
    logging.info(f"Point Cloud Only | Acc@1 {top1.avg:.3f}% | Acc@3 {top3.avg:.3f}% | Acc@5 {top5.avg:.3f}%")
    logging.info(f"{'='*70}")
    
    return {'acc1': top1.avg, 'acc3': top3.avg, 'acc5': top5.avg}


# ============ 主函数 ============

def get_args():
    """获取命令行参数"""
    parser = argparse.ArgumentParser(description='Multimodal Uni3D Zero-shot Testing')
    
    # 模型参数
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--model_type', type=str, default='two_stage', 
                        choices=['two_stage', 'multimodal'], help='模型类型')
    parser.add_argument('--stage', type=int, default=2, help='Two-stage 模型的阶段 (1 或 2)')
    parser.add_argument('--pc_model', type=str, default='eva_giant_patch14_560.m30m_ft_in22k_in1k', 
                        help='点云 Transformer 模型')
    parser.add_argument('--embed_dim', type=int, default=1280, help='CLIP 嵌入维度')
    parser.add_argument('--pc_feat_dim', type=int, default=1408, help='点云特征维度')
    parser.add_argument('--pc_encoder_dim', type=int, default=512, help='点云编码器维度')
    parser.add_argument('--group_size', type=int, default=64, help='点云分组大小')
    parser.add_argument('--num_group', type=int, default=512, help='点云分组数量')
    parser.add_argument('--drop_path_rate', type=float, default=0.2, help='DropPath 率')
    parser.add_argument('--patch_dropout', type=float, default=0.5, help='Patch Dropout 率')
    parser.add_argument('--use_embed', action='store_true', default=True, help='使用预计算的embedding')
    parser.add_argument('--use_fusion_blocks', action='store_true', default=True, help='使用融合块')
    # parser.add_argument('--pretrained_pc', type=int, default=2, help='融合块数量')
    
    # CLIP 参数
    parser.add_argument('--clip_model', type=str, default='ViT-bigG-14', help='CLIP 模型')
    parser.add_argument('--pretrained', type=str, default='/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/clip_model/open_clip_pytorch_model.bin', help='CLIP 预训练权重')
    
    # 数据集选择
    parser.add_argument('--dataset', type=str, default='objaverse_lvis',
                        choices=['objaverse_lvis', 'modelnet40', 'scanobjnn'],
                        help='测试数据集类型')
    
    # Objaverse-LVIS 数据路径参数
    parser.add_argument('--pc_path', type=str, 
                        default='./data/test_datasets/lvis/lvis_test.txt',
                        help='[Objaverse-LVIS] 测试数据列表文件路径')
    parser.add_argument('--pc_path_root', type=str, 
                        default='/cfs_160T/serenasnliu/3d-datasets/openshape/Objaverse/000-xxx.tar/objaverse-processed/merged_for_training_final/Objaverse/',
                        help='[Objaverse-LVIS] 点云数据根目录')
    
    # ModelNet40 / ScanObjectNN 数据路径参数
    parser.add_argument('--data_path', type=str, default=None,
                        help='[ModelNet40/ScanObjectNN] 数据路径。如不指定则使用默认路径')
    
    # 通用数据参数
    parser.add_argument('--npoints', type=int, default=10000, help='点云采样点数')
    parser.add_argument('--use_height', action='store_true', default=False, help='是否使用高度特征')
    parser.add_argument('--openshape_setting', action='store_true', default=True, help='使用 OpenShape 设置')
    
    # 标签和模板 (可根据数据集自动设置)
    parser.add_argument('--label_key', type=str, default=None,
                        help='labels.json 中的 key。如不指定则根据数据集自动选择')
    parser.add_argument('--prompt_template', type=str, default=None,
                        help='templates.json 中的 key。如不指定则根据数据集自动选择')
    
    # 测试参数
    parser.add_argument('--modal', type=str, default='v', 
                        choices=['i', 'v', 't', 'iv', 'it', 'vt', 'ivt', 'all'],
                        help='测试模态组合。选择 "all" 将依次测试所有 7 种组合')
    parser.add_argument('--batch_size', type=int, default=2, help='批大小')
    parser.add_argument('--workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--print_freq', type=int, default=100, help='打印频率')
    
    # 特征缓存参数
    parser.add_argument('--save_cache', action='store_true', default=True,
                        help='保存提取的特征到缓存文件')
    parser.add_argument('--use_cache', action='store_true', default=False,
                        help='使用已缓存的特征（如果存在）')
    parser.add_argument('--no_save_cache', dest='save_cache', action='store_false',
                        help='不保存特征缓存')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda:7', help='设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 根据数据集自动设置 label_key 和 prompt_template
    if args.label_key is None:
        args.label_key = DATASET_CONFIGS[args.dataset]['label_key']
    if args.prompt_template is None:
        args.prompt_template = DATASET_CONFIGS[args.dataset]['prompt_template']
    
    return args


def main():
    """主函数"""
    args = get_args()
    
    # 设置日志
    logger = setup_logging()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    if torch.cuda.is_available() and 'cuda' in args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')
        logging.warning("CUDA not available, using CPU")
    args.device = device
    
    logging.info(f"\n{'='*70}")
    logging.info(f"Multimodal Uni3D Zero-shot Testing")
    logging.info(f"{'='*70}")
    logging.info(f"Device: {device}")
    logging.info(f"Checkpoint: {args.checkpoint}")
    logging.info(f"Modal: {args.modal}")
    logging.info(f"Data path: {args.pc_path}")
    logging.info(f"Save cache: {args.save_cache}")
    logging.info(f"Use cache: {args.use_cache}")
    logging.info(f"{'='*70}\n")
    
    # 加载模型
    logging.info("=> Loading models...")
    model = load_multimodal_model(args, device)
    clip_model, tokenizer = load_clip_model(args, device)
    
    # ============ 快速验证模型 ============
    logging.info("=> Quick model verification...")
    with torch.no_grad():
        try:
            # 创建一个随机输入进行前向传播测试
            dummy_pc = torch.randn(1, args.npoints, 6, device=device)
            dummy_image_embed = torch.randn(1, args.embed_dim, device=device)
            dummy_text_embed = torch.randn(1, args.embed_dim, device=device)
            
            # 测试 encode_multimodal
            uni3d_model = get_model(model)
            fused_feat, _, _ = uni3d_model.encode_multimodal(
                point=dummy_pc, image_embed=dummy_image_embed, text_embed=dummy_text_embed, modal='v'
            )
            
            logging.info(f"   ✓ Forward pass successful!")
            logging.info(f"   Output shape: {fused_feat.shape}")
            logging.info(f"   Output mean: {fused_feat.mean().item():.6f}, std: {fused_feat.std().item():.6f}")
            logging.info(f"   Output[:5]: {fused_feat[0][:5].tolist()}")
            
            # 再次用不同的随机输入测试，检查输出是否变化
            dummy_pc2 = torch.randn(1, args.npoints, 6, device=device)
            fused_feat2, _, _ = uni3d_model.encode_multimodal(
                point=dummy_pc2, image_embed=dummy_image_embed, text_embed=dummy_text_embed, modal='v'
            )
            
            # 计算两次输出的相似度
            sim = F.cosine_similarity(fused_feat, fused_feat2, dim=-1).item()
            logging.info(f"   Cosine similarity between two random inputs: {sim:.6f}")
            
            if sim > 0.99:
                logging.warning(f"   ⚠️ WARNING: Two different inputs produced nearly identical outputs!")
                logging.warning(f"      This might indicate a problem with the model or weights.")
        except Exception as e:
            logging.error(f"   ✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 加载测试数据集
    logging.info("=> Loading test dataset...")
    test_loader = load_test_dataset(args)
    
    # 运行测试
    logging.info("=> Starting zero-shot testing...")
    
    results = {}
    
    if args.modal == 'all':
        # 测试所有模态组合
        all_modals = ['v', 'iv', 'vt', 'ivt', 'i', 't', 'it']
        for modal in all_modals:
            logging.info(f"\n{'#'*70}")
            logging.info(f"Testing modal: {modal}")
            logging.info(f"{'#'*70}\n")
            
            try:
                with amp.autocast():
                    result = test_zeroshot_multimodal(
                        test_loader, model, clip_model, tokenizer, args, modal=modal
                    )
                results[modal] = result
            except Exception as e:
                logging.error(f"Error testing modal {modal}: {e}")
                results[modal] = {'acc1': 0.0, 'acc3': 0.0, 'acc5': 0.0, 'error': str(e)}
        
        # 打印汇总结果
        logging.info(f"\n{'='*70}")
        logging.info("SUMMARY - All Modal Combinations")
        logging.info(f"{'='*70}")
        for modal, result in results.items():
            if 'error' not in result:
                logging.info(f"  {modal:5s}: Acc@1 {result['acc1']:.2f}% | Acc@3 {result['acc3']:.2f}% | Acc@5 {result['acc5']:.2f}%")
            else:
                logging.info(f"  {modal:5s}: ERROR - {result['error']}")
        logging.info(f"{'='*70}\n")
    
    else:
        # 测试单个模态组合
        with amp.autocast():
            if args.modal == 'v':
                # 也运行纯点云测试作为对比
                result_pc_only = test_zeroshot_pointcloud_only(
                    test_loader, model, clip_model, tokenizer, args
                )
                results['pc_only'] = result_pc_only
            
            result = test_zeroshot_multimodal(
                test_loader, model, clip_model, tokenizer, args, modal=args.modal
            )
            results[args.modal] = result
    
    # 保存结果
    output_dir = os.path.dirname(args.checkpoint) or '.'
    result_file = os.path.join(output_dir, f'test_results_{args.modal}.json')
    
    # 转换结果为可序列化格式
    serializable_results = {}
    for modal, result in results.items():
        serializable_results[modal] = {
            'acc1': float(result['acc1']),
            'acc3': float(result['acc3']),
            'acc5': float(result['acc5'])
        }
    
    with open(result_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    logging.info(f"=> Results saved to: {result_file}")
    
    return results


if __name__ == '__main__':
    main()
