"""
Uni3D 最简推理脚本
用法: python simple_inference.py
"""

import torch
import numpy as np
import open_clip
import timm
from models.point_encoder import PointcloudEncoder
from models.uni3d import Uni3D

# ============== 配置参数 ==============
class Args:
    """推理所需的基本参数"""
    # 模型配置 (使用 base 版本，最小最快)
    pc_model = "eva_giant_patch14_560"  # 可选: eva02_tiny/small/base/large, eva_giant
    pc_feat_dim = 1408                     # 对应: 192/384/768/1024/1408
    embed_dim = 1024
    pc_encoder_dim = 512
    group_size = 64
    num_group = 512
    patch_dropout = 0.0
    drop_path_rate = 0.0
    
    # 文件路径 (请根据实际情况修改)
    clip_model_name = "EVA02-E-14-plus"
    clip_model_path = "./clip_model/open_clip_pytorch_model.bin"  # CLIP模型路径
    uni3d_ckpt_path = "./checkpoints/model.pt"                     # Uni3D模型路径
    pretrained_pc = ""  # 点云编码器预训练权重(可选)

def load_uni3d_model(args, device):
    """加载 Uni3D 模型"""
    print("=> 加载点云编码器...")
    point_transformer = timm.create_model(
        args.pc_model, 
        checkpoint_path=args.pretrained_pc,
        drop_path_rate=args.drop_path_rate
    )
    point_encoder = PointcloudEncoder(point_transformer, args)
    model = Uni3D(point_encoder=point_encoder)
    
    # 加载预训练权重
    print(f"=> 加载 Uni3D 权重: {args.uni3d_ckpt_path}")
    checkpoint = torch.load(args.uni3d_ckpt_path, map_location='cpu')
    sd = checkpoint['module'] if 'module' in checkpoint else checkpoint
    # 移除 'module.' 前缀
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    
    model.to(device)
    model.eval()
    return model

def load_clip_model(args, device):
    """加载 CLIP 模型"""
    print(f"=> 加载 CLIP 模型: {args.clip_model_name}")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=args.clip_model_name,
        pretrained=args.clip_model_path
    )
    clip_model.to(device)
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer(args.clip_model_name)
    return clip_model, tokenizer, preprocess

def preprocess_point_cloud(points, num_points=10000):
    """
    预处理点云数据
    Input:
        points: numpy array, shape (N, 3) 或 (N, 6)，6维时后3维是颜色
        num_points: 采样点数
    Output:
        xyz: tensor, shape (1, num_points, 3)
        rgb: tensor, shape (1, num_points, 3)
    """
    # 确保点云格式
    if points.shape[1] == 3:
        # 没有颜色信息，使用默认颜色
        xyz = points
        rgb = np.zeros_like(xyz)  # 默认黑色
    else:
        xyz = points[:, :3]
        rgb = points[:, 3:6]
    
    # 随机采样到固定点数
    if xyz.shape[0] > num_points:
        indices = np.random.choice(xyz.shape[0], num_points, replace=False)
    else:
        indices = np.random.choice(xyz.shape[0], num_points, replace=True)
    
    xyz = xyz[indices]
    rgb = rgb[indices]
    
    # 归一化到单位球
    centroid = xyz.mean(axis=0)
    xyz = xyz - centroid
    max_dist = np.max(np.linalg.norm(xyz, axis=1))
    xyz = xyz / max_dist
    
    # 转换为 tensor
    xyz = torch.from_numpy(xyz).float().unsqueeze(0)  # (1, N, 3)
    rgb = torch.from_numpy(rgb).float().unsqueeze(0)  # (1, N, 3)
    
    return xyz, rgb

def encode_point_cloud(model, xyz, rgb, device):
    """编码点云获取特征向量"""
    with torch.no_grad():
        xyz = xyz.to(device)
        rgb = rgb.to(device)
        feature = torch.cat((xyz, rgb), dim=-1)
        pc_embed = model.encode_pc(feature)
        pc_embed = pc_embed / pc_embed.norm(dim=-1, keepdim=True)
    return pc_embed

def encode_text(clip_model, tokenizer, texts, device):
    """编码文本获取特征向量"""
    with torch.no_grad():
        tokens = tokenizer(texts).to(device)
        text_embed = clip_model.encode_text(tokens)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
    return text_embed

def classify_point_cloud(pc_embed, text_embed, labels):
    """
    零样本分类
    返回预测类别和置信度
    """
    similarity = (pc_embed @ text_embed.T).squeeze(0)  # (num_classes,)
    probs = similarity.softmax(dim=-1)
    pred_idx = probs.argmax().item()
    return labels[pred_idx], probs[pred_idx].item(), probs

# ============== 主函数 ==============
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化参数
    args = Args()
    
    # 加载模型
    model = load_uni3d_model(args, device)
    clip_model, tokenizer, _ = load_clip_model(args, device)
    
    print("\n" + "="*50)
    print("模型加载完成！开始推理演示...")
    print("="*50)
    
    # ========== 示例1: 生成随机点云测试 ==========
    print("\n[示例1] 使用随机点云测试")
    random_points = np.random.randn(10000, 3).astype(np.float32)
    xyz, rgb = preprocess_point_cloud(random_points)
    pc_embed = encode_point_cloud(model, xyz, rgb, device)
    print(f"点云特征维度: {pc_embed.shape}")  # 应该是 (1, 1024)
    
    # ========== 示例2: 零样本分类 ==========
    print("\n[示例2] 零样本分类演示")
    # 定义候选类别
    candidate_labels = ["a chair", "a table", "a car", "an airplane", "a lamp"]
    
    # 生成文本模板
    text_prompts = [f"a point cloud model of {label}" for label in candidate_labels]
    text_embed = encode_text(clip_model, tokenizer, text_prompts, device)
    
    # 分类
    pred_label, confidence, all_probs = classify_point_cloud(pc_embed, text_embed, candidate_labels)
    
    print(f"预测类别: {pred_label}")
    print(f"置信度: {confidence:.2%}")
    print("各类别概率:")
    for label, prob in zip(candidate_labels, all_probs.tolist()):
        print(f"  {label}: {prob:.2%}")
    
    # ========== 示例3: 从文件加载点云 ==========
    print("\n[示例3] 从文件加载点云 (示例代码)")
    print("""
    # 加载 .npy 格式点云
    points = np.load("your_pointcloud.npy")  # shape: (N, 3) 或 (N, 6)
    xyz, rgb = preprocess_point_cloud(points)
    pc_embed = encode_point_cloud(model, xyz, rgb, device)
    
    # 加载 .ply 格式点云 (需要 open3d)
    import open3d as o3d
    pcd = o3d.io.read_point_cloud("your_pointcloud.ply")
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(points)
    points_with_color = np.concatenate([points, colors], axis=1)
    xyz, rgb = preprocess_point_cloud(points_with_color)
    """)
    
    print("\n" + "="*50)
    print("✅ 推理演示完成！")
    print("="*50)

if __name__ == "__main__":
    main()
