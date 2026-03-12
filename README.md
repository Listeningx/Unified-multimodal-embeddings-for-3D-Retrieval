# overview
代码仓库https://github.com/Listeningx/Unified-multimodal-embeddings-for-3D-Retrieval.git
本项目从Uni3d项目代码改编而来。将文本、图像、点云三种模态的资产表示联合起来编码embedding，用于高质量资产检索。另外我们提供了专注类内检索排序质量的基准测试以及构建代码。复现需要先安装pointnet2，https://github.com/baaivision/Uni3D

# pretrained model
open_clip:
位置：./clip_model/...
型号：OpenCLIP (ViT-bigG-14, laion2b_s39b_b160k).
https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/tree/main

uni3d:
位置：./checkpoints/model.pt
https://huggingface.co/BAAI/Uni3D/blob/main/modelzoo/uni3d-g/model.pt

checkpoints:
https://huggingface.co/Listeningx/remu3d-checkpoints

# train
运行脚本有train_multimodal_dropout.sh（随机丢弃模态）和train_multimodal.sh（全模态）
```bash
# ============ 训练模式选择 ============
# 可选值: "single" (单机单卡), "multi" (单机多卡 DDP), "deepspeed" (DeepSpeed 多卡)
# 使用方法: 
#   bash scripts/train_multimodal_dropout.sh          # 默认使用多卡 DDP
#   bash scripts/train_multimodal_dropout.sh single   # 使用单机单卡
#   bash scripts/train_multimodal_dropout.sh multi    # 使用多卡 DDP
#   bash scripts/train_multimodal_dropout.sh deepspeed # 使用 DeepSpeed
#   bash scripts/train_multimodal_dropout.sh resume <output_dir_path>  # 恢复训练
#     例如: bash scripts/train_multimodal_dropout.sh resume ./output_multimodal/20260130_163844


# ============ 训练模式选择 ============
# 可选值: "single" (单机单卡), "multi" (单机多卡 DDP), "deepspeed" (DeepSpeed 多卡)
# 使用方法: 
#   bash scripts/train_multimodal.sh          # 默认使用多卡 DDP
#   bash scripts/train_multimodal.sh single   # 使用单机单卡
#   bash scripts/train_multimodal.sh multi    # 使用多卡 DDP
#   bash scripts/train_multimodal.sh deepspeed # 使用 DeepSpeed
```

# evaluation
模型测试脚本：eval_retrieval_uni3d.sh
关键参数说明：
| 参数 | 值 | 说明 |
|-----|-----|-----|
| --json_dir | eval_input_qwen3.5max | 评测输入数据目录 |
| --dataset | objaverse | 使用 Objaverse 数据集 |
| --query_modal | 循环变量 | query 端使用的模态组合 |
| --gallery_modal | 循环变量 | gallery 端使用的模态组合 |
| --output_dir | objaverse/multimodal | 结果输出目录 |
| --checkpoint | mp_rank_00_model_states.pt | 模型 checkpoint 文件 |
| --model_type | multimodal | 使用多模态模型 |
| --pc_cache_dir | objaverse/pointcloud_cache | 点云特征缓存目录（加速重复运行） |
| --batch_size | 32 | 批大小 |

# 训练数据
数据集：ensembled_embedding，由openshape工作整理发布https://github.com/Colin97/OpenShape_code
https://huggingface.co/datasets/OpenShape/openshape-training-data/tree/main

索引文件：
| 索引文件 | 路径 | 样本数 | 使用条件 |
|----------|------|--------|----------|
| 无 LVIS | ./data/train_no_lvis_uni3d.json | 828,197 | use_lvis=False（默认） |
| 含 LVIS | ./data/train_all_uni3d.json | 874,402 | use_lvis=True |

辅助文件：
GPT-4 文本过滤：./data/gpt4_filter.json	
标记文本质量（flag: "Y"/"N"），过滤低质量文本描述

以上三个大json文件：
链接：https://pan.quark.cn/s/77ed34c05a41
提取码：bw7r

