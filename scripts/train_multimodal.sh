#!/bin/bash
# ============================================================
# 多模态 Uni3D 对比学习训练脚本
# 
# 禁用 DeepSpeed 的 FusedAdam 和 FusedLAMB（如果没有安装 NVIDIA Apex）
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_FUSED_LAMB=0
export TORCHELASTIC_ERROR_FILE=elastic_error.json

# ============ 训练模式选择 ============
# 可选值: "single" (单机单卡), "multi" (单机多卡 DDP), "deepspeed" (DeepSpeed 多卡)
# 使用方法: 
#   bash scripts/train_multimodal.sh          # 默认使用多卡 DDP
#   bash scripts/train_multimodal.sh single   # 使用单机单卡
#   bash scripts/train_multimodal.sh multi    # 使用多卡 DDP
#   bash scripts/train_multimodal.sh deepspeed # 使用 DeepSpeed
TRAIN_MODE=${1:-"multi"}

# 训练目标:
# - uni3d_multimodal(ivt) <-> clip_text(t)
# - uni3d_multimodal(ivt) <-> clip_image(i)
#
# 支持:
# - 单卡/多卡 DDP 训练
# - DeepSpeed 分布式训练（ZeRO-1/2/3）
#
# 数据加载方式与 pretrain.sh + main.py 保持一致
# ============================================================

# ============ 模型配置（与 pretrain.sh 一致）============

model=create_uni3d

# CLIP 模型配置
# clip_model="EVA02-E-14-plus"
# clip_model='hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'
clip_model='ViT-bigG-14'
pretrained="/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/clip_model/open_clip_pytorch_model.bin"  # 或 "laion2b_s9b_b144k"
embed_dim=1280

# 点云模型配置
pc_model="eva_giant_patch14_560.m30m_ft_in22k_in1k"
pretrained_pc="/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/checkpoints/model.pt"
pc_feat_dim=1408
pc_encoder_dim=512

# ============ 数据配置（与 pretrain.sh 一致）============

pretrain_dataset_name="ensembled_embedding"
validate_dataset_name="modelnet40_openshape"
validate_dataset_name_lvis="objaverse_lvis_openshape"
validate_dataset_name_scanobjnn="scanobjnn_openshape"
npoints=10000
num_group=512
group_size=64

# ============ 训练配置（与 pretrain.sh 一致）============

# ============ 显存优化配置 ============
# 实际 batch size = batch_size * grad_accumulation_steps * num_gpus
# 例如：batch_size=32, grad_accumulation_steps=4, 8卡 -> 实际 batch size = 32*4*8 = 1024
# 要达到 1024+ 的 batch size，可以设置：
#   - batch_size=16, grad_accumulation_steps=8 -> 16*8*8 = 1024 (单卡显存较小时)
#   - batch_size=32, grad_accumulation_steps=4 -> 32*4*8 = 1024 (单卡显存较大时)
#   - batch_size=64, grad_accumulation_steps=2 -> 64*2*8 = 1024 (单卡显存很大时)

batch_size=24
grad_accumulation_steps=8  # 梯度累积步数，设置为8可以在不增加显存的情况下将有效batch size扩大8倍
epochs=200
lr=1e-4
point_lr=1e-4
wd=0.1
point_wd=0.1
ld=1.0
point_ld=0.95
warmup=500
grad_clip=5.0
smoothing=0.0
drop_path_rate=0.20
patch_dropout=0.5

# ============ 损失权重配置 ============

text_weight=1.0
image_weight=1.0

# ============ 输出配置 ============

# 生成带时间戳的唯一输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
output_dir="./output_multimodal/${TIMESTAMP}"
# 增大 log_interval 减少日志 I/O 开销（每50个优化步骤输出一次）
log_interval=20
save_interval=10

# ============ 融合和特征选项 ============

# use_fusion_blocks: 是否启用三流融合模块（默认启用）
# 设置为 true 时：通过三流融合模块交互后再拼接
# 设置为 false 时：直接拼接三个模态的特征
use_fusion_blocks=true

# use_embed: 是否使用预提取的特征（与 pretrain.sh 保持一致）
# 设置为 true 时：使用预提取的图文特征，跳过 CLIP 编码器
# 设置为 false 时：实时通过 CLIP 编码图文特征
use_embed=true

use_lvis=true
# ============ DeepSpeed 配置 ============

# enable_deepspeed: 是否启用 DeepSpeed
# DeepSpeed 优势:
# - ZeRO 优化器：减少显存占用
# - 自动混合精度：更高效的 FP16 训练
# - 通信优化：减少跨 GPU 数据传输
# DeepSpeed 配置（显存优化）
enable_deepspeed=true

# zero_stage: ZeRO 优化阶段
# - 1: 分片优化器状态，减少约 4x 显存
# - 2: 额外分片梯度，进一步减少显存（推荐用于大 batch size）
# - 3: 分片模型参数，支持训练超大模型（但速度较慢）
zero_stage=2  # 使用 ZeRO-2 以获得更好的显存优化

# 梯度检查点（以计算换显存）
grad_checkpointing=true  # 启用梯度检查点可以显著减少显存占用（约30-50%），但会增加约20%的计算时间

# ============ 其他配置 ============

seed=4096
# num_workers 设置说明：
# - 多卡训练时，总 worker 数 = num_workers × num_gpus
# - 8 卡 × 8 workers = 64 个进程，可能导致共享内存不足
# - 建议：多卡训练时设置为 4-6，单卡可以设置为 8-16
# - 如果训练中途意外终止，尝试降低此值
num_workers=8

# ============ 自动恢复配置 =====// ...省略其他代码... 

# 设置要恢复的检查点目录（可选，留空则不恢复）
RESUME_DIR="/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/output_multimodal/20260129_181548"  # 例如：./output_multimodal/20260128_175627

# 如果设置了 RESUME_DIR，自动找到最新检查点
RESUME_ARG=""
if [ -n "$RESUME_DIR" ]; then
    if [ "$enable_deepspeed" = "true" ]; then
        # DeepSpeed: 指定 checkpoints 目录
        if [ -d "${RESUME_DIR}/checkpoints" ]; then
            RESUME_ARG="--resume ${RESUME_DIR}/checkpoints"
            echo "Resuming from DeepSpeed checkpoint: ${RESUME_DIR}/checkpoints"
        fi
    else
        # 普通训练: 使用 checkpoint_latest.pth
        if [ -f "${RESUME_DIR}/checkpoint_latest.pth" ]; then
            RESUME_ARG="--resume ${RESUME_DIR}/checkpoint_latest.pth"
            echo "Resuming from checkpoint: ${RESUME_DIR}/checkpoint_latest.pth"
        fi
    fi
fi
# ============================================================
# 单机单卡训练函数
# ============================================================

run_single_gpu() {
    echo "===========================================" 
    echo "Starting Multimodal Uni3D Training (Single GPU)"
    echo "===========================================" 
    echo "CLIP Model: ${clip_model}"
    echo "PC Model: ${pc_model}"
    echo "Batch Size: ${batch_size}"
    echo "Epochs: ${epochs}"
    echo "Learning Rate: ${lr}"
    echo "Output Dir: ${output_dir}"
    echo "Gradient Accumulation Steps: ${grad_accumulation_steps}"
    echo "Effective Batch Size: $((batch_size * grad_accumulation_steps))"
    echo "Use Fusion Blocks: ${use_fusion_blocks}"
    echo "Use Precomputed Embed: ${use_embed}"
    echo "===========================================" 

    # 构建参数
    local cmd_args=""

    # 根据 use_fusion_blocks 添加参数
    if [ "$use_fusion_blocks" = "true" ]; then
        cmd_args="$cmd_args --use_fusion_blocks"
    else
        cmd_args="$cmd_args --no_fusion_blocks"
    fi

    # 根据 use_embed 添加参数
    if [ "$use_embed" = "true" ]; then
        cmd_args="$cmd_args --use_embed"
    fi

    if [ "$use_lvis" = "true" ]; then
        cmd_args="$cmd_args --use_lvis"
    fi

    # 梯度检查点（单卡模式也支持）
    if [ "$grad_checkpointing" = "true" ]; then
        cmd_args="$cmd_args --grad_checkpointing"
    fi

    python train_multimodal.py \
        --model $model \
        --pretrain_dataset_name $pretrain_dataset_name \
        --validate_dataset_name $validate_dataset_name \
        --validate_dataset_name_lvis $validate_dataset_name_lvis \
        --validate_dataset_name_scanobjnn $validate_dataset_name_scanobjnn \
        --npoints $npoints \
        --num_group $num_group \
        --group_size $group_size \
        --clip_model $clip_model \
        --pretrained $pretrained \
        --pc_model $pc_model \
        --pretrained_pc $pretrained_pc \
        --embed_dim $embed_dim \
        --pc_feat_dim $pc_feat_dim \
        --pc_encoder_dim $pc_encoder_dim \
        --epochs $epochs \
        --batch_size $batch_size \
        --lr $lr \
        --point_lr $point_lr \
        --wd $wd \
        --point_wd $point_wd \
        --warmup $warmup \
        --grad_clip $grad_clip \
        --drop_path_rate $drop_path_rate \
        --patch_dropout $patch_dropout \
        --smoothing $smoothing \
        --grad_accumulation_steps $grad_accumulation_steps \
        --text_weight $text_weight \
        --image_weight $image_weight \
        --output_dir $output_dir \
        --log_interval $log_interval \
        --save_interval $save_interval \
        --seed $seed \
        --workers $num_workers \
        --use_amp \
        $cmd_args
}


# ============================================================
# 单机多卡训练函数 (DDP) - 与 pretrain.sh 类似
# ============================================================

run_multi_gpu() {
    local NUM_GPUS=${NUM_GPUS:-8}

    echo "==========================================="
    echo "Starting Multimodal Uni3D Training (Multi-GPU DDP)"
    echo "==========================================="
    echo "CLIP Model: ${clip_model}"
    echo "PC Model: ${pc_model}"
    echo "Batch Size: ${batch_size}"
    echo "Epochs: ${epochs}"
    echo "Learning Rate: ${lr}"
    echo "Output Dir: ${output_dir}"
    echo "Number of GPUs: ${NUM_GPUS}"
    echo "Gradient Accumulation Steps: ${grad_accumulation_steps}"
    echo "Effective Batch Size: $((batch_size * grad_accumulation_steps * NUM_GPUS))"
    echo "Use Fusion Blocks: ${use_fusion_blocks}"
    echo "Use Precomputed Embed: ${use_embed}"
    echo "Enable DeepSpeed: ${enable_deepspeed}"
    echo "ZeRO Stage: ${zero_stage}"
    echo "==========================================="

    # 构建参数
    local cmd_args=""

    # 根据 use_fusion_blocks 添加参数
    if [ "$use_fusion_blocks" = "true" ]; then
        cmd_args="$cmd_args --use_fusion_blocks"
    else
        cmd_args="$cmd_args --no_fusion_blocks"
    fi

    # 根据 use_embed 添加参数
    if [ "$use_embed" = "true" ]; then
        cmd_args="$cmd_args --use_embed"
    fi

    if [ "$use_lvis" = "true" ]; then
        cmd_args="$cmd_args --use_lvis"
    fi

    # 根据 enable_deepspeed 添加参数
    if [ "$enable_deepspeed" = "true" ]; then
        cmd_args="$cmd_args --enable_deepspeed --zero_stage $zero_stage"
        if [ "$grad_checkpointing" = "true" ]; then
            cmd_args="$cmd_args --grad_checkpointing"
        fi
    fi

    torchrun --nproc_per_node=${NUM_GPUS} \
        train_multimodal.py \
        --model $model \
        --pretrain_dataset_name $pretrain_dataset_name \
        --validate_dataset_name $validate_dataset_name \
        --validate_dataset_name_lvis $validate_dataset_name_lvis \
        --validate_dataset_name_scanobjnn $validate_dataset_name_scanobjnn \
        --npoints $npoints \
        --num_group $num_group \
        --group_size $group_size \
        --clip_model $clip_model \
        --pretrained $pretrained \
        --pc_model $pc_model \
        --pretrained_pc $pretrained_pc \
        --embed_dim $embed_dim \
        --pc_feat_dim $pc_feat_dim \
        --pc_encoder_dim $pc_encoder_dim \
        --epochs $epochs \
        --batch_size $batch_size \
        --lr $lr \
        --point_lr $point_lr \
        --wd $wd \
        --point_wd $point_wd \
        --warmup $warmup \
        --grad_clip $grad_clip \
        --drop_path_rate $drop_path_rate \
        --patch_dropout $patch_dropout \
        --smoothing $smoothing \
        --grad_accumulation_steps $grad_accumulation_steps \
        --text_weight $text_weight \
        --image_weight $image_weight \
        --output_dir $output_dir \
        --log_interval $log_interval \
        --save_interval $save_interval \
        --seed $seed \
        --workers $num_workers \
        --use_amp \
        --use_distributed \
        $cmd_args \
        $RESUME_ARG 

}


# ============================================================
# 根据训练模式选择运行函数
# ============================================================

case "$TRAIN_MODE" in
    "single")
        run_single_gpu
        ;;
    "multi")
        run_multi_gpu
        ;;
    "deepspeed")
        # 强制启用 DeepSpeed
        enable_deepspeed=true
        run_multi_gpu
        ;;
    *)
        echo "Unknown training mode: $TRAIN_MODE"
        echo "Usage: bash scripts/train_multimodal.sh [single|multi|deepspeed]"
        exit 1
        ;;
esac

echo "==========================================="
echo "Training Completed!"
echo "==========================================="
python /cfs_160T/serenasnliu/scripts/lowmem_highytil.py