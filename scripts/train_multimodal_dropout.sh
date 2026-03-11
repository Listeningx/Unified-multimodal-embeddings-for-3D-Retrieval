#!/bin/bash
# ============================================================
# 多模态 Uni3D 对比学习训练脚本 (带 Modality Dropout)
# 
# 与 train_multimodal.sh 的区别：
# - 使用 train_multimodal_dropout.py 而不是 train_multimodal.py
# - 支持随机的 Modality Dropout，训练时随机丢弃某些模态
# - 可配置各模态组合的出现概率
#
# 禁用 DeepSpeed 的 FusedAdam 和 FusedLAMB（如果没有安装 NVIDIA Apex）
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_FUSED_LAMB=0
export TORCHELASTIC_ERROR_FILE=elastic_error.json
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600           # NCCL 超时 1 小时（默认 1800s），防止 CephFS I/O 抖动导致 NCCL 崩溃
export NCCL_IB_TIMEOUT=230          # InfiniBand 超时（如果使用 IB 网络）

# ============ 训练模式选择 ============
# 可选值: "single" (单机单卡), "multi" (单机多卡 DDP), "deepspeed" (DeepSpeed 多卡)
# 使用方法: 
#   bash scripts/train_multimodal_dropout.sh          # 默认使用多卡 DDP
#   bash scripts/train_multimodal_dropout.sh single   # 使用单机单卡
#   bash scripts/train_multimodal_dropout.sh multi    # 使用多卡 DDP
#   bash scripts/train_multimodal_dropout.sh deepspeed # 使用 DeepSpeed
#   bash scripts/train_multimodal_dropout.sh resume <output_dir_path>  # 恢复训练
#     例如: bash scripts/train_multimodal_dropout.sh resume ./output_multimodal/20260130_163844
TRAIN_MODE=${1:-"multi"}
RESUME_OUTPUT_DIR=${2:-""}  # resume 模式下为输出目录路径

# 训练目标:
# - uni3d_multimodal(随机模态组合) <-> clip_text(t)  [仅当有 t 时]
# - uni3d_multimodal(随机模态组合) <-> clip_image(i) [仅当有 i 时]
#
# Modality Dropout 概率（默认配置）:
# - ivt: 50%  (完整三模态)
# - iv:  15%  (图像+点云)
# - vt:  15%  (点云+文本)
# - v:   10%  (仅点云)
# - it:  5%   (图像+文本，无点云，比例极小)
# - i:   2.5% (仅图像)
# - t:   2.5% (仅文本)
#
# 支持:
# - 单卡/多卡 DDP 训练
# - DeepSpeed 分布式训练（ZeRO-1/2/3）
# - 可配置的模态 dropout 概率
#
# 数据加载方式与 pretrain.sh + main.py 保持一致
# ============================================================

# ============ 模型配置（与 pretrain.sh 一致）============

model=create_uni3d

# CLIP 模型配置
# clip_model="EVA02-E-14-plus"
# clip_model='hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'
clip_model='ViT-bigG-14'
pretrained="Uni3D/clip_model/open_clip_pytorch_model.bin"  # 或 "laion2b_s9b_b144k"
embed_dim=1280

# 点云模型配置
pc_model="eva_giant_patch14_560.m30m_ft_in22k_in1k"
pretrained_pc="Uni3D/checkpoints/model.pt"
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

batch_size=36
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

# ============ Modality Dropout 配置 ============

# enable_modality_dropout: 是否启用 Modality Dropout（默认启用）
# 设置为 true 时：训练时随机丢弃某些模态，提高模型对缺失模态的鲁棒性
# 设置为 false 时：始终使用完整的 ivt 三模态
enable_modality_dropout=true

# 各模态组合的出现概率（总和为 1）
# 注意：it 组合（图像+文本，无点云）的比例非常小，因为这种情况不太常见
modal_prob_ivt=0.20   # 完整三模态
modal_prob_iv=0.1    # 图像+点云
modal_prob_vt=0.1    # 点云+文本
modal_prob_v=0.50     # 仅点云（核心模态）
modal_prob_it=0.05    # 图像+文本（无点云，比例极小）
modal_prob_i=0.025    # 仅图像
modal_prob_t=0.025    # 仅文本

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
# RESUME_DIR: 设置为输出目录路径（不含 checkpoints），例如：./output_multimodal/20260130_163844
# 如果使用 resume 模式启动，RESUME_DIR 会自动从命令行参数获取
RESUME_DIR=""

# 如果设置了 RESUME_DIR，自动找到最新检查点
RESUME_ARG=""
if [ -n "$RESUME_DIR" ]; then
    if [ "$enable_deepspeed" = "true" ]; then
        # DeepSpeed: 指定 checkpoints 目录
        if [ -d "${RESUME_DIR}/checkpoints" ]; then
            RESUME_ARG="--resume ${RESUME_DIR}/checkpoints"
            echo "Resuming from DeepSpeed checkpoint: ${RESUME_DIR}/checkpoints"
        else
            echo "Warning: DeepSpeed checkpoints dir not found at ${RESUME_DIR}/checkpoints"
        fi
    else
        # 普通训练: 使用 checkpoint_latest.pth
        if [ -f "${RESUME_DIR}/checkpoint_latest.pth" ]; then
            RESUME_ARG="--resume ${RESUME_DIR}/checkpoint_latest.pth"
            echo "Resuming from checkpoint: ${RESUME_DIR}/checkpoint_latest.pth"
        else
            echo "Warning: checkpoint_latest.pth not found at ${RESUME_DIR}/"
        fi
    fi
    # 恢复训练时使用原来的输出目录
    output_dir="${RESUME_DIR}"
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
    echo "Enable Modality Dropout: ${enable_modality_dropout}"
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

    # Modality Dropout 参数
    if [ "$enable_modality_dropout" = "true" ]; then
        cmd_args="$cmd_args --enable_modality_dropout"
        cmd_args="$cmd_args --modal_prob_ivt $modal_prob_ivt"
        cmd_args="$cmd_args --modal_prob_iv $modal_prob_iv"
        cmd_args="$cmd_args --modal_prob_vt $modal_prob_vt"
        cmd_args="$cmd_args --modal_prob_v $modal_prob_v"
        cmd_args="$cmd_args --modal_prob_it $modal_prob_it"
        cmd_args="$cmd_args --modal_prob_i $modal_prob_i"
        cmd_args="$cmd_args --modal_prob_t $modal_prob_t"
    else
        cmd_args="$cmd_args --no_modality_dropout"
    fi

    python train_multimodal_dropout.py \
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
        $RESUME_ARG \
        $cmd_args
}


# ============================================================
# 单机多卡训练函数 (DDP) - 与 pretrain.sh 类似
# ============================================================

run_multi_gpu() {
    local NUM_GPUS=${NUM_GPUS:-8}

    echo "==========================================="
    echo "Starting Multimodal Uni3D Training (Multi-GPU DDP with Modality Dropout)"
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
    echo "Enable Modality Dropout: ${enable_modality_dropout}"
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

    # Modality Dropout 参数
    if [ "$enable_modality_dropout" = "true" ]; then
        cmd_args="$cmd_args --enable_modality_dropout"
        cmd_args="$cmd_args --modal_prob_ivt $modal_prob_ivt"
        cmd_args="$cmd_args --modal_prob_iv $modal_prob_iv"
        cmd_args="$cmd_args --modal_prob_vt $modal_prob_vt"
        cmd_args="$cmd_args --modal_prob_v $modal_prob_v"
        cmd_args="$cmd_args --modal_prob_it $modal_prob_it"
        cmd_args="$cmd_args --modal_prob_i $modal_prob_i"
        cmd_args="$cmd_args --modal_prob_t $modal_prob_t"
    else
        cmd_args="$cmd_args --no_modality_dropout"
    fi

    torchrun --nproc_per_node=${NUM_GPUS} \
        train_multimodal_dropout.py \
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
    "resume")
        # 恢复训练模式
        if [ -z "$RESUME_OUTPUT_DIR" ]; then
            echo "Error: Resume requires output directory path"
            echo "Usage: bash scripts/train_multimodal_dropout.sh resume <output_dir_path>"
            echo "Example: bash scripts/train_multimodal_dropout.sh resume ./output_multimodal/20260130_163844"
            exit 1
        fi
        
        if [ ! -d "${RESUME_OUTPUT_DIR}/checkpoints" ]; then
            echo "Error: Checkpoints directory not found at ${RESUME_OUTPUT_DIR}/checkpoints"
            exit 1
        fi
        
        # 设置 RESUME_DIR 并重新生成 RESUME_ARG
        RESUME_DIR="$RESUME_OUTPUT_DIR"
        if [ "$enable_deepspeed" = "true" ]; then
            RESUME_ARG="--resume ${RESUME_DIR}/checkpoints"
        else
            RESUME_ARG="--resume ${RESUME_DIR}/checkpoint_latest.pth"
        fi
        # 恢复训练时使用原来的输出目录
        output_dir="${RESUME_DIR}"
        
        LATEST_TAG=$(cat "${RESUME_DIR}/checkpoints/latest" 2>/dev/null || echo "unknown")
        
        # Resume 模式下减少 workers，降低 CephFS 并发 I/O 压力，减少 Data Time 飙升风险
        num_workers=2
        
        echo "==========================================="
        echo "Resuming Training from Checkpoint"
        echo "  Resume Dir: $RESUME_DIR"
        echo "  Latest Tag: $LATEST_TAG"
        echo "  Output Dir: $output_dir (same as before)"
        echo "  Workers: $num_workers (reduced for resume stability)"
        echo "==========================================="
        
        run_multi_gpu
        ;;
    *)
        echo "Unknown training mode: $TRAIN_MODE"
        echo "Usage: bash scripts/train_multimodal_dropout.sh [single|multi|deepspeed|resume <output_dir>]"
        exit 1
        ;;
esac

echo "==========================================="
echo "Training Completed!"
echo "==========================================="
