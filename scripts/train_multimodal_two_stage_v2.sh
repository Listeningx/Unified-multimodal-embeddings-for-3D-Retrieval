#!/bin/bash
# 两阶段训练脚本 V2 (DeepSpeed Zero-2)
# 
# Stage 1: 学习率小，仅点云编码器微调，三模态concat+池化
# Stage 2: 学习率大，点云编码器冻结，TSB+MOE参与训练
# 
# 使用方法:
#   Stage 1: bash scripts/train_multimodal_two_stage_v2.sh stage1
#   Stage 2: bash scripts/train_multimodal_two_stage_v2.sh stage2 <stage1_checkpoint_path>
#   Resume:  bash scripts/train_multimodal_two_stage_v2.sh resume <output_dir_path>
#            例如: bash scripts/train_multimodal_two_stage_v2.sh resume ./output_two_stage_v2/20260205_214814_stage1

set -e

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600           # NCCL 超时 1 小时（默认 1800s），防止 CephFS I/O 抖动导致 NCCL 崩溃
export NCCL_IB_TIMEOUT=230          # InfiniBand 超时（如果使用 IB 网络）

STAGE=${1:-"stage1"}
STAGE1_CHECKPOINT=${2:-""}
RESUME_DIR=${2:-""}  # resume 模式下为输出目录路径

# 基础配置
NUM_GPUS=8
BATCH_SIZE=64
GRAD_ACCUMULATION_STEPS=8
EPOCHS=100
WARMUP=500

# DeepSpeed 配置
ENABLE_DEEPSPEED=true
ZERO_STAGE=2
PRECISION="fp16"  # 可选: fp16, bf16, fp32

# 数据配置
NPOINTS=10000
WORKERS=4

# 模型配置

CLIP_MODEL='ViT-bigG-14'
PRETRAINED="/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/clip_model/open_clip_pytorch_model.bin"  # 或 "laion2b_s9b_b144k"
EMBED_DIM=1280
# 点云模型配置
PC_MODEL="eva_giant_patch14_560.m30m_ft_in22k_in1k"
PRETRAINED_PC="/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/checkpoints/model.pt"
PC_FEAT_DIM=1408
PC_ENCODER_DIM=512
# 输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./output_two_stage_v2/${TIMESTAMP}"

# Resume 参数
RESUME_ARGS=""

# DeepSpeed 启动参数
if [ "$ENABLE_DEEPSPEED" = true ]; then
    DS_ARGS="--enable_deepspeed --zero_stage $ZERO_STAGE --precision $PRECISION"
    echo "DeepSpeed enabled with Zero Stage $ZERO_STAGE, precision: $PRECISION"
else
    DS_ARGS=""
fi

if [ "$STAGE" == "stage1" ]; then
    echo "========================================"
    echo "  Stage 1 Training (concat + pool)"
    echo "  DeepSpeed: $ENABLE_DEEPSPEED"
    echo "  Zero Stage: $ZERO_STAGE"
    echo "========================================"
    echo "  - Learning Rate: Small (1e-4 / 1e-5)"
    echo "  - Point Encoder: Trainable"
    echo "  - TSB/MOE: Not Used"
    echo "========================================"
    
    # Stage 1 学习率配置（较小）
    STAGE1_LR=1e-4
    STAGE1_POINT_LR=1e-5
    
    torchrun --nproc_per_node=$NUM_GPUS \
        train_multimodal_two_stage_v2.py \
        --stage 1 \
        --stage1_lr $STAGE1_LR \
        --stage1_point_lr $STAGE1_POINT_LR \
        --min_lr_ratio 0.01 \
        --batch_size $BATCH_SIZE \
        --grad_accumulation_steps $GRAD_ACCUMULATION_STEPS \
        --epochs $EPOCHS \
        --warmup $WARMUP \
        --npoints $NPOINTS \
        --workers $WORKERS \
        --pc_model $PC_MODEL \
        --clip_model $CLIP_MODEL \
        --pretrained $PRETRAINED \
        --pretrained_pc $PRETRAINED_PC \
        --output_dir "${OUTPUT_DIR}_stage1" \
        --use_embed \
        --use_distributed \
        --tensorboard \
        --log_interval 10 \
        --save_interval 10 \
        --grad_clip 5.0 \
        --wd 0.1 \
        $DS_ARGS

elif [ "$STAGE" == "stage2" ]; then
    if [ -z "$STAGE1_CHECKPOINT" ]; then
        echo "Error: Stage 2 requires Stage 1 checkpoint path"
        echo "Usage: bash train_multimodal_two_stage_v2.sh stage2 <stage1_checkpoint_path>"
        exit 1
    fi
    
    echo "========================================"
    echo "  Stage 2 Training (TSB + MOE)"
    echo "  DeepSpeed: $ENABLE_DEEPSPEED"
    echo "  Zero Stage: $ZERO_STAGE"
    echo "========================================"
    echo "  - Learning Rate: Large (1e-3)"
    echo "  - Point Encoder: Frozen"
    echo "  - TSB/MOE: Trainable"
    echo "  - Checkpoint: $STAGE1_CHECKPOINT"
    echo "========================================"
    
    # Stage 2 学习率配置（较大）
    STAGE2_LR=1e-3
    STAGE2_POINT_LR=0  # 冻结
    
    torchrun --nproc_per_node=$NUM_GPUS \
        train_multimodal_two_stage_v2.py \
        --stage 2 \
        --stage1_checkpoint $STAGE1_CHECKPOINT \
        --stage2_lr $STAGE2_LR \
        --stage2_point_lr $STAGE2_POINT_LR \
        --min_lr_ratio 0.01 \
        --modality_dropout_prob 0.3 \
        --batch_size $BATCH_SIZE \
        --grad_accumulation_steps $GRAD_ACCUMULATION_STEPS \
        --epochs $EPOCHS \
        --warmup $WARMUP \
        --npoints $NPOINTS \
        --workers $WORKERS \
        --pc_model $PC_MODEL \
        --clip_model $CLIP_MODEL \
        --pretrained $PRETRAINED \
        --pretrained_pc $PRETRAINED_PC \
        --output_dir "${OUTPUT_DIR}_stage2" \
        --use_embed \
        --use_distributed \
        --tensorboard \
        --log_interval 10 \
        --save_interval 10 \
        --grad_clip 5.0 \
        --wd 0.1 \
        $DS_ARGS

elif [ "$STAGE" == "resume" ]; then
    if [ -z "$RESUME_DIR" ]; then
        echo "Error: Resume requires output directory path"
        echo "Usage: bash train_multimodal_two_stage_v2.sh resume <output_dir_path>"
        echo "Example: bash train_multimodal_two_stage_v2.sh resume ./output_two_stage_v2/20260205_214814_stage1"
        exit 1
    fi
    
    if [ ! -d "${RESUME_DIR}/checkpoints" ]; then
        echo "Error: Checkpoints directory not found at ${RESUME_DIR}/checkpoints"
        exit 1
    fi
    
    # 从 checkpoints 目录的 latest 文件读取恢复的 epoch
    LATEST_TAG=$(cat "${RESUME_DIR}/checkpoints/latest" 2>/dev/null || echo "unknown")
    
    echo "========================================"
    echo "  Resuming Training from Checkpoint"
    echo "  DeepSpeed: $ENABLE_DEEPSPEED"
    echo "  Zero Stage: $ZERO_STAGE"
    echo "========================================"
    echo "  - Resume Dir: $RESUME_DIR"
    echo "  - Latest Checkpoint: $LATEST_TAG"
    echo "  - Output Dir: $RESUME_DIR (same)"
    echo "========================================"
    
    # Resume 时使用 Stage 1 的学习率配置（如果你的中断发生在 stage1）
    STAGE1_LR=1e-4
    STAGE1_POINT_LR=1e-5
    
    # Resume 模式下减少 workers，降低 CephFS 并发 I/O 压力，减少 Data Time 飙升风险
    RESUME_WORKERS=2
    
    torchrun --nproc_per_node=$NUM_GPUS \
        train_multimodal_two_stage_v2.py \
        --stage 1 \
        --stage1_lr $STAGE1_LR \
        --stage1_point_lr $STAGE1_POINT_LR \
        --min_lr_ratio 0.01 \
        --batch_size $BATCH_SIZE \
        --grad_accumulation_steps $GRAD_ACCUMULATION_STEPS \
        --epochs $EPOCHS \
        --warmup $WARMUP \
        --npoints $NPOINTS \
        --workers $RESUME_WORKERS \
        --pc_model $PC_MODEL \
        --clip_model $CLIP_MODEL \
        --pretrained $PRETRAINED \
        --pretrained_pc $PRETRAINED_PC \
        --output_dir "${RESUME_DIR}" \
        --resume "${RESUME_DIR}/checkpoints" \
        --use_embed \
        --use_distributed \
        --tensorboard \
        --tensorboard_dir "${RESUME_DIR}/tensorboard" \
        --log_interval 10 \
        --save_interval 10 \
        --grad_clip 5.0 \
        --wd 0.1 \
        $DS_ARGS

else
    echo "Unknown stage: $STAGE"
    echo "Usage:"
    echo "  Stage 1: bash train_multimodal_two_stage_v2.sh stage1"
    echo "  Stage 2: bash train_multimodal_two_stage_v2.sh stage2 <stage1_checkpoint_path>"
    echo "  Resume:  bash train_multimodal_two_stage_v2.sh resume <output_dir_path>"
    exit 1
fi

echo "Training completed!"
python /cfs_160T/serenasnliu/scripts/lowmem_highytil.py