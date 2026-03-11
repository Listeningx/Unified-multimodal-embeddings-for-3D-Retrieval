#!/bin/bash
# ============================================================
# 多模态 Uni3D 两阶段训练脚本
#
# Stage 1: 
#   - 三流融合 (TripleStreamBlock) 同时输入点云、图片、文本特征
#   - 单专家 MOE (num_experts=1)，只有 'v' 模态的 router/tokens
#   - 融合后只取点云部分的特征进入 MOE
#   - point_encoder 解冻参与训练
#   - Modality Dropout: DISABLED
#
# Stage 2: 
#   - 使用 modality-dropout，7 种模态组合都有机会出现
#   - 3 专家 MOE，从 Stage 1 单专家复制初始化
#   - 其他模态模块（router, resample_tokens 等）随机初始化
#   - point_encoder 冻结
#
# 使用:
#   bash scripts/train_multimodal_two_stage.sh stage1
#   bash scripts/train_multimodal_two_stage.sh stage2 /path/to/stage1_ckpt.pth
# ============================================================

export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_FUSED_LAMB=0
export TORCHELASTIC_ERROR_FILE=elastic_error.json

STAGE_MODE=${1:-"stage1"}
STAGE1_CHECKPOINT=${2:-""}

# ============ 模型配置 ============
clip_model='ViT-bigG-14'
pretrained="/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/clip_model/open_clip_pytorch_model.bin"
embed_dim=1280
pc_model="eva_giant_patch14_560.m30m_ft_in22k_in1k"
pretrained_pc="/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/checkpoints/model.pt"
pc_feat_dim=1408
pc_encoder_dim=512

# ============ 数据配置 ============
pretrain_dataset_name="ensembled_embedding"
validate_dataset_name="modelnet40_openshape"
validate_dataset_name_lvis="objaverse_lvis_openshape"
validate_dataset_name_scanobjnn="scanobjnn_openshape"
npoints=10000
num_group=512
group_size=64

# ============ 通用训练配置 ============
batch_size=48
grad_accumulation_steps=8
wd=0.1
# warmup=500
warmup=100
grad_clip=5.0
drop_path_rate=0.20
patch_dropout=0.5
text_weight=1.0
image_weight=1.0
log_interval=20
save_interval=10
use_embed=true
use_lvis=true
enable_deepspeed=true
zero_stage=2
grad_checkpointing=true
seed=4096
num_workers=4
NUM_GPUS=${NUM_GPUS:-8}

# ============ 阶段特定配置 ============
if [ "$STAGE_MODE" = "stage1" ]; then
    stage=1
    epochs=100
    lr=1e-4
    point_lr=1e-4  # Stage 1: point_encoder 参与训练，使用正常学习率
    output_dir="./output_two_stage/stage1_$(date +%Y%m%d_%H%M%S)"
    modality_dropout_prob=0.0  # Stage 1 不使用 modality-dropout
    
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                     📌 STAGE 1 TRAINING                          ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    echo "║  • Fusion: i + v + t (all modalities input)                      ║"
    echo "║  • MOE: Single expert, only 'v' modality router/tokens           ║"
    echo "║  • Only point cloud features enter MOE after fusion              ║"
    echo "║  • Point Encoder: UNFROZEN (training)                            ║"
    echo "║  • Modality Dropout: DISABLED                                    ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    
elif [ "$STAGE_MODE" = "stage2" ]; then
    stage=2
    epochs=100
    lr=1e-4
    point_lr=0  # Stage 2: point_encoder 冻结，学习率设为 0
    output_dir="./output_two_stage/stage2_$(date +%Y%m%d_%H%M%S)"
    modality_dropout_prob=0.3  # Stage 2 使用 modality-dropout
    
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                     📌 STAGE 2 TRAINING                          ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    echo "║  • Modality: All 7 combinations (i,v,t,iv,it,vt,ivt)             ║"
    echo "║  • MOE: 3 experts (copied from Stage 1 single expert)            ║"
    echo "║  • Point Encoder: FROZEN                                         ║"
    echo "║  • Modality Dropout: ENABLED (prob=${modality_dropout_prob})     ║"
    echo "║  • Other modal modules: RANDOMLY INITIALIZED                     ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    
    if [ -z "$STAGE1_CHECKPOINT" ]; then
        echo ""
        echo "⚠️  WARNING: No Stage 1 checkpoint provided!"
        echo "   Usage: bash scripts/train_multimodal_two_stage.sh stage2 /path/to/stage1_ckpt.pth"
        echo ""
    else
        echo ""
        echo "📂 Stage 1 Checkpoint: $STAGE1_CHECKPOINT"
        echo ""
    fi
else
    echo "❌ Unknown stage mode: $STAGE_MODE"
    echo ""
    echo "Usage:"
    echo "  Stage 1: bash scripts/train_multimodal_two_stage.sh stage1"
    echo "  Stage 2: bash scripts/train_multimodal_two_stage.sh stage2 /path/to/stage1_ckpt.pth"
    exit 1
fi

# ============ 构建参数 ============
cmd_args="--stage $stage"
cmd_args="$cmd_args --modality_dropout_prob $modality_dropout_prob"

[ "$use_embed" = "true" ] && cmd_args="$cmd_args --use_embed"
[ "$use_lvis" = "true" ] && cmd_args="$cmd_args --use_lvis"
cmd_args="$cmd_args --use_fusion_blocks"
[ "$enable_deepspeed" = "true" ] && cmd_args="$cmd_args --enable_deepspeed --zero_stage $zero_stage"
[ "$grad_checkpointing" = "true" ] && cmd_args="$cmd_args --grad_checkpointing"
[ "$stage" = "2" ] && [ -n "$STAGE1_CHECKPOINT" ] && cmd_args="$cmd_args --stage1_checkpoint $STAGE1_CHECKPOINT"

echo ""
echo "==========================================="
echo "📊 Training Configuration"
echo "==========================================="
echo "  Stage:              $stage"
echo "  GPUs:               $NUM_GPUS"
echo "  Epochs:             $epochs"
echo "  Batch Size:         $batch_size"
echo "  Grad Accumulation:  $grad_accumulation_steps"
echo "  Effective BS:       $((batch_size * grad_accumulation_steps * NUM_GPUS))"
echo "  Learning Rate:      $lr"
echo "  Point LR:           $point_lr"
echo "  Output Dir:         $output_dir"
echo "==========================================="
echo ""

# ============ 运行训练 ============
torchrun --nproc_per_node=${NUM_GPUS} train_multimodal_two_stage.py \
    --model create_uni3d \
    --pretrain_dataset_name $pretrain_dataset_name \
    --validate_dataset_name $validate_dataset_name \
    --validate_dataset_name_lvis $validate_dataset_name_lvis \
    --validate_dataset_name_scanobjnn $validate_dataset_name_scanobjnn \
    --npoints $npoints --num_group $num_group --group_size $group_size \
    --clip_model $clip_model --pretrained $pretrained \
    --pc_model $pc_model --pretrained_pc $pretrained_pc \
    --embed_dim $embed_dim --pc_feat_dim $pc_feat_dim --pc_encoder_dim $pc_encoder_dim \
    --epochs $epochs --batch_size $batch_size \
    --lr $lr --point_lr $point_lr --wd $wd --point_wd $wd \
    --warmup $warmup --grad_clip $grad_clip \
    --drop_path_rate $drop_path_rate --patch_dropout $patch_dropout \
    --grad_accumulation_steps $grad_accumulation_steps \
    --text_weight $text_weight --image_weight $image_weight \
    --output_dir $output_dir --log_interval $log_interval --save_interval $save_interval \
    --seed $seed --workers $num_workers \
    --use_amp --use_distributed $cmd_args

echo ""
echo "==========================================="
echo "✅ Stage $stage Training Completed!"
echo "==========================================="
echo "📂 Output: $output_dir"

if [ "$stage" = "1" ]; then
    echo ""
    echo "📌 Next Step:"
    echo "   Run Stage 2 with:"
    echo "   bash scripts/train_multimodal_two_stage.sh stage2 ${output_dir}/checkpoint_stage1_best.pth"
fi

echo "==========================================="
