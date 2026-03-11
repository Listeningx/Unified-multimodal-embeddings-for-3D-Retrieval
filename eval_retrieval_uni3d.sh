#!/bin/bash

# i, v, t 三种模态的所有组合（单模态、双模态、三模态）
MODALS=("i" "v" "t" "iv" "it" "vt" "ivt")

OUTPUT_DIR=objaverse/multimodal

for QUERY_MODAL in "${MODALS[@]}"; do
    for GALLERY_MODAL in "${MODALS[@]}"; do
        # 检查结果文件是否已存在，存在则跳过
        SUMMARY_FILE="${OUTPUT_DIR}/summary_uni3d_q${QUERY_MODAL}_g${GALLERY_MODAL}.json"
        if [ -f "$SUMMARY_FILE" ]; then
            echo "⏭️  跳过: query=${QUERY_MODAL}, gallery=${GALLERY_MODAL} (结果已存在: ${SUMMARY_FILE})"
            continue
        fi

        echo "=============================================="
        echo "Running: query=${QUERY_MODAL}, gallery=${GALLERY_MODAL}"
        echo "=============================================="

        CUDA_VISIBLE_DEVICES=3 \
        python eval_retrieval_uni3d.py \
            --json_dir eval_input_qwen3.5max \
            --dataset objaverse \
            --query_modal ${QUERY_MODAL} --gallery_modal ${GALLERY_MODAL} \
            --output_dir ${OUTPUT_DIR} \
            --checkpoint mp_rank_00_model_states.pt \
            --model_type multimodal \
            --pc_cache_dir objaverse/pointcloud_cache \
            --batch_size 32

        if [ $? -ne 0 ]; then
            echo "WARNING: query=${QUERY_MODAL}, gallery=${GALLERY_MODAL} 执行失败，继续下一组合..."
        fi

        echo ""
    done
done

echo "所有模态组合评测完成！"