#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 eval_input_qwen3.5max 目录下所有 JSON 文件中提取 image 路径，
去重后复制到专门的文件夹中。
"""

import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

# 源目录：JSON文件所在目录
JSON_DIR = "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/data/test_datasets/objaverse/eval_input_qwen3.5max"

# 目标目录：存放复制的图片
OUTPUT_DIR = "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/data/test_datasets/objaverse/collected_images"

def main():
    json_dir = Path(JSON_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 收集所有JSON文件（递归搜索子目录）
    json_files = sorted(json_dir.rglob("*.json"))
    # 排除 conversion_summary.json
    json_files = [f for f in json_files if "conversion_summary" not in f.name]
    print(f"找到 {len(json_files)} 个 JSON 文件")

    # 2. 收集所有不重复的 image 路径
    unique_images = set()
    for jf in tqdm(json_files, desc="扫描 JSON 文件"):
        try:
            with open(jf, "r") as f:
                data = json.load(f)
            for item in data.get("data", []):
                img_path = item.get("image", "")
                if img_path:
                    unique_images.add(img_path)
        except Exception as e:
            print(f"⚠️ 读取 {jf} 失败: {e}")

    print(f"\n共发现 {len(unique_images)} 个不重复的 image 路径")

    # 3. 复制图片，处理可能的文件名冲突
    # 先检测是否有文件名重复
    name_to_paths = {}
    for img_path in unique_images:
        basename = os.path.basename(img_path)
        if basename not in name_to_paths:
            name_to_paths[basename] = []
        name_to_paths[basename].append(img_path)

    # 统计冲突
    conflicts = {k: v for k, v in name_to_paths.items() if len(v) > 1}
    if conflicts:
        print(f"⚠️ 存在 {len(conflicts)} 个文件名冲突（同名但不同路径），将使用物体ID作为前缀区分")

    # 4. 执行复制
    copied = 0
    skipped = 0
    failed = 0

    for img_path in tqdm(sorted(unique_images), desc="复制图片"):
        if not os.path.exists(img_path):
            print(f"  ⚠️ 文件不存在: {img_path}")
            failed += 1
            continue

        basename = os.path.basename(img_path)

        # 如果该文件名有冲突，使用上级目录的物体ID作为前缀
        if basename in conflicts:
            # 从路径中提取物体ID（倒数第二级目录名）
            # 路径格式：.../物体ID/basecolor/front.png
            parts = img_path.split("/")
            if len(parts) >= 3:
                obj_id = parts[-3]  # 物体ID目录
                dest_name = f"{obj_id}_{basename}"
            else:
                dest_name = img_path.replace("/", "_")
        else:
            dest_name = basename

        dest_path = output_dir / dest_name

        if dest_path.exists():
            skipped += 1
            continue

        try:
            shutil.copy2(img_path, dest_path)
            copied += 1
        except Exception as e:
            print(f"  ⚠️ 复制失败 {img_path}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"复制完成！")
    print(f"  成功复制: {copied}")
    print(f"  已存在跳过: {skipped}")
    print(f"  失败: {failed}")
    print(f"  目标目录: {OUTPUT_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
