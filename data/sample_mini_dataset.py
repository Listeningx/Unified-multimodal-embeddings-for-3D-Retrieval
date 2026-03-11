#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 train_no_lvis_uni3d.json 中随机选择 20% 的数据作为迷你训练集
"""

import json
import random

# 设置随机种子，保证可复现性
random.seed(42)

# 输入和输出文件路径
input_file = "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/data/train_all_uni3d.json"
output_file = "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/data/train_all_uni3d_mini.json"

print(f"正在读取文件: {input_file}")

# 读取原始JSON文件
with open(input_file, 'r') as f:
    data = json.load(f)

total_count = len(data)
print(f"原始数据总量: {total_count}")

# 随机选择20%的数据
sample_count = int(total_count * 0.2)
print(f"将随机选择 20% 的数据: {sample_count} 条")

# 获取所有键并随机采样
all_keys = list(data.keys())
sampled_keys = random.sample(all_keys, sample_count)

# 创建迷你数据集
mini_data = {k: data[k] for k in sampled_keys}

print(f"迷你训练集大小: {len(mini_data)}")

# 保存迷你数据集
print(f"正在保存到: {output_file}")
with open(output_file, 'w') as f:
    json.dump(mini_data, f, indent=4)

print("完成！")
print(f"原始数据量: {total_count}")
print(f"迷你数据量: {len(mini_data)} ({len(mini_data)/total_count*100:.1f}%)")
