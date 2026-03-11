import numpy as np
import os

# 定义要读取的.npy字典文件路径
npy_file_path = "/cfs_160T/serenasnliu/3d-datasets/3DFUTURE/objaverse-processed/merged_for_training_final/3D-FUTURE/9439bfd4-9d1a-430a-9c5f-acc038e13823.npy"

def read_npy_dict_file(file_path):
    """
    读取包含字典的.npy文件，解析每个键值对的详细信息
    :param file_path: .npy文件的绝对路径
    :return: 读取到的字典（失败返回None）
    """
    # 1. 基础路径校验
    if not os.path.exists(file_path):
        print(f"错误：文件不存在 -> {file_path}")
        return None
    if not os.path.isfile(file_path):
        print(f"错误：路径不是文件 -> {file_path}")
        return None

    try:
        # 2. 读取.npy文件（allow_pickle=True是读取字典的关键）
        data = np.load(file_path, allow_pickle=True).item()
        print(f"✅ 成功读取字典文件：{file_path}")
        print(f"📚 字典中包含 {len(data)} 个键值对")
        print("="*80)

        # 3. 遍历字典，解析每个键值对的详细信息
        for idx, (key, value) in enumerate(data.items(), 1):
            print(f"\n【第 {idx} 个键值对】")
            print(f"🔑 键名：{key}")
            print(f"📌 值的类型：{type(value).__name__}")
            
            # 分情况解析值的信息（兼容numpy数组、列表、标量等）
            if isinstance(value, np.ndarray):
                # 若是numpy数组，展示形状、数据类型、前几个元素
                print(f"📏 数组形状：{value.shape}")
                print(f"🔢 数组数据类型：{value.dtype}")
                # 预览前3个元素（避免数据量过大）
                preview = value.flatten()[:3] if value.ndim > 1 else value[:3]
                print(f"👀 前3个元素预览：{preview}")
            elif isinstance(value, (list, tuple)):
                # 若是列表/元组，展示长度、元素类型
                print(f"📏 长度：{len(value)}")
                if len(value) > 0:
                    print(f"🔢 第一个元素类型：{type(value[0]).__name__}")
                    print(f"👀 前3个元素预览：{value[:3]}")
                    if isinstance(value[0], dict):
                        print(value[0]['original'].shape)
            else:
                # 其他类型（字符串、数字等）直接展示值
                print(f"📊 值：{value}")
                if isinstance(value, dict):
                    print(value['original'].shape)
        
        return data

    except AttributeError as e:
        print(f"错误：读取的文件内容不是字典类型 -> {e}")
        return None
    except ValueError as e:
        print(f"错误：文件不是有效的.npy格式 -> {e}")
        return None
    except MemoryError as e:
        print(f"错误：文件过大，内存不足 -> {e}")
        return None
    except Exception as e:
        print(f"读取文件时发生未知错误 -> {e}")
        return None

# 调用函数读取字典文件
npy_dict_data = read_npy_dict_file(npy_file_path)

# 若读取成功，可在此处添加自定义处理逻辑
if npy_dict_data is not None:
    # 示例：获取某个特定键的值（比如假设存在'points'键）
    target_key = "points"
    if target_key in npy_dict_data:
        print(f"\n🎯 提取键 '{target_key}' 的值：{npy_dict_data[target_key]}")