"""
将检查点中的模型权重上传到 Hugging Face Hub
只上传 mp_rank_00_model_states.pt（模型权重），跳过优化器状态和缓存文件

使用前请确保：
1. pip install -U huggingface_hub
2. 在下方填写 HF_TOKEN（需要 Write 权限）
   获取地址: https://huggingface.co/settings/tokens
3. 修改下方 REPO_ID 为你的仓库名
"""

import os
from huggingface_hub import HfApi, create_repo

# ========== 配置项 ==========
# 填写你的 Hugging Face Access Token（Write 权限）
# 获取地址: https://huggingface.co/settings/tokens
HF_TOKEN = "REMOVED_SECRET"

# 修改为你的 Hugging Face 用户名/仓库名
REPO_ID = "Listeningx/remu3d-checkpoints"

# 检查点根目录
CKPT_DIR = "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/output_multimodal/20260130_163844/checkpoints"

# 上传到仓库中的目标路径前缀
PATH_PREFIX = "checkpoints"
# =============================

def main():
    # 直接通过 token 参数传递，避免 login() 的 configparser bug
    api = HfApi(token=HF_TOKEN)
    print("已连接 Hugging Face Hub")

    # 创建仓库（如果不存在）
    try:
        create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, token=HF_TOKEN)
        print(f"仓库已就绪: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"创建仓库时出错: {e}")
        return

    # 遍历所有子目录，按名称排序
    subdirs = sorted([
        d for d in os.listdir(CKPT_DIR)
        if os.path.isdir(os.path.join(CKPT_DIR, d))
    ])

    print(f"\n找到 {len(subdirs)} 个检查点目录: {subdirs}\n")

    for subdir in subdirs:
        model_file = os.path.join(CKPT_DIR, subdir, "mp_rank_00_model_states.pt")

        if not os.path.exists(model_file):
            print(f"[跳过] {subdir}/ 下未找到 mp_rank_00_model_states.pt")
            continue

        path_in_repo = f"{PATH_PREFIX}/{subdir}/mp_rank_00_model_states.pt"
        file_size_gb = os.path.getsize(model_file) / (1024 ** 3)

        print(f"[上传中] {subdir}/mp_rank_00_model_states.pt ({file_size_gb:.2f} GB) -> {path_in_repo}")

        try:
            api.upload_file(
                path_or_fileobj=model_file,
                path_in_repo=path_in_repo,
                repo_id=REPO_ID,
                repo_type="model",
            )
            print(f"[完成] {subdir}")
        except Exception as e:
            print(f"[失败] {subdir}: {e}")
            print("  你可以重新运行脚本，已上传的文件不会重复上传")
            continue

    print(f"\n全部上传完成！访问: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()
