#!/bin/bash
# ============================================================
# SFT Checkpoint 下载脚本 (对应 使用文档/train/快速启动.md 3.1 节)
# 运行环境: verl_train_2 (脚本内部自动激活)
# 使用方式 (在项目根目录下执行):
#   bash train_scrips/install/download_sft_ckpt.sh
# ============================================================

set -euo pipefail

ENV_NAME="verl_train_2"          # 与 install_verl_train.sh 创建的环境一致

# ========== 自定义变量: 下载目录 ==========
# 换用户/机器时, 修改 LOCAL_CKPT_DIR 即可
LOCAL_CKPT_DIR="/diskpool/home/xuxz/test_download"

conda activate "${ENV_NAME}"

# ========== Search checkpoint-10500 (3.5 epoch, 推荐) ==========
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('afordb/sft_result_webshop_and_search_based_on_qwen',
    local_dir='${LOCAL_CKPT_DIR}/search_3.5epoch_10500it',
    allow_patterns='search/3.5epoch_10500it/*')
"

# ========== WebShop checkpoint-8800 (5 epoch, 推荐) ==========
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('afordb/sft_result_webshop_and_search_based_on_qwen',
    local_dir='${LOCAL_CKPT_DIR}/webshop_5epoch_8800it',
    allow_patterns='webshop/5epoch_8800it/*')
"

echo "下载完成: ${LOCAL_CKPT_DIR}"
echo "若模型放置位置与训练脚本默认路径不同, 请同步修改训练脚本中的 actor_rollout_ref.model.path"
