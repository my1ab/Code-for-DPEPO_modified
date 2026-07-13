#!/bin/bash
# ============================================================
# verl_train 环境完整安装脚本
# 对应环境: /diskpool/home/xuxz/miniconda3/envs/verl_train
# Python: 3.11.15  |  CUDA: 12.4  |  Torch: 2.6.0
# ============================================================
# 使用方式:
#   bash scripts/install_verl_train.sh
#
# 说明:
# - 脚本按依赖层次分组安装，高版本 pip 会自动解析依赖树，
#   因此很多包不需要逐个列出，它们会作为核心包的依赖自动安装。
# - 版本固定仅针对需要精确匹配的关键包。
# ============================================================

set -euo pipefail  # 遇到错误即停止

# ---------- 颜色输出 ----------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; }

ENV_NAME="${1:-verl_train}"  # 可通过命令行参数指定环境名，默认 verl_train
PYTHON_VERSION="3.11"
TORCH_VERSION="2.6.0"
CUDA_VERSION="cu124"
VLLM_VERSION="0.8.5"
FLASH_ATTN_VERSION="2.7.4.post1"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ============================================================
# Step 0: 创建 conda 环境
# ============================================================
log "Step 0: 创建 conda 环境 '${ENV_NAME}' (Python ${PYTHON_VERSION})"
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# 升级 pip
pip install --upgrade pip

# ============================================================
# Step 1: 安装 PyTorch 全家桶（必须先装，后续所有包依赖它）
# ============================================================
log "Step 1: 安装 PyTorch ${TORCH_VERSION} (CUDA 12.4)"
pip3 install \
    torch=="${TORCH_VERSION}" \
    torchvision=="${TORCH_VERSION%.*}.0" \
    torchaudio=="${TORCH_VERSION%.*}.0" \
    --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"

# ============================================================
# Step 2: 安装 FlashAttention（编译耗时，需 --no-build-isolation）
# ============================================================
log "Step 2: 安装 FlashAttention ${FLASH_ATTN_VERSION}"
# 直接安装预编译 wheel，避免本地编译
pip install flash-attn=="${FLASH_ATTN_VERSION}" --no-build-isolation

# FlashInfer (vLLM 的 prefix-prefilling 需要)
# 注意: 需要根据 Python 版本选择对应 wheel
log "Step 2b: 安装 FlashInfer"
pip install flashinfer-python --find-links https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python

# xformers (加速 attention)
pip install "xformers<0.0.30"

# ============================================================
# Step 3: 安装 vLLM（会连带安装大量依赖）
#   - 自动带出: numpy, requests, aiohttp, pydantic, starlette,
#     fastapi, uvicorn, pyzmq, msgpack, orjson, msgspec,
#     safetensors, tokenizers, huggingface_hub, filelock, psutil ...
# ============================================================
log "Step 3: 安装 vLLM ${VLLM_VERSION}"
pip install "vllm==${VLLM_VERSION}"

# ============================================================
# Step 4: 安装 Transformers 生态
#   - transformers 自动带出: tokenizers, safetensors, huggingface_hub,
#     filelock, fsspec, regex, tqdm, Jinja2, MarkupSafe ...
#   - accelerate 自动带出: numpy, psutil ...
#   - datasets 自动带出: dill, multiprocess, pyarrow, pandas, aiohttp ...
#   - peft 自动带出: huggingface_hub, safetensors ...
# ============================================================
log "Step 4: 安装 Transformers 生态 (transformers, accelerate, datasets, peft)"
pip install \
    "transformers[hf_xet]==4.51.1" \
    "accelerate==1.13.0" \
    "datasets==4.8.5" \
    "peft==0.17.0"

# ============================================================
# Step 5: 安装分布式训练框架
#   - ray[default] 自动带出: grpcio, protobuf, google-auth, cachetools,
#     cloudpickle, aiosignal, frozenlist ...
# ============================================================
log "Step 5: 安装分布式训练框架 (Ray, tensordict, torchrl)"
pip install \
    "ray[default]==2.47.1" \
    "tensordict==0.12.4" \
    "torchrl==0.12.0"

# ============================================================
# Step 6: 安装 Agent 环境
#   - alfworld 依赖: gym, nltk, spacy 等
#   - spacy 自动带出: thinc, preshed, cymem, murmurhash, blis, wasabi,
#     srsly, catalogue, confection, typer, langcodes ...
# ============================================================
log "Step 6: 安装 Agent 环境 (gymnasium, alfworld, scienceworld, 等)"
pip install \
    "gymnasium==0.29.1" \
    "stable-baselines3==2.6.0" \
    "alfworld==0.4.2" \
    "scienceworld==1.2.3" \
    "jericho==3.3.1" \
    "textworld==1.7.0"

# 下载 spacy 模型 (alfworld 需要)
log "Step 6b: 下载 spacy 模型"
python -m spacy download en_core_web_sm

# alfworld 额外数据
log "Step 6c: 下载 ALFWorld 游戏文件（如果可用）"
alfworld-download -f 2>/dev/null || warn "alfworld-download 不可用, 可手动运行"

# ============================================================
# Step 7: 安装搜索/检索相关包（search 实验用）
# ============================================================
log "Step 7: 安装检索相关包"
pip install \
    "pyserini==0.17.0" \
    "rank-bm25==0.2.2" \
    "faiss==1.9.0" \
    "nmslib==2.1.2"

# ============================================================
# Step 8: 安装其他工具包
#   - 这些包部分会被上面的包自动带出，但为了版本锁定显式安装
# ============================================================
log "Step 8: 安装其他工具包"
pip install \
    "hydra-core==1.3.2" \
    "wandb==0.27.1" \
    "openai==2.33.0" \
    "gradio==4.26.0" \
    "outlines==0.1.11" \
    "modelscope==1.37.0" \
    "matplotlib" \
    "scikit-learn==1.8.0" \
    "tiktoken==0.12.0" \
    "einops==0.8.2" \
    "sentencepiece==0.2.1" \
    "codetiming" \
    "pylatexenc" \
    "qwen-vl-utils[decord]" \
    "pybind11" \
    "nltk==3.9.4"

# ============================================================
# Step 9: 安装本项目 (verl)
# ============================================================
log "Step 9: 安装本项目 verl (editable mode)"
cd "${PROJECT_DIR}"
pip install -e .

# ============================================================
# Step 10: 开发/测试工具
# ============================================================
log "Step 10: 安装开发测试工具"
pip install \
    "pytest==8.3.5" \
    "ruff==0.15.12" \
    "py-spy"

# ============================================================
# 验证安装
# ============================================================
log "验证安装..."
echo "---"
python --version
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import vllm; print(f'vLLM {vllm.__version__}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
python -c "import ray; print(f'Ray {ray.__version__}')"
python -c "import flash_attn; print(f'FlashAttention {flash_attn.__version__}')"
echo "---"
log "${GREEN}verl_train 环境安装完成！${NC}"
log "使用: conda activate ${ENV_NAME}"
