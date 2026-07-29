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

# /diskpool/home/xuxz/miniconda3/envs/verl_train/bin/python --version 查看python版本
PYTHON_VERSION="3.11"
TORCH_VERSION="2.6.0"
TORCHVISION_VERSION="0.21.0"
CUDA_VERSION="cu124"
VLLM_VERSION="0.8.5"
FLASH_ATTN_VERSION="2.7.4.post1"
# 脚本位于 <项目根>/train_scrips/install/，需向上两级到达项目根
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

# ============================================================
# Step 0: 创建 conda 环境
# ============================================================
ENV_NAME="verl_train_2"  # 可通过命令行参数指定环境名，默认 verl_train

eval "$(conda shell.bash hook)"

# 检查环境是否已存在
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    warn "conda 环境 '${ENV_NAME}' 已存在，跳过创建，直接激活"
    conda activate "${ENV_NAME}"
else
    log "Step 0: 创建 conda 环境 '${ENV_NAME}' (Python ${PYTHON_VERSION})"
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    conda activate "${ENV_NAME}"
fi

# 升级 pip
pip install --upgrade pip
# 固定 setuptools 版本，避免新版 setuptools 严格执行 setup.py 中的约束导致 tensordict 被降级
pip install "setuptools==68.0.0"

# ============================================================
# Step 1: 安装 PyTorch 全家桶（必须先装，后续所有包依赖它）
# ============================================================
log "Step 1: 安装 PyTorch ${TORCH_VERSION} (CUDA 12.4)"
pip3 install \
    torch=="${TORCH_VERSION}" \
    torchvision=="${TORCHVISION_VERSION}" \
    torchaudio=="${TORCH_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"

# ============================================================
# Step 2: 安装 FlashAttention（编译耗时，需 --no-build-isolation）
# --no-build-isolation 意味着 pip 不会自动安装构建依赖，
# 因此必须先手动安装 psutil, ninja（flash-attn 的 setup.py 依赖）
# ============================================================
log "Step 2a: 安装 flash-attn 构建依赖 (psutil, ninja)"
pip install psutil ninja

log "Step 2: 安装 FlashAttention ${FLASH_ATTN_VERSION}"
# flash-attn 的 setup.py 会下载预编译 wheel，pip 在移动 wheel 时
# 需要 TMPDIR 与缓存目录在同一文件系统，否则报 cross-device link 错误。
# 将 TMPDIR 设到 /diskpool 下避免跨设备。
export TMPDIR="/diskpool/home/xuxz/.cache/tmp"
mkdir -p "${TMPDIR}"
# 直接安装预编译 wheel，避免本地编译
pip install flash-attn=="${FLASH_ATTN_VERSION}" --no-build-isolation

# FlashInfer (vLLM 的 prefix-prefilling 可选加速)
# 注意: 该包在当前环境中未实际安装，vLLM 0.8.5 不强制依赖，设为可选
log "Step 2b: 安装 FlashInfer (可选)"
pip install flashinfer-python --find-links https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python || warn "FlashInfer 安装失败, 跳过 (vLLM 不强制依赖)"

# xformers (加速 attention)
pip install "xformers<0.0.30"

# ============================================================
# Step 3: 安装 vLLM（会连带安装大量依赖）
#   - 自动带出: numpy, requests, aiohttp, pydantic, starlette,
#     fastapi, uvicorn, pyzmq, msgpack, orjson, msgspec,
#     safetensors, tokenizers, huggingface_hub, filelock, psutil ...
# ============================================================
log "Step 3: 安装 vLLM ${VLLM_VERSION}"
# 先固定 numpy<2 和 protobuf<7，避免 vLLM 依赖解析安装不兼容的新版本
pip install "numpy==1.26.4" "protobuf==6.33.6"
pip install "vllm==${VLLM_VERSION}"
# vLLM 可能升级 numpy/protobuf，强制恢复
pip install "numpy==1.26.4" "protobuf==6.33.6"

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
    "textworld==1.7.0" \
    "gym==0.24.0"

# 下载 spacy 模型 (alfworld 需要)
# 注意: spacy CLI (python -m spacy download) 在 typer 0.9.4 + click 8.3.3 下会报错
#       "Secondary flag is not valid for non-boolean flag"，改用 pip 直接安装 wheel
# log "Step 6b: 下载 spacy 模型"
# pip install "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" || warn "en_core_web_sm 下载失败, 可稍后手动安装"
# pip install "en_core_web_lg @ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl" || warn "en_core_web_lg 下载失败, 可稍后手动安装"

# alfworld 额外数据
# log "Step 6c: 下载 ALFWorld 游戏文件（如果可用）"
# alfworld-download -f 2>/dev/null || warn "alfworld-download 不可用, 可手动运行"

# ============================================================
# Step 7: 安装搜索/检索相关包（search 实验用）
#   注意: faiss 1.9.0 在 PyPI 上的包名是 faiss-cpu（不是 faiss），
#         原环境通过 conda-forge 安装了 faiss=1.9.0，
#         这里用 pip 安装 faiss-cpu，功能等价。
# ============================================================
log "Step 7: 安装检索相关包"
pip install \
    "pyserini==0.17.0" \
    "rank-bm25==0.2.2" \
    "faiss-cpu==1.9.0" \
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
    "matplotlib==3.10.9" \
    "scikit-learn==1.8.0" \
    "tiktoken==0.12.0" \
    "einops==0.8.2" \
    "sentencepiece==0.2.1" \
    "codetiming" \
    "pylatexenc==2.10" \
    "qwen-vl-utils[decord]" \
    "pybind11" \
    "nltk==3.9.4" \
    "mlflow==3.13.0" \
    "tensorboardX==2.6.5" \
    "sentence-transformers==5.6.0" \
    "selenium==4.2.0" \
    "Flask==2.1.2" \
    "flask-cors==6.0.2"
# 固定 numpy/protobuf，防止上面安装的包升级了它们
pip install "numpy==1.26.4" "protobuf==6.33.6"

# ============================================================
# Step 9: 安装本项目 (verl)
#   注意: setup.py 中声明了 tensordict<=0.6.2，会降级已安装的 0.12.4。
#   使用 --no-deps 避免依赖解析覆盖前面已固定的版本。
#   verl 的所有依赖在前面的步骤中已显式安装。
# ============================================================
log "Step 9: 安装本项目 verl (editable mode, --no-deps)"
cd "${PROJECT_DIR}"
pip install -e . --no-deps
# 确保 tensordict 未被降级
pip install "tensordict==0.12.4" 2>/dev/null || warn "tensordict==0.12.4 安装失败"

# ============================================================
# Step 10: 开发/测试工具
# ============================================================
log "Step 10: 安装开发测试工具"
pip install \
    "pytest==8.3.5" \
    "ruff==0.15.12" \
    "py-spy"

# ============================================================
# Step 11: 版本对齐
#   前面的步骤中，vLLM / flashinfer / spacy 等包的传递依赖
#   会被 pip 解析为最新版本，与原环境不一致。
#   这里统一强制对齐到原环境 verl_train 的版本。
# ============================================================
log "Step 11: 版本对齐（强制固定所有传递依赖版本）"
pip install --no-deps \
    "aiohappyeyeballs==2.6.1" \
    "aiohttp==3.13.5" \
    "alembic==1.18.4" \
    "annotated-doc==0.0.4" \
    "annotated-types==0.7.0" \
    "anyio==4.13.0" \
    "av==17.0.1" \
    "blake3==1.0.8" \
    "cachetools==7.1.0" \
    "certifi==2026.4.22" \
    "cffi==2.0.0" \
    "charset-normalizer==2.0.12" \
    "click==8.3.3" \
    "confection==0.1.5" \
    "cryptography==47.0.0" \
    "cuda-pathfinder==1.5.4" \
    "cupy-cuda12x==14.0.1" \
    "Cython==3.2.4" \
    "databricks-sdk==0.114.0" \
    "distlib==0.4.1" \
    "docker==7.1.0" \
    "fastapi==0.136.1" \
    "fastapi-cli==0.0.24" \
    "fastapi-cloud-cli==0.17.1" \
    "filelock==3.25.2" \
    "fonttools==4.62.1" \
    "gguf==0.18.0" \
    "GitPython==3.1.50" \
    "google-api-core==2.31.0" \
    "googleapis-common-protos==1.74.0" \
    "google-auth==2.53.0" \
    "greenlet==3.5.1" \
    "grpcio==1.80.0" \
    "hf-xet==1.4.3" \
    "httptools==0.7.1" \
    "huey==3.0.1" \
    "idna==3.13" \
    "jiter==0.14.0" \
    "lightgbm==4.6.0" \
    "markdown-it-py==4.0.0" \
    "mistral_common==1.11.1" \
    "more-itertools==11.0.2" \
    "msgpack==1.1.2" \
    "narwhals==2.20.0" \
    "omegaconf==2.3.0" \
    "onnxruntime==1.25.1" \
    "opencv-python-headless==4.13.0.92" \
    "opentelemetry-api==1.42.1" \
    "opentelemetry-exporter-prometheus==0.63b1" \
    "opentelemetry-proto==1.42.1" \
    "opentelemetry-sdk==1.42.1" \
    "opentelemetry-semantic-conventions==0.63b1" \
    "orjson==3.11.8" \
    "pandas==2.2.3" \
    "platformdirs==4.10.0" \
    "prettytable==3.17.0" \
    "prometheus_client==0.25.0" \
    "prometheus-fastapi-instrumentator==7.1.0" \
    "prompt_toolkit==3.0.52" \
    "propcache==0.4.1" \
    "proto-plus==1.28.0" \
    "pyasn1==0.6.3" \
    "pydantic==2.13.3" \
    "pydantic_core==2.46.3" \
    "pydantic-settings==2.14.0" \
    "pyOpenSSL==26.1.0" \
    "python-discovery==1.4.0" \
    "python-multipart==0.0.27" \
    "pytz==2026.1.post1" \
    "pyvers==0.2.2" \
    "PyYAML==6.0.2" \
    "regex==2026.4.4" \
    "requests==2.33.1" \
    "rich-toolkit==0.19.7" \
    "rignore==0.7.6" \
    "rpds-py==0.30.0" \
    "safetensors==0.7.0" \
    "sentry-sdk==2.58.0" \
    "SQLAlchemy==2.0.50" \
    "starlette==0.52.1" \
    "thinc==8.2.5" \
    "tqdm==4.67.3" \
    "typer==0.9.4" \
    "tzdata==2026.2" \
    "uvicorn==0.46.0" \
    "virtualenv==21.4.2" \
    "watchfiles==1.1.1" \
    "wcwidth==0.7.0" \
    "weasel==0.3.4" \
    "Werkzeug==2.1.0" \
    "wrapt==2.1.2" \
    "xxhash==3.7.0" \
    "yarl==1.23.0" \
    "zipp==3.23.1"

# 安装原环境有但新环境缺失的包
pip install \
    "backports.zstd==1.3.0" \
    "beautifulsoup4==4.11.1" \
    "Brotli==1.2.0" \
    "cbor2==6.1.3" \
    "cleantext==1.1.4" \
    "colorama==0.4.6" \
    "commonmark==0.9.1" \
    "env==0.1.0" \
    "exceptiongroup==1.3.1" \
    "frozendict==2.4.7" \
    "gdown==6.0.0" \
    "h2==4.3.0" \
    "hpack==4.1.0" \
    "hyperframe==6.1.0" \
    "langcodes==3.5.1" \
    "Levenshtein==0.27.3" \
    "nvidia-cufile-cu12==1.13.1.3" \
    "openai-harmony==0.0.8" \
    "pybase64==1.4.3" \
    "python-Levenshtein==0.27.3" \
    "RapidFuzz==3.14.5" \
    "requests-mock==1.12.1" \
    "setproctitle==1.3.7" \
    "smart-open==6.4.0" \
    "soundfile==0.14.0" \
    "soupsieve==2.8.3" \
    "soxr==1.1.0" \
    "thefuzz==0.19.0" \
    "torchdata==0.11.0" \
    "train==0.0.5" \
    "typer-slim==0.24.0" \
    "ujson==5.12.0"

# 移除新环境中多余的包（flashinfer 的依赖，原环境没有）
pip uninstall -y flashinfer-python apache-tvm-ffi cuda-tile nccl4py nvidia-cudnn-frontend nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-cu12 nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-core nvidia-ml-py tabulate 2>/dev/null || warn "部分多余包卸载失败, 可手动处理"
# 移除 detect-installer（fastapi-cloud-cli 新版带入，原环境无）
pip uninstall -y detect-installer 2>/dev/null || true
# 移除 cuda-python 及其依赖（flashinfer 残留依赖，原环境无）
pip uninstall -y cuda-python cuda-bindings cuda-core nvidia-cuda-nvdisasm 2>/dev/null || true
# typer 需要用 --no-deps 强制降级，否则 spacy 等包会拉回新版
pip install --no-deps "typer==0.9.4"

# 最终再固定一次核心包
pip install --no-deps "numpy==1.26.4" "protobuf==6.33.6" "tensordict==0.12.4"

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
