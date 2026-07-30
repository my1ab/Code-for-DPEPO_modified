#!/bin/bash
# ============================================================
# 数据集下载 + 检索索引准备 + 检索服务器启动 一键脚本
# 对应文档: 使用文档/quick_start.md 第 1.4 章
# 运行环境: verl_train_2 (由 install_verl_train.sh 一键安装)
# ============================================================
# 使用方式 (在项目根目录 /diskpool/home/xuxz/Code-for-DPEPO 下执行):
#   bash train_scrips/install/setup_data_verl_train.sh             # 完整数据准备 (WebShop + Search)
#   bash train_scrips/install/setup_data_verl_train.sh webshop     # 仅 WebShop 数据集下载 + 商品索引
#   bash train_scrips/install/setup_data_verl_train.sh search      # 仅 Search 预处理 + 检索索引下载
#   bash train_scrips/install/setup_data_verl_train.sh retriever   # 仅启动检索服务器 (训练/验证前启动)
# ============================================================

set -euo pipefail

# ---------- 颜色输出 ----------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; }

ENV_NAME="verl_train_2"          # 与 install_verl_train.sh 创建的环境一致
# 手动设置：用户目录 + 项目名
PROJECT_NAME="Code-for-DPEPO"    # 手动设置项目名
HOME_DIR="/diskpool/home/xuxz"
#  以下根据 用户目录 + 项目名 自动设置
PROJECT_DIR="${HOME_DIR}/${PROJECT_NAME}"
DATA_DIR="${HOME_DIR}/data"
WEBSHOP_DIR="${PROJECT_DIR}/agent_system/environments/env_package/webshop/webshop"
SEARCH_ENGINE_DIR="${WEBSHOP_DIR}/search_engine"
SEARCHR1_DIR="${DATA_DIR}/searchR1"

MODE="${1:-all}"                 # 默认执行完整数据准备 (all)

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# ============================================================
# 1.4.1 WebShop: 下载数据集
# ============================================================
download_webshop_data() {
    log "1.4.1 WebShop: 下载数据集 (ModelScope)"
    if [ -f "${DATA_DIR}/items_shuffle.json" ]; then
        warn "已检测到 ${DATA_DIR}/items_shuffle.json，WebShop 数据已存在，跳过下载"
        return
    fi
    mkdir -p "${DATA_DIR}"
    cd "${DATA_DIR}"
    # 从 ModelScope 定向下载 dataset/webshop 子目录下的 5 个数据文件
    modelscope download --model afordb/sft_result_webshop_and_search_based_on_qwen \
        --local_dir . \
        dataset/webshop/items_human_ins.json \
        dataset/webshop/items_ins_v2.json \
        dataset/webshop/items_ins_v2_1000.json \
        dataset/webshop/items_shuffle.json \
        dataset/webshop/items_shuffle_1000.json

    # 下载后文件在 $DATA_DIR/dataset/webshop/ 下，移动到 $DATA_DIR 根目录
    if [ -d "${DATA_DIR}/dataset/webshop" ]; then
        mv "${DATA_DIR}"/dataset/webshop/*.json "${DATA_DIR}"/
        rm -rf "${DATA_DIR}/dataset"
    fi
    log "WebShop 数据下载完成: ${DATA_DIR} 下应包含 5 个 json 文件"
}

# ============================================================
# 1.4.1 WebShop: 建立商品搜索索引 (Lucene)
# ============================================================
build_webshop_index() {
    log "1.4.1 WebShop: 建立商品搜索索引 (convert + run_indexing)"
    cd "${SEARCH_ENGINE_DIR}"

    # Step 0: 创建 convert 输出目录 (resources*/documents.jsonl 写入目录)
    mkdir -p resources resources_100 resources_1k resources_100k

    # Step 1: convert 商品 JSON → Lucene 文档集 (读取 ~/data 下数据)
    python convert_product_file_format.py

    # Step 2: 建立 Lucene 索引 (resources* → indexes*; pyserini 会自动创建索引目录, mkdir 可省略)
    mkdir -p indexes
    ./run_indexing.sh

    cd "${WEBSHOP_DIR}"
    log "WebShop 商品搜索索引建立完成"
}

# ============================================================
# 1.4.2 Search Step 1: 预处理数据集
# ============================================================
preprocess_search_data() {
    log "1.4.2 Search Step 1: 预处理数据集 (输出 ~/data/searchR1_processed_direct)"
    cd "${PROJECT_DIR}"

    # 使用 HF 镜像加速下载
    export HF_ENDPOINT=https://hf-mirror.com

    python examples/data_preprocess/preprocess_search_r1_dataset.py
}

# ============================================================
# 1.4.2 Search Step 2: 下载检索索引 + 合并
# ============================================================
download_search_index() {
    log "1.4.2 Search Step 2: 下载检索索引 (~/data/searchR1)"
    cd "${PROJECT_DIR}"
    mkdir -p "${SEARCHR1_DIR}"

    python examples/search/searchr1_download.py --local_dir "${SEARCHR1_DIR}"

    # 合并分片 part_* → e5_Flat.index
    if [ -f "${SEARCHR1_DIR}/e5_Flat.index" ]; then
        warn "e5_Flat.index 已存在，跳过分片合并"
    else
        cat "${SEARCHR1_DIR}"/part_* > "${SEARCHR1_DIR}/e5_Flat.index"
    fi

    # 解压 Wikipedia 语料库
    if [ -f "${SEARCHR1_DIR}/wiki-18.jsonl" ]; then
        warn "wiki-18.jsonl 已存在，跳过解压"
    else
        gzip -d "${SEARCHR1_DIR}/wiki-18.jsonl.gz"
    fi
    log "Search 检索索引下载完成"
}

# ============================================================
# 1.4.2 Search Step 3: 启动检索服务器 (训练/验证前启动)
# ============================================================
launch_retriever() {
    log "1.4.2 Search Step 3: 启动检索服务器 (faiss + MMAP 分片加载, PORT=8010)"
    cd "${PROJECT_DIR}"

    # faiss 索引 + MMAP 分片加载；脚本内部自动后台运行并等待就绪
    bash examples/search/retriever/retrieval_launch_faiss.sh
}

# ============================================================
# 主流程
# ============================================================
case "${MODE}" in
    all)
        log "开始完整数据准备 (WebShop + Search)"
        download_webshop_data
        build_webshop_index
        preprocess_search_data
        download_search_index
        log "数据准备全部完成，请参照文档 1.4.3 确认 ~/data/ 目录结构"
        log "训练/验证前请先启动检索服务器: bash train_scrips/install/setup_data_verl_train.sh retriever"
        ;;
    webshop)
        download_webshop_data
        build_webshop_index
        ;;
    search)
        preprocess_search_data
        download_search_index
        ;;
    retriever)
        launch_retriever
        ;;
    *)
        err "未知参数: ${MODE}"
        echo "用法: bash ${0} [all|webshop|search|retriever]"
        echo "  (无参数)  = 完整数据准备 (WebShop 数据+索引, Search 预处理+索引下载)"
        echo "  webshop   = 仅 WebShop 数据集下载 + 商品搜索索引"
        echo "  search    = 仅 Search 数据预处理 + 检索索引下载"
        echo "  retriever = 仅启动检索服务器 (训练/验证前启动)"
        exit 1
        ;;
esac
