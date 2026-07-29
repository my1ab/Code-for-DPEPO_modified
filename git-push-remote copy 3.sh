#!/bin/bash
# Git 提交并推送到远端仓库脚本
# 自动排除大文件/目录、清理历史大文件、强制推送

set -e

# ======================== 配置 ========================
REMOTE_URL="git@github.com:my1ab/Code-for-DPEPO_modified.git"
REPO_NAME="Code-for-DPEPO_modified"
TARGET_BRANCH="main"

# 需要排除的路径（不提交、不推送）
EXCLUDE_PATHS=(
    2gpu_emb_search_080 2gpu_emb_search_090 2gpu_emb_search_noemb
    3emb_model_bs1 3emb_model_bs2 3emb_model_bs4 3emb_model_bs4_webshop 3emb_model_resume
    交接文档 解析文件 修改日志
    webshop_para_full_result webshop_checkpoint_para webshop_checkpoint file_sft_search
    test_bert_similarity/学习轨迹 test_bert_similarity/验证轨迹 test_bert_similarity/验证轨迹search
    test_bert_similarity/file_for_sft_webshop test_bert_similarity/log test_bert_similarity/models
    "*.pt" "*.ckpt" "*.safetensors" "*.tar.gz" "__pycache__/" "*.pyc" "*.pyo"
)

SIZE_LIMIT=104857600  # 100MB (GitHub 硬限制)

# ======================== 函数 ========================

# 获取文件大小（兼容 Linux/macOS），失败返回 0
get_file_size() {
    stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null || echo 0
}

# ======================== 初始化 ========================

echo "=== 初始化 ==="
git config user.name "my1ab"
git config user.email "my1ab@example.com"

# 设置远程仓库
if ! git remote get-url "$REPO_NAME" &>/dev/null; then
    git remote add "$REPO_NAME" "$REMOTE_URL"
else
    git remote set-url "$REPO_NAME" "$REMOTE_URL"
fi
git remote -v

# 确保目标分支存在
git show-ref --verify --quiet "refs/heads/$TARGET_BRANCH" || git branch "$TARGET_BRANCH"

# ======================== 清理历史大文件 ========================

echo ""
echo "=== 清理历史提交中的大文件（>${SIZE_LIMIT} bytes）==="
if command -v git-filter-repo &>/dev/null; then
    LARGE_COUNT=$(git rev-list --objects --all 2>/dev/null \
        | git cat-file --batch-check='%(objecttype) %(objectsize)' 2>/dev/null \
        | awk -v lim="$SIZE_LIMIT" '$1=="blob" && $2 > lim' | wc -l)
    if [ "$LARGE_COUNT" -gt 0 ]; then
        echo "⚠️  历史中有 $LARGE_COUNT 个大文件，使用 filter-repo 清理..."
        git branch backup-before-filterrepo 2>/dev/null || true
        STASH_DONE=false
        if ! git diff --quiet || ! git diff --cached --quiet; then
            git stash push -m "auto-stash-before-filterrepo" 2>&1
            STASH_DONE=true
        fi
        git filter-repo --strip-blobs-bigger-than 100M --force 2>&1
        rm -f .git/filter-repo/already_ran
        git remote add "$REPO_NAME" "$REMOTE_URL" 2>/dev/null || true
        [ "$STASH_DONE" = true ] && git stash pop 2>&1 || true
        echo "✅ 清理完成（备份: backup-before-filterrepo）"
    else
        echo "✅ 历史中无大文件"
    fi
else
    echo "⚠️  git-filter-repo 未安装，跳过（安装: pip install git-filter-repo）"
fi

# ======================== 暂存文件 ========================

echo ""
echo "=== 暂存文件（排除 EXCLUDE_PATHS）==="

# 1. 清空暂存区
git reset HEAD -- .

# 2. 从索引中移除已跟踪的排除路径
for path in "${EXCLUDE_PATHS[@]}"; do
    git rm --cached -r "$path" 2>/dev/null || true
done

# 3. 添加所有文件（排除指定路径）
GIT_ADD_ARGS=("-A")
for path in "${EXCLUDE_PATHS[@]}"; do
    GIT_ADD_ARGS+=(":(exclude)${path}")
done
git add "${GIT_ADD_ARGS[@]}"

# 4. 单独处理 coldstart_test 子目录（只保留根目录文件）
if [ -d "coldstart_test" ]; then
    for subdir in coldstart_test/*/; do
        [ -d "$subdir" ] && git rm --cached -r "$subdir" 2>/dev/null || true
    done
fi

# 5. 移除暂存区中仍超过限制的大文件
LARGE_FILES=""
while IFS= read -r f; do
    [ -f "$f" ] || continue
    size=$(get_file_size "$f")
    if [ "$size" -gt "$SIZE_LIMIT" ] 2>/dev/null; then
        printf "  ⚠️  %.1f MB\t%s（已移除）\n" "$(echo "scale=1; $size/1048576" | bc)" "$f"
        git rm --cached "$f" 2>/dev/null || true
        LARGE_FILES=1
    fi
done < <(git diff --cached --name-only)
[ -n "$LARGE_FILES" ] && echo "⚠️  已从暂存区移除大文件" || echo "✅ 暂存区无大文件"

# ======================== 提交并推送 ========================

echo ""
echo "=== 提交并推送 ==="
if git diff --cached --quiet; then
    echo "暂存区为空，直接推送当前分支"
else
    git commit -m "Update project files"
fi
git push "$REPO_NAME" HEAD:"$TARGET_BRANCH" -f

echo ""
echo "=== 操作完成 ==="
