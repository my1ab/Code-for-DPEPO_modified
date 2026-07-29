#!/bin/bash


# REMOTE_URL="git@github.com:my1ab/Code-for-DPEPO_modified.git"  # SSH方式
# my1ab
# 
# REMOTE_URL="https://github.com/my1ab/Code-for-DPEPO_modified.git"
REMOTE_URL="git@github.com:my1ab/Code-for-DPEPO_modified.git"
REPO_NAME="Code-for-DPEPO_modified"
TARGET_BRANCH="main"
if ! git remote get-url $REPO_NAME &>/dev/null; then
    echo "远程仓库 $REPO_NAME 不存在，正在添加..."
    git remote add $REPO_NAME "$REMOTE_URL"
else
    echo "远程仓库 $REPO_NAME 已存在，更新URL以包含token认证"
    git remote set-url $REPO_NAME "$REMOTE_URL"
fi
git remote -v


# Git 日常提交和推送到远端仓库脚本

set -e

echo "=== 设置 Git 用户信息 ==="
git config user.name "my1ab"
git config user.email "my1ab@example.com"

echo ""
echo "=== 检查当前目录 ==="
pwd

echo ""
echo "=== 检查 Git 状态 ==="
git status

echo ""
echo "=== 检查并创建目标分支 ==="
# 手动选择目标分支
TARGET_BRANCH="main"
# TARGET_BRANCH="my-verl"

if git show-ref --verify --quiet "refs/heads/$TARGET_BRANCH"; then
    echo "分支 $TARGET_BRANCH 已存在"
else
    echo "分支 $TARGET_BRANCH 不存在，创建该分支"
    git branch $TARGET_BRANCH
fi

echo ""
echo "=== 定义需要排除的路径 ==="
EXCLUDE_PATHS=(
    # 排除coldstart_test下的所有子文件夹
    2gpu_emb_search_080
    2gpu_emb_search_090
    2gpu_emb_search_noemb
    3emb_model_bs1
    3emb_model_bs2
    3emb_model_bs4
    3emb_model_bs4_webshop
    3emb_model_resume
    交接文档
    解析文件
    修改日志
    webshop_para_full_result
    webshop_checkpoint_para
    webshop_checkpoint
    file_sft_search
    models
    test_bert_similarity/学习轨迹
    test_bert_similarity/验证轨迹
    test_bert_similarity/验证轨迹search
    test_bert_similarity/file_for_sft_webshop
    test_bert_similarity/log
    test_bert_similarity/models
    "*.pt"
    "*.ckpt"
    "*.safetensors"
    "*.tar.gz"
    "__pycache__/"
    "*.pyc"
    "*.pyo"
)

echo ""
echo "=== 检查本地是否有未推送且含大文件的提交，如有则用 filter-repo 清理 ==="
REMOTE_REF="${REPO_NAME}/${TARGET_BRANCH}"
REMOTE_COMMIT=$(git rev-parse "$REMOTE_REF" 2>/dev/null || echo "")
if [ -n "$REMOTE_COMMIT" ]; then
    # 扫描未推送提交中是否有超过 100MB 的文件
    BAD_LARGE=$(git rev-list --objects "$REMOTE_COMMIT"..HEAD 2>/dev/null \
        | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' \
        | awk '$1=="blob" && $2 > 104857600 {print $3}')
    # 扫描未推送提交中是否有 LFS pointer 文件（pointer 本身很小，按大小检测不到）
    LFS_FILES=$(git grep -l '^version https://git-lfs' $(git rev-list "$REMOTE_COMMIT"..HEAD) 2>/dev/null \
        | sed 's/^[^:]*://' | sort -u)
    if [ -n "$BAD_LARGE" ] || [ -n "$LFS_FILES" ]; then
        [ -n "$BAD_LARGE" ] && echo "⚠️  检测到超过 100MB 的大文件：" && echo "$BAD_LARGE" | head -10
        [ -n "$LFS_FILES" ] && echo "⚠️  检测到 LFS pointer 文件：" && echo "$LFS_FILES" | head -10
        if command -v git-filter-repo &>/dev/null; then
            echo "  使用 git-filter-repo 清理大文件..."
            git branch backup-before-filterrepo 2>/dev/null || true
            # stash 未提交的修改（filter-repo 要求干净工作区）
            STASH_DONE=false
            if ! git diff --quiet || ! git diff --cached --quiet; then
                git stash push -m "auto-stash-before-filterrepo" 2>&1 && STASH_DONE=true
            fi
            # 收集需要从历史中移除的路径：大文件 + LFS pointer 文件
            CLEANUP_ARGS=()
            [ -s <(echo "$BAD_LARGE") ] && while IFS= read -r f; do
                [ -n "$f" ] && CLEANUP_ARGS+=(--path "$f")
            done <<< "$BAD_LARGE"
            [ -s <(echo "$LFS_FILES") ] && while IFS= read -r f; do
                [ -n "$f" ] && CLEANUP_ARGS+=(--path "$f")
            done <<< "$LFS_FILES"
            if [ ${#CLEANUP_ARGS[@]} -gt 0 ]; then
                git filter-repo "${CLEANUP_ARGS[@]}" --invert-paths --force 2>&1
                rm -f .git/filter-repo/already_ran
            fi
            # 兜底：清理所有超过 100MB 的 blob
            git filter-repo --strip-blobs-bigger-than 100M --force 2>&1
            rm -f .git/filter-repo/already_ran
            # filter-repo 会移除远程引用，重新添加
            git remote add $REPO_NAME "$REMOTE_URL" 2>/dev/null || true
            REMOTE_COMMIT=$(git rev-parse "$REMOTE_REF" 2>/dev/null || echo "")
            [ "$STASH_DONE" = true ] && git stash pop 2>&1 || true
            echo "✅ 历史清理完成"
        else
            echo "  git-filter-repo 未安装，回退到 git reset --soft"
            echo "  安装方式: pip install git-filter-repo"
            git reset --soft "$REMOTE_COMMIT"
        fi
    else
        echo "✅ 未推送的提交中没有大文件"
    fi
else
    echo "未找到远端提交，跳过大文件检查"
fi

echo ""
echo "=== 先清空所有暂存区，保证干净的状态 ==="
git reset HEAD -- .  # 取消所有暂存的文件  但git add可以覆盖这个操作
git status

echo ""
echo "=== 从 Git 索引中移除 EXCLUDE_PATHS 中已跟踪的文件 ==="
# git add :(exclude) 只能阻止新文件被添加，不会移除已跟踪的文件
for path in "${EXCLUDE_PATHS[@]}"; do
    git rm --cached -r "$path" 2>/dev/null || true
done

echo ""
echo "=== 添加所有文件（自动排除 EXCLUDE_PATHS 中的路径）==="
# 使用 Git pathspec magic（:(exclude) 长格式）在 git add 时直接排除指定路径  不需要添加后删除
GIT_ADD_ARGS=("-A")
for path in "${EXCLUDE_PATHS[@]}"; do
    GIT_ADD_ARGS+=(":(exclude)${path}")
done
git add "${GIT_ADD_ARGS[@]}"
echo "已执行: git add -A 并排除 ${#EXCLUDE_PATHS[@]} 个路径模式"

echo ""
echo "=== 验证暂存区中是否还有超过 100MB 的大文件 ==="
LARGE_FILES=$(git diff --cached --name-only | while read f; do
    [ -f "$f" ] || continue
    size=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
    if [ "$size" -gt 104857600 ] 2>/dev/null; then
        printf "%d MB\t%s\n" $((size/1048576)) "$f"
    fi
done)
if [ -n "$LARGE_FILES" ]; then
    echo "⚠️  暂存区中仍有超过 100MB 的文件："
    echo "$LARGE_FILES"
    echo "$LARGE_FILES" | while read line; do
        git rm --cached "${line#*\t}" 2>/dev/null || true
    done
    echo "已移除大文件"
else
    echo "✅ 暂存区中没有超过 100MB 的文件"
fi

# 单独处理coldstart_test下的所有子文件夹，确保只保留coldstart_test根目录下的文件
if [ -d "coldstart_test" ]; then
    echo "排除coldstart_test下的所有子文件夹:"
    # 查找coldstart_test下的所有一级子目录
    for subdir in coldstart_test/*/; do
        if [ -d "$subdir" ]; then
            echo "  排除子目录: $subdir"
            git reset HEAD "$subdir" 2>/dev/null || true
            git rm --cached -r "$subdir" 2>/dev/null || true
        fi
    done
fi

echo ""
echo "=== 检查暂存状态 ==="
git status

echo ""
echo "=== 提交更改 ==="
if git diff --cached --quiet; then
    echo "暂存区为空，无更改可提交"
    # 即使没有新的提交，也尝试推送当前分支到远端，确保远端仓库同步
    echo ""
    echo "=== 尝试推送当前分支到远端仓库 $TARGET_BRANCH 分支，确保同步 ==="
    git push $REPO_NAME HEAD:$TARGET_BRANCH -f
else
    git commit -m "Update project files"
    echo ""
    echo "=== 推送到远端仓库 $TARGET_BRANCH 分支 ==="
    # 格式: git push <远程名> <来源>:<目标> -f
    git push $REPO_NAME HEAD:$TARGET_BRANCH -f
fi



echo ""
echo "=== 操作完成 ==="