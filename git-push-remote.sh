#!/bin/bash


REMOTE_URL="git@github.com:my1ab/Code-for-DPEPO_modified.git"  # SSH方式
if ! git remote get-url Code-for-DPEPO_modified &>/dev/null; then
    echo "远程仓库 Code-for-DPEPO_modified 不存在，正在添加..."
    git remote add Code-for-DPEPO_modified "$REMOTE_URL"
else
    echo "远程仓库 Code-for-DPEPO_modified 已存在，更新URL以包含token认证"
    git remote set-url Code-for-DPEPO_modified "$REMOTE_URL"
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
echo "=== 先清空所有暂存区，保证干净的状态 ==="
git reset HEAD -- .  # 取消所有暂存的文件
git status

echo ""
echo "=== 正在添加所有已追踪和新文件 ==="
git add -A  # 添加所有修改、新增和删除的文件，等同于git add -u + git add .

echo ""
echo "=== 定义需要排除的路径 ==="
EXCLUDE_PATHS=(
    "coldstart_test/*/"  # 排除coldstart_test下的所有子文件夹
    "coldstart_result_webshop/"
    "coldstart_test_new/model_hislen8_result_v2/"

    "*.pt"
    "*.ckpt"
    "*.safetensors"
    "*.tar.gz"
    "__pycache__/"
    "*.pyc"
    "*.pyo"
)

echo ""
echo "=== 从暂存区排除不需要的路径 ==="
# 首先处理普通排除路径
for path in "${EXCLUDE_PATHS[@]}"; do
    echo "排除: $path"
    # 使用递归方式移除整个目录
    git reset HEAD "$path" 2>/dev/null || true
    git rm --cached -r "$path" 2>/dev/null || true
done

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


# echo ""
# echo "=== 暂存区大小统计 ==="
# git diff --cached --stat

# echo ""
# echo "=== 暂存区总大小 ==="
# TOTAL_SIZE=$(git diff --cached --numstat | awk '{sum+=$1+$2} END {print sum/1024/1024}')
# echo "总大小: $TOTAL_SIZE MB"

echo ""
echo "=== 提交更改 ==="
if git diff --cached --quiet; then
    echo "暂存区为空，无更改可提交"
    # 即使没有新的提交，也尝试推送当前分支到远端，确保远端仓库同步
    echo ""
    echo "=== 尝试推送当前分支到远端仓库 $TARGET_BRANCH 分支，确保同步 ==="
    git push Code-for-DPEPO_modified HEAD:$TARGET_BRANCH -f
else
    git commit -m "Update project files"
    echo ""
    echo "=== 推送到远端仓库 $TARGET_BRANCH 分支 ==="
    # 格式: git push <远程名> <来源>:<目标> -f
    git push Code-for-DPEPO_modified HEAD:$TARGET_BRANCH -f
fi



echo ""
echo "=== 操作完成 ==="