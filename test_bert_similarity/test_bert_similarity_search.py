"""
使用 BERT 嵌入模型计算文本相似度测试（Search 任务版）

基于 transformers 的 EmbeddingManager（来自 agent_system/multi_turn_rollout/emb.py），
使用 BERT-based 模型（如 BAAI/bge-large-en-v1.5）在 CPU 上推理，
对输入文本生成嵌入向量，并计算余弦相似度。

与 test_bert_similarity.py 的差异：
1. EmbeddingManager 实例化时传入 task_type='search'，切换到 Search 任务策略
   （search/answer 走词集匹配+嵌入兜底，无 click 精确匹配分支）
2. 测试 4 / 5 使用 test_data_search.py 中的 SEARCH_PAIRS / SEARCH_TEXTS
   而非 WEBSHOP_PAIRS / WEBSHOP_TEXTS，验证 Search 任务的相似度阈值
3. 测试 1 / 2 / 3 仍使用 test_data.py 中的通用 TEXT_PAIRS / TEXTS 做基础嵌入校验
"""

import argparse
import os
import sys
from pathlib import Path

# 将 agent_system 加入 sys.path，以便 import EmbeddingManager
sys.path.insert(0, str(Path(__file__).parent.parent / "agent_system"))
from multi_turn_rollout.emb import EmbeddingManager

from test_data import TEXT_PAIRS, TEXTS  # noqa: F401  (保留通用嵌入测试引用)

# test_data_search.py：Search（QA 检索）任务测试样例数据
# 源自 验证轨迹search/ 下的真实轨迹（bamboogle/hotpotqa/musique/nq/popqa/triviaqa/2wikimultihopqa）
# from test_data_search import SEARCH_PAIRS, SEARCH_TEXTS
# from test_data_search_1 import SEARCH_PAIRS, SEARCH_TEXTS
from test_data_search_2 import SEARCH_PAIRS, SEARCH_TEXTS

# ── 配置 ──────────────────────────────────────────────
MODEL_DIR = str(Path(__file__).parent / "models" / "bge-large-en-v1.5")
# 从命令行参数读取阈值（run_search.sh 中通过 --threshold 传入）
parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=float, default=0.893, help="相似度阈值")
args, _ = parser.parse_known_args()
SIMILARITY_THRESHOLD = args.threshold
GPU_MEMORY_UTILIZATION = 0.1
# Search 任务策略：search/answer 走词集匹配+嵌入兜底，无 click 分支
TASK_TYPE = "search"


# ── 测试 1: 使用 EmbeddingManager.similarity 计算文本对相似度 ──
def test_pairwise_similarity(emb_mgr: EmbeddingManager):
    """使用 EmbeddingManager.similarity 计算文本对的余弦相似度"""
    print("\n" + "=" * 70)
    print("测试 1: 使用 EmbeddingManager.similarity 计算文本对相似度")
    print("=" * 70)

    results = []
    for text_a, text_b in TEXT_PAIRS:
        sim = emb_mgr.similarity(text_a, text_b)
        results.append((text_a, text_b, sim))
        print(f"\n  文本A: {text_a}")
        print(f"  文本B: {text_b}")
        print(f"  相似度: {sim:.6f}")

    print(f"\n  共计算 {len(results)} 对文本的相似度")
    return results


# ── 测试 2: 使用 EmbeddingManager.embed + cosine_similarity ──
def test_embed_cosine(emb_mgr: EmbeddingManager):
    """使用 EmbeddingManager.embed 生成嵌入向量，然后手动计算余弦相似度"""
    print("\n" + "=" * 70)
    print("测试 2: 使用 EmbeddingManager.embed + 手动余弦相似度")
    print("=" * 70)

    results = []
    for text_a, text_b in TEXT_PAIRS:
        emb_a = emb_mgr.embed(text_a)
        emb_b = emb_mgr.embed(text_b)
        sim = emb_mgr.cosine_similarity(emb_a, emb_b)
        results.append((text_a, text_b, sim))
        print(f"\n  文本A: {text_a}")
        print(f"  文本B: {text_b}")
        print(f"  余弦相似度: {sim:.6f}")

    print(f"\n  共计算 {len(results)} 对文本的相似度")
    return results


# ── 测试 3: 批量嵌入并构建相似度矩阵 ──────────────────
def test_similarity_matrix(emb_mgr: EmbeddingManager):
    """批量生成嵌入并构建 NxN 相似度矩阵"""
    print("\n" + "=" * 70)
    print("测试 3: 批量嵌入并构建相似度矩阵")
    print("=" * 70)

    # 批量嵌入
    embeddings = emb_mgr.embed_batch(TEXTS)
    print(f"  已生成 {len(embeddings)} 个嵌入向量，维度: {len(embeddings[0])}")

    # 构建相似度矩阵
    n = len(TEXTS)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            matrix[i][j] = emb_mgr.cosine_similarity(embeddings[i], embeddings[j])

    # 打印矩阵
    print(f"\n  相似度矩阵 ({n}x{n}):")
    header = "          " + "".join(f"  T{j:<4}" for j in range(n))
    print(header)
    for i in range(n):
        row_label = f"  T{i} "
        row_vals = "".join(f"  {matrix[i][j]:.3f}" for j in range(n))
        print(f"{row_label}{row_vals}")

    # 打印文本标签
    print("\n  文本标签:")
    for i, text in enumerate(TEXTS):
        print(f"  T{i}: {text}")

    # 找出最相似和最不相似的文本对（排除自身）
    print("\n  最相似的文本对（排除自身）:")
    best_sim, best_pair = -1, None
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] > best_sim:
                best_sim = matrix[i][j]
                best_pair = (i, j)
    if best_pair:
        i, j = best_pair
        print(f"    T{i} ↔ T{j}: {best_sim:.6f}")
        print(f"    \"{TEXTS[i]}\"")
        print(f"    \"{TEXTS[j]}\"")

    print("\n  最不相似的文本对:")
    worst_sim, worst_pair = 1, None
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] < worst_sim:
                worst_sim = matrix[i][j]
                worst_pair = (i, j)
    if worst_pair:
        i, j = worst_pair
        print(f"    T{i} ↔ T{j}: {worst_sim:.6f}")
        print(f"    \"{TEXTS[i]}\"")
        print(f"    \"{TEXTS[j]}\"")

    return matrix


# ── 测试 4: Search 动作相似度验证 ────────────────────
def test_search_pairs(emb_mgr: EmbeddingManager, threshold: float = 0.975):
    """测试 Search 动作对的相似度（方案 B+E: search/answer 词集匹配+嵌入兜底）。

    Args:
        emb_mgr: EmbeddingManager 实例（必须以 task_type='search' 构造）
        threshold: search/answer 嵌入相似度阈值（仅用于展示，实际阈值在 emb_mgr 内）
    """
    print("\n" + "=" * 70)
    print("测试 4: Search 动作相似度验证（方案 B+E: search/answer 词集匹配+嵌入兜底）")
    print(f"  search/answer 阈值: {threshold}  |  null: 精确匹配  |  词集匹配: 优先  |  task_type={emb_mgr.task_type}")
    print("=" * 70)

    # 按类别分组统计
    categories = {}
    correct = 0
    total = 0
    threshold_errors = []
    should_match_sims = []
    should_not_match_sims = []
    # 原始相似度（阈值过滤前、线性放大前），用于更准确的阈值分析
    should_match_raw_sims = []
    should_not_match_raw_sims = []

    print(f"\n  {'#':<3} {'动作A':<42} {'动作B':<42} {'原始':>7} {'归一':>7}  {'判定':>2}  {'期望':>2}  类别")
    print("  " + "-" * 130)

    for idx, (action_a, action_b, should_match, category) in enumerate(SEARCH_PAIRS):
        sim = emb_mgr.action_similarity(action_a, action_b)
        raw_sim, note = emb_mgr.raw_similarity(action_a, action_b)

        actual_match = sim > 0.0
        # should_match=None 表示边界情况，不参与正确率统计
        if should_match is not None:
            total += 1
            is_correct = (actual_match == should_match)
            if is_correct:
                correct += 1
            else:
                threshold_errors.append((action_a, action_b, sim, raw_sim, actual_match, should_match, category))
        else:
            is_correct = "N/A"

        # 收集应判/不应判的相似度（用于阈值分析）
        if should_match is True:
            should_match_sims.append(sim)
            should_match_raw_sims.append(raw_sim)
        elif should_match is False:
            should_not_match_sims.append(sim)
            should_not_match_raw_sims.append(raw_sim)

        # 紧凑标记
        correct_str = "OK" if is_correct is True else ("XX" if is_correct is False else "~~")
        expect_str = "Y" if should_match is True else ("N" if should_match is False else "?")
        match_str = "Y" if actual_match else "N"

        print(f"  {idx:<3d} {action_a:<42s} {action_b:<42s} {raw_sim:7.4f} {sim:7.4f}  {match_str:>2s}  {expect_str:>2s}  {correct_str:>2s}  {category} [{note}]")

        # 按类别收集
        if category not in categories:
            categories[category] = []
        categories[category].append(sim)

    # 汇总
    print(f"\n  阈值判定正确率: {correct}/{total} = {correct/total*100:.1f}%" if total > 0 else "  无可统计的非边界对")
    if threshold_errors:
        print(f"\n  阈值判定错误（{len(threshold_errors)} 对）:")
        for a, b, sim, raw_sim, actual, expected, cat in threshold_errors:
            tag = "漏检" if (not actual and expected) else "误检"
            print(f"    [{tag}] {a:<42s} vs {b:<42s} raw={raw_sim:.4f} norm={sim:.4f} [{cat}]")

    # 按类别统计相似度范围
    print(f"\n  各类别相似度范围:")
    print(f"  {'类别':<22s} {'数量':>4s} {'最小':>7s} {'最大':>7s} {'平均':>7s}  分布")
    print("  " + "-" * 65)
    for cat, sims in sorted(categories.items()):
        avg = sum(sims) / len(sims)
        bar = "#" * int(avg * 20)
        print(f"  {cat:<22s} {len(sims):>4d} {min(sims):>7.4f} {max(sims):>7.4f} {avg:>7.4f}  {bar}")

    # 阈值建议
    print(f"\n  阈值分析 (当前={threshold}):")
    if should_match_sims and should_not_match_sims:
        lowest_match = min(should_match_sims)
        highest_no_match = max(should_not_match_sims)
        gap = lowest_match - highest_no_match
        print(f"    [归一化相似度]")
        print(f"    应判重复的最低归一化相似度: {lowest_match:.4f}")
        print(f"    不应判重复的最高归一化相似度: {highest_no_match:.4f}")
        print(f"    分离间隙: {gap:.4f}")
        if gap > 0:
            suggested = (lowest_match + highest_no_match) / 2
            print(f"    建议阈值（中点）: {suggested:.4f}")
        else:
            print(f"    ⚠ 分离间隙为负，存在重叠区域，阈值无法完美区分")
            print(f"      重叠区间: [{highest_no_match:.4f}, {lowest_match:.4f}]")

    if should_match_raw_sims and should_not_match_raw_sims:
        lowest_raw_match = min(should_match_raw_sims)
        highest_raw_no_match = max(should_not_match_raw_sims)
        raw_gap = lowest_raw_match - highest_raw_no_match
        print(f"    [原始相似度（阈值过滤前、线性放大前）]")
        print(f"    应判重复的最低原始相似度: {lowest_raw_match:.4f}")
        print(f"    不应判重复的最高原始相似度: {highest_raw_no_match:.4f}")
        print(f"    分离间隙: {raw_gap:.4f}")
        if raw_gap > 0:
            suggested_raw = (lowest_raw_match + highest_raw_no_match) / 2
            print(f"    建议阈值（中点）: {suggested_raw:.4f}")
        else:
            print(f"    ⚠ 分离间隙为负，存在重叠区域，阈值无法完美区分")
            print(f"      重叠区间: [{highest_raw_no_match:.4f}, {lowest_raw_match:.4f}]")
    return categories


# ── 测试 5: Search 动作相似度矩阵 ────────────────────
def test_search_matrix(emb_mgr: EmbeddingManager):
    """构建 Search 动作的相似度矩阵。"""
    print("\n" + "=" * 70)
    print("测试 5: Search 动作相似度矩阵")
    print("=" * 70)

    texts = SEARCH_TEXTS
    n = len(texts)

    embeddings = emb_mgr.embed_batch(texts)
    print(f"  已生成 {len(embeddings)} 个嵌入向量，维度: {len(embeddings[0])}")

    # 构建相似度矩阵
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            matrix[i][j] = emb_mgr.cosine_similarity(embeddings[i], embeddings[j])

    # 打印矩阵
    print(f"\n  相似度矩阵 ({n}x{n}):")
    # 截断显示，每个值占 6 字符
    header = "              " + "".join(f"  S{j:<3d}" for j in range(n))
    print(header)
    for i in range(n):
        label = texts[i][:12]
        row_vals = "".join(f"  {matrix[i][j]:.3f}" for j in range(n))
        print(f"  S{i} {label:<12s}{row_vals}")

    # 打印文本标签
    print("\n  动作标签:")
    for i, text in enumerate(texts):
        print(f"  S{i}: {text}")

    # 统计相似度分布
    all_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            all_sims.append(matrix[i][j])

    all_sims.sort(reverse=True)
    print(f"\n  相似度分布（排除自身，共 {len(all_sims)} 对）:")
    print(f"    最高: {all_sims[0]:.4f}")
    print(f"    最低: {all_sims[-1]:.4f}")
    print(f"    中位数: {all_sims[len(all_sims)//2]:.4f}")
    print(f"    平均: {sum(all_sims)/len(all_sims):.4f}")

    # 分段统计
    ranges = [(0.9, 1.0), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.5, 0.6), (0.0, 0.5)]
    print(f"\n  相似度分段统计:")
    for lo, hi in ranges:
        count = sum(1 for s in all_sims if lo <= s < hi)
        # bar = "█" * count
        # print(f"    [{lo:.1f}, {hi:.1f}): {count:>3d} {bar}")
        bar = "|" * count
        print(f"    [{lo:.1f}, {hi:.1f}): {count:>3d} {bar}")

    return matrix


# ── 测试 6: 使用 EmbeddingManager 替代 Matryoshka 维度测试 ──
# Matryoshka 降维嵌入需要 vLLM 的 PoolingParams(dimensions=...) 支持，
# EmbeddingManager 基于 transformers 暂不实现此功能，故移除该测试。


# ── 模型初始化 ──────────────────────────────────────────
# EmbeddingManager 内置本地目录校验，初始化时模型必须已存在。
# 如需要下载模型，请使用旧版 ensure_model_downloaded 或手动放置到 MODEL_DIR。


# ── 主函数 ────────────────────────────────────────────
def main():
    print("=" * 70)
    print("BERT 嵌入模型相似度测试（基于 EmbeddingManager）- Search 任务版")
    print(f"模型目录: {MODEL_DIR}")
    print(f"相似度阈值: {SIMILARITY_THRESHOLD}")
    print(f"任务类型: {TASK_TYPE}")
    print("=" * 70)

    # 加载 EmbeddingManager（CPU 推理，绕过 GPU 问题）
    # task_type='search' 启用 search/answer 词集匹配+嵌入兜底策略
    print(f"\n[1/1] 正在加载 EmbeddingManager (task_type={TASK_TYPE}) ...")
    emb_mgr = EmbeddingManager(
        model_path=MODEL_DIR,
        similarity_threshold=SIMILARITY_THRESHOLD,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        task_type=TASK_TYPE,
    )
    print("EmbeddingManager 加载完成!")

    # 运行测试
    # test_pairwise_similarity(emb_mgr)
    # test_embed_cosine(emb_mgr)
    # test_similarity_matrix(emb_mgr)
    test_search_pairs(emb_mgr, threshold=SIMILARITY_THRESHOLD)
    # test_search_matrix(emb_mgr)

    print("\n" + "=" * 70)
    print("所有测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    print(f'test started')
    main()
