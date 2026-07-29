"""
使用 BERT 嵌入模型计算文本相似度测试

基于 vLLM 的 pooling 模型 API，使用 BERT-based 模型（如 BAAI/bge-base-en-v1.5）
对输入文本生成嵌入向量，并计算余弦相似度。

参考文档: https://docs.vllm.ai/en/latest/models/pooling_models/embed/
"""

import math
import os
from pathlib import Path

# 禁用 vLLM 的 tqdm 进度条（必须在 import vllm 之前设置）
os.environ["TQDM_DISABLE"] = "1"
from vllm import LLM, PoolingParams

from test_data import TEXT_PAIRS, TEXTS, WEBSHOP_PAIRS, WEBSHOP_TEXTS

# test_data_search.py：Search（QA 检索）任务测试样例数据
# 源自 验证轨迹search/ 下的真实轨迹（bamboogle/hotpotqa/musique/nq/popqa/triviaqa/2wikimultihopqa）
# 当前仅 import 引用，暂不参与测试逻辑
from test_data_search import SEARCH_PAIRS, SEARCH_TEXTS  # noqa: F401





# ── 工具函数 ──────────────────────────────────────────
def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """计算两个向量的余弦相似度"""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

# ── 配置 ──────────────────────────────────────────────
# BERT-based 嵌入模型 (HuggingFace repo id)
# MODEL_NAME = "BAAI/bge-base-en-v1.5"  
MODEL_NAME = "BAAI/bge-large-en-v1.5"  
MODEL_DIR = Path(__file__).parent / "models" / MODEL_NAME.split("/")[-1]  # 本地模型目录
TRUST_REMOTE_CODE = True
UTILIZATION = 0.1


def action_similarity(action_a: str, action_b: str, llm: LLM,
                      search_threshold: float = 0.975) -> float:
    """方案 B + E：分类型相似度计算。

    - click: 大小写不敏感精确匹配，返回 0.0 或 1.0
    - search: 词集匹配优先（方案 E），嵌入兜底，≥ 阈值返回相似度值，否则 0.0
    - 跨类型: 返回 0.0
    - null: null==null 返回 1.0，否则 0.0

    方案 E：在嵌入计算之前，先对 search 内容做词集精确匹配，
    捕获所有"词序重排"变体，避免嵌入模型对词序不稳定导致的漏检。
    """
   

    # null 处理
    if action_a == "null" or action_b == "null":
        return 1.0 if action_a == action_b else 0.0
    
    action_a = action_a.lower()
    action_b = action_b.lower()

    # 提取动作类型
    type_a = action_a.split("[")[0] if "[" in action_a else "unknown"
    type_b = action_b.split("[")[0] if "[" in action_b else "unknown"

    # 不同动作类型 -> 不相似
    if type_a != type_b:
        return 0.0

    if type_a == "click":
        # click 按键：精确匹配（已转小写）
        # 仅click转小写
        target_a = action_a[6:-1]
        target_b = action_b[6:-1]
        return 1.0 if target_a == target_b else 0.0

    if type_a == "search":
        # 保留大小写
        content_a = action_a[7:-1]
        content_b = action_b[7:-1]

        # 精确匹配快速路径
        if content_a == content_b:
            return 1.0

        # 方案 E：词集匹配
        # 完全相同词集的不同排列 -> 一定是重复
        words_a = set(content_a.split())
        words_b = set(content_b.split())
        if words_a == words_b:
            return 1.0

        # 嵌入语义相似度兜底
        (out_a,) = llm.embed(content_a)
        (out_b,) = llm.embed(content_b)
        sim = cosine_similarity(out_a.outputs.embedding, out_b.outputs.embedding)
        # 方案一：阈值过滤 + 范围放大
        #   1) 低于阈值 search_threshold 的相似度贡献为 0
        #   2) 通过阈值的相似度从 [search_threshold, 1.0] 线性放大到 [0, 1]
        if sim < search_threshold:
            return 0.0
        return (sim - search_threshold) / (1.0 - search_threshold)

    return 0.0


# ── 测试 1: 使用 LLM.score 计算文本对相似度 ────────────
def test_pairwise_score(llm: LLM):
    """使用 LLM.score 直接计算文本对的相似度分数"""
    print("\n" + "=" * 70)
    print("测试 1: 使用 LLM.score 计算文本对相似度")
    print("=" * 70)

    results = []
    for text_a, text_b in TEXT_PAIRS:
        (output,) = llm.score(text_a, text_b)
        score = output.outputs.score
        results.append((text_a, text_b, score))
        print(f"\n  文本A: {text_a}")
        print(f"  文本B: {text_b}")
        print(f"  相似度: {score:.6f}")

    print(f"\n  共计算 {len(results)} 对文本的相似度")
    return results


# ── 测试 2: 使用 LLM.embed 计算余弦相似度 ────────────
def test_embed_cosine(llm: LLM):
    """使用 LLM.embed 生成嵌入向量，然后手动计算余弦相似度"""
    print("\n" + "=" * 70)
    print("测试 2: 使用 LLM.embed + 手动余弦相似度")
    print("=" * 70)

    results = []
    for text_a, text_b in TEXT_PAIRS:
        (out_a,) = llm.embed(text_a)
        (out_b,) = llm.embed(text_b)
        emb_a = out_a.outputs.embedding
        emb_b = out_b.outputs.embedding
        sim = cosine_similarity(emb_a, emb_b)
        results.append((text_a, text_b, sim))
        print(f"\n  文本A: {text_a}")
        print(f"  文本B: {text_b}")
        print(f"  余弦相似度: {sim:.6f}")

    print(f"\n  共计算 {len(results)} 对文本的相似度")
    return results


# ── 测试 3: 批量嵌入并构建相似度矩阵 ──────────────────
def test_similarity_matrix(llm: LLM):
    """批量生成嵌入并构建 NxN 相似度矩阵"""
    print("\n" + "=" * 70)
    print("测试 3: 批量嵌入并构建相似度矩阵")
    print("=" * 70)

    # 批量嵌入
    outputs = llm.embed(TEXTS)
    embeddings = [out.outputs.embedding for out in outputs]
    print(f"  已生成 {len(embeddings)} 个嵌入向量，维度: {len(embeddings[0])}")

    # 构建相似度矩阵
    n = len(TEXTS)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            matrix[i][j] = cosine_similarity(embeddings[i], embeddings[j])

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


# ── 测试 4: WebShop 动作相似度验证 ────────────────────
def test_webshop_pairs(llm: LLM, threshold: float = 0.975):
    """测试 WebShop 动作对的相似度（方案 B：click 精确匹配 + search 嵌入阈值）。

    Args:
        llm: vLLM 模型实例（仅 search 动作使用）
        threshold: search 嵌入相似度阈值（click 不受此阈值影响，走精确匹配）
    """
    print("\n" + "=" * 70)
    print("测试 4: WebShop 动作相似度验证（方案 B+E: click精确 + search词集匹配+嵌入兜底）")
    print(f"  search 阈值: {threshold}  |  click: 精确匹配(lower)  |  词集匹配: 优先")
    print("=" * 70)

    # 按类别分组统计
    categories = {}
    correct = 0
    total = 0
    threshold_errors = []
    should_match_sims = []
    should_not_match_sims = []

    print(f"\n  {'#':<3} {'动作A':<28} {'动作B':<28} {'相似度':>7}  {'阈值':>4}  {'期望':>4}  {'判定':>2}  类别")
    print("  " + "-" * 100)

    for idx, (action_a, action_b, should_match, category) in enumerate(WEBSHOP_PAIRS):
        sim = action_similarity(action_a, action_b, llm, search_threshold=threshold)

        actual_match = sim > 0.0
        # should_match=None 表示边界情况，不参与正确率统计
        if should_match is not None:
            total += 1
            is_correct = (actual_match == should_match)
            if is_correct:
                correct += 1
            else:
                threshold_errors.append((action_a, action_b, sim, actual_match, should_match, category))
        else:
            is_correct = "N/A"

        # 收集应判/不应判的相似度（用于阈值分析）
        if should_match is True:
            should_match_sims.append(sim)
        elif should_match is False:
            should_not_match_sims.append(sim)

        # 紧凑标记
        correct_str = "OK" if is_correct is True else ("XX" if is_correct is False else "~~")
        expect_str = "Y" if should_match is True else ("N" if should_match is False else "?")
        match_str = "Y" if actual_match else "N"

        print(f"  {idx:<3d} {action_a:<28s} {action_b:<28s} {sim:7.4f}  {match_str:>4s}  {expect_str:>4s}  {correct_str:>2s}  {category}")

        # 按类别收集
        if category not in categories:
            categories[category] = []
        categories[category].append(sim)

    # 汇总
    print(f"\n  阈值判定正确率: {correct}/{total} = {correct/total*100:.1f}%" if total > 0 else "  无可统计的非边界对")
    if threshold_errors:
        print(f"\n  阈值判定错误（{len(threshold_errors)} 对）:")
        for a, b, sim, actual, expected, cat in threshold_errors:
            tag = "漏检" if (not actual and expected) else "误检"
            print(f"    [{tag}] {a:<28s} vs {b:<28s} sim={sim:.4f} [{cat}]")

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
        print(f"    应判重复的最低相似度: {lowest_match:.4f}")
        print(f"    不应判重复的最高相似度: {highest_no_match:.4f}")
        print(f"    分离间隙: {gap:.4f}")
        if gap > 0:
            suggested = (lowest_match + highest_no_match) / 2
            print(f"    建议阈值（中点）: {suggested:.4f}")
        else:
            print(f"    ⚠ 分离间隙为负，存在重叠区域，阈值无法完美区分")
            print(f"      重叠区间: [{highest_no_match:.4f}, {lowest_match:.4f}]")
    return categories


# ── 测试 5: WebShop 动作相似度矩阵 ────────────────────
def test_webshop_matrix(llm: LLM):
    """构建 WebShop 动作的相似度矩阵。"""
    print("\n" + "=" * 70)
    print("测试 5: WebShop 动作相似度矩阵")
    print("=" * 70)

    texts = WEBSHOP_TEXTS
    n = len(texts)

    outputs = llm.embed(texts)
    embeddings = [out.outputs.embedding for out in outputs]
    print(f"  已生成 {len(embeddings)} 个嵌入向量，维度: {len(embeddings[0])}")

    # 构建相似度矩阵
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            matrix[i][j] = cosine_similarity(embeddings[i], embeddings[j])

    # 打印矩阵
    print(f"\n  相似度矩阵 ({n}x{n}):")
    # 截断显示，每个值占 6 字符
    header = "              " + "".join(f"  W{j:<3d}" for j in range(n))
    print(header)
    for i in range(n):
        label = texts[i][:12]
        row_vals = "".join(f"  {matrix[i][j]:.3f}" for j in range(n))
        print(f"  W{i} {label:<12s}{row_vals}")

    # 打印文本标签
    print("\n  动作标签:")
    for i, text in enumerate(texts):
        print(f"  W{i}: {text}")

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


# ── 测试 6: 使用 Matryoshka 降维嵌入 ──────────────────
def test_matryoshka_dimensions(llm: LLM):
    """测试不同输出维度对相似度的影响（需要模型支持 Matryoshka）"""
    print("\n" + "=" * 70)
    print("测试 6: Matryoshka 不同维度下的嵌入相似度")
    print("=" * 70)

    text_a, text_b = TEXT_PAIRS[0]  # 使用第一对文本

    for dim in [768, 384, 128, 32]:
        try:
            (out_a,) = llm.embed(
                [text_a], pooling_params=PoolingParams(dimensions=dim)
            )
            (out_b,) = llm.embed(
                [text_b], pooling_params=PoolingParams(dimensions=dim)
            )
            emb_a = out_a.outputs.embedding
            emb_b = out_b.outputs.embedding
            sim = cosine_similarity(emb_a, emb_b)
            print(f"  维度={dim:<4d}  相似度={sim:.6f}  (向量长度={len(emb_a)})")
        except Exception as e:
            print(f"  维度={dim:<4d}  不支持: {e}")
            break


# ── 模型下载 ──────────────────────────────────────────
def ensure_model_downloaded(model_name: str, local_dir: Path) -> str:
    """
    确保模型已下载到本地目录。
    - 如果本地目录已存在且非空，直接使用本地路径
    - 否则从 HuggingFace Hub 下载到指定目录

    返回: 本地模型路径
    """
    # 检查本地是否已有模型文件
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"  本地已有模型缓存: {local_dir}")
        return str(local_dir)

    # 从 HuggingFace 下载
    print(f"  正在从 HuggingFace 下载模型: {model_name}")
    print(f"  下载到: {local_dir}")
    os.makedirs(local_dir.parent, exist_ok=True)

    # 优先使用 huggingface_hub 下载
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=model_name,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,  # 下载真实文件而非软链接
        )
    except ImportError:
        # 回退到 transformers 的下载方式
        from transformers import AutoModel, AutoTokenizer
        AutoTokenizer.from_pretrained(model_name)
        AutoModel.from_pretrained(model_name)
        # transformers 默认缓存到 ~/.cache/huggingface，复制到目标目录
        import shutil
        from transformers.utils import default_cache_path
        cache_dir = os.path.join(default_cache_path, f"models--{model_name.replace('/', '--')}")
        if os.path.exists(cache_dir):
            shutil.copytree(cache_dir, str(local_dir), dirs_exist_ok=True)
        else:
            print(f"  警告: 无法确定缓存位置，请手动移动模型文件到 {local_dir}")
    print(f"  下载完成!")
    return str(local_dir)


# ── 主函数 ────────────────────────────────────────────
def main():
    print("=" * 70)
    print("BERT 嵌入模型相似度测试")
    print(f"模型: {MODEL_NAME}")
    print(f"本地目录: {MODEL_DIR}")
    print("=" * 70)

    # 1. 下载模型到本地
    print(f"\n[1/2] 确保模型已下载到本地 ...")
    local_model_path = ensure_model_downloaded(MODEL_NAME, MODEL_DIR)
   

    # 2. 加载本地模型
    print(f"\n[2/2] 正在加载本地模型: {local_model_path} ...")
    llm = LLM(
        model=local_model_path,
        task="embed",
        trust_remote_code=TRUST_REMOTE_CODE,
        gpu_memory_utilization=UTILIZATION,  # GPU 7 已被占用约12G，降低显存利用率
        # dtype="float32",  # 使用 float32 避免 float16 降精度误差
    )
    print("模型加载完成!")

    # exit(0)

    # 运行测试
    # vllm内部自动计算
    test_pairwise_score(llm)
    # 嵌入后手动计算
    test_embed_cosine(llm)
    test_similarity_matrix(llm)
    # WebShop 动作验证（方案 B: click 精确匹配 + search 嵌入阈值 0.975）
    test_webshop_pairs(llm, threshold=0.975)
    test_webshop_matrix(llm)
    # test_matryoshka_dimensions(llm)

    print("\n" + "=" * 70)
    print("所有测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    print(f'test started')
    main()
