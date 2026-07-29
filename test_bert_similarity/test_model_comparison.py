"""
快速对比不同 sentence embedding 模型在同义/不同词对上的表现
使用 sentence-transformers 库的 SentenceTransformer 接口
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 强制 CPU
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.insert(0, "/diskpool/home/xuxz/Code-for-DPEPO")

# 测试对: (text_a, text_b, should_match, category)
TEST_PAIRS = [
    # === 应判重复 Y: 单词不同义但语义相同 ===
    ("head of NASA during Apollo 11", "NASA administrator Apollo 11", True, "同义-head=administrator"),
    ("head of NASA during Apollo 11", "who was in charge of NASA during Apollo 11", True, "同义-head=in charge of"),
    ("The Terminal director", "who directed The Terminal", True, "同义-director=directed"),
    ("Britney Spears birthplace city state", "Britney Spears born in what city", True, "同义-birthplace=born in"),
    ("Alexandros Matsas birthplace", "Alexandros Matsas born", True, "同义-birthplace=born"),
    ("Alicia Keys education university", "Alicia Keys studied at", True, "同义-education=studied at"),
    ("Beaches 1988 film director", "Beaches 1988 Garry Marshall", True, "同义-director=Garry Marshall"),
    ("Battle of Tarawa date", "Battle of Tarawa 1943", True, "同义-date=1943"),
    ("Alessandro Allori Bronzino adoption", "Alessandro Allori Bronzino uncle death", True, "同义-adoption=uncle death"),
    ("screenwriter of Before Midnight", "writer Before Midnight Before Sunrise After Midnight", True, "同义-screenwriter=writer"),
    ("Bankim Chandra Chattopadhyay brother", "Bankim Chandra Chattopadhyay sibling", True, "同义-brother=sibling"),
    ("Bob Boles opera", "Bob Boles role in opera", True, "同义-opera=role in opera"),

    # === 不应判重复 N: 词汇相似但含义不同 ===
    ("Andy Murray brother", "Andy Murray sister", False, "不同-brother vs sister"),
    ("The Believers composer", "The Believers lyricist", False, "不同-composer vs lyricist"),
    ("head of NASA during Apollo 11", "Apollo 11 mission commander NASA", False, "不同-head vs commander"),
    ("The Terminal director", "The Terminal producer", False, "不同-director vs producer"),
    ("Andy Murray brother", "Andrey Murray brother tennis", False, "不同-Andy vs Andrey"),
    ("Beaches 1988 film director", "Beaches 1988 film director American", False, "不同-+American"),
    ("Beaches 1988 film director", "Beaches 1988 film director British", False, "不同-+British"),
    ("Beaches 1988 film director", "Beaches 1988 film director nationality", False, "不同-+nationality"),
    ("Came Home father", "Came Home paternal father", False, "不同-+paternal"),
    ("Came Home father", "Came Home horse father", False, "不同-+horse"),
    ("Came Home father", "Came Home movie father", False, "不同-+movie"),
    ("Bankim Chandra Chattopadhyay brother", "Bankim Chandra Chattopadhyay sister", False, "不同-brother vs sister"),
    ("Albany International Airport location", "Albany International Airport town", False, "不同-location vs town"),
]

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def test_model(model_name, model_path_or_repo):
    """用给定模型测试所有对，返回 raw cosine 相似度"""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_path_or_repo, device="cpu")
        
        # 收集所有文本
        texts = []
        for a, b, _, _ in TEST_PAIRS:
            texts.extend([a, b])
        
        # 批量编码
        embeddings = model.encode(texts, normalize_embeddings=True)
        
        # 计算每对的 cosine 相似度
        results = []
        for i, (a, b, should_match, cat) in enumerate(TEST_PAIRS):
            emb_a = embeddings[i*2:i*2+1]
            emb_b = embeddings[i*2+1:i*2+2]
            sim = cosine_similarity(emb_a, emb_b)[0][0]
            results.append((a, b, should_match, cat, sim))
        
        return results
    except Exception as e:
        print(f"  [错误] {model_name}: {e}")
        return None

def analyze_results(model_name, results):
    """分析结果: 比较应判Y和应判N的分布"""
    if results is None:
        return
    
    print(f"\n{'='*80}")
    print(f"模型: {model_name}")
    print(f"{'='*80}")
    
    y_sims = [(r[3], r[4]) for r in results if r[2]]  # 应判重复
    n_sims = [(r[3], r[4]) for r in results if not r[2]]  # 不应判重复
    
    y_raws = [s for _, s in y_sims]
    n_raws = [s for _, s in n_sims]
    
    y_min, y_max = min(y_raws), max(y_raws)
    n_min, n_max = min(n_raws), max(n_raws)
    
    print(f"\n  应判重复(Y): min={y_min:.4f} max={y_max:.4f} avg={np.mean(y_raws):.4f}")
    print(f"  不应判(N):   min={n_min:.4f} max={n_max:.4f} avg={np.mean(n_raws):.4f}")
    print(f"  分离间隙: {y_min - n_max:.4f} ({'✓ 可分' if y_min > n_max else '✗ 重叠'})")
    
    # 找最佳阈值
    all_sims = [(s, True) for s in y_raws] + [(s, False) for s in n_raws]
    all_sims.sort(key=lambda x: x[0], reverse=True)
    
    best_acc = 0
    best_thresh = 0
    for i in range(len(all_sims)):
        thresh = all_sims[i][0]
        tp = sum(1 for s, is_y in all_sims if s >= thresh and is_y)
        fp = sum(1 for s, is_y in all_sims if s >= thresh and not is_y)
        fn = sum(1 for s, is_y in all_sims if s < thresh and is_y)
        tn = sum(1 for s, is_y in all_sims if s < thresh and not is_y)
        acc = (tp + tn) / len(all_sims)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    
    print(f"  最佳阈值: {best_thresh:.4f} (准确率 {best_acc:.1%})")
    
    # 打印每对的 raw
    print(f"\n  {'类别':<35} {'Y/N':>4} {'raw':>8}")
    print(f"  {'-'*55}")
    for a, b, should_match, cat, sim in sorted(results, key=lambda x: x[4], reverse=True):
        tag = "Y" if should_match else "N"
        print(f"  {cat:<35} {tag:>4} {sim:>8.4f}")
    
    return best_acc, best_thresh

# 测试多个模型
if __name__ == "__main__":
    models_to_test = [
        ("bge-large-en-v1.5 (当前)", "/diskpool/home/xuxz/Code-for-DPEPO/test_bert_similarity/models/bge-large-en-v1.5"),
        ("bge-base-en-v1.5", "/diskpool/home/xuxz/Code-for-DPEPO/test_bert_similarity/models/bge-base-en-v1.5"),
        # 以下需要联网下载（首次使用会自动下载到缓存）
        ("all-mpnet-base-v2", "sentence-transformers/all-mpnet-base-v2"),
        ("e5-large-v2", "intfloat/e5-large-v2"),
        ("gte-large", "thenlper/gte-large"),
        ("bge-large-zh-v1.5", "BAAI/bge-large-zh-v1.5"),  # 中文模型作对照
    ]
    
    all_results = {}
    for name, path in models_to_test:
        print(f"\n正在加载模型: {name} ...")
        results = test_model(name, path)
        if results:
            acc, thresh = analyze_results(name, results)
            all_results[name] = (acc, thresh)
    
    # 汇总对比
    print(f"\n{'='*80}")
    print("模型对比汇总")
    print(f"{'='*80}")
    print(f"  {'模型':<35} {'最佳准确率':>12} {'最佳阈值':>10}")
    print(f"  {'-'*60}")
    for name, (acc, thresh) in sorted(all_results.items(), key=lambda x: -x[1][0]):
        print(f"  {name:<35} {acc:>12.1%} {thresh:>10.4f}")
