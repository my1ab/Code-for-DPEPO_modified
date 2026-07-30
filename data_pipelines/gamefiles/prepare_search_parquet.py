"""
Prepare Search training data in parquet format (search1 任务).

采样方式（源自 coldstart_search_local_3epoch.py 的 SearchTaskSampler）:
  - 固定范围: source parquet 中 [start, end) 的行索引区间
  - Seed 随机采样: 用固定 seed 构建 shuffle table (random permutation)
  - 排除逻辑: success_indices_merged_500_search.txt 中的 logical_idx
    对应 shuffle table 中的位置，这些位置上的候选被排除。

输出约定 (与 WebShop 版本一致):
  - gamefiles/search/  → 只存储 index JSON 文件 (env.env_path 用)
  - verl_train_data/search/ → 只存储 parquet 数据文件

Usage:
    python data_pipelines/gamefiles/prepare_search_parquet.py
"""

import json
import os
import re
import numpy as np
import pandas as pd

# ============================================================
# 配置参数
# ============================================================

# ===== 路径集中配置（迁移时只需设环境变量 DPEPO_USER_HOME / DPEPO_PROJECT_NAME） =====
_DPEPO_USER_HOME = os.environ.get('DPEPO_USER_HOME', '/diskpool/home/xuxz')
_DPEPO_PROJECT_NAME = os.environ.get('DPEPO_PROJECT_NAME', 'Code-for-DPEPO')
_DPEPO_CODE_BASE = os.path.join(_DPEPO_USER_HOME, _DPEPO_PROJECT_NAME)
# =================================================================================

# Search 原始数据路径（位于 ${USER_HOME}/data/searchR1_processed_direct）
SEARCH_DATA_DIR = os.path.join(_DPEPO_USER_HOME, 'data', 'searchR1_processed_direct')

# 输出路径（自动根据 _DPEPO_CODE_BASE 拼接）
JSON_DIR = os.path.join(_DPEPO_CODE_BASE, 'data_pipelines', 'gamefiles', 'search')
PARQUET_DIR = os.path.join(_DPEPO_CODE_BASE, 'data_pipelines', 'verl_train_data', 'search')
EXCLUDE_LIST = os.path.join(_DPEPO_CODE_BASE, 'data_pipelines', 'gamefiles', 'success_indices_merged_500_search.txt')

# 采样参数
TRAIN_SIZE = 500          # 训练集 size
TEST_SIZE = 50            # 测试集 size
SEED = 1                  # 随机种子 (同 exclude 文件中的 seed=1)
EXCLUDE_SUCCESS = True  

# JSON 输出格式控制
#   False → {"0": physical_idx} (WebShop 兼容格式, 仅索引)
#   True  → {"0": {"physical_idx": ..., "extra_info": {...}}}
INCLUDE_EXTRA_INFO = False

# 数据分区配置 (参照 coldstart_search_local_3epoch.py 的 SearchTaskSampler._SPLIT_CONFIG)
#   全部使用 (0, None) 即全部行,  采样通过 shuffle table 控制
SPLIT_CONFIG = {
    "test":  {"file": "test.parquet",  "range": (0, None)},
    "train": {"file": "train.parquet", "range": (0, None)},
}

# ============================================================
# 工具函数
# ============================================================

def load_success_logical_indices(exclude_path: str) -> set:
    """从 success_indices_merged_500_search.txt 中解析 logical_idx 集。

    文件格式:
        success indices merged:[3, 4, 5, ...]
        len_list = 500 (truncated to 500)
        seed=1  train.parquet

    Returns:
        set[int]: 需要排除的 logical_idx (shuffle table 中的位置)
    """
    # 读取整个排除列表文件内容
    with open(exclude_path) as f:
        text = f.read()
    # 用正则匹配 "success indices merged:[...]" 行，提取方括号内的索引列表
    # DOTALL 允许跨行匹配，防止列表换行
    match = re.search(r'success indices merged:\s*\[(.*?)\]', text, re.DOTALL)
    if not match:
        # 未找到匹配行则返回空集合，同时打印警告
        print(f"[Warning] No 'success indices merged' found in {exclude_path}")
        return set()
    # 解析逗号分隔的字符串，转为整数集合（集合自动去重）
    indices = {int(x.strip()) for x in match.group(1).split(',') if x.strip()}
    print(f"[Exclude] Loaded {len(indices)} logical indices to exclude from {exclude_path}")
    return indices


def read_env_kwargs(row: pd.Series) -> dict:
    """从 parquet 行中提取 env_kwargs (兼容字符串和 dict 两种存储格式)。"""
    env_kwargs = row.get("env_kwargs", {})
    if isinstance(env_kwargs, str):
        env_kwargs = json.loads(env_kwargs)
    if not env_kwargs:
        env_kwargs = {
            "question": row.get("question", ""),
            "ground_truth": row.get("ground_truth", {}),
            "data_source": row.get("data_source", "unknown"),
        }
    return env_kwargs


def convert_to_json_serializable(obj):
    """递归将 numpy 类型转换为 JSON 可序列化的 Python 原生类型。"""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def load_source_data(split: str):
    """加载原始 search parquet，返回 (df, goal_idxs)。

    Args:
        split: 'train' 或 'test'

    Returns:
        (df, goal_idxs): 数据 DataFrame 和可用行索引列表
    """
    cfg = SPLIT_CONFIG.get(split)
    if cfg is None:
        raise ValueError(f"Unknown split '{split}'. Valid: {list(SPLIT_CONFIG.keys())}")

    data_path = os.path.join(SEARCH_DATA_DIR, cfg["file"])
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")

    print(f"[Load] Loading {split} data from {data_path}")
    df = pd.read_parquet(data_path)
    total_rows = len(df)
    print(f"       Total rows: {total_rows}")

    start, end = cfg["range"]
    if end is None:
        end = total_rows
    else:
        end = min(end, total_rows)
    start = min(start, end)

    goal_idxs = list(range(start, end))
    print(f"       Split '{split}': goal_idxs = [{start}, {end})  ({len(goal_idxs)} tasks)")
    return df, goal_idxs


def build_shuffle_table(n: int, seed: int, table_size: int = None) -> list:
    """构建 shuffle table (固定范围 + seed 随机采样)。

    同 coldstart_search_local_3epoch.py 的 SearchTaskSampler._build_shuffle_table:
      - seed >= 0: rng.choice(n, size=table_size, replace=False)

    Args:
        n: 候选池大小 (goal_idxs 长度)
        seed: 随机种子
        table_size: shuffle table 长度, 默认 = n

    Returns:
        list[int]: shuffle table, 每个元素是 goal_idxs 中的位置索引
    """
    if table_size is None:
        table_size = n
    rng = np.random.RandomState(seed)
    table = list(rng.choice(n, size=table_size, replace=False))
    print(f"[Shuffle] Built shuffle table: size={len(table)}/{table_size}, seed={seed}")
    return table


def sample_from_shuffle_table(shuffle_table: list, goal_idxs: list, df: pd.DataFrame,
                               logical_idx: int) -> dict:
    """从 shuffle table 中采样一个任务。

    同 coldstart_search_local_3epoch.py 的 SearchTaskSampler.sample。

    Returns:
        dict: {"question": ..., "ground_truth": ..., "data_source": ...,
               "physical_idx": ..., "logical_idx": ...}
    """
    pos_in_goal = shuffle_table[logical_idx]
    idx = goal_idxs[pos_in_goal]
    row = df.iloc[idx]

    env_kwargs = read_env_kwargs(row)
    question = env_kwargs.get("question") or row.get("question", "")
    ground_truth = env_kwargs.get("ground_truth") or row.get("ground_truth", {})
    data_source = env_kwargs.get("data_source") or row.get("data_source", "unknown")

    if not question:
        raise KeyError(f"Missing 'question' in row {idx} (env_kwargs={env_kwargs})")

    return {
        "question": question,
        "ground_truth": ground_truth,
        "data_source": data_source,
        "physical_idx": int(idx),
        "logical_idx": logical_idx,
    }


def build_parquet_rows(samples: list, split: str) -> list:
    rows = []
    for s in samples:
        # 构建行
        rows.append({
            'answer': '',
            'data_source': s['data_source'],
            'prompt': [{'role': 'user', 'content': 'The prompt is dynamic obtained from envs'}],
            'ability': 'agent',
            'gamefile': s['physical_idx'],  # gamefile = int physical_idx (与 JSON 一致)
            'extra_info': {
                'split': split,
                'ground_truth': s['ground_truth'],
                'physical_idx': s['physical_idx'],
                'logical_idx': s['logical_idx'],
            }
        })
    return rows


def build_task_json(samples: list, split: str, include_extra: bool):
    """构建 task JSON dict。

    include_extra=False: {"0": physical_idx, "1": physical_idx, ...}
    include_extra=True:  {"0": {"physical_idx": ..., "extra_info": {...}}, ...}
    """
    task_dict = {}
    for i, s in enumerate(samples):
        key = str(i)
        if include_extra:
            task_dict[key] = {
                "physical_idx": s["physical_idx"],
                "extra_info": {
                    "question": s["question"],
                    "ground_truth": convert_to_json_serializable(s["ground_truth"]),
                    "data_source": s["data_source"],
                    "split": split,
                    "logical_idx": s["logical_idx"],
                },
            }
        else:
            task_dict[key] = s["physical_idx"]
    return task_dict


def write(shuffle_table, exclude_indices, train_goal_idxs, df_train,
                    exclude_success: bool):
    """生成并写出某一版本的训练数据（不含 test——test 在 main 中统一写出）。"""
    tag = 'excluded' if exclude_success else 'non-excluded'

    if exclude_success:
        selected = [pos for pos in range(len(shuffle_table)) if pos not in exclude_indices]
        print(f"[{tag}] Before exclusion: {len(shuffle_table)} positions, "
              f"after: {len(selected)} (removed {len(shuffle_table) - len(selected)})")
    else:
        selected = list(range(len(shuffle_table)))
        print(f"[{tag}] No exclusion, candidates: {len(selected)}")

    selected_positions = selected[:TRAIN_SIZE]
    print(f"[{tag}] Taking first {len(selected_positions)} positions as train set")

    train_samples_local = []
    for logical_idx in selected_positions:
        sample = sample_from_shuffle_table(shuffle_table, train_goal_idxs, df_train, logical_idx)
        train_samples_local.append(sample)
    print(f"[{tag}] Sampled {len(train_samples_local)} training tasks")

    suffix = '_excluded' if exclude_success else ''

    ptrain = build_parquet_rows(train_samples_local, 'train')
    df_t = pd.DataFrame(ptrain)
    train_parquet_path = os.path.join(PARQUET_DIR, f'search_train{suffix}.parquet')
    df_t.to_parquet(train_parquet_path, index=False)
    print(f"[{tag}] Parquet train: {len(ptrain)} → {train_parquet_path}")

    train_task_dict = build_task_json(train_samples_local, 'train', INCLUDE_EXTRA_INFO)
    train_json_path = os.path.join(JSON_DIR, f'search_train_tasks{suffix}.json')
    with open(train_json_path, 'w') as f:
        json.dump(train_task_dict, f, indent=4, ensure_ascii=False)
    print(f"[{tag}] JSON train: {len(train_task_dict)} → {train_json_path}")

    used = sorted(s['logical_idx'] for s in train_samples_local)
    print(f"[{tag}] Used logical indices (first 20): {used[:20]}...")
    print(f"[{tag}] Used logical indices (last 20):  ...{used[-20:]}")
    print(f"[{tag}] Total train: {len(train_samples_local)}")


# ============================================================
# 主函数
# ============================================================

def main():
    os.makedirs(JSON_DIR, exist_ok=True)
    os.makedirs(PARQUET_DIR, exist_ok=True)

    # ── 1. 加载排除列表 ──────────────────────────────────────
    exclude_indices = load_success_logical_indices(EXCLUDE_LIST)
    print(f"[Exclude] {len(exclude_indices)} indices to exclude from train pool")
    print('=' * 80)

    # ── 2. 加载训练数据 (train.parquet) ──────────────────────
    df_train, train_goal_idxs = load_source_data("train")
    n_train = len(train_goal_idxs)

    # ── 3. 构建 shuffle table (seed 随机采样) ────────────────
    # table_size = 所有可用行数
    shuffle_table = build_shuffle_table(n_train, seed=SEED, table_size=n_train)

    # ── 4. 采样测试数据 (从 test.parquet, 与 train 同 seed=1) ─
    df_test, test_goal_idxs = load_source_data("test")
    n_test = len(test_goal_idxs)

    test_shuffle_table = build_shuffle_table(n_test, seed=SEED, table_size=n_test)

    test_samples = []
    for i in range(min(TEST_SIZE, n_test)):
        sample = sample_from_shuffle_table(test_shuffle_table, test_goal_idxs, df_test, i)
        test_samples.append(sample)
    print(f"[Test] Sampled {len(test_samples)} test tasks")

    # ── 9. 生成两种训练版本 ──────────────────────────────────
    # print('=' * 80)
    # print('>>> Generating non-excluded version (no success exclusion)')
    # print('=' * 80)
    # write(shuffle_table, exclude_indices, train_goal_idxs, df_train,
    #                 exclude_success=False)

    print()
    print('=' * 80)
    print('>>> Generating excluded version (with success exclusion)')
    print('=' * 80)
    write(shuffle_table, exclude_indices, train_goal_idxs, df_train,
                    exclude_success=EXCLUDE_SUCCESS)

    # ── 10. 统一写出 test 数据（仅一份，不受 exclude 影响） ──
    ptest = build_parquet_rows(test_samples, 'test')
    df_v = pd.DataFrame(ptest)
    test_parquet_path = os.path.join(PARQUET_DIR, 'search_test.parquet')
    df_v.to_parquet(test_parquet_path, index=False)

    test_task_dict = build_task_json(test_samples, 'test', INCLUDE_EXTRA_INFO)
    test_json_path = os.path.join(JSON_DIR, 'search_test_tasks.json')
    with open(test_json_path, 'w') as f:
        json.dump(test_task_dict, f, indent=4, ensure_ascii=False)

    print(f"\n[Test] Parquet: {len(ptest)} → {test_parquet_path}")
    print(f"[Test] JSON:    {len(test_task_dict)} → {test_json_path}")

    # ── 11. 打印使用说明 ──────────────────────────────────────
    print()
    print("=" * 80)
    print("Done! Use these paths in training:")
    print("  Non-excluded:")
    print("    data.train_files=" + PARQUET_DIR + "/search_train.parquet")
    print("    env.env_path=" + JSON_DIR + "/search_train_tasks.json")
    print("  Excluded:")
    print("    data.train_files=" + PARQUET_DIR + "/search_train_excluded.parquet")
    print("    env.env_path=" + JSON_DIR + "/search_train_tasks_excluded.json")
    print("  Test:")
    print("    data.val_files=" + PARQUET_DIR + "/search_test.parquet")
    print("=" * 80)


if __name__ == '__main__':
    main()