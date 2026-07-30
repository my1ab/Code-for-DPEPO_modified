"""
Prepare WebShop training data in parquet format.

Following the convention:
  - gamefiles/webshop/  → only stores index JSON files (seed lists for env.env_path)
  - verl_train_data/webshop/ → only stores parquet data files

Usage:
    python data_pipelines/gamefiles/prepare_webshop_parquet.py
"""

import json
import os
import pandas as pd

# ===== 路径集中配置（迁移时只需设环境变量 DPEPO_USER_HOME / DPEPO_PROJECT_NAME） =====
_DPEPO_USER_HOME = os.environ.get('DPEPO_USER_HOME', '/diskpool/home/xuxz')
_DPEPO_PROJECT_NAME = os.environ.get('DPEPO_PROJECT_NAME', 'Code-for-DPEPO')
_DPEPO_CODE_BASE = os.path.join(_DPEPO_USER_HOME, _DPEPO_PROJECT_NAME)
# =================================================================================


TRAIN_SIZE = 500
TEST_SIZE = 50
SEED = 42
GOAL_NUM = 6910  # number of goals when human_goals=False in WebShop envs

# Goal index ranges matching agent_system/environments/env_package/webshop/envs.py:
#   test:  range(0, 500)       → 0-499
#   sft:   range(600, GOAL_NUM) → 600-6909
#   train: range(600, GOAL_NUM) → 600-6909
TRAIN_GOAL_START = 600
TEST_GOAL_END = 500

# 路径设置（自动根据 _DPEPO_CODE_BASE 拼接）
JSON_DIR = os.path.join(_DPEPO_CODE_BASE, 'data_pipelines', 'gamefiles', 'webshop')
PARQUET_DIR = os.path.join(_DPEPO_CODE_BASE, 'data_pipelines', 'verl_train_data', 'webshop')
SUCCESS_LOG = os.path.join(_DPEPO_CODE_BASE, 'data_pipelines', 'gamefiles', 'success_indices_merged_webshop.txt')



def read_json(filepath):
    return json.load(open(filepath, 'r'))


def load_success_logical_indices(log_path, list_len=500):
    """Parse the success log and return a set of 0-based linear indices.

    Args:
        log_path: path to the success log file.
        list_len: length of the original training data list that these
                  linear indices refer to (default 500).
                  Takes the first list_len indices in order.
    """
    import re
    with open(log_path) as f:
        text = f.read()
    match = re.search(r'Success task indices: \[(.*?)\]', text, re.DOTALL)
    if not match:
        print(f"[Warning] No 'Success task indices' found in {log_path}")
        return set()
    nums = [int(x.strip()) for x in match.group(1).split(',') if x.strip()]
    truncated = set(nums[:list_len])
    print(f"[Success] Loaded {len(truncated)} previously solved linear indices (first {list_len}) from {log_path}")
    return truncated

# EXCLUDE_SUCCESS=True  # Set to True to exclude previously solved tasks based on SUCCESS_LOG
EXCLUDE_SUCCESS=False  # Set to True to exclude previously solved tasks based on SUCCESS_LOG
def main(exclude_success=EXCLUDE_SUCCESS):
    os.makedirs(JSON_DIR, exist_ok=True)
    os.makedirs(PARQUET_DIR, exist_ok=True)

    rng = __import__('numpy').random.RandomState(SEED)

    # Goal pools matching envs.py split ranges exactly:
    #   test:  range(500)          → goal indices 0-499
    #   train: range(600, GOAL_NUM) → goal indices 600-6909
    train_pool = list(range(600, GOAL_NUM))
    test_pool = list(range(500))

    print(f"[Range] Test pool:  0-499 ({len(test_pool)} candidates)")
    print(f"[Range] Train pool: 600-{GOAL_NUM-1} ({len(train_pool)} candidates)")

    print('='*80)
    print(f"exclude_success={exclude_success}")
    print('='*80)

    # Step 1: generate 1500 candidates from train pool using rng.choice
    # (same method as envs.py reset())
    train_candidates = rng.choice(train_pool, size=1500, replace=False).tolist()
    test_idx = rng.choice(test_pool, size=min(TEST_SIZE, len(test_pool)), replace=False).tolist()
    print(f"[Candidates] Generated {len(train_candidates)} train candidates")

    # Step 2: remove entries at positions listed in success_linear_indices
    if exclude_success:
        success_linear_indices = load_success_logical_indices(SUCCESS_LOG)
        train_candidates_excluded = [g for i, g in enumerate(train_candidates) if i not in success_linear_indices]
        print(f"[Exclude] Removed {len(success_linear_indices)} positions, remaining: {len(train_candidates_excluded)}")
    else:
        train_candidates_excluded = train_candidates
        print(f"[Exclude] No exclusion applied, candidates remain: {len(train_candidates_excluded)}")

    # Step 3: take first TRAIN_SIZE as final training indices
    print(f"[Final] Taking first {TRAIN_SIZE} train candidates with exclude_success: {exclude_success}")
    train_idx = train_candidates_excluded[:TRAIN_SIZE]
    # train_idx = train_candidates_excluded

    # Build suffix for filenames to indicate exclude status
    suffix = '_excluded' if exclude_success else ''

    # --- Build parquet data ---
    parquet_train_data = []
    for seed in train_idx:
        parquet_train_data.append({
            'answer': '',
            'data_source': 'webshop',
            'prompt': [{'role': 'user', 'content': 'The prompt is dynamic obtained from envs'}],
            'ability': 'agent',
            'gamefile': seed,
            'extra_info': {
                'split': 'train',
            }
        })

    parquet_test_data = []
    for seed in test_idx:
        parquet_test_data.append({
            'answer': '',
            'data_source': 'webshop',
            'prompt': [{'role': 'user', 'content': 'The prompt is dynamic obtained from envs'}],
            'ability': 'agent',
            'gamefile': seed,
            'extra_info': {
                'split': 'test',
            }
        })

    # --- Parquet → PARQUET_DIR ---
    df_train = pd.DataFrame(parquet_train_data)
    train_parquet = os.path.join(PARQUET_DIR, f'webshop_train{suffix}.parquet')
    df_train.to_parquet(train_parquet, index=False)
    print(f"[Train] Saved {len(parquet_train_data)} samples to {train_parquet}")

    df_test = pd.DataFrame(parquet_test_data)
    test_parquet = os.path.join(PARQUET_DIR, f'webshop_test{suffix}.parquet')
    df_test.to_parquet(test_parquet, index=False)
    print(f"[Test]  Saved {len(parquet_test_data)} samples to {test_parquet}")

    # --- Task JSON files (index only) → JSON_DIR ---
    train_json = {str(i): seed for i, seed in enumerate(train_idx)}
    train_json_path = os.path.join(JSON_DIR, f'webshop_train_tasks{suffix}.json')
    with open(train_json_path, 'w') as f:
        json.dump(train_json, f, indent=4)
    print(f"[Train JSON] Saved {len(train_json)} tasks to {train_json_path}")

    test_json = {str(i): seed for i, seed in enumerate(test_idx)}
    test_json_path = os.path.join(JSON_DIR, f'webshop_test_tasks{suffix}.json')
    with open(test_json_path, 'w') as f:
        json.dump(test_json, f, indent=4)
    print(f"[Test JSON]  Saved {len(test_json)} tasks to {test_json_path}")

    print("\nDone! Use these paths in training:")
    print(f"  data.train_files={train_parquet}")
    print(f"  data.val_files={test_parquet}")
    print(f"  env.env_path={train_json_path}")


if __name__ == '__main__':
    main(exclude_success=True)  # First run with exclude setting to generate the excluded version
    print("="*80)
    print("="*80)
    # main(exclude_success=False)  # Run again with opposite exclude setting to generate both versions
