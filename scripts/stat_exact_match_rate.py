"""
统计真实轨迹在 embedding 关闭（精确匹配）下的 penalty 触发率。

参照 rollout_loop_parallel_search.py 的 calculate_penalties 三条精确匹配路径：
  1. 深度重复（calculate_depth_repeat）：同一 env 的当前动作与历史动作完全相等
  2. 转移重复（calculate_transition_repeat）：(prev, curr) 元组完全相等
  3. 宽度重复（_calculate_width_repeat_weighted）：同一 step 内多个 env 动作完全相等

数据源：test_bert_similarity/验证轨迹search/*_success.json
  结构：List[Dict], 每条含 trajectory: List[Dict]
        assistant step 含 get_actions: Dict[str, str]  键 "0"~"4"
"""

import json
import os
import re
from collections import defaultdict
from itertools import combinations

TRAJ_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'test_bert_similarity', '验证轨迹search'
)

# 规范化：将 <search>q</search> -> search[q], <answer>a</answer> -> answer[a], null -> null
def normalize_action(raw):
    if raw is None or raw.strip() == '' or raw.strip().lower() == 'null':
        return 'null'
    m = re.search(r'<(search|answer)>(.*?)</\1>', raw, re.DOTALL)
    if m:
        return f'{m.group(1)}[{m.group(2).strip()}]'
    return raw.strip()


def load_actions_from_file(filepath):
    """从一个 success.json 中提取所有 (trajectory_id, step_idx, env_idx, action) 四元组。

    返回：
        traj_actions: List[List[Dict[str, str]]]
            每条 trajectory 是一个 list，元素是 step，
            step 是 {env_idx_str: normalized_action} 字典。
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    traj_actions = []
    for item in data:
        traj = item.get('trajectory', [])
        steps = []
        for step_msg in traj:
            if not isinstance(step_msg, dict):
                continue
            if step_msg.get('role') != 'assistant':
                continue
            ga = step_msg.get('get_actions')
            if not isinstance(ga, dict):
                continue
            norm = {k: normalize_action(v) for k, v in ga.items()}
            steps.append(norm)
        if steps:
            traj_actions.append(steps)
    return traj_actions


def analyze_dataset(name, traj_actions):
    """对一个数据集统计三种精确匹配 penalty 的触发情况。"""
    n_traj = len(traj_actions)
    n_steps = sum(len(t) for t in traj_actions)
    n_envs_per_step = 5  # 固定 num_parallel=5

    # ---- 全局动作收集（去重统计） ----
    all_actions = []  # 全部动作（含 null）
    all_actions_non_null = []
    for traj in traj_actions:
        for step in traj:
            for env_idx, act in step.items():
                all_actions.append(act)
                if act != 'null':
                    all_actions_non_null.append(act)

    total_actions = len(all_actions)
    unique_actions = len(set(all_actions))
    unique_non_null = len(set(all_actions_non_null))

    # ---- 1. 宽度重复（同一 step 内多个 env 动作完全相等）----
    # 模拟 _calculate_width_repeat_weighted 精确匹配分支
    # COUNT_width_repeat = len(actions) - len(set(actions))
    width_repeat_total = 0  # 累计重复计数
    width_repeat_steps = 0  # 触发宽度重复的 step 数
    width_total_steps = 0
    for traj in traj_actions:
        for step in traj:
            actions = list(step.values())
            actions_wo_null = [a for a in actions if a != 'null']
            if len(actions_wo_null) <= 1:
                cnt = 0
            else:
                cnt = len(actions_wo_null) - len(set(actions_wo_null))
            width_repeat_total += cnt
            if cnt > 0:
                width_repeat_steps += 1
            width_total_steps += 1

    # ---- 2. 深度重复（同一 env 的当前动作与历史动作完全相等）----
    # 模拟 calculate_depth_repeat：每个 step 的每个 env 动作与前面所有历史比较
    # 注意：原代码不区分 null 与非 null，null == null 也会计数
    depth_repeat_total = 0  # 累计深度匹配次数
    depth_repeat_steps_envs = 0  # 触发深度重复的 (step, env) 数
    depth_total_step_envs = 0
    # 拆分 null 与非 null
    depth_repeat_total_null = 0
    depth_repeat_total_nonnull = 0
    depth_repeat_step_envs_null = 0
    depth_repeat_step_envs_nonnull = 0
    depth_total_null = 0
    depth_total_nonnull = 0
    for traj in traj_actions:
        # 每个 env 累积历史
        env_histories = defaultdict(list)
        for step in traj:
            for env_idx, act in step.items():
                hist = env_histories[env_idx]
                if hist:
                    cnt = sum(1 for h in reversed(hist) if h == act)
                else:
                    cnt = 0
                depth_repeat_total += cnt
                if cnt > 0:
                    depth_repeat_steps_envs += 1
                depth_total_step_envs += 1
                # 拆分
                if act == 'null':
                    depth_total_null += 1
                    depth_repeat_total_null += cnt
                    if cnt > 0:
                        depth_repeat_step_envs_null += 1
                else:
                    depth_total_nonnull += 1
                    depth_repeat_total_nonnull += cnt
                    if cnt > 0:
                        depth_repeat_step_envs_nonnull += 1
                env_histories[env_idx].append(act)

    # ---- 3. 转移重复（(prev, curr) 元组与历史转移对完全相等）----
    # 模拟 calculate_transition_repeat：state_action_pair_a.count(last_state_action)
    trans_repeat_total = 0
    trans_repeat_steps_envs = 0
    trans_total_step_envs = 0
    for traj in traj_actions:
        env_transitions = defaultdict(list)  # 每个 env 累积 (prev, curr) 元组
        env_prev = {}  # 每个 env 的上一个动作
        for step in traj:
            for env_idx, act in step.items():
                if env_idx in env_prev:
                    curr_pair = (env_prev[env_idx], act)
                    pairs = env_transitions[env_idx]
                    cnt = pairs.count(curr_pair)
                    trans_repeat_total += cnt
                    if cnt > 0:
                        trans_repeat_steps_envs += 1
                    trans_total_step_envs += 1
                    env_transitions[env_idx].append(curr_pair)
                env_prev[env_idx] = act

    # ---- 4. 综合：完全匹配可被捕获的动作比例 ----
    # "可被完全匹配捕获" 意味着该 (step, env) 的动作在精确匹配下至少触发一种 penalty
    captured_step_envs = 0
    total_step_envs = 0
    for traj in traj_actions:
        env_histories = defaultdict(list)
        env_prev = {}
        for step in traj:
            # 先算宽度重复（针对整个 step）
            step_actions = list(step.values())
            step_non_null = [a for a in step_actions if a != 'null']
            width_set = set()
            if len(step_non_null) > 1:
                seen = set()
                for a in step_non_null:
                    if a in seen:
                        width_set.add(a)
                    seen.add(a)

            for env_idx, act in step.items():
                total_step_envs += 1
                captured = False
                # 宽度
                if act in width_set:
                    captured = True
                # 深度
                if not captured and env_histories[env_idx]:
                    if act in env_histories[env_idx]:
                        captured = True
                # 转移
                if not captured and env_idx in env_prev:
                    curr_pair = (env_prev[env_idx], act)
                    if curr_pair in env_transitions.get(env_idx, []):
                        captured = True
                if captured:
                    captured_step_envs += 1
                env_histories[env_idx].append(act)
                if env_idx in env_prev:
                    env_transitions[env_idx].append((env_prev[env_idx], act))
                env_prev[env_idx] = act

    return {
        'name': name,
        'n_traj': n_traj,
        'n_steps': n_steps,
        'total_actions': total_actions,
        'unique_actions': unique_actions,
        'unique_ratio': unique_actions / total_actions if total_actions else 0,
        'unique_non_null': unique_non_null,
        # 宽度
        'width_repeat_total': width_repeat_total,
        'width_repeat_steps': width_repeat_steps,
        'width_total_steps': width_total_steps,
        'width_step_rate': width_repeat_steps / width_total_steps if width_total_steps else 0,
        # 深度
        'depth_repeat_total': depth_repeat_total,
        'depth_repeat_step_envs': depth_repeat_steps_envs,
        'depth_total_step_envs': depth_total_step_envs,
        'depth_step_env_rate': depth_repeat_steps_envs / depth_total_step_envs if depth_total_step_envs else 0,
        'depth_repeat_total_null': depth_repeat_total_null,
        'depth_repeat_total_nonnull': depth_repeat_total_nonnull,
        'depth_repeat_step_envs_null': depth_repeat_step_envs_null,
        'depth_repeat_step_envs_nonnull': depth_repeat_step_envs_nonnull,
        'depth_total_null': depth_total_null,
        'depth_total_nonnull': depth_total_nonnull,
        # 转移
        'trans_repeat_total': trans_repeat_total,
        'trans_repeat_step_envs': trans_repeat_steps_envs,
        'trans_total_step_envs': trans_total_step_envs,
        'trans_step_env_rate': trans_repeat_steps_envs / trans_total_step_envs if trans_total_step_envs else 0,
        # 综合
        'captured_step_envs': captured_step_envs,
        'total_step_envs': total_step_envs,
        'captured_rate': captured_step_envs / total_step_envs if total_step_envs else 0,
    }


def main():
    files = [
        ('Bamboogle', 'search_coldstart_bamboogle_125_success.json'),
        ('HotpotQA', 'search_coldstart_hotpotqa_7405_success.json'),
        ('MuSiQue', 'search_coldstart_musique_2417_success.json'),
        ('NQ', 'search_coldstart_nq_3610_success.json'),
        ('PopQA', 'search_coldstart_popqa_14267_success.json'),
        ('TriviaQA', 'search_coldstart_triviaqa_11313_success.json'),
        ('2WikiMultiHopQA', 'search_coldstart_2wikimultihopqa_12576_success.json'),
    ]

    all_results = []
    grand_total = defaultdict(int)
    # 额外的拆分统计
    split_total = defaultdict(int)

    for name, fname in files:
        path = os.path.join(TRAJ_DIR, fname)
        if not os.path.exists(path):
            print(f'[skip] {name}: file not found')
            continue
        traj_actions = load_actions_from_file(path)
        r = analyze_dataset(name, traj_actions)
        all_results.append(r)
        for k in ['n_traj', 'n_steps', 'total_actions', 'unique_actions',
                  'unique_non_null', 'width_repeat_total', 'width_repeat_steps',
                  'width_total_steps', 'depth_repeat_total', 'depth_repeat_step_envs',
                  'depth_total_step_envs', 'trans_repeat_total', 'trans_repeat_step_envs',
                  'trans_total_step_envs', 'captured_step_envs', 'total_step_envs',
                  'depth_repeat_total_null', 'depth_repeat_total_nonnull',
                  'depth_repeat_step_envs_null', 'depth_repeat_step_envs_nonnull',
                  'depth_total_null', 'depth_total_nonnull']:
            grand_total[k] += r[k]

    # 打印每个数据集
    print('=' * 120)
    print(f'{"数据集":<20} {"轨迹":>6} {"步数":>6} {"总动作":>8} {"唯一动作":>8} {"去重率%":>8}')
    print('-' * 120)
    for r in all_results:
        print(f'{r["name"]:<20} {r["n_traj"]:>6} {r["n_steps"]:>6} '
              f'{r["total_actions"]:>8} {r["unique_actions"]:>8} '
              f'{r["unique_ratio"]*100:>7.2f}%')
    print('-' * 120)
    print(f'{"合计":<20} {grand_total["n_traj"]:>6} {grand_total["n_steps"]:>6} '
          f'{grand_total["total_actions"]:>8} {grand_total["unique_actions"]:>8} '
          f'{grand_total["unique_actions"]/grand_total["total_actions"]*100:>7.2f}%')

    print()
    print('=' * 120)
    print('宽度重复（同一 step 内多个 env 动作完全相等）')
    print(f'{"数据集":<20} {"触发步数":>8} {"总步数":>8} {"触发率%":>8} {"累计计数":>10}')
    print('-' * 120)
    for r in all_results:
        print(f'{r["name"]:<20} {r["width_repeat_steps"]:>8} {r["width_total_steps"]:>8} '
              f'{r["width_step_rate"]*100:>7.2f}% {r["width_repeat_total"]:>10}')
    print('-' * 120)
    print(f'{"合计":<20} {grand_total["width_repeat_steps"]:>8} {grand_total["width_total_steps"]:>8} '
          f'{grand_total["width_repeat_steps"]/grand_total["width_total_steps"]*100:>7.2f}% '
          f'{grand_total["width_repeat_total"]:>10}')

    print()
    print('=' * 120)
    print('深度重复（同一 env 的当前动作与历史动作完全相等）')
    print(f'{"数据集":<20} {"触发次数":>8} {"总比较":>8} {"触发率%":>8} {"累计计数":>10}')
    print('-' * 120)
    for r in all_results:
        print(f'{r["name"]:<20} {r["depth_repeat_step_envs"]:>8} {r["depth_total_step_envs"]:>8} '
              f'{r["depth_step_env_rate"]*100:>7.2f}% {r["depth_repeat_total"]:>10}')
    print('-' * 120)
    print(f'{"合计":<20} {grand_total["depth_repeat_step_envs"]:>8} {grand_total["depth_total_step_envs"]:>8} '
          f'{grand_total["depth_repeat_step_envs"]/grand_total["depth_total_step_envs"]*100:>7.2f}% '
          f'{grand_total["depth_repeat_total"]:>10}')

    print()
    print('=' * 120)
    print('转移重复（(prev, curr) 元组与历史转移对完全相等）')
    print(f'{"数据集":<20} {"触发次数":>8} {"总比较":>8} {"触发率%":>8} {"累计计数":>10}')
    print('-' * 120)
    for r in all_results:
        print(f'{r["name"]:<20} {r["trans_repeat_step_envs"]:>8} {r["trans_total_step_envs"]:>8} '
              f'{r["trans_step_env_rate"]*100:>7.2f}% {r["trans_repeat_total"]:>10}')
    print('-' * 120)
    print(f'{"合计":<20} {grand_total["trans_repeat_step_envs"]:>8} {grand_total["trans_total_step_envs"]:>8} '
          f'{grand_total["trans_repeat_step_envs"]/grand_total["trans_total_step_envs"]*100:>7.2f}% '
          f'{grand_total["trans_repeat_total"]:>10}')

    print()
    print('=' * 120)
    print('综合：精确匹配可捕获的 (step, env) 动作比例')
    print(f'{"数据集":<20} {"捕获数":>8} {"总数":>8} {"捕获率%":>8}')
    print('-' * 120)
    for r in all_results:
        print(f'{r["name"]:<20} {r["captured_step_envs"]:>8} {r["total_step_envs"]:>8} '
              f'{r["captured_rate"]*100:>7.2f}%')
    print('-' * 120)
    print(f'{"合计":<20} {grand_total["captured_step_envs"]:>8} {grand_total["total_step_envs"]:>8} '
          f'{grand_total["captured_step_envs"]/grand_total["total_step_envs"]*100:>7.2f}%')

    # ---- 关键拆分：null vs 非 null 的深度重复率 ----
    print()
    print('=' * 120)
    print('关键拆分：深度重复中 null vs 非 null 动作')
    print(f'{"数据集":<20} {"null总数":>8} {"null触发":>8} {"null率%":>8} '
          f'{"非null总":>8} {"非null触":>8} {"非null率%":>10}')
    print('-' * 120)
    for r in all_results:
        null_rate = r['depth_repeat_step_envs_null'] / r['depth_total_null'] * 100 if r['depth_total_null'] else 0
        nonnull_rate = r['depth_repeat_step_envs_nonnull'] / r['depth_total_nonnull'] * 100 if r['depth_total_nonnull'] else 0
        print(f'{r["name"]:<20} {r["depth_total_null"]:>8} {r["depth_repeat_step_envs_null"]:>8} {null_rate:>7.2f}% '
              f'{r["depth_total_nonnull"]:>8} {r["depth_repeat_step_envs_nonnull"]:>8} {nonnull_rate:>9.2f}%')
    print('-' * 120)
    g_null_rate = grand_total['depth_repeat_step_envs_null'] / grand_total['depth_total_null'] * 100 if grand_total['depth_total_null'] else 0
    g_nonnull_rate = grand_total['depth_repeat_step_envs_nonnull'] / grand_total['depth_total_nonnull'] * 100 if grand_total['depth_total_nonnull'] else 0
    print(f'{"合计":<20} {grand_total["depth_total_null"]:>8} {grand_total["depth_repeat_step_envs_null"]:>8} {g_null_rate:>7.2f}% '
          f'{grand_total["depth_total_nonnull"]:>8} {grand_total["depth_repeat_step_envs_nonnull"]:>8} {g_nonnull_rate:>9.2f}%')

    # ---- 关键：非 null 动作的深度重复详细 ----
    print()
    print('=' * 120)
    print('非 null 动作深度重复详情（即 search/answer 的精确匹配捕获率）')
    print(f'{"数据集":<20} {"非null总":>8} {"触发次数":>8} {"累计计数":>8} {"触发率%":>8}')
    print('-' * 120)
    for r in all_results:
        rate = r['depth_repeat_step_envs_nonnull'] / r['depth_total_nonnull'] * 100 if r['depth_total_nonnull'] else 0
        print(f'{r["name"]:<20} {r["depth_total_nonnull"]:>8} {r["depth_repeat_step_envs_nonnull"]:>8} '
              f'{r["depth_repeat_total_nonnull"]:>8} {rate:>7.2f}%')
    print('-' * 120)
    g_rate = grand_total['depth_repeat_step_envs_nonnull'] / grand_total['depth_total_nonnull'] * 100 if grand_total['depth_total_nonnull'] else 0
    print(f'{"合计":<20} {grand_total["depth_total_nonnull"]:>8} {grand_total["depth_repeat_step_envs_nonnull"]:>8} '
          f'{grand_total["depth_repeat_total_nonnull"]:>8} {g_rate:>7.2f}%')

    # 额外：按动作类型分组的去重率
    print()
    print('=' * 120)
    print('按动作类型分组：完全匹配的覆盖率（即 set/total 的比例，越低说明变体越多）')
    type_stats = defaultdict(lambda: {'total': 0, 'unique': set()})
    for r in all_results:
        # 重新加载原始动作做类型统计
        name = r['name']
        fname = dict([(n[0], n[1]) for n in files])[name]
        path = os.path.join(TRAJ_DIR, fname)
        traj_actions = load_actions_from_file(path)
        for traj in traj_actions:
            for step in traj:
                for env_idx, act in step.items():
                    if act.startswith('search['):
                        t = 'search'
                    elif act.startswith('answer['):
                        t = 'answer'
                    else:
                        t = 'null'
                    type_stats[t]['total'] += 1
                    type_stats[t]['unique'].add(act)

    print(f'{"类型":<10} {"总动作":>8} {"唯一动作":>8} {"去重率%":>8}')
    print('-' * 120)
    for t, st in sorted(type_stats.items()):
        u = len(st['unique'])
        total = st['total']
        print(f'{t:<10} {total:>8} {u:>8} {u/total*100 if total else 0:>7.2f}%')


if __name__ == '__main__':
    main()
