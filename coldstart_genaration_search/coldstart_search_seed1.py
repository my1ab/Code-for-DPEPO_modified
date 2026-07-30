"""
Search Task Coldstart Data Generation Script
Based on: coldstart_test_search/coldstart_para_his_test_1.5B_hislen8_epoch3.5_v2.py

Generates coldstart trajectories for the Search-R1 task.
The agent interacts with a search engine to answer questions.
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
import json
import time
import re
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'verl-agent'))

# from agent_system.environments.env_package.webshop import build_webshop_envs
from agent_system.environments.env_package.search import build_search_envs
print("import build_search_envs success")

from coldstart_genaration_search.prompts_search import (
    SYSTEM_PROMPT_SEARCH_PARA,
    USER_PROMPT_NO_HIS_PARA,
    USER_PROMPT_HIS_PARA,
)
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# Global Configuration — 统一在此处设置所有超参数
# ============================================================
local_model = None
local_tokenizer = None

# ── 仅在函数内部使用的全局常量 ────────────────────────
MAX_CONTEXT_LENGTH = 32768    # 本地模型最大上下文长度 (仅本地模式生效)
BASE_MODEL_PATH = '/diskpool/home/xuxz/ms-swift/model/Qwen2.5-1.5B-Instruct'  # 本地模型路径 (仅本地模式生效)
# SEARCH_URL = 'http://127.0.0.1:8000/retrieve'  # 检索服务器地址
# 自定义端口
PORT=8010
SEARCH_URL = f'http://127.0.0.1:{PORT}/retrieve'
SEARCH_TOPK = 3               # 每次搜索返回的 top-k 文档数
SEARCH_TIMEOUT = 60           # 搜索请求超时时间 (秒)
SEARCH_LOG_REQUESTS = False   # 是否记录搜索请求日志

# ── 数据路径 ────────────────────────────────────────
SEARCH_DATA_DIR = os.path.expanduser('~/data/searchR1_processed_direct')



# ============================================================
# Model Loading
# ============================================================
def load_local_model(tokenizer_path=None, model_path=None, show=1):
    global local_model, local_tokenizer
    if model_path is not None:
        print(f"\n{'='*60}")
        print(f"Loading tokenizer and model from checkpoint: {tokenizer_path}")
        print(f"{'='*60}")

        print("Loading tokenizer...")
        try:
            local_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                local_files_only=True
            )
            print(f"Tokenizer loaded: {local_tokenizer.__class__.__name__}")

            if hasattr(local_tokenizer, 'model_max_length'):
                local_tokenizer.model_max_length = MAX_CONTEXT_LENGTH
                print(f"Set tokenizer model_max_length to: {MAX_CONTEXT_LENGTH}")

            local_tokenizer.truncation_side = "right"
            print(f"Set tokenizer truncation_side to: {local_tokenizer.truncation_side}")
            print(f"{'='*60}")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            import traceback
            traceback.print_exc()
            raise

        if show:
            print(f"\nLoading model weights from {model_path}")
        local_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype='bfloat16',
            device_map='auto',
            attn_implementation='flash_attention_2',
            local_files_only=True,
            use_safetensors=True
        )
        local_model.eval()

        if show:
            print(f"\n{'='*60}")
            print(f"Model loaded successfully!")
            print(f"Model type: {local_model.__class__.__name__}")
            print(f"Model device: {local_model.device}")
            print(f"Model dtype: {local_model.dtype}")
            print(f"Number of parameters: {sum(p.numel() for p in local_model.parameters()):,}")
            print(f"Max context length: {MAX_CONTEXT_LENGTH}")
            print(f"{'='*60}")

        print("load_local_model success")
        print(f"{'='*60}")
    return local_model, local_tokenizer


def local_model_infer(messages, max_new_tokens=4096, show=0):
    """Run inference with the local model."""
    model, tokenizer = load_local_model()
    text = tokenizer.apply_chat_template(messages, tokenize=False)

    max_input_length = MAX_CONTEXT_LENGTH - max_new_tokens

    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=False,
        padding=False
    ).to(model.device)

    input_length = inputs['input_ids'].shape[1]

    if input_length > max_input_length:
        if show:
            print(f"WARNING: Input truncated to max length!")
            print(f"Original input tokens: {input_length}, truncating to {max_input_length}")
        inputs = {
            'input_ids': inputs['input_ids'][:, :max_input_length],
            'attention_mask': inputs['attention_mask'][:, :max_input_length]
        }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
            use_cache=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    actual_input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    if response.startswith(actual_input_text):
        response = response[len(actual_input_text):].strip()
    return response

from openai import OpenAI, APIConnectionError, APITimeoutError

DEEPSEEK_RETRY_MAX = 5          # 最大重试次数
DEEPSEEK_RETRY_INTERVAL = 1     # 重试间隔（秒）


def deepseek(messages, ds_model=1, effort=0, show=0, turn=None):
    """Use DeepSeek API for inference. 自动重试最多 DEEPSEEK_RETRY_MAX 次。

    Args:
        turn: 当前轮次，用于日志打印（可选）。
    """
    # client = OpenAI(api_key="sk-b718f52386c34ffeb714f684d225f688", base_url="https://api.deepseek.com")
    client = OpenAI(api_key="sk-a8d675e7b9f14343b8d38e23718fc21a", base_url="https://api.deepseek.com")

    # 强制flash + high
    ds_model=1 
    effort=1
    model_name = "deepseek-v4-flash" if ds_model == 1 else "deepseek-v4-pro"

    turn_tag = f"[turn {turn}] "

    last_exception = None
    for attempt in range(1, DEEPSEEK_RETRY_MAX + 1):
        try:
            if effort == 0:
                response = client.chat.completions.create(
                    model="deepseek-v4-flash",
                    extra_body={"thinking": {"type": "disabled"}},
                    messages=messages,
                    stream=False,
                )
            else:
                reasoning_effort = "high" if effort == 1 else "max"
                response = client.chat.completions.create(
                    model="deepseek-v4-flash",
                    extra_body={"thinking": {"type": "enabled"}},
                    reasoning_effort=reasoning_effort,
                    messages=messages,
                    stream=False,
                )

            if show:
                print(f'{turn_tag}model: {model_name}, effort: {"disabled" if effort == 0 else reasoning_effort}')
            return response.choices[0].message.content

        except (APIConnectionError, APITimeoutError) as e:
            # 连接类异常则重试
            last_exception = e
            if attempt < DEEPSEEK_RETRY_MAX:
                print(f"  {turn_tag}[Retry {attempt}/{DEEPSEEK_RETRY_MAX}] DeepSeek API 连接失败，{DEEPSEEK_RETRY_INTERVAL}s 后重试...")
                time.sleep(DEEPSEEK_RETRY_INTERVAL)
            else:
                print(f"  {turn_tag}[Retry {attempt}/{DEEPSEEK_RETRY_MAX}] 已达最大重试次数，放弃。")
                raise last_exception
        except Exception as e:
            # 非连接类异常（如 API key 错误、请求格式错误等）不重试，直接抛出
            raise e


# ============================================================
# Model Test — 以 ChatML 格式测试 DeepSeek API 可达性
# ============================================================
def test_model(ds_model=1, effort=0, use_local_model=False):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "introduce yourself."},
    ]

    if use_local_model:
        print("=" * 60)
        print("Testing Local model connectivity...")
        print("=" * 60)
        try:
            response = local_model_infer(messages, show=1)
            print(f"\nResponse: {response}")
            print(f"\n{'='*60}")
            print("Local model test PASSED!")
            print(f"{'='*60}")
            return True
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"Local model test FAILED: {e}")
            print(f"{'='*60}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("=" * 60)
        print("Testing DeepSeek API connectivity...")
        print("=" * 60)
        try:
            response = deepseek(messages, ds_model=ds_model, effort=effort, show=1)
            print(f"\nResponse: {response}")
            print(f"\n{'='*60}")
            print("DeepSeek API test PASSED!")
            print(f"{'='*60}")
            return True
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"DeepSeek API test FAILED: {e}")
            print(f"{'='*60}")
            return False


# ============================================================
# Prompt Helpers
# ============================================================
def format_parallel_observations(obs_list, total_envs):
    """Format observations from multiple envs into tagged string."""
    parts = []
    for i in range(total_envs):
        obs = obs_list[i] if i < len(obs_list) else ""
        parts.append(f"<observation_{i+1}>\n{obs}\n</observation_{i+1}>")
    return "\n".join(parts)


def format_parallel_history(history_list, turn, his_len, total_envs):
    """Format per-env history into prompt string (v2 reference style)."""
    lines = []
    for env_idx in range(total_envs):
        env_lines = [f"In Environment {env_idx+1}"]
        if his_len < 0:
            start = 0
        else:
            start = max(0, turn - his_len)
        for t_idx in range(start, turn):
            action = history_list[env_idx]['Action'][t_idx]
            obs = history_list[env_idx]['Observation'][t_idx]
            obs_trimmed = obs[:2000] if obs and len(obs) > 2000 else (obs or "")
            env_lines.append(f"  Action {t_idx+1}: {action}")
            env_lines.append(f"  Observation {t_idx+1}: {obs_trimmed}")
        lines.append("\n".join(env_lines))
    return "\n\n".join(lines)


def format_parallel_last_history(history_list, turn, total_envs):
    """Format only the most recent step from each env."""
    lines = []
    for env_idx in range(total_envs):
        action = history_list[env_idx]['Action'][-1] if history_list[env_idx]['Action'] else "null"
        obs = history_list[env_idx]['Observation'][-1] if history_list[env_idx]['Observation'] else ""
        obs_trimmed = obs[:2000] if obs and len(obs) > 2000 else (obs or "")
        lines.append(f"In Environment {env_idx+1}")
        lines.append(f"  Action {turn}: {action}")
        lines.append(f"  Observation {turn}: {obs_trimmed}")
    return "\n".join(lines)


def get_search_system_message_para(num_para, total_envs):
    return {"role": "system", "content": SYSTEM_PROMPT_SEARCH_PARA.format(
        num_parallel=num_para, total_envs=total_envs
    )}


def get_search_user_message_para(question, initial_observations, total_envs, num_para,
                                 history_info=None, last_history=None):
    init_obs_str = format_parallel_observations(initial_observations, total_envs)
    if history_info is not None and last_history is not None:
        content = USER_PROMPT_HIS_PARA.format(
            task_description=question,
            initial_observation=init_obs_str,
            history_info=history_info,
            last_history=last_history,
            total_envs=total_envs,
            num_parallel=num_para,
        )
    else:
        content = USER_PROMPT_NO_HIS_PARA.format(
            question=question,
            observations=init_obs_str,
            total_envs=total_envs,
            num_parallel=num_para,
        )
    return {"role": "user", "content": content}


# ============================================================
# Action Extraction
# ============================================================
def extract_parallel_search_action(text, total_envs=1):
    """Extract parallel actions from model output with <parallel>/<env_i> tags.

    Format:
      <parallel>
      <env_1><search>query1</search></env_1>
      <env_2><answer>answer2</answer></env_2>
      </parallel>

    Returns:
        dict with keys:
          - think: str or None
          - actions: dict {env_idx: {"type": "search"/"answer", "content": "...", "raw": "..."}}
          - all_null: bool
    """
    # Extract think
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, text, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else None

    # Extract parallel env actions
    actions = {}
    parallel_pattern = r'<parallel>(.*?)</parallel>'
    parallel_match = re.search(parallel_pattern, text, re.DOTALL)
    if parallel_match:
        parallel_content = parallel_match.group(1)
        env_pattern = r'<env_(\d+)>(.*?)</env_\d+>'
        for env_idx_str, raw_action in re.findall(env_pattern, parallel_content, re.DOTALL):
            env_idx = int(env_idx_str) - 1  # convert to 0-based
            raw_action = raw_action.strip()
            # Parse inner action type
            inner_search = re.search(r'<search>(.*?)</search>', raw_action, re.DOTALL)
            inner_answer = re.search(r'<answer>(.*?)</answer>', raw_action, re.DOTALL)
            # search专用动作处理
            if inner_search:
                actions[env_idx] = {
                    "type": "search",
                    "content": inner_search.group(1).strip(),
                    "raw": raw_action,
                }
            elif inner_answer:
                actions[env_idx] = {
                    "type": "answer",
                    "content": inner_answer.group(1).strip(),
                    "raw": raw_action,
                }
            else:
                actions[env_idx] = {
                    "type": None,
                    "content": raw_action,
                    "raw": raw_action,
                }

    # Fill in null for missing envs
    for i in range(total_envs):
        if i not in actions:
            actions[i] = {"type": None, "content": None, "raw": "null"}

    all_null = all(a["type"] is None for a in actions.values())

    return {
        'think': think_content,
        'actions': actions,
        'all_null': all_null,
    }


# ============================================================
# Unique filename helper
# ============================================================
def get_unique_filename(file_path):
    if not os.path.exists(file_path):
        return file_path
    dir_name = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    name, ext = os.path.splitext(file_name)
    counter = 1
    while True:
        new_file_name = f"{name}_{counter}{ext}"
        new_file_path = os.path.join(dir_name, new_file_name) if dir_name else new_file_name
        if not os.path.exists(new_file_path):
            return new_file_path
        counter += 1


# ============================================================
# Trajectory Generation
# ============================================================
def get_single_trajectory(env, question, ground_truth, data_source,
                          physical_idx=0, logical_idx=0,
                          max_turns=10, show_turn=False, his_len=5,
                          use_local_model=True, ds_model=1, effort=0,
                          group_n=1, env_num=1, num_para=1):
    """
    Generate a single trajectory for one search question,
    supporting parallel multi-environment exploration.

    ═══════════════════════════════════════════════════════════════
    SEARCH ENV API (compared to WebShop):

    env.reset(kwargs: List[Dict]) -> (obs_list, info_list)
      - kwargs MUST contain: "ground_truth", "question", "data_source"
      - obs_list: the questions themselves (strings)
      - info_list: metadata dicts

    env.step(actions: List[str]) -> (obs_list, reward_list, done_list, info_list)
      - ALWAYS returns exactly 4 lists
      - obs_list: search results (str) if search action, "" if answer action
      - reward_list: float (only meaningful when done=True, 1.0=correct)
      - done_list: bool (True if <answer> found or max_turns reached)
      - info_list: dict with key "won" = bool(done and reward >= 1.0)
    ═══════════════════════════════════════════════════════════════

    Args:
        env: SearchMultiProcessEnv instance
        question: The question string
        ground_truth: Ground truth answer(s)
        data_source: Data source name
        physical_idx: 真实行索引 (数据文件中的行号)
        logical_idx: 逻辑位置索引 (随机数表中的位置)
        max_turns: Maximum number of search/answer turns
        show_turn: Whether to print turn info
        his_len: History window length (-1 for full history)
        use_local_model: Use local model vs deepseek API
        group_n: Number of groups (for batch sizing)
        env_num: Number of envs per group
        num_para: Max number of parallel actions per turn

    Returns:
        (messages, success_flag, status_msg, seperated_list)
    """
    total_envs = group_n * env_num
    num_para = min(num_para, total_envs)

    messages = []
    success_flag = 0
    seperated_list = []

    # ── Reset environment with duplicated kwargs ────────────────
    kwargs = [{
        "ground_truth": ground_truth,
        "question": question,
        "data_source": data_source
    }] * total_envs
    # 
    obs_list, info_list = env.reset(kwargs)
    current_obs_list = list(obs_list)
    initial_obs_list = list(obs_list)  # 保存起始观察，供历史轮模板的 {initial_observation} 使用

    # ── Per-environment history ─────────────────────────────────
    history_list = []
    for _ in range(total_envs):
        history_list.append({'Action': [], 'Observation': []})

    # ── System message ──────────────────────────────────────────
    messages.append(get_search_system_message_para(num_para, total_envs))

    null_count = 0

    for turn in range(max_turns):
        # ── Build prompts with observations ─────────────────────
        if turn == 0:
            user_msg = get_search_user_message_para(
                question, initial_obs_list, total_envs, num_para,
                history_info=None, last_history=None
            )
        else:
            # Parallel history format
            history_info = format_parallel_history(history_list, turn, his_len, total_envs)
            last_history = format_parallel_last_history(history_list, turn, total_envs)
            user_msg = get_search_user_message_para(
                question, initial_obs_list, total_envs, num_para,
                history_info=history_info, last_history=last_history
            )

        messages.append(user_msg)

        # ── Prepare partial messages (for training data) ─────────
        partial_messages = [
            {"physical_idx": physical_idx, "logical_idx": logical_idx, "turn": turn},
            messages[0],  # system message
            messages[-1],  # current turn user message
        ]

        # ── Model inference ──────────────────────────────────────
        if use_local_model:
            assistant_response = local_model_infer(messages=partial_messages[1:])
        else:
            assistant_response = deepseek(messages=partial_messages[1:],
                                          ds_model=ds_model, effort=effort,
                                          turn=turn)

        partial_messages.append({"role": "assistant", "content": assistant_response})
        seperated_list.append({"messages": partial_messages.copy()})

        # ── Extract actions ──────────────────────────────────────
        para_result = extract_parallel_search_action(assistant_response, total_envs)
        step_actions_dict = {}
        for env_idx in range(total_envs):
            a = para_result['actions'][env_idx]
            step_actions_dict[env_idx] = a["raw"] if a["type"] is not None else "null"
        all_null = para_result['all_null']

        # ── Build action list for env.step() ─────────────────────
        env_actions = []
        for env_idx in range(total_envs):
            raw = step_actions_dict.get(env_idx, "null")
            if raw == "null" or raw is None:
                env_actions.append("")
            else:
                env_actions.append(raw)

        # ── Step the environment ─────────────────────────────────
        if not all_null:
            obs_list, reward_list, done_list, info_list = env.step(env_actions)
            null_count = 0
        else:
            # All null — skip step, keep observations as-is
            null_count += 1
            obs_list = [""] * total_envs
            reward_list = [None] * total_envs
            done_list   = [False] * total_envs
            info_list   = [{}] * total_envs

        # Truncate to match total_envs (v2 reference style)
        new_obs_list = list(obs_list)[:total_envs]
        new_reward_list = list(reward_list)[:total_envs]
        new_done_list = list(done_list)[:total_envs]
        new_info_list = list(info_list)[:total_envs]

        # ── Record history for each env ──────────────────────────
        for env_idx in range(total_envs):
            action_raw = step_actions_dict.get(env_idx, "null")
            action_content = action_raw
            inner_s = re.search(r'<search>(.*?)</search>', action_raw, re.DOTALL)
            inner_a = re.search(r'<answer>(.*?)</answer>', action_raw, re.DOTALL)
            if inner_s:
                action_content = inner_s.group(1).strip()
            elif inner_a:
                action_content = inner_a.group(1).strip()
            history_list[env_idx]['Action'].append(action_content if action_content else 'null')
            history_list[env_idx]['Observation'].append(new_obs_list[env_idx])

        current_obs_list = new_obs_list

        # ── Build single assistant message ───────────────────────
        assistant_msg = {
            # "physical_idx": physical_idx,
            # "logical_idx": logical_idx,
            "role": "assistant",
            "content": assistant_response,
            "rewards": json.dumps(new_reward_list),
            "get_actions": step_actions_dict,
            "think": para_result['think'],
        }
        if show_turn:
            assistant_msg["turn"] = turn + 1
        messages.append(assistant_msg)

        # ── Null termination check ───────────────────────────────
        if all_null and null_count >= 2:
            # status_msg = f"Task physical_idx={physical_idx}, logical_idx={logical_idx} exit(all null) at turn {turn + 1}"
            status_msg = f"Task {logical_idx} exit(all null) at turn {turn + 1} (physical_idx={physical_idx})"
            if show_turn:
                print(status_msg)
            return messages, success_flag, status_msg, seperated_list

        # ── Termination conditions ───────────────────────────────
        any_done = any(new_done_list)
        any_success = any(r is not None and r >= 1.0 for r in new_reward_list)

        if any_done:
            completed_idx = [i for i, d in enumerate(new_done_list[:total_envs]) if d]
            if any_success:
                success_flag = 1
                # status_msg = f"Task physical_idx={physical_idx}, logical_idx={logical_idx} SUCCESS at turn {turn + 1} in environments {completed_idx} data_source={data_source}"
                status_msg = f"Task {logical_idx} SUCCESS at turn {turn + 1} in envs {completed_idx} data_source={data_source} (physical_idx={physical_idx})"
            else:
                # status_msg = f"Task physical_idx={physical_idx}, logical_idx={logical_idx} FAILED at turn {turn + 1} in environments {completed_idx}"
                status_msg = f"Task {logical_idx} FAILED at turn {turn + 1} in envs {completed_idx} data_source={data_source} (physical_idx={physical_idx})"
            if show_turn:
                print(status_msg)
            break

    else:
        # status_msg = f"Task physical_idx={physical_idx}, logical_idx={logical_idx} out of max turn"
        status_msg = f"Task {logical_idx} out of max turn (physical_idx={physical_idx})"
        if show_turn:
            print(status_msg)

    return messages, success_flag, status_msg, seperated_list


# ============================================================
# Task Sampler — 管理与 WebShop 环境类似的采样逻辑
# ============================================================
class SearchTaskSampler:
    """Search 任务的采样管理器，模仿 WebShop 的 goal 管理方式。

    WebShop (
      agent_system/environments/env_package/webshop/envs.py
      └── WebshopMultiProcessEnv
    ) 在 __init__ 中加载所有 goal 并通过 split 确定可用索引范围,
    然后在 reset() 中按 sequential / exclude / random 模式采样。

    Search 没有内置的环境级采样（SearchMultiProcessEnv 只透传外部参数），
    因此本 Sampler 在外部实现等效逻辑。

    用法:
        sampler = SearchTaskSampler(
            data_path="~/data/searchR1_processed_direct/test.parquet",
            split="test",
            seed=-1,        # <0 → sequential, >=0 → random
        )
        # 通过 current_idx 选取随机数表中的元素
        tasks = sampler.sample(current_idx=42)
        for task in tasks:
            question = task["question"]
            ground_truth = task["ground_truth"]
            ...
    """

    # ── split → parquet 文件名 / 行索引范围的映射 ──────────────
    # 对应 WebShop 的 goal_idxs 逻辑:
    #   split='test' → range(500),  test.parquet
    #   split='train' → range(1500, len(goals)),  train.parquet
    #   split='sft'   → range(600, len(goals)),  train.parquet (与 train 同文件)
    _SPLIT_CONFIG = {
        "test":  {"file": "test.parquet",  "range": (0, None)},
        "train": {"file": "train.parquet", "range": (0, None)},
        # "sft":   {"file": "train.parquet", "range": (600, None)},
        # "sft":   {"file": "train.parquet", "range": (0, None)},
        # "sft":   {"file": "train.parquet", "range": (0, 79167)},
        "sft":   {"file": "train.parquet", "range": (0, None)},
        # "all":   {"file": "test.parquet",  "range": (0, None)},
    }

    def __init__(
        self,
        data_dir: str = None,
        split: str = "test",
        seed: int = 42,
        batch_size: int = 1,
    ):
        """
        Args:
            data_dir: parquet 所在目录, 默认使用 SEARCH_DATA_DIR
            split: 数据分区名, 决定文件和行范围:
                     test  → test.parquet  (0~499)
                     train → train.parquet (1500~end)
                     sft   → train.parquet (0~end)
            seed: 随机种子; <0 → sequential 顺序采样, >=0 → 随机采样 (使用 seed 初始化 RNG)
            batch_size: 每次 sample() 返回的任务数
        """
        data_dir = SEARCH_DATA_DIR

        # 根据 split 确定 parquet 文件
        # "sft":   {"file": "train.parquet", "range": (600, None)}
        split_cfg = self._SPLIT_CONFIG.get(split)
        if split_cfg is None:
            raise ValueError(f"Unknown split '{split}'. Valid: {list(self._SPLIT_CONFIG.keys())}")

        data_path = os.path.join(data_dir, split_cfg["file"])
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data not found: {data_path}")

        print(f"SearchTaskSampler loading: {data_path}")
        self._df = pd.read_parquet(data_path)
        self._total_rows = len(self._df)
        print(f"  Total rows: {self._total_rows}")

        # 确定可用索引范围 (对应 WebShop 的 self.goal_idxs)
        self.start, self.end = split_cfg["range"]

        if self.end is None:
            self.end = self._total_rows
        else:
            self.end = min(self.end, self._total_rows)
        self.start = min(self.start, self.end)

        self._goal_idxs = list(range(self.start, self.end))
        print(f"  Split '{split}': goal_idxs = [{self.start}, {self.end})  ({len(self._goal_idxs)} tasks)")

        # ── 采样模式选择 ────────────────────────────────────────
        # seed < 0 → sequential 顺序采样, seed >= 0 → 随机采样
        if seed < 0:
            self._sequential = True
            self._seed = seed
            self._rng = None
            mode = "sequential"
        else:
            self._seed = seed
            self._rng = np.random.RandomState(seed)
            self._sequential = False
            mode = f"random (seed={seed})"

        self._batch_size = batch_size
        self._shuffle_table = None    # 随机数表，首次 sample() 时构建

        print(f"Sampling mode: {mode}")
        # print(f"Seed: {seed}")

    # ── 公开接口 ─────────────────────────────────────────────────

    @property
    def goal_start(self) -> int:
        """当前采样的起始索引。"""
        return self._goal_idxs[0] if self._goal_idxs else 0

    @property
    def goal_end(self) -> int:
        """当前采样的结束索引(不含)。"""
        return self._goal_idxs[-1] + 1 if self._goal_idxs else 0

    @property
    def total_tasks(self) -> int:
        return len(self._goal_idxs)

    @property
    def seed(self) -> int:
        """创建时传入的随机种子。"""
        return self._seed

    @property
    def remaining(self) -> int:
        return len(self._shuffle_table) if self._shuffle_table is not None else len(self._goal_idxs)

    def has_next(self) -> bool:
        return True

    def reset(self) -> None:
        """重置采样状态 (清空随机数表，下次 sample 时重建)。"""
        self._shuffle_table = None

    # table_size=10000
    def _build_shuffle_table(self):
        """构建随机数表：固定长度 1000，每个元素是 _goal_idxs 中的位置索引。
        
        - seed < 0 (sequential): 按顺序填充 [0, 1, 2, ..., 999] % n
        - seed >= 0 (random): 从 RNG 随机抽取 (可重复)
        
        Args:
            table_size: 随机数表长度; 默认 None 则使用 self.end - self.start
        """
        n = len(self._goal_idxs)
    
        table_size = self.end - self.start
        if n == 0:
            self._shuffle_table = []
            return
        if self._sequential:
            self._shuffle_table = [i for i in range(table_size)]
        else:
            # self._shuffle_table = list(self._rng.choice(n, size=table_size, replace=True))
            self._shuffle_table = list(self._rng.choice(n, size=table_size, replace=False))
        print(f'Shuffle table built: {len(self._shuffle_table)}/{table_size} entries')

    def sample(self, current_idx: int = 0) -> list[dict]:
        """从随机数表中选取一个任务。

        首次调用时自动构建随机数表。

        Args:
            current_idx: 随机数表中的位置索引 (0-based)

        Returns:
            list[dict]: 包含一个 dict，字段:
                "question":     str
                "ground_truth": str or dict
                "data_source":  str
                "physical_idx": int (真实行索引)
                "logical_idx":  int (随机数表中的逻辑位置)
        """
        if self._shuffle_table is None:
            raise KeyError("self._shuffle_table not built yet. Call _build_shuffle_table() to build it.")
        # self._build_shuffle_table()

        if current_idx < 0 or current_idx >= len(self._shuffle_table):
            raise IndexError(
                f"current_idx {current_idx} out of range [0, {len(self._shuffle_table)})"
            )

        pos_in_goal = self._shuffle_table[current_idx]
        idx = self._goal_idxs[pos_in_goal]

        row = self._df.iloc[idx]
        env_kwargs = row.get("env_kwargs", {})
        if isinstance(env_kwargs, str):
            env_kwargs = json.loads(env_kwargs)

        question = env_kwargs.get("question") or row.get("question")
        ground_truth = env_kwargs.get("ground_truth") or row.get("ground_truth")
        data_source = env_kwargs.get("data_source") or row.get("data_source")

        if not question:
            raise KeyError(f"Missing 'question' in row {idx} (env_kwargs={env_kwargs})")
        if not ground_truth:
            raise KeyError(f"Missing 'ground_truth' in row {idx}")
        if not data_source:
            raise KeyError(f"Missing 'data_source' in row {idx}")

        return [{
            "question": question,
            "ground_truth": ground_truth,
            "data_source": data_source,
            "physical_idx": int(idx),
            "logical_idx": current_idx,
        }]

    def __len__(self) -> int:
        return self.total_tasks

    def __repr__(self) -> str:
        tbl_status = "built" if self._shuffle_table is not None else "unbuilt"
        return (
            f"SearchTaskSampler(total={self.total_tasks}, "
            f"shuffle_table={tbl_status}, "
            f"sequential={self._sequential})"
        )


# ============================================================
# Main Evaluation
# ============================================================
def evaluate_coldstart_data(output_file, sampler, max_turns=25,
                            show_turn=False, his_len=5,
                            save_traj=1, use_local_model=True,
                            ds_model=1, effort=0,
                            start_idx=None, end_idx=None,
                            group_n=1, env_num=1, num_para=1):
    """
    Generate coldstart trajectories for search task.

    Args:
        output_file: JSON 输出路径
        sampler: SearchTaskSampler 实例 (管理采样顺序和范围)
        max_turns: 每问最大搜索轮数
        show_turn: 是否逐轮打印状态
        his_len: 历史窗口长度
        save_traj: 是否保存轨迹
        use_local_model: 是否使用本地模型
        ds_model: DeepSeek 模型 (1=Flash, 2=Pro)
        effort: DeepSeek thinking 级别
        start_idx: 在随机序列中的起始逻辑位置
        end_idx: 在随机序列中的结束逻辑位置
        group_n: 并行组数
        env_num: 每组环境数
        num_para: 每轮最大并行动作数
    """
    log_file = output_file.replace('.json', '.log')

    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))

    print(f'Start time: {start_time_str}')
    print(f'log_file: {log_file}')
    print(f"{'─'*80}")

    # Log configuration
    total_envs = group_n * env_num
    config_lines = [
        f'Evaluating search coldstart: {sampler.total_tasks} samples',
        f'max_turns: {max_turns}, his_len: {his_len}',
        f'use_local_model: {use_local_model}',
        f'sampler: {sampler}',
        f'search_url: {SEARCH_URL}, topk: {SEARCH_TOPK}',
        f'group_n: {group_n}, env_num: {env_num}, total_envs: {total_envs}, num_para: {num_para}',
    ]
    with open(log_file, 'w') as f:
        for line in config_lines:
            f.write(line + '\n')
            print(line)

    # Build environment
    print(f'building environment')
    env_config = OmegaConf.create({
        # 环境知道max_turns  会强制判0
        'max_steps': max_turns,
        'search': {
            'search_url': SEARCH_URL,
            'topk': SEARCH_TOPK,
            'timeout': SEARCH_TIMEOUT,
            'log_requests': SEARCH_LOG_REQUESTS,
        }
    })

    # 调用接口 (seed 必须 ≥0, sampler 中 seed<0 表示强制 sequential 模式)
    env_seed = max(sampler.seed, 0)
    env = build_search_envs(
        seed=env_seed,
        env_num=env_num,
        group_n=group_n,
        is_train=False,
        env_config=env_config,
    )

    print(f'Environment built successfully')

    # Track results
    all_trajectories = []
    success_trajectories = []
    seperated_trajectories = []
    success_indices = []
    success_count = 0
    total_count = 0

    # ── 用 start_idx/end_idx 从随机数表中定位逻辑分片 ─────
    if start_idx is not None and end_idx is not None:
        if end_idx < start_idx:
            raise ValueError(f"end_idx ({end_idx}) must be >= start_idx ({start_idx})")
        print(f"Processing logical range: [{start_idx}, {end_idx}]")

    for pos in tqdm(range(start_idx, end_idx + 1), desc="Evaluating test data"):
        tasks = sampler.sample(current_idx=pos)
        if not tasks:
            break
        task = tasks[0]
        physical_idx = task["physical_idx"]
        logical_idx = task["logical_idx"]
        question = task["question"]
        ground_truth = task["ground_truth"]
        data_source = task["data_source"]

        if not question:
            print(f"WARNING: Skipping physical_idx={physical_idx}, logical_idx={logical_idx} - empty question")
            continue

        try:
            trajectory, success_flag, status_msg, seperated_list = get_single_trajectory(
                env=env,
                question=question,
                ground_truth=ground_truth,
                data_source=data_source,
                physical_idx=physical_idx,
                logical_idx=logical_idx,
                max_turns=max_turns,
                show_turn=show_turn,
                his_len=his_len,
                use_local_model=use_local_model,
                ds_model=ds_model,
                effort=effort,
                group_n=group_n,
                env_num=env_num,
                num_para=num_para,
            )

            # 保存轨迹时附带 physical_idx 和 logical_idx
            all_trajectories.append({
                "physical_idx": physical_idx,
                "logical_idx": logical_idx,
                "trajectory": trajectory
            })
            seperated_trajectories.append(seperated_list)
            total_count += 1

            with open(log_file, 'a') as f:
                f.write(status_msg + '\n')

            if success_flag == 1:
                success_trajectories.append({
                    "physical_idx": physical_idx,
                    "logical_idx": logical_idx,
                    "trajectory": trajectory
                })
                # success_indices.append(physical_idx)
                success_indices.append(logical_idx)
                success_count += 1

        except Exception as e:
            error_msg = f"Error evaluating physical_idx={physical_idx}, logical_idx={logical_idx}: {e}"
            print(f"\n  {error_msg}")
            with open(log_file, 'a') as f:
                f.write(error_msg + '\n')
            import traceback
            traceback.print_exc()
            continue

    # Calculate success rate
    success_rate = success_count / total_count if total_count > 0 else 0
    print(f"\n{'='*60}")
    print(f"Total: {total_count}, Success: {success_count}, Rate: {success_rate:.2%}")
    print(f"{'='*60}")

    # Save results
    if save_traj:
        with open(output_file, 'w') as f:
            json.dump(all_trajectories, f, indent=4)
        print(f"Trajectories saved to {output_file}")

        if success_trajectories:
            success_output = output_file.replace('.json', '_success.json')
            with open(success_output, 'w') as f:
                json.dump(success_trajectories, f, indent=4)
            print(f"Success trajectories saved to {success_output}")

        # if seperated_trajectories:
        #     seperated_output = output_file.replace('.json', '_seperated.json')
        #     with open(seperated_output, 'w') as f:
        #         json.dump(seperated_trajectories, f, indent=4)
        #     print(f"Seperated data saved to {seperated_output}")
        #     print(f"Seperated count: {len(seperated_trajectories)}")

    # Final log
    with open(log_file, 'a') as f:
        f.write(f"\nTotal: {total_count}, Success: {success_count}, Rate: {success_rate:.2%}\n")
        f.write(f"Success indices: {success_indices}\n")
        f.write(f"Seperated count: {len(seperated_trajectories)}\n")
        if save_traj:
            f.write(f"Trajectories saved to {output_file}\n")
            if success_trajectories:
                success_output = output_file.replace('.json', '_success.json')
                f.write(f"Success trajectories saved to {success_output}\n")
            if seperated_trajectories:
                seperated_output = output_file.replace('.json', '_seperated.json')
                f.write(f"Seperated data saved to {seperated_output}\n")

    env.close()

    end_time = time.time()
    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    elapsed_time = end_time - start_time
    print(f'Start time: {start_time_str}')
    print(f'End time: {end_time_str}')
    print(f'Elapsed time: {elapsed_time:.1f}s')

    with open(log_file, 'a') as f:
        f.write(f"Start time: {start_time_str}\n")
        f.write(f"End time: {end_time_str}\n")
        f.write(f"Elapsed time: {elapsed_time:.1f}s\n")

    return all_trajectories


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    """
    ================================================================
    Search Coldstart Data Generation
    ================================================================
    REQUIREMENTS:
    1. A running search/retrieval server at SEARCH_URL (default: http://127.0.0.1:8000/retrieve)
       - Start with: bash examples/search/retriever/retrieval_launch.sh
    2. Preprocessed search data at SEARCH_DATA_DIR (default: ~/data/searchR1_processed_direct/)
       - Generate with: python examples/data_preprocess/preprocess_search_r1_dataset.py
    3. (Optional) Local model checkpoint for local inference
       - Or use DeepSeek API (use_local_model=False)
    ================================================================

    split → parquet 文件:
      test  → test.parquet  (range 0~499)
      train → train.parquet (range 1500~end)
      sft   → train.parquet (range 600~end)
    ================================================================
    """
    

    # ================================================================
    # Sampler & Output Setup - 读取全局超参数
    # ================================================================
    # 本地模型校验点路径 (仅 USE_LOCAL_MODEL=True 时生效)
    # CHECKPOINT_PATH = '/diskpool/home/xuxz/ms-swift/checkpoint/Qwen2.5-1.5B-Instruct-Parallel-Epoch5/...'
    # load_local_model(tokenizer_path=CHECKPOINT_PATH, model_path=CHECKPOINT_PATH, show=1)

    # ── 局部超参数 (手动设定) ─────────────────────────────
    # 模型推理
    ds_model = 1               # DeepSeek 模型: 1=Flash, 2=Pro (仅 API 模式生效)
    effort = 1                 # DeepSeek thinking: 0=disabled, 1=high, 2=max

    # 环境交互
              # 每问最大搜索/回答轮数

    # 历史窗口
    his_len = 8                # 上下文历史窗口长度 (-1=使用全部历史)

    # 并行探索设置
    group_n = 5                # 并行组数
    env_num = 1                # 每组环境数 (total_envs = group_n * env_num)
    num_para = group_n               # 每轮最大并行动作数 (<= total_envs)

    # 采样控制
    seed = 1                  # 随机种子: <0 → sequential, >=0 → 随机采样
                 # 数据分区: test/train/sft, 决定 parquet 文件和行范围
    show_turn = True           # 是否逐轮打印状态信息

    

    # ckpt_path = "/diskpool/home/xuxz/ms-swift/checkpoint_search/Qwen2.5-1.5B-Instruct-Parallel-Epoch5-hislen8/v0-20260705-053350/checkpoint-7230"
    ckpt_path = "/diskpool/home/xuxz/ms-swift/checkpoint_search/Qwen2.5-1.5B-Instruct-Parallel-Epoch5-hislen8/v0-20260705-053350/checkpoint-4500"
    # load_local_model(tokenizer_path=ckpt_path, model_path=ckpt_path, show=1)
    test_model(ds_model, effort, use_local_model=False)
    print(f'[DEBUG] test done')

    # ── Sampler & Output Setup ────────────────────────────
    sampler = SearchTaskSampler(
        data_dir=SEARCH_DATA_DIR,
        split="sft",
        # split="test",
        seed=seed,
    )

    

    # 输出控制 (逻辑标记, 从 0 开始, 仅用于文件名标注, 不影响实际采样)
    save_traj = 1              # 是否保存轨迹 (1=保存, 0=不保存)

    # ── 分片配置：手动管理，支持任意分片方案 ──────────────
    # 每项为 (start, end) 闭区间，顺序不重复
    slice_ranges = [
        # (0, 2),
        # (3, 5),
        # (0, 99),
        # (100, 199),
        # (200, 299),
        # (300, 399),
        # (400, 499),
        # (500, 599),
        # (600, 699),
        # (700, 799),
        # (800, 899),
        # (900, 999),
        # (1000, 1099),
        # (1100, 1199),
        # (1200, 1299),
        # (1300, 1399),
        # (1400, 1499),
        # (1500, 1599),
        # (1600, 1699),
        # (1700, 1799),
        (1792, 1799),
        (1800, 1899),
        (1900, 1999)
    ]
    
    single_index=[
        0, 17, 29, 31
    ]
    
    print(f"\n{'='*80}")
    print(f"Generating with sampler: {sampler}")
    print(f"Parallel config: group_n={group_n}, env_num={env_num}, total_envs={group_n * env_num}, num_para={num_para}")
    print(f"Slice ranges: {slice_ranges}")
    print(f"{'='*80}\n")

    # 重置 sampler 状态，然后每组分片前同步计数器
    sampler.reset()
    
    # OUTPUT_BASE_DIR = '/diskpool/home/xuxz/verl-agent/coldstart_test_search/test_seed1'
    OUTPUT_BASE_DIR = '/diskpool/home/xuxz/verl-agent/coldstart_genaration_search/result_search_seed1'
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    # test_single = 0
    # if test_single:
    #     range1 = [(x, x) for x in single_index]
    #     OUTPUT_BASE_DIR = '/diskpool/home/xuxz/verl-agent/coldstart_genaration_search/result_search_single'
    #     os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    #     print(f'test_single, single_index: {single_index}')
    # else:
    #     range1 = slice_ranges
    range1 = slice_ranges
    
    # max_turns = 25
    # max_turns = 20
    max_turns = 15
    for chunk_start, chunk_end in range1:
        output_file = get_unique_filename(
            os.path.join(OUTPUT_BASE_DIR, f'search_coldstart_{chunk_start}_{chunk_end}.json')
        )
        print(f"{'─'*60}")
        print(f"Processing chunk [{chunk_start}, {chunk_end}] → {os.path.basename(output_file)}")
        print(f"{'─'*60}")
        sampler._build_shuffle_table()
        print(f"{'─'*60}")

        evaluate_coldstart_data(
            output_file=output_file,
            sampler=sampler,
            max_turns=max_turns,
            show_turn=show_turn,
            his_len=his_len,
            save_traj=save_traj,
            use_local_model=False,
            ds_model=1,
            effort=1,
            start_idx=chunk_start,
            end_idx=chunk_end,
            group_n=group_n,
            env_num=env_num,
            num_para=num_para,
        )

    print(f"\n{'='*80}")
    print("All chunks completed!")
    print(f"{'='*80}")
