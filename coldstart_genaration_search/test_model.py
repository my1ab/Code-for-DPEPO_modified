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
SEARCH_URL = 'http://127.0.0.1:8000/retrieve'  # 检索服务器地址
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
    # client = OpenAI(api_key="sk-3fa0dedd2f1043fa9a861f864108a15d", base_url="https://api.deepseek.com")
    # client = OpenAI(api_key="sk-ca982270521c4b8184115c7928c96801", base_url="https://api.deepseek.com")
    # client = OpenAI(api_key="sk-a82b8e25c4a5412b8ccf875a2bb15943", base_url="https://api.deepseek.com")
    client = OpenAI(api_key="sk-b718f52386c34ffeb714f684d225f688", base_url="https://api.deepseek.com")

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
def test_model(ds_model=1, effort=0):
    """
    用简单例句测试 DeepSeek API 是否可达。

    Args:
        ds_model: 1=Flash, 2=Pro
        effort: 0=disabled thinking, 1=high, 2=max
    """
    print("=" * 60)
    print("Testing DeepSeek API connectivity...")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Please respond with a short greeting."},
    ]

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
    use_local_model = False    # True=使用本地模型, False=使用 DeepSeek API
    ds_model = 1               # DeepSeek 模型: 1=Flash, 2=Pro (仅 API 模式生效)
    effort = 1                 # DeepSeek thinking: 0=disabled, 1=high, 2=max

    # 环境交互
    max_turns = 25             # 每问最大搜索/回答轮数

    # 历史窗口
    his_len = 8                # 上下文历史窗口长度 (-1=使用全部历史)

    # 并行探索设置
    group_n = 5                # 并行组数
    env_num = 1                # 每组环境数 (total_envs = group_n * env_num)
    num_para = group_n               # 每轮最大并行动作数 (<= total_envs)

    # 采样控制
    seed = -1                  # 随机种子: <0 → sequential, >=0 → 随机采样
    split = "sft"              # 数据分区: test/train/sft/all, 决定 parquet 文件和默认 range
    sampler_range = None       # 实际数据范围 (start, end), 覆盖 split 默认; None=使用 split 默认
    show_turn = True           # 是否逐轮打印状态信息

    

    test_model(ds_model, effort)
    print(f'[DEBUG] test done')
    # exit(0)

    # ── Sampler & Output Setup ────────────────────────────
    sampler = SearchTaskSampler(
        data_dir=SEARCH_DATA_DIR,
        split=split,                # 由 split 变量决定 parquet 文件和默认 range
        split_range=sampler_range,  # 手动覆盖实际数据范围; None=使用 split 默认
        seed=seed,
    )

    

    # 输出控制 (逻辑标记, 从 0 开始, 仅用于文件名标注, 不影响实际采样)
    save_traj = 1              # 是否保存轨迹 (1=保存, 0=不保存)

    # ── 分片配置：手动管理，支持任意分片方案 ──────────────
    # 每项为 (start, end) 闭区间，顺序不重复
    slice_ranges = [
        # (20, 24),
        # (25, 29),
        (0,49)
        # (10, 14),
        # (0, 4),
        # (5, 9),
        # (10, 14),
        # (15, 19),
        # (0, 99),
        # (100, 199),
        # (200, 299),
        # (300, 399),
        # (400, 499),
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
    
    OUTPUT_BASE_DIR = '/diskpool/home/xuxz/verl-agent/coldstart_genaration_search/result_search'
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    test_single = 1
    if test_single:
        range1 = single_index
        OUTPUT_BASE_DIR = '/diskpool/home/xuxz/verl-agent/coldstart_genaration_search/result_search_single'
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    else:
        range1 = slice_ranges
        
    for chunk_start, chunk_end in range1:
        # 将 sampler 内部指针同步到 chunk 的起始位置
        sampler._goal_idx_counter = chunk_start

        output_file = get_unique_filename(
            os.path.join(OUTPUT_BASE_DIR, f'search_coldstart_{chunk_start}_{chunk_end}.json')
        )
        print(f"\n{'─'*60}")
        print(f"Processing chunk [{chunk_start}, {chunk_end}] → {os.path.basename(output_file)}")
        print(f"{'─'*60}")

        evaluate_coldstart_data(
            output_file=output_file,
            sampler=sampler,
            max_turns=max_turns,
            show_turn=show_turn,
            his_len=his_len,
            save_traj=save_traj,
            use_local_model=use_local_model,
            ds_model=ds_model,
            effort=effort,
            start_idx=chunk_start,
            end_idx=chunk_end,
            group_n=group_n,
            env_num=env_num,
            num_para=num_para,
        )

    print(f"\n{'='*80}")
    print("All chunks completed!")
    print(f"{'='*80}")
