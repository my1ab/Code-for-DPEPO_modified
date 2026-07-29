# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Search-specific parallel rollout loop.

Adapted from rollout_loop_parallel_webshop.py for Search-R1 task.
Key differences from WebShop:
  1. Uses search-specific prompts (prompts_search.py) with <search>/<answer> actions
  2. Search actions: <search>query</search> and <answer>answer</answer>
  3. Observations are search result text (no 'available_actions' dict)
  4. Task (question) is passed directly, not extracted from [SEP] format

Modification Summary (对齐 coldstart_search_local.py 的 Search 字段交互方式):
  - Import: .prompts_webshop → .prompts_search (SYSTEM_PROMPT_SEARCH_PARA, USER_PROMPT_NO_HIS_PARA, USER_PROMPT_HIS_PARA)
  - 类名: TrajectoryCollectorParallelWebShop → TrajectoryCollectorParallelSearch
  - step==0: reason_prompt_para + "Next Possible Actions" → USER_PROMPT_NO_HIS_PARA + 纯 <observation_i> 包装
  - step>0:  reason_prompt_para_his + "Next Possible Actions" → USER_PROMPT_HIS_PARA, 去掉所有 "Next Possible Actions"
  - System prompt: system_message_para → SYSTEM_PROMPT_SEARCH_PARA
  - task 字段: 占位符 'Find and purchase a product' → 直接用 start_obs (即 question)
  - Observation 格式: 含 "admissible_commands" 描述 → 纯 <observation_i>\n{obs}\n</observation_i>
  - action extraction 保留: extract_think_and_actions 提取 <env_i> 标签, search_projection 解析内层 <search>/<answer>
"""

import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict, Optional
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.torch_functional import pad_sequence_to_length
from .prompts_search import SYSTEM_PROMPT_SEARCH_PARA, USER_PROMPT_NO_HIS_PARA, USER_PROMPT_HIS_PARA
import re
from tqdm import tqdm

import json
import os
import time



def append_to_json_file(data, filename):
    """Append a dict to a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = []

    if isinstance(existing_data, list):
        existing_data.append(data)
    else:
        existing_data = [existing_data, data]

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)


def extract_think_and_actions(text, num_parallel, print_debug=True):
    """
    Extract think content and action dict from model output.
    照搬coldstart_para_his_test文件的健壮逻辑：提前初始化所有num_parallel个环境的action为"null"，只填充合法的动作
    Returns:
        dict with keys:
            - 'think': str or None
            - 'actions': dict mapping env_index -> action string
    """
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, text, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else None

    # 提前初始化所有num_parallel个环境的动作都是null，env_manager的索引是1-based，所以key是1~num_parallel
    actions_dict = {idx: "null" for idx in range(1, num_parallel + 1)}
    
    actions_pattern = r'<env_(\d+)>(.*?)</env_\d+>'
    matches = re.findall(actions_pattern, text, re.DOTALL)
    
    for env_index, action in matches:
        env_index = int(env_index)
        # env_manager里的self.env_pools是1-based索引，valid indices是1~len(self.env_pools)，和模型输出的<env_1>标签完全对齐
        if 1 <= env_index <= num_parallel:
            action = action.strip()
            if action and action != 'None' and action != 'null':
                actions_dict[env_index] = action
    
    # 打印提取的动作数量，调试null_count统计问题
    if print_debug:
        print(f"[DEBUG] [extract_think_and_actions] 提取到动作数量(排除null): {len([v for v in actions_dict.values() if v != 'null'])}, 原始匹配动作列表长度: {len(matches)}, think内容是否存在: {think_content is not None}，num_parallel={num_parallel}")
    
    return {
        'think': think_content,
        'actions': actions_dict
    }


def non_tensor_to_list_of_dict(batch: DataProto) -> list[dict]:
    """Convert DataProto non-tensor batch to list of dicts."""
    tensors = batch.batch
    non_tensor = batch.non_tensor_batch
    batch_size = len(tensors['input_ids'])
    total_data_list = []
    for bs in range(batch_size):
        save_dict = {}
        for key, val in non_tensor.items():
            save_dict[key] = val[bs]
        total_data_list.append(save_dict)
    return total_data_list


# 模块级全局任务计数器，生命周期与程序一致，不受类创建销毁影响
GLOBAL_TASK_COUNTER = 0
import datetime


class TrajectoryCollectorParallelSearch:
    """
    Trajectory collector for Search parallel environment training.
    
    Handles prompt construction with search-specific templates,
    history management, and multi-step environment interaction.
    """

    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        # 从 config.custom 读取调试和保存开关，支持 shell 脚本传入覆盖
        if hasattr(config, 'custom'):
            self.print_debug = config.custom.get('print_debug', True)
            self.save_traj = config.custom.get('save_traj', False)
        else:
            self.print_debug = True
            self.save_traj = False

    def format_available_actions(self, available_actions):
        """
        Format search available actions into a readable string.
        Search env does not have structured available_actions like WebShop;
        this is kept for interface compatibility.
        """
        return ['<search>query</search>', '<answer>answer</answer>']

    def preprocess_single_sample(
        self,
        item: int,
        step: int,
        task: str,
        start_obs: str,
        start_possible_action: str,
        history_actions: List,
        history_observations: List,
        last_action: Dict,
        last_observation: Dict,
        last_possible_actions: Dict,
        start_available_actions: Optional[List] = None,
        history_available_actions: Optional[List] = None,
        last_available_actions: Optional[Dict] = None,
        num_parallel=5,
        add_limit_prompt=True,
        total_envs=5,
    ):
        """
        Process a single Search sample into model input format.
        
        Uses search-specific prompt templates from prompts_search.py (coldstart_search_local.py style).
        - step==0: uses USER_PROMPT_NO_HIS_PARA (initial prompt with observation, no "Next Possible Actions")
        - step>0:  uses USER_PROMPT_HIS_PARA (with history context, no "Next Possible Actions")
        """
        if step == 0:
            # Initial step: build current observation prompt (search-style, no "Next Possible Actions")
            obs_prompt = ''
            for idx in range(1, total_envs + 1):
                obs_prompt += (
                    f'<observation_{idx}>\n{start_obs}\n</observation_{idx}>\n'
                )

            prompt = USER_PROMPT_NO_HIS_PARA.format(
                question=task,
                observations=obs_prompt,
                total_envs=self.config.env.num_parallel,
                num_parallel=self.config.env.num_parallel,
            )
        else:
            # 构建带有 <observation_i> 标签包装的 initial_observation（对齐coldstart_search_local.py风格）
            # Search环境没有"Next Possible Actions"，observations直接是搜索结果文本
            initial_obs_prompt = ''
            for idx in range(1, total_envs + 1):
                initial_obs_prompt += (
                    f'<observation_{idx}>\n{start_obs}\n</observation_{idx}>\n'
                )

            # Build current observations prompt for all environments (search-style, no "Next Possible Actions")
            obs_prompt = ''
            for idx in range(1, total_envs + 1):
                if isinstance(last_observation, dict):
                    obv = last_observation.get(idx, start_obs)
                elif (idx - 1) < len(last_observation):
                    obv = last_observation[idx - 1]
                else:
                    obv = start_obs
                obs_prompt += (
                    f'<observation_{idx}>\n{obv}\n</observation_{idx}>\n'
                )

            # Build history info for all environments
            history_start = "You have already taken multiple actions in multiple parallel environments. Below are the most recent observations and the corresponding actions you took:\n"
            his_len = self.config.env.get('history_length', -1)

            if his_len < 0:
                # Use full history
                history_lines = []
                for env_idx in range(1, total_envs + 1):  # 统一使用1-based索引访问字典
                    env_history = []
                    if history_actions and env_idx in history_actions:  # 检查字典中是否存在该键
                        env_act_list = history_actions[env_idx] if isinstance(history_actions[env_idx], list) else list(history_actions[env_idx].values())
                        env_obs_list = history_observations[env_idx] if isinstance(history_observations[env_idx], list) else list(history_observations[env_idx].values())
                        for t_idx in range(min(len(env_act_list), len(env_obs_list))):
                            env_history.append(f"Action {t_idx + 1}: {env_act_list[t_idx]}")
                            env_history.append(f"Observation {t_idx + 1}: {env_obs_list[t_idx]}")
                    if env_history:
                        history_lines.append(f"In Environment {env_idx}\n" + "\n".join(env_history))  # env_idx已经是1-based，直接使用
                history_info = history_start + "\n\n".join(history_lines)
            else:
                # Use partial history (last his_len steps per env)
                history_partial_lines = []
                for env_idx in range(1, total_envs + 1):
                    env_history = []
                    if history_actions and env_idx in history_actions:  # 检查字典中是否存在该键
                        env_act_list = history_actions[env_idx] if isinstance(history_actions[env_idx], list) else list(history_actions[env_idx].values())
                        env_obs_list = history_observations[env_idx] if isinstance(history_observations[env_idx], list) else list(history_observations[env_idx].values())
                        start_idx = max(0, len(env_act_list) - his_len)
                        for t_idx in range(start_idx, len(env_act_list)):
                            env_history.append(f"Action {t_idx + 1}: {env_act_list[t_idx]}")
                            if t_idx < len(env_obs_list):
                                env_history.append(f"Observation {t_idx + 1}: {env_obs_list[t_idx]}")
                    if env_history:
                        history_partial_lines.append(f"In Environment {env_idx}\n" + "\n".join(env_history))  # env_idx已经是1-based，直接使用
                history_info = history_start + "\n\n".join(history_partial_lines)

            # Build last step info (search-style: no "Next Possible Actions")
            last_history_lines = []
            for env_idx in range(total_envs):
                action = last_action.get(env_idx + 1, "null") if isinstance(last_action, dict) else (
                    last_action[env_idx] if env_idx < len(last_action) else "null"
                )
                obv = last_observation[env_idx] if isinstance(last_observation, list) and env_idx < len(last_observation) else (
                    last_observation.get(env_idx, start_obs) if isinstance(last_observation, dict) else start_obs
                )
                env_history_lines = [f"Action {step}: {action}"]
                env_history_lines.append(f"Observation {step}: {obv}")
                last_history_lines.append(f"In Environment {env_idx + 1}\n" + "\n".join(env_history_lines))
            last_history = "\n\n".join(last_history_lines)

            prompt = USER_PROMPT_HIS_PARA.format(
                task_description=task,
                initial_observation=initial_obs_prompt,
                history_info=history_info,
                last_history=last_history,
                num_parallel=self.config.env.num_parallel,
                total_envs=self.config.env.num_parallel,
            )

        # 使用search专用的system prompt，对齐coldstart_search_local.py
        generation_completion = [
            {'role': 'system', 'content': SYSTEM_PROMPT_SEARCH_PARA.format(num_parallel=self.config.env.num_parallel, total_envs=self.config.env.num_parallel)},
            {'role': 'user', 'content': prompt}
        ]

        chat = np.array(generation_completion)

        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False
        )

        # Initialize return dict
        row_dict = {}

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.config.data.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        # 
        raw_prompt_ids = self.tokenizer.encode(prompt_with_chat_template, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length:]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.config.data.max_prompt_length]
                
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'index': item,
            'data_source': 'agent'
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()

        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto,
        step: int,
        start_obs: List,
        start_possible_actions: List,
        history_actions: List,
        history_observations: List,
        last_actions: List,
        last_observations: List,
        last_possible_actions: List,
        num_parallel: int,
        add_limit_prompt: bool,
        total_envs: int,
        active_masks: np.ndarray = None,
    ) -> DataProto:
        """
        Process a batch of Search observation samples.
        对齐 coldstart_search_local.py 的交互方式：
        - start_obs 本身即为question（search env reset返回的obs_list即为question）
        - task 直接使用 start_obs（question），无需占位符
        - 无 "Next Possible Actions" 信息
        """
        data_all_infos = {
            'start_obs': start_obs,
            'start_possible_actions': start_possible_actions,
            'history_actions': history_actions,
            'history_observations': history_observations,
            'last_actions': last_actions,
            'last_observations': last_observations,
            'last_possible_actions': last_possible_actions,
        }

        # Convert to list of dict
        length = len(start_obs)
        save_list = []
        for batch_idx in range(length):
            save_dict = {}
            for key, value in data_all_infos.items():
                save_dict[key] = value[batch_idx]
            save_list.append(save_dict)

        # Search：start_obs即为question，直接用作task（对齐coldstart_search_local.py）
        for entry in save_list:
            entry['task'] = entry['start_obs'] if isinstance(entry['start_obs'], str) else str(entry['start_obs'])

        processed_samples = []

        for item, entry in enumerate(save_list):
            # 即使任务已完成，也要继续处理，保持batch大小一致，避免维度不匹配问题
            # 所有任务都必须处理，不能跳过，确保返回的batch大小始终与输入一致
            processed = self.preprocess_single_sample(
                item=item,
                step=step,
                task=entry['task'],
                start_obs=entry['start_obs'],
                start_possible_action=entry['start_possible_actions'],
                history_actions=entry['history_actions'],
                history_observations=entry['history_observations'],
                last_action=entry['last_actions'],
                last_observation=entry['last_observations'],
                last_possible_actions=entry['last_possible_actions'],
                # num_parallel=num_parallel,
                num_parallel=self.config.env.num_parallel,
                # num_parallel=total_envs,  
                add_limit_prompt=add_limit_prompt,
                total_envs=total_envs,
            )
            processed_samples.append(processed)

        batch = collate_fn(processed_samples)

        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch

    def gather_rollout_data(
        self,
        total_batch_list: List[List[Dict]],
        episode_rewards: np.ndarray,
        episode_lengths: np.ndarray,
        traj_uid: np.ndarray,
        tool_callings: np.ndarray,
        success_flags: np.ndarray = None,
        status_msgs: List = None,
        world_size=None
    ) -> DataProto:
        """Collect and organize trajectory data, aligned with official rollout_loop_parallel.py."""
        batch_size = len(total_batch_list)
        if self.print_debug:
            print(f'[DEBUG] into gather_rollout_data, batch_size={batch_size}')
        
        # 设置默认值，保持向后兼容性
        # if success_flags is None:
        #     success_flags = np.zeros(batch_size, dtype=int)
        # if status_msgs is None:
        #     status_msgs = ["" for _ in range(batch_size)]

        # 完全和官方rollout_loop_parallel.py保持一致的逻辑
        effective_batch = [] 
        for bs in range(batch_size):
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # 完全对齐官方实现，直接在原数据上添加字段
                    data['episode_rewards'] = episode_rewards[bs] 
                    data['episode_lengths'] = episode_lengths[bs]
                    data['tool_callings'] = tool_callings[bs]
                    # webshop特有字段保留
                    # data['success_flag'] = int(success_flags[bs])
                    # data['status_msg'] = str(status_msgs[bs])
                    # if 'active_masks' not in data:
                    #     data['active_masks'] = True
                    effective_batch.append(data)
        
        # 完全和官方一样的padding流程，保留必要的多卡对齐padding
        gen_batch = DataProto.from_single_dict(data=collate_fn(effective_batch))
        # 和官方代码一致，保留pad_dataproto_to_divisor用于多卡训练的batch对齐
        # if hasattr(self.config, 'world_size'):
        # if world_size != None:
        #     # padded_gen_batch, pad_info = pad_dataproto_to_divisor(gen_batch, self.config.world_size)
        #     print(f'[DEBUG] world_size={world_size} in gather')
        #     padded_gen_batch, pad_info = pad_dataproto_to_divisor(gen_batch, world_size)
        #     # if pad_info > 0:
        #     #     padded_gen_batch.meta_info['padded_info'] = pad_info
        #     final_batch = padded_gen_batch
        # else:
        #     final_batch = gen_batch
        
        # import gc
        # del total_batch_list, episode_rewards, episode_lengths, traj_uid, tool_callings
        # del success_flags, status_msgs, effective_batch, gen_batch
        # gc.collect()
        # if torch.cuda.is_available():
            # torch.cuda.empty_cache()
    
        return gen_batch

    # ===================== Calculate 系列机制（对齐 rollout_loop_parallel.py） =====================
    def calculate_process_reward(self, total_batch_list):
        # TODO: Check the boundary situation

        # merge multi step data into a full trajectory
        group_data = []
        for trajectory in total_batch_list:
            save_dict = {}
            save_dict['gamefile'] = trajectory[0]['gamefile']
            if 'expert_actions' in trajectory[0]:
                save_dict['expert_actions'] = trajectory[0]['expert_actions']
            else:
                # WebShop may not have expert_actions; fall back to empty list
                print('[warning]no expert actions for process reward')
                save_dict['expert_actions'] = []

            save_dict['parallel_action'] = {}
            save_dict['parallel_obs'] = {}
            for i in range(self.config.env.num_parallel):
                save_dict['parallel_action'][i + 1] = []
                save_dict['parallel_obs'][i + 1] = []

            for step_data in trajectory:
                action_dict = step_data['action_dict']
                observation_list = step_data['observation']

                idx = 0
                for env_idx, action in action_dict.items():
                    if env_idx in set(range(1, self.config.env.num_parallel + 1)):
                        cur_action = action
                        cur_obs = observation_list[idx]
                        idx += 1
                        save_dict['parallel_action'][env_idx].append(cur_action.strip(' ').strip('\n').strip('\n\n'))
                        save_dict['parallel_obs'][env_idx].append(cur_obs)

            group_data.append(save_dict)

        # Calculate the rewards
        total_process_rewards = []

        for predict in group_data:
            parallel_actions = predict['parallel_action']
            expert_action_list = predict['expert_actions']

            parallel_lcs_rewards = []
            # Iterate each parallel sub-trajectory
            for env_idx, action_list in parallel_actions.items():
                reward = self.count_longest_ordered_subsequence(expert_action_list, action_list)
                parallel_lcs_rewards.append(reward)

            action_reward = max(parallel_lcs_rewards)

            total_process_rewards.append(round(action_reward / len(expert_action_list), 5) if expert_action_list else 0.0)

        return total_process_rewards

    def calculate_penalties(self, history_actions, action_dict):
        # history_actions: Containing history actions in each env
        # sample: only the action_dict is useful
        action_keys = set(action_dict.keys()) & set(range(1, self.config.env.num_parallel + 1))
        for his_key, value in history_actions.items():
            history_actions[his_key] = [his_act.strip() for his_act in value]
        for act_key, value in action_dict.items():
            action_dict[act_key] = value.strip()

        action_penalty_per_env = []
        for key in action_keys:
            env_action = action_dict[key].strip()
            env_history_actions = history_actions[key]

            # Calculate the Simple Repeat Count
            COUNT_repeat_penalty = self.calculate_depth_repeat(
                history_actions=env_history_actions,
                action=env_action
            )

            if len(env_history_actions) == 0:
                last_state_action = (env_action)
            else:
                last_state_action = (env_history_actions[-1], env_action)

            # Depth Transition Repeat Count
            COUNT_depth_transition_penalty = self.calculate_transition_repeat(
                history_actions=env_history_actions,
                last_state_action=last_state_action
            )

            # Width Transition Repeat Count
            LIST_width_repeat_count = []  # TODO: Add Pooling Operation
            for w_idx in list(set(action_keys) - set([key])):
                width_history_actions = history_actions[w_idx] + [action_dict[w_idx]]
                repeat_count = self.calculate_transition_repeat(
                    history_actions=width_history_actions,
                    last_state_action=last_state_action
                )
                LIST_width_repeat_count.append(repeat_count)

            COUNT_width_transition_repeat = sum(LIST_width_repeat_count)

            depth_alpha = self.config.reward_model.depth_alpha
            W_depth_repeat = depth_alpha ** COUNT_repeat_penalty

            depth_t_gamma = self.config.reward_model.depth_t_gamma
            W_depth_t_repeat = depth_t_gamma ** COUNT_depth_transition_penalty

            width_t_beta = self.config.reward_model.width_t_beta
            W_width_t_repeat = width_t_beta ** COUNT_width_transition_repeat

            W_list = [W_depth_repeat, W_depth_t_repeat, W_width_t_repeat]
            pooling_kind_weight = sum(W_list) / len(W_list)

            action_penalty_per_env.append(pooling_kind_weight)

        actions_wo_look = [elem for elem in action_dict.values() if elem != 'look']
        COUNT_width_repeat = len(actions_wo_look) - len(set(actions_wo_look))
        width_omega = self.config.reward_model.width_omega
        W_width_repeat = width_omega ** COUNT_width_repeat

        pooling_w_action = sum(action_penalty_per_env) / len(action_penalty_per_env) if action_penalty_per_env else 0.0

        W = (W_width_repeat + pooling_w_action) / 2

        return round(W, 4)

    def get_state_action_pair(self, action_history):
        return [(a, b) for a, b in zip(action_history, action_history[1:])]

    def calculate_transition_repeat(self, history_actions, last_state_action):
        # Transition Repeat
        full_action_list = history_actions

        state_action_pair_a = self.get_state_action_pair(full_action_list)

        if len(state_action_pair_a) == 0:
            return 0

        repeat_count = state_action_pair_a.count(last_state_action)

        return repeat_count

    def calculate_depth_repeat(self, history_actions, action):
        repeat_count = 0
        for history_action in reversed(history_actions):
            if action == history_action:
                repeat_count += 1

        return repeat_count

    def calculate_width_repeat_rate(self, action_dict):
        num_actions = len(action_dict)
        num_dedu_actions = len(set(action_dict.values()))
        return num_actions - num_dedu_actions

    def count_longest_ordered_subsequence(self, ground_truth, prediction):
        if not ground_truth or not prediction:
            return 0

        i = 0  # 指向 ground_truth
        j = 0  # 指向 agent

        matched = 0
        while i < len(ground_truth) and j < len(prediction):
            if ground_truth[i] == prediction[j]:
                matched += 1
                i += 1   # 只有匹配成功才前进 ground_truth
            j += 1       # agent 永远前进

        return matched

    # ===================== End of Calculate 系列机制 =====================

    def vanilla_multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
        is_train: bool = True,
    ) -> DataProto:
        """
        Collect trajectories through parallel Search agent-environment loop.
        
        Parameters match rollout_loop_parallel.py interface but use search-specific
        prompting and history construction (对齐coldstart_search_local.py交互方式).
        """
        # 声明使用模块级全局变量
        global GLOBAL_TASK_COUNTER
        # 训练阶段先对gen_batch进行repeat，确保batch_size正确扩展为train_batch_size*group_n
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
        batch_size = len(gen_batch.batch)

        # ---------- ID Preparation ----------
        group_ids = []
        uid_batch = []
        # 训练阶段使用config.env.rollout.n创建多个重复样本，验证阶段group_n=1节省资源
        group_n = self.config.env.rollout.n if is_train else 1
        # 验证时 [DEBUG] group_n = 1 in rollout loop, is_train=False, batch_size=50
        if self.print_debug:
            print(f'[DEBUG] group_n = {group_n} in rollout loop, is_train={is_train}, batch_size={batch_size}')
        for i in range(batch_size):
            if i % group_n == 0:
                group_id = 0
                uid = str(uuid.uuid4())
            else:
                group_id += 1
            group_ids.append(group_id)
            uid_batch.append(uid)
        uid_batch = np.array(uid_batch, dtype=object)
        group_ids = np.array(group_ids, dtype=object)

        gen_batch.non_tensor_batch['uid'] = uid_batch
        gen_batch.non_tensor_batch['group_id'] = group_ids

        # num_parallel = self.config.env.num_parallel
        add_limit_prompt = self.config.env.get('add_limit_prompt', True)
        # total_envs = group_n

        # Initial observations from the environment
        non_tensor_batch = non_tensor_to_list_of_dict(gen_batch)
        start_obs, start_possible_actions = envs.get_start_info_group(non_tensor_batch)

        length_obs = len(start_obs)
        assert len(gen_batch.batch) == length_obs, \
            f"Batch Size:{len(gen_batch.batch)} does not match Observations Size: {length_obs}"

        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        
        # 参考coldstart_para_his_test_1.5B_hislen8_epoch3.5_v2.py添加状态跟踪变量（仅保留成功/失败判断的核心功能）
        null_count = np.zeros(batch_size, dtype=int)  # 每个样本连续null动作的次数
        turn_out_range = np.ones(batch_size, dtype=bool)  # 标记任务是否仍在有效轮次内
        success_flags = np.zeros(batch_size, dtype=int)  # 每个样本的成功标记：0=未完成，1=成功
        status_msgs = ["" for _ in range(batch_size)]  # 每个样本的状态消息
        # 确保长度严格匹配batch_size，避免后续维度错误
        assert len(success_flags) == batch_size, "success_flags长度与batch_size不匹配"
        assert len(status_msgs) == batch_size, "status_msgs长度与batch_size不匹配"

        # Trajectory collection loop（恢复copy.py原始循环逻辑，仅保留必要的成功判断功能）
        # _step为当前步数
        for _step in tqdm(range(self.config.env.max_steps)):
            if self.print_debug:
                print(f'[DEBUG] running task {GLOBAL_TASK_COUNTER} step {_step} of total {self.config.env.max_steps}')
            # 逐个元素取反  即is_done真时将active_masks伪
            active_masks = np.logical_not(is_done)
            # 如果所有任务都完成了  即全伪  则提前退出  但之后已经有is_done判断
            # if not active_masks.any():  
                # break

            non_tensor_batch = non_tensor_to_list_of_dict(gen_batch)
            history_actions, history_observations = envs.get_history_info_group(non_tensor_batch)
            last_actions, last_observations, last_possible_actions = envs.get_last_actions_info_group(non_tensor_batch)

            # 1. Preprocess batch
            batch = self.preprocess_batch(
                gen_batch=gen_batch,
                step=_step,
                start_obs=start_obs,
                start_possible_actions=start_possible_actions,
                history_actions=history_actions,
                history_observations=history_observations,
                last_actions=last_actions,
                last_observations=last_observations,
                last_possible_actions=last_possible_actions,
                # num_parallel=5,
                num_parallel=self.config.env.num_parallel,
                add_limit_prompt=add_limit_prompt,
                # total_envs=5,
                total_envs=self.config.env.num_parallel,
                active_masks=active_masks,
            )

            # 2. Generate model output
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")

            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            batch_input.meta_info = gen_batch.meta_info

            # 只对 active（未完成）样本做模型推理，已完成样本跳过以减少计算量
            active_indices = np.where(active_masks)[0]
            num_active = len(active_indices)
            if num_active == batch_size:
                # 全部 active，走原始路径
                batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
                batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
                batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)
            else:
                # 只对 active 样本推理，减少 GPU 计算量
                # batch_input_active = batch_input[active_masks]
                batch_input_active = batch_input[active_indices]
                batch_input_active_padded, pad_size_active = pad_dataproto_to_divisor(
                    batch_input_active, actor_rollout_wg.world_size
                )
                batch_output_active_padded = actor_rollout_wg.generate_sequences(
                    batch_input_active_padded
                )
                batch_output_active = unpad_dataproto(
                    batch_output_active_padded, pad_size=pad_size_active
                )

                # 将 active 推理结果映射回全 batch
                from tensordict import TensorDict
                full_batch = {}
                for key, tensor in batch_output_active.batch.items():
                    full_shape = list(tensor.shape)
                    full_shape[0] = batch_size
                    # 先用0占位
                    full_tensor = torch.full(
                        full_shape, fill_value=0,
                        dtype=tensor.dtype, device=tensor.device
                    )
                    # 两步按key写回 active_indices为activemask判定仍未结束的样本
                    full_tensor[active_indices] = tensor
                    full_batch[key] = full_tensor

                full_non_tensor = {}
                for key, val in batch_output_active.non_tensor_batch.items():
                    full_val = np.empty(batch_size, dtype=object)
                    # 逐元素赋值，避免 num_active 维数 >1 时 numpy 广播失败
                    for j, idx in enumerate(active_indices):
                        full_val[idx] = val[j]
                    full_non_tensor[key] = full_val

                batch_output = DataProto(
                    batch=TensorDict(full_batch, batch_size=(batch_size,)),
                    non_tensor_batch=full_non_tensor,
                    meta_info=batch_input.meta_info
                )

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid
            batch.non_tensor_batch['group_id'] = group_ids

            batch = batch.union(batch_output)
            batch.non_tensor_batch['gamefile'] = gen_batch.non_tensor_batch['gamefile']
            if 'expert_actions' in gen_batch.non_tensor_batch:
                batch.non_tensor_batch['expert_actions'] = gen_batch.non_tensor_batch['expert_actions']

            text_actions = self.tokenizer.batch_decode(
                batch.batch['responses'],
                skip_special_tokens=True
            )

            batch.non_tensor_batch['action'] = text_actions
            # 提取动作
            batch.non_tensor_batch['action_dict'] = [
                extract_think_and_actions(elem, num_parallel=self.config.env.num_parallel, print_debug=self.print_debug)['actions'] for elem in text_actions
            ]

            # 格式化输入
            parallel_actions_dict = to_list_of_dict(batch)

            # 补null动作  避免无效交互
            for i in range(batch_size):
               
                if not active_masks[i]:
                    action_dict = parallel_actions_dict[i].get('action_dict', {})
                    for env_idx in action_dict:
                        action_dict[env_idx] = "null"

            # 3. Interact with environments
            # 选用def step_group(self, gourped_samples):
            # 进行动作
            dict_grouped_output = envs.step_group(parallel_actions_dict)

            # 计算平行惩罚
            # ---------Parallel Penalties (对齐 rollout_loop_parallel.py) ----------- #
            if hasattr(self.config, 'reward_model') and self.config.reward_model.get('parallel_reward', False):
                # 重新获取 history_actions（当前步的环境已更新）
                cur_history_actions, _ = envs.get_history_info_group(non_tensor_batch)
                for sample, his_acts in zip(dict_grouped_output, cur_history_actions):
                    action_dict = sample['action_dict']
                    if action_dict and len(action_dict) != 0:
                        W = self.calculate_penalties(
                            history_actions=his_acts,
                            action_dict=action_dict,
                        )
                    else:
                        W = self.config.reward_model.get('no_action_penalty', 1.0)
                    sample['penalty_W'] = W

            single_dict_grouped_output = collate_fn(dict_grouped_output)

            next_obs = single_dict_grouped_output['observation']

            np_rewards = np.array(single_dict_grouped_output['rewards'], dtype=object)
            rewards = np.array([
                np.max(lst) if len(lst) > 0 else 0
                for lst in np_rewards
            ])
            # single_dict_grouped_output = collate_fn(dict_grouped_output)
            dones = single_dict_grouped_output['dones']
            infos = single_dict_grouped_output['possible_actions']

            # 对 possible_actions 做 2D padding：将所有环境内的可选动作列表补全到相同长度（全局最大长度），
            # 缺项填充 "null"。这样 possible_actions 保持为 (batch_size, max_len) 的 2D numpy 数组，
            # 避免不同 step 之间长度变化导致 collate_fn 的 broadcast 错误。(5,12)  (5,)
            # possible_actions_raw = single_dict_grouped_output['possible_actions']
            # if len(possible_actions_raw) > 0:
            #     max_len = max(len(lst) for lst in possible_actions_raw)
            #     # max_len = 12
            #     padded_list = []
            #     for lst in possible_actions_raw:
            #         current = list(lst)
            #         if len(current) < max_len:
            #             current += ["null"] * (max_len - len(current))
            #         padded_list.append(current)
            #     single_dict_grouped_output['possible_actions'] = np.array(padded_list, dtype=object)
            

            batch = DataProto.from_single_dict(
                data=single_dict_grouped_output,
                meta_info=gen_batch.meta_info
            )

            # 记录稀疏奖励
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)

            # 移除辅助字段（来自 step_group），这些字段在后续步形状可能变化，会导致 collate_fn 广播失败
            # 且它们对 PPO 训练无意义，只保留训练所需的字段
            # 安全可删除的字段分析：
            #   - 'observation': 被 calculate_process_reward 使用 → 不能删除
            #   - 'dones': 每步长度固定，但类型可能变化(list of bools vs array)；已提前提取为dones变量用于判断，不再需要 → 安全删除
            #   - 'possible_actions': 每步长度不同(环境状态变化导致可选动作数变化)，collate时形状不一致 → 安全删除
            #   - 'concated_observation': 仅在coldstart中使用，本文件未使用 → 安全删除
            #   - 'task': 固定值，但ppo训练不需要 → 安全删除
            #   - 'success_flags','status_msgs': 仅在coldstart中使用，本文件未使用 → 安全删除
            aux_fields = ['dones', 'possible_actions', 'concated_observation', 'task', 'success_flags', 'status_msgs']
            for field in aux_fields:
                if field in batch.non_tensor_batch:
                    del batch.non_tensor_batch[field]

            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                if not active_masks[i]:
                    # 已完成的样本不再记录轨迹数据，避免噪声数据影响PPO训练
                    continue
                # 未完成则添加batch_list
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # 参考coldstart_para_his_test_1.5B_hislen8_epoch3.5_v2.py添加每样本状态判断
            if self.print_debug:
                print(f"[DEBUG] dones = {dones} and not done yet")
                print(f"[DEBUG] is_done = {is_done}")
            # print(f"rewards = {current_rewards}")
            # 内层：逐个处理每个样本的状态 判定+更新
            for bs in range(batch_size):
                # active_masks伪 已经完成的任务跳过判定处理
                if not active_masks[bs]:
                    continue  
                    
                # 提取并检查当前样本的动作是否全部为null
                current_actions = batch.non_tensor_batch['action_dict'][bs]
                if not current_actions or len(current_actions) == 0:
                    all_invalid = True
                else:
                    # step_actions = [current_actions.get(idx, 'null') for idx in range(total_envs)]
                    # 标记 修复：环境的worker编号从1开始，所以idx从1开始取，避免永远取不到action返回null
                    total_envs = self.config.env.num_parallel
                    step_actions = [current_actions.get(idx+1, 'null') for idx in range(total_envs)]
                    all_invalid = all(a is None or a == 'null' or a == 'None' for a in step_actions)
                
                # 更新null计数
                if all_invalid:
                    null_count[bs] += 1
                else:
                    null_count[bs] = 0
                
                # 检查任务是否完成
                # any_done = dones[bs] if isinstance(dones[bs], bool) else any(dones[bs])
                any_done = any(dones[bs])
                
                if any_done:
                    # 任务已结束，判断是否成功
                    current_rewards = np_rewards[bs]
                    # 修复numpy数组布尔判断错误，检查数组是否非空且包含有效奖励
                    if current_rewards is not None and len(current_rewards) > 0:
                        for reward in current_rewards:
                            if reward is not None and reward > 0:
                                success_flags[bs] = 1
                                break
                    status = "SUCCESS" if success_flags[bs] == 1 else "FAILED"
                    
                    # 最终日志格式与coldstart完全一致
                    worker_id_in_task = bs % group_n  # 当前任务组内的worker索引（0~group_n-1），对应你的5个环境副本0~4
                    completed_idx = worker_id_in_task  # 用worker在组内的索引作为环境编号，符合你对环境副本的理解
                    # status_msgs[bs] = f"Task {task_idx} {status} at turn {_step + 1} in environments {completed_idx}"
                    status_msgs[bs] = f"Task {GLOBAL_TASK_COUNTER} sample {bs} {status} at turn {_step + 1}"
                    if self.print_debug:
                        print(f'[DEBUG] status_msgs[bs] = {status_msgs[bs]}')
                        print(f"dones = {dones}")
                        print(f"np_rewards = {np_rewards}")
                    is_done[bs] = True
                    turn_out_range[bs] = False
                    # 只有训练阶段(group_n>1)才需要把整个组的所有worker都标记为完成，保持所有轨迹长度一致
                    # 验证阶段(group_n=1)只标记当前bs的任务，避免影响其他任务的轨迹收集
                    if is_train:
                        n = self.config.env.rollout.n
                        group_idx = bs // n  # 0-based group index
                        # 整个组全部置为1
                        range1 = range(group_idx * n, (group_idx + 1) * n)
                        for g_bs in range1:
                            if not is_done[g_bs]:
                                is_done[g_bs] = True
                                turn_out_range[g_bs] = False
                    # else:
                        # 验证阶段只标记当前任务，让其他任务自然完成，收集完整的验证轨迹
                        is_done[bs] = True
                        turn_out_range[bs] = False
                    if self.print_debug:
                        print(f'[DEBUG] is_done trans to {is_done}')
                elif null_count[bs] >= 2:
                    status_msgs[bs] = f"Task {GLOBAL_TASK_COUNTER} exit(all null) at turn {_step + 1}"
                    if self.print_debug:
                        print(f'[DEBUG] status_msgs[bs] = {status_msgs[bs]}')
                    is_done[bs] = True
                    turn_out_range[bs] = False
                    # 要求单条轨迹结束时
                    if is_train:
                        n = self.config.env.rollout.n
                        group_idx = bs // n  # 0-based group index
                        range1 = range(group_idx * n, (group_idx + 1) * n)
                        for g_bs in range1:
                            if not is_done[g_bs]:
                                is_done[g_bs] = True
                                turn_out_range[g_bs] = False
                    else:
                        # 验证阶段只标记当前任务，让其他任务自然完成，收集完整的验证轨迹
                        is_done[bs] = True
                        turn_out_range[bs] = False
                    if self.print_debug:
                        print(f'[DEBUG] is_done trans to {is_done}')
            
            # 检查是否所有任务都已完成，无论batch_size是多少，只要全部完成就立即退出
            if is_done.all():
                print(f"All tasks completed at turn {_step + 1}, exiting rollout loop.")
                break
        # ------------------ Calculation Process Reward (对齐 rollout_loop_parallel.py) ---------------------
        # reward_model和process_reward true时计算
        if hasattr(self.config, 'reward_model') and self.config.reward_model.get('process_reward', False):
            process_reward = self.calculate_process_reward(
                total_batch_list=total_batch_list,
            )
            # We only apply `process reward` when the trajectory fails
            process_reward = np.array(process_reward)
            # 仅当环境奖励为 0 时才用 process_reward 替代
            episode_rewards = np.where(episode_rewards == 0, process_reward, episode_rewards)

        # batch循环结束后处理仍在进行中的任务（超过配置的最大步数限制）
        for bs in range(batch_size):
            if turn_out_range[bs]:
                status_msgs[bs] = f"Task {GLOBAL_TASK_COUNTER} sample {bs} out of max turn"
                print(status_msgs[bs])
        
        # 添加成功判断和统计功能，参考coldstart_para_his_test_1.5B_hislen8_epoch3.5_v2.py
        success_count = np.sum(success_flags)
        success_rate = success_count / batch_size if batch_size > 0 else 0
        print(f"\n{'='*60}")
        print(f"Rollout Summary:")
        print(f"Total tasks: {batch_size // group_n}, Successful tasks: {success_count}, Success rate: {success_rate:.2%}")
        print(f"Average episode length: {np.mean(episode_lengths):.2f} steps")
        print(f"Average episode reward: {np.mean(episode_rewards):.4f}")
        print(f"{'='*60}")

        
        
        # process_reward = np.array(process_reward)
        # episode_rewards = np.where(episode_rewards == 0, process_reward, episode_rewards)
        return total_batch_list, episode_rewards, episode_lengths, traj_uid, tool_callings, success_flags, status_msgs

    # 新增多轮训练函数，与 rollout_loop_parallel.py 接口一致，但使用 Search 特定的 prompting 和历史构建（对齐coldstart_search_local.py）
    def multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
        is_train: bool = True,
    ) -> DataProto:
        """
        Wrapper method matching the interface in ray_trainer.py.
        Handles training-mode repetition and rollout data gathering.
        """
        # 移除重复的repeat逻辑，统一由vanilla_multi_turn_loop内的group_n逻辑处理batch扩展，避免重复repeat导致batch_size加倍

        total_batch_list, episode_rewards, episode_lengths, traj_uid, tool_callings, success_flags, status_msgs = \
            self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                is_train=is_train,
            )
        
        global GLOBAL_TASK_COUNTER
        # 返回成功标记和状态消息，保持与参考文件相同的轨迹信息结构
        show_case = 1
        # 分别保存到case sample文件夹
        print("="*80)
        if self.print_debug:
            print(f'[DEBUG] show_case {show_case} save_traj {self.save_traj}')
        # print(f'[DEBUG] total_batch_list = {total_batch_list}')
        # 将 total_batch_list 以文本格式写入独立的日志文件（代替 print 到控制台）
        log_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        if show_case:
            log_dir = "case"
            os.makedirs(log_dir, exist_ok=True)
            log_filename = os.path.join(log_dir, f"total_batch_list_{log_time}.log")
            with open(log_filename, 'w', encoding='utf-8') as f:
                f.write(f"[DEBUG] total_batch_list = {total_batch_list}\n")
            print(f'[保存轨迹] total_batch_list has been logged to {log_filename}')
        print("="*80)
        
        # 当前批次的所有任务处理完成后，递增全局计数器
        
        
        
        # 如果全局开关开启，保存当前批次的所有轨迹到新的JSON文件
        if self.save_traj:
            # 生成唯一的文件名，包含全局任务计数器和时间戳，避免覆盖
            timestamp = int(time.time() * 1000)
            # filename = f"sample/traj_batch_{GLOBAL_TASK_COUNTER}_{timestamp}.json"
            filename = f"sample/traj_task_{GLOBAL_TASK_COUNTER}_{log_time}.json"
            # 辅助函数：递归将所有Tensor和numpy数组转换为Python原生类型，确保JSON可序列化
            def tensor_to_native(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.tolist()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.generic):  # 处理numpy的标量类型，如np.int64、np.float64等
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: tensor_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [tensor_to_native(item) for item in obj]
                else:
                    return obj
            
            # 创建保存目录（如果不存在）
            save_dir = os.path.dirname(filename)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            # 需要在保存前去除的长向量键（这些字段数据量大但对轨迹分析无意义）
            _LONG_VEC_KEYS = {
                'prompts', 'responses', 'input_ids',
                'rollout_log_probs', 'attention_mask', 'position_ids',
            }
            def strip_long_vectors(batch_list):
                """递归去除 total_batch_list 中每条记录的长向量字段"""
                for group in batch_list:
                    for item in group:
                        for key in list(item.keys()):
                            if key in _LONG_VEC_KEYS:
                                del item[key]
                return batch_list

            # 准备要保存的数据，包含所有轨迹相关信息，先转换所有非原生类型，再去除长向量
            # total_batch_list, episode_rewards, episode_lengths, traj_uid, tool_callings, success_flags, status_msgs
            _cleaned_batch_list = strip_long_vectors(tensor_to_native(total_batch_list))
            save_data = {
                "batch_idx": GLOBAL_TASK_COUNTER,
                "total_batch_list": _cleaned_batch_list,
                "episode_rewards": tensor_to_native(episode_rewards),
                "episode_lengths": tensor_to_native(episode_lengths),
                "traj_uid": tensor_to_native(traj_uid),
                "tool_callings": tensor_to_native(tool_callings),
                "success_flags": tensor_to_native(success_flags),
                "status_msgs": status_msgs
            }
            # 写入JSON文件（使用 default=str 兜底，防止深层 ndarray 导致写一半崩溃）
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=4, default=str)
            print(f"[保存轨迹] 当前批次轨迹已保存到: {filename}")
        
        GLOBAL_TASK_COUNTER += 1

        # 只有训练阶段需要返回gather_rollout_data生成的DataProto送入PPO训练，验证阶段不需要
        if is_train:
            gen_batch_output: DataProto = self.gather_rollout_data(
                total_batch_list=total_batch_list,
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                traj_uid=traj_uid,
                tool_callings=tool_callings,
                success_flags=success_flags,
                status_msgs=status_msgs,
                world_size=actor_rollout_wg.world_size
            )
            if self.print_debug:
                print(f'[DEBUG] going backward computing')
            return gen_batch_output
        else:
            # 验证阶段使用与参考文件完全相同的成功统计逻辑，不调用gather_rollout_data，避免维度不匹配
            success_count = np.sum(success_flags)
            success_rate = success_count / len(success_flags) if len(success_flags) > 0 else 0
            success_task_indices = [i for i, flag in enumerate(success_flags) if flag == 1]
            
            print(f"\n{'='*60}")
            print(f"Validation Complete Summary:")
            print(f"Total validation tasks: {len(success_flags)}")
            print(f"Successful tasks: {success_count}")
            print(f"Success rate: {success_rate:.2%}")
            print(f"Success task indices: {success_task_indices}")
            print(f"Average episode length: {np.mean(episode_lengths):.2f} steps")
            print(f"Average episode reward: {np.mean(episode_rewards):.4f}")
            print(f"{'='*60}")
            
            # 返回完整的轨迹信息，与参考文件格式保持一致
            return total_batch_list, episode_rewards, episode_lengths, traj_uid, tool_callings, success_flags, status_msgs

            # loss 计算真正依赖的字段只有 responses、attention_mask、input_ids、rewards/episode_rewards、uid/group_id。其他字段（observation、action_dict、gamefile 等）都是辅助性的，不参与梯度计算。

