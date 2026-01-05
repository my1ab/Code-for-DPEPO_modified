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

import torch 
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer 
import uuid 
from verl.models.transformers.qwen2_vl import get_rope_index
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data 
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict 
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from .prompts import *
import re
from tqdm import tqdm
from typing import List,Any

import json

def append_to_json_file(data, filename):
    """
    将字典追加到JSON文件中
    """
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

def extract_think_and_actions(text):
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, text, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else None
    
    actions_pattern = r'<env_\d+>(.*?)</env_\d+>'
    actions = re.findall(actions_pattern, text, re.DOTALL)
    actions_dict = {}
    for index,action in enumerate(actions):
        actions_dict[index + 1] = action
    # actions = [{index+1:action} ]
    
    return {
        'think': think_content,
        'actions': actions_dict
    }

def non_tensor_to_list_of_dict(batch: DataProto) -> list[dict]:
    tensors = batch.batch
    non_tensor = batch.non_tensor_batch
    batch_size = len(tensors['input_ids']) 
    total_data_list = []
    for bs in range(batch_size): 
        save_dict = dict() 
        for key,val in non_tensor.items():
            save_dict[key] = val[bs]
        total_data_list.append(save_dict)
    return total_data_list 

class TrajectoryCollectorParallel:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
    
    def preprocess_single_sample(
        self,
        item: int,
        # gen_batch: DataProto,
        step: int,
        task: str,
        start_obs: str,
        start_possible_action: str,
        history_actions: List,
        history_observations: List, 
        last_action: Dict,
        last_observation: Dict,
        last_possible_actions: Dict,
        num_parallel=5,
        add_limit_prompt=True,
    ): 
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model. 
        
        Parameters: 
            item (int): Sample index in the batch 
            gen_batch (DataProto): Batch data containing original prompts 
        
        Returns: 
            dict: Contains processed input data such as input_ids, attention_mask, etc. 
        """
        
        # data_source = gen_batch.non_tensor_batch['data_source'][item]
        
        # At the Start 
        if step == 0: 
            is_start = len(last_action) == 0 #  and len(history_observations) 
            assert is_start, "The history is not empty at the very begining."
            if type(start_possible_action) == List:
                admissible_actions = [elem for elem in start_possible_action if elem != 'help']
            else:
                admissible_actions = start_possible_action
            
            prompt = compressed_prompt_initial.format(
                task_description=task,
                current_observation=start_obs,
                admissible_actions=admissible_actions
            )

        else: 
            if type(start_possible_action) == List:
                last_possible_actions = [elem for elem in start_possible_action if elem != 'help']
            else:
                last_possible_actions = last_possible_actions
            
            last_action_obv = '' 
            for last_env_idx,action in last_action.items():
                action_cur_env = action 
                poa_cur_env = last_possible_actions
                observation_cur_env = last_observation[last_env_idx] 
                
                last_action_obv += f'In Environment {last_env_idx}\n'
                last_action_obv += f'Action: {action_cur_env}\n'
                last_action_obv += f'Observation: {observation_cur_env}\n'
                last_action_obv += f'Admissible Actions: {poa_cur_env}\n'
            
            history_prompt = '' 
            for step_his_env_idx in history_actions.keys():
                action_cur_env = history_actions[step_his_env_idx]
                obs_cur_env = history_observations[step_his_env_idx] 
                # history_prompt += f'In Environment {step_his_env_idx}\n'
                has_action = False 
                current_history = ''
                for turn_history_idx,(action,z_obs) in enumerate(zip(action_cur_env,obs_cur_env)):
                    has_action = True
                    current_history += f'Action {turn_history_idx + 1}: {action}\n'
                    current_history += f'Observation {turn_history_idx + 1}: {z_obs}\n'
                
                if has_action:
                    history_prompt += f'In Environment {step_his_env_idx}\n' + current_history
            
            if history_prompt != '':
                history_prompt = f'\nYou have already taken multiple actions in multiple parallel environments. Below are the most recent observations and the corresponding actions you took:\n{history_prompt}\n'

            prompt = compressed_prompt_process.format(
                task_description=task,
                initial_observation=start_obs,
                history_info=history_prompt,
                last_history=last_action_obv,
            ) 
        
        if add_limit_prompt:
            prompt = prompt + f'\nYou can explore up to {num_parallel} different environments, ranging from 1 to {num_parallel}.'
        
        generation_completion  = [
            {'role':'system','content':system_prompt},
            {'role':'user','content':prompt}
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
        
        # Process multimodal data
        raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.config.data.truncation,
        )
        
        # append_to_json_file(
        #     data = {
        #         'step':step,
        #         'prompt':prompt_with_chat_template,
        #         'length':attention_mask.sum(dim=1).item()
        #     },
        #     filename='/data/home/zhangjs/disk/project/verl-agent/checkout_input.json'
        # )
        
        position_ids = compute_position_id_with_mask(attention_mask) 

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict 
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            # 'anchor_obs': _obs_anchor,
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
        add_limit_prompt: bool
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        # breakpoint()
        # At the Start 
        data_all_infos = { 
            'start_obs':start_obs,
            'start_possible_actions': start_possible_actions,
            'history_actions':history_actions,
            'history_observations': history_observations,
            'last_actions': last_actions,
            'last_observations': last_observations,
            'last_possible_actions':last_possible_actions,
        }
        
        # Convert to list of dict
        length = len(start_obs)
        save_list = []
        for batch_idx in range(length):
            save_dict = {}
            for key,value in data_all_infos.items():
                if key == 'start_obs': 
                    if '\n\nYour task is to: ' in value[batch_idx]:
                        start_obv, task = value[batch_idx].split('\n\nYour task is to: ')
                        save_dict['start_obs'] = start_obv
                        save_dict['task'] = task
                    else:
                        start_obv, task = value[batch_idx].split('Task Description:\n')
                        save_dict['start_obs'] = start_obv
                        save_dict['task'] = task
                else:
                    save_dict[key] = value[batch_idx]
            save_list.append(save_dict)

        processed_samples = []

        for item, entry in enumerate(save_list):
            # Extract per-sample observations 
            processed = self.preprocess_single_sample(
                item=item,
                # gen_batch=gen_batch,
                step=step,
                task=task,
                start_obs=entry['start_obs'],
                start_possible_action=entry['start_possible_actions'],
                history_actions=entry['history_actions'], 
                history_observations=entry['history_observations'], 
                last_action=entry['last_actions'],
                last_observation=entry['last_observations'],
                last_possible_actions=entry['last_possible_actions'],
                num_parallel=num_parallel,
                add_limit_prompt=add_limit_prompt,
            ) 
            processed_samples.append(processed)

        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
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
            # success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        # success_rate = {} 
        # for key, value in success.items():
        #     success_rate[key] = np.mean(value)
        
        effective_batch = [] 
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs] 
                    # episode_lengths 
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # # success_rate
                    # for key, value in success_rate.items():
                    #     data[key] = value

                    effective_batch.append(data)
        
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        ) 
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)


        # ----------Updated ID Preparation----------
        group_ids = []
        uid_batch = []
        for i in range(batch_size):
            if i % self.config.env.rollout.n == 0:
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
        
        # Initial observations from the environment 
        non_tensor_batch = non_tensor_to_list_of_dict(gen_batch) 
        start_obs,start_possible_actions = envs.get_start_info_group(non_tensor_batch)  
        
        length_obs = len(start_obs) 
        assert len(gen_batch.batch) == length_obs, f"Batch Size:{len(gen_batch.batch)} does not match Observations Size: {length_obs}"
        
        
        
        is_done = np.zeros(batch_size, dtype=bool) 
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32) 
        # Trajectory collection loop
        
        
        # total_trajectory = {}
        for _step in tqdm(range(self.config.env.max_steps)): 
            # TODO: In this step, need to check the group id and sample id 
            # 1. Prepare the Inputs for Generation
            active_masks = np.logical_not(is_done) 
            # Active 4 recording wheather the current trajectories is done.
            # Get the history infos 
            
            non_tensor_batch = non_tensor_to_list_of_dict(gen_batch) 
            '''gen_batch
                - group_id
                - uuid
                - gamefile
                - other dummy infos
            '''
            # breakpoint() 
            history_actions, history_observations = envs.get_history_info_group(non_tensor_batch) 
            last_actions, last_observations, last_possible_actions = envs.get_last_actions_info_group(non_tensor_batch) 
            
            # ## Save Trajectories 
            # for sample, his_action, his_obs, last_action, last_obs, done in zip(non_tensor_batch,history_actions,history_observations,last_actions,last_observations,active_masks):
            #     if not done:
            #         gamefile = sample['gamefile']
            #         group_id = sample['group_id']
            #         if gamefile not in total_trajectory:
            #             total_trajectory[gamefile] = {}
            #             if group_id not in total_trajectory[gamefile]:
            #                 total_trajectory[gamefile][group_id] = {}
            #                 total_trajectory[gamefile][group_id]['actions'] = []
            #                 total_trajectory[gamefile][group_id]['observations'] = []

            #         total_trajectory[gamefile][group_id]['actions'].append(last_action) 
            #         total_trajectory[gamefile][group_id]['observations'].append(last_obs) 



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
                num_parallel=self.config.env.num_parallel,
                add_limit_prompt=self.config.env.add_limit_prompt
            ) 
            '''batch
                - input_ids
                - attention_mask
                - position_ids
                - raw_prompt_ids
                - index
                - data_source
                - raw_prompt
            '''

            # 2. Generate the output of per sample
            # get the inputs 
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            
            # Get the Batch Input
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info
            
            # Generate 
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size) # pad to be divisible by dp_size 
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size) # unpad 
            
            batch.non_tensor_batch['uid'] = uid_batch 
            batch.non_tensor_batch['traj_uid'] = traj_uid 
            # Updated Group ID 
            batch.non_tensor_batch['group_id'] = group_ids
            
            batch = batch.union(batch_output) 
            batch.non_tensor_batch['gamefile'] = gen_batch.non_tensor_batch['gamefile']
            batch.non_tensor_batch['expert_actions'] = gen_batch.non_tensor_batch['expert_actions']
            
            text_actions = self.tokenizer.batch_decode(
                batch.batch['responses'], 
                skip_special_tokens=True
            ) 
            
            batch.non_tensor_batch['action'] = text_actions
            batch.non_tensor_batch['action_dict'] = [extract_think_and_actions(elem)['actions'] for elem in text_actions]
            # Process the text actions here 
            parallel_actions_dict = to_list_of_dict(batch) 
            
            # Interact with the Environments 
            dict_grouped_output = envs.step_group(parallel_actions_dict) 

            # ---------Parallel Penalties ----------- #
            history_actions, history_observations = envs.get_history_info_group(non_tensor_batch) 
            if self.config.reward_model.parallel_reward:
                for sample, cur_history_actions in zip(dict_grouped_output,history_actions):
                    action_dict = sample['action_dict'] 
                    if len(action_dict) != 0:
                        W = self.calculate_penalties(
                            history_actions=cur_history_actions,
                            action_dict=action_dict,
                        ) 
                    else:
                        W = self.config.reward_model.no_action_penalty
                    sample['penalty_W'] = W 
            # history_actions: List, len(history_actions): batch size * n 
            # dict_grouped_output: List[Dict], len(history_actions): batch size * n 
            # history_actions, history_observations = envs.get_history_info_group(non_tensor_batch) 
            # Check the content of 
            # - Get the history actions first
            # - Get the last action
            
            single_dict_grouped_output = collate_fn(dict_grouped_output) 
            # print(single_dict_grouped_output['penalty_W']) 
            
            # breakpoint() 
            next_obs = single_dict_grouped_output['observation']

            # current step's reward 
            np_rewards = np.array(single_dict_grouped_output['rewards'], dtype=object)
            rewards = np.array([
                np.max(lst) if len(lst) > 0 else 0
                for lst in np_rewards
            ]) 
            dones = single_dict_grouped_output['dones'] 
            infos = single_dict_grouped_output['possible_actions'] # actually is possible_actions 
            
            # Create DataProto with preserved metadata
            batch = DataProto.from_single_dict(
                data=single_dict_grouped_output,
                meta_info=gen_batch.meta_info
            )
            
            # Update rewards and episode lengths for active environments
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True) 
            
            
            batch_list: list[dict] = to_list_of_dict(batch) 
            
            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])
            
            is_done = np.logical_or(is_done, rewards)

            # is_all_success = np.logical_or(is_done, rewards)
            # Break if all environments are done
            # if is_done.any():
            #     print() 
            if is_done.all():
                break 
        
        # breakpoint() 
        # ------------------ Calculation Process Reward ---------------------
        if self.config.reward_model.process_reward:
            process_reward = self.calculate_process_reward(
                total_batch_list=total_batch_list,
            ) 
            # We only apply `process reward` when the trajectory fails 
            process_reward = np.array(process_reward) 
            episode_rewards = np.where(episode_rewards == 0, process_reward, episode_rewards)

        # ------------------ Add Some Save Opreations ---------------------
        # import pickle

        # with open('/data/home/zhangjs/disk/project/verl-agent/trajectories.pkl', 'wb') as f:
        #     pickle.dump(total_batch_list, f) 
        
        # breakpoint() 
        return total_batch_list, episode_rewards, episode_lengths, traj_uid, tool_callings

    def calculate_process_reward(self,total_batch_list):
        # TODO: Check the boundary situation

        # merge multi step data into a full trajectory
        group_data = []
        for trajectory in total_batch_list:
            save_dict = {} 
            save_dict['gamefile'] = trajectory[0]['gamefile']
            save_dict['expert_actions'] = trajectory[0]['expert_actions']

            save_dict['parallel_action'] = {}
            save_dict['parallel_obs'] = {} 
            for i in range(self.config.env.num_parallel):
                save_dict['parallel_action'][i+1] = []
                save_dict['parallel_obs'][i+1] = []
            
            for step_data in trajectory:
                action_dict = step_data['action_dict']
                observation_list = step_data['observation']
                
                idx = 0 
                for env_idx, action in action_dict.items(): 
                    if env_idx in set(range(1,self.config.env.num_parallel + 1)):
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
                reward = self.count_longest_ordered_subsequence(expert_action_list,action_list)
                parallel_lcs_rewards.append(reward) 
            
            action_reward = max(parallel_lcs_rewards)
            
            total_process_rewards.append(round(action_reward / len(expert_action_list),5))
        
        return total_process_rewards

    def calculate_penalties(self,history_actions,action_dict):
        # history_actions: Containing history actions in each env
        # sample: only the action_dict is useful
        action_keys = set(action_dict.keys()) & set(range(1,self.config.env.num_parallel + 1))
        for his_key,value in history_actions.items():
            history_actions[his_key] = [his_act.strip() for his_act in value]
        for act_key,value in action_dict.items():
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
                last_state_action = (env_history_actions[-1],env_action)

            # Depth Transition Repeat Count
            COUNT_depth_transition_penalty = self.calculate_transition_repeat(
                history_actions=env_history_actions,
                last_state_action=last_state_action
            )

            # Width Transition Repeat Count 
            LIST_width_repeat_count = []  # TODO: Add Pooling Opreation 
            for w_idx in list(set(action_keys) - set([key])):
                width_history_actions=history_actions[w_idx] + [action_dict[w_idx]]
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

            # width_omega = self.self.config.reward_model.width_omega
            # W_width_repeat = width_omega ** (COUNT_width_transition_repeat - 1)
            
            W_list = [W_depth_repeat,W_depth_t_repeat,W_width_t_repeat]
            pooling_kind_weight = sum(W_list) / len(W_list)

            action_penalty_per_env.append(pooling_kind_weight)
        
        actions_wo_look = [elem for elem in action_dict.values() if elem != 'look']
        COUNT_width_repeat = len(actions_wo_look) - len(set(actions_wo_look)) 
        width_omega = self.config.reward_model.width_omega
        W_width_repeat = width_omega ** COUNT_width_repeat
        
        pooling_w_action = sum(action_penalty_per_env) / len(action_penalty_per_env)
        
        W = (W_width_repeat + pooling_w_action) / 2
        
        return round(W,4) 


    def get_state_action_pair(
        self,
        action_history
        ):
        return [(a, b) for a, b in zip(action_history, action_history[1:])]

    def calculate_transition_repeat(
        self,
        history_actions: List,
        last_state_action,
    ):  
        # Transition Repeat
        full_action_list = history_actions

        state_action_pair_a = self.get_state_action_pair(full_action_list)

        if len(state_action_pair_a) == 0:
            return 0
        
        repeat_count = state_action_pair_a.count(last_state_action)
        
        return repeat_count 

    def calculate_depth_repeat(
        self,
        history_actions: List,
        action: Any,
    ):
        repeat_count = 0
        for history_action in reversed(history_actions):
            if action == history_action:
                repeat_count += 1 
        
        return repeat_count

    def calculate_width_repeat_rate(
        self,
        action_dict
    ):
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

    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """ 
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:
            
            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            ) 
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(
                batch_list=batch_list, 
                episode_rewards=episode_rewards, 
                episode_lengths=episode_lengths, 
                success=success, 
                traj_uid=traj_uid, 
                tool_callings=tool_callings, 
                config=self.config,
                last_try=(try_count == max_try_count),
            )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
        
        # Initial observations from the environment
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_traj_uid, totoal_tool_callings = \
                self.vanilla_multi_turn_loop(
                    gen_batch=gen_batch,
                    actor_rollout_wg=actor_rollout_wg,
                    envs=envs,
                ) 
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)
        
        # Add Judgement Here

        if is_train:
            # Create trajectory data 
            gen_batch_output: DataProto = self.gather_rollout_data(
                total_batch_list=total_batch_list,
                episode_rewards=total_episode_rewards,
                episode_lengths=total_episode_lengths,
                # success=total_success,
                traj_uid=total_traj_uid,
                tool_callings=totoal_tool_callings,
            )
            
            return gen_batch_output

        else:
            return total_batch_list, total_episode_rewards, total_episode_lengths, total_traj_uid, totoal_tool_callings