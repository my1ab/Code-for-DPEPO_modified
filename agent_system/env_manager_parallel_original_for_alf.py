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

from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory
from omegaconf import OmegaConf

from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos) 
        # initialize the history buffer 
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions}) 
        self.pre_text_obs = text_obs 
        # Build the next prompts or observations for next step
        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None: 
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
    
    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
        
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break


# ---------------
## -----Import Lines
# standard libraries
import re
import os
import json
import yaml
import tempfile 
from copy import deepcopy
from os.path import join as pjoin

import ray
import torch
import numpy as np
import torchvision.transforms as T

import gymnasium as gym
from openai import OpenAI
from gymnasium import spaces

import textworld 
import textworld.gym 
from tqdm import tqdm

from alfworld.info import ALFWORLD_DATA 
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredExpert, AlfredExpertType
from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment


# Some Env settings
os.environ['TMPDIR'] = '/diskpool/tmp'   
tempfile.tempdir = '/diskpool/tmp'  
ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]

# Some Util Functions
def read_json(file_path):
    return json.load(open(file_path,'r'))



def get_env_name(game_file):
    return game_file.split('json_2.1.1/train/')[-1].replace('/game.tw-pddl','') 

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward


def deepseek(messages):
    client = OpenAI(api_key="sk-2b259711001b4a01b9267a258a92b75e", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False,
        temperature=1.5
    )
    
    return response.choices[0].message.content 

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

# Basic Env Manager
class Env:
    def __init__(self,game_file):
        self.gamefile = game_file
        env,obs,infos = self.build_env(gamefile=game_file)
        self.env = env
        self.start_obv = obs 
        self.start_infos = infos 
        self.last_command = []
        self.auto_reset = False
        self.is_done = False
    
    def step(self,action):
        # if self.is_done is True:
        #     obs, reward, done, infos = self.last_command[-1] 

        #     if self.auto_reset:
        #         reward, done = 0., False
        #         obs, infos = self.reset() 

        # else:
        #     obs, rewards, dones, infos = self.env.step(action) 
        obs, rewards, dones, infos = self.env.step(action) 

        if dones:
            self.is_done = True 

        self.last_command.append(
            {
                'action':action,
                'observation':obs,
                'rewards':rewards,
                'dones':dones,
                'possible_commands':infos['admissible_commands'],
                'game_file':infos['admissible_commands']
            }
        )
        
        return obs, rewards, dones, infos
    
    def reset(self):
        obs, infos = self.env.reset() 
        return obs, infos
    
    def build_env(self,gamefile):
        request_infos = textworld.EnvInfos(facts=True,admissible_commands=True,extras=["gamefile"])

        env_id = textworld.gym.register_game(gamefile, request_infos, wrappers=[AlfredDemangler(),])
        
        env = textworld.gym.make(env_id)
        obs, infos = env.reset()

        return env,obs,infos

# Sinlge Parallel Env Manager 
class ParallelAlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, game_files, num_parallel): 
        self.env_pools = {} 

        self.action_manager = {} 
        self.obs_manager = {}

        self.last_action_manager = {}
        self.last_obs_manager = {}
        self.last_poa_manager = {}
        

        for parallel_idx in range(num_parallel): 
            self.action_manager[parallel_idx + 1] = []
            self.obs_manager[parallel_idx + 1] = []

            self.env_pools[parallel_idx + 1] = Env(game_files)
        
        self.start_obv = self.env_pools[1].start_obv 
        self.admissible_commands = self.env_pools[1].start_infos['admissible_commands'] 

        # self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        # self.env.seed(seed)

    def show_basis_infos(self):
        return self.start_obv,self.admissible_commands
    
    def get_history_infos(self):
        return self.action_manager, self.obs_manager 
    
    def get_last_actions(self):
        return self.last_action_manager, self.last_obs_manager, self.last_poa_manager

    
    def step(self, action_dict):
        obs,scores,dones,infos = [],[],[],[]
        obs_prompt = ''

        # Only at the start stage, this code does not run
        if len(self.last_action_manager) != 0 and len(self.last_obs_manager) != 0:
            for last_action_idx,last_action in self.last_action_manager.items():
                self.action_manager[last_action_idx].append(last_action)
            
            for last_obs_idx,last_obs in self.last_obs_manager.items():
                self.obs_manager[last_obs_idx].append(last_obs) 

            self.last_action_manager = {}
            self.last_obs_manager = {}
            self.last_poa_manager = {} 

        for action_index,action in action_dict.items():
            if action_index <= len(self.env_pools):
                sub_env = self.env_pools[action_index] 
                
                ob,reward,done,info = sub_env.step(action) 

                self.last_action_manager[action_index] = action 
                self.last_obs_manager[action_index] = ob
                self.last_poa_manager[action_index] = info['admissible_commands']
                
                admissible_commands = ','.join(info['admissible_commands']) 

                obs_prompt += f'<observation_{action_index}>\nThe observation and next candidated actions of {action_index}-th environment are:\nObservation:\n{ob}\nNext Possible Actions:\n{admissible_commands}\n</observation_{action_index}>\n'

                obs.append(ob) 
                scores.append(reward)
                dones.append(done)
                infos.append(info) 
            else:
                obs.append(f'The action index {action_index} is invalid. Valid indices range from 1 to {len(self.env_pools)}.') 
                scores.append(0) 
                dones.append(False) 
                infos.append({'admissible_commands': f'The action index {action_index} is invalid. Valid indices range from 1 to {len(self.env_pools)}.'}) 
                
    
        return obs, scores, dones, infos, obs_prompt 
    
    
    def reset(self):
        """Reset the environment"""

        for env in self.env_pools.values():
            obs, infos = env.reset() 
        
        return obs, infos

# Env Manager that manages the Grouped Parallel Workers
class ParallelAlfworldEnvs(gym.Env):
    def __init__(self, 
                 game_files,
                 group_n, 
                 resources_per_worker, 
                 num_parallel=10,
                 env_kwargs={}):
        super().__init__() 
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
        
        self.multi_modal = False
        # self.num_processes = env_num * group_n
        self.group_n = group_n
        
        # Create Ray remote actors instead of processes 
        env_worker = ray.remote(**resources_per_worker)(ParallelAlfworldWorker)
        self.workers = [] 
        self.workers_dict = {} 
        for game_file in tqdm(game_files): 
            env_name = get_env_name(game_file)
            self.workers_dict[env_name] = []
            for group_idx in range(self.group_n):
                worker = env_worker.remote(game_file,num_parallel)
                self.workers.append(worker) 
                self.workers_dict[env_name].append(worker)
        
        # self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def get_start_info_group(self,gourped_samples):
        futures = [] 
        for sample in gourped_samples:
            gamefile = sample['gamefile']
            group_id = sample['group_id']

            sub_gamefile_name = get_env_name(gamefile) 
            current_worker = self.workers_dict[sub_gamefile_name][group_id]
            future = current_worker.show_basis_infos.remote()
            futures.append(future) 
        results = ray.get(futures) 

        obvs = [elem[0] for elem in results]
        possible_actions = [elem[1] for elem in results]
        
        return obvs, possible_actions 
    
    def get_history_info_group(self, gourped_samples):
        futures = [] 
        for sample in gourped_samples:
            gamefile = sample['gamefile']
            group_id = sample['group_id']
            
            sub_gamefile_name = get_env_name(gamefile) 
            current_worker = self.workers_dict[sub_gamefile_name][group_id]
            future = current_worker.get_history_infos.remote()
            futures.append(future) 
        results = ray.get(futures) 

        actions = [elem[0] for elem in results]
        observations = [elem[1] for elem in results]
        
        return actions, observations 
    
    def get_last_actions_info_group(self, gourped_samples):
        futures = [] 
        for sample in gourped_samples:
            gamefile = sample['gamefile']
            group_id = sample['group_id']

            sub_gamefile_name = get_env_name(gamefile) 
            current_worker = self.workers_dict[sub_gamefile_name][group_id]
            future = current_worker.get_last_actions.remote()
            futures.append(future) 
        results = ray.get(futures) 

        actions = [elem[0] for elem in results]
        observations = [elem[1] for elem in results]
        poas = [elem[2] for elem in results]
        
        return actions, observations, poas
    
    def get_last_actions_group(self, gourped_samples):
        futures = [] 
        for sample in gourped_samples: 
            gamefile = sample['gamefile'] 
            group_id = sample['group_id'] 

            sub_gamefile_name = get_env_name(gamefile) 
            current_worker = self.workers_dict[sub_gamefile_name][group_id]
            future = current_worker.get_history_infos.remote()
            futures.append(future) 
        results = ray.get(futures) 
        
        actions = [elem[0] for elem in results]
        observations = [elem[1] for elem in results]
        
        return actions, observations 
    
    def step_group(self, gourped_samples):
        # [
        #     {'uuid':'xxx','group_id':'xxx','actions':'xxx','gamefile':'xxx'},
        #     {'uuid':'xxx','group_id':'xxx','actions':'xxx','gamefile':'xxx'}
        # ] 
        futures = [] 
        for sample in gourped_samples:
            gamefile = sample['gamefile']
            action = sample['action_dict']
            group_id = sample['group_id']

            sub_gamefile_name = get_env_name(gamefile) 
            current_worker = self.workers_dict[sub_gamefile_name][group_id]
            future = current_worker.step.remote(action)
            futures.append(future)
        
        observation_list = []
        scores_list = []
        dones_list = []
        infos_list = []
        obs_prompt_list = []

        results = ray.get(futures)
        for result,sample in zip(results,gourped_samples):
            obs = result[0]
            scores = result[1]
            dones = result[2]
            infos = result[3] 
            prompts = result[4] 
            # sample['result'] = {
            #     'observation':obs,
            #     'rewards':scores,
            #     'dones':dones, 
            #     'possible_actions':[elem['admissible_commands'] for elem in infos],
            #     'concated_observation':prompts
            # } 
            sample.update({
                'observation':obs,
                'rewards':scores,
                'dones':dones, 
                'possible_actions':[elem['admissible_commands'] for elem in infos],
                # 'possible_actions':[elem['admissible_commands'] for elem in infos],
                'concated_observation':prompts
            }) 
            # observation_list.append(obs)
            # scores_list.append(scores) 
            # dones_list.append(dones) 
            # infos_list.append(infos)  
            # obs_prompt_list.append(prompts)

        return gourped_samples
    
    # This actions should be a parsed dict: 
    # - key: index for the environment 
    # - value: corresponding actions 
    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = [] 
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i]) 
            futures.append(future) 
        
        # Collect results
        observation_list = []
        scores_list = []
        dones_list = []
        infos_list = []
        obs_prompt_list = []
        
        results = ray.get(futures)
        for i, (obs, scores, dones, infos,prompts) in enumerate(results):
            observation_list.append(obs)
            scores_list.append(scores)
            dones_list.append(dones)
            infos_list.append(infos)
            obs_prompt_list.append(prompts)
        
        return observation_list, scores_list, dones_list, infos_list, obs_prompt_list
    
    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)
        
        obs = []
        infos = [] 
        results = ray.get(futures)
        for obv,info in results:
            obs.append(obv)
            infos.append(info)
        
        return obs, infos 
    
    def step_file(self,game_file,action):
        sub_gamefile = get_env_name(game_file)
        worker = self.workers_dict[sub_gamefile] 
        future = worker.step.remote(action) 
        results = ray.get(future) 
        # results = future.results() 
        return results[0], results[1], results[2], results[3], results[4] 
    
    def get_start_info_file(self,game_file):
        sub_gamefile = get_env_name(game_file)
        worker = self.workers_dict[sub_gamefile] 
        future = worker.show_basis_infos.remote()
        results = ray.get(future)

        return results[0],results[1] # obv,infos


    def reset_file(self,game_file):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        sub_gamefile = get_env_name(game_file)
        worker = self.workers_dict[sub_gamefile] 

        future = worker.reset.remote()
        result = ray.get(future)
        
        return result[0], result[1]
    
    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands 

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_parallel_alfworld_envs(gamefiles,
                                 group_n, 
                                 resources_per_worker, 
                                 num_parallel,
                                 env_kwargs={}):
    print(f'Here is the num_parallel: {num_parallel}') 
    return ParallelAlfworldEnvs(gamefiles,
                                group_n, 
                                resources_per_worker, 
                                num_parallel=num_parallel
                                ) 