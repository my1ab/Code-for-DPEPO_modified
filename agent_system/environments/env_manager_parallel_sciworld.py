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


# ---------------
import os
import re
import ray 
import tempfile 
import gymnasium as gym
from typing import List
from scienceworld import ScienceWorldEnv 
from agent_system.environments.prompts import *

# Some Env settings
os.environ['TMPDIR'] = '/diskpool/tmp'   
tempfile.tempdir = '/diskpool/tmp'  
ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]

# Some Util Functions
def read_json(file_path):
    return json.load(open(file_path,'r'))


def get_env_name(game_file):
    return f'{game_file[0]}_{game_file[1]}'

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward


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
        env,obs,infos = self.build_env(gamefile=game_file)  # Build the Environment
        # Initialize some class attributes
        self.env = env
        self.start_obv = obs  # Record the start observation
        self.start_infos = infos  # Record the start infos
        self.last_command = [] # Record the actions in last step
        self.auto_reset = True
        self.is_done = False
    
    # Execute a action in this environment
    def step(self,action):
        observation, reward, done, infos = self.env.step(action) 
        valid_actions = self.env.get_possible_actions() 
        valid_objs = self.env.get_possible_objects() 
        valid_action_strs = f"Valid_actions: {valid_actions}, OBJ needs to be replaced with one of the following objects: {valid_objs}\n example: <env_i>focus on door</env_i>"
        
        infos['admissible_commands'] = valid_action_strs
        infos['observation_text'] = observation 
        infos["possible_actions"] = self.env.get_valid_action_object_combinations() 
        infos['score'] = reward 

        
        isCompleted = done
        won = isCompleted and infos["score"] > 0

        reward = 1 if won else 0
        
        return observation, reward, won, infos
    
    # Reset the status of current env
    def reset(self):
        obs, infos = self.env.reset() 
        valid_actions = self.env.get_possible_actions() 
        valid_objs = self.env.get_possible_objects()

        valid_action_strs = f"Valid_actions: {valid_actions}, OBJ needs to be replaced with one of the following objects: {valid_objs}\n example: <env_i>focus on door</env_i>"
        
        infos['admissible_commands'] = valid_action_strs
        return obs, infos
    
    # Build the environment with gamefile
    def build_env(self,gamefile):
        task_name = gamefile[0]
        variant_id = gamefile[1]
        env = ScienceWorldEnv()
        
        
        # Don't Need to figure out the code here, just know what can it do.
        env.load(task_name, variant_id) 
        initial_obs, initial_info = env.reset() 
        valid_actions = env.get_possible_actions() 
        valid_objs = env.get_possible_objects()
        
        valid_action_strs = f"Valid_actions: {valid_actions}, OBJ needs to be replaced with one of the following objects: {valid_objs}\n example: <env_i>focus on door</env_i>"
        
        initial_info['admissible_commands'] = valid_action_strs
        
        return env,initial_obs,initial_info


# This class is implemented for parallel agent that can explore multiple parallel
# Multiple paralllel environments serve for a single Agent 
class ParallelSciWorldWorker:
    def __init__(self, game_files, num_parallel): 
        # For Saving Parallel Environments
        self.env_pools = {} 

        self.action_manager = {} 
        self.obs_manager = {}

        self.last_action_manager = {}
        self.last_obs_manager = {}
        self.last_poa_manager = {}

        # Initialize 
        for parallel_idx in range(num_parallel): 
            self.action_manager[parallel_idx + 1] = []
            self.obs_manager[parallel_idx + 1] = []
            print(f'gamefiles:{game_files}')
            self.env_pools[parallel_idx + 1] = Env(game_files)
        
        # Record the start `observations` and `possible commands` in next step.
        task_desc = self.env_pools[1].start_infos['taskDesc'] 
        self.start_obv = self.env_pools[1].start_obv + task_desc 
        self.admissible_commands = self.env_pools[1].start_infos['admissible_commands'] 
        
        # self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        # self.env.seed(seed)
    
    def show_basis_infos(self):
        return self.start_obv,self.admissible_commands
    
    def get_history_infos(self):
        return self.action_manager, self.obs_manager 

    def get_last_actions(self):
        return self.last_action_manager, self.last_obs_manager, self.last_poa_manager
    
    # Execute parallel actions in parallel Environments
    def step(self, action_dict):
        obs,scores,dones,infos = [],[],[],[]
        obs_prompt = ''

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

                admissible_commands = info['admissible_commands']
                if type(admissible_commands) == List:
                    admissible_commands = ','.join(admissible_commands) 
                else:
                    admissible_commands = admissible_commands
                
                self.last_action_manager[action_index] = action 
                self.last_obs_manager[action_index] = ob
                self.last_poa_manager[action_index] = admissible_commands
                
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
    
    # Reset
    def reset(self):
        """Reset the environment"""

        for env in self.env_pools.values():
            obs, infos = env.reset() 
        
        return obs, infos  

# ----------------------- New 

# For a single task, sample a group of answers, each group has `group_n` answer
# This class is implemented for `GRPO` Algorithms or some situations requiring sample multiple answer
class ParallelSciWorldEnvs(gym.Env):
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
        env_worker = ray.remote(**resources_per_worker)(ParallelSciWorldWorker)
        self.workers = [] 
        self.workers_dict = {} 
        
        for game_file in tqdm(game_files): 
            env_name = get_env_name(game_file)
            self.workers_dict[env_name] = []

            for group_idx in range(self.group_n):
                worker = env_worker.remote(game_file, num_parallel)
                self.workers.append(worker) 
                self.workers_dict[get_env_name(game_file)].append(worker)
    
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

    def step_group(self, gourped_samples):
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
            
            sample.update({
                'observation':obs,
                'rewards':scores,
                'dones':dones, 
                'possible_actions':[elem['admissible_commands'] for elem in infos],
                # 'possible_actions':[elem['admissible_commands'] for elem in infos],
                'concated_observation':prompts
            }) 
        
        return gourped_samples

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
        if type(game_file) != str:
            sub_gamefile = get_env_name(game_file)
        else:
            sub_gamefile = game_file
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
    
    # def step_group(self, game_files, action_dicts): 
    #     futures = [] 
    #     for game_file,action_dict in zip(game_files, action_dicts):
    #         current_worker = self.workers_dict[game_file]
    #         future = current_worker.step.remote(action_dict)
    #         futures.append(future)

    #     result_list = []
    #     results = ray.get(futures) 
    #     for game_file, result in zip(game_files,results):
    #         obs = result[0]
    #         scores = result[1]
    #         dones = result[2]
    #         infos = result[3] 
    #         prompts = result[4] 

    #         result_list.append({
    #             'game_file':game_file,
    #             'observation':obs,
    #             'rewards':scores,
    #             'dones':dones, 
    #             'possible_actions':[elem['admissible_commands'] for elem in infos],
    #             'concated_observation':prompts
    #         }) 
            
    #         # observation_list.append(obs)
    #         # scores_list.append(scores) 
    #         # dones_list.append(dones) 
    #         # infos_list.append(infos)  
    #         # obs_prompt_list.append(prompts) 

    #     return result_list


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

def build_parallel_sciworld_envs(gamefiles,
                                 group_n, 
                                 resources_per_worker, 
                                 num_parallel,
                                 env_kwargs={}):
    return ParallelSciWorldEnvs(gamefiles,
                                group_n, 
                                resources_per_worker, 
                                num_parallel=num_parallel
                                ) 