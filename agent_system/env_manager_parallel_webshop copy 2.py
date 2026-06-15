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

import gym
from openai import OpenAI

from tqdm import tqdm

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
# Basic Env Manager
class Env:
    def __init__(self, game_file):
        self.gamefile = game_file
        self.env_path = '/data/home/zhangjs/disk/project/verl-agent/agent_system/environments/env_package/webshop/webshop'
        self.env_kwargs = {
            'observation_mode': 'text', 
            'num_products': None, 
            'human_goals': False,
            'file_path': '/data/home/zhangjs/disk/project/verl-agent/agent_system/environments/env_package/webshop/webshop/data/items_shuffle_1000.json',
            'attr_path': '/data/home/zhangjs/disk/project/verl-agent/agent_system/environments/env_package/webshop/webshop/data/items_ins_v2_1000.json'
        } 
        env, obs, infos = self.build_env(
            gamefile=game_file,
            env_kwargs=self.env_kwargs,
        )
        self.env = env 
        self.start_obv = obs 
        self.start_infos = infos 
        self.last_command = []
        self.auto_reset = False
        self.is_done = False
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action) 
        
        # 处理 info 可能为 None 的情况
        info = dict(info or {})
        available_actions = self.env.get_available_actions()  # 直接从环境获取

        # Formated the obs and actions
        formated_obs = self.format_obs(obs, add_task=False) 
        possible_actions = self.format_possible_actions(available_actions) 
        
        if done:
            self.is_done = True 

        self.last_command.append(
            {
                'action': action,
                'observation': formated_obs,
                'rewards': reward,
                'dones': done,
                'possible_commands': possible_actions,
                'game_file': self.gamefile
            }
        )
        
        copied_info = dict(info) 
        copied_info['admissible_commands'] = possible_actions
        copied_info['task_score'] = reward
        if done and reward == 1.0:
            copied_info['won'] = True
        else:
            copied_info['won'] = False

        return formated_obs, reward, done, copied_info
    
    def reset(self):
        obs, infos = self.env.reset(session=self.gamefile) 
        
        # 处理 infos 可能为 None 的情况
        infos = dict(infos or {})
        self.task = self.extract_task(obs) 
        formated_obs = self.format_obs(obs, add_task=True) 
        
        copied_infos = {}
        available_actions = self.env.get_available_actions()  # 直接从环境获取
        copied_infos['admissible_commands'] = self.format_possible_actions(available_actions)

        return formated_obs, copied_infos

    
    def build_env(self, gamefile, env_kwargs): # the gamefile is actually seed
        import sys 
        import os 
        sys.path.append(self.env_path) 

        from web_agent_site.envs import WebAgentTextEnv
        
        env_kwargs['seed'] = gamefile 
        env = gym.make('WebAgentTextEnv-v0', **env_kwargs) 
        
        obs, infos = env.reset(session=gamefile) 
        
        # 处理 infos 可能为 None 的情况
        infos = dict(infos or {})
        self.task = self.extract_task(obs) 
        formated_obs = self.format_obs(obs, add_task=True) 

        copied_infos = {}
        available_actions = env.get_available_actions()  # 直接从环境获取
        copied_infos['admissible_commands'] = self.format_possible_actions(available_actions)

        return env, formated_obs, copied_infos
    
    def extract_task(self, text_obs):
        parts = text_obs.split(" [SEP] ")
        task = parts[2]
        return task
    
    def format_obs(self, text_obs, add_task=True):
        parts = text_obs.split(" [SEP] ")
        # the index of self.tasks[i] in parts
        try:
            index = parts.index(self.task) 
            reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
        except:
            reformatted_obs = text_obs
        if add_task:
            return reformatted_obs + '\n\nYour task is to: ' + self.task
        else:
            return reformatted_obs
    
    def format_possible_actions(self, possible_actions):
        actions = []
    
        for key in possible_actions.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if possible_actions["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in possible_actions["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
    
    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render)
        return rendered
    
    def get_available_actions(self):
        """Get available actions"""
        return self.env.get_available_actions()
    
    def get_goals(self):
        """Get environment goals"""
        return self.env.server.goals
    
    def close(self):
        """Close the environment"""
        self.env.close()
    
    

# Sinlge Parallel Env Manager 
class ParallelWebShopWorker:
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
            if action_index < len(self.env_pools):
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
class ParallelWebShopEnvs(gym.Env):
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
        env_worker = ray.remote(**resources_per_worker)(ParallelWebShopWorker)
        self.workers = [] 
        self.workers_dict = {} 
        for game_file in tqdm(game_files): 
            # env_name = get_env_name(game_file)
            self.workers_dict[game_file] = []
            for group_idx in range(self.group_n):
                worker = env_worker.remote(game_file,num_parallel)
                self.workers.append(worker) 
                self.workers_dict[game_file].append(worker)
            print(len(self.workers_dict[game_file])) 

        print(len(self.workers_dict))
        
        # self.prev_admissible_commands = [None for _ in range(self.num_processes)]
    
    def get_start_info_group(self,gourped_samples):
        futures = [] 
        for sample in gourped_samples:
            gamefile = sample['gamefile']
            group_id = sample['group_id']

            # sub_gamefile_name = get_env_name(gamefile) 
            current_worker = self.workers_dict[gamefile][group_id]
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

            # sub_gamefile_name = get_env_name(gamefile) 
            current_worker = self.workers_dict[gamefile][group_id]
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

            # sub_gamefile_name = get_env_name(gamefile) 
            current_worker = self.workers_dict[gamefile][group_id]
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

            # sub_gamefile_name = get_env_name(gamefile) 
            current_worker = self.workers_dict[gamefile][group_id]
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

            # sub_gamefile_name = get_env_name(gamefile) 
            current_worker = self.workers_dict[gamefile][group_id]
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
        # sub_gamefile = get_env_name(game_file)
        worker = self.workers_dict[game_file] 
        future = worker.step.remote(action) 
        results = ray.get(future) 
        # results = future.results() 
        return results[0], results[1], results[2], results[3], results[4] 
    
    def get_start_info_file(self,game_file):
        # sub_gamefile = get_env_name(game_file)
        worker = self.workers_dict[game_file] 
        future = worker.show_basis_infos.remote()
        results = ray.get(future)

        return results[0],results[1] # obv,infos


    def reset_file(self,game_file):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        # sub_gamefile = get_env_name(game_file)
        worker = self.workers_dict[game_file] 

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

def build_parallel_webshop_envs(gamefiles,
                                 group_n, 
                                 resources_per_worker, 
                                 num_parallel,
                                 env_kwargs={}):
    return ParallelWebShopEnvs(gamefiles,
                                group_n, 
                                resources_per_worker, 
                                num_parallel=num_parallel
                                ) 