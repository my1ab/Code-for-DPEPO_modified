# Base Import
import os 
import json  
import ray
import yaml 
import time
import torch
import tempfile 
import numpy as np
from os.path import join as pjoin
import torchvision.transforms as T
from collections import defaultdict
from typing import List, Tuple, Dict, Union, Any 

from tqdm import tqdm

# ALF-World Import
import gymnasium as gym
from gymnasium import spaces

import textworld 
import textworld.gym 
from alfworld.info import ALFWORLD_DATA 
from agent_system.environments.prompts import *
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredExpert, AlfredExpertType
from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment 

# Util Functions
import re
from openai import OpenAI
from copy import deepcopy

# ALFWorld Prompts - Modified for ALFRED environment
system_prompt = """
Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

You are an expert agent operating in the ALFRED embodied environment.
Given a task, you need to reason first in your mind.
Your reasoning process must be enclosed within <think> </think> tags,
for example: <think> reasoning process here </think>.

After thinking, you may take actions. You can either explore multiple parallel environments with multiple actions or take an action in a specific environment.
At the very beginning, every environment have the same status,but each environment is independent, they do not share state changes after actions are taken.
So, parallel actions are executed simultaneously across different environments. The parallel actions are not carried out sequentially.
You must wrap each action in specific environment tags like <env_i> </env_i> to indicate which environment you are acting in.

To take multiple actions at the same time in different environment, use the <parallel> </parallel> tags and wrap each action within its corresponding <env_i> </env_i> tag, where i refers to the i-th environment:

<parallel>
<env_1> possible action 1 </env_1>
...
<env_i> possible action 2 </env_i>
</parallel>

Your output must follow the rules below.
"""

reason_prompt = """
You are an expert agent operating in the ALFRED Embodied Environment. 
Your task is to: {task}.
Your current observation is: {current_observation}
Next Possible Actions: {admissible_actions}

Please output your reasoning and actions following the format:
<think>your reasoning here</think>
<parallel>
<env_1>action 1</env_1>
...
</parallel>
"""

def get_env_name(game_file):
    return game_file.split('json_2.1.1/train/')[-1].replace('/game.tw-pddl','') 

def deepseek(messages):
    client = OpenAI(api_key="Your DeepSeek API Here", base_url="https://api.deepseek.com")

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
    
    # 返回一个动作列表  代表多个并行环境
    return {
        'think': think_content,
        'actions': actions_dict
    }

def read_json(file_path):
    data = json.load(open(file_path,'r'))
    return data

# Set the TMPDIR avoiding the fking `No Space Left On Device`
os.environ['TMPDIR'] = '/diskpool/tmp'   
tempfile.tempdir = '/diskpool/tmp'

# Env Manager Classes

# Base Env
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
        if self.is_done is True:
            obs, reward, done, infos = self.last_command[-1]

            if self.auto_reset:
                reward, done = 0., False
                obs, infos = self.reset() 
        else:
            # def build_env(self,gamefile):
            # Don't Need to figure out the code here, just know what can it do.
            # request_infos = textworld.EnvInfos(facts=True,admissible_commands=True,extras=["gamefile"])

            # env_id = textworld.gym.register_game(gamefile, request_infos, wrappers=[AlfredDemangler(),])
            
            # env = textworld.gym.make(env_id)
            # obs, infos = env.reset()

            # return env,obs,infos
            obs, rewards, dones, infos = self.env.step(action) 
            # `obs` means the result of executing `action` in enviroment
            # There are `admissible_commands` in infos which means the actions that agent can take in next step.
            
        # 成功失败
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
    
    # Reset the status of current env
    def reset(self):
        obs, infos = self.env.reset() 
        return obs, infos
    
    # Build the environment with gamefile
    def build_env(self,gamefile):
        # Don't Need to figure out the code here, just know what can it do.
        request_infos = textworld.EnvInfos(facts=True,admissible_commands=True,extras=["gamefile"])

        env_id = textworld.gym.register_game(gamefile, request_infos, wrappers=[AlfredDemangler(),])
        
        env = textworld.gym.make(env_id)
        obs, infos = env.reset()

        return env,obs,infos


# This class is implemented for parallel agent that can explore multiple parallel
# Multiple paralllel environments serve for a single Agent
class ParallelAlfworldWorker:
    def __init__(self, game_files, num_parallel, num_copied): 
        # For Saving Parallel Environments
        self.env_pools = {} 
        # Initialize 
        for parallel_idx in range(num_parallel): 
            self.env_pools[parallel_idx + 1] = Env(game_files) if num_copied == 0 else [Env(game_files) for _ in range(num_copied)] 
        
        # Record the start `observations` and `possible commands` in next step.
        self.start_obv = self.env_pools[1].start_obv 
        self.admissible_commands = self.env_pools[1].start_infos['admissible_commands'] 

        # self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        # self.env.seed(seed)

    def show_basis_infos(self):
        return self.start_obv,self.admissible_commands
    
    # Execute parallel actions in parallel Environments
    def step(self, action_dict): 
        obs,scores,dones,infos = [],[],[],[]
        obs_prompt = '' 
        for action_indx,action in action_dict.items():
            sub_env = self.env_pools[action_indx]
            ob,reward,done,info = sub_env.step(action)

            admissible_commands = ','.join(info['admissible_commands']) 

            obs_prompt += f'<observation_{action_indx}>\nThe observation and next candidated actions of {action_indx}-th environment are:\nObservation:\n{ob}\nNext Possible Actions:\n{admissible_commands}\n</observation_{action_indx}>\n'

            obs.append(ob) 
            scores.append(reward)
            dones.append(done)
            infos.append(info)
    
        return obs, scores, dones, infos, obs_prompt
    
    # Reset
    def reset(self):
        """Reset the environment"""

        for env in self.env_pools.values():
            obs, infos = env.reset() 
        
        return obs, infos

# For a single task, sample a group of answers, each group has `group_n` answer
# This class is implemented for `GRPO` Algorithms or some situations requiring sample multiple answer
class ParallelAlfworldEnvs(gym.Env):
    def __init__(self, 
                 game_files,
                 group_n, 
                 resources_per_worker, 
                 is_train=True, 
                 num_parallel=10,
                 num_copied=0,
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

        for game_file in game_files: 
            worker = env_worker.remote(game_file,num_parallel, num_copied)
            self.workers.append(worker) 
            self.workers_dict[get_env_name(game_file)] = worker
    
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
                                #  env_num, 
                                 group_n, 
                                 resources_per_worker, 
                                 num_parallel,
                                 num_copied,
                                 is_train=True, 
                                 env_kwargs={}):
    return ParallelAlfworldEnvs(gamefiles,
                                # env_num, 
                                group_n, 
                                resources_per_worker, 
                                is_train,
                                num_parallel=num_parallel,
                                num_copied=num_copied) 

# Main Logic - Modified to generate ALFWorld.json format
# Ã¤Â½Â¿Ã¥â€¦Â¶Ã§Â¬Â¦Ã¥ÂË†Ã¨Â¿â€Ã¥â€ºÅ¾ALFWorld.jsonÃ¦Â Â¼Ã¥Â¼ÂÃ§Å¡â€žÃ¨Â½Â¨Ã¨Â¿Â¹
def get_single_trajectory(parallel_env, game_file, turns=50):
    """
    Generate a single trajectory and return data in ALFWorld.json format
    """
    obs, admissible_commands = parallel_env.get_start_info_file(game_file)
    
    # Build the initial prompt in ALFWorld.json format
    formatted_reason_prompt = reason_prompt.format(
        task="cool some egg and put it in microwave.",  # Default task for demonstration
        current_observation=obs,
        admissible_actions=','.join(admissible_commands)
    )
    
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{formatted_reason_prompt}"
    
    # Get LLM response
    conversations = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': formatted_reason_prompt}
    ]
    
    output = deepseek(messages=conversations)
    result = extract_think_and_actions(output)
    
    # Build completion in ALFWorld.json format
    if result['think'] and result['actions']:
        actions_str = "\n".join([f"<env_{k}>{v}</env_{k}>" for k, v in result['actions'].items()])
        completion = f"<think>\n{result['think']}\n</think>\n\n<parallel>\n{actions_str}\n</parallel>"
    else:
        completion = f"<think>\nNo reasoning available\n</think>\n\n<parallel>\n</parallel>"
    
    # Calculate length
    length = len(prompt) + len(completion)
    
    return {
        'prompt': prompt,
        'completion': completion,
        'length': length
    }

def generate_coldstart_data(game_path, output_file, max_files=5, turns=50):
    """
    Generate coldstart data in ALFWorld.json format
    """
    data = read_json(game_path)
    todo_files = list(data.items())[:max_files]
    todo_game_files = [elem[-1] for elem in todo_files]
    
    print('Loading Environments...')
    env_start_time = time.time()
    parallel_env = build_parallel_alfworld_envs(
        gamefiles=todo_game_files,
        group_n=1,
        resources_per_worker={'num_cpus': 0.1},
        num_parallel=10,
        num_copied=0,
        is_train=True
    )
    
    output = parallel_env.reset()
    print(f'Environments loaded, took {time.time() - env_start_time} seconds')
    
    # Generate data
    coldstart_data = []
    for key, game_file in tqdm(todo_files, desc="Generating coldstart data"):
        try:
            trajectory = get_single_trajectory(parallel_env, game_file, turns=turns)
            coldstart_data.append(trajectory)
        except Exception as e:
            print(f"Error processing {game_file}: {e}")
            continue
    
    # Save to output file
    with open(output_file, 'w') as f:
        json.dump(coldstart_data, f, indent=4)
    
    print(f"Coldstart data saved to {output_file}")
    print(f"Total {len(coldstart_data)} entries generated")
    
    # Cleanup
    parallel_env.close()
    ray.shutdown()
    
    return coldstart_data

if __name__ == "__main__":
    # Configuration
    GAME_PATH = '/dir_path/gamefiles_train.json'  # Update this path
    OUTPUT_FILE = '/home/dpepo/Code-for-DPEPO-main/data_pipelines/ALFWorld_coldstart.json'
    
    # Generate coldstart data
    generate_coldstart_data(GAME_PATH, OUTPUT_FILE, max_files=5, turns=50)