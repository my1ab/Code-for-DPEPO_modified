# ReACT Prompts 
ReACT_ALFWORLD_TAGGING_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment.  
I will provide you with a successful trajectory. You need to supplement the reasoning process.

**Reason using ONLY ONE tag pair and express your reasoning in one sentence:**
<think> Your supplemented reasoning process here</think>

**Here are some aspects for you to generate better, logical, appropriate reasoning process based on the observation and status.**
Decompose a complex overall task into clear subgoals, listing each milestone as a separate point. Focus on outlining the full sequence required to complete the overall task, not just the immediate next action.
This approach is typically used at the initial stage of a task, or when significant problems or uncertainties arise that may require re-planning.
All points must be listed explicitly and separately, such as: Step 1: xxx; Step 2: xxx; Step 3: xxx; and so on.

When immediate next steps have a clear exploratory nature—such as when searching for an unknown object or information—use current observations to think outside the box and generate as many possible hypotheses, locations, items, or actions as possible.
Employ this approach when results are unexpected, information is lacking, or obstacles demand creative and innovative problem-solving.

Reflect on the current state, task progress, objectives, and reasons for failures when the task has stalled for an extended period, incorrect actions have been taken, or the current situation has been misjudged. Analyze potential causes for errors or lack of progress, and consider alternative strategies or perspectives to overcome obstacles.
This is especially useful when several consecutive actions do not yield breakthroughs, or when persistent mistakes indicate the need for a deeper reassessment.

Continuously track the current progress and history of reasoning and execution throughout the task.
Firstly recall the current subgoal based on the previously established overall plan, then consider the next action required to achieve this subgoal.
Typically used when task outcomes are as expected and no other mode of reasoning is required.

**When you generating the reasoning at timesteps t, you can only see the timesteps at 1~t-1. For the analysis of past states, use the past tense; for what is to be done next in reasoning, use the future tense like `will`.
In your generated reason process, you must use `I` instaed of some others like `AI`**
You need to output a list in JSON format, with the same length as the trajectory. Each element should contain two key-value pairs, for example:  
```json
[{{"reason": "<think>The book may be in the cabinet, shelf, so in the next steps I need to search these locations.</think>", "action": "go to shelf 1"}},
{{"reason": "<think>Currently, my sub-goal is to obtain item A. I have already spotted A, and in order to accomplish this objective, I need to pick it up.</think>", "action": "pick up A"}}]
```
The "action" field must match the action in the trajectory, and the "reason" field should be a reasonable reasoning process inferred from the context of previous actions and the next few actions.
the task description and initial observations are as follows:{init_status}
now the trajectory is as follows: {traj}
"""


ALFWORLD_TAGGING_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment.  
I will provide you with a successful trajectory. You need to supplement the reasoning process.

**Reason using ONLY ONE tag pair and express your reasoning in one concise, brief sentence:**

<planning>
Decompose a complex overall task into clear subgoals, listing each milestone as a separate point. Focus on outlining the full sequence required to complete the overall task, not just the immediate next action.
This approach is typically used at the initial stage of a task, or when significant problems or uncertainties arise that may require re-planning.
All points must be listed explicitly and separately, such as: Step 1: xxx; Step 2: xxx; Step 3: xxx; and so on.
</planning>

<explore>
When immediate next steps have a clear exploratory nature—such as when searching for an unknown object or information—use current observations to think outside the box and generate as many possible hypotheses, locations, items, or actions as possible.
Employ this approach when results are unexpected, information is lacking, or obstacles demand creative and innovative problem-solving.
</explore>

<reflection>
Reflect on the current state, task progress, objectives, and reasons for failures when the task has stalled for an extended period, incorrect actions have been taken, or the current situation has been misjudged. Analyze potential causes for errors or lack of progress, and consider alternative strategies or perspectives to overcome obstacles.
This is especially useful when several consecutive actions do not yield breakthroughs, or when persistent mistakes indicate the need for a deeper reassessment.
</reflection>

<monitor>  
Continuously track the current progress and history of reasoning and execution throughout the task.
Firstly recall the current subgoal based on the previously established overall plan, then consider the next action required to achieve this subgoal.
Typically used when task outcomes are as expected and no other mode of reasoning is required.
</monitor>

You need to output a list in JSON format, with the same length as the trajectory. Each element should contain two key-value pairs, for example:  
```json
[{{"reason": "<explore>The book may be in the cabinet, shelf, so in the next steps I need to search these locations.</explore>", "action": "go to shelf 1"}},
{{"reason": "<monitor>Currently, my sub-goal is to obtain item A. I have already spotted A, and in order to accomplish this objective, I need to pick it up.</monitor>", "action": "pick up A"}}]
```
The "action" field must match the action in the trajectory, and the "reason" field should be a reasonable reasoning process inferred from the context of previous actions and the next few actions.

now the trajectory is as follows: {traj}
"""


ALFWORLD_TEMPLATE_NO_HIS_CS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}

Now it's your turn to take an action, following these steps:

1. First, reason using ONLY ONE tag pair and express your reasoning in one concise, brief sentence:

<planning>
Plan or replan the entire task by breaking it down into high-level steps. Focus on outlining the full sequence required to complete the overall task, not just the immediate next action. 
Use this at the beginning of complex tasks or whenever the previous plan is incorrect or insufficient.
It is necessary to list all the points separately. eg, step 1: xxx, step 2: xxx, step 3: xxx, etc.
</planning>

<explore>
When results are unexpected or information is lacking, use current observations to think outside the box and list as many possible locations, items, or actions as possible.
Use this approach when facing obstacles that require creative and innovative thinking.
</explore>

<reflection>
Analyze the reasons for errors in task execution and correct them by exploring alternative approaches. 'No known action matches that input.' indicates the action is invalid.
This is typically used when several consecutive actions yield no substantial progress. 
</reflection>

<monitor>  
Continuously track the current progress and history of reasoning and execution throughout the task. Recall the current subgoal and consider the next concrete action, ensuring agent alignment with the overall plan.  
Typically used when task outcomes are as expected and no other mode of reasoning is required.
</monitor>

2. After your reasoning, you MUST select and present an admissible action for the current step within <action> </action> tags.

Specify the next action the agent should take to progress toward the task goal, following these guidelines:
1. Object and Receptacle References: Use specific identifiers:
- [obj id] for objects (e.g., apple 1).
- [recep id] for receptacles (e.g., countertop 1).
2. Action Validity: Follow the exact format below. Any deviation renders the action invalid:
Valid actions: go to [recep id], take [obj id] from [recep id], put [obj id] in/on [recep id], open/close [recep id], use [obj id], heat/cool/clean [obj id] with [recep id]

<action>Choose the most appropriate action from the valid actions.</action>
"""



ALFWORLD_TEMPLATE_CS = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observaitons and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}

Your previous overall plan is: {planning}.

Now it's your turn to take an action, following these steps:

1. First, reason using ONLY ONE tag pair and express your reasoning in one concise, brief sentence:

<planning>
Plan or replan the entire task by breaking it down into high-level steps. Focus on outlining the full sequence required to complete the overall task, not just the immediate next action. 
Use this at the beginning of complex tasks or whenever the previous plan is incorrect or insufficient.
It is necessary to list all the points separately. eg, step 1: xxx, step 2: xxx, step 3: xxx, etc.
</planning>

<explore>
When results are unexpected or information is lacking, use current observations to think outside the box and list as many possible locations, items, or actions as possible.
Use this approach when facing obstacles that require creative and innovative thinking.
</explore>

<reflection>
Analyze the reasons for errors in task execution and correct them by exploring alternative approaches. 'No known action matches that input.' indicates the action is invalid.
This is typically used when several consecutive actions yield no substantial progress. 
</reflection>

<monitor>  
Continuously track the current progress and history of reasoning and execution throughout the task. Recall the current subgoal and consider the next concrete action, ensuring agent alignment with the overall plan.  
Typically used when task outcomes are as expected and no other mode of reasoning is required.
</monitor>

2. After your reasoning, you MUST select and present an admissible action for the current step within <action> </action> tags.

Specify the next action the agent should take to progress toward the task goal, following these guidelines:
1. Object and Receptacle References: Use specific identifiers:
- [obj id] for objects (e.g., apple 1).
- [recep id] for receptacles (e.g., countertop 1).
2. Action Validity: Follow the exact format below. Any deviation renders the action invalid:
Valid actions: go to [recep id], take [obj id] from [recep id], put [obj id] in/on [recep id], open/close [recep id], use [obj id], heat/cool/clean [obj id] with [recep id]

<action>Choose the most appropriate action from the valid actions.</action>
"""




### Base Class
from typing import List, Tuple, Dict, Union, Any
import torch
import numpy as np
import os
from agent_system.environments.prompts import *
from collections import defaultdict
import time

def to_numpy(data):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        pass
    elif isinstance(data, (int, float, bool, Tuple, List)):
        data = np.array(data)
    else:
        raise ValueError(f"Unsupported type: {type(data)})")
    return data
class EnvironmentManagerBase:
    def __init__(self, envs, projection_f, config):
        """
        Initialize the environment manager.
        
        Parameters:
        - envs: The environment instance, usually a vectorized environment containing multiple sub-environments.
        - projection_f: A function that maps text actions to environment actions.
        - config: Configuration object.
        """
        self.envs = envs
        self.projection_f = projection_f
        self.config = config

    def reset(self, kwargs) -> Dict[str, Any]:
        """
        Reset all environments and return the initial observations.
        Parameters:
        - kwargs (Dict): Additional keyword arguments for resetting the environment.

        Returns:
        - next_observations (Dict):
          - 'text' (None or List[str]): The textual observation.
          - 'image' (np.ndarray or torch.Tensor): The image observation as either a NumPy array or a PyTorch tensor.
          - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        """
        obs, infos = self.envs.reset()
        return {'text': None, 'image': obs, 'anchor': None}, infos
    
    def step(self, text_actions: List[str]):
        """
        Execute text actions and return the next state, rewards, done flags, and additional information.
        
        Parameters:
        - text_actions (List[str]): A list of text actions to execute.
        
        Returns:
        - next_observations (Dict):
          - 'text' (None or List[str]): The textual observation.
          - 'image' (np.ndarray or torch.Tensor): The image observation as either a NumPy array or a PyTorch tensor.
          - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        - rewards (np.ndarry or torch.Tensor): The rewards returned by the environment.
        - dones (np.ndarray or torch.Tensor): Done flags indicating which environments have completed.
        - infos (List[Dict]): Additional environment information.
        
        Exceptions:
        - NotImplementedError: If an observation key is not in ('text', 'image').
        """
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_observations = {
            'text': None, # Implement this if needed
            'image': next_obs,
            'anchor': None # For GiGPO only. anchor observation without any histories, hint, etc. Implement this if needed
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)
        
        return next_observations, rewards, dones, infos

    def build_text_obs(self,) -> List[str]:
        """
        This function builds the text observation for the agent.
        
        Returns:
        - postprocess_text_obs (List[str]): A list of processed text observations.
        """
        pass

    def close(self) -> None:
        """
        Close the environment and release resources.
        """
        self.envs.close()

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Evaluate if the episodes are successful or not. 
        (Default) implementation is to check info['won'] of the last step.
        
        Returns:
        - success (np.ndarray or torch.Tensor): 1 if the episode is successful, 0 otherwise.
        """
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        batch_size = len(total_batch_list)
        
        success = defaultdict(list)
        
        for bs in range(batch_size):
            self._process_batch(bs, total_batch_list, total_infos, success)
        
        assert len(success['success_rate']) == batch_size

        return {key: np.array(value) for key, value in success.items()}
    
    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                return
            
    def save_image(self, image, step):
        """
        Save an image to a file.
        
        Parameters:
        - image (np.ndarray or torch.Tensor): The image to save.
        - path (str): The path to save the image.
        """
        path = os.path.join(os.path.dirname(__file__), os.path.join("images", self.config.env.env_name))
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, f"step{step}.png")
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError(f"Unsupported type: {type(image)})")
        
        if len(image.shape) == 4:
            image = image[0]
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        if image.max() <= 1.0:
            image = (image * 255)

        image = image.astype(np.uint8)
        
        from PIL import Image
        image = Image.fromarray(image)
        image.save(path)
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

import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

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

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env):
        self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        self.env.seed(seed)
    
    def step(self, action):
        """Execute a step in the environment"""
        actions = [action] 
        
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        return obs, scores, dones, infos
    
    def reset(self):
        """Reset the environment"""
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        return obs, infos
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type'] 
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv') 
        self.num_processes = env_num * group_n
        self.group_n = group_n
        
        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(config, seed + (i // self.group_n), base_env)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future) 

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        images = ray.get(futures)
        return images

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

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs) 
import os
import yaml
import tempfile 

os.environ['TMPDIR'] = '/diskpool/tmp'   
tempfile.tempdir = '/diskpool/tmp'  
def get_env_name(game_file):
    return game_file.split('json_2.1.1/train/')[-1].replace('/game.tw-pddl','') 
import os 
import yaml 
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray

from os.path import join as pjoin
import os

import textworld 
import textworld.gym 

from alfworld.info import ALFWORLD_DATA 
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredExpert, AlfredExpertType

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

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


class Env:
    def __init__(self,game_file):
        self.gamefile = game_file
        env,obs,infos = self.build_env(gamefile=game_file)
        self.env = env
        self.start_obv = obs 
        self.start_infos = infos 
        self.last_command = []
        self.auto_reset = True
        self.is_done = False
    
    def step(self,action):
        if self.is_done is True:
            obs, reward, done, infos = self.last_command[-1]

            if self.auto_reset:
                reward, done = 0., False
                obs, infos = self.reset()

        else:
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


class ParallelAlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, game_files, num_parallel, num_copied): 
        self.env_pools = {} 
        
        for parallel_idx in range(num_parallel): 
            self.env_pools[parallel_idx + 1] = Env(game_files) if num_copied == 0 else [Env(game_files) for _ in range(num_copied)] 
        
        self.start_obv = self.env_pools[1].start_obv 
        self.admissible_commands = self.env_pools[1].start_infos['admissible_commands'] 

        # self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        # self.env.seed(seed)

    def show_basis_infos(self):
        return self.start_obv,self.admissible_commands
        
    
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
    
    def reset(self):
        """Reset the environment"""

        for env in self.env_pools.values():
            obs, infos = env.reset() 
        
        return obs, infos

    

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

        # self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    
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

import re
from openai import OpenAI
from copy import deepcopy

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
    # actions = [{index+1:action} ]
    
    return {
        'think': think_content,
        'actions': actions_dict
    }

import json

def read_json(file_path):
    return json.load(open(file_path,'r'))
