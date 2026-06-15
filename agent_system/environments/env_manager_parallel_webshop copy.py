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
WebShop Parallel Environment Manager for verl training.

Re-exports the parallel WebShop environment builder from the top-level module
and provides a WebShopEnvironmentManager compatible with EnvironmentManagerBase.
"""

from typing import List, Dict, Any
from collections import defaultdict
import numpy as np
import os

from agent_system.env_manager_parallel_webshop import (
    Env,
    ParallelWebShopWorker,
    ParallelWebShopEnvs,
    build_parallel_webshop_envs,
)
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.environments.prompts.webshop import WEBSHOP_TEMPLATE_NO_HIS, WEBSHOP_TEMPLATE
from agent_system.memory import SimpleMemory
from agent_system.multi_turn_rollout.utils import to_list_of_dict


class WebShopEnvironmentManager(EnvironmentManagerBase):
    """
    WebShop environment manager wrapping ParallelWebShopEnvs.
    Implements the EnvironmentManagerBase interface for verl training.
    """

    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        """Reset all environments and return initial observations."""
        text_obs, infos = self.envs.reset()
        self.memory.reset(batch_size=len(text_obs))
        self.tasks = []
        self.extract_task(text_obs)
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, infos, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos

    def step(self, text_actions: List[str]):
        """Execute text actions and return next state."""
        # Convert flat text_actions to grouped action dicts
        # This is a simplified single-group step; for grouped use collector directly
        action_dicts = [{i + 1: act for i, act in enumerate(text_actions)}]
        results = self.envs.step_group(action_dicts)
        
        obs_list = []
        rewards_list = []
        dones_list = []
        infos_list = []
        
        for r in results:
            obs_list.append(r.get('observation', []))
            rewards_list.append(r.get('rewards', []))
            dones_list.append(r.get('dones', []))
            infos_list.append(r.get('possible_actions', []))

        # Flatten from grouped structure
        text_obs = [item for sublist in obs_list for item in (sublist if isinstance(sublist, list) else [sublist])]
        rewards = [item for sublist in rewards_list for item in (sublist if isinstance(sublist, list) else [sublist])]
        dones = [item for sublist in dones_list for item in (sublist if isinstance(sublist, list) else [sublist])]

        self.memory.store({'text_obs': self.pre_text_obs, 'action': text_actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, infos_list)
        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos_list

    def extract_task(self, text_obs: List[str]):
        """Extract task descriptions from observations."""
        for obs in text_obs:
            if isinstance(obs, str) and 'Your task is to: ' in obs:
                task_start = obs.find('Your task is to: ')
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                self.tasks.append('Find and purchase a product')

    def build_text_obs(self, text_obs: List[str], infos_list=None, init: bool = False) -> List[str]:
        """Build formatted text observations with history."""
        postprocess_text_obs = []
        
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                self.config.env.history_length,
                obs_key="text_obs",
                action_key="action"
            )

        for i in range(len(text_obs)):
            # Extract available actions
            if infos_list and i < len(infos_list):
                info = infos_list[i]
                if isinstance(info, dict):
                    available = info.get('admissible_commands', [])
                elif isinstance(info, list):
                    available = info
                else:
                    available = []
            else:
                available = []

            available_str = "\n".join([f"  - {a}" for a in available])

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i] if i < len(self.tasks) else 'Find and purchase a product',
                    current_observation=text_obs[i] if isinstance(text_obs[i], str) else str(text_obs[i]),
                    available_actions=available_str,
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i] if i < len(self.tasks) else 'Find and purchase a product',
                    step_count=len(self.memory[i]) if i < len(self.memory) else 0,
                    history_length=valid_lens[i] if i < len(valid_lens) else 0,
                    action_history=memory_contexts[i] if i < len(memory_contexts) else '',
                    current_step=len(self.memory[i]) + 1 if i < len(self.memory) else 1,
                    current_observation=text_obs[i] if isinstance(text_obs[i], str) else str(text_obs[i]),
                    available_actions=available_str,
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """Evaluate episode success based on reward."""
        total_infos = kwargs.get('total_infos', [])
        total_batch_list = kwargs.get('total_batch_list', [])
        batch_size = len(total_batch_list)
        success = defaultdict(list)

        for bs in range(batch_size):
            self._process_batch(bs, total_batch_list, total_infos, success)

        return {key: np.array(value) for key, value in success.items()}

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item.get('active_masks', False):
                info = total_infos[batch_idx][i] if batch_idx < len(total_infos) else {}
                reward = info.get('reward', 0) if isinstance(info, dict) else 0
                won_value = 1.0 if reward and float(reward) > 0 else 0.0
                success['success_rate'].append(won_value)
                return

    def close(self):
        """Close all environments."""
        self.envs.close()
