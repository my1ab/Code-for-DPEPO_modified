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

import ray
import gym
import numpy as np
import os
import sys

# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *WebAgentTextEnv* instance.
    """
    
    def __init__(self, seed, env_kwargs):
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os
        # Get the path to this file's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Add webshop directory for WebAgentTextEnv imports (web_agent_site is in this path)
        webshop_root = os.path.join(current_dir, 'webshop')
        if webshop_root not in sys.path:
            sys.path.append(webshop_root)
        web_agent_site_path = os.path.join(webshop_root, 'web_agent_site')
        if web_agent_site_path not in sys.path:
            sys.path.append(web_agent_site_path)
        
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)
        
        env_kwargs['seed'] = seed
        # 修改 消除了环境检查器的错误
        self.env = gym.make('WebAgentTextEnv-v0', disable_env_checker=True, **env_kwargs)
    
    def step(self, action):
        """Execute a step in the environment"""
        # obs, reward, done, info = self.env.step(action)
        step_result = self.env.step(action)
        # 修改 兼容自定义的返回格式
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, done, info, *_ = step_result
        info = dict(info or {})  # make a *copy* so we can mutate safely
        info['available_actions'] = self.env.get_available_actions()
        info['task_score'] = reward

        # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
        if done and reward == 1.0:
            info['won'] = True
            reward = 10.0
        else:
            info['won'] = False
            reward = 0

        return obs, reward, done, info
    
    def reset(self, idx):
        """Reset the environment with given session index"""
        obs, info = self.env.reset(session=idx)
        info = dict(info or {})
        info['available_actions'] = self.env.get_available_actions()
        info['won'] = False
        return obs, info
    
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


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        split: str = 'train',
        env_kwargs: dict = None,
        exclude: bool = False,
        sequential: bool = False,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            # Get the path to this file's directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Add project root (parent of agent_system) to PYTHONPATH for Ray workers
            project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
            
            # Get current PYTHONPATH
            current_pythonpath = os.environ.get('PYTHONPATH', '')
            if current_pythonpath:
                pythonpath_entries = current_pythonpath.split(os.pathsep)
            else:
                pythonpath_entries = []
            
            # Add project root if not already in PYTHONPATH
            if project_root not in pythonpath_entries:
                pythonpath_entries.append(project_root)
            
            # Create runtime_env to set PYTHONPATH for all Ray workers
            runtime_env = {
                'env_vars': {
                    'PYTHONPATH': os.pathsep.join(pythonpath_entries)
                }
            }
            
            # 启动ray之前添加路径
            ray.init(runtime_env=runtime_env)

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        self.split = split
        self.exclude = exclude
        self.sequential = sequential
        # if not is_train: assert group_n == 1

        # 此处用到指定的seed42
        self._rng = np.random.RandomState(seed)

        self._env_kwargs = env_kwargs if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}

        # -------------------------- Ray actors setup --------------------------
        env_worker = ray.remote(**resources_per_worker)(WebshopWorker)
        self._workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(seed + (i // self.group_n), self._env_kwargs)
            self._workers.append(worker)

        # Get goals from the first worker
        goals_future = self._workers[0].get_goals.remote()
        goals = ray.get(goals_future)

        # 6910 total
        if split == 'test':
            self.goal_idxs = range(500)
        elif split == 'eval':
            self.goal_idxs = range(500, 1500)
        elif split == 'train':
            self.goal_idxs = range(1500, len(goals))
        elif split == 'sft':
            # self.goal_idxs = range(len(goals))
            # self.goal_idxs = range(500, len(goals))
            self.goal_idxs = range(600, len(goals))
            # self.goal_idxs = range(1500, len(goals))
            # if len(goals) >= 2500:
            #     self.goal_idxs = range(2500, len(goals))
            # else:
            #     self.goal_idxs = range(len(goals))
        else:
            self.goal_idxs = range(len(goals))

        # 以下为原始代码，此时为顺序读取
        # ------- original ----------#
        # if args.num is None:
        #     if split == 'test':
        #         self.goal_idxs = range(500)
        #     elif split == 'eval':
        #         self.goal_idxs = range(500, 1500)
        #     elif split == 'train':
        #         self.goal_idxs = range(1500, len(self.env.server.goals))
        #     elif split == 'sft':
        #         # self.goal_idxs = range(min(2500, len(goals)))
        #         self.goal_idxs = range(1500, len(self.env.server.goals))
        # else:
        #     self.goal_idxs = range(len(self.env.server.goals))

        # if not self.is_train:
        #     self.goal_idxs = range(500)
        # else:
        #     self.goal_idxs = range(500, len(goals))
            
        print(f"\n Split: {split}, Goal indices: {self.goal_idxs}, Exclude: {self.exclude}, Sequential: {self.sequential} \n")
        
        # Track used goal indices to avoid repetition (especially useful for test split)
        self._used_goal_idxs = set()
        # 顺序取索引的计数器
        self._goal_idx_counter = 0

        
    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self._workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        # Original reset logic (before fix):
        # idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        # idx = np.repeat(idx, self.group_n).tolist()
        # 
        # Fixed reset logic (handles empty goal_idxs and optionally excludes used goals):
        # 基于上文决定的范围与seed采样得到idx
        # self._rng为可复现的随机数生成器  线性放缩到指定范围
        goal_idxs_list = list(self.goal_idxs)
        
        if self.sequential:
            # 顺序不重复取索引模式
            if len(goal_idxs_list) == 0:
                raise ValueError("No goals available to sample from")
            
            # 计算本次需要取的goal数量 不看group_n
            # num_goals_needed = min(self.env_num, len(goal_idxs_list))
            num_goals_needed = self.env_num
            # 检查是否有足够的剩余goal
            if self._goal_idx_counter + num_goals_needed > len(goal_idxs_list):
                raise ValueError(f"No goals available to sample from (ran out of sequential indices: used {self._goal_idx_counter}/{len(goal_idxs_list)})")
            
            # 顺序取索引
            idx = goal_idxs_list[self._goal_idx_counter : self._goal_idx_counter + num_goals_needed]
            # 更新计数器
            self._goal_idx_counter += num_goals_needed
        elif self.exclude:
            # Exclude used goal indices
            available_goal_idxs = [g for g in goal_idxs_list if g not in self._used_goal_idxs]
            
            if len(available_goal_idxs) == 0:
                raise ValueError("No goals available to sample from (all goals have been used)")
            
            idx = self._rng.choice(available_goal_idxs, size=min(self.env_num, len(available_goal_idxs)), replace=False)
            
            # Mark these goals as used
            self._used_goal_idxs.update(idx)
        else:
            # Original behavior: allow repetition
            # Original reset logic (before adding exclude feature):
            # idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
            # idx = np.repeat(idx, self.group_n).tolist()
            # 
            # Original fixed version (handles empty goal_idxs):
            if len(goal_idxs_list) == 0:
                raise ValueError("No goals available to sample from")
            
            idx = self._rng.choice(goal_idxs_list, size=min(self.env_num, len(goal_idxs_list)), replace=False)
        
        # 按group_n重复idx  相当于环境copy
        idx = np.repeat(idx, self.group_n).tolist()

        # Send reset commands to all workers
        futures = []
        for worker, i in zip(self._workers, idx):
            future = worker.reset.remote(i)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            future = self._workers[env_idx].render.remote(mode)
            return ray.get(future)

        futures = []
        for worker in self._workers:
            future = worker.render.remote(mode)
            futures.append(future)
        
        return ray.get(futures)

    # ------------------------------------------------------------------
    # Clean-up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # Close all workers and kill Ray actors
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)
        
        # Wait for all workers to close
        ray.get(close_futures)
        
        # Kill all Ray actors
        for worker in self._workers:
            ray.kill(worker)
            
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    split: str = 'train',
    env_kwargs: dict = None,
    exclude: bool = False,
    sequential: bool = False,
):
    """Mirror *build_sokoban_envs* so higher-level code can swap seamlessly.
    
    Args:
        exclude: If True, excludes already sampled goal indices to avoid repetition.
                 Useful when testing to ensure each goal is only used once.
        sequential: If True, samples goal indices sequentially in order without repetition.
                    This takes precedence over exclude parameter.
    """
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        split=split,
        env_kwargs=env_kwargs,
        exclude=exclude,
        sequential=sequential,
    )