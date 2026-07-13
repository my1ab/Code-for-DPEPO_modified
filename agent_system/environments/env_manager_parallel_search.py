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

# Basic Env Manager — Search version (uses SearchEnv instead of WebShop)
class Env:
    def __init__(self, game_file, env_kwargs=None, ground_truth=None):
        """
        Args:
            game_file: For Search env, this is the question string.
            env_kwargs: Dict with keys like 'search_url', 'topk', 'timeout', 'max_turns',
                        and optionally 'parquet_path' for int→question resolution.
            ground_truth: Optional ground truth answer for reward computation.
                          If None, falls back to game_file (question).
        """
        self.gamefile = game_file  # question text
        self.ground_truth = ground_truth if ground_truth is not None else game_file
        # Only use hardcoded defaults when env_kwargs is None (not called with empty dict {}).
        # When env_kwargs = {} (e.g. from lazy mode), each downstream .get() call in
        # build_env / reset has its own per-key defaults, so empty dict is fine.
        self.env_kwargs = env_kwargs if env_kwargs is not None else {
            'search_url': 'http://localhost:8081/search',
            'topk': 10,
            'timeout': 30,
            'log_requests': False,
            'max_turns': 10,
        }
        env, obs, infos = self.build_env(
            gamefile=game_file,
            env_kwargs=self.env_kwargs,
        )
        self.env = env
        # Search: start_obv is the question itself (used as task in rollout loop)
        self.start_obv = game_file
        self.start_infos = infos
        self.last_command = []
        self.auto_reset = False
        self.is_done = False

    def step(self, action):
        """Execute one action in the Search environment.

        SearchEnv.step returns a BaseTextEnvStepOutput (TypedDict) with:
            - observations: List[Dict[str, str]]  (OpenAI message format)
            - reward: float
            - done: bool
            - metadata: Dict
            - postprocessed_action: Optional[str]
        """
        out = self.env.step(action)

        # Extract observation text from OpenAI message format
        obs_list = out.get("observations", [])
        obs = "" if len(obs_list) == 0 else obs_list[0].get("content", "").strip()

        reward = out.get("reward", 0.0)
        done = out.get("done", False)
        info = dict(out.get("metadata", {}))

        # Possible actions for Search are always <search> and <answer>
        possible_actions = self.format_possible_actions(None)

        if done:
            self.is_done = True

        self.last_command.append(
            {
                'action': action,
                'observation': obs,
                'rewards': reward,
                'dones': done,
                'possible_commands': possible_actions,
                'game_file': self.gamefile,
            }
        )

        copied_info = dict(info)
        copied_info['admissible_commands'] = possible_actions
        copied_info['task_score'] = reward

        # Determine 'won' flag: Search environment considers reward >= 1.0 as success
        # (see envs.py _sync_step: info["won"] = bool(done and reward >= 1.0))
        won = bool(done and reward >= 1.0)
        copied_info['won'] = won

        return obs, reward, done, copied_info

    def reset(self):
        """Reset the Search environment with the same question."""
        extras = {
            "ground_truth": self.ground_truth,
            "max_turns": self.env_kwargs.get("max_turns", 10),
            "data_source": "search",
        }
        self.env.reset(extras)

        self.task = self.gamefile  # For Search, the question IS the task
        self.last_command = []
        self.is_done = False

        # Reset returns the question as observation (no [SEP] formatting needed)
        copied_infos = {}
        copied_infos['admissible_commands'] = self.format_possible_actions(None)

        return self.gamefile, copied_infos

    def build_env(self, gamefile, env_kwargs):
        """Build a SearchEnv instance.

        Args:
            gamefile: The question string.
            env_kwargs: Dict with search configuration.
        """
        from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.env import SearchEnv
        from omegaconf import DictConfig

        search_cfg = DictConfig({
            "search_url": env_kwargs.get("search_url", "http://localhost:8081/search"),
            "topk": env_kwargs.get("topk", 10),
            "timeout": env_kwargs.get("timeout", 30),
            "log_requests": env_kwargs.get("log_requests", False),
        })
        env = SearchEnv(search_cfg)

        extras = {
            "ground_truth": self.ground_truth,
            "max_turns": env_kwargs.get("max_turns", 10),
            "data_source": "search",
        }
        env.reset(extras)

        self.task = gamefile

        copied_infos = {}
        copied_infos['admissible_commands'] = self.format_possible_actions(None)

        return env, gamefile, copied_infos

    def extract_task(self, text_obs):
        """For Search, the task is the question (gamefile) itself."""
        return self.gamefile

    def format_obs(self, text_obs, add_task=True):
        """For Search, observations are plain text (search results), no [SEP] parsing needed."""
        # text_obs is already clean search result text; just return as-is
        if add_task:
            return text_obs + '\n\nYour task is to: ' + self.gamefile
        return text_obs

    def format_possible_actions(self, possible_actions):
        """Search env has two action types: <search> and <answer>.

        Unlike WebShop, there is no 'has_search_bar' or 'clickables' dict;
        possible_actions parameter is ignored.
        """
        return ['<search>query</search>', '<answer>answer</answer>']

    def render(self, mode_for_render):
        """Render the environment (not implemented for Search)."""
        pass

    def get_available_actions(self):
        """Get available actions (not applicable for Search)."""
        return ['<search>query</search>', '<answer>answer</answer>']

    def get_goals(self):
        """Get environment goals. For Search, returns the question."""
        return [self.gamefile]

    def close(self):
        """Close the environment."""
        if hasattr(self, 'env'):
            self.env.close()
    
    

# Sinlge Parallel Env Manager (Search version)
class ParallelSearchWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one Search environment instance.
    """
    
    def __init__(self, game_files, num_parallel, ground_truth=None, env_kwargs=None): 
        self.env_pools = {} 

        self.action_manager = {} 
        self.obs_manager = {}

        self.last_action_manager = {}
        self.last_obs_manager = {}
        self.last_poa_manager = {}
        
        # 从 env_kwargs 中提取 search 所需的配置
        search_env_kwargs = {}
        if env_kwargs:
            for key in ['search_url', 'topk', 'timeout', 'log_requests', 'max_turns']:
                if key in env_kwargs:
                    search_env_kwargs[key] = env_kwargs[key]

        for parallel_idx in range(num_parallel): 
            self.action_manager[parallel_idx + 1] = []
            self.obs_manager[parallel_idx + 1] = []

            self.env_pools[parallel_idx + 1] = Env(game_files, env_kwargs=search_env_kwargs, ground_truth=ground_truth)
        
        self.start_obv = self.env_pools[1].start_obv 
        self.admissible_commands = self.env_pools[1].start_infos['admissible_commands'] 

    def show_basis_infos(self):
        return self.start_obv, self.admissible_commands
    
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
            # 注意: env_pools 的 key 是 1-based (1~num_parallel), action_index 来自模型输出的 <env_1> 标签也是 1-based
            # 所以用 <= 而不是 < 来确保最后一个环境(num_parallel)也能被访问到
            if action_index in self.env_pools:
                sub_env = self.env_pools[action_index] 
                
                ob,reward,done,info = sub_env.step(action) 

                self.last_action_manager[action_index] = action 
                self.last_obs_manager[action_index] = ob
                self.last_poa_manager[action_index] = info['admissible_commands']
                
                # Search: observation prompt does NOT include "Next Possible Actions"
                # (aligned with coldstart_search_local.py style)
                obs_prompt += f'<observation_{action_index}>\n{ob}\n</observation_{action_index}>\n'

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

# Env Manager that manages the Grouped Parallel Workers (Search version)
class ParallelSearchEnvs(gym.Env):
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
        self.num_processes = group_n
        self.group_n = group_n
        
        # ---------- Resolve int → question + ground_truth via parquet ----------
        # If env_kwargs contains 'parquet_path', game_files are treated as
        # parquet row indices (int), and the question/ground_truth are loaded
        # from the parquet file. This mirrors WebShop's int→goal resolution.
        self._df = None
        self._gt_map = {}  # gamefile (int) → ground_truth
        parquet_path = env_kwargs.get('parquet_path', None)
        if parquet_path is not None:
            import pandas as pd
            self._df = pd.read_parquet(parquet_path)
            print(f"[ParallelSearchEnvs] Loaded parquet: {parquet_path} ({len(self._df)} rows)")
        
        # Create Ray remote actors instead of processes 
        env_worker = ray.remote(**resources_per_worker)(ParallelSearchWorker)
        self.workers = [] 
        self.workers_dict = {} 
        for game_file in tqdm(game_files): 
            # 类型探测: 如果 game_file 是字符串(question text)，直接使用，跳过 parquet iloc
            # 这在 lazy 模式下发生，此时 batch.non_tensor_batch['gamefile'] 已经是 question 字符串
            if isinstance(game_file, str):
                question = game_file
                gt = None
            # Resolve int→question text if parquet is available
            elif self._df is not None:
                row = self._df.iloc[game_file]
                ek = row['env_kwargs']
                if isinstance(ek, str):
                    ek = json.loads(ek)
                question = ek['question']
                gt = ek.get('ground_truth', question)
                self._gt_map[game_file] = gt
            else:
                question = str(game_file)
                gt = None
            
            self.workers_dict[game_file] = []  # key by int (parquet row index)
            for group_idx in range(self.group_n):
                worker = env_worker.remote(question, num_parallel, ground_truth=gt, env_kwargs=env_kwargs)
                self.workers.append(worker) 
                self.workers_dict[game_file].append(worker)
            print(f"group_n for {game_file}: {len(self.workers_dict[game_file])}") 

        print(f"Total unique game files: {len(self.workers_dict)}")
        
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

def build_parallel_search_envs(gamefiles,
                                 group_n, 
                                 resources_per_worker, 
                                 num_parallel,
                                 env_kwargs={}):
    return ParallelSearchEnvs(gamefiles,
                                group_n, 
                                resources_per_worker, 
                                num_parallel=num_parallel,
                                env_kwargs=env_kwargs
                                )


# ============================================================
# Test: 初始化 ParallelSearchEnvs 并输出所有环境信息到日志
# ============================================================
def test_parallel_search_envs(n=3, group_n=5):
    """
    测试函数：初始化 ParallelSearchEnvs 类（group_n=5, num_parallel=5），
    输出所有环境的问题 + 答案 + physical_id + logical_id 到一个 log 文件。

    每个 task 会创建 group_n × num_parallel = 5×5 = 25 个并行子环境。

    使用 dpepo_search 环境的默认配置:
      - parquet: ~/data/searchR1_processed_direct/train.parquet
      - JSON:    data_pipelines/gamefiles/search/search_train_tasks_excluded.json
      - search_url: http://127.0.0.1:8000/retrieve
      - topk: 3, timeout: 60
      - num_parallel: 5 (每个 worker 的并行子环境数)
      - group_n: 5     (每个 task 的 Ray worker 副本数)
      - resources_per_worker: 1 CPU
    """
    import pandas as pd
    import datetime

    # ---------- 路径配置 ----------
    parquet_path = os.path.expanduser('~/data/searchR1_processed_direct/train.parquet')
    json_path = '/diskpool/home/xuxz/Code-for-DPEPO/data_pipelines/gamefiles/search/search_train_tasks_excluded.json'
    # log_dir = os.path.join(os.path.dirname(__file__), 'case')
    log_dir = os.path.dirname(__file__)
    os.makedirs(log_dir, exist_ok=True)
    log_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'test_parallel_search_envs_{log_timestamp}.log')
    num_parallel = group_n  # 保持对称

    # ---------- 1. 加载 JSON，取前 n 个 physical_idx ----------
    task_map = read_json(json_path)
    physical_indices = [task_map[str(i)] for i in range(n)]
    print(f"[Test] 从 JSON 读取前 {n} 个 physical_idx: {physical_indices}")

    # ---------- 2. 从 parquet 预读所有数据 ----------
    df = pd.read_parquet(parquet_path)
    print(f"[Test] Parquet 总行数: {len(df)}")

    # 构建行索引→数据的查找表
    row_data = {}
    for phys_idx in physical_indices:
        row = df.iloc[phys_idx]
        ek = row['env_kwargs']
        if isinstance(ek, str):
            ek = json.loads(ek)
        q = ek['question']
        gt = ek.get('ground_truth', {})
        ds = ek.get('data_source', 'N/A')
        row_data[phys_idx] = {'question': q, 'ground_truth': gt, 'data_source': ds}

    # ---------- 3. 初始化 ParallelSearchEnvs ----------
    resources_per_worker = {"num_cpus": 1}

    print(f"[Test] 初始化 ParallelSearchEnvs(n={n}, group_n={group_n}, num_parallel={num_parallel}) ...")
    envs = ParallelSearchEnvs(
        game_files=physical_indices,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        num_parallel=num_parallel,
        env_kwargs={
            'parquet_path': parquet_path,
            'search_url': 'http://127.0.0.1:8000/retrieve',
            'topk': 3,
            'timeout': 60,
        }
    )

    # ---------- 4. 构建 grouped_samples 并调用 get_start_info_group ----------
    gourped_samples = []
    for i, phys_idx in enumerate(physical_indices):
        for g in range(group_n):
            gourped_samples.append({
                'gamefile': phys_idx,
                'group_id': g,
            })

    print(f"[Test] 调用 get_start_info_group (共 {len(gourped_samples)} 个样本)...")
    obvs, possible_actions = envs.get_start_info_group(gourped_samples)

    # ---------- 5. 写入日志文件 ----------
    log_lines = []
    log_lines.append(f"Test ParallelSearchEnvs — {log_timestamp}")
    log_lines.append(f"n={n}, group_n={group_n}, num_parallel={num_parallel}")
    log_lines.append(f"Parquet: {parquet_path}")
    log_lines.append(f"JSON:    {json_path}")
    log_lines.append(f"Physical indices: {physical_indices}")
    log_lines.append("=" * 100)

    sample_idx = 0  # gourped_samples 中的全局索引
    for i, phys_idx in enumerate(physical_indices):
        rd = row_data[phys_idx]
        q = rd['question']
        gt = rd['ground_truth']
        ds = rd['data_source']

        # ground_truth 格式转换
        if isinstance(gt, dict):
            target = gt.get('target', gt)
            if hasattr(target, '__iter__') and not isinstance(target, str):
                gt_str = '; '.join(str(t) for t in target)
            else:
                gt_str = str(target)
        else:
            gt_str = str(gt)

        log_lines.append(f"\n{'─' * 100}")
        log_lines.append(f"Task [{i}]  logical_id={i}  physical_idx={phys_idx}  data_source={ds}")
        log_lines.append(f"  question:     {q}")
        log_lines.append(f"  ground_truth: {gt_str}")
        log_lines.append(f"{'─' * 100}")

        for g in range(group_n):
            obv = obvs[sample_idx]
            log_lines.append(f"  └─ group_id={g}, worker_idx={sample_idx}")
            log_lines.append(f"       start_obs:           {obv}")
            log_lines.append(f"       admissible_commands: {possible_actions[sample_idx]}")
            log_lines.append(f"       physical_idx:        {phys_idx}")
            log_lines.append(f"       logical_id:          {i}")
            sample_idx += 1

    # 写入文件
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    print(f"[Test] 日志已写入: {log_path}")

    # ---------- 6. 清理 ----------
    print(f"[Test] 清理环境 ...")
    envs.close()
    print(f"[Test] 测试完成 ✓")
    return envs


if __name__ == '__main__':
    test_parallel_search_envs(n=3, group_n=5) 