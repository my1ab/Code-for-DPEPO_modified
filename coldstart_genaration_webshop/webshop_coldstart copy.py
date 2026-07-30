# 开始时间




# Base Import
import os 
import sys
import json  
import ray
import time
import numpy as np
from tqdm import tqdm

# Add parent directory to sys.path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'verl-agent'))
# Add webshop directory to sys.path for web_agent_site module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'verl-agent/agent_system/environments/env_package/webshop/webshop'))

# WebShop Import
import gym
from agent_system.environments.env_package.webshop import webshop_projection
# /home/dpepo/verl-agent/agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py

# from agent_system.environments.prompts.webshop import WEBSHOP_TEMPLATE_NO_HIS as AGENT_WEBSHOP_TEMPLATE_NO_HIS
# from agent_system.environments.prompts.webshop import WEBSHOP_TEMPLATE as AGENT_WEBSHOP_TEMPLATE
from prompts_webshop import system_prompt
# 引入本地官方 prompt
from coldstart_genaration_webshop.prompt_copy.webshop import WEBSHOP_TEMPLATE_NO_HIS, WEBSHOP_TEMPLATE, system_message
from prompts_webshop import system_prompt, reason_prompt
# /home/dpepo/verl-agent/agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py
from agent_system.environments.env_package.webshop.webshop.web_agent_site.envs.web_agent_text_env \
import WebAgentTextEnv
from agent_system.environments.env_package.webshop import build_webshop_envs
print("import WebAgentTextEnv success")

def deepseek(messages):
    # sk-05267e6863d6455eb1a8c2fc92df3005
    client = OpenAI(api_key="sk-05267e6863d6455eb1a8c2fc92df3005", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False,
        temperature=1.5
    )
    
    return response.choices[0].message.content 

def test_api_connection():
    """
    Test function to verify the remote API connection works correctly.
    Sends a simple test message and prints the response.
    """
    print("Testing DeepSeek API connection...")
    
    try:
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! Please respond with a simple greeting."}
        ]
        
        response = deepseek(test_messages)
        print(f"API Response: {response}")
        print("API connection test PASSED!")
        return True
        
    except Exception as e:
        print(f"API connection test FAILED: {e}")
        print("Please check your API key and network connection.")
        return False

def validate_data_format(data):
    """
    Validate that the generated data format matches the expected format.
    """
    print("\nValidating data format...")
    
    if not isinstance(data, list):
        print("ERROR: Data is not a list")
        return False
    
    required_fields = ['prompt', 'completion', 'length']
    optional_fields = ['turn']
    
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            print(f"ERROR: Entry {i} is not a dictionary")
            return False
        
        # Check required fields
        for field in required_fields:
            if field not in entry:
                print(f"ERROR: Entry {i} missing required field: {field}")
                return False
        
        # Check field types
        if not isinstance(entry['prompt'], str):
            print(f"ERROR: Entry {i} 'prompt' is not a string")
            return False
        
        if not isinstance(entry['completion'], str):
            print(f"ERROR: Entry {i} 'completion' is not a string")
            return False
        
        if not isinstance(entry['length'], int):
            print(f"ERROR: Entry {i} 'length' is not an integer")
            return False
        
        # Check prompt format
        if not entry['prompt'].startswith('<|begin_of_text|>'):
            print(f"WARNING: Entry {i} 'prompt' doesn't start with <|begin_of_text|>")
        
        # Check completion format
        if '<think>' not in entry['completion'] or '</think>' not in entry['completion']:
            print(f"WARNING: Entry {i} 'completion' missing <think> tags")
        
        if '<action>' not in entry['completion'] or '</action>' not in entry['completion']:
            print(f"WARNING: Entry {i} 'completion' missing <action> tags")
    
    print(f"SUCCESS: All {len(data)} entries validated!")
    return True

class WebshopSingleEnv:
    """A non-Ray, single-process wrapper around WebAgentTextEnv.
    Mirrors the interface of WebshopMultiProcessEnv for compatibility.
    """
    def __init__(
        self,
        seed: int,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
        env_kwargs: dict = None,
    ) -> None:
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        
        # 种子42随机
        self._rng = np.random.RandomState(seed)
        self._env_kwargs = env_kwargs if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}
        
        # 封装WebAgentTextEnv
        self._env = WebAgentTextEnv(**self._env_kwargs)
        
        # Get goals from the environment's server
        self.goals = self._env.server.goals
        
        # 训练集从500开始
        num_goals = len(self.goals)
        # train_split = min(500, num_goals // 2)
        # if not self.is_train:
        #     self.goal_idxs = range(min(train_split, num_goals))
        # else:
        #     self.goal_idxs = range(train_split, num_goals)
        self.goal_idxs = range(num_goals)
        
        # 打印
        print(f"Loaded {len(self.goal_idxs)} goals (train={is_train})")
        
        # 顺序取索引的计数器
        self._goal_idx_counter = 0
    
    # 自定义的step  制作列表并转换接口
    def step(self, actions: list[str]):
        """Execute a step in the environment.
        Args:
            actions: List of actions (one per environment, but we only use the first)
        Returns:
            obs_list, reward_list, done_list, info_list
        """
        action = actions[0] if actions else None
        
        # 环境交互并得到reward
        # 此处用到WebAgentTextEnv的两个核心方法
        # self._env = WebAgentTextEnv(**self._env_kwargs)
        # return obs_list, reward_list, done_list, info_list, reward
        obs, reward, done, info, action_valid = self._env.step(action)
        info = dict(info or {})
        info['available_actions'] = self._env.get_available_actions()['clickables']
        info['task_score'] = reward
        

        # self.num_processes = env_num * group_n 总共商品数
        obs_list = [obs] * self.num_processes
        reward_list = [reward] * self.num_processes
        done_list = [done] * self.num_processes
        info_list = [info] * self.num_processes
        
        return obs_list, reward_list, done_list, info_list, reward, action_valid
    
    def reset(self):
        """Reset the environment with sequential goal index."""
        # idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        # 顺序取索引
        idx = [self._goal_idx_counter % len(self.goal_idxs)]
        self._goal_idx_counter += 1
        idx = np.repeat(idx, self.group_n).tolist()
        
        obs, info = self._env.reset(session=idx[0])
        # info被设为none导致没有goal  需要从环境变量获取
        info = dict(info or {})
        info['available_actions'] = self._env.get_available_actions()['clickables']
        info['won'] = False
        # 牛逼
        # baseline_models/env.py中
        # info.update({'goal': self.env.instruction_text, ...})
        info['goal'] = self._env.instruction_text
        
        obs_list = [obs] * self.num_processes
        info_list = [info] * self.num_processes
        
        return obs_list, info_list
    
    def close(self):
        """Close the environment."""
        self._env.close()
    
    def get_available_actions(self):
        """Get available actions from the environment."""
        return self._env.get_available_actions()
    
    def get_goals(self):
        """Get environment goals."""
        return self._env.server.goals
# exit(0)
# Util Functions
import re
from openai import OpenAI



def extract_think_and_action(text):
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, text, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else None
    
    action_pattern = r'<action>(.*?)</action>'
    action_match = re.search(action_pattern, text, re.DOTALL)
    action_content = action_match.group(1).strip() if action_match else None
    
    return {
        'think': think_content,
        'action': action_content
    }

# 根据任务描述+动作字符串规范动作
# search[keywords] or click[value]
def normalize_action(action, task_description):
    if not action or action == 'None':
        return None
    return action


def read_json(file_path):
    data = json.load(open(file_path,'r'))
    return data

def get_unique_filename(file_path):
    """
    Check if a file exists and generate a unique filename by appending a number if needed.
    
    Args:
        file_path: Original file path
        
    Returns:
        Unique file path
    """
    if not os.path.exists(file_path):
        return file_path
    
    dir_name = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    name, ext = os.path.splitext(file_name)
    
    counter = 1
    while True:
        new_file_name = f"{name}_{counter}{ext}"
        new_file_path = os.path.join(dir_name, new_file_name) if dir_name else new_file_name
        if not os.path.exists(new_file_path):
            return new_file_path
        counter += 1

# Main Logic - Generate coldstart data for WebShop
def get_single_trajectory(env, env_idx=0, turns=50,  show_turn=False, use_history=True,
                          use_para=0, num_para=1):
    """
    Generate a single trajectory for WebShop environment with multiple turns
    Returns messages in the format:
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]
    
    Args:
        show_turn: If True, adds turn number markers to message contents
        use_history: If True, use WEBSHOP_TEMPLATE with history, else use WEBSHOP_TEMPLATE_NO_HIS
    """
    messages = []
    
    # Reset the environment
    obs_list, info_list = env.reset()
    obs = obs_list[env_idx]
    info = info_list[env_idx]
    
    # Get task description from info or use default
    task_description = info.get('goal', 'Find and purchase a product')
    available_actions = info.get('available_actions', [])
    
    # 历史记录列表
    action_history = []
    history_length = 5  # 保留最近5步历史
    
    # 选择模板 平行与历史
    # system_message = 'You are an expert autonomous agent operating in the WebShop e-commerce environment.'
    if use_para == 0:
        if use_history:
            prompt_used = WEBSHOP_TEMPLATE
        else:
            prompt_used = WEBSHOP_TEMPLATE_NO_HIS
    elif use_para == 1:
        prompt_used = reason_prompt

    # Add system message - only once at the beginning
    
    messages.append({"role": "system", "content": system_message})
    
    for turn in range(turns):
        # Build user prompt with task description, current observation, and available actions
        if use_para == 0 and use_history:
            # 使用支持历史记录的官方模板
            history_str = "\n".join([f"Step {i+1}: Observation: {h['obs']}, Action: {h['action']}" for i, h in enumerate(action_history[-history_length:])])
            formatted_prompt = prompt_used.format(
                task_description=task_description,
                step_count=turn,
                history_length=min(len(action_history), history_length),
                action_history=history_str,
                current_step=turn + 1,
                current_observation=obs,
                available_actions="\n".join([f"  - {action}" for action in available_actions])
            )
        elif use_para == 1:
            formatted_prompt = prompt_used.format(
            
            pass
        else:
            # 使用无历史模板
            formatted_prompt = prompt_used.format(
                task_description=task_description,
                current_observation=obs,
                available_actions="\n".join([f"  - {action}" for action in available_actions])
            )
        
        # Add user message (user提出需求)
        # if show_turn:
        #     messages.append({"role": "user", "content": formatted_prompt, "turn": turn + 1})
        # else:
        #     messages.append({"role": "user", "content": formatted_prompt})
        messages.append({"role": "user", "content": formatted_prompt})
        
       
        output = deepseek(messages=messages)
        # return {
        #     'think': think_content,
        #     'action': action_content
        #       }
        result = extract_think_and_action(output, p
        
        # assistant_response即content
        if result['think'] and result['action']:
            assistant_response = f"<think>\n{result['think']}\n</think>\n\n<action>{result['action']}</action>"
        else:
            assistant_response = f"<think>\nNo reasoning available\n</think>\n\n<action>None</action>"
        
        # Execute action and get next observation
        normalized_action = None
        done_list = [False]
        
        # 有动作时规范动作
        if result['action'] and result['action'] != 'None':
            # Normalize action to match WebAgentTextEnv expected format
            normalized_action = normalize_action(result['action'], task_description)
            
            if normalized_action:
                # 转为列表并执行动作
                actions = [normalized_action]
                # env = WebshopSingleEnv
                obs_list, reward_list, done_list, info_list, reward, action_valid = env.step(actions)
                # 记录历史信息（当前observation和原始action）
                action_history.append({
                    'obs': obs,
                    'action': result['action']  # Store original action for consistency
                })
                obs = obs_list[env_idx]
                info = info_list[env_idx]
                available_actions = info.get('available_actions', [])
            else:
                reward = None
        else:
            reward = None
        
        
        
        # Add assistant message (assistant作回复)
        assistant_msg = {
            "role": "assistant",
            "content": assistant_response,
            "reward": reward,
            "original_action": result['action'] if result['action'] else None,
            "normalized_action": normalized_action,
            "executed_actions": [normalized_action] if normalized_action else None,
            "action_valid": action_valid
        }
        if show_turn:
            assistant_msg["turn"] = turn + 1
        messages.append(assistant_msg)
        
        # Check if task is done
        is_done = done_list[env_idx] if (normalized_action and len(done_list) > env_idx) else False
        if result['action'] and result['action'] != 'None' and is_done:
            print(f"Task completed at turn {turn + 1}")
            break
        elif not result['action'] or result['action'] == 'None' or normalized_action is None:
            # No valid action, end trajectory
            break
    
    return messages

def generate_coldstart_data(output_file, num_samples=500, turns=50, use_ray=False, use_para=0, show_turn=False, 
                            use_history=True,load_all = 0, human_goals=False, num_para=1):
    """
    Generate coldstart data for WebShop in JSON format.
    
    Args:
        output_file: Path to save the generated data
        num_samples: Number of samples to generate
        turns: Maximum number of turns per trajectory
        use_ray: Whether to use Ray for parallelization (default: False)
        show_turn: If True, adds turn number markers to message contents (default: False)
        use_history: If True, use WEBSHOP_TEMPLATE with history, else use WEBSHOP_TEMPLATE_NO_HIS (default: True)
    """
    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    print(f'Start time: {start_time_str}')
    print(f'Generating {num_samples} samples with {turns} turns each...')
    print(f'Turn markers enabled: {show_turn}')
    
    print(f'Building WebShop environments {"(Ray)" if use_ray else "(non-Ray)"}...')
    env_start_time = time.time()
    
    base_path = '/home/dpepo/Code-for-DPEPO-main/verl-agent/agent_system/environments/env_package/webshop/webshop/data/'
    
    if load_all == 0:
        file_path = os.path.join(base_path, 'items_shuffle_1000.json')
        attr_path = os.path.join(base_path, 'items_ins_v2_1000.json')
    else:
        file_path = os.path.join(base_path, 'items_shuffle.json')
        attr_path = os.path.join(base_path, 'items_ins_v2.json')

    print(f'file_path: {file_path} \n attr_path: {attr_path}')
    
    env_kwargs = {
        'observation_mode': 'text',
        'num_products': None,
        # 'human_goals': False,
        # 'human_goals': True,
        'human_goals': human_goals,
        'file_path': file_path,
        'attr_path': attr_path
    }

    
    # Build environment based on use_ray flag
    # 使用use_ray区分是否并行化
    if use_ray:
        # if not ray.is_initialized():
        #     ray.init()
        # 此处通过WebshopMultiProcessEnv创建并行环境
        # env = build_webshop_envs(
        #     seed=42,
        #     env_num=num_para,
        #     group_n=1,
        #     resources_per_worker={'num_cpus': 0.5},
        #     is_train=True,
        #     env_kwargs=env_kwargs
        # )
        env = WebshopSingleEnv(
            seed=42,
            env_num=1,
            group_n=1,
            is_train=True,
            env_kwargs=env_kwargs
        )
    else:
        env = WebshopSingleEnv(
            seed=42,
            env_num=1,
            group_n=1,
            is_train=True,
            env_kwargs=env_kwargs
        )
    
    print(f'Environment built, took {time.time() - env_start_time} seconds')
    
    # Generate data
    coldstart_data = []
    for i in tqdm(range(num_samples), desc="Generating coldstart data"):
        try:
            # 得到单条轨迹
            # env = WebshopSingleEnv
            trajectory = get_single_trajectory(env, env_idx=0, turns=turns, use_para=use_para, 
                                               show_turn=show_turn, num_para=num_para, use_history=use_history)
            # For non-Ray mode, each trajectory is a messages list
            coldstart_data.append(trajectory)
        except Exception as e:
            print(f"Error generating sample {i}: {e}")
            continue
    
    # Validate data format for non-Ray mode
    if not use_ray:
        print("\nValidating data format...")
        if isinstance(coldstart_data, list):
            for i, messages in enumerate(coldstart_data):
                if not isinstance(messages, list):
                    print(f"ERROR: Sample {i} is not a list of messages")
                    continue
                for j, msg in enumerate(messages):
                    if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                        print(f"ERROR: Sample {i}, Message {j} has invalid format")
            print(f"SUCCESS: All {len(coldstart_data)} samples validated!")
    
    # Get unique filename if file exists
    output_file = get_unique_filename(output_file)
    
    # Save to output file
    with open(output_file, 'w') as f:
        json.dump(coldstart_data, f, indent=4)
    
    print(f"Coldstart data saved to {output_file}")
    print(f"Total {len(coldstart_data)} entries generated")
    
    # Cleanup
    env.close()
    if use_ray and ray.is_initialized():
        ray.shutdown()
    
    end_time = time.time()
    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    elapsed_time = end_time - start_time
    print(f'Start time: {start_time_str}')
    print(f'End time: {end_time_str}')
    print(f'Elapsed time: {elapsed_time}s')
    
    return coldstart_data

if __name__ == "__main__":
    # Configuration
    OUTPUT_FILE = '/home/dpepo/Code-for-DPEPO-main/coldstart_genaration_webshop/WebShop_coldstart.json'
    
    # Generate coldstart data
    move = 2
    if move == 1:
        test_api_connection()
    elif move == 2:
        test_api_connection()
        # generate_coldstart_data(OUTPUT_FILE, num_samples=1, turns=25, use_ray=False, use_para=0,
        #                         show_turn=1, use_history=1)
        generate_coldstart_data(OUTPUT_FILE, num_samples=200, turns=50, use_ray=False, use_para=0,
                                show_turn=1, use_history=1, load_all=0, human_goals=False, 
                                num_para=1)
        # show_turn为bool型 但可以赋值为1/0 true表示所有非0数