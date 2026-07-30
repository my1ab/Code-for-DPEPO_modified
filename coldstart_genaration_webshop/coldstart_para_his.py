# 开始时间


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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
# 引入本地官方 prompt
# from webshop import WEBSHOP_TEMPLATE_NO_HIS, WEBSHOP_TEMPLATE, system_message
# from coldstart_genaration_webshop.prompts_webshop规则版 import system_message_para, reason_prompt_para
from coldstart_genaration_webshop.prompts_webshop import system_message_para, reason_prompt_para
from prompts_webshop import reason_prompt_para_his
# /home/dpepo/verl-agent/agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py
from agent_system.environments.env_package.webshop.webshop.web_agent_site.envs.web_agent_text_env \
import WebAgentTextEnv
from agent_system.environments.env_package.webshop import build_webshop_envs
print("import WebAgentTextEnv success")

# exit(0)

remote = 0

def deepseek(messages, ds_model=1, effort=1, show=0):
    # sk-05267e6863d6455eb1a8c2fc92df3005
    # client = OpenAI(api_key="sk-05267e6863d6455eb1a8c2fc92df3005", base_url="https://api.deepseek.com")
    # sk-d588ac9454c84f4186db19750c4c8a11
    client = OpenAI(api_key="sk-d588ac9454c84f4186db19750c4c8a11", base_url="https://api.deepseek.com")

    if ds_model == 1:
        model_name = "deepseek-v4-flash"
        # print("exit")
        # exit(0)
    elif ds_model == 2:
        model_name = "deepseek-v4-pro"

    if effort == 1:
        reasoning_effort = "high"
        # print("exit")
        # exit(0)
    elif effort == 2:
        reasoning_effort = "max"

    response = client.chat.completions.create(
        # model="deepseek-chat",
        model="deepseek-v4-flash",
        # reasoning_effort="high",
        # model=model_name,
        # reasoning_effort=reasoning_effort,
        # extra_body={"thinking": {"type": "enabled"}},
        extra_body={"thinking": {"type": "disabled"}},
        messages=messages,
        stream=False,
        # temperature=1,
    )

    if(show):
        if effort!=0:
            print(f'model: {model_name}, effort: {reasoning_effort}')
        else:
            print(f'model: {model_name}, disabled thinking')
        
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
            # {"role": "user", "content": "Hello! Please respond with a simple greeting."}
            {"role": "user", "content": "We are testing whether the API connection works."}
        ]
        
        response = deepseek(test_messages)
        print(f"API Response: {response}")
        print("API connection test PASSED!")
        return True
        
    except Exception as e:
        print(f"API connection test FAILED: {e}")
        print("Please check your API key and network connection.")
        return False


# Util Functions
import re
from openai import OpenAI


def extract_think_and_action(text, use_para=0, num_para=1, total_envs=1):
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, text, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else None
    
    # 默认null  与不合规字符的处理相同（0-based索引）
    actions_dict = {idx: "null" for idx in range(total_envs)}
    
    env_pattern = r'<env_(\d+)>(.*?)</env_\d+>'
    matches = re.findall(env_pattern, text, re.DOTALL)
    
    for env_index, action in matches:
        env_index = int(env_index)
        # LLM输出1-based索引，转换为0-based存储
        if 1 <= env_index <= total_envs:
            action = action.strip()
            if action and action != 'None' and action != 'null':
                actions_dict[env_index - 1] = action
    
    return {
        'think': think_content,
        'actions': actions_dict
    }

# 根据任务描述+动作字符串规范动作
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
def get_single_trajectory(env, task_idx, env_idx=0, turns=50, show_turn=False,
                          use_para=0, num_para=1, env_num=50, group_n=1, prompt=1, ds_model=1, effort=1):
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
        use_para: If > 0, use parallel environment mode with <env_i> tags
        num_para: Number of parallel environments to select each turn
        env_num: Number of environments per group
        group_n: Number of groups
        prompt: 1=传统prompt, 3=历史压缩prompt
    """
    messages = []
    success_flag = 0
    
    # Reset the environment
    obs_list, info_list = env.reset()
    
    # Initialize observations and infos for all parallel environments
    total_envs = group_n * env_num
    obs_parallel = obs_list[:total_envs]
    info_parallel = info_list[:total_envs]
    task_descriptions = [info.get('goal', 'Find and purchase a product') for info in info_parallel]
    available_actions_parallel = [info.get('available_actions', []) for info in info_parallel]
    
    # 保存初始观察（用于历史压缩prompt）
    initial_observation = obs_prompt_initial = ''
    for idx in range(total_envs):
        admissible_commands = "\n".join([f"  - {action}" for action in available_actions_parallel[idx]])
        obs_prompt_initial += f'<observation_{idx}>\nThe observation and next candidated actions of {idx}-th environment are:\nObservation:\n{obs_parallel[idx]}\nNext Possible Actions:\n{admissible_commands}\n</observation_{idx}>\n'
    initial_observation = obs_prompt_initial
    
    # 三维历史记录列表：history_list[env_idx][key][turn]
    # 维度：环境索引 × 键值(Action/Observation/AdmissibleActions) × 轮次索引
    # history_list[env_idx]['Action'][turn_idx] = 第turn_idx轮第env_idx个环境的动作
    # history_list[env_idx]['Observation'][turn_idx] = 第turn_idx轮执行后的观察
    # history_list[env_idx]['AdmissibleActions'][turn_idx] = 第turn_idx轮执行后的可用动作
    history_list = []
    for env_idx in range(total_envs):
        history_list.append({
            'Action': [],
            'Observation': [],
            'AdmissibleActions': []
        })
    
    # 存储每轮的 partial_messages
    seperated_list = []
    
    # Add system message - only once at the beginning
    messages.append({"role": "system", "content": system_message_para.format(num_parallel=num_para, total_envs=total_envs)})
    
    status_msg = f"not defined"

    null_count = 0

    for turn in range(turns):
        # Build user prompt with task description, current observation, and available actions
        # 多环境并行模式：构建包含所有环境观察的prompt
        obs_prompt = ''
        for idx in range(total_envs):
            admissible_commands = "\n".join([f"  - {action}" for action in available_actions_parallel[idx]])
            obs_prompt += f'<observation_{idx}>\nThe observation and next candidated actions of {idx}-th environment are:\nObservation:\n{obs_parallel[idx]}\nNext Possible Actions:\n{admissible_commands}\n</observation_{idx}>\n'
        
        admissible_actions = "\n".join([f"  - {action}" for env_actions in available_actions_parallel for action in env_actions])
        
        if prompt == 3:
            if turn == 0:
                # 首轮使用传统prompt
                message_sent = reason_prompt_para.format(
                    task_description=task_descriptions[0],
                    current_observation=obs_prompt,
                    admissible_actions=admissible_actions,
                    num_parallel=num_para,
                    total_envs=total_envs
                )
            else:
                # 之后每轮使用历史压缩prompt (reason_prompt_para_his)
                # history_list 包含前 turn 轮的记录，所有历史信息直接从history_list获取
                
                # 构建历史信息（压缩摘要）：直接从history_list生成，对齐temp_history_list格式
                history_lines = []
                for env_idx in range(total_envs):
                    env_history = []
                    for t_idx in range(turn):  # 前 turn 轮（0到turn-1）
                        action = history_list[env_idx]['Action'][t_idx]
                        obs = history_list[env_idx]['Observation'][t_idx]
                        action_line = f"Action {t_idx+1}: {action}"
                        env_history.append(action_line)
                        obs_line = f"Observation {t_idx+1}: {obs}"
                        env_history.append(obs_line)
                    if env_history:
                        # 环境索引从1开始（与系统消息保持一致）
                        history_lines.append(f"In Environment {env_idx+1}\n" + "\n".join(env_history))
                history_info = "\n\n".join(history_lines)
                history_info = "You have already taken multiple actions in multiple parallel environments. Below are the most recent observations and the corresponding actions you took:\n" + history_info
                
                # 上一步历史（第 turn-1 轮）：直接从history_list获取，对齐temp_last_history格式
                last_history_lines = []
                for env_idx in range(total_envs):
                    action = history_list[env_idx]['Action'][turn-1]
                    obs = history_list[env_idx]['Observation'][turn-1]
                    adm_actions = history_list[env_idx]['AdmissibleActions'][turn-1]
                    # Action编号使用当前turn（从1开始），环境索引从1开始（与系统消息保持一致）
                    env_history_lines = [f"Action {turn}: {action}"]
                    env_history_lines.append(f"Observation {turn}: {obs}")
                    if adm_actions:
                        env_history_lines.append(f"Next Possible Actions: {', '.join(adm_actions)}")
                    last_history_lines.append(f"In Environment {env_idx+1}\n" + "\n".join(env_history_lines))
                last_history = "\n\n".join(last_history_lines)
                
                message_sent = reason_prompt_para_his.format(
                    task_description=task_descriptions[0],
                    initial_observation=initial_observation,
                    history_info=history_info,
                    last_history=last_history,
                    num_parallel=num_para,
                    total_envs=total_envs
                )
        else:
            # prompt == 1: 使用传统prompt
            message_sent = reason_prompt_para.format(
                task_description=task_descriptions[0],
                current_observation=obs_prompt,
                admissible_actions=admissible_actions,
                num_parallel=num_para,
                total_envs=total_envs
            )
        
        # Add user message (user提出需求) - 用于保存训练数据
        messages.append({"role": "user", "content": message_sent})
        
        # 如果使用压缩 则可丢弃上下文
        if prompt == 3:
            partial_messages = [
                {"task_idx": task_idx, "turn": turn},
                messages[0],  # system message
                messages[-1]  # 当前轮次 user message
            ]
            assistant_response = deepseek(messages=partial_messages[1:], ds_model=ds_model, effort=effort)
            # 注意得到回答后要新增到partial_messages
            partial_messages.append({"role": "assistant", "content": assistant_response})
            
            seperated_list.append({"messages": partial_messages.copy()})
        else:
            # 使用完整message上下文
            assistant_response = deepseek(messages=messages, ds_model=ds_model, effort=effort)
        # 模板匹配提取思考和动作
        result = extract_think_and_action(assistant_response, use_para=use_para, num_para=num_para, total_envs=total_envs)
        
        # 需要同时有思考和动作
        # 多环境并行模式
        # if result['think'] and result.get('actions'):
        #     actions_str = "\n".join([f"<env_{k}>{v}</env_{k}>" for k, v in result['actions'].items()])
        #     assistant_response = f"<think>\n{result['think']}\n</think>\n\n<parallel>\n{actions_str}\n</parallel>"
        # else:
        #     assistant_response = f"<think>\nNo reasoning available\n</think>\n\n<parallel>\n</parallel>"
        
        # Execute action and get next observation
        done_list = [False] * total_envs
        reward_list = [None] * total_envs
        
        # 多环境并行模式动作执行
        if result.get('actions') and len(result['actions']) > 0:
            # 直接从result构建动作列表（extract_think_and_action已转换为0-based）
            step_actions = [result['actions'].get(idx, 'null') for idx in range(total_envs)]
            
            # 兼容不同环境的 step 返回值
            step_result = env.step(step_actions)
            if len(step_result) == 6:
                obs_list, reward_list, done_list, info_list, rewards, action_valids = step_result
            elif len(step_result) == 5:
                obs_list, reward_list, done_list, info_list, action_valids = step_result
            else:
                obs_list, reward_list, done_list, info_list = step_result
                action_valids = True
            
            # 更新 reward_list（截断到 total_envs）
            reward_list = reward_list[:total_envs]
            
            # 更新每个环境的状态
            obs_parallel = obs_list[:total_envs]
            info_parallel = info_list[:total_envs]
            task_descriptions = [info.get('goal', 'Find and purchase a product') for info in info_parallel]
            available_actions_parallel = [info.get('available_actions', []) for info in info_parallel]
        else:
            # 没有有效动作，reward_list为 None
            reward_list = [None] * total_envs
            step_actions = ['null'] * total_envs
            obs_parallel = [''] * total_envs
            available_actions_parallel = [[] for _ in range(total_envs)]
        
        # 更新记录history_list
        # 使用三维history_list：history_list[env_idx][key][turn]
        for env_idx in range(total_envs):
            action = step_actions[env_idx] if env_idx < len(step_actions) else 'null'
            obs = obs_parallel[env_idx] if env_idx < len(obs_parallel) else 'no observation'
            adm_actions = available_actions_parallel[env_idx] if env_idx < len(available_actions_parallel) else []
            history_list[env_idx]['Action'].append(action)
            history_list[env_idx]['Observation'].append(obs)
            history_list[env_idx]['AdmissibleActions'].append(adm_actions)
        
        # Add assistant message (assistant作回复)
        # for idx in range(total_envs):
                # # 0开
                # action = result['actions'].get(idx)
                # normalized_actions.append(action)
        assistant_msg = {
            "task_idx": task_idx,
            "role": "assistant",
            "content": assistant_response,
            "rewards": json.dumps(reward_list),
            "get_actions": result.get('actions', {}),
        }
        
        if show_turn:
            assistant_msg["turn"] = turn + 1
        messages.append(assistant_msg)
        

        any_done = any(done_list[:total_envs])

        if not result.get('actions') or len(result['actions']) == 0:
            all_invalid = True
        else:
            step_actions = [result['actions'].get(idx, 'null') for idx in range(total_envs)]
            all_invalid = all(a is None or a == 'null' or a == 'None' for a in step_actions)
        # 连续两次全部非法才会退出
        if all_invalid:
            null_count += 1
        else:
            null_count = 0

        turn_in_range = 0
        if any_done:
            completed_idx = [i for i, done in enumerate(done_list[:total_envs]) if done]
            # 检查 reward_list 中只要有一个大于0即设定 success_flag
            turn_in_range = 1
            if reward_list:
                for reward in reward_list:
                    if reward is not None and reward > 0:
                        success_flag = 1
                        break
            status = "SUCCESS" if success_flag == 1 else "FAILED"
            status_msg = f"Task {task_idx} {status} at turn {turn + 1} in environments {completed_idx}"
            print(status_msg)
            break
        elif null_count >= 2:
            turn_in_range = 1
            status_msg = f"Task {task_idx} exit(all null) at turn {turn + 1}"
            print(status_msg)
            break

    if turn_in_range == 0:
        status_msg = f"Task {task_idx} out of max turn"
        print(status_msg)
    return messages, success_flag, status_msg, seperated_list

def generate_coldstart_data(output_file, num_cpus=0.1, end_idx=500, turns=50, use_ray=False, use_para=0, show_turn=False, 
                            load_all=0, human_goals=False, num_para=1, group_n=1, env_num=50,
                            only_test=0, prompt=1, limit_goals=-1, ds_model=1, effort=1, seed=42,
                            start_index=0):
    """
    Generate coldstart data for WebShop in JSON format.
    
    Args:
        output_file: Path to save the generated data
        end_idx: End index (inclusive) for data generation
        turns: Maximum number of turns per trajectory
        show_turn: If True, adds turn number markers to message contents (default: False)
        start_index: Start index for data generation (default: 0)
        ds_model: Model selection (1: flash model, 2: pro model)
        prompt: 1=传统prompt, 3=历史压缩prompt
    """
    # output_file = get_unique_filename(output_file)
    log_file = output_file.replace('.json', '.txt')
    seperated_file = output_file.replace('.json', 'seperated_success.json')

    # 当log文件存在但output文件不存在时，删除log文件并重新开始
    if os.path.exists(log_file) and not os.path.exists(output_file):
        print(f"Warning: log file exists ({log_file}) but output file does not exist.")
        print("This indicates a previous run was interrupted.")
        print("Deleting incomplete log file and starting fresh...")
        os.remove(log_file)

    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    
    
    
    print(f'Start time: {start_time_str}')
    print(f'log_file: {log_file}')
    
    if end_idx < start_index:
        print(f"Error: end_idx ({end_idx}) must be >= start_index ({start_index})")
        return
    
    model_name = "deepseek-v4-flash" if ds_model == 1 else "deepseek-v4-pro"
    env_msg1 = f'Generating samples (idx {start_index} to {end_idx}) with {turns} max turns each...'
    env_msg2 = f'group_n: {group_n}, env_num: {env_num}, num_para: {num_para}'
    env_msg3 = f'load_all: {load_all}, human_goals: {human_goals}, limit_goals: {limit_goals}, seed: {seed}, start_index: {start_index}, end_idx: {end_idx}'
    env_msg4 = f'ds_model: {ds_model} ({model_name}, effort: {effort})'
    env_msg5 = f'prompt: {prompt}'
    with open(log_file, 'w') as f:
        f.write(env_msg1 + '\n')
        f.write(env_msg2 + '\n')
        f.write(env_msg3 + '\n')
        f.write(env_msg4 + '\n')
        f.write(env_msg5 + '\n')
    print(f"\n{'='*80}\n")
    print(env_msg1), print(env_msg2), print(env_msg3), print(env_msg4), print(env_msg5)
    print(f"\n{'='*80}\n")
    
    expected_num_para = group_n * env_num
    # print(f"num_para ({num_para}) > group_n * env_num ({group_n} * {env_num} = {expected_num_para})")
    if num_para > expected_num_para:
        print(f"num_para ({num_para}) > group_n * env_num ({group_n} * {env_num} = {expected_num_para})")
        print("exiting...")
        exit(1)
    
    env_start_time = time.time()
    # Get unique filename if file exists
    
    # base_path = '/home/dpepo/verl-agent/agent_system/environments/env_package/webshop/webshop/data/'
    if remote == 0:
        base_path = '/home/dpepo/data/'
    else:
        base_path = '/diskpool/home/xuxz/data/'    

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
        'human_goals': human_goals,
        'limit_goals': limit_goals,
        'file_path': file_path,
        'attr_path': attr_path
    }
    
    # Build environment - parallel mode
    # SFT数据生成使用 'sft' split（1500-4000），与RL训练（4000-）完全隔离
    env = build_webshop_envs(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker={'num_cpus': num_cpus},
        is_train=True,
        split='sft',
        env_kwargs=env_kwargs
    )
    
    print(f'Environment built, took {time.time() - env_start_time} seconds')
    
    # 如果需要从中间开始，先跳过前 start_index 次 reset 以同步随机数状态
    if start_index > 0:
        print(f"\n{'='*80}\n")
        skip_start_time = time.time()
        print(f"Skipping first {start_index} resets to sync random state...")
        for _ in range(start_index):
            env.reset()
        skip_time = time.time() - skip_start_time
        print(f"Done skipping {start_index} resets, took {skip_time} seconds.")
        print(f"\n{'='*80}\n")
        with open(log_file, 'a') as f:
            f.write(f"Done skipping {start_index} resets, took {skip_time} seconds.\n")
    
    # Generate data
    coldstart_data = []
    success_data = []  # 存储成功的轨迹（格式与原message相同）
    seperated_data = []  # 存储每轮的 partial_messages
    success_task_indices = []  # 存储成功任务的task_idx
    
    
    if only_test == 1:
        print('only_test True')
        env.close()
        
        return

    for i in tqdm(range(start_index, end_idx + 1), desc="Generating coldstart data"):
        try:
            # 得到单条轨迹
            # env = WebshopSingleEnv
            trajectory, success_flag, status_msg, seperated_list = get_single_trajectory(env, task_idx=i, env_idx=0, turns=turns, use_para=use_para, 
                                               show_turn=show_turn, num_para=num_para,
                                               env_num=env_num, group_n=group_n, prompt=prompt, ds_model=ds_model, 
                                               effort=effort
                                               )
            # For non-Ray mode, each trajectory is a messages list
            coldstart_data.append(trajectory)
            
            # 将状态信息写入日志文件
            with open(log_file, 'a') as f:
                f.write(status_msg + '\n')
            
            # 无论成功与否，都保存 seperated_list
            seperated_data.append(seperated_list)
            
            # 如果成功，添加到成功轨迹列表
            if success_flag == 1:
                success_data.append(trajectory)
                # 记录成功的 task_idx
                success_task_indices.append(i)
        except Exception as e:
            print(f"Error generating sample {i}: {e}")
            continue
    
    
    # Save to output file
    with open(output_file, 'w') as f:
        json.dump(coldstart_data, f, indent=4)
    
    print(f"Coldstart data saved to {output_file}")
    print(f"Total {len(coldstart_data)} entries generated")
    
    # 保存成功轨迹到新的 JSON 文件（格式与原message相同）
    if success_data:
        success_output_file = output_file.replace('.json', '_success.json')
        with open(success_output_file, 'w') as f:
            json.dump(success_data, f, indent=4)
        
        success_count = len(success_data)
        success_rate = success_count / len(coldstart_data) if coldstart_data else 0
        
        print(f"Success trajectories saved to {success_output_file}")
        print(f"Success count: {success_count}, Success rate: {success_rate:.2%}")
        
        # 保存 seperated_list 到 JSON 文件
        if seperated_data:
            seperated_output_file = output_file.replace('.json', '_seperated.json')
            with open(seperated_output_file, 'w') as f:
                json.dump(seperated_data, f, indent=4)
            
            print(f"Seperated data saved to {seperated_output_file}")
            print(f"Seperated data count: {len(seperated_data)}")
        
        # 追加写入日志文件（不覆盖之前的状态信息）
        # w则覆盖 a则添加
        with open(log_file, 'a') as f:
            f.write("\n")
            f.write(f"Success trajectories saved to {success_output_file}\n")
            f.write(f"Success count: {success_count}, Success rate: {success_rate:.2%}\n")
            f.write(f"Total entries: {len(coldstart_data)}\n")
            f.write(f"Mode: parallel\n")
            f.write(f"Number of parallel environments: {num_para}\n")
            f.write(f"Success task indices: {success_task_indices}\n")
    else:
        print("No success trajectories generated.")
        # 追加写入日志文件（不覆盖之前的状态信息）
        with open(log_file, 'a') as f:
            f.write("\n")
            f.write("No success trajectories generated.\n")
        
    # Cleanup
    env.close()
    
    end_time = time.time()
    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    elapsed_time = end_time - start_time
    print(f'Start time: {start_time_str}')
    print(f'End time: {end_time_str}')
    print(f'Elapsed time: {elapsed_time}s')
    
    # 将用时信息写入日志文件
    if os.path.exists(log_file):
        with open(log_file, 'a') as f:
            f.write(f"Start time: {start_time_str}\n")
            f.write(f"End time: {end_time_str}\n")
            f.write(f"Elapsed time: {elapsed_time}s\n")
    
    return coldstart_data

if __name__ == "__main__":
    # Configuration
    # /home/dpepo/verl-agent/coldstart_genaration_webshop/webshop_coldstart.py
    remote = 1

    if remote == 0:
        OUTPUT_FILE_BASE = f'/home/dpepo/verl-agent/coldstart_result_webshop/WebShop_coldstart'  
    else:
        OUTPUT_FILE_BASE = f'/diskpool/home/xuxz/verl-agent/coldstart_result_webshop/WebShop_coldstart'
        
    
    
    # Generate coldstart data for each seed in the list
    move = 2
    if move == 1:
        test_api_connection()
    elif move == 2:
        test_api_connection()
        # 手动输入的 seed 列表
        # SEEDS = [42, 100, 200, 300]  # 可以根据需要修改
        # SEEDS = [42, 100] 
        SEEDS = [42]
        for seed in SEEDS:
            # 为每个 seed 生成唯一的输出文件名
            OUTPUT_FILE = f'{OUTPUT_FILE_BASE}_seed_{seed}.json'
            
            if OUTPUT_FILE and not os.path.exists(os.path.dirname(OUTPUT_FILE)):
                os.makedirs(os.path.dirname(OUTPUT_FILE))

            max_turns = 25

            # print(f"\n{'='*80}\n")
            # print(f"Generating data with seed={seed}")
            # output_file = get_unique_filename(OUTPUT_FILE)
            # print(f"Output file: {output_file}")
            # print(f"\n{'='*80}\n")

            # generate_coldstart_data(output_file,  turns=max_turns, 
            #                         num_cpus=1,
            #                         use_para=1, num_para=5, group_n=1, env_num=5,
            #                         show_turn=1,  
            #                         load_all=0, 
            #                         human_goals=False,
            #                         prompt=3,

            #                         ds_model=1, 
            #                         effort=1,
            #                         seed=seed,
            #                         start_index=0,
            #                         end_idx=99,
            #                         only_test=0
            #                         )
            
            # print(f"\n{'='*80}\n")
            # print(f"Generating data with seed={seed}")
            # output_file = get_unique_filename(OUTPUT_FILE)
            # print(f"Output file: {output_file}")
            # print(f"\n{'='*80}\n")
            
            # generate_coldstart_data(output_file,  turns=max_turns, 
            #                         num_cpus=1,
            #                         use_para=1, num_para=5, group_n=1, env_num=5,
            #                         show_turn=1,  
            #                         load_all=0, 
            #                         human_goals=False,
            #                         prompt=3,

            #                         ds_model=1, 
            #                         effort=1,
            #                         seed=seed,
            #                         start_index=100,
            #                         end_idx=199,
            #                         only_test=0
            #                         )
            
            # print(f"\n{'='*80}\n")
            # print(f"Generating data with seed={seed}")
            # output_file = get_unique_filename(OUTPUT_FILE)
            # print(f"Output file: {output_file}")
            # print(f"\n{'='*80}\n")
            
            # generate_coldstart_data(output_file,  turns=max_turns, 
            #                         num_cpus=1,
            #                         use_para=1, num_para=5, group_n=1, env_num=5,
            #                         show_turn=1,  
            #                         load_all=0, 
            #                         human_goals=False,
            #                         prompt=3,

            #                         ds_model=1, 
            #                         effort=1,
            #                         seed=seed,
                                    
            #                         start_index=200,
            #                         end_idx=299,
            #                         only_test=0
            #                         )
            
            # print(f"\n{'='*80}\n")
            # print(f"Generating data with seed={seed}")
            # output_file = get_unique_filename(OUTPUT_FILE)
            # print(f"Output file: {output_file}")
            # print(f"\n{'='*80}\n")
            
            # generate_coldstart_data(output_file,  turns=max_turns, 
            #                         num_cpus=1,
            #                         use_para=1, num_para=5, group_n=1, env_num=5,
            #                         show_turn=1,  
            #                         load_all=0, 
            #                         human_goals=False,
            #                         prompt=3,

            #                         ds_model=1, 
            #                         effort=1,
            #                         seed=seed,
                                    
            #                         start_index=300,
            #                         end_idx=399,
            #                         only_test=0
            #                         )
            
            # print(f"\n{'='*80}\n")
            # print(f"Generating data with seed={seed}")
            # output_file = get_unique_filename(OUTPUT_FILE)
            # print(f"Output file: {output_file}")
            # print(f"\n{'='*80}\n")
            
            # generate_coldstart_data(output_file,  turns=max_turns, 
            #                         num_cpus=1,
            #                         use_para=1, num_para=5, group_n=1, env_num=5,
            #                         show_turn=1,  
            #                         load_all=0, 
            #                         human_goals=False,
            #                         prompt=3,

            #                         ds_model=1, 
            #                         effort=1,
            #                         seed=seed,
                                    
            #                         start_index=400,
            #                         end_idx=499,
            #                         only_test=0
            #                         )
            
            # print(f"Generating data with seed={seed}")
            # output_file = get_unique_filename(OUTPUT_FILE)
            # print(f"Output file: {output_file}")
            # print(f"\n{'='*80}\n")
            
            # generate_coldstart_data(output_file,  turns=max_turns, 
            #                         num_cpus=1,
            #                         use_para=1, num_para=5, group_n=1, env_num=5,
            #                         show_turn=1,  
            #                         load_all=0, 
            #                         human_goals=False,
            #                         prompt=3,

            #                         ds_model=1, 
            #                         effort=1,
            #                         seed=seed,
                                    
            #                         start_index=500,
            #                         end_idx=599,
            #                         only_test=0
            #                         )
            
            # print(f"Generating data with seed={seed}")
            # output_file = get_unique_filename(OUTPUT_FILE)
            # print(f"Output file: {output_file}")
            # print(f"\n{'='*80}\n")
            
            # generate_coldstart_data(output_file,  turns=max_turns, 
            #                         num_cpus=1,
            #                         use_para=1, num_para=5, group_n=1, env_num=5,
            #                         show_turn=1,  
            #                         load_all=0, 
            #                         human_goals=False,
            #                         prompt=3,

            #                         ds_model=1, 
            #                         effort=1,
            #                         seed=seed,
                                    
            #                         start_index=600,
            #                         end_idx=699,
            #                         only_test=0
            #                         )
            
            # print(f"Generating data with seed={seed}")
            # output_file = get_unique_filename(OUTPUT_FILE)
            # print(f"Output file: {output_file}")
            # print(f"\n{'='*80}\n")
            
            # generate_coldstart_data(output_file,  turns=max_turns, 
            #                         num_cpus=1,
            #                         use_para=1, num_para=5, group_n=1, env_num=5,
            #                         show_turn=1,  
            #                         load_all=0, 
            #                         human_goals=False,
            #                         prompt=3,

            #                         ds_model=1, 
            #                         effort=1,
            #                         seed=seed,
                                    
            #                         start_index=700,
            #                         end_idx=799,
            #                         only_test=0
            #                         )
            
            # print(f"Generating data with seed={seed}")
            # output_file = get_unique_filename(OUTPUT_FILE)
            # print(f"Output file: {output_file}")
            # print(f"\n{'='*80}\n")
            
            # generate_coldstart_data(output_file,  turns=max_turns, 
            #                         num_cpus=1,
            #                         use_para=1, num_para=5, group_n=1, env_num=5,
            #                         show_turn=1,  
            #                         load_all=0, 
            #                         human_goals=False,
            #                         prompt=3,

            #                         ds_model=1, 
            #                         effort=1,
            #                         seed=seed,
                                    
            #                         start_index=800,
            #                         end_idx=899,
            #                         only_test=0
            #                         )

            print(f"Generating data with seed={seed}")
            output_file = get_unique_filename(OUTPUT_FILE)
            print(f"Output file: {output_file}")
            print(f"\n{'='*80}\n")
            
            generate_coldstart_data(output_file,  turns=max_turns, 
                                    num_cpus=1,
                                    use_para=1, num_para=5, group_n=1, env_num=5,
                                    show_turn=1,  
                                    load_all=0, 
                                    human_goals=False,
                                    prompt=3,

                                    ds_model=1, 
                                    effort=1,
                                    seed=seed,
                                    
                                    # start_index=900,
                                    # end_idx=999,
                                    start_index=1000,
                                    end_idx=1099,
                                    only_test=0
                                    )
            
            
            
            print(f"\n{'='*80}\n")
            print(f"Finished generating data with seed={seed}")
            print(f"\n{'='*80}\n")
        
        # show_turn为bool型 但可以赋值为1/0 true表示所有非0数