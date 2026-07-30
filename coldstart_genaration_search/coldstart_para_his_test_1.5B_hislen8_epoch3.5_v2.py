import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os 
import sys
import json  
import ray
import time
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'verl-agent'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'verl-agent/agent_system/environments/env_package/webshop/webshop'))

import gym
from agent_system.environments.env_package.webshop import webshop_projection
from coldstart_genaration_webshop.prompts_webshop import system_message_para, reason_prompt_para, reason_prompt_para_his
from agent_system.environments.env_package.webshop.webshop.web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
from agent_system.environments.env_package.webshop import build_webshop_envs
print("import WebAgentTextEnv success")

remote = 0


from transformers import AutoTokenizer, AutoModelForCausalLM

# Qwen2.5 1.5B 最大上下文长度
MAX_CONTEXT_LENGTH = 32768
local_model = None
local_tokenizer = None


def load_local_model(tokenizer_path = None, 
                     model_path = None, show=1):
    global local_model, local_tokenizer
    # 当传入时重新定义 不传入时直接使用全局变量
    if model_path is not None:
        print(f"\n{'='*60}")
        print(f"Loading tokenizer and model from checkpoint: {tokenizer_path}")
        print(f" ")
        print(f"{'='*60}")
        
        print("Loading tokenizer...")
        try:
            local_tokenizer = AutoTokenizer.from_pretrained(
                # model_path,
                tokenizer_path,
                local_files_only=True
                )
            # exit(0)
            print(f"Tokenizer loaded: {local_tokenizer.__class__.__name__}")
            
            if hasattr(local_tokenizer, 'model_max_length'):
                local_tokenizer.model_max_length = MAX_CONTEXT_LENGTH
                print(f"Set tokenizer model_max_length to: {MAX_CONTEXT_LENGTH}")
            
            # 设置截断策略：从右侧截断，保留左侧（开头）信息
            local_tokenizer.truncation_side = "right"
            print(f"Set tokenizer truncation_side to: {local_tokenizer.truncation_side}")
            print(f"{'='*60}")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        if show:
            print(f"\nLoading model weights from {model_path}")
        local_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype='bfloat16',
            device_map='auto',
            attn_implementation='flash_attention_2',
            local_files_only=True,
            use_safetensors=True
        )
        
        local_model.eval()
        
        if show:
            print(f"\n{'='*60}")
            print(f"Model loaded successfully!")
            print(f"Model type: {local_model.__class__.__name__}")
            print(f"Model device: {local_model.device}")
            print(f"Model dtype: {local_model.dtype}")
            print(f"Number of parameters: {sum(p.numel() for p in local_model.parameters()):,}")
            print(f"Max context length: {MAX_CONTEXT_LENGTH}")
        
        print(f"{'='*60}")
        print("load_local_model success")
        print(f"{'='*60}")
    # exit(0)
    return local_model, local_tokenizer

def test_model_output(type=1):
    """测试模型输出，使用简单例句验证"""
    print(f"\n{'='*60}")
    print("TESTING MODEL OUTPUT")
    print(f"{'='*60}")
    
    model, tokenizer = load_local_model()
    
    if type == 1:
        test_messages = [
            {"role": "system", "content": "you are a helpful assistant."},
            {"role": "user", "content": "introduce yourself."}
        ]
        
        print(f"Input messages:")
        for msg in test_messages:
            print(f"  - {msg['role']}: {msg['content'][:50]}...")
        
        text = tokenizer.apply_chat_template(test_messages, tokenize=False)
        print(f"\nFormatted text length: {len(text)}")
        print(f"First 200 chars: {text[:200]}...")
        
        inputs = tokenizer(text, return_tensors='pt').to(model.device)
        print(f"\nInput tokens: {inputs['input_ids'].shape[1]}")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8192,
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(text):].strip()

    # 简单测试：直接输入文本
    elif type == 2:
        test_text = "introduce yourself"
        
        print(f"Input text: {test_text}")
        
        inputs = tokenizer(test_text, return_tensors='pt').to(model.device)
        print(f"Input tokens: {inputs['input_ids'].shape[1]}")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8192,
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\n{'='*60}")
    print("MODEL OUTPUT:")
    print(response)
    print(f"\n{'='*60}")
    
    return response

def local_model_infer(messages, max_new_tokens=4096, show=1):
    model, tokenizer = load_local_model()
    
    # 不使用 add_generation_prompt，确保回答不包含额外的 prompt 格式
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # 最大化输入长度，预留 max_new_tokens 给输出
    max_input_length = MAX_CONTEXT_LENGTH - max_new_tokens
    
    # 单次 tokenize：不截断，得到真实原始长度
    inputs = tokenizer(
        text, 
        return_tensors='pt',
        truncation=False,
        padding=False
    ).to(model.device)
    
    input_length = inputs['input_ids'].shape[1]
    
    if input_length > max_input_length:
        # 手动切片，保留前 max_input_length 个 token
        if show:
            print(f"WARNING: Input truncated to max length!")
            print(f"Original input tokens: {input_length}, truncating to {max_input_length}")
        inputs = {
            'input_ids': inputs['input_ids'][:, :max_input_length],
            'attention_mask': inputs['attention_mask'][:, :max_input_length]
        }
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            # 标记输出结束位置
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
            use_cache=True
            # 注意：不使用 max_length 参数，因为与 max_new_tokens 互斥
            # 输入长度已在 tokenizer 阶段通过 truncation 限制为 MAX_CONTEXT_LENGTH - max_new_tokens
            # max_length=MAX_CONTEXT_LENGTH  # 移除避免警告
        )
    
    # skip_special_tokens设置是否跳过特殊token，如 bos eos 等
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # 截断操作 确保只返回模型生成的回答，不包含输入的 prompt
    # 使用实际输入模型的文本长度来截取（处理截断情况）
    actual_input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    if response.startswith(actual_input_text):
        print(f'start with input and stripping')
        response = response[len(actual_input_text):].strip()
    else:
        print(f'not start with input')
    
    # if show:
    #     print(f"Local model inference completed")
    #     print(f"Response length: {len(response)}")
    
    return response

def deepseek(messages, ds_model=1, effort=1, show=0):
    from openai import OpenAI
    client = OpenAI(api_key="sk-d588ac9454c84f4186db19750c4c8a11", base_url="https://api.deepseek.com")

    if ds_model == 1:
        model_name = "deepseek-v4-flash"
    elif ds_model == 2:
        model_name = "deepseek-v4-pro"

    if effort == 1:
        reasoning_effort = "high"
    elif effort == 2:
        reasoning_effort = "max"

    if effort == 0:
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            extra_body={"thinking": {"type": "disabled"}},
            # extra_body={"thinking": {"type": "enabled"}},
            # reasoning_effort="high",
            messages=messages,
            stream=False,
        )
    else:
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            # extra_body={"thinking": {"type": "disabled"}},
            extra_body={"thinking": {"type": "enabled"}},
            reasoning_effort="high",
            messages=messages,
            stream=False,
        )
        

    if(show):
        if effort!=0:
            print(f'model: {model_name}, effort: {reasoning_effort}')
        else:
            print(f'model: {model_name}, disabled thinking')
        
    return response.choices[0].message.content 

def extract_think_and_action(text, use_para=0, num_para=1, total_envs=1):
    # print(f"DEBUG extract - text type: {type(text)}")
    # print(f"DEBUG extract - text: {text[:200] if isinstance(text, str) else text}")
    
    import re
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, text, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else None
    
    actions_dict = {idx: "null" for idx in range(total_envs)}
    
    env_pattern = r'<env_(\d+)>(.*?)</env_\d+>'
    matches = re.findall(env_pattern, text, re.DOTALL)
    
    for env_index, action in matches:
        env_index = int(env_index)
        if 1 <= env_index <= total_envs:
            action = action.strip()
            if action and action != 'None' and action != 'null':
                actions_dict[env_index - 1] = action
    
    return {
        'think': think_content,
        'actions': actions_dict
    }

def get_unique_filename(file_path):
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

def get_single_trajectory(env, task_idx, env_idx=0, turns=50, show_turn=False,
                          use_para=0, num_para=1, env_num=50, group_n=1, prompt=1, ds_model=1, effort=1,
                          use_local_model=True, his_len=5):
    messages = []
    success_flag = 0
    
    obs_list, info_list = env.reset()
    
    
    total_envs = group_n * env_num
    obs_parallel = obs_list[:total_envs]
    info_parallel = info_list[:total_envs]
    

    
    task_descriptions = []
    available_actions_parallel = []
    
    for info in info_parallel:
        
        if isinstance(info, dict):
            task_descriptions.append(info.get('goal', 'Find and purchase a product'))
            available_actions_parallel.append(info.get('available_actions', []))
        else:
            print(f"WARNING - info is not a dict, type: {type(info)}")
            task_descriptions.append('Find and purchase a product')
            available_actions_parallel.append([])
    
    obs_prompt_initial = ''
    # 注意环境索引1开始
    for idx in range(1, total_envs + 1):
        admissible_commands = "\n".join([f"  - {action}" for action in available_actions_parallel[idx - 1]])
        obs_prompt_initial += f'<observation_{idx}>\nThe observation and next candidated actions of {idx}-th environment are:\nObservation:\n{obs_parallel[idx - 1]}\nNext Possible Actions:\n{admissible_commands}\n</observation_{idx}>\n'
    initial_observation = obs_prompt_initial
    
    history_list = []
    for env_idx in range(total_envs):
        history_list.append({
            'Action': [],
            'Observation': [],
            'AdmissibleActions': []
        })
    
    seperated_list = []
    
    messages.append({"role": "system", "content": system_message_para.format(num_parallel=num_para, total_envs=total_envs)})
    
    status_msg = f"not defined"
    null_count = 0

    for turn in range(turns):
        
        # 注意环境索引1开始
        obs_prompt = ''
        for idx in range(1, total_envs + 1):
            admissible_commands = "\n".join([f"  - {action}" for action in available_actions_parallel[idx - 1]])
            obs_prompt += f'<observation_{idx}>\nThe observation and next candidated actions of {idx}-th environment are:\nObservation:\n{obs_parallel[idx - 1]}\nNext Possible Actions:\n{admissible_commands}\n</observation_{idx}>\n'
        
        admissible_actions = "\n".join([f"  - {action}" for env_actions in available_actions_parallel for action in env_actions])
        
        if prompt == 3:
            if turn == 0:
                message_sent = reason_prompt_para.format(
                    task_description=task_descriptions[0],
                    current_observation=obs_prompt,
                    admissible_actions=admissible_actions,
                    num_parallel=num_para,
                    total_envs=total_envs
                )
            else:
                history_start = "You have already taken multiple actions in multiple parallel environments. Below are the most recent observations and the corresponding actions you took:\n"
                
                if his_len < 0:
                    # 使用完整历史
                    history_lines = []
                    for env_idx in range(total_envs):
                        env_history = []
                        for t_idx in range(turn):
                            action = history_list[env_idx]['Action'][t_idx]
                            obs = history_list[env_idx]['Observation'][t_idx]
                            action_line = f"Action {t_idx+1}: {action}"
                            env_history.append(action_line)
                            obs_line = f"Observation {t_idx+1}: {obs}"
                            env_history.append(obs_line)
                        if env_history:
                            history_lines.append(f"In Environment {env_idx+1}\n" + "\n".join(env_history))
                    history_info = history_start + "\n\n".join(history_lines)
                else:
                    # 使用部分历史：每个环境仅保留最后his_len条，idx保持原始值
                    history_partial_lines = []
                    for env_idx in range(total_envs):
                        env_history = []
                        start_idx = max(0, turn - his_len)
                        for t_idx in range(start_idx, turn):
                            action = history_list[env_idx]['Action'][t_idx]
                            obs = history_list[env_idx]['Observation'][t_idx]
                            action_line = f"Action {t_idx+1}: {action}"
                            env_history.append(action_line)
                            obs_line = f"Observation {t_idx+1}: {obs}"
                            env_history.append(obs_line)
                        if env_history:
                            history_partial_lines.append(f"In Environment {env_idx+1}\n" + "\n".join(env_history))
                    history_info = history_start + "\n\n".join(history_partial_lines)
                
                last_history_lines = []
                for env_idx in range(total_envs):
                    action = history_list[env_idx]['Action'][turn-1]
                    obs = history_list[env_idx]['Observation'][turn-1]
                    adm_actions = history_list[env_idx]['AdmissibleActions'][turn-1]
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
        # else:
        #     message_sent = reason_prompt_para.format(
        #         task_description=task_descriptions[0],
        #         current_observation=obs_prompt,
        #         admissible_actions=admissible_actions,
        #         num_parallel=num_para,
        #         total_envs=total_envs
        #     )
        
        messages.append({"role": "user", "content": message_sent})
        
        if prompt == 3:
            # 如果使用压缩 则可丢弃上下文
            partial_messages = [
                {"task_idx": task_idx, "turn": turn},
                messages[0],  # system message
                messages[-1]  # 当前轮次 user message
            ]

            if turn == 0:
                print(f' ')

            if use_local_model:
                assistant_response = local_model_infer(messages=partial_messages[1:])
            else:
                assistant_response = deepseek(messages=partial_messages[1:], ds_model=ds_model, effort=effort)
            
            # 注意得到回答后要新增到partial_messages
            partial_messages.append({"role": "assistant", "content": assistant_response})
            seperated_list.append({"messages": partial_messages.copy()})
        # else:
        #     if use_local_model:
        #         assistant_response = local_model_infer(messages=messages)
        #     else:
        #         assistant_response = deepseek(messages=messages, ds_model=ds_model, effort=effort)
    
        result = extract_think_and_action(assistant_response, use_para=use_para, num_para=num_para, total_envs=total_envs)
       
        done_list = [False] * total_envs
        reward_list = [None] * total_envs
        
        if result.get('actions') and len(result['actions']) > 0:
            step_actions = [result['actions'].get(idx, 'null') for idx in range(total_envs)]
            
            step_result = env.step(step_actions)
            if len(step_result) == 6:
                obs_list, reward_list, done_list, info_list, rewards, action_valids = step_result
            elif len(step_result) == 5:
                obs_list, reward_list, done_list, info_list, action_valids = step_result
            else:
                obs_list, reward_list, done_list, info_list = step_result
                action_valids = True
            
            reward_list = reward_list[:total_envs]
            obs_parallel = obs_list[:total_envs]
            info_parallel = info_list[:total_envs]
            task_descriptions = [info.get('goal', 'Find and purchase a product') for info in info_parallel]
            available_actions_parallel = [info.get('available_actions', []) for info in info_parallel]
        else:
            reward_list = [None] * total_envs
            step_actions = ['null'] * total_envs
            obs_parallel = [''] * total_envs
            available_actions_parallel = [[] for _ in range(total_envs)]
        
        
        for env_idx in range(total_envs):
            action = step_actions[env_idx] if env_idx < len(step_actions) else 'null'
            obs = obs_parallel[env_idx] if env_idx < len(obs_parallel) else 'no observation'
            adm_actions = available_actions_parallel[env_idx] if env_idx < len(available_actions_parallel) else []
            history_list[env_idx]['Action'].append(action)
            history_list[env_idx]['Observation'].append(obs)
            history_list[env_idx]['AdmissibleActions'].append(adm_actions)
        
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
        
        if all_invalid:
            null_count += 1
        else:
            null_count = 0

        turn_in_range = 0
        if any_done:
            completed_idx = [i for i, done in enumerate(done_list[:total_envs]) if done]
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

def evaluate_coldstart_data(output_file, num_cpus=0.1, end_idx=100, turns=50, use_para=0, show_turn=False, 
                            load_all=0, human_goals=False, num_para=1, group_n=1, env_num=50,
                            prompt=1, limit_goals=-1, ds_model=1, effort=1, seed=42,
                            start_index=0, save_traj=1, use_local_model=True,
                            checkpoint_path=None,
                            split='test', his_len=5):
    # log_file = output_file.replace('.json', '_test.log')
    log_file = output_file.replace('.json', '.log')
    seperated_file = output_file.replace('.json', '_test_seperated.json')

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
    env_msg1 = f'Evaluating test samples (idx {start_index} to {end_idx}) with {turns} max turns each...'
    env_msg2 = f'group_n: {group_n}, env_num: {env_num}, num_para: {num_para}'
    env_msg3 = f'load_all: {load_all}, human_goals: {human_goals}, limit_goals: {limit_goals}, seed: {seed}, start_index: {start_index}, end_idx: {end_idx}'
    if use_local_model:
        env_msg4 = f'using local model (Qwen2.5-1.5/3/7B), path: {checkpoint_path}'
        env_msg5 = f'prompt: {prompt}, his_len: {his_len}, Max context length: {MAX_CONTEXT_LENGTH}'
    else:
        env_msg4 = f'using ds_model: {ds_model} ({model_name}, effort: {effort})'
        env_msg5 = f'prompt: {prompt}, his_len: {his_len}'
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
    if num_para > expected_num_para:
        print(f"num_para ({num_para}) > group_n * env_num ({group_n} * {env_num} = {expected_num_para})")
        print("exiting...")
        exit(1)
    
    env_start_time = time.time()

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
    
    
    env = build_webshop_envs(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker={'num_cpus': num_cpus},
        is_train=False,
        split=split,
        env_kwargs=env_kwargs,
        # 当前配置：env_num=1, group_n=5 → 创建5个相同环境副本，用sequential=True顺序获取测试目标
        # 每次reset获取1个新目标，5个副本都用同一个目标，正好顺序测试完test集的500个目标
        exclude=True,
        sequential=True,
    )
    
    print(f'Environment built, took {time.time() - env_start_time} seconds')
    
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
    
    test_data = []
    success_data = []
    seperated_data = []
    success_task_indices = []
    
    for i in tqdm(range(start_index, end_idx + 1), desc="Evaluating test data"):
        try:
            trajectory, success_flag, status_msg, seperated_list = get_single_trajectory(env, task_idx=i, env_idx=0, turns=turns, use_para=use_para, 
                                               show_turn=show_turn, num_para=num_para,
                                               env_num=env_num, group_n=group_n, prompt=prompt, ds_model=ds_model, 
                                               effort=effort, use_local_model=use_local_model,
                                               his_len=his_len
                                               )
            test_data.append(trajectory)
            
            with open(log_file, 'a') as f:
                f.write(status_msg + '\n')
            
            seperated_data.append(seperated_list)
            
            if success_flag == 1:
                success_data.append(trajectory)
                success_task_indices.append(i)
        except Exception as e:
            print(f"Error evaluating sample {i}: {e}")
            continue
    

    print(f"Total {len(test_data)} entries evaluated")
    
    success_count = len(success_data)
    success_rate = success_count / len(test_data) if test_data else 0
    
    if save_traj != 0:
        with open(output_file, 'w') as f:
            json.dump(test_data, f, indent=4)
        
        print(f"Test data saved to {output_file}")
        
        if success_data:
            success_output_file = output_file.replace('.json', '_success.json')
            with open(success_output_file, 'w') as f:
                json.dump(success_data, f, indent=4)
            
            print(f"Success trajectories saved to {success_output_file}")
            
        if seperated_data:
            seperated_output_file = output_file.replace('.json', '_test_seperated.json')
            with open(seperated_output_file, 'w') as f:
                json.dump(seperated_data, f, indent=4)
            
            print(f"Seperated data saved to {seperated_output_file}")
            print(f"Seperated data count: {len(seperated_data)}")
    
    print(f"Success count: {success_count}, Success rate: {success_rate:.2%}")
    
    with open(log_file, 'a') as f:
        f.write("\n")
        f.write(f"Success count: {success_count}, Success rate: {success_rate:.2%}\n")
        f.write(f"Total entries: {len(test_data)}\n")
        f.write(f"Mode: parallel\n")
        f.write(f"Number of parallel environments: {num_para}\n")
        if success_data:
            if save_traj != 0:
                f.write(f"Success trajectories saved to {success_output_file}\n")
            f.write(f"Success task indices: {success_task_indices}\n")
        else:
            print("No success trajectories generated.")
            f.write("No success trajectories generated.\n")
        
    env.close()
    
    end_time = time.time()
    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    elapsed_time = end_time - start_time
    print(f'Start time: {start_time_str}')
    print(f'End time: {end_time_str}')
    print(f'Elapsed time: {elapsed_time}s')
    
    if os.path.exists(log_file):
        with open(log_file, 'a') as f:
            f.write(f"Start time: {start_time_str}\n")
            f.write(f"End time: {end_time_str}\n")
            f.write(f"Elapsed time: {elapsed_time}s\n")
    
    return test_data

def check_model():
    print(f"\n{'='*80}")
    print("PRE-LOADING MODEL BEFORE TESTING")
    print(f"{'='*80}")
    print(f"Model path: {LOCAL_MODEL_PATH}")
    print("Loading model...")
    
    try:
        model, tokenizer = load_local_model()
        print(f"\n{'='*80}")
        print("MODEL LOADED SUCCESSFULLY!")
        print(f"{'='*80}\n")
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"FAILED TO LOAD MODEL!")
        print(f"Error: {e}")
        print(f"{'='*80}\n")
        exit(1)
    
    print(f"\n{'='*80}")
    print("RUNNING MODEL OUTPUT TEST")
    print(f"{'='*80}")
    try:
        test_model_output()
        print(f"\n{'='*80}")
        print("MODEL OUTPUT TEST PASSED!")
        print(f"{'='*80}\n")
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"MODEL OUTPUT TEST FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*80}\n")

# 原始模型路径（用于加载 tokenizer）
BASE_MODEL_PATH = '/diskpool/home/xuxz/ms-swift/model/Qwen2.5-1.5B-Instruct'
BASE_MODEL_PATH = '/diskpool/home/xuxz/ms-swift/model/Qwen2.5-3B-Instruct'

# 训练后的 checkpoint 路径（用于加载模型权重）
# CHECKPOINT_PATH = '/diskpool/home/xuxz/ms-swift/checkpoint/Qwen2.5-1.5B-Instruct-Parallel-Epoch1/v13-20260518-183416/checkpoint-1500'
# CHECKPOINT_PATH = '/diskpool/home/xuxz/ms-swift/checkpoint/Qwen2.5-7B-Instruct-Parallel-Epoch1/v2-20260518-204044/checkpoint-1500'
# CHECKPOINT_PATH = '/diskpool/home/xuxz/ms-swift/checkpoint/Qwen2.5-1.5B-Instruct-Parallel-Epoch5/v2-20260520-111355/checkpoint-7655'
CHECKPOINT_PATH = '/diskpool/home/xuxz/ms-swift/checkpoint/Qwen2.5-3B-Instruct-Parallel-Epoch5/v5-20260521-151400/checkpoint-7655'


if __name__ == "__main__":
    remote = 1

    if remote == 0:
        OUTPUT_FILE_BASE = f'/home/dpepo/verl-agent/coldstart_test/WebShop_test'  
    else:
        OUTPUT_FILE_BASE = f'/diskpool/home/xuxz/verl-agent/coldstart_test/WebShop_test'
    
    # check_model()
    
    SEEDS = [42]
    for seed in SEEDS:
        OUTPUT_FILE = f'{OUTPUT_FILE_BASE}_seed_{seed}.json'
        
        if OUTPUT_FILE and not os.path.exists(os.path.dirname(OUTPUT_FILE)):
            os.makedirs(os.path.dirname(OUTPUT_FILE))

        max_turns = 50
        HIS_LEN = 8  # 历史窗口长度，-1表示使用完整历史

        FILENAME = '1.5B_epoch3.5_hislen8_test_v2.json'
        # FOLDER = 'result'
        FOLDER = 'model_hislen8_result_v2'
        BASEPATH = os.path.join(FOLDER, FILENAME)
        os.makedirs(FOLDER, exist_ok=True)

        # CHECKPOINT_PATH = '/diskpool/home/xuxz/ms-swift/checkpoint/Qwen2.5-1.5B-Instruct-Parallel-Epoch5/v4-20260529-110357/checkpoint-6250'
        # CHECKPOINT_PATH = '/diskpool/home/xuxz/ms-swift/model/Qwen2.5-1.5B-Instruct'
        # CHECKPOINT_PATH = '/diskpool/home/xuxz/ms-swift/checkpoint/Qwen2.5-1.5B-Instruct-Parallel-Epoch5/v4-20260529-110357/checkpoint-8800'
        CHECKPOINT_PATH = '/diskpool/home/xuxz/ms-swift/checkpoint/Qwen2.5-1.5B-Instruct-Parallel-Epoch5-hislen8/v0-20260602-201729/checkpoint-6250'
        print(f"\n{'='*80}\n")
        output_file = get_unique_filename(BASEPATH)
        print(f"Output file: {output_file}")
        print(f"\n{'='*80}\n")
        # 使用全局变量存储  避免了重复加载模型
        load_local_model(tokenizer_path = CHECKPOINT_PATH, 
                         model_path = CHECKPOINT_PATH,
                         show=0
                         )
        print(f"\n{'='*80}\n")
        evaluate_coldstart_data(output_file,  turns=max_turns, 
                                num_cpus=1,
                                use_para=1, num_para=5, group_n=5, env_num=1,
                                show_turn=1,  
                                load_all=0, 
                                human_goals=False,
                                prompt=3,
                                ds_model=1, 
                                effort=0,
                                seed=seed,

                                # start_index=0,
                                start_index=0,
                                end_idx=99,
                                save_traj=1,
                                use_local_model=1,
                                checkpoint_path=CHECKPOINT_PATH,
                                split='test',
                                his_len=HIS_LEN,
                                # split='sft'
                                )

        print(f"\n{'='*80}\n")
        output_file = get_unique_filename(BASEPATH)
        print(f"Output file: {output_file}")
        print(f"\n{'='*80}\n")
        evaluate_coldstart_data(output_file,  turns=max_turns, 
                                num_cpus=1,
                                use_para=1, num_para=5, group_n=5, env_num=1,
                                show_turn=1,  
                                load_all=0, 
                                human_goals=False,
                                prompt=3,
                                ds_model=1, 
                                effort=0,
                                seed=seed,

                                start_index=100,
                                end_idx=199,
                                save_traj=1,
                                use_local_model=1,
                                checkpoint_path=CHECKPOINT_PATH,
                                split='test',
                                his_len=HIS_LEN,
                                )
        
         # gap
        print(f"\n{'='*80}\n")
        output_file = get_unique_filename(BASEPATH)
        print(f"Output file: {output_file}")
        print(f"\n{'='*80}\n")

        print(f"\n{'='*80}\n")
        evaluate_coldstart_data(output_file,  turns=max_turns,
                                num_cpus=1,
                                use_para=1, num_para=5, group_n=5, env_num=1,
                                show_turn=1,  
                                load_all=0, 
                                human_goals=False,
                                prompt=3,
                                ds_model=1, 
                                effort=0,
                                seed=seed,

                                start_index=200,
                                end_idx=499,
                                save_traj=1,
                                use_local_model=1,
                                checkpoint_path=CHECKPOINT_PATH,
                                split='test',
                                his_len=HIS_LEN,
                                # split='sft'
                                )