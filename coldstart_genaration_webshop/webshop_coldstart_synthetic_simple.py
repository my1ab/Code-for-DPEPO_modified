import os
import sys
import json
import re
import time
import random
from tqdm import tqdm
from openai import OpenAI

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'verl-agent'))
from agent_system.environments.prompts.webshop import WEBSHOP_TEMPLATE_NO_HIS

# 简单样例
def deepseek(messages):
    client = OpenAI(api_key="sk-05267e6863d6455eb1a8c2fc92df3005", base_url="https://api.deepseek.com")
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False,
        temperature=1.5
    )
    
    return response.choices[0].message.content

def test_api_connection():
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
        return False

def extract_think_and_action(text):
    think_pattern = r'<think>(.*?)</think>'
    action_pattern = r'<action>(.*?)</action>'
    
    think_match = re.search(think_pattern, text, re.DOTALL)
    action_match = re.search(action_pattern, text, re.DOTALL)
    
    think = think_match.group(1).strip() if think_match else ""
    action = action_match.group(1).strip() if action_match else ""
    
    return think, action

def load_webshop_data(base_path):
    """Load WebShop product data and goals directly from JSON files."""
    print("Loading WebShop data...")
    
    items_file = os.path.join(base_path, 'verl-agent/agent_system/environments/env_package/webshop/webshop/data/items_shuffle_1000.json')
    attrs_file = os.path.join(base_path, 'verl-agent/agent_system/environments/env_package/webshop/webshop/data/items_ins_v2_1000.json')
    
    with open(items_file, 'r') as f:
        items_data = json.load(f)
    
    with open(attrs_file, 'r') as f:
        attrs_data = json.load(f)
    
    print(f"Loaded {len(items_data)} products and {len(attrs_data)} attribute items")
    return items_data, attrs_data

def generate_goals(items_data, num_goals=100):
    """Generate synthetic goals from product data."""
    goals = []
    
    for i, product in enumerate(items_data):
        if i >= num_goals:
            break
            
        asin = product.get('asin', product.get('ASIN', f'product_{i}'))
        title = product.get('Title', '')
        category = product.get('category', 'unknown')
        attributes = product.get('attributes', {})
        
        goal = {
            'asin': asin,
            'name': title,
            'category': category,
            'query': title.split()[0] if title else 'product',
            'instruction_text': f"Find and purchase a {category} product: {title}",
            'attributes': [k for k, v in (attributes.items() if isinstance(attributes, dict) else []) if v]
        }
        goals.append(goal)
    
    return goals

class SimpleWebShopEnv:
    """Simplified WebShop environment without Ray or WebAgentTextEnv."""
    
    def __init__(self, items_data, attrs_data):
        self.items_list = items_data
        self.items_data = {item.get('asin', item.get('ASIN', f'product_{i}')): item for i, item in enumerate(items_data)}
        self.attrs_data = attrs_data
        self.goals = generate_goals(items_data)
        self.current_goal = None
        self.current_state = 'search'  # search, results, item, purchase
        self.search_results = []
        self.current_item = None
        self.steps = 0
        
    def reset(self, goal_idx=None):
        if goal_idx is None:
            goal_idx = random.randint(0, len(self.goals) - 1)
        
        self.current_goal = self.goals[goal_idx]
        self.current_state = 'search'
        self.search_results = []
        self.current_item = None
        self.steps = 0
        
        obs = f"Search for products. Your goal is: {self.current_goal['instruction_text']}"
        info = {
            'goal': self.current_goal['instruction_text'],
            'available_actions': self._get_available_actions(),
            'won': False
        }
        
        return obs, info
    
    def _get_available_actions(self):
        if self.current_state == 'search':
            return ['search', 'finish']
        elif self.current_state == 'results':
            actions = [f'click[{asin}]' for asin in self.search_results[:10]]
            actions.append('search')
            actions.append('finish')
            return actions
        elif self.current_state == 'item':
            return ['click[description]', 'click[features]', 'click[reviews]', 'click[buy]', 'click[< prev]', 'finish']
        elif self.current_state == 'purchase':
            return ['finish']
        return ['search', 'finish']
    
    def step(self, action):
        self.steps += 1
        done = False
        reward = 0
        info = {}
        
        if action.startswith('search'):
            query = self.current_goal['query']
            self.search_results = [asin for asin in list(self.items_data.keys())[:20]]
            self.current_state = 'results'
            
            results_text = "\n".join([f"- {self.items_data[asin].get('Title', asin)}" for asin in self.search_results[:5]])
            obs = f"Search results for '{query}':\n{results_text}"
            
        elif action.startswith('click['):
            target = action[6:-1]
            
            if target in self.items_data:
                self.current_item = target
                self.current_state = 'item'
                product = self.items_data[target]
                obs = f"Product details for {product.get('Title', target)}:\nPrice: {product.get('Price', 'N/A')}\nRating: {product.get('Rating', 'N/A')}"
                
            elif target == 'buy':
                if self.current_item == self.current_goal['asin']:
                    reward = 10
                    done = True
                    obs = "Purchase successful! You found the correct product."
                else:
                    reward = 0
                    done = True
                    obs = "Purchase completed but it may not match the target product."
                self.current_state = 'purchase'
                
            elif target == '< prev':
                if self.current_state == 'item':
                    self.current_state = 'results'
                    obs = "Back to search results"
                else:
                    self.current_state = 'search'
                    obs = "Back to search page"
                    
            elif target in ['description', 'features', 'reviews']:
                obs = f"Viewing {target} for {self.items_data.get(self.current_item, {}).get('Title', self.current_item)}"
                
        elif action == 'finish':
            done = True
            obs = "Task finished"
            
        else:
            obs = "Invalid action. Available actions: " + ", ".join(self._get_available_actions())
            
        info = {
            'available_actions': self._get_available_actions(),
            'won': (reward == 10),
            'task_score': reward
        }
        
        return obs, reward, done, info

def get_single_trajectory(env, turns=50):
    trajectory_data = []
    
    obs, info = env.reset()
    task_description = info.get('goal', 'Find and purchase a product')
    
    for turn in range(turns):
        available_actions = info.get('available_actions', [])
        
        prompt = WEBSHOP_TEMPLATE_NO_HIS.format(
            task_description=task_description,
            current_observation=obs,
            available_actions="\n".join(available_actions)
        )
        
        prompt = "<|begin_of_text|>" + prompt
        
        messages = [
            {"role": "system", "content": "You are an expert autonomous agent that can interact with a web shopping environment."},
            {"role": "user", "content": prompt}
        ]
        
        response = deepseek(messages)
        think, action = extract_think_and_action(response)
        
        if not action:
            action = random.choice(available_actions) if available_actions else "finish"
        
        completion = f"<think>\n{think}\n</think>\n\n<action>{action}</action>"
        length = len(prompt) + len(completion)
        
        trajectory_data.append({
            'prompt': prompt,
            'completion': completion,
            'length': length,
            'turn': turn + 1
        })
        
        obs, reward, done, info = env.step(action)
        
        if done:
            break
    
    return trajectory_data

def validate_data_format(data):
    print("\nValidating data format...")
    
    if not isinstance(data, list):
        print("ERROR: Data is not a list")
        return False
    
    required_fields = ['prompt', 'completion', 'length']
    
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            print(f"ERROR: Entry {i} is not a dictionary")
            return False
        
        for field in required_fields:
            if field not in entry:
                print(f"ERROR: Entry {i} missing required field: {field}")
                return False
        
        if not isinstance(entry['prompt'], str):
            print(f"ERROR: Entry {i} 'prompt' is not a string")
            return False
        
        if not isinstance(entry['completion'], str):
            print(f"ERROR: Entry {i} 'completion' is not a string")
            return False
        
        if not isinstance(entry['length'], int):
            print(f"ERROR: Entry {i} 'length' is not an integer")
            return False
        
        if not entry['prompt'].startswith('<|begin_of_text|>'):
            print(f"WARNING: Entry {i} 'prompt' doesn't start with <|begin_of_text|>")
        
        if '<think>' not in entry['completion'] or '</think>' not in entry['completion']:
            print(f"WARNING: Entry {i} 'completion' missing <think> tags")
        
        if '<action>' not in entry['completion'] or '</action>' not in entry['completion']:
            print(f"WARNING: Entry {i} 'completion' missing <action> tags")
    
    print(f"SUCCESS: All {len(data)} entries validated!")
    return True

def generate_coldstart_data(output_file, num_samples=10, turns=50):
    print('Loading WebShop data...')
    base_path = '/home/dpepo/Code-for-DPEPO-main'
    
    try:
        items_data, attrs_data = load_webshop_data(base_path)
    except Exception as e:
        print(f"Failed to load WebShop data: {e}")
        return None
    
    print('Creating simplified WebShop environment...')
    # self._env = WebAgentTextEnv(**self._env_kwargs)
    env = SimpleWebShopEnv(items_data, attrs_data)
    
    print(f'Generating {num_samples} trajectories...')
    coldstart_data = []
    
    for i in tqdm(range(num_samples), desc="Generating coldstart data"):
        try:
            trajectory = get_single_trajectory(env, turns=turns)
            coldstart_data.extend(trajectory)
        except Exception as e:
            print(f"Error generating sample {i}: {e}")
            continue
    
    validate_data_format(coldstart_data)
    
    with open(output_file, 'w') as f:
        json.dump(coldstart_data, f, indent=4)
    
    print(f"Coldstart data saved to {output_file}")
    print(f"Total {len(coldstart_data)} entries generated")
    
    return coldstart_data

if __name__ == "__main__":
    OUTPUT_FILE = '/home/dpepo/Code-for-DPEPO-main/coldstart_genaration_webshop/WebShop_coldstart_simple.json'
    
    test_api_connection()
    
    generate_coldstart_data(OUTPUT_FILE, num_samples=5, turns=10)