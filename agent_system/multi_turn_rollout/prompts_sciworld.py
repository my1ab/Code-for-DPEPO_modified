system_prompt = '''You are an expert agent operating in the ScienceWorld environment, which is a text-based virtual environment centered around accomplishing tasks from the elementary science curriculum.
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
<env_2> possible action 2 </env_2>
...
<env_i> possible action i </env_i>
</parallel>

Your output must follow the rules above.
'''


history_prompt = """You have already taken multiple actions in multiple parallel environments. Below are the most recent observaitons and the corresponding actions you took: {action_history}
"""

action_explanation = """Here the explanations of actions you may take:
[
{{"action": "open OBJ", "description": "open a container"}},
{{"action": "close OBJ", "description": "close a container"}},
{{"action": "activate OBJ", "description": "activate a device"}},
{{"action": "deactivate OBJ", "description": "deactivate a device"}},
{{"action": "connect OBJ to OBJ", "description": "connect electrical components"}},
{{"action": "disconnect OBJ", "description": "disconnect electrical components"}},
{{"action": "use OBJ [on OBJ]", "description": "use a device/item"}},
{{"action": "look around", "description": "describe the current room"}},
{{"action": "look at OBJ", "description": "describe an object in detail"}},
{{"action": "look in OBJ", "description": "describe a container's contents"}},
{{"action": "read OBJ", "description": "read a note or book"}},
{{"action": "move OBJ to OBJ", "description": "move an object to a container"}},
{{"action": "pick up OBJ", "description": "move an object to the inventory"}},
{{"action": "put down OBJ", "description": "drop an inventory item"}},
{{"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"}},
{{"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"}},
{{"action": "mix OBJ", "description": "chemically mix a container"}},
{{"action": "go to LOC", "description": "move to a new location"}},
{{"action": "eat OBJ", "description": "eat a food"}},
{{"action": "flush OBJ", "description": "flush a toilet"}},
{{"action": "focus on OBJ", "description": "signal intent on a task object"}},
{{"action": "wait", "description": "take no action for 10 iterations"}},
{{"action": "wait1", "description": "take no action for 1 iteration"}},
{{"action": "task", "description": "describe current task"}},
{{"action": "inventory", "description": "list your inventory"}}
]
"""

compressed_prompt_initial = """You are an expert agent operating in the ScienceWorld environment, which is a text-based virtual environment centered around accomplishing tasks from the elementary science curriculum.
Your task is to: {task_description}
{action_explanation}
Your current observation is: {current_observation} 
Your admissible actions of the current situation are: {admissible_actions}."""

compressed_prompt_process = """You are an expert agent operating in the ScienceWorld environment, which is a text-based virtual environment centered around accomplishing tasks from the elementary science curriculum.
Your task is to: {task_description}.
{action_explanation}
Your initial observation is: {initial_observation}.{history_info}
In your last step, your actions, corresponding observations and admissible actions are:\n{last_history}"""