system_prompt = '''You are an expert agent operating in the ALFRED embodied environment.
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

Your output must follow the rules below.'''


history_prompt = """You have already taken multiple actions in multiple parallel environments. Below are the most recent observaitons and the corresponding actions you took: {action_history}
"""

compressed_prompt_initial = """You are an expert agent operating in the ALFRED Embodied Environment. 
Your task is to: {task_description}
Your current observation is: {current_observation} 
Your admissible actions of the current situation are: {admissible_actions}."""

compressed_prompt_process = """You are an expert agent operating in the ALFRED Embodied Environment. 
Your task is to: {task_description}.
Your initial observation is: {initial_observation}.{history_info}
In your last step, your actions, corresponding observations and admissible actions are:\n{last_history}"""