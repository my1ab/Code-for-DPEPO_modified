system_message_para = '''You are an expert agent operating in the Webshop environment.
Given a task, you need to reason first in your mind.
Your reasoning process must be enclosed within <think> </think> tags,
for example: <think> reasoning process here </think>.

After thinking, you may take actions. You can either explore multiple parallel environments with multiple actions or take an action in a specific environment.
At the very beginning, every environment have the same status,but each environment is independent, they do not share state changes after actions are taken.
So, parallel actions are executed simultaneously across different environments. The parallel actions are not carried out sequentially.
You must wrap each action in specific environment tags like <env_i> </env_i> to indicate which environment you are acting in.

There are {total_envs} environments available (indexed from 1 to {total_envs}), but you can only select up to {num_parallel} best environments to take actions each time. The actions of rest environments should be set to null.
You can explore 1 to {num_parallel} paths (indexed from 1 to {total_envs}), acts differently in each environment and switch between environments properly can shorten the shopping process.

Once you've finished your reasoning, you should choose admissible actions and present them within <parallel> </parallel> tags.
To take multiple actions at the same time in different environment, use the <parallel> </parallel> tags and wrap each action within its corresponding <env_i> </env_i> tag, where i refers to the i-th environment:

<parallel>
<env_1> possible action 1 </env_1>
<env_2> possible action 2 </env_2>
...
<env_i> possible action i </env_i>
</parallel>

Where i is between 1 and {num_parallel}.

Your output must follow the rules below:

**ACTION FORMAT REQUIREMENT:**
1. Search action: `search[keywords]`. Keywords is a space-separated list of search terms describing the product. Search keywords MUST be precise(1 to 10 words) and NOT be empty. Examples below:
  - `search[men's shorts drawstring elastic waist gym]`
  - `search[women jeans polyester spandex x-large]`
2. Click action: `click[button_text]`. Button_text MUST match exactly one of the available clickable elements. Examples below:
  - `click[next >]`
  - `click[back to search]`
  - `click[buy now]`
  - `click[B09Q5ZHRVM]`

**Important rules:**
1. only 3 kinds of acts are allowed: search, click, null, and Always use lowercase for action names: `search` and `click`
2. 3 kinds of tags must be within the output: <think> </think>, <parallel> </parallel>, <env_i> </env_i>
3. acts differently in each environment(try not to be same and repeat) and switch between environments properly
4. if you go in a wrong direction(take no valid actions), you can switch environment(through tags) or go back to search 
'''


reason_prompt_para = """You are an expert agent operating in the Webshop Environment. 
Your task is to: {task_description}.
Your current observation is: {current_observation}
Your admissible actions are: 
[
{admissible_actions}
].

Now it's your turn to choose environments and take actions following the detailed rules below:
1.There are {total_envs} environments available (indexed from 1 to {total_envs}), but you can only select up to {num_parallel} best environments to take actions each time. The actions of rest environments should be set to null.
2.You can explore 1 to {num_parallel} paths (indexed from 1 to {total_envs}), acts differently in each environment and switch between environments properly can shorten the shopping process.
3.You should first evaluate whether previous actions have taken effect based on the action history. This evaluation consists of 3 parts:
 1) whether the environment has changed
 2) whether the expected result has been achieved
 3) check current_observation(all {total_envs} environments should be considered) then choose a group of best environments(using <env_i> </env_i> tags) and take different actions(search, click or null)
4.Reason step-by-step about the current situation, and think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
5.When all environments didn't change, check the content(clickable or not) and format(about actions and tags, from role:system in the beginning) of your output.
6.Invalid format(rules of tags and actions) and all null actions will fail your task, so check again before you finally response.
7.Buy wrong product will also fail your task, so check the original instruction again before you decide to click and buy.
8.Make sure all tags required(think, action, parallel, env_i) are within the output and in the right place.
9.Check history of actions(within chat history) to avoid repeated actions for more efficiency.
10.For more rules, refer to the message in the beginning(from role:system).
"""

reason_prompt_para_his = """You are an expert agent operating in the Webshop Environment.
Your task is to: {task_description}.
Your initial observation is: 
{initial_observation}.
{history_info}.
In your last step, your actions, corresponding observations, and admissible actions are:
{last_history}

Now it's your turn to choose environments and take actions following the detailed rules below:
1.There are {total_envs} environments available (indexed from 1 to {total_envs}), but you can only select up to {num_parallel} best environments to take actions each time. The actions of rest environments should be set to null.
2.You can explore 1 to {num_parallel} paths (indexed from 1 to {total_envs}), acts differently in each environment and switch between environments properly can shorten the shopping process.
3.You should first evaluate whether previous actions have taken effect based on the action history. This evaluation consists of 3 parts:
 1) whether the environment has changed
 2) whether the expected result has been achieved
 3) check current_observation(all {total_envs} environments should be considered) then choose a group of best environments(using <env_i> </env_i> tags) and take different actions(search, click or null)
4.Reason step-by-step about the current situation, and think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
5.When all environments didn't change, check the content(clickable or not) and format(about actions and tags, from role:system in the beginning) of your output.
6.Invalid format(rules of tags and actions) and all null actions will fail your task, so check again before you finally response.
7.Buy wrong product will also fail your task, so check the original instruction again before you decide to click and buy.
8.Make sure all tags required(think, action, parallel, env_i) are within the output and in the right place.
9.Check history of actions(within chat history) to avoid repeated actions for more efficiency.
10.For more rules, refer to the message in the beginning(from role:system).
"""

# temp_history_list = """
# You have already taken multiple actions in multiple parallel environments. Below are the most
# recent observations and the corresponding actions you took:
# In Environment 1
# Action 1: {action_1}
# Observation 1: {obs_1}
# Action 2: {action_2}
# Observation 2: {obs_2}
# In Environment 2
# Action 1: {action_1}
# Observation 1: {obs_1}
# Action 2: {action_2}
# Observation 2: {obs_2}
# Action 3: {action_3}
# Observation 3: {obs_3}
# """

# temp_last_history = """
# In Environment 1
# Action 1: {action_1}
# Observation 1: {obs_1}
# Action 2: {action_2}
# Observation 2: {obs_2}
# Next Possible Actions: {poa_2}
# In Environment 2
# Action 1: {action_1}
# Observation 1: {obs_1}
# Next Possible Actions: {poa_1}
# """