system_message_para2 = '''You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Given a task, you need to follow the steps: reason, choose environments, take actions.
Here are rules:
Your reasoning follow the rules required below:
Your reasoning process must be enclosed within <think> </think> tags, for example: <think> reasoning process here </think>.
You should first evaluate whether previous actions based on the action history. This evaluation consists of 3 steps:
 1) whether the environment has changed
 2) whether the expected result has been achieved
 3) check current_observation(all {total_envs} environments should be considered) then choose a group of best environments(using <env_i> </env_i> tags) and take different actions(search, click or null)
Once you've finished your reasoning, you should choose admissible actions and present them within <parallel> </parallel> tags.

You choose environments follow the rules required below:
There are {total_envs} environments available (indexed from 0 to {total_envs}-1), but you can only select up to {num_parallel} best environments to take actions each time. The actions of rest environments should be set to null.
You can explore 1 to {num_parallel} paths (indexed from 0 to {total_envs}-1), acts differently in each environment and switch between environments properly can shorten the shopping process.
To take multiple actions at the same time in different environment, use the <parallel> </parallel> tags and wrap each action within its corresponding <env_i> </env_i> tag, where i refers to the i-th environment:

**ENV FORMAT REQUIREMENT:**
<parallel>
<env_0> possible action 0 </env_0>
<env_1> possible action 1 </env_1>
...
<env_i> possible action i </env_i>
</parallel>
Where i is between 0 and {num_parallel}-1.

Your possible actions follow the rules required below:
**ACTION FORMAT REQUIREMENT:**
1. Search action: `search[keywords]`. Keywords is a space-separated list of search terms describing the product. Search keywords MUST be precise(1 to 10 words) and NOT be empty. Examples below:
  - `search[men's shorts drawstring elastic waist gym]`
  - `search[women jeans polyester spandex x-large]`
2. Click action: `click[button_text]`. Button_text MUST match exactly one of the available clickable elements. Examples below:
  - `click[next >]`
  - `click[back to search]`
  - `click[buy now]`
  - `click[B09Q5ZHRVM]`
  

You should follow the rules(about reasoning, choose environments, possible actions) above. Here are details:
**Important rules:**
1. only 3 kinds of acts are allowed: search, click, null, and Always use lowercase for action names: `search` and `click`
2. 3 kinds of tags must be within the output: <think> </think>, <parallel> </parallel>, <env_i> </env_i>
3. acts differently in each environment(try not to be same and repeat) and switch between environments properly
4. if you go in a wrong direction(take no valid actions), you can switch environment(through tags) or go back to search 
5.When all environments didn't change, check the content(clickable or not) and format(about actions and tags, from role:system in the beginning) of your output.
6.Invalid format(rules of tags and actions) and all null actions will fail your task, so check again before you finally response.
7.Buy wrong product will also fail your task, so check the original instruction again before you decide to click and buy.
8.Make sure all tags required(think, action, parallel, env_i) are within the output and in the right place.
"""
'''


reason_prompt_para2 = """You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions are: {admissible_actions}.

You should reason step by step(within tags: <think> </think>) and choose your actions(within tags: <parallel> </parallel>, <env_i> </env_i>)(only admissible actions). 
Steps and Rules required are from role:system in the beginning. You must check rules before finally output.
The most important thing: 
1.Before you click[buy now], check the product information cause you fail if you buy the wrong product.
"""

reason_prompt_para2_with_history = '''
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are: 
[
{available_actions}
].

You should reason step by step(within tags: <think> </think>) and choose your actions(within tags: <parallel> </parallel>, <env_i> </env_i>)(only admissible actions). 
Steps and Rules required are from role:system in the beginning. You must check rules before finally output.
Additional details: 
1.Before you click[buy now], check the product information cause you fail if you buy the wrong product.
'''
