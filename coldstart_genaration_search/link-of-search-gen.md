# Search Coldstart Data Generation — 函数/文件调用链路

> **概览：文件流**
> ```
> 数据准备                                         冷启动生成                                   保存结果
> ┌──────────────────────┐    ┌───────────────────────────────────────┐    ┌──────────────────────┐
> │ raw parquet          │    │ coldstart_search.py                   │    │ *.json               │
> │ (HuggingFace)        │───→│   ├── prompts_search.py (prompts)     │───→│   ├─ 完整轨迹          │
> │         ↓            │    │   ├── build_search_envs (环境)         │    │   ├─ _success.json    │
> │ processed parquet    │    │   │     └── SearchEnv + SearchTool    │    │   └─ _seperated.json  │
> │ (train / test)       │    │   └── Model (local Qwen / DeepSeek)  │    │                      │
> └──────────────────────┘    └───────────────────────────────────────┘    └──────────────────────┘
> ```
>
> **概览：函数调用流（单条轨迹）**   `[文件路径]`
> ```
> evaluate_coldstart_data()                                                       [coldstart_search.py]
>   ├── load_search_data()          → pd.read_parquet → df                        [coldstart_search.py]
>   ├── build_search_envs()         → SearchMultiProcessEnv                       [search/envs.py]
>   └── for each row:
>         └── get_single_trajectory()                                             [coldstart_search.py]
>               ├── env.reset(kwargs)                                             [search/envs.py → SearchMultiProcessEnv]
>               │     └── SearchEnv.reset(extras)  → 初始化 ground_truth, history [search/env.py]
>               │
>               └── for turn in range(max_turns):
>                     ├── 构建 prompt (system + user + history)                   [coldstart_search.py + prompts_search.py]
>                     ├── Model: local_model_infer() / deepseek()                 [coldstart_search.py]
>                     ├── extract_search_action()  → action_type + content        [coldstart_search.py]
>                     │
>                     ├── if search:
>                     │     env.step("<search>query</search>")                    [search/envs.py → SearchMultiProcessEnv]
>                     │       └── SearchEnv.step()                                [search/env.py]
>                     │             ├── _execute_tool()                           [search/env.py]
>                     │             │     └── call_search_api() → HTTP POST       [search/tools/search.py]
>                     │             └── return obs, reward=0, done=False
>                     │
>                     ├── if answer:
>                     │     env.step("<answer>ans</answer>")                      [search/envs.py → SearchMultiProcessEnv]
>                     │       └── SearchEnv.step()                                [search/env.py]
>                     │             ├── _get_reward()                             [search/env.py]
>                     │             │     └── compute_score() → EM                [search/utils.py]
>                     │             └── return obs="", reward=1/0, done=True
>                     │
>                     └── 终止判断: SUCCESS / WRONG / null(2次) / max_turns
> ```
>
> 路径简写对照：
> - `coldstart_search.py`        = `coldstart_genaration_search/coldstart_search.py`
> - `prompts_search.py`          = `coldstart_genaration_search/prompts_search.py`
> - `search/envs.py`             = `agent_system/environments/env_package/search/envs.py`
> - `search/env.py`              = `agent_system/environments/env_package/search/third_party/skyrl_gym/envs/search/env.py`
> - `search/tools/search.py`     = `agent_system/environments/env_package/search/third_party/skyrl_gym/tools/search.py`
> - `search/utils.py`            = `agent_system/environments/env_package/search/third_party/skyrl_gym/envs/search/utils.py`
> ```

## 总览

```
coldstart_genaration_search/
├── coldstart_search.py     ← 主入口
└── prompts_search.py       ← Prompt 模板
```

依赖的外部模块（`verl-agent/` 下）：

```
agent_system/environments/env_package/search/
├── __init__.py              → 导出 build_search_envs, search_projection
├── envs.py                  → SearchMultiProcessEnv (多进程包装器)
├── projection.py            → search_projection (动作合法性检查)
└── third_party/skyrl_gym/
    ├── envs/search/
    │   └── env.py           → SearchEnv (底层单环境)
    └── tools/
        └── search.py        → SearchToolGroup → call_search_api → HTTP POST 检索服务
```

数据预处理：

```
examples/data_preprocess/
├── download_search_r1_dataset.py       → 从 HuggingFace 下载 raw parquet
└── preprocess_search_r1_dataset.py     → 转换为训练/测试 parquet
```

---

## 详细调用链路

### 1. 数据准备阶段（独立运行）

```
download_search_r1_dataset.py
  → HuggingFace "PeterJinGo/nq_hotpotqa_train" → ~/data/searchR1_raw/{train,test}.parquet

preprocess_search_r1_dataset.py
  → 读 raw parquet
  → process_single_row():
      提取: question, ground_truth, data_source
      组装: env_kwargs = {"ground_truth": ..., "question": ..., "data_source": ...}
      输出: ~/data/searchR1_processed_direct/{train,test}.parquet
        列: data_source, prompt, reward_model, extra_info, metadata, env_kwargs
```

### 2. 冷启动生成主流程

```
coldstart_search.py :: __main__
  ├── 配置: USE_LOCAL_MODEL, MAX_TURNS, HIS_LEN, SEED ...
  ├── CHUNKS = [(start_idx, n_samples, desc), ...]
  │
  └── for each chunk:
       └── evaluate_coldstart_data(
              output_file, max_samples, max_turns,
              show_turn, his_len, seed, start_index, ...)
            │
            ├── [1] load_search_data(data_path)
            │     └── pd.read_parquet("~/data/searchR1_processed_direct/test.parquet")
            │     └── → df (iterrows)
            │
            ├── [2] build_search_envs(seed, env_num=1, group_n=1, is_train=False, env_config)
            │     └── agent_system/.../search/envs.py :: build_search_envs()
            │           └── → SearchMultiProcessEnv(seed, env_num, group_n, is_train, env_config)
            │                 └── 初始化: self.batch_size = env_num * group_n
            │                 └── 创建 N 个 SearchEnv(env_config.search):
            │                       └── third_party/.../search/env.py :: SearchEnv.__init__()
            │                             └── SearchToolGroup(search_url, topk, timeout, ...)
            │                 └── 线程池: ThreadPoolExecutor
            │                 └── 事件循环: asyncio
            │
            └── for each row in df (tqdm):
                 │
                 ├── 提取: question, ground_truth, data_source (从 row.env_kwargs)
                 │
                 └── get_single_trajectory(
                        env, question, ground_truth, data_source,
                        task_idx, max_turns, show_turn, his_len, ...)
                      │
                      ├── [3] env.reset(kwargs=[{ground_truth, question, data_source}])
                      │     └── SearchMultiProcessEnv.reset()
                      │           └── 并行: for each SearchEnv → _sync_reset(env, kw)
                      │                 ├── SearchEnv.reset(extras={ground_truth, max_turns, data_source})
                      │                 │     └── self.ground_truth = extras["ground_truth"]
                      │                 │     └── self.max_turns = extras["max_turns"]
                      │                 │     └── self.chat_history = []
                      │                 │     └── self.done = False
                      │                 │     └── self.turns = 0
                      │                 └── return obs=kwargs["question"], info={data_source}
                      └── → obs_list=[question], info_list=[{data_source}]
                      │
                      ├── 初始化 history_list = {Action: [], Observation: []}
                      ├── messages = [system_msg]
                      │
                      └── for turn in range(max_turns):
                            │
                            ├── 构建 user_msg (含历史)
                            │   └── 第 1 轮: SEARCH_PROMPT_NO_HIS.format(question)
                            │   └── 第 2+ 轮: SEARCH_PROMPT_HIS.format(question, step_count, history)
                            │
                            ├── partial_messages = [{task_idx,turn}, system_msg, user_msg]
                            │
                            ├── [4] 模型推理
                            │   ├── use_local_model=True:
                            │   │     └── local_model_infer(messages=[system, user])
                            │   │           ├── tokenizer.apply_chat_template()
                            │   │           └── model.generate(max_new_tokens, temperature=0.7, ...)
                            │   │           └── → 解码 response
                            │   └── use_local_model=False:
                            │         └── deepseek(messages=[system, user])
                            │               └── OpenAI(api_key, base_url).chat.completions.create()
                            │               └── → response.choices[0].message.content
                            │
                            ├── partial_messages += [{role:"assistant", content: response}]
                            ├── seperated_list.append({messages: partial_messages})
                            │
                            ├── [5] extract_search_action(response)
                            │     └── re.search(r'<think>(.*?)</think>', ...)
                            │     └── re.search(r'<search>(.*?)</search>', ...)
                            │     └── re.search(r'<answer>(.*?)</answer>', ...)
                            │     └── → {think, action_type, action_content}
                            │
                            ├── [6] env.step([env_action])
                            │   └── action_type='search':
                            │   │     env_action = "<search>{query}</search>"
                            │   │     └── SearchMultiProcessEnv.step([action])
                            │   │           └── 并行: _sync_step(env, action)
                            │   │                 └── SearchEnv.step(action)
                            │   │                       ├── self.turns += 1
                            │   │                       ├── self.chat_history.append({role:"assistant", content:action})
                            │   │                       ├── _is_done(action)
                            │   │                       │     └── <answer> found OR turns >= max_turns
                            │   │                       ├── _parse_action(action)
                            │   │                       │     └── re.search(r'<search>(.*?)</search>')
                            │   │                       ├── _execute_tool("SearchToolGroup", "search", [query])
                            │   │                       │     └── SearchToolGroup.search(query)
                            │   │                       │           └── call_search_api(search_url, query, topk, ...)
                            │   │                       │                 └── HTTP POST → retrieval server
                            │   │                       │                 └── → JSON results
                            │   │                       │                 └── → str: "\n<information>...results...</information>\n"
                            │   │                       ├── self.chat_history.append({role:"user", content:observation})
                            │   │                       └── return BaseTextEnvStepOutput(
                            │   │                                observations=[{role:"user", content:"<information>..."}],
                            │   │                                reward=0, done=False, metadata={...})
                            │   │                 └── _sync_step 解包:
                            │   │                       obs = observations[0]["content"]
                            │   │                       reward, done, info = ...
                            │   │                 → (obs, reward, done, info)
                            │   │           → (obs_list, reward_list, done_list, info_list)
                            │   │
                            │   └── action_type='answer':
                            │         env_action = "<answer>{content}</answer>"
                            │         └── SearchEnv.step(action)
                            │               ├── _is_done → True (含 <answer>)
                            │               ├── _get_reward(action, done=True)
                            │               │     └── compute_score(chat_history_str, ground_truth)
                            │               │           ├── extract_solution() → <answer> 内容
                            │               │           └── em_check(answer, ground_truth["target"])
                            │               │           → 1.0 或 0.0
                            │               └── return BaseTextEnvStepOutput(
                            │                        observations=[], reward=1.0/0.0, done=True, ...)
                            │         → info["won"] = bool(done and reward >= 1.0)
                            │
                            ├── 解包结果: new_obs, new_reward, new_done, won
                            ├── history_list 记录
                            │
                            └── 终止判断:
                                  ├── action_type='answer':
                                  │     ├── won=True  → SUCCESS
                                  │     └── won=False → 回答错误
                                  ├── new_done=True  → 搜索时耗尽 max_turns
                                  ├── null_count>=2  → 连续空动作
                                  └── for-else      → 超出 max_turns
                      │
                      └── return (messages, success_flag, status_msg, seperated_list)
            │
            └── 保存结果:
                  ├── {output_file}              → 完整轨迹
                  ├── {output_file}_success.json → 成功轨迹
                  └── {output_file}_seperated.json → seperated 格式训练数据
```

---

## 关键数据流

### reset 数据流

```
parquet row
  → env_kwargs = {"ground_truth": {"target": "answer"}, "question": "Q?", "data_source": "nq"}
  → env.reset([env_kwargs])
    → SearchMultiProcessEnv._sync_reset(env, kw)
      → SearchEnv.reset(extras={"ground_truth": ..., "max_turns": N, "data_source": ...})
        → self.ground_truth = ...
        → self.chat_history = []
      → return kw["question"], {"data_source": ...}
    → return ["Q?"], [{"data_source": "nq"}]
```

### step 数据流 (search 动作)

```
"<search>some query</search>"
  → SearchMultiProcessEnv.step(["<search>some query</search>"])
    → SearchEnv.step("<search>some query</search>")
      → chat_history += [{"role": "assistant", "content": "<search>some query</search>"}]
      → _parse_action → ["some query"]
      → _execute_tool("SearchToolGroup", "search", ["some query"])
        → SearchToolGroup.search(["some query"])
          → call_search_api(search_url, "some query", topk=3, ...)
            → HTTP POST 到检索服务器
            → return "\n<information>...检索结果文本...</information>\n"
      → chat_history += [{"role": "user", "content": "\n<information>...\n"}]
      → return BaseTextEnvStepOutput(
          observations=[{"role": "user", "content": "\n<information>...\n"}],
          reward=0,
          done=False,
          metadata={...})
    → _sync_step 解包:
        obs = "\n<information>...\n"
        reward = 0
        done = False
        info = {..., "won": False}
    → return ["\n<information>...\n"], [0], [False], [{...}]
```

### step 数据流 (answer 动作)

```
"<answer>Beijing</answer>"
  → SearchEnv.step("<answer>Beijing</answer>")
    → _is_done → True
    → _get_reward:
        chat_history_str = 拼接所有 chat_history
        compute_score(chat_history_str, ground_truth)
          → extract_solution() → "Beijing"
          → em_check("Beijing", ground_truth["target"])
          → 1.0 或 0.0
    → return BaseTextEnvStepOutput(
        observations=[],
        reward=1.0/0.0,
        done=True,
        metadata={"data_source": ..., "tool_calling": False})
    → _sync_step 解包:
        obs = "" (observations 为空)
        reward = 1.0 或 0.0
        done = True
        info = {..., "won": True/False}
```

---

## 文件函数索引

| 文件 | 函数/类 | 作用 |
|------|---------|------|
| `coldstart_search.py` | `__main__` | 入口，配置分块，循环调用 |
| | `evaluate_coldstart_data()` | 主控：加载数据→建环境→逐条生成→保存 |
| | `get_single_trajectory()` | 单问题轨迹生成：reset→多轮搜索/回答→终止 |
| | `load_search_data()` | 从 parquet 加载测试数据 |
| | `load_local_model()` | 加载本地 Qwen 模型 |
| | `local_model_infer()` | 本地模型推理 |
| | `deepseek()` | DeepSeek API 推理 |
| | `get_search_system_message()` | 组装 system message |
| | `get_search_user_message()` | 组装 user message (含历史) |
| | `extract_search_action()` | 正则提取 think/search/answer |
| | `get_unique_filename()` | 避免文件名冲突 |
| `prompts_search.py` | `SYSTEM_MESSAGE_SEARCH` | 系统 prompt |
| | `SEARCH_PROMPT_NO_HIS` | 首轮 user prompt |
| | `SEARCH_PROMPT_HIS` | 带历史的 user prompt |
| `search/envs.py` | `SearchMultiProcessEnv` | 多环境并行包装器 |
| | `_sync_reset()` | 同步 reset 单个 env |
| | `_sync_step()` | 同步 step 单个 env |
| | `build_search_envs()` | 工厂函数 |
| `search/third_party/.../env.py` | `SearchEnv` | 底层搜索环境 |
| | `reset(extras)` | 初始化，设 ground_truth/max_turns |
| | `step(action)` | 执行搜索或回答 |
| | `_parse_action()` | 解析 <search> 标签 |
| | `_execute_tool()` | 调用 SearchToolGroup |
| | `_get_reward()` | 计算 reward (EM) |
| | `_is_done()` | 判断终止 |
| `search/third_party/.../utils.py` | `compute_score()` | EM 评分函数 |
| | `extract_solution()` | 从 <answer> 提取答案 |
| | `em_check()` | 精确匹配 |
| `search/third_party/.../search.py` | `SearchToolGroup` | 搜索工具组 |
| | `call_search_api()` | HTTP POST 到检索服务器 |
| `search/projection.py` | `search_projection()` | 动作合法性检查 |
| `data_preprocess/...py` | `process_single_row()` | 原始数据→训练格式 |

