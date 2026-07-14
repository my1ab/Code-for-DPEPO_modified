import json
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ===== 路径集中配置 =====
_DPEPO_USER_HOME = os.environ.get('DPEPO_USER_HOME', '/diskpool/home/xuxz')
_DPEPO_PROJECT_NAME = os.environ.get('DPEPO_PROJECT_NAME', 'Code-for-DPEPO')
_DPEPO_CODE_BASE = os.path.join(_DPEPO_USER_HOME, _DPEPO_PROJECT_NAME)
_GAMEFILES_DIR = os.path.join(_DPEPO_CODE_BASE, 'data_pipelines', 'gamefiles')
# ======================


def read_json(filepath):
    return json.load(open(filepath, 'r'))


def main(exclude_success=False):
    filepath = os.path.join(_GAMEFILES_DIR, 'alfworld', 'gamefiles_train.json')
    total_train_data = read_json(filepath=filepath)

    gamefiles = list(total_train_data.values())

    if exclude_success:
        synthestic = '/data/home/zhangjs/disk/project/verl-agent/data_pipelines/datas/parallel_vanilla/win.json'

        synthestic_data = read_json(synthestic)
        synthestic_gamefiles = [elem['game_file'] for elem in synthestic_data]

        left_gamefiles = list(set(gamefiles) - set(synthestic_gamefiles))
    else:
        left_gamefiles = gamefiles

    parquet_train_data = []
    for file in left_gamefiles:
        parquet_train_data.append(
            {
                'answer': '',
                'data_source': 'alfworld',
                'prompt': [{'role': 'user', 'content': 'The prompt is dynamic obtained from envs'}],
                'ability': 'agent',
                'gamefile': file,
                'extra_info': {
                    'split': 'train'
                }
            }
        )

    current_data = parquet_train_data[:500]
    df = pd.DataFrame(current_data)

    # df.to_parquet('../verl_train_data/dedu_parallel_train_data_500.parquet', index=False)
    df.to_parquet(os.path.join(_GAMEFILES_DIR, 'dedu_parallel_train_data_500.parquet'), index=False)

    game_files = {}
    for idx, elem in enumerate(current_data):
        gamefile = elem['gamefile']
        game_files[str(idx)] = gamefile

    with open(os.path.join(_GAMEFILES_DIR, 'dedu_parallel_train_data_500.json'), 'w') as f:
        json.dump(game_files, f, indent=4)


if __name__ == '__main__':
    main()
