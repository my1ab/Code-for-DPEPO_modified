import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def read_json(filepath):
    return json.load(open(filepath, 'r'))


def main():
    # filepath = 'gamefiles_eval.json'
    filepath = '/diskpool/home/xuxz/Code-for-DPEPO/data_pipelines/gamefiles/alfworld/gamefiles_eval.json'
    total_train_data = read_json(filepath=filepath)
    
    gamefiles = list(total_train_data.values())

    parquet_train_data = []
    for file in gamefiles:
        parquet_train_data.append(
            {
                'answer': '',
                'data_source': 'alfworld',
                'prompt': [{'role': 'user', 'content': 'The prompt is dynamic obtained from envs'}],
                'ability': 'agent',
                'gamefile': file,
                'extra_info': {
                    'split': 'test'
                }
            }
        )

    current_data = parquet_train_data[:500]
    df = pd.DataFrame(current_data)

    # df.to_parquet('../verl_train_data/test_indomain.parquet', index=False)
    df.to_parquet('/diskpool/home/xuxz/Code-for-DPEPO/data_pipelines/gamefiles/test_indomain.parquet', index=False)

    game_files = {}
    for idx, elem in enumerate(current_data):
        gamefile = elem['gamefile']
        game_files[str(idx)] = gamefile

    # with open('../gamefiles/dedu_parallel_train_data_500.json', 'w') as f:
    with open('/diskpool/home/xuxz/Code-for-DPEPO/data_pipelines/gamefiles/test_indomain.json', 'w') as f:
        json.dump(game_files, f, indent=4)


if __name__ == '__main__':
    main()
