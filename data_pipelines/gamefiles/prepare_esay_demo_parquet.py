"""
功能：将 ALFWorld 的游戏文件路径列表转换为 Parquet + JSON 格式的训练数据文件。

用法：直接运行即可生成 parquet 和 json 文件。
"""

import json
import pandas as pd


def read_json(filepath):
    return json.load(open(filepath, 'r'))


def main():
    # ALFWorld 训练集游戏文件路径列表
    gamefiles = [
        "/diskpool/home/xuxz/.cache/alfworld/json_2.1.1/train/look_at_obj_in_light-Book-None-DeskLamp-320/trial_T20190909_152445_177541/game.tw-pddl",
        "/diskpool/home/xuxz/.cache/alfworld/json_2.1.1/train/pick_clean_then_place_in_recep-Bowl-None-DiningTable-23/trial_T20190907_221735_616141/game.tw-pddl",
        "/diskpool/home/xuxz/.cache/alfworld/json_2.1.1/train/pick_two_obj_and_place-Pen-None-GarbageCan-321/trial_T20190907_201828_429337/game.tw-pddl",
        "/diskpool/home/xuxz/.cache/alfworld/json_2.1.1/train/pick_two_obj_and_place-Pen-None-GarbageCan-321/trial_T20190907_201720_953919/game.tw-pddl",
        "/diskpool/home/xuxz/.cache/alfworld/json_2.1.1/train/look_at_obj_in_light-Mug-None-DeskLamp-301/trial_T20190908_155916_103990/game.tw-pddl",
        "/diskpool/home/xuxz/.cache/alfworld/json_2.1.1/train/pick_and_place_simple-Candle-None-Toilet-429/trial_T20190908_052248_516834/game.tw-pddl",
        "/diskpool/home/xuxz/.cache/alfworld/json_2.1.1/train/pick_cool_then_place_in_recep-Bread-None-CounterTop-11/trial_T20190907_185214_077230/game.tw-pddl",
    ]

    # 构建训练数据列表
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
                    'split': 'train'
                }
            }
        )

    # 转换为 DataFrame 并保存为 Parquet
    df = pd.DataFrame(parquet_train_data)
    # df.to_parquet('../verl_train_data/parallel_train_data_demo_easy.parquet', index=False)
    df.to_parquet('/diskpool/home/xuxz/Code-for-DPEPO/data_pipelines/gamefiles/parallel_train_data_demo_easy.parquet', index=False)
    print(f"已生成 parquet 文件，共 {len(df)} 条数据")

    # 同时保存为同名 JSON 文件
    json_path = '/diskpool/home/xuxz/Code-for-DPEPO/data_pipelines/gamefiles/parallel_train_data_demo_easy.json'
    df.to_json(json_path, orient='records', force_ascii=False, indent=2)
    print(f"已生成 json 文件: {json_path}")


if __name__ == '__main__':
    main()
