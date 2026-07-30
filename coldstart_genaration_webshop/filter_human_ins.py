import json
import os
from tqdm import tqdm

def filter_human_ins_by_asin(human_ins_path, items_shuffle_path, output_path):
    if not os.path.exists(human_ins_path):
        print(f"错误：人类标注文件不存在 - {human_ins_path}")
        return
    
    if not os.path.exists(items_shuffle_path):
        print(f"错误：商品数据文件不存在 - {items_shuffle_path}")
        return
    
    print(f"开始加载商品数据文件: {items_shuffle_path}")
    with open(items_shuffle_path, 'r', encoding='utf-8') as f:
        items_data = json.load(f)
    
    asin_set = set()
    # for item in items_data:
    for item in tqdm(items_data, desc="Loading ASINs", total=len(items_data)):
        if 'asin' in item:
            asin_set.add(item['asin'])
    
    print(f"从 {items_shuffle_path} 中加载了 {len(asin_set)} 个唯一 ASIN")
    
    with open(human_ins_path, 'r', encoding='utf-8') as f:
        human_ins_data = json.load(f)
    
    filtered_data = {}
    removed_count = 0
    kept_count = 0
    
    for asin, annotations in tqdm(human_ins_data.items(), desc="Filtering annotations", total=len(human_ins_data)):
        if asin in asin_set:
            filtered_data[asin] = annotations
            kept_count += len(annotations) if isinstance(annotations, list) else 1
        else:
            removed_count += len(annotations) if isinstance(annotations, list) else 1
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    
    print(f"过滤完成！")
    print(f"原始商品数量: {len(human_ins_data)}")
    print(f"过滤后商品数量: {len(filtered_data)}")
    print(f"移除的标注数量: {removed_count}")
    print(f"保留的标注数量: {kept_count}")
    print(f"结果已保存到: {output_path}")

if __name__ == "__main__":
    # human_ins_path = '/home/dpepo/data/items_human_ins.json'
    # items_shuffle_path = '/home/dpepo/verl-agent/items_shuffle_1000.json'
    # items_shuffle_path = '/home/dpepo/data/items_shuffle.json'
    # output_path = '/home/dpepo/data/items_human_ins_filtered.json'
    human_ins_path = '/diskpool/home/xuxz/data/items_human_ins.json'
    items_shuffle_path = '/diskpool/home/xuxz/data/items_shuffle.json'
    output_path = '/diskpool/home/xuxz/data/items_human_ins_filtered.json'
    
    filter_human_ins_by_asin(human_ins_path, items_shuffle_path, output_path)