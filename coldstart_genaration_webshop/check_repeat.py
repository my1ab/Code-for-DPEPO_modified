import os
import json
from tqdm import tqdm

def check_dataset_overlap():
    """
    检查全量数据集和1000个数据集是否有重合
    数据集路径：/home/dpepo/data/
    - 全量数据集: items_shuffle.json, items_ins_v2.json
    - 1000个数据集: items_shuffle_1000.json, items_ins_v2_1000.json
    """
    # base_path = '/home/dpepo/data/'
    base_path = '/diskpool/home/xuxz/data/'
    
    # 定义文件路径
    files = {
        'full_shuffle': os.path.join(base_path, 'items_shuffle.json'),
        'full_ins': os.path.join(base_path, 'items_ins_v2.json'),
        'small_shuffle': os.path.join(base_path, 'items_shuffle_1000.json'),
        'small_ins': os.path.join(base_path, 'items_ins_v2_1000.json')
    }
    
    # 检查文件是否存在
    for name, path in files.items():
        if os.path.exists(path):
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: {path} (不存在)")
    
    print("\n" + "="*60)
    
    # 读取数据集并提取产品ID（同时统计重复）
    def load_and_extract_ids(file_path):
        """
        支持两种数据结构：
        1. 列表结构：items_shuffle系列，使用顶层小写 asin 字段作为ID
        2. 字典结构：items_ins_v2系列，key 即为 ASIN
        
        ASIN统一转换为大写，确保大小写不敏感的比较
        """
        if not os.path.exists(file_path):
            return set(), 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            total_count = 0
            ids = set()
            
            # 根据数据结构提取ID
            if isinstance(data, list):
                # 列表结构：items_shuffle系列，每个元素是产品
                total_count = len(data)
                for item in tqdm(data, desc=f"  处理 {os.path.basename(file_path)}", leave=False):
                    if isinstance(item, dict) and 'asin' in item:
                        # 直接使用顶层小写 asin 字段
                        ids.add(str(item['asin']).upper())
                return ids, total_count
            elif isinstance(data, dict):
                # 字典结构：items_ins_v2系列，key 即为 ASIN
                # 先检查是否为嵌套列表结构（如 {'products': [...]}）
                for key in ['products', 'items', 'data']:
                    if key in data and isinstance(data[key], list):
                        total_count = len(data[key])
                        for item in tqdm(data[key], desc=f"  处理 {os.path.basename(file_path)}", leave=False):
                            if isinstance(item, dict) and 'asin' in item:
                                ids.add(str(item['asin']).upper())
                        return ids, total_count
                # items_ins_v2系列：直接的产品字典（key为ASIN）
                return set(str(k).upper() for k in data.keys()), len(data)
            else:
                print(f"  无法解析 {file_path} 的数据结构")
                return set(), 0
        except Exception as e:
            print(f"  读取 {file_path} 失败: {e}")
            return set(), 0
    
    # 提取各数据集的产品ID
    print("提取产品ID...")
    
    full_shuffle_ids, full_shuffle_total = load_and_extract_ids(files['full_shuffle'])
    small_shuffle_ids, small_shuffle_total = load_and_extract_ids(files['small_shuffle'])
    
    full_ins_ids, full_ins_total = load_and_extract_ids(files['full_ins'])
    small_ins_ids, small_ins_total = load_and_extract_ids(files['small_ins'])
    
    print("\n数据集基本统计:")
    print(f"  items_shuffle.json: 总条目数={full_shuffle_total}, 唯一产品数={len(full_shuffle_ids)}, 重复数={full_shuffle_total - len(full_shuffle_ids)}")
    print(f"  items_shuffle_1000.json: 总条目数={small_shuffle_total}, 唯一产品数={len(small_shuffle_ids)}, 重复数={small_shuffle_total - len(small_shuffle_ids)}")
    print(f"  items_ins_v2.json: 总条目数={full_ins_total}, 唯一产品数={len(full_ins_ids)}, 重复数={full_ins_total - len(full_ins_ids)}")
    print(f"  items_ins_v2_1000.json: 总条目数={small_ins_total}, 唯一产品数={len(small_ins_ids)}, 重复数={small_ins_total - len(small_ins_ids)}")
    
    print("\n" + "="*60)
    
    # 检查重合情况
    print("检查重合情况...")
    
    # items_shuffle 系列的重合
    if full_shuffle_ids and small_shuffle_ids:
        overlap_shuffle = full_shuffle_ids & small_shuffle_ids
        print(f"\nitems_shuffle系列重合分析:")
        print(f"  全量数据集大小: {len(full_shuffle_ids)}")
        print(f"  1000数据集大小: {len(small_shuffle_ids)}")
        print(f"  重合数量: {len(overlap_shuffle)}")
        print(f"  重合率(1000数据集在全量中的占比): {len(overlap_shuffle)/len(small_shuffle_ids)*100:.2f}%")
        print(f"  重合率(全量数据集在1000中的占比): {len(overlap_shuffle)/len(full_shuffle_ids)*100:.2f}%")
        
        if len(overlap_shuffle) == len(small_shuffle_ids):
            print("  ✓ 1000数据集是全量数据集的子集")
        elif len(overlap_shuffle) == 0:
            print("  ✗ 两个数据集完全不重合")
        else:
            print("  ⚠ 部分重合")
    
    # items_ins_v2 系列的重合
    if full_ins_ids and small_ins_ids:
        overlap_ins = full_ins_ids & small_ins_ids
        print(f"\nitems_ins_v2系列重合分析:")
        print(f"  全量数据集大小: {len(full_ins_ids)}")
        print(f"  1000数据集大小: {len(small_ins_ids)}")
        print(f"  重合数量: {len(overlap_ins)}")
        print(f"  重合率(1000数据集在全量中的占比): {len(overlap_ins)/len(small_ins_ids)*100:.2f}%")
        print(f"  重合率(全量数据集在1000中的占比): {len(overlap_ins)/len(full_ins_ids)*100:.2f}%")
        
        if len(overlap_ins) == len(small_ins_ids):
            print("  ✓ 1000数据集是全量数据集的子集")
        elif len(overlap_ins) == 0:
            print("  ✗ 两个数据集完全不重合")
        else:
            print("  ⚠ 部分重合")
    
    # 检查两个系列之间的对应关系
    print("\n" + "="*60)
    print("检查两个系列之间的ID对应关系...")
    
    if small_shuffle_ids and small_ins_ids:
        shuffle_not_in_ins = small_shuffle_ids - small_ins_ids
        ins_not_in_shuffle = small_ins_ids - small_shuffle_ids
        
        print(f"\n1000数据集系列:")
        print(f"  items_shuffle_1000.json 中有但 items_ins_v2_1000.json 中没有的ID数量: {len(shuffle_not_in_ins)}")
        print(f"  items_ins_v2_1000.json 中有但 items_shuffle_1000.json 中没有的ID数量: {len(ins_not_in_shuffle)}")
        
        if len(shuffle_not_in_ins) == 0 and len(ins_not_in_shuffle) == 0:
            print("  ✓ 两个1000数据集的ID完全一致")
        elif len(shuffle_not_in_ins) == 0:
            print("  ⚠ items_shuffle_1000.json 的ID是 items_ins_v2_1000.json 的子集")
        elif len(ins_not_in_shuffle) == 0:
            print("  ⚠ items_ins_v2_1000.json 的ID是 items_shuffle_1000.json 的子集")
    
    if full_shuffle_ids and full_ins_ids:
        shuffle_not_in_ins_full = full_shuffle_ids - full_ins_ids
        ins_not_in_shuffle_full = full_ins_ids - full_shuffle_ids
        
        print(f"\n全量数据集系列:")
        print(f"  items_shuffle.json 中有但 items_ins_v2.json 中没有的ID数量: {len(shuffle_not_in_ins_full)}")
        print(f"  items_ins_v2.json 中有但 items_shuffle.json 中没有的ID数量: {len(ins_not_in_shuffle_full)}")
        
        if len(shuffle_not_in_ins_full) == 0 and len(ins_not_in_shuffle_full) == 0:
            print("  ✓ 两个全量数据集的ID完全一致")

if __name__ == "__main__":
    print("="*60)
    print("数据集重合检查工具")
    print("="*60)
    check_dataset_overlap()
    print("\n" + "="*60)
    print("检查完成")