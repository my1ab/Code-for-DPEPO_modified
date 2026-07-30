import json
import os
from tqdm import tqdm

def check_human_ins_file(file_path):
    if not os.path.exists(file_path):
        print(f"错误：文件不存在 - {file_path}")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_items = len(data)
    items_with_instruction = 0
    items_with_attributes = 0
    items_with_instruction_attributes = 0
    
    instruction_attr_counts = []
    attributes_counts = []
    
    for item_id, item_data in tqdm(data.items(), desc="Processing items", total=total_items):
        if isinstance(item_data, list):
            for annotation in item_data:
                if 'instruction' in annotation and annotation['instruction']:
                    items_with_instruction += 1
                
                if 'attributes' in annotation and annotation['attributes']:
                    items_with_attributes += 1
                    attributes_counts.append(len(annotation['attributes']))
                
                if 'instruction_attributes' in annotation and annotation['instruction_attributes']:
                    items_with_instruction_attributes += 1
                    instruction_attr_counts.append(len(annotation['instruction_attributes']))
        else:
            if 'instruction' in item_data and item_data['instruction']:
                items_with_instruction += 1
            
            if 'attributes' in item_data and item_data['attributes']:
                items_with_attributes += 1
                attributes_counts.append(len(item_data['attributes']))
            
            if 'instruction_attributes' in item_data and item_data['instruction_attributes']:
                items_with_instruction_attributes += 1
                instruction_attr_counts.append(len(item_data['instruction_attributes']))
    
    avg_attrs = sum(attributes_counts) / len(attributes_counts) if attributes_counts else 0
    avg_instr_attrs = sum(instruction_attr_counts) / len(instruction_attr_counts) if instruction_attr_counts else 0
    
    print("=" * 60)
    print("items_human_ins.json 文件检查报告")
    print("=" * 60)
    print(f"文件路径: {file_path}")
    print(f"总商品数量: {total_items}")
    print(f"包含 instruction 的商品数量: {items_with_instruction} ({(items_with_instruction/total_items)*100:.2f}%)")
    print(f"包含 attributes 的商品数量: {items_with_attributes} ({(items_with_attributes/total_items)*100:.2f}%)")
    print(f"包含 instruction_attributes 的商品数量: {items_with_instruction_attributes} ({(items_with_instruction_attributes/total_items)*100:.2f}%)")
    print(f"平均每个商品的 attributes 数量: {avg_attrs:.2f}")
    print(f"平均每个商品的 instruction_attributes 数量: {avg_instr_attrs:.2f}")
    print("=" * 60)
    
    if total_items > 0:
        first_item = list(data.keys())[0]
        print("\n第一个商品的结构示例:")
        print(f"商品ID: {first_item}")
        print(f"内容: {json.dumps(data[first_item], indent=2, ensure_ascii=False)}")

if __name__ == "__main__":
    file_path = '/home/dpepo/data/items_human_ins.json'
    check_human_ins_file(file_path)
    
    print("\n" + "=" * 60)
    print("尝试检查备用路径...")
    print("=" * 60)
    print(f'end check human ins file')
    
    # alt_path = '/home/dpepo/verl-agent/items_ins_v2_1000.json'
    # if os.path.exists(alt_path):
    #     print(f"检查备用文件: {alt_path}")
    #     check_human_ins_file(alt_path)
    # else:
    #     print("备用文件也不存在")