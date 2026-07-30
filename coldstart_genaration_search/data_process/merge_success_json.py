import json
import glob
import os
import re


def merge(input_files, output_path, max_trajectories):
    """Merge multiple success JSON files (sorted by range) into one."""
    all_data = []
    for fpath in input_files:
        with open(fpath, "r") as f:
            data = json.load(f)
            all_data.extend(data)

    # 截断到轨迹上限
    if len(all_data) > max_trajectories:
        print(f"Trajectory count ({len(all_data)}) exceeds limit ({max_trajectories}), truncating...")
        all_data = all_data[:max_trajectories]

    with open(output_path, "w") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"Merged {len(all_data)} success trajectories -> {output_path} (max {max_trajectories})")
    return all_data



def convert_to_seperated(success_data, output_path, include_metadata):
    """
    Convert success JSON format to seperated format and write to file.

    Each trajectory -> list of turn entries.
    Each entry has messages=[metadata, system, user, assistant]:
      messages[0] = {"task_idx": N, "turn": N}          # physical_idx/logical_idx (when include_metadata=True)
      messages[1] = {"role": "system", "content": ...}  # role+content
      messages[2] = {"role": "user",   "content": ...}  # role+content
      messages[3] = {"role": "assistant", "content": ...} # role+content

    Note: The success JSON uses "logical_idx" rather than "task_idx".
          The script uses logical_idx as the task identifier.

    Args:
        success_data: 输入的 success JSON 数据
        output_path: 输出文件路径
        include_metadata: 是否在 messages 开头附加 task_idx 和 turn 元数据
    """
    seperated_output = []

    for item in success_data:
        task_idx = item.get("logical_idx", 0)
        trajectory = item["trajectory"]

        # trajectory[0] is always system
        system_entry = {"role": "system", "content": trajectory[0]["content"]}

        task_entries = []
        # trajectory[1:] alternates: user, assistant, user, assistant, ...
        for i in range(1, len(trajectory), 2):
            if i + 1 >= len(trajectory):
                break
            user_msg = trajectory[i]
            assistant_msg = trajectory[i + 1]
            turn = assistant_msg.get("turn", (i + 1) // 2)

            if include_metadata:
                entry = {
                    "messages": [
                        {"task_idx": task_idx, "turn": turn},
                        system_entry,
                        {"role": "user", "content": user_msg["content"]},
                        {"role": "assistant", "content": assistant_msg["content"]},
                    ]
                }
            else:
                entry = {
                    "messages": [
                        system_entry,
                        {"role": "user", "content": user_msg["content"]},
                        {"role": "assistant", "content": assistant_msg["content"]},
                    ]
                }
            task_entries.append(entry)

        seperated_output.append(task_entries)

    with open(output_path, "w") as f:
        json.dump(seperated_output, f, indent=2, ensure_ascii=False)

    total_tasks = len(seperated_output)
    total_entries = sum(len(task) for task in seperated_output)
    print(f"Converted {total_tasks} tasks to seperated format -> {output_path}")
    print(f"Total turn entries: {total_entries}")

    return seperated_output


def convert_to_formatted(success_data, output_path):
    """
    Convert success JSON to flat turn-level format.

    Each turn -> single dict with one "messages" array = [system, user, assistant].
    ALL turns from ALL trajectories are flattened into one list (no trajectory grouping).

    Args:
        success_data: 输入的 success JSON 数据 (list of {task_idx, trajectory})
        output_path: 输出文件路径
    """
    formatted_output = []

    for item in success_data:
        trajectory = item["trajectory"]
        system_entry = {"role": "system", "content": trajectory[0]["content"]}

        # trajectory[1:] alternates: user, assistant, user, assistant, ...
        for i in range(1, len(trajectory), 2):
            if i + 1 >= len(trajectory):
                break
            user_msg = trajectory[i]
            assistant_msg = trajectory[i + 1]

            entry = {
                "messages": [
                    system_entry,
                    {"role": "user", "content": user_msg["content"]},
                    {"role": "assistant", "content": assistant_msg["content"]},
                ]
            }
            formatted_output.append(entry)

    with open(output_path, "w") as f:
        json.dump(formatted_output, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(formatted_output)} turn entries to formatted format -> {output_path}")
    return formatted_output


def discover_success_files(base_dir):
    """扫描一级目录下所有 `*_success.json` 文件，按 range 排序后返回。"""
    pattern = os.path.join(base_dir, "*_success.json")
    files = glob.glob(pattern)

    def extract_range(fpath):
        """从文件名中提取 (start, end) 用于排序。"""
        basename = os.path.basename(fpath)
        match = re.search(r'(\d+)_(\d+)_success\.json$', basename)
        if match:
            return int(match.group(1)), int(match.group(2))
        return (0, 0)

    files.sort(key=extract_range)
    return files


if __name__ == "__main__":
    # 脚本所在目录的父目录即为 result_search 一级目录
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(BASE_DIR, "merge", "merged")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 自动发现所有 success JSON 文件
    input_files = discover_success_files(BASE_DIR)

    if not input_files:
        print("未找到任何 *_success.json 文件！")
        exit(1)

    print(f"发现 {len(input_files)} 个 success 文件:")
    for f in input_files:
        print(f"  {os.path.basename(f)}")

    # 从文件名推断总 range
    first_name = os.path.basename(input_files[0])
    last_name = os.path.basename(input_files[-1])
    start_match = re.search(r'(\d+)_(\d+)_success\.json$', first_name)
    end_match = re.search(r'(\d+)_(\d+)_success\.json$', last_name)
    if start_match and end_match:
        range_label = f"{start_match.group(1)}_{end_match.group(2)}"
    else:
        range_label = "all"

    # 1. 合并 success JSON
    merged_path = os.path.join(OUTPUT_DIR, f"search_coldstart_{range_label}_success_merged_500.json")
    merged_data = merge(input_files, merged_path, max_trajectories=500)

    # 2. 转换为 seperated 格式并写入文件
    seperated_path = os.path.join(OUTPUT_DIR, f"search_coldstart_{range_label}_seperated_merged_cleaned_500.json")
    seperated_data = convert_to_seperated(merged_data, seperated_path, 
                                          include_metadata=False)

    # 3. 转换为 flat conversation 格式（对齐模板 example.json）
    formatted_path = os.path.join(OUTPUT_DIR, f"search_coldstart_{range_label}_bracket_formatted.json")
    formatted_data = convert_to_formatted(merged_data, formatted_path)
