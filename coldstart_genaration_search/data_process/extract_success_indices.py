import re
import json
import glob
import os

log_dir = "/diskpool/home/xuxz/verl-agent/coldstart_genaration_search/result_search"
log_pattern = "search_coldstart_*.log"
log_files = sorted(glob.glob(os.path.join(log_dir, log_pattern)))

output_file = "/diskpool/home/xuxz/verl-agent/coldstart_genaration_search/result_search/data_after_process/success_indices_merged.txt"

all_indices = []
for fpath in log_files:
    with open(fpath, "r") as f:
        for line in f:
            m = re.match(r"Success indices:\s*(\[.*\])", line.strip())
            if m:
                indices = json.loads(m.group(1))
                all_indices.extend(indices)

all_indices.sort()

with open(output_file, "w") as f:
    f.write(f"success indices merged:{json.dumps(all_indices)}\n")
    f.write(f"len_list = {len(all_indices)}\n")

print(f"Total success indices: {len(all_indices)}")
print(f"Success indices merged:{json.dumps(all_indices)}")
print(f"len_list = {len(all_indices)}")
