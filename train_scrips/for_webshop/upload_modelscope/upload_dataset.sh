# modelscope login --token ms-c584573c-009e-45dc-b83d-52d52faa6ac3

# 上传 items_human_ins.json
modelscope upload \
  --repo-type model \
  --commit-message "upload items_human_ins.json" \
  afordb/sft_result_webshop_and_search_based_on_qwen \
  /diskpool/home/xuxz/data/items_human_ins.json \
  dataset/webshop/items_human_ins.json

# 上传 items_ins_v2_1000.json
modelscope upload \
  --repo-type model \
  --commit-message "upload items_ins_v2_1000.json" \
  afordb/sft_result_webshop_and_search_based_on_qwen \
  /diskpool/home/xuxz/data/items_ins_v2_1000.json \
  dataset/webshop/items_ins_v2_1000.json

# 上传 items_ins_v2.json
modelscope upload \
  --repo-type model \
  --commit-message "upload items_ins_v2.json" \
  afordb/sft_result_webshop_and_search_based_on_qwen \
  /diskpool/home/xuxz/data/items_ins_v2.json \
  dataset/webshop/items_ins_v2.json

# 上传 items_shuffle_1000.json
modelscope upload \
  --repo-type model \
  --commit-message "upload items_shuffle_1000.json" \
  afordb/sft_result_webshop_and_search_based_on_qwen \
  /diskpool/home/xuxz/data/items_shuffle_1000.json \
  dataset/webshop/items_shuffle_1000.json

# 上传 items_shuffle.json
modelscope upload \
  --repo-type model \
  --commit-message "upload items_shuffle.json" \
  afordb/sft_result_webshop_and_search_based_on_qwen \
  /diskpool/home/xuxz/data/items_shuffle.json \
  dataset/webshop/items_shuffle.json