# modelscope login --token ms-c584573c-009e-45dc-b83d-52d52faa6ac3
# 上传 checkpoint-10500
modelscope upload \
  --repo-type model \
  --commit-message "upload checkpoint-10500 (search)" \
  --exclude='global_step*' \
  afordb/sft_result_webshop_and_search_based_on_qwen \
  /diskpool/home/xuxz/ms-swift/checkpoint_search/Qwen2.5-1.5B-Instruct-Parallel-Epoch5-hislen8_seed1_1000sample/v1-20260715-204318/checkpoint-10500 \
  search/3.5epoch_10500it

# 上传 checkpoint-14865
modelscope upload \
  --repo-type model \
  --commit-message "upload checkpoint-14865 (search)" \
  --exclude='global_step*' \
  afordb/sft_result_webshop_and_search_based_on_qwen \
  /diskpool/home/xuxz/ms-swift/checkpoint_search/Qwen2.5-1.5B-Instruct-Parallel-Epoch5-hislen8_seed1_1000sample/v1-20260715-204318/checkpoint-14865 \
  search/5epoch_14865it

# 上传 checkpoint-6250 (webshop)
modelscope upload \
  --repo-type model \
  --commit-message "upload checkpoint-6250 (webshop)" \
  --exclude='global_step*' \
  afordb/sft_result_webshop_and_search_based_on_qwen \
  /diskpool/home/xuxz/ms-swift/checkpoint/Qwen2.5-1.5B-Instruct-Parallel-Epoch5-hislen8/v0-20260602-201729/checkpoint-6250 \
  webshop/3.5epoch_6250it

# 上传 checkpoint-8800 (webshop)
modelscope upload \
  --repo-type model \
  --commit-message "upload checkpoint-8800 (webshop)" \
  --exclude='global_step*' \
  afordb/sft_result_webshop_and_search_based_on_qwen \
  /diskpool/home/xuxz/ms-swift/checkpoint/Qwen2.5-1.5B-Instruct-Parallel-Epoch5-hislen8/v0-20260602-201729/checkpoint-8800 \
  webshop/5epoch_8800it
