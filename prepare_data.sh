# # We only use data preparation to indicate the modality and the data size.
export CUDA_VISIBLE_DEVICES=5
echo GPU:$CUDA_VISIBLE_DEVICES

train_data_size=16
val_data_size=128
group_size=8

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size