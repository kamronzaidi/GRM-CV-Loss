devices=0,1,2,3
n_gpu=4
main_process_port=9994



# Data Generation

python rlhf/data_generation/sample_dataset.py

CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port}  \
    rlhf/data_generation/obtain_gold_score.py \
    --per_device_batch_size 64 \
    --max_length 1024 \
    --data_path "rlhf/data_generation/data/unified_20k" \
    --model_path "Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback" \
    --save_path "rlhf/data/obtain_gold_score" \
    --save_name "unified_sampled_gold_score" \
    --mode "train" \
    --num_splits 6 
    
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port}  \
    rlhf/data_generation/obtain_gold_score.py \
    --per_device_batch_size 64 \
    --max_length 1024 \
    --data_path "rlhf/data_generation/data/unified_1k" \
    --model_path "Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback" \
    --save_path "rlhf/data/obtain_gold_score" \
    --save_name "unified_sampled_gold_score" \
    --mode "test" \
    --num_splits 1