pairs=('alpaca-13b,gpt-3.5-turbo' 'alpaca-13b,vicuna-13b-v1.2' 'alpaca-13b,claude-v1' 'claude-v1,llama-13b' 'claude-v1,gpt-3.5-turbo' 'gpt-3.5-turbo,vicuna-13b-v1.2' 'gpt-4,llama-13b' 'gpt-4,vicuna-13b-v1.2' 'llama-13b,vicuna-13b-v1.2' 'alpaca-13b,llama-13b' 'claude-v1,gpt-4' 'gpt-3.5-turbo,gpt-4' 'gpt-3.5-turbo,llama-13b' 'alpaca-13b,gpt-4' 'claude-v1,vicuna-13b-v1.2')
num_humans=(48 96 144 192 240 288 336 384) #48 96 144 192 240 288 336 384 432 480 528 576 624 672 720 768 #72 60 #

## Train
devices=1,2,3
n_gpu=3
dataset_name='lmsys/mt_bench_human_judgments'
dataset_mode='40K'
base_model='Ray2333/GRM-Gemma2-2B-sftreg'
wandb_name="GRM_pairs" #"GRM_seed1"
log_dir_train='../save_reward_models'
main_process_port=9995

learning_rate=1e-5
lora_r=32
lora_alpha=64
max_length=1024
num_train_epochs=2
gradient_accumulation_steps=4

weight_ratio=0.01
layer_type='mlp' 
sft_only=True
reference_free=True

## Eval
gpu='3'
port=8992
per_device_eval_batch_size=1
#peft_name='../save_reward_models/GRM-Gemma2-2B-sftreg_GRM_pairs_len1024_lora32_1e-05_datamt_bench_human_judgments/logs/checkpoint-1' #'../save_reward_models/GRM-Gemma2-2B-sftreg_GRM_pair_only_len1024_lora32_1e-05_datamt_bench_human_judgments/logs/checkpoint-8'
num_layers=1
log_dir_eval='./eval_GRM'
save_all_data=True


for num_human in "${num_humans[@]}"; do
    for pair in "${pairs[@]}"; do

        rm -rf /zfsauton2/home/kzaidi/10707/Generalizable-Reward-Model/save_reward_models/GRM-Gemma2-2B-sftreg_GRM_pairs_len1024_lora32_1e-05_datamt_bench_human_judgments

        cd ../reward_models
        CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port} run_grm_reward_train.py \
            --base_model ${base_model}  --wandb_name ${wandb_name}   --log_dir ${log_dir_train} \
            --num_train_epochs ${num_train_epochs} \
            --max_length ${max_length} \
            --use_lora True \
            --gradient_accumulation_steps ${gradient_accumulation_steps} \
            --learning_rate ${learning_rate} \
            --dataset ${dataset_name} --dataset_mode ${dataset_mode} \
            --weight_ratio ${weight_ratio}  --layer_type ${layer_type} \
            --reference_free ${reference_free} --sft_only ${sft_only} \
            --pair ${pair} --num_human ${num_human}

        cd ../rm_eval
        peft_name=$(ls -dt ../save_reward_models/GRM-Gemma2-2B-sftreg_GRM_pairs_len1024_lora32_1e-05_datamt_bench_human_judgments/logs/*/ | head -n 1)
        CUDA_VISIBLE_DEVICES=${gpu} accelerate launch --main_process_port ${port} eval_grm.py --base_model ${base_model} --peft_name ${peft_name} \
                                             --per_device_eval_batch_size ${per_device_eval_batch_size} \
                                             --max_length ${max_length} --log_dir ${log_dir_eval} --save_all_data ${save_all_data} \
                                              --task mtbench --layer_type ${layer_type} --num_layers ${num_layers} --pair ${pair} --affix ${num_human}
    done
done