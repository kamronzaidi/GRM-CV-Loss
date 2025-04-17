devices=1,2,3
n_gpu=3
dataset_name='lmsys/mt_bench_human_judgments'
dataset_mode='40K'
base_model='Ray2333/GRM-Gemma2-2B-sftreg'
wandb_name="GRM_pairs" #"GRM_seed1"
log_dir='../save_reward_models'
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

pair='alpaca-13b,gpt-3.5-turbo'
num_human=144

cd ../reward_models
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port} run_grm_reward_train.py \
    --base_model ${base_model}  --wandb_name ${wandb_name}   --log_dir ${log_dir} \
    --num_train_epochs ${num_train_epochs} \
    --max_length ${max_length} \
    --use_lora True \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${learning_rate} \
    --dataset ${dataset_name} --dataset_mode ${dataset_mode} \
    --weight_ratio ${weight_ratio}  --layer_type ${layer_type} \
    --reference_free ${reference_free} --sft_only ${sft_only} \
    --pair ${pair} --num_human ${num_human}
