
gpu='3'
port=8992
max_length=1024
per_device_eval_batch_size=1
base_model="Ray2333/GRM-Gemma2-2B-sftreg"
peft_name='../save_reward_models/GRM-Gemma2-2B-sftreg_GRM_notrain_len1024_lora32_0.0_datamt_bench_human_judgments/logs/checkpoint-1' #'../save_reward_models/GRM-Gemma2-2B-sftreg_GRM_pair_only_len1024_lora32_1e-05_datamt_bench_human_judgments/logs/checkpoint-8'
layer_type='mlp' # linear
num_layers=1
log_dir='./eval_GRM'
save_all_data=True
pair='alpaca-13b,gpt-3.5-turbo'
affix='0'

cd ../rm_eval
CUDA_VISIBLE_DEVICES=${gpu} accelerate launch --main_process_port ${port} eval_grm.py --base_model ${base_model} --peft_name ${peft_name} \
                                             --per_device_eval_batch_size ${per_device_eval_batch_size} \
                                             --max_length ${max_length} --log_dir ${log_dir} --save_all_data ${save_all_data} \
                                              --task mtbench --layer_type ${layer_type} --num_layers ${num_layers} --pair ${pair} --affix ${affix}



