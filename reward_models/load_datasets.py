import numpy as np
import os
import torch
import torch.nn as nn
from datasets import load_dataset, concatenate_datasets

# for vanilla chosen and reject style dataset, such as dendrydong/preference_700K
def build_dataset(data_path, tokenizer, split='train', size=None, model_name=''):
    ds = load_dataset(data_path, split=split)
    
    if size is not None:
        ds = ds.select(range(0, size))

    def formatting_func(example):
        kwargs = {"padding": True, "truncation": True, "max_length": tokenizer.max_length, "return_tensors": "pt"}
        chosen_messages = example['chosen']
        rejected_messages = example['rejected']
        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)

        if 'GRM' in model_name:
            # add label mask for sft and dpo training
            prompt = example['chosen'][:-1]
            prompt_template = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            tokens_prompt = tokenizer.encode_plus(prompt_template, **kwargs)['input_ids'][0]
            label_chosen = tokens_chosen["input_ids"][0].clone()
            label_chosen[:len(tokens_prompt)] = -100
            label_rejected = tokens_rejected["input_ids"][0].clone()
            label_rejected[:len(tokens_prompt)] = -100
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "label_chosen": label_chosen,  'label_rejected': label_rejected
            }
        else:
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
            }

    ds = ds.map(formatting_func, batched=False, num_proc=10) 
    remove_columns = []
    for col in ds.column_names:
        if 'input' not in col and 'attention' not in col and 'label' not in col:
            remove_columns.append(col)
    ds = ds.remove_columns(remove_columns)

    ds.set_format(type="torch")
    return ds


# for UnifiedFeedback
def build_dataset_UF(data_path, tokenizer, split='train', size=None, mode='', model_name=''):
    try:
        ds = load_dataset(data_path, 'all', split=split)
    except:
        ds = load_dataset(data_path, split=split)
    
    # filter data with the same rating
    ds = ds.filter(lambda example: example['conv_A_rating'] != example['conv_B_rating'], num_proc=30)

    if len(mode):
        if mode == '40k' or mode == '40K':
            ds = ds.select(range(0, len(ds), 20)) 
        elif mode == '400k' or mode == '400K':
            ds = ds.select(range(0, len(ds), 2)) 

    if size is not None:
        ds = ds.select(range(0, size))

    def formatting_func(example):
        kwargs = {"padding": True, "truncation": True, "max_length": tokenizer.max_length, "return_tensors": "pt"}
        if example['conv_A_rating'] > example['conv_B_rating']:
            chosen_messages = example['conv_A']
            rejected_messages = example['conv_B']
            margin = example['conv_A_rating'] - example['conv_B_rating']
        else:
            chosen_messages = example['conv_B']
            rejected_messages = example['conv_A']
            margin = example['conv_B_rating'] - example['conv_A_rating']
        
        if 'summarize' in example['source']:
            chosen_messages[0]['content'] = 'Generate one-sentence summary for the following post: ' + chosen_messages[0]['content'].strip()
            rejected_messages[0]['content'] = 'Generate one-sentence summary for the following post: ' + rejected_messages[0]['content'].strip()
        
        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
        if 'GRM' in model_name:
            # add label mask for sft and dpo training
            prompt = [example['conv_A'][0]]
            prompt_template = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            tokens_prompt = tokenizer.encode_plus(prompt_template, **kwargs)['input_ids'][0]
            label_chosen = tokens_chosen["input_ids"][0].clone()
            label_chosen[:len(tokens_prompt)] = -100
            label_rejected = tokens_rejected["input_ids"][0].clone()
            label_rejected[:len(tokens_prompt)] = -100
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "label_chosen": label_chosen,  'label_rejected': label_rejected,
                # "margin": margin, # GRM does not need this
            }
        else:
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "margin": margin, 
            }
        

    ds = ds.map(formatting_func, batched=False, num_proc=10)
    # ds = ds.filter(lambda x: len(x["input_ids_chosen"]) <= script_args.max_length and len(x["input_ids_rejected"]) <= script_args.max_length, num_proc=30)
    remove_columns = []
    for col in ds.column_names:
        if 'input' not in col and 'attention' not in col and 'margin' not in col and 'label' not in col:
            remove_columns.append(col)
    ds = ds.remove_columns(remove_columns)

    ds.set_format(type="torch")
    return ds


# for Skywork Reward Preference 80K
def build_dataset_SK(data_path, tokenizer, split='train', size=None, model_name=''):
    ds = load_dataset(data_path, split=split)

    if size is not None:
        ds = ds.select(range(0, size))

    def formatting_func(example):
        kwargs = {"padding": True, "truncation": True, "max_length": tokenizer.max_length, "return_tensors": "pt"}
        prompt = example['chosen'][0]['content']

        chosen_messages = example['chosen']
        rejected_messages = example['rejected']

        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
        if 'GRM' in model_name:
            # add label mask for sft and dpo
            prompt_template = tokenizer.apply_chat_template([{"content": prompt, "role": "user" }], tokenize=False, add_generation_prompt=True)
            tokens_prompt = tokenizer.encode_plus(prompt_template, **kwargs)['input_ids'][0]
            label_chosen = tokens_chosen["input_ids"][0].clone()
            label_chosen[:len(tokens_prompt)] = -100
            label_rejected = tokens_rejected["input_ids"][0].clone()
            label_rejected[:len(tokens_prompt)] = -100
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "label_chosen": label_chosen,  'label_rejected': label_rejected
            }
        else:
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
            }

    ds = ds.map(formatting_func, batched=False, num_proc=10) 
    ds.set_format(type="torch")
    return ds

#New for MT_Bench
def build_dataset_MT(data_path, tokenizer, split='train', size=None, mode='', model_name='', pair='', num_human=-1):
    ds = load_dataset(data_path, split=split)
        
    # filter data with the same rating
    ds = ds.filter(lambda example: example['winner'] != 'tie', num_proc=30)
    
    if size is not None:
        ds = ds.select(range(0, size))
        
    if pair:
        model1, model2 = pair.split(',')
        ds = ds.filter(lambda x: (x['model_a']==model1 and x['model_b']==model2) or (x['model_a']==model2 and x['model_b']==model1), num_proc=10)
        
    # if num_human >= 0:
    #     ds = ds.select(np.random.randint(0, len(ds), size=num_human))

    def formatting_func(example):
        kwargs = {"padding": True, "truncation": True, "max_length": tokenizer.max_length, "return_tensors": "pt"}
        # chosen_a = ds.filter(lambda example: example['winner'] == 'model_a', num_proc=30).rename_column("conversation_a", "chosen").rename_column("conversation_b", "rejected")
        # chosen_b = ds.filter(lambda example: example['winner'] == 'model_b', num_proc=30).rename_column("conversation_b", "chosen").rename_column("conversation_a", "rejected")
        # chosen_messages = concatenate_datasets([chosen_a['chosen'], chosen_b['chosen']])
        chosen = 'conversation_a' if example['winner'] == 'model_a' else 'conversation_b'
        rejected = 'conversation_b' if example['winner'] == 'model_a' else 'conversation_a'
        chosen_messages = example[chosen]
        rejected_messages = example[rejected]
        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)

        if 'GRM' in model_name:
            # add label mask for sft and dpo training
            prompt = example[chosen][:-1]
            prompt_template = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            tokens_prompt = tokenizer.encode_plus(prompt_template, **kwargs)['input_ids'][0]
            label_chosen = tokens_chosen["input_ids"][0].clone()
            label_chosen[:len(tokens_prompt)] = -100
            label_rejected = tokens_rejected["input_ids"][0].clone()
            label_rejected[:len(tokens_prompt)] = -100
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "label_chosen": label_chosen,  'label_rejected': label_rejected
            }
        else:
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
            }

    ds = ds.map(formatting_func, batched=False, num_proc=10) 
    remove_columns = []
    for col in ds.column_names:
        if 'input' not in col and 'attention' not in col and 'label' not in col:
            remove_columns.append(col)
    ds = ds.remove_columns(remove_columns)

    ds.set_format(type="torch")
    return ds

def load_train_eval_dataset(data_path, tokenizer, size=None, mode='', model_name='', pair='', num_human=-1):
    if 'Unified' in data_path:
        # mode is only used for loading training data
        train_dataset = build_dataset_UF(data_path, tokenizer, split='train', size=size, mode=mode, model_name=model_name) 
        eval_dataset = build_dataset_UF(data_path, tokenizer, split='val', model_name=model_name)
    elif 'Skywork' in data_path:
        dataset = build_dataset_SK(data_path, tokenizer, split='train', size=size, model_name=model_name)
        dataset_split = dataset.train_test_split(test_size=0.005)
        train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']
    #New
    elif 'mt_bench' in data_path:
        dataset = build_dataset_MT(data_path, tokenizer, split='human', size=size, model_name=model_name, pair=pair, num_human=num_human)
        #dataset_split = dataset.train_test_split(test_size=0.01)
        #train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']
        eval_dataset = dataset
        train_dataset = dataset.select(np.random.randint(0, len(dataset), size=num_human)) if num_human >= 0 else dataset
    #End
    else:
        dataset = build_dataset(data_path, tokenizer, split='train', size=size, model_name=model_name) 
        dataset_split = dataset.train_test_split(test_size=0.01)
        train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']
    return train_dataset, eval_dataset
