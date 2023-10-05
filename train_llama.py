import os
import torch
import numpy as np
import json
from transformers import BitsAndBytesConfig, AutoTokenizer, \
    DataCollatorForLanguageModeling, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback
from peft import (LoraConfig, TaskType, get_peft_model,
                  prepare_model_for_int8_training)
from torch.utils.data import DataLoader
from utils import set_seed, get_checkpoint_name
from data import load_all_samples
from mutual_dataset import MutualDataset
import matplotlib.pyplot as plt
from llama_tokenize import Llama_dataset, Llama_with_sent_ids_dataset
from llama_collator import LLama_DataCollatorForLanguageModeling
import random
import pickle
import argparse
#! you need pip install accelerate bitsandbytes

def generate_and_collect_info(trainer, dev_loader, tokenizer, device, LORA_R):
    generated_info = [] # [ (sentence_id, generated_text, perplexity) ]
    with torch.no_grad():
        for batch in dev_loader:
            inputs = {key: value.to(device) for key, value in batch.items() if key not in ['sentences_id']} #sentence_id is useful only for metrics
            inputs.pop('labels')

            outputs_ids = trainer.model.generate( 
                **inputs,
                max_new_tokens=30,
                output_scores=True,
                return_dict_in_generate=True,
                temperature = 1
            )

            generated_tokens_ids = outputs_ids['sequences'] #(batch_size, input_length+max_new_tokens)
            generated_scores = outputs_ids['scores'] #it's a tuple of len max_new_tokens where each (batch_size, vocab_size)
            
            # Convert the tuple of tensors into a single tensor with dimensions (max_new_tokens, batch_size, vocab_size)
            scores_tensor = torch.stack(generated_scores)

            # Convert the scores into probabilities (max_new_tokens, batch_size, vocab_size)
            probs = torch.nn.functional.softmax(scores_tensor, dim=-1)

            # Make sure generated_tokens_ids is a tensor of shape (batch_size, input_length + max_new_tokens)
            generated_tokens_ids = torch.tensor(generated_tokens_ids)

            # Gather the probabilities of the generated tokens
            actual_probs = torch.gather(probs.permute(1,0,2), 2, generated_tokens_ids[:,-30:].unsqueeze(-1)).squeeze(dim=-1)

            # Calculate the negative log likelihood for each sequence in the batch
            nll_per_sequence = -torch.log(actual_probs).sum(dim=1)

            perplexity = torch.exp(nll_per_sequence)

            batch_generated_text = tokenizer.batch_decode(generated_tokens_ids[:, -30:], skip_special_tokens=True)

            generated_info.extend([(sent_id.item(), gen_text, perpl) for gen_text, perpl, sent_id in zip(batch_generated_text, perplexity, 
                                                                                                  batch['sentences_id'])])
    
    # Save index_list to a binary file
    with open(f'inference_info_r_{LORA_R}.pkl', 'wb') as file:
        pickle.dump(generated_info, file)

def print_args(args):
    # Step 4: Print the parsed arguments
    print("Parsed Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

def get_quantize_4bits_config():# Activate 4-bit precision base model loading
    use_4bit = True
    # Activate nested quantization for 4-bit base models
    use_nested_quant = False
    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"
    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"
    # Optimizer to use, original is paged_adamw_32bit
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    return bnb_config

def create_tokenizer(config, MY_TOKEN):
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'], use_fast=True, token=MY_TOKEN)

    # tokenizer info
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer



def main(args):
    print('Training')
    print_args(args)
    config = vars(args) # convert to dict
    base_dir = os.path.join(config['data_dir'], config['dataset_name'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device,' device!!!!')
    set_seed(config['seed'])
    MY_TOKEN = config['hf_token']

    train_samples = load_all_samples(base_dir, 'train')
    dev_samples = load_all_samples(base_dir, 'dev')

    tokenizer = create_tokenizer(config, MY_TOKEN)

    # shuffle data and take sublists
    indexed_train_list = [i for i in range(len(train_samples))]
    random.shuffle(indexed_train_list)
    shuffled_samples = [train_samples[i] for i in indexed_train_list]
    FINETUNE_SIZE = config['finetune_size']
    train_samples = shuffled_samples[:FINETUNE_SIZE]
    dev_samples = shuffled_samples[FINETUNE_SIZE:FINETUNE_SIZE+FINETUNE_SIZE]

    # Save index_list to a binary file
    with open('index_list.pkl', 'wb') as file:
        pickle.dump(indexed_train_list, file)

    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    train_dataset = Llama_dataset(tokenizer, train_samples, do_generate=False)
    print('dev')
    dev_dataset = Llama_dataset(tokenizer, dev_samples, do_generate=False)
    kwargs = {}
    optim = "paged_adamw_32bit" if config['quantize_4bits'] and not config['debug'] else "adamw_torch" # default
    if not config['debug']:
        if config['quantize_4bits']:
            bnb_config = get_quantize_4bits_config()
            kwargs['quantization_config'] = bnb_config
        else: 
            kwargs['load_in_8bit'] =True
        model = AutoModelForCausalLM.from_pretrained(config['model_name'], token = MY_TOKEN, device_map="auto", **kwargs)
    else:
        print('We are in debug mode so we take only the first few sentences')
        train_samples =  train_samples[:4]
        dev_samples =  dev_samples[:4]
        config['batch_size'] = 2
        model = AutoModelForCausalLM.from_pretrained('gpt2', device_map="auto", **kwargs)

        
    model = model.to(device)
    model.resize_token_embeddings(model.config.vocab_size + 1)

    if not config['quantize_4bits']:
        model = prepare_model_for_int8_training(model)

    # here we define the modules we use for LORA
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'] if 'llama' in config['model_name'] and not config['debug'] else None # edit with your desired target modules
    LORA_R = config['rank']
    peft_config = LoraConfig(
                    task_type="CAUSAL_LM", inference_mode=False, r=LORA_R, lora_alpha=config['lora_alpha'], lora_dropout=config['lora_dropout'],
                    bias = 'none',
                    target_modules=target_modules
                    ) #! maybe we should add target_modules but I am not sure that the allowed values are the same for every model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_collate_fn = LLama_DataCollatorForLanguageModeling(tokenizer=tokenizer, pad_to_multiple_of=8, mlm=False)#train_dataset.collate_fn
    dev_collate_fn = LLama_DataCollatorForLanguageModeling(tokenizer=tokenizer, pad_to_multiple_of=8, mlm=False)#dev_dataset.collate_fn

    training_arguments = TrainingArguments(
        output_dir=config['out_dir'],
        per_device_train_batch_size=config['batch_size'],
        num_train_epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        warmup_ratio=config['warmup_steps'],
        load_best_model_at_end= True,
        weight_decay=config['weight_decay'],
        save_total_limit = 1,
        seed=config['seed'],
        evaluation_strategy='epoch',
        save_strategy ='epoch',
        metric_for_best_model="eval_loss",
        optim= optim
    )


    trainer = Trainer(model, training_arguments, train_dataset=train_dataset, eval_dataset=dev_dataset, data_collator = train_collate_fn,
            tokenizer=tokenizer)
    print('Training...')
    if config['do_train']: trainer.train()
    
    trainer.model.eval()

    DEV_BATCH_SIZE = config['eval_batch_size']
    dev_dataset = Llama_with_sent_ids_dataset(tokenizer, dev_samples, do_generate=True,dev_ids= indexed_train_list[FINETUNE_SIZE:FINETUNE_SIZE+FINETUNE_SIZE])
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=DEV_BATCH_SIZE, collate_fn=dev_collate_fn)
    print('Inference...')
    generate_and_collect_info(trainer, dev_loader, tokenizer, device, LORA_R)

def load_config(path):
    # Open and read the JSON file
    with open(path, 'r') as file:
        config = json.load(file)
    return config

#! In gpu without any arguments only 0-shot evalutation will be done and in 8bits
def parse_option():
    parser = argparse.ArgumentParser(description="LLama parser")
    
    #! If you want to train add --do_train, if you want to use 4bits pass also --quantize_4bits
    parser.add_argument('--debug', action='store_true',help='store_true default is false. If you pass --debug it will be true')# store_false, store_true
    parser.add_argument('--do_train', action='store_true',help='default is doing only evaluation')
    # general info
    parser.add_argument('--hf_token', type=str, default='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', help='You can get it from here https://huggingface.co/settings/tokens')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--tokenizer_name', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--dataset_name', type=str, default='mutual')
    parser.add_argument('--finetune_size', type=int, default=1000)
    # directories
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--out_dir', type=str, default='checkpoints')
    # model hyperparams
    parser.add_argument('--max_seq_length', type=int, default=256)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=0)
    # lora hyperparams for parameter efficient finetuning
    parser.add_argument('--rank', type=int, default=16, help='The bigger, the better, as it allows us to update more parameters, but it also increases memory usage.') 
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--quantize_4bits', action='store_true',help='default is 8bit. If you run the script with --quantize_4bits it will be true else false')

    # Parse the command line arguments
    parsed_args = parser.parse_args()
    return parsed_args

if __name__ == "__main__":
    parsed_args = parse_option()
    main(parsed_args)