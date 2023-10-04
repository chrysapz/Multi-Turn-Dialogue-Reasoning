import os
import torch
import numpy as np
import json
from transformers import AutoModelForSequenceClassification, AdamW, AutoTokenizer, \
    AutoModelForMultipleChoice, DataCollatorForLanguageModeling, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import (LoraConfig, TaskType, get_peft_model,
                  prepare_model_for_int8_training)
from torch.utils.data import DataLoader
from utils import set_seed, get_checkpoint_name
from data import load_all_samples
from mutual_dataset import MutualDataset
import matplotlib.pyplot as plt
from llama_tokenize import Llama_dataset
from llama_collator import LLama_DataCollatorForLanguageModeling
import random
import pickle
#! you need pip install accelerate bitsandbytes

def main(config):
    print('Training')
    print(config)
    base_dir = os.path.join(config['data_dir'], config['dataset_name'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device,' device!!!!')
    set_seed(config['seed'])
    MY_TOKEN = 'put your code'
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'], use_fast=True, token=MY_TOKEN)

    # tokenizer info
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_samples = load_all_samples(base_dir, 'train')
    dev_samples = load_all_samples(base_dir, 'dev')

    indexed_train_list = [i for i in range(len(train_samples))]
    
    random.shuffle(indexed_train_list)
    shuffled_samples = [train_samples[i] for i in indexed_train_list]
    FINETUNE_SIZE = 1000
    train_samples = shuffled_samples[:FINETUNE_SIZE]
    dev_samples = shuffled_samples[FINETUNE_SIZE:FINETUNE_SIZE+FINETUNE_SIZE]

    # Save index_list to a binary file
    with open('index_list.pkl', 'wb') as file:
        pickle.dump(indexed_train_list, file)

    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    if config['debug']:
        print('We are in debug so we take only the first sentence')
        train_samples =  train_samples[:10]
        dev_samples =  dev_samples[:10]
        config['batch_size'] = 2
        model = AutoModelForCausalLM.from_pretrained('gpt2', load_in_8bit=True, device_map="auto")
    #! todo add attention_masks
    train_dataset = Llama_dataset(tokenizer, train_samples)
    dev_dataset = Llama_dataset(tokenizer, dev_samples)

    if not config['debug']:
        model = AutoModelForCausalLM.from_pretrained(config['model_name'], token = MY_TOKEN, load_in_8bit=True, device_map="auto")
    model = model.to(device)
    model.resize_token_embeddings(model.config.vocab_size + 1)

    # #quaa
    model = prepare_model_for_int8_training(model)

    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'] if 'llama' in config['model_name'] and not config['debug'] else None # edit with your desired target modules
    peft_config = LoraConfig(
                    task_type="CAUSAL_LM", inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
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
        evaluation_strategy="steps", #epoch
        save_strategy ='steps', #epoch
        metric_for_best_model="eval_loss"
    )

    trainer = Trainer(model, training_arguments, train_dataset=train_dataset, eval_dataset=dev_dataset, data_collator = train_collate_fn,
            tokenizer=tokenizer)
    print('Training...')
    trainer.train()


    # model.eval()

    # DEV_BATCH_SIZE = 1
    # dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=DEV_BATCH_SIZE, collate_fn=dev_collate_fn)
    # # Calculate perplexity
    # generated_info = {'sentence_id':[], 'generated_ids':[], 'perplexity':[]} #(sentence_id, generated_ids, perplexity_of_generated)
    # with torch.no_grad():
    #     for batch in dev_loader:
    #         inputs = {key: value.to(device) for key, value in batch.items() if key not in ['sentence_id']} #sentence_id is useful only for metrics
    #         # inputs.pop('labels')
    #         # note that the difference between input_ids and labels is that in labels we have -100 in ignore tokens
    #         outputs_ids = model.generate( #! maybe add trainer.model?
    #             **inputs,
    #             max_new_tokens=30,
    #             output_scores=True,
    #             return_dict_in_generate=True,
    #             temperature = 1
    #         )
    #         whole_sequences_ids = outputs_ids['sequences'] #(batch_size, input_length+max_new_tokens)
    #         generated_scores = outputs_ids['scores'] #it's a tuple of len max_new_tokens where each (batch_size, vocab_size)
            
    #         output_text = tokenizer.decode(whole_sequences_ids[0], skip_special_tokens=True)

    #         #! not correct
    #         # Calculate cross-entropy loss for each sequence in the batch
    #         for i in range(len(whole_sequences_ids)):
    #             # Get the logits for the generated sequence
    #             generated_logits = generated_scores[i]

    #             # Calculate the cross-entropy loss
    #             loss = torch.nn.functional.cross_entropy(generated_logits, inputs['labels'][i])
    #             perplexity = torch.exp(loss).item()            
    #             # Print or store the loss for this sequence
    #             print(f"Loss for sequence {i}: {loss.item()}")

    #         if DEV_BATCH_SIZE == 1:
    #             generated_info['sentence_id'].append(batch['sentence_id'][0])
    #             generated_info['generated_ids'].append(outputs)
    #             generated_info['perplexity'].append(perplexity)
    #         else:
    #             raise ValueError('We do not support batch size > 1')




def load_config(path):
    # Open and read the JSON file
    with open(path, 'r') as file:
        config = json.load(file)
    return config

if __name__ == "__main__":
    config_path = os.path.join("conf", "config_llama.json")
    config = load_config(config_path)
    main(config)