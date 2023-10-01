import os
import torch
import numpy as np
import json
from transformers import AutoModelForSequenceClassification, AdamW, AutoTokenizer, \
    AutoModelForMultipleChoice, DataCollatorForLanguageModeling, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorWithPadding
from peft import (LoraConfig, TaskType, get_peft_model,
                  prepare_model_for_int8_training)
from torch.utils.data import DataLoader
from utils import set_seed, get_checkpoint_name
from data import load_all_samples
from mutual_dataset import MutualDataset
from llama_mutual_dataset import Seperate_Context_Option_Dataset, Concat_History_Option_Dataset
import matplotlib.pyplot as plt

dict_input_label_format = {
    'concat': Concat_History_Option_Dataset,
    'seperate': Seperate_Context_Option_Dataset
}

def main(config):
    input_label_format = 'seperate'

    print('Training')
    print(config)
    base_dir = os.path.join(config['data_dir'], config['dataset_name'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config['seed'])

    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'], use_fast=True)

    train_samples = load_all_samples(base_dir, 'train')
    dev_samples = load_all_samples(base_dir, 'dev')
    if config['debug']:
        print('We are in debug so we take only the first sentence')
        train_samples =  train_samples[:1]
        dev_samples =  dev_samples[:1]

    train_dataset = dict_input_label_format[input_label_format](train_samples, tokenizer, config['max_seq_length'])
    dev_dataset = dict_input_label_format[input_label_format](dev_samples, tokenizer, config['max_seq_length'])

    #! todo change config and use llama
    model = AutoModelForCausalLM.from_pretrained(config['model_name'], num_labels = 2)
    model = model.to(device)

    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'] if 'llama' in config['model_name'] else None # edit with your desired target modules
    peft_config = LoraConfig(
                    task_type="CAUSAL_LM", inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.05,
                    bias = 'none',
                    target_modules=target_modules
                    ) #! maybe we should add target_modules but I am not sure that the allowed values are the same for every model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    #! DataCollatorForLanguageModeling sets as labels the input and puts -100 in the padding token
    train_collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)#train_dataset.collate_fn
    dev_collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)#dev_dataset.collate_fn

    # # t_total = len(train_dataloader) * config['epochs']
    # # # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=t_total)

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


    model.eval()

    DEV_BATCH_SIZE = 1
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=DEV_BATCH_SIZE, collate_fn=dev_collate_fn)
    # Calculate perplexity
    generated_info = {'sentence_id':[], 'generated_ids':[], 'perplexity':[]} #(sentence_id, generated_ids, perplexity_of_generated)
    with torch.no_grad():
        for batch in dev_loader:
            inputs = {key: value.to(device) for key, value in batch.items() if key not in ['sentence_id']} #sentence_id is useful only for metrics
            # inputs.pop('labels')
            # note that the difference between input_ids and labels is that in labels we have -100 in ignore tokens
            outputs_ids = model.generate( #! maybe add trainer.model?
                **inputs,
                max_new_tokens=30,
                output_scores=True,
                return_dict_in_generate=True
                temperature = 1
            )
            whole_sequences_ids = outputs_ids['sequences'] #(batch_size, input_length+max_new_tokens)
            generated_scores = outputs_ids['scores'] #it's a tuple of len max_new_tokens where each (batch_size, vocab_size)
            
            output_text = tokenizer.decode(whole_sequences_ids[0], skip_special_tokens=True)

            #! not correct
            # Calculate cross-entropy loss for each sequence in the batch
            for i in range(len(whole_sequences_ids)):
                # Get the logits for the generated sequence
                generated_logits = generated_scores[i]

                # Calculate the cross-entropy loss
                loss = torch.nn.functional.cross_entropy(generated_logits, inputs['labels'][i])
                perplexity = torch.exp(loss).item()            
                # Print or store the loss for this sequence
                print(f"Loss for sequence {i}: {loss.item()}")

            if DEV_BATCH_SIZE == 1:
                generated_info['sentence_id'].append(batch['sentence_id'][0])
                generated_info['generated_ids'].append(outputs)
                generated_info['perplexity'].append(perplexity)
            else:
                raise ValueError('We do not support batch size > 1')




def load_config(path):
    # Open and read the JSON file
    with open(path, 'r') as file:
        config = json.load(file)
    return config

if __name__ == "__main__":
    config_path = os.path.join("conf", "config_llama.json")
    config = load_config(config_path)
    main(config)