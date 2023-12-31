import os
import torch
import json
from transformers import BitsAndBytesConfig, AutoTokenizer, \
    AutoModelForCausalLM, TrainingArguments, Trainer
from peft import (LoraConfig, TaskType, get_peft_model,
                  prepare_model_for_int8_training)
from torch.utils.data import DataLoader
from utils import set_seed, print_args, load_pickle
from data import load_all_samples
import matplotlib.pyplot as plt
from llama_tokenize import Llama_dataset, Llama_with_sent_ids_dataset, Llama_next_word_dataset
from llama_collator import LLama_DataCollatorForLanguageModeling, DataCollatorForLanguageModeling
import random
import pickle
import argparse
from tqdm import tqdm
#! you need pip install accelerate bitsandbytes

def generate_and_collect_info(trainer, dev_loader, tokenizer, device, lora_dict, unshuffled_dev_samples):
    """
    Generate text sequences using a trained model and save information about the generated sequences.

    Args:
        trainer (Trainer): The model trainer.
        dev_loader (DataLoader): DataLoader for the development dataset.
        tokenizer (PreTrainedTokenizer): Tokenizer for encoding/decoding text.
        device (torch.device): Device to run inference on (e.g., 'cuda' or 'cpu').
        lora_dict (dict): Dict with keys 'rank','lora_alpha','lora_dropout'. Useful for naming the pickle files
        unshuffled_dev_samples (list):

     Returns:
        all_generated_info (dict): A dictionary containing information about the generated sequences. Each entry in the dictionary represents a
         generated sequence and includes the following key-value pairs:
            a. gen_text (str): The generated text sequence.
            b. perpl (float): The perplexity score associated with the generated sequence.
            c. without_dummy (int): An indicator specifying whether the generated sequence contains dummy tokens.
            d. true_label (str): The true label of the input sentence, if available.
            
    Generates text sequences using the provided model and collects information about each generated sequence,
    including the sentence ID, generated text, and perplexity. 

    Note:
        - The generated sequences are limited to a maximum of 30 new tokens.
        - The generated information is saved to pickle files with specific names based on the values of lora_dict.
    """
    all_generated_info = {} 
    context_header = '--------BELOW IS THE CONTEXT HISTORY--------'
    generated_header = '--------BELOW IS THE GENERATED TEXT--------'
    epsilon = 1e-10
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dev_loader)):
            inputs = {key: value.to(device) for key, value in batch.items() if key not in ['sentences_id','without_dummy','labels']} #sentence_id is useful only for metrics
   
            outputs_ids = trainer.model.generate( 
                **inputs,
                max_new_tokens=30,
                output_scores=True,
                return_dict_in_generate=True,
                top_p = lora_dict['top_p'])

            generated_tokens_ids = outputs_ids['sequences'] #(batch_size, input_length+max_new_tokens)
            generated_scores = outputs_ids['scores'] #it's a tuple of len max_new_tokens where each (batch_size, vocab_size)
            
            # Convert the tuple of tensors into a single tensor with dimensions (max_new_tokens, batch_size, vocab_size)
            scores_tensor = torch.stack(generated_scores)

            # Convert the scores into probabilities (max_new_tokens, batch_size, vocab_size)
            probs = torch.nn.functional.softmax(scores_tensor, dim=-1)

            # Make sure generated_tokens_ids is a tensor of shape (batch_size, input_length + max_new_tokens)
            generated_tokens_ids = torch.tensor(generated_tokens_ids)

            until = int(probs.size(0))
            # Gather the probabilities of the generated tokens probs: (batch_size, generated_timesteps=29, vocab) gen (batch_size, generated_timesteps=30, 1)
            actual_probs = torch.gather(probs.permute(1,0,2), 2, generated_tokens_ids[:,-until:].unsqueeze(-1)).squeeze(dim=-1)
            
            actual_probs = actual_probs.clamp(min=epsilon)
            # Calculate the negative log likelihood for each sequence in the batch
            is_not_pad_id = generated_tokens_ids[:, -until:] != tokenizer.pad_token_id # batch_size, 30 with binary flags
            negative_log_prob = -torch.log(actual_probs) # batch_size, 30
            log_prob_after_mask =  negative_log_prob * is_not_pad_id # batch_size, 30 where we masked if we have padding id
            non_pad_nll_per_sequence = torch.sum(log_prob_after_mask,dim=-1) # batch_size same as nll_per_sequence if no pad tokens
            count_not_padd = torch.count_nonzero(is_not_pad_id, dim=1)
            if torch.any(count_not_padd == 0): continue # for safety if it only generates the pad token!  
            mean_non_pad_nll_per_sequence = non_pad_nll_per_sequence / count_not_padd

            # nll_per_sequence = negative_log_prob.sum(dim=1) #! what we had initially

            perplexity = torch.exp(mean_non_pad_nll_per_sequence)

            
            batch_generated_text = tokenizer.batch_decode(generated_tokens_ids[:, -until:], skip_special_tokens=True)
            batch_input_text = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)

            if i % 20 == 0:
                output = [
                    f'{context_header}\n{input_}\n{generated_header}\n{gen}\n'
                    for input_, gen in zip(batch_input_text, batch_generated_text)
                ]

                print('\n'.join(output))

            batch_generated_info = {
                  sent_id.item(): {
                      'gen_text': gen_text,
                      'perpl': perpl.item(),
                      'without_dummy': without_dummy.item(),
                  }
              
              for gen_text, perpl, sent_id, without_dummy
              in zip(batch_generated_text, perplexity, batch['sentences_id'], batch['without_dummy'])
            }
            all_generated_info.update(batch_generated_info)

    # now we will go the dict we created and add there the true label as well
    for initial_sent_id in all_generated_info:
        
        answers_id, options, article = unshuffled_dev_samples[initial_sent_id]
        true_label_txt = options[answers_id]

        all_generated_info[initial_sent_id]['true_label'] = true_label_txt

    # for path name
    formatted_pairs = [f"{key}_{value}" for key, value in lora_dict.items()]
    result_string = "_".join(formatted_pairs)

    directory_path = 'generated_text' # os.path.join('generated_text', f'inference_{result_string}.pkl')
    if not os.path.exists(directory_path):
    # If it doesn't exist, create it along with any intermediate directories
        os.makedirs(directory_path)
    # Save index_list to a binary file
    save_file_path = os.path.join(directory_path, f'inference_{result_string}.pkl')
    with open(save_file_path, 'wb') as file:
        pickle.dump(all_generated_info, file)

    return all_generated_info

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
    """
    Creates and configures a tokenizer for natural language processing tasks.

    Args:
        config (dict): A dictionary containing configuration parameters for the tokenizer.
        MY_TOKEN (str): The huggingface custom token to be used in the tokenizer.

    Returns:
        tokenizer: An instance of a tokenizer configured according to the provided parameters.
    """
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'], use_fast=True, token=MY_TOKEN)

    # tokenizer info
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer



def main(args):
    print_args(args)
    config = vars(args) # convert to dict

    llama_eval_indices = load_pickle('eval_ids_llama.pkl') #4500 from 6k
    indexed_random_list = load_pickle('random_ids.pkl') #6k

    llama_train_indices = [] #1500
    for id in indexed_random_list:
        if id not in llama_eval_indices:
            llama_train_indices.append(id)

    base_dir = os.path.join(config['data_dir'], config['dataset_name'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device,' device!!!!')
    set_seed(config['seed'])
    MY_TOKEN = config['hf_token']

    train_samples = load_all_samples(base_dir, 'train')

    tokenizer = create_tokenizer(config, MY_TOKEN)

    shuffled_train_samples = [train_samples[i] for i in llama_train_indices]
    shuffled_dev_samples = [train_samples[i] for i in llama_eval_indices]

    kwargs = {} # different argument based on whether we use 4bits or 8bits
    optim = "paged_adamw_32bit" if config['bits'] ==4 and not config['debug'] else "adamw_torch" # default
    if not config['debug']:
        if config['bits']==4:
            bnb_config = get_quantize_4bits_config()
            kwargs['quantization_config'] = bnb_config
        elif config['bits']==8: 
            kwargs['load_in_8bit'] =True
        elif config['bits']==16:
            kwargs['torch_dtype'] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(config['model_name'], token = MY_TOKEN, device_map="auto", **kwargs)
    else:
        print('We are in debug mode so we take only the first few sentences')
        config['batch_size'] = 2
        if config['bits']==8: 
            kwargs['load_in_8bit'] =True
        elif config['bits']==16:
            kwargs['torch_dtype'] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained('gpt2', device_map="auto",**kwargs)

    if config['use_context']:
        train_dataset = Llama_next_word_dataset(tokenizer, shuffled_train_samples, do_generate=False, use_context=config['use_context'])
        dev_dataset = Llama_next_word_dataset(tokenizer, shuffled_dev_samples, do_generate=False,use_context=config['use_context'])
        train_collate_fn = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8, mlm=False)
        dev_collate_fn = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8, mlm=False)
    else:
        train_dataset = Llama_dataset(tokenizer, shuffled_train_samples, do_generate=False, use_context=config['use_context'])
        dev_dataset = Llama_dataset(tokenizer, shuffled_dev_samples, do_generate=False,use_context=config['use_context'])
        train_collate_fn = LLama_DataCollatorForLanguageModeling(tokenizer=tokenizer, pad_to_multiple_of=8, mlm=False)
        dev_collate_fn = LLama_DataCollatorForLanguageModeling(tokenizer=tokenizer, pad_to_multiple_of=8, mlm=False)
        

    # Resize token embeddings to accommodate the pad_token
    model.resize_token_embeddings(model.config.vocab_size + 1) # because we added pad_token

    # Prepare the model for INT8 training if not using 4-bit quantization
    if config['bits'] == 8:
        model = prepare_model_for_int8_training(model)

    # here we define the modules we use for LORA
    target_modules = ["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"] if 'llama' in config['model_name'] and not config['debug'] else None

    # Configure LORA (Loose Rank Attention) model
    peft_config = LoraConfig(
                    task_type="CAUSAL_LM", inference_mode=False, r=config['rank'], lora_alpha=config['lora_alpha'], lora_dropout=config['lora_dropout'],
                    bias = 'none',
                    target_modules=target_modules
                    ) #! maybe we should add target_modules but I am not sure that the allowed values are the same for every model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    dict_info_for_path = {key: config[key] for key in ['rank','lora_alpha','lora_dropout']} # to be used when saving the pickle during inference
    dict_info_for_path['loss'] = 'all' if config['use_context'] else 'labels' 
    dict_info_for_path['lr'] = config['learning_rate']
    dict_info_for_path['top_p'] = config['top_p']
    
    if config['bits']==16:
        bf=True
    else:
        bf=False
    training_arguments = TrainingArguments(
        output_dir=config['out_dir'],
        per_device_train_batch_size=config['train_batch_size'],
        num_train_epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        warmup_ratio=config['warmup_steps'],
        load_best_model_at_end= True,
        weight_decay=config['weight_decay'],
        save_total_limit = 1,
        seed=config['seed'],
        evaluation_strategy='epoch',
        save_strategy ='epoch',
        metric_for_best_model="loss",
        optim= optim,
        greater_is_better = False,
        bf16=bf # from https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama_2/scripts/sft_llama2.py#L101
    )


    trainer = Trainer(model, training_arguments, train_dataset=train_dataset,eval_dataset=dev_dataset, data_collator = train_collate_fn,
            tokenizer=tokenizer)
  
    if config['do_train']: 
        print('Training...')
        trainer.train()
        dict_info_for_path['train'] = 'finetuned'
    else:
        dict_info_for_path['train'] = 'not_finetuned'
    trainer.model.eval()

    dev_dataset = Llama_with_sent_ids_dataset(tokenizer, shuffled_dev_samples, do_generate=True,dev_ids= llama_eval_indices, use_context=config['use_context'])
    dev_collate_fn = LLama_DataCollatorForLanguageModeling(tokenizer=tokenizer, pad_to_multiple_of=8, mlm=False)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=16, collate_fn=dev_collate_fn)

    # Perform inference and collect information
    print('Inference...')
    all_generated_info = generate_and_collect_info(trainer, dev_loader, tokenizer, device,dict_info_for_path, train_samples)

def load_config(path):
    # Open and read the JSON file
    with open(path, 'r') as file:
        config = json.load(file)
    return config

#! In gpu without any arguments only 0-shot evalutation will be done and in 8bits. You won't be in debug mode
def parse_option():
    parser = argparse.ArgumentParser(description="LLama parser")
    
    #! If you want to train add --do_train, if you want to use 4bits pass also --do_train to train --use context
    parser.add_argument('--debug', action='store_true',help='store_true default is false. If you pass --debug it will be true')# store_false, store_true
    parser.add_argument('--use_context', action='store_true',help='store_true default is false.')# store_false, store_true
    parser.add_argument('--do_train', action='store_true',help='default is doing only evaluation')
    # general info
    parser.add_argument('--hf_token', type=str, default='hf_token', help='You can get it from here https://huggingface.co/settings/tokens')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--tokenizer_name', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--dataset_name', type=str, default='mutual')
    parser.add_argument('--finetune_size', type=int, default=1500)
    # directories
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--out_dir', type=str, default='checkpoints')
    # model hyperparams
    parser.add_argument('--max_seq_length', type=int, default=256)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=1.0)
    # lora hyperparams for parameter efficient finetuning
    parser.add_argument('--rank', type=int, default=16, help='The bigger, the better, as it allows us to update more parameters, but it also increases memory usage.') 
    parser.add_argument('--lora_alpha', type=int, default=8)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--bits', type=int,default=8)

    # Parse the command line arguments
    parsed_args = parser.parse_args()
    return parsed_args

if __name__ == "__main__":
    parsed_args = parse_option()
    main(parsed_args)