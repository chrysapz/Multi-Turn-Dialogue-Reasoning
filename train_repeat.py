import os
import torch
import numpy as np
import json
from copy import deepcopy
from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
    DataCollatorWithPadding,  RobertaTokenizerFast
from torch.utils.data import DataLoader
from utils import set_seed, get_checkpoint_name
from data import load_all_samples
from mutual_dataset import MutualDataset
import matplotlib.pyplot as plt
from evaluate import evaluate_data, calculate_probs, group_data, sort_grouped_data
from time import gmtime, strftime
from tqdm import tqdm
from collections import defaultdict
from utils import repeat_golds_training_data, repeat_training_data_based_on_sim, load_pickle,  print_args, create_pickle, create_dicts_from_tuples, create_sub_dict
import pandas as pd
import argparse
from trainer import train



NUM_TRAIN_EXAMPLES = 6000
repeat_type = {'sim': repeat_training_data_based_on_sim,
   'gold': repeat_golds_training_data
}


def main(config):
    print('Training')
    print_args(args)
    config = vars(args) # convert to dict

    # config['sim']  =True
    # config['repeat_pickle'] = None#'sim_inference.pkl' #os.path.join('pickles','sim_augment_inference.pkl')
    # config['repeat_type']= 'gold'
    # config['debug'] = True
    # Set up the data directory and device
    base_dir = os.path.join(config['data_dir'], config['dataset_name'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device ',device)
    set_seed(config['seed'])

    checkpoint_name = get_checkpoint_name(config)
    out_dir = config['out_dir']
    save_folder = os.path.join(out_dir, checkpoint_name)
    if config['debug']:
        save_folder = os.path.join(save_folder, 'debug')

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Load the tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(config['tokenizer_name'], use_fast=True)

    # Load training and val data
    initial_train_samples = load_all_samples(base_dir, 'train')
    # Read the serialized data from the file and deserialize it
    train_random_indices = load_pickle('random_ids.pkl')
    val_random_indices = load_pickle('val_random_ids.pkl')

    train_samples = [initial_train_samples[i] for i in train_random_indices]
    dev_samples = [initial_train_samples[i] for i in val_random_indices]

    # shuffled_samples = [initial_train_samples[i] for i in indexed_train_list]

    # train_samples = shuffled_samples[:NUM_TRAIN_EXAMPLES]
    # train_random_indices = indexed_train_list[:NUM_TRAIN_EXAMPLES]
    train_id2history, train_id2options, train_id2label_id = create_dicts_from_tuples(train_samples, train_random_indices)

    # dev_samples = shuffled_samples[NUM_TRAIN_EXAMPLES:] 
    # val_random_indices = indexed_train_list[NUM_TRAIN_EXAMPLES:]
    val_id2history, val_id2options, val_id2label_id = create_dicts_from_tuples(dev_samples, val_random_indices)

    test_samples = load_all_samples(base_dir, 'dev')
    test_indices = list(range(len(test_samples)))
    test_id2history, test_id2options, test_id2label_id = create_dicts_from_tuples(test_samples, test_indices)


    if config['repeat_pickle'] is not None:
        preprocessed_generated_info = load_pickle(config['repeat_pickle'])

        if config['repeat_type']=='sim':
            train_id2options, train_id2label_id = repeat_training_data_based_on_sim(train_id2options, train_id2label_id, preprocessed_generated_info)
        elif config['repeat_type']=='gold': # we consider everything as gold
            train_id2options, train_id2label_id = repeat_golds_training_data(train_id2options, train_id2label_id, preprocessed_generated_info)

    if config['debug']:
        train_k_ids = [2650, 2868]
        test_k_ids = [2,1]
        val_k_ids = [2860,5806]
        config['epochs']=2

        train_id2history, train_id2options, train_id2label_id = create_sub_dict(train_id2history, train_id2options, train_id2label_id, train_k_ids)
        val_id2history, val_id2options, val_id2label_id = create_sub_dict(val_id2history, val_id2options, val_id2label_id, val_k_ids)
        test_id2history, test_id2options, test_id2label_id = create_sub_dict(test_id2history, test_id2options, test_id2label_id, test_k_ids)

    assert(len(train_id2history) == len(train_id2options) == len(train_id2label_id))
    assert(len(val_id2history) == len(val_id2options) == len(val_id2label_id))
    assert(len(test_id2history) == len(test_id2options) == len(test_id2label_id))

    print(f'len training set {len(train_id2history)} len val set {len(val_id2history)} len test set {len(test_id2history)}')

    # tokenize and create datasets for training and eval datat
    train_dataset = MutualDataset(train_id2history, train_id2options, train_id2label_id, tokenizer, config['max_seq_length'])
    dev_dataset = MutualDataset(val_id2history, val_id2options, val_id2label_id, tokenizer, config['max_seq_length'])
    test_dataset = MutualDataset(test_id2history, test_id2options, test_id2label_id, tokenizer, config['max_seq_length'])

    # create the model
    model = AutoModelForSequenceClassification.from_pretrained(config['model_name'], num_labels = 2)
    model = model.to(device)

    # Initialize collate functions and data loaders for training and development
    train_collate_fn = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config['batch_size'], collate_fn=train_collate_fn)

    dev_collate_fn = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=config['batch_size'], collate_fn=dev_collate_fn)

    dev_collate_fn = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=config['batch_size'], collate_fn=dev_collate_fn)

    # common trick applied also in the paper https://github.com/Nealcly/MuTual/blob/master/baseline/multi-choice/run_multiple_choice.py#L101
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay':config['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    #! they pass epsilon as an argument in their code. Maybe we can tune it at a later stage if our results deviate from theirs.
    # Initialize the optimizer

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config['learning_rate'], eps=config['adam_epsilon'])
    # t_total = len(train_dataloader) * config['epochs']
    # # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=t_total)
    # Train the model
    model_info, avg_loss, confidence, variability, correctness = train(model, train_loader, dev_loader, optimizer, config, device)

    # Save the model
    #! be cautious not to exhaust memory since we save it every time
    if not config['debug']:

        save_name = os.path.join(save_folder, 'model')

        torch.save(model_info, save_name)

        print('Test...')
        model = AutoModelForSequenceClassification.from_pretrained(config['model_name'], num_labels = 2)
        # load the model we just trained
        # checkpoint = torch.load(save_name)
        model.load_state_dict(model_info['model_state_dict'])
        model = model.to(device)
        preds, labels, avg_loss, metrics, grouped_data, labeled_data = evaluate_data(model, test_loader, config, device)

        # save loss
        out_path = os.path.join(save_folder, "training_loss.png")
        plt.plot(avg_loss)
        plt.savefig(out_path)

        # save pickles for confidence, variability and correctness
        path_confidence_pickle = os.path.join(save_folder, f'labeled_data.pkl')
        create_pickle(labeled_data, path_confidence_pickle)

        # save pickles for confidence, variability and correctness
        path_confidence_pickle = os.path.join(save_folder, f'confidence.pkl')
        create_pickle(confidence, path_confidence_pickle)

        path_variability_pickle = os.path.join(save_folder, f'variability.pkl')
        create_pickle(variability, path_variability_pickle)

        path_correctness_pickle = os.path.join(save_folder, f'correctness.pkl')
        create_pickle(correctness, path_correctness_pickle)

        path_probs_pickle = os.path.join(save_folder, f'dict_probs.pkl')
        create_pickle(grouped_data, path_probs_pickle)

        json_name = os.path.join(save_folder,'config.json')
        with open(json_name, "w") as json_file:
            json.dump(config, json_file)

def load_config(path):
    """
    Load a configuration from a JSON file.
    
    Args:
        path (str): The path to the JSON file containing the configuration data.
        
    Returns:
        dict: A dictionary containing the configuration data.
    """
    with open(path, 'r') as file:
        config = json.load(file)
    return config

def parse_option():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="My Argument Parser")

    # Add arguments with default values manually
    parser.add_argument("--mode", type=str, default="binary")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--dataset_name", type=str, default="mutual")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--tokenizer_name", type=str, default="roberta-base")
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--calculate_probs", type=bool, default=True)
    # parser.add_argument("--consider_only_gold", action='store_true',help='default is to consider the augmented labels as noisy')
    parser.add_argument('--debug', action='store_true',help='default is not to debug')
    parser.add_argument('--sim', action='store_true',help='default is not to filter using similarities')
    #todo
    # parser.add_argument('--repeat', type=int, default=0, help='default is not to repeat training data')
    parser.add_argument('--repeat_type', type=str,default='gold',
                         help='default is repeat based on gold. The other option is to repeat based on similarity')
    
    parser.add_argument('--repeat_pickle', type=str,default=None,
                         help='default is not to repeat data. The repeat_pickle file should contain the similarities')

    # Parse the command-line arguments
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_option()
    main(args)
    