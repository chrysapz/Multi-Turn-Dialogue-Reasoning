import os
from utils import load_pickle
from transformers import T5Tokenizer
import re
import os
import torch
import numpy as np
import json
from copy import deepcopy
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
    AutoModelForMultipleChoice, DataCollatorWithPadding, AutoModelForCausalLM, RobertaTokenizerFast
from torch.utils.data import DataLoader
from utils import set_seed, get_checkpoint_name
from data import load_all_samples
from mutual_dataset import MutualDataset
import matplotlib.pyplot as plt
from evaluate import evaluate_data, calculate_probs, group_data, sort_grouped_data
from time import gmtime, strftime
from tqdm import tqdm
from collections import defaultdict
from utils import calculate_true_label_probs, repeat_golds_training_data, repeat_training_data_based_on_sim, add_augmented_as_gold, add_augmented_label_based_on_sim, load_pickle, count_true_label_correct, calculate_mean, calculate_variability, print_args, create_pickle, create_dicts_from_tuples
import pandas as pd
import argparse
from similarities import calculate_similarities, get_sim_key
from manual_filtering import preprocess_augmented_labels, add_start_to_augmented_labels, data_statistics
from trainer import train
import time
# # Decode the token IDs
# decoded_tokens = tokenizer.batch_decode(tmp_generated_ids, skip_special_tokens=True)
NUM_TRAIN_EXAMPLES = 6000

def find_option_text(concat):
    # Find the last position of 'm : ' or 'f : '
    last_m_position = concat.rfind('m :')
    last_f_position = concat.rfind('f :')

    # Find the maximum of the two positions
    last_position = max(last_m_position, last_f_position)

    # Check if any of the substrings were found
    
    return concat[:last_position], concat[last_position:]  # 4 is the length of 'm : ' or 'f : '


def create_new_dialogues(id2t5input, all_generated_ids, tokenizer, sent_ids):
    extra_id_str = '<extra_id_'
    dummy_id = 10000
    dummy2initial = {}
    generated_id2history, generated_id2options, generated_id2label_id = {}, {}, {}
    for id in id2t5input:
        dummy_id +=1
        dummy2initial[dummy_id] = id

        t5_input = id2t5input[id]
        
        index_generated_ids = sent_ids.index(id)
        tmp_generated_ids = all_generated_ids[index_generated_ids]


        

        semi_correct_generated_text = tokenizer.batch_decode(tmp_generated_ids, skip_special_tokens=True)[1:] # avoid padding at the start
        semi_correct_generated_text = ' '.join(semi_correct_generated_text)

        # hack= ' '.join(tokenizer.batch_decode([tmp_generated_ids]))
        # decoded_tokens = tokenizer.batch_decode([tmp_generated_ids], skip_special_tokens=True)
        # # Apply the function to the decoded tokens
        # concatenated_tokens = concatenate_subwords(decoded_tokens)

        # Extract segments between <extra_id_X> using regex
        segments = re.split(r'<extra_id_\d+>',semi_correct_generated_text )

        # Remove empty segments
        segments = [s.strip() for s in segments if s.strip()]

        # Replace segments in the input text
        output_text = t5_input
        for i, segment in enumerate(segments):
            placeholder = f"<extra_id_{i}>"
            output_text = output_text.replace(placeholder, segment)

        hist, true_new_option = find_option_text(output_text)
        generated_id2history[dummy_id] = hist
        generated_id2options[dummy_id] = [true_new_option] # then repeat the ones from the initial
        generated_id2label_id[dummy_id] = [0] # always put first position in the options label

    return generated_id2history, generated_id2options, generated_id2label_id, dummy2initial
        # mask_id = 0
        # current_extra_id = f'{extra_id_str}_{mask_id}'

        # next_extra_id = f'{extra_id_str}_{mask_id}'

        # t5_input.replace(current_extra_id,)
        # a=1

def create_sub_dict(id2history, id2options, id2label_id, k_ids):
    """
    For debugging, create sub-dictionaries from input dictionaries based on a list of selected keys.

    Args:
        id2history (dict): A dictionary mapping IDs to history data.
        id2options (dict): A dictionary mapping IDs to options data.
        id2label_id (dict): A dictionary mapping IDs to label IDs.
        k_ids (list): A list of keys (IDs) to select from the input dictionaries.

    Returns:
        tuple: A tuple containing three dictionaries:
            - sub_id2history (dict): A sub-dictionary containing selected history data.
            - sub_id2options (dict): A sub-dictionary containing selected options data.
            - sub_id2label_id (dict): A sub-dictionary containing selected label IDs.
    """
    sub_id2history = {id: id2history[id] for id in k_ids}
    sub_id2options = {id: id2options[id] for id in k_ids}
    sub_id2label_id = {id: id2label_id[id] for id in k_ids}

    return sub_id2history, sub_id2options, sub_id2label_id

def add_existing_negative_options(generated_train_id2options, dummy2initial, train_id2options, train_id2label_id):
    new_generated_train_id2options = deepcopy(generated_train_id2options)

    for dummy_id in generated_train_id2options:
        initial_id = dummy2initial[dummy_id]
        
        initial_options = train_id2options[initial_id] # list
        initial_label_id = train_id2label_id[initial_id][0]

        negative_options = [option for i, option in enumerate(initial_options) if i != initial_label_id]

        new_generated_train_id2options[dummy_id].extend(negative_options)
    return new_generated_train_id2options

from select_masks import create_dialogues

def main(config):
    print('Training')
    print_args(args)
    config = vars(args) # convert to dict
    # config['debug'] = True
    
    out_dir = config['out_dir']
    # config['augment'] = 'finetuned.pkl'
    # Set up the data directory and device
    base_dir = os.path.join(config['data_dir'], config['dataset_name'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device ',device)
    set_seed(config['seed'])

    checkpoint_name = get_checkpoint_name(config)
    save_folder = os.path.join(out_dir, checkpoint_name)
    if config['debug']:
        save_folder = os.path.join(save_folder, 'debug')
        config['epochs']=2

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

    train_id2history, train_id2options, train_id2label_id = create_dicts_from_tuples(train_samples, train_random_indices)

    val_id2history, val_id2options, val_id2label_id = create_dicts_from_tuples(dev_samples, val_random_indices)

    test_samples = load_all_samples(base_dir, 'dev')
    test_indices = list(range(len(test_samples)))
    test_id2history, test_id2options, test_id2label_id = create_dicts_from_tuples(test_samples, test_indices)

    config_2 = {'mode': 'binary', 'data_dir': 'data', 'out_dir': 'checkpoints', 'dataset_name': 'mutual', 'debug': False, 'model_name': 'google/t5-v1_1-xl', 'batch_size': 64, 'n_gram_start': 1, 'n_gram_finish': 1}

    id2t5, keywords_samples, dicts, generated_ids, count_masks, sent_ids, generated_dialogues = create_dialogues(config_2)
    # print('id2t5',id2t5[982])
    # print('train_id2history',train_id2history[982])

    t5_tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xl')

    generated_train_id2history, generated_train_id2options, generated_train_id2label_id, dummy2initial = create_new_dialogues(id2t5, generated_ids, t5_tokenizer, sent_ids)
    
    new_generated_train_id2options = add_existing_negative_options(generated_train_id2options, dummy2initial, train_id2options, train_id2label_id)

    #!todo update train_id2history, train_id2options, train_id2label_id using the ones below
    #generated_train_id2history, generated_train_id2options, generated_train_id2label_id
    train_id2options.update(new_generated_train_id2options)
    train_id2label_id.update(generated_train_id2label_id)
    train_id2history.update(generated_train_id2history)

    create_pickle(train_id2options,'train_options.pkl')
    create_pickle(train_id2label_id,'train_id2labelid.pkl')
    create_pickle(train_id2history,'train_id2history.pkl')
    create_pickle(dummy2initial,'dummy.pkl')

    if config['debug']:
        train_k_ids = [2650, 2868]
        test_k_ids = [2,1]
        val_k_ids = [2860,5806]

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
        out_dir = config['out_dir']

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
    parser.add_argument('--augment', type=str,default=None,
                         help='default is not to augment training data')

    # Parse the command-line arguments
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_option()
    main(args)