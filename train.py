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
import pickle
from manual_filtering import preprocess_augmented_labels, remove_last_sentence, add_start_to_augmented_labels, length_statistics, remove_using_similarity

NUM_TRAIN_EXAMPLES = 6000
repeat_type = {'sim': repeat_training_data_based_on_sim,
   'gold': repeat_golds_training_data
}


def train(model, train_loader, dev_loader, optimizer, config, device):
    """
    Train a Roberta model using the specified data and hyperparameters.

    Args:
        model (nn.Module): The Roberta model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        dev_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        config (dict): A dictionary containing hyperparameters and configuration settings.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') on which to perform training.

    Returns:
        tuple: A tuple containing the following elements:
            a. best_model_info (dict): A dictionary containing information about the best model that will be saved (e.g. optimizer, best_model, epoch corresponding to best model).
            b. epoch_loss (list): A list of training loss values for each epoch.
            c. confidence (dict): for each sentence_id, which is the key, we measure the mean model probability of the true label (float value)
            d. variability (dict): for each sentence_id, which is the key, we measure the spread of the true label (float value)
            e. correctness (dict): for each sentence_id, which is the key, we measure the fraction of times it assings the higher probability to the correct label (float value)
    """
    print('start training')
    epochs = config['epochs']

    epoch_loss = []
    all_epochs_preds = []
    all_epochs_labels = []
    best_r1 = 0
    # for dataset cartography
    true_label_dict_probs = defaultdict(list)
    count_true_pred_dict = defaultdict(list)
    numerators = defaultdict(list)
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0.0

        cur_epoch_preds = []
        cur_epoch_labels = []
        cur_sentence_ids = []
        cur_option_ids = []
        for batch_num, batch in enumerate(train_loader):
            if batch_num%400==0:
                print('inside training ', batch_num)
            optimizer.zero_grad()

            inputs = {key: value.to(device) for key, value in batch.items() if key not in ['sentence_id','option_id']} #sentence_id is useful only for metrics
            outputs = model(**inputs)
            loss = outputs[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
            # scheduler.step()  # Update learning rate schedule
            total_loss += loss.item()

            cur_epoch_preds.append(outputs.logits.cpu().detach())
            cur_epoch_labels.append(batch['labels'].cpu().detach())

            cur_sentence_ids.extend(batch['sentence_id'])
            cur_option_ids.extend(batch['option_id'])
            # if config['debug']: break #! todo remove this
        
        # Concatenate training predictions and training labels
        cur_epoch_preds = torch.cat(cur_epoch_preds, dim=0)
        cur_epoch_labels = torch.cat(cur_epoch_labels, dim=0)

        if config['calculate_probs']: # useful for data maps
            cur_epoch_preds = calculate_probs(cur_epoch_preds)


        # for dataset cartography
        grouped_data, labeled_data = group_data(cur_sentence_ids, cur_option_ids, cur_epoch_preds, cur_epoch_labels)
        # sorted_data = sort_grouped_data(grouped_data)
        true_label_dict_probs = calculate_true_label_probs(grouped_data, labeled_data, true_label_dict_probs)
        count_true_pred_dict = count_true_label_correct(grouped_data, labeled_data, count_true_pred_dict)

        avg_train_loss = total_loss / len(train_loader)
        epoch_loss.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{epochs} | Training Loss: {avg_train_loss}")

        eval_preds, eval_labels, eval_loss, metrics, sorted_data = evaluate_data(model, dev_loader, config, device)
        if metrics['r1'] > best_r1 or config['debug']:
            best_model = deepcopy(model)
            best_epoch = epoch
            best_optimizer = deepcopy(optimizer)
            best_r1 = metrics['r1']
    
    confidence = calculate_mean(true_label_dict_probs)
    variability = calculate_variability(true_label_dict_probs, confidence)
    correctness = calculate_mean(count_true_pred_dict)

    #create dataframe for data_plots
    df = pd.DataFrame(list(correctness.items()), columns=['sentence_id', 'correctness'])
    df['confidence'] = df['sentence_id'].map(confidence)
    df['variability'] = df['sentence_id'].map(variability)

    best_model_info = {'model_state_dict': best_model.state_dict(), 'epoch':best_epoch,'optimizer_state_dict':best_optimizer.state_dict(), 'r1':best_r1}

    return best_model_info, epoch_loss, confidence, variability, correctness

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

def main(config):
    print('Training')
    print_args(args)
    config = vars(args) # convert to dict
    # config['sim']  =True
    # # config['debug'] = True
    # config['augment'] = 'only_inference_colab_greedy_decoding.pkl'
    # Set up the data directory and device
    base_dir = os.path.join(config['data_dir'], config['dataset_name'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device ',device)
    set_seed(config['seed'])

    # Load the tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(config['tokenizer_name'], use_fast=True)

    # Load training and val data
    initial_train_samples = load_all_samples(base_dir, 'train')
    # Read the serialized data from the file and deserialize it
    indexed_train_list = load_pickle('index_list.pkl')

    shuffled_samples = [initial_train_samples[i] for i in indexed_train_list]

    train_samples = shuffled_samples[:NUM_TRAIN_EXAMPLES]
    train_random_indices = indexed_train_list[:NUM_TRAIN_EXAMPLES]
    train_id2history, train_id2options, train_id2label_id = create_dicts_from_tuples(train_samples, train_random_indices)

    dev_samples = shuffled_samples[NUM_TRAIN_EXAMPLES:] 
    val_random_indices = indexed_train_list[NUM_TRAIN_EXAMPLES:]
    val_id2history, val_id2options, val_id2label_id = create_dicts_from_tuples(dev_samples, val_random_indices)

    test_samples = load_all_samples(base_dir, 'dev')
    test_indices = list(range(len(test_samples)))
    test_id2history, test_id2options, test_id2label_id = create_dicts_from_tuples(test_samples, test_indices)

    if config['augment'] is not None:
        pickle_path = os.path.join('generated_text',config['augment'])
        generated_info = load_pickle(pickle_path)
        print('add start remove new line')
        # REMOVE '\n', truncate last and remove whole empty
        preprocessed_generated_info = preprocess_augmented_labels(generated_info)

        ratios = length_statistics(tokenizer, preprocessed_generated_info)

        new_generated_info = add_start_to_augmented_labels(preprocessed_generated_info, train_id2options)

        if config['sim']:
            print('with cosine similarity we consider generated augmented data with cosine > mean as gold otherwise they are noisy')
            model_name = 'all-distilroberta-v1'
            metric =  'cosine'
            sim_key = get_sim_key(model_name, metric)
            new_generated_info, avg_score, std_score, _,_,_, _ = calculate_similarities(64, new_generated_info, model_name, metric)
            # new_generated_info = remove_using_similarity(new_generated_info, avg_score, sim_key, config['consider_only_gold'])
            train_id2options, train_id2label_id, new_generated_info = add_augmented_label_based_on_sim(new_generated_info, train_id2options, train_id2label_id, avg_score)
            print('*'*12)
            print(f'number of new examples we are going to add: {len(new_generated_info)} ') # assume 1 per sent_id!
            print('without cosine similarity we consider all generated augmented data as gold')
            new_pickle_name = 'sim_'+config['augment'] 
            create_pickle(new_generated_info, new_pickle_name)
        # generated_info = preprocess_augmented_labels(generated_info, train_id2options)
        # print('remove last sentence')
        # generated_info = remove_last_sentence(generated_info)
        else:
            #! without sim filtering means that we consider everything as gold
            print('without cosine similarity we consider all generated augmented data as gold')
            train_id2options, train_id2label_id = add_augmented_as_gold(new_generated_info, train_id2options, train_id2label_id)
            new_pickle_name = 'manually_'+config['augment'] 
            create_pickle(new_generated_info, new_pickle_name)
        # train_id2history = create_dicts_from_tuples(train_samples, train_random_indices)

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

        train_id2history, train_id2options, train_id2label_id = create_sub_dict(train_id2history, train_id2options, train_id2label_id, train_k_ids)
        val_id2history, val_id2options, val_id2label_id = create_sub_dict(val_id2history, val_id2options, val_id2label_id, val_k_ids)
        test_id2history, test_id2options, test_id2label_id = create_sub_dict(test_id2history, test_id2options, test_id2label_id, test_k_ids)

        # train_samples = train_samples[:k] # k * num_options in the dataset below
        # dev_samples = dev_samples[:k]
        # test_samples = test_samples[:k] # k * num_options in the dataset below
        # config['epochs'] = 2
        # config['batch_size'] = 2
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


        # name =''

        # if config['augment'] is None:
        #     name += config['repeat_type']
        # if config['augment'] is not None:
        #     name += config['augment'][:-4]+'_'
        # else:
        #     name = 'baseline'

        # if config['sim']: 
        #     name += 'sim'

    
        # if config['augment'] is None and config['repeat'] == 0:
        #     name = 'baseline'

        checkpoint_name = 'baseline' #get_checkpoint_name(config)
        save_folder = os.path.join(out_dir, checkpoint_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_name = os.path.join(save_folder, 'model')
        torch.save(model_info, save_name)

        print('Test...')
        model = AutoModelForSequenceClassification.from_pretrained(config['model_name'], num_labels = 2)
        # load the model we just trained
        checkpoint = torch.load(save_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        preds, labels, avg_loss, metrics, sorted_data = evaluate_data(model, test_loader, config, device)

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
        create_pickle(sorted_data, path_probs_pickle)





    # below are just some checks
    # new_model =  AutoModelForSequenceClassification.from_pretrained(config['model_name'], num_labels = 2)
    # new_optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config['learning_rate'], eps=config['adam_epsilon'])
   
    # import gc
    # gc.collect()
    # torch.cuda.empty_cache()

    # checkpoint = torch.load(save_name, map_location ='cpu')
    # new_model.load_state_dict(checkpoint['model_state_dict'])
    # new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # #! check that we use the updated weights
    # assert(new_model.classifier.dense.weight == model.classifier.dense.weight.to('cpu')).all().item()
    # pretr = AutoModelForSequenceClassification.from_pretrained(config['model_name'], num_labels = 2)
    # assert(not (new_model.classifier.dense.weight == pretr.classifier.dense.weight.to('cpu')).all().item())
    
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
                         help='default is not to repeat data')
    parser.add_argument('--augment', type=str,default=None,
                         help='default is not to augment training data')

    # Parse the command-line arguments
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_option()
    main(args)
    