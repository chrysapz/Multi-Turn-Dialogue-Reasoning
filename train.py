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
from utils import calculate_true_label_probs, count_true_label_correct, calculate_mean, calculate_variability

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

        eval_preds, eval_labels, eval_loss, metrics = evaluate_data(model, dev_loader, config, device)
        if metrics['r1'] > best_r1:
            best_model = deepcopy(model)
            best_epoch = epoch
            best_optimizer = deepcopy(optimizer)
    
    confidence = calculate_mean(true_label_dict_probs)
    variability = calculate_variability(true_label_dict_probs, confidence)
    correctness = calculate_mean(count_true_pred_dict)

    best_model_info = {'model_state_dict': best_model.state_dict(), 'epoch':best_epoch,'optimizer_state_dict':best_optimizer.state_dict()}

    return best_model_info, epoch_loss, confidence, variability, correctness


def main(config):
    print('Training')
    print(config)

    # Set up the data directory and device
    base_dir = os.path.join(config['data_dir'], config['dataset_name'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config['seed'])

    # Load the tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(config['tokenizer_name'], use_fast=True)

    # Load training and val data
    train_samples = load_all_samples(base_dir, 'train')
    dev_samples = load_all_samples(base_dir, 'dev')

    if config['debug']:
        k = 2
        train_samples = train_samples[:k] # k * num_options in the dataset below
        dev_samples = dev_samples[:k] # k * num_options in the dataset below
        config['epochs'] = 2
        config['batch_size'] = 2

    # tokenize and create datasets for training and eval datat
    train_dataset = MutualDataset(train_samples, tokenizer, config['max_seq_length'])
    dev_dataset = MutualDataset(dev_samples, tokenizer, config['max_seq_length'])

    # create the model
    model = AutoModelForSequenceClassification.from_pretrained(config['model_name'], num_labels = 2)
    model = model.to(device)

    # Initialize collate functions and data loaders for training and development
    train_collate_fn = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config['batch_size'], collate_fn=train_collate_fn)

    dev_collate_fn = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=config['batch_size'], collate_fn=dev_collate_fn)

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

    # Save the model and training loss plot
    #! be cautious not to exhaust memory since we save it every time
    if not config['debug']:
        out_dir = config['out_dir']
        checkpoint_name = get_checkpoint_name(config)
        save_folder = os.path.join(out_dir, checkpoint_name)
        os.makedirs(save_folder)
        save_name = os.path.join(save_folder, 'model')
        torch.save(model_info, save_name)

        out_path = os.path.join(save_folder, "training_loss.png")
        plt.plot(avg_loss)
        plt.savefig(out_path)

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

if __name__ == "__main__":
    config_path = os.path.join("conf", "config.json")
    config = load_config(config_path)
    main(config)
    