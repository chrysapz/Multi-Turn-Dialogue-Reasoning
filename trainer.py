
import torch
import numpy as np

from copy import deepcopy

from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
    AutoModelForMultipleChoice, DataCollatorWithPadding, AutoModelForCausalLM, RobertaTokenizerFast
from torch.utils.data import DataLoader

from mutual_dataset import MutualDataset
import matplotlib.pyplot as plt
from evaluate import evaluate_data, calculate_probs, group_data, sort_grouped_data
from time import gmtime, strftime
from tqdm import tqdm
from collections import defaultdict
from utils import calculate_true_label_probs, count_true_label_correct, calculate_mean, calculate_variability
import pandas as pd


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
    best_r1 = 0
    # for dataset cartography
    true_label_dict_probs = defaultdict(list)
    count_true_pred_dict = defaultdict(list)
    numerators = defaultdict(list)
    no_improvement_count = 0  # Initialize a counter for early stopping
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

        eval_preds, eval_labels, eval_loss, metrics, grouped_data = evaluate_data(model, dev_loader, config, device)
        if metrics['r1'] > best_r1 or config['debug']:
            best_model = deepcopy(model)
            best_epoch = epoch
            best_optimizer = deepcopy(optimizer)
            best_r1 = metrics['r1']
            no_improvement_count = 0  # Reset the counter
        else:
            no_improvement_count += 1

        if no_improvement_count >= 3:
            print(f"Early stopping triggered after {no_improvement_count} epochs of no improvement.")
            break
    
    confidence = calculate_mean(true_label_dict_probs)
    variability = calculate_variability(true_label_dict_probs, confidence)
    correctness = calculate_mean(count_true_pred_dict)

    #create dataframe for data_plots
    df = pd.DataFrame(list(correctness.items()), columns=['sentence_id', 'correctness'])
    df['confidence'] = df['sentence_id'].map(confidence)
    df['variability'] = df['sentence_id'].map(variability)

    best_model_info = {'model_state_dict': best_model.state_dict(), 'epoch':best_epoch,'optimizer_state_dict':best_optimizer.state_dict(), 'r1':best_r1}

    return best_model_info, epoch_loss, confidence, variability, correctness