import json
import os
import torch
from utils import set_seed, get_checkpoint_name
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.nn import functional as F
from collections import defaultdict
from utils import calculate_IR_metrics
from data import load_all_samples
from torch.utils.data import DataLoader
import numpy as np
import math
from utils import calculate_true_label_probs, count_true_label_correct, RPF1, RPF1_binary, create_pickle

def evaluate_data(model, data_loader, config, device):
    """
    Evaluate the given model on a data loader and calculate various metrics.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader containing the evaluation data.
        config (dict): A configuration dictionary with options for evaluation.
        device (torch.device): The device (CPU or GPU) on which to perform evaluation.

    Returns:
        tuple: A tuple containing:
            a. preds (torch.Tensor): Model predictions.
            b. labels (torch.Tensor): Ground truth labels.
            c. avg_loss (float): Average loss over the evaluation dataset.
            d. metrics (dict): Dictionary of evaluation metrics (e.g., 'r1', 'r2', 'mrr') if not in debug mode.

    Note:
        This function evaluates the given model on a provided data loader, calculates the average loss,
        and optionally computes additional evaluation metrics such as Recall@1 (r1), Recall@2 (r2),
        and Mean Reciprocal Rank (MRR) if not in debug mode. 
    """
    model.eval()
    print('Evaluate...')
    total_loss = 0
    labels = []
    preds = []
    sentence_ids = []
    option_ids = []

    for i, batch in enumerate(data_loader):
        inputs = {key: value.to(device) for key, value in batch.items() if key not in ['sentence_id','option_id']}
        outputs = model(**inputs)
        total_loss += outputs.loss.item()
        # Get predictions
        preds.append(outputs.logits.cpu().detach())
        labels.append(batch['labels'].cpu().detach())

        sentence_ids.extend(batch['sentence_id'])
        option_ids.extend(batch['option_id'])
        
    
    # Concatenate predictions and labels
    preds = torch.cat(preds, dim=0) # torch tensor (num_val_examples, 2)
    labels = torch.cat(labels, dim=0) # torch tensor (num_val_examples)

    if config['calculate_probs']:
        preds = calculate_probs(preds) # torch tensor (num_val_examples, 2)

    grouped_data, labeled_data = group_data(sentence_ids, option_ids, preds, labels)


    sorted_data = sort_grouped_data(grouped_data)
    metrics = {}
    r_1, r_2, mrr = calculate_IR_metrics(sorted_data, labeled_data)
    p, r, f1 = RPF1(grouped_data, labeled_data)

    binary_precision, binary_recall, binary_f1_score = RPF1_binary(preds, labels)

    metrics['r1'] = r_1
    metrics['r2'] = r_2
    metrics['mrr'] = mrr
    metrics['precision'] = p
    metrics['recall'] = r 
    metrics['f1'] = f1
    metrics['binary_precision'] = binary_precision 
    metrics['binary_recall'] = binary_recall
    metrics['binary_f1_score'] = binary_f1_score

    avg_loss = total_loss / len(data_loader)
    model.train()

    return preds, labels, avg_loss, metrics, grouped_data, labeled_data

def group_data(sentence_ids, option_ids, probabilities, labels):
    """
    Group data based on sentence IDs and create a mapping of labeled data.

    Args:
        sentence_ids (list[int]): List of sentence IDs.
        option_ids (list[int]): List of option IDs.
        probabilities (list[float]): List of tuples containing prediction probabilities for negative and positive class respectively
        labels (list[int]): List of labels (0 or 1).

    Returns:
        tuple (dict, dict): A tuple containing two dictionaries -
            a. grouped_data: A dictionary mapping sentence IDs to a list of (option_id, positive_probability) tuples.
            b. labeled_data: A dictionary mapping sentence IDs to the option_id with a positive label.
    """
    grouped_data = defaultdict(list) # {sentence_id : [(option_id, predict_positive_prob),..]}
    labeled_data = defaultdict(int) # {sentence_id : option_id}
    # Iterate through the lists and group based on sentence IDs
    for sentence_id, option_id, probability, label in zip(sentence_ids, option_ids, probabilities, labels):
        #tensor to scalar
        sentence_id, option_id = sentence_id.item(), option_id.item()
        #! take at position 1 because 
        pos_probability, label = probability[1].item(), label.item()

        grouped_data[sentence_id].append((option_id, pos_probability))
        if label == 1:
            labeled_data[sentence_id] = option_id

    return grouped_data, labeled_data

def sort_grouped_data(grouped_data): # grouped_data {sentence_id : [(option_id, predict_positive_prob),..]}
    # for each sentence_id sort the options in decreasing order
    """
    Sort grouped data based on prediction probabilities.

    Args:
        grouped_data (dict): A dictionary mapping sentence IDs to a list of (option_id, positive_probability) tuples produced from group_data method.

    Returns:
        sorted_grouped_data (dict): A dictionary mapping sentence IDs to a list of option IDs sorted in decreasing order of probabilities.
    """
    sorted_grouped_data = {}
    for sentence_id in grouped_data.keys():
        # Create pairs of values and probabilities
        pairs = grouped_data[sentence_id]
        
        # Sort the pairs based on probabilities
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        
        # Extract the sorted option_ids from the sorted pairs
        sorted_option_ids = [pair[0] for pair in sorted_pairs]
        
        # Store the sorted values in the result dictionary
        sorted_grouped_data[sentence_id] = sorted_option_ids
    return sorted_grouped_data

def calculate_probs(logits):
    """
    Calculate probabilities from logits using softmax.

    Args:
        logits (tensor): Input logits.

    Returns:
        probs (tenosr): Probability values obtained using softmax in the last dimension.
    """
    probs = F.softmax(logits, dim=-1)
    return probs

def load_config(path):
    # Open and read the JSON file
    with open(path, 'r') as file:
        config = json.load(file)
    return config
