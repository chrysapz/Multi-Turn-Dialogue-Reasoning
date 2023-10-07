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
from utils import calculate_true_label_probs, count_true_label_correct

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
    if not config['debug']:
        r_1, r_2, mrr = calculate_IR_metrics(sorted_data, labeled_data)
        p, r, f1 = RPF1(grouped_data, labeled_data)
        metrics['r1'] = r_1
        metrics['r2'] = r_2
        metrics['mrr'] = mrr
        metrics['precision'] = p
        metrics['recall'] = r 
        metrics['f1'] = f1

    avg_loss = total_loss / len(data_loader)
    model.train()

    return preds, labels, avg_loss, metrics

def RPF1(grouped_data, labeled_data):

  
    TP, FP, FN = 0, 0, 0

    max_scores = {key: max(value, key=lambda x: x[1])[0] for key, value in grouped_data.items()}
    print("these are max scores: ", max_scores)
    for sentence_id, pred in max_scores.items():
        correct_option_id = labeled_data[sentence_id]
        if pred == correct_option_id:
            TP += 1
        else:
            FP += 1
            FN += 1
    
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0  
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0         
    
    return precision, recall, f1


def confidence(grouped_data, labeled_data, true_label_dict_probs):
    '''
    grouped_data: {sentence_id:[(option_id_epoch_0, prob_epoch_0)]]}
    labeled_data: {sentence_id:corrept_option_id}
    true_label_dict_probs: {sentence_id: [probs_epoch_0,probs_epoch_1]}
    '''
    for sentence_id in grouped_data:
        option_probs_pairs = grouped_data[sentence_id]
        correct_option_id = labeled_data[sentence_id]
        for option_id, prob in option_probs_pairs:
            if option_id == correct_option_id:
                true_label_prob = prob
                true_label_dict_probs[sentence_id].append(true_label_prob)

 
    return true_label_dict_probs

'''grouped_data = {0:[(1,0.6),(0,0.211)], 1:[(1,0.111),(0,0.8)]}
labeled_data = {0:1,1:0}
true_label_dict_probs = defaultdict(list)'''

def correctness(grouped_data, labeled_data, true_pred_dict_probs):
    '''the fraction of times the
    model correctly labels xi across epochs, named
    correctness; this score only has 1 + E possible
    values. Intuitively, a high-confidence instance is
    “easier” for the given learner
    
    ιδια φιλοσοφια με απο πανω απλα 0 η 1 αναλογα αν το HIgher pred ηταν για το σωστο και μετα θα γινει avg over epochs
    true_label_dict_probs: {sentence_id: [correct_epoch_0, correct_epoch_1]}'''

    max_scores = {key: max(value, key=lambda x: x[1])[0] for key, value in grouped_data.items()}
    for sentence_id, pred in max_scores.items():
        correct_option_id = labeled_data[sentence_id]
        if pred == correct_option_id:
            true_label_asserted = 1 if pred == correct_option_id else 0

            true_pred_dict_probs[sentence_id].append(true_label_asserted)            
    
    return true_pred_dict_probs

def variability_numerators(confidence, grouped_data, labeled_data, numerators):
    '''measures the spread of p across epochs using std
    grouped_data: {sentence_id:[(option_id_epoch_0, prob_epoch_0)]]}
    labeled_data: {sentence_id:corrept_option_id}
    true_label_dict_probs: {sentence_id: [probs_epoch_0,probs_epoch_1]}'''
    
    for sentence_id in grouped_data:
        option_probs_pairs = grouped_data[sentence_id]
        correct_option_id = labeled_data[sentence_id]
        for option_id, prob in option_probs_pairs:
            if option_id == correct_option_id:
                true_label_prob = prob
                numerator = (true_label_prob - confidence[sentence_id])**2
                numerators[sentence_id].append(numerator)

    return numerators



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

def main(config):

    from train import dataset_per_mode
    
    print('Evaluation')
    print(config)
    base_dir = os.path.join(config['data_dir'], config['dataset_name'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config['seed'])
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])

    dataset_class = dataset_per_mode[config['mode']]

    dev_samples = load_all_samples(base_dir, 'dev')
    dev_dataset = dataset_class(dev_samples, tokenizer, config['max_seq_length'])

    dev_collate_fn = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=config['batch_size'], collate_fn=dev_collate_fn)

    #load the model
    #todo change this. Now we only allow checkpoints from the same config file
    out_dir = config['out_dir']
    checkpoint_name = 'mutual_roberta-base_9_22_20_16' # get_checkpoint_name(config)
    save_path = os.path.join(out_dir, checkpoint_name)
    model = AutoModelForSequenceClassification.from_pretrained(save_path, num_labels = 2)

    preds, labels, avg_loss = evaluate_data(model, dev_loader, config, device)

def load_config(path):
    # Open and read the JSON file
    with open(path, 'r') as file:
        config = json.load(file)
    return config

# just tests
if __name__ == "__main__":
    grouped_data = {0:[(1,0.4),(0,0.2)], 1:[(1,0.1),(0,0.6)]}
    labeled_data = {0:1,1:0}
    true_label_dict_probs = defaultdict(list)
    true_pred_dict_probs = defaultdict(list)
    numerators = defaultdict(list)

    pre, re, f1 = RPF1(grouped_data, labeled_data)
    print("First time: ",pre,re,f1)
    #true_label_dict_probs = confidence(grouped_data, labeled_data, true_label_dict_probs)
    #true_pred_dict_probs = correctness(grouped_data, labeled_data, true_pred_dict_probs)

    grouped_data = {0:[(1,0.6),(0,0.211)], 1:[(1,0.8),(0,0.1)]}
    labeled_data = {0:1,1:0}
    pre, re, f1 = RPF1(grouped_data, labeled_data)
    print("Second time: ",pre,re,f1)

    #true_label_dict_probs = confidence(grouped_data, labeled_data, true_label_dict_probs)
    #true_pred_dict_probs = correctness(grouped_data, labeled_data, true_pred_dict_probs)


    true_avg_dict_probs = defaultdict(float)
    true_avg_dict_preds = defaultdict(float)
    for sentence_id in true_label_dict_probs:
        true_avg_dict_probs[sentence_id] = np.mean(true_label_dict_probs[sentence_id])

    for sentence_id in true_pred_dict_probs:
        true_avg_dict_preds[sentence_id] = np.mean(true_pred_dict_probs[sentence_id])
    
    avg_numerators = defaultdict(float)
    for sentence_id in numerators:
        avg_numerators[sentence_id] = np.mean(numerators[sentence_id])

    num_epochs = 10
    std = (avg_numerators / num_epochs)**0.5

    print(true_pred_dict_probs)
    print(true_avg_dict_preds)
    

    a=1
    '''
    grouped_data: {sentence_id:[ [(option_id_epoch_0, prob_epoch_0)],  [(option_id_epoch_0, prob_epoch_0)]]}
    labeled_data: {sentence_id:corrept_option_id}
    true_label_dict_probs: {sentence_id: [probs_epoch_0,probs_epoch_1]}
    '''

    # true_label_dict_avg_probs = defaultdict(float)
    # for sentence_id in true_label_prob:
    #     true_label_dict_avg_probs[sentence_id] = np.mean(true_label_prob[sentence_id])

    # # just for check
    # # grouped_data = {1:[(1,0.4), (3,0.7)], 2:[(3,0.8),(2,0.5)]}
    # # an = sort_grouped_data(grouped_data)
    # sentence_ids = torch.tensor([0,1,0,1,0,1,0,1])
    # option_ids = torch.tensor([1,2,0,1,2,3,3,0])
    # preds = torch.tensor([[0,0.99],[0,0.99],[1,0],[1,0],[1,0],[1,0],[0.1,0.9],[0.4,0.6]])
    # labels = torch.tensor([0,0,0,0,0,0,1,1])
    # assert (len(sentence_ids) == len(option_ids) == len(preds) == len(labels))
    # grouped_data, labeled_data = group_data(sentence_ids, option_ids, preds, labels)
    # sorted_data = sort_grouped_data(grouped_data)
    # r_1, r_2, mrr = calculate_IR_metrics(sorted_data, labeled_data)
    # a=1