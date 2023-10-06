import torch
import numpy as np
import datetime
import random
from collections import defaultdict

# see https://github.com/Nealcly/MuTual/blob/master/eval_sample/eval.py
def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    
    Args:
        seed (int): The seed value to set for random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_checkpoint_name(config):
    """
    Generate a checkpoint name based on the configuration.

    Args:
        config (dict): A dictionary containing configuration information.

    Returns:
        config_name (string): A checkpoint name composed of dataset name, model name, and date information.
    """
    now = datetime.datetime.now()
    date_info = f'{now.month}_{now.day}_{now.hour}_{now.minute}'
    config_name =  f"{config['dataset_name']}_{config['model_name']}_{date_info}"

    print('date info ', date_info)
    return config_name

def calculate_IR_metrics(sorted_grouped_data, labeled_data):
    """
    Calculate information retrieval metrics including R@1, R@2, and MRR. https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)

    Args:
        sorted_grouped_data (dict): A dictionary containing sorted grouped data.
        labeled_data (dict): A dictionary containing labeled data.

    Returns:
        r_1 (float): Recall at 1
        r_2 (float): Recall at 2
        mrr (float): Mean Reciprocal Rank
    """
    counter_r1, counter_r2, mrr = 0, 0, 0
    for sentence_id in sorted_grouped_data:
        golden = labeled_data[sentence_id]
        output_list = sorted_grouped_data[sentence_id]
        assert sorted(output_list) == [0,1,2,3]
        index = output_list.index(golden)
        if index == 0:
            counter_r1 += 1
        elif index == 1:
            counter_r2 += 1
        mrr += 1 / (index + 1)
    
    count_data = len(sorted_grouped_data)
    mrr = mrr/count_data
    r_1 = counter_r1 / count_data
    r_2 = (counter_r2 +counter_r1) / count_data
    print("R@1: %.3f \t R@2: %.3f \t MRR %.3f" %(r_1, r_2, mrr))
    return r_1, r_2, mrr

def calculate_true_label_probs(grouped_data, labeled_data, true_label_dict_probs):
    '''
    Appends the probability of the correct label in the current epoch to the true_label_dict_probs.
    Useful to later calculate the confidence of the model predictions. 

    Parameters:
        grouped_data (dict): A dictionary of grouped data with sentence IDs as keys and lists of (option_id, probability) pairs as values.
        labeled_data (dict): A dictionary mapping sentence IDs to their correct option IDs.
        true_label_dict_probs (dict): A dictionary with sentence IDs as keys and lists of label probabilities for each epoch.

    Returns:
        true_label_dict_probs (dict): An updated dictionary with sentence IDs as keys and lists of label probabilities for each epoch.
    
    Note: 
        You should call this function at the end of each epoch by passing the same true_label_dict_probs.
        When training finishes, you must calculate the average for every sentence_id
    '''
    for sentence_id in grouped_data:
        option_probs_pairs = grouped_data[sentence_id]
        correct_option_id = labeled_data[sentence_id]
        for option_id, prob in option_probs_pairs:
            if option_id == correct_option_id:
                true_label_prob = prob
                true_label_dict_probs[sentence_id].append(true_label_prob)

 
    return true_label_dict_probs


def count_true_label_correct(grouped_data, labeled_data, count_true_pred_dict):
    """
    Appends 1  to count_true_pred_dict if the true label had the highest probality among the options otherwise it appends 0.
    Useful to later calculate the correctness of the model predictions 

    Parameters:
        grouped_data (dict): A dictionary of grouped data with sentence IDs as keys and lists of (option_id, probability) pairs as values.
        labeled_data (dict): A dictionary mapping sentence IDs to their correct option IDs.
        count_true_pred_dict (dict): A dictionary with sentence_ids as keys and lists as values representing whether the model gave the correct label the biggest probability.

    Returns:
        count_true_pred_dict (dict): An updated dictionary with sentence_ids as keys and lists with binary values representing whether the model gave the correct label the biggest probability.

    Note: 
        You should call this function at the end of each epoch by passing the count_true_pred_dict.
        When training finishes, you must calculate the average for every sentence_id
    """

    max_scores = {key: max(value, key=lambda x: x[1])[0] for key, value in grouped_data.items()}
    for sentence_id, max_pred in max_scores.items():
        correct_option_id = labeled_data[sentence_id]
        # if pred == correct_option_id:
        true_label_asserted = 1 if max_pred == correct_option_id else 0

        count_true_pred_dict[sentence_id].append(true_label_asserted)            
    
    return count_true_pred_dict

def calculate_mean(true_label_dict_probs):
    """
    Calculate the average scores for each sentence based on a dictionary.

    Args:
    true_label_dict_probs (dict): A dictionary where keys are sentence IDs, and values are lists of values
                                 across training epochs.

    Returns:
    avg_dict_probs (dict): A dictionary where keys are sentence IDs, and values are the average scores
                          calculated from the probabilities for each sentence.
    """
    avg_dict_probs = {}
    for sentence_id in true_label_dict_probs:
        avg_dict_probs[sentence_id] = np.mean(true_label_dict_probs[sentence_id])
    return avg_dict_probs


def calculate_variability(true_label_dict_probs, true_avg_dict_probs):
    """
    Calculates the variability which measures the spread across training epochs of the probability of the true label using std

    Args:
        true_label_dict_probs (dict): A dictionary with keys sentences_id and values containing probabilities of the true label at every epoch
                            for different epochs.
         true_avg_dict_probs (dict): A dictionary of confidence values for each sentence.
    Returns:
        variability (dict): A dictionary containing the calculated variability for each sentence.
    
    Note: 
        You should call this function only when training finishes!

    """

    numerators = defaultdict(list)
    for sentence_id in true_label_dict_probs:
        confidence = true_avg_dict_probs[sentence_id]
        true_label_probs_across_epochs = true_label_dict_probs[sentence_id]
        for true_label_prob in true_label_probs_across_epochs:
                numerator = (true_label_prob - confidence)**2
                numerators[sentence_id].append(numerator)

    variability = {}
    for sentence_id in true_label_dict_probs:
        variability[sentence_id] = np.sqrt(np.mean(numerators[sentence_id]))

    return variability
# just tests
if __name__=='__main__':
    sorted_grouped_data = {1:[2,0,1,3],2:[1,2,0,3]}
    labeled_data = {1:2,2:1}
    r_1, r_2, mrr = calculate_IR_metrics(sorted_grouped_data, labeled_data)
    assert(r_1 == r_2== mrr == 1.0)

    sorted_grouped_data = {1:[2,0,1,3],2:[1,2,0,3]}
    labeled_data = {1:2,2:0}
    r_1, r_2, mrr = calculate_IR_metrics(sorted_grouped_data, labeled_data)
    assert(r_1 == 1/2 == r_2)
    assert(mrr == 2/3)
