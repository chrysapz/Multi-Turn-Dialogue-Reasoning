import json
import os
import torch
from utils import set_seed, get_checkpoint_name
from data import create_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.nn import functional as F
from collections import defaultdict
from utils import calculate_IR_metrics

def evaluate_data(model, dataset, config, tokenizer, device):
    collate_fn = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    data_loader = DataLoader(dataset, shuffle=False, batch_size=config['batch_size'], collate_fn = collate_fn)
    
    model.eval()
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
        if config['debug'] and i >= 2: # debug
            break
    
    # Concatenate predictions and labels
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)

    if config['calculate_probs']: # useful for data maps
        preds = calculate_probs(preds)

    #
    grouped_data, labeled_data = group_data(sentence_ids, option_ids, preds, labels)
    sorted_data = sort_grouped_data(grouped_data)
    if not config['debug']:
        r_1, r_2, mrr = calculate_IR_metrics(sorted_data, labeled_data)

    avg_loss = total_loss / len(data_loader)
    model.train()

    return preds, labels, avg_loss

def group_data(sentence_ids, option_ids, probabilities, labels):
    grouped_data = defaultdict(list) # {sentence_id : [(option_id, predict_positive_prob),..]}
    labeled_data = defaultdict(int) # {sentence_id : true_label}
    # Iterate through the lists and group based on sentence IDs
    for sentence_id, option_id, probability, label in zip(sentence_ids, option_ids, probabilities, labels):
        #tensor to scalar
        sentence_id, option_id = sentence_id.item(), option_id.item()
        #! take at position 1 because 
        pos_probability, label = probability[1].item(), label.item()

        grouped_data[sentence_id].append((option_id, pos_probability))
        labeled_data[sentence_id] = label

    return grouped_data, labeled_data

def sort_grouped_data(grouped_data): # grouped_data {sentence_id : [(option_id, predict_positive_prob),..]}
    # for each sentence_id sort the options in decreasing order
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
    probs = F.softmax(logits, dim=-1)
    return probs

def main(config):
    print('Evaluation')
    print(config)
    base_dir = os.path.join(config['data_dir'], config['dataset_name'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config['seed'])
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])

    dev_dataset = create_dataset(base_dir, 'dev', tokenizer, config['max_seq_length'])

    #load the model
    #todo change this. Now we only allow checkpoints from the same config file
    out_dir = config['out_dir']
    checkpoint_name = get_checkpoint_name(config)
    save_path = os.path.join(out_dir, checkpoint_name)
    model = AutoModelForSequenceClassification.from_pretrained(save_path, num_labels = 2)

    preds, labels, avg_loss = evaluate_data(model, dev_dataset, config, tokenizer, device)

def load_config(path):
    # Open and read the JSON file
    with open(path, 'r') as file:
        config = json.load(file)
    return config

if __name__ == "__main__":
    # just for check
    # grouped_data = {1:[(1,0.4), (3,0.7)], 2:[(3,0.8),(2,0.5)]}
    # an = sort_grouped_data(grouped_data)

    config_path = os.path.join("conf", "config.json")
    config = load_config(config_path)
    main(config)