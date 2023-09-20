import json
import os
import torch
from train import set_seed, get_checkpoint_name
from data import create_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.nn import functional as F

def evaluate(model, dataset, config, tokenizer, device):
    collate_fn = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    data_loader = DataLoader(dataset, shuffle=True, batch_size=config['batch_size'], collate_fn = collate_fn)
    
    model.eval()
    total_loss = 0
    labels = []
    preds = []
    for i, batch in enumerate(data_loader):
        inputs = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**inputs)
        total_loss += outputs.loss.item()
        # Get predictions
        preds.append(outputs.logits.cpu().detach())
        labels.append(batch['labels'].cpu().detach())
        # if i >1: # debug
        #     break
    
    # Concatenate predictions and labels
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)

    avg_loss = total_loss / len(data_loader)
    model.train()

    return preds, labels, avg_loss

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

    preds, labels, avg_loss = evaluate(model, dev_dataset, config, tokenizer, device)

def load_config(path):
    # Open and read the JSON file
    with open(path, 'r') as file:
        config = json.load(file)
    return config

if __name__ == "__main__":
    config_path = os.path.join("conf", "config.json")
    config = load_config(config_path)
    main(config)