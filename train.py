import os
import torch
import numpy as np
import json
from transformers import AutoModelForSequenceClassification, AdamW, AutoTokenizer, AutoModelForMultipleChoice, DataCollatorWithPadding
from torch.utils.data import DataLoader
from utils import set_seed, get_checkpoint_name
from data import load_all_samples
from mutual_dataset import MutualDataset
import matplotlib.pyplot as plt
from evaluate import evaluate_data, calculate_probs

def train(model, train_loader, dev_loader, optimizer, config, device):
    
    epochs = config['epochs']

    epoch_loss = []
    all_epochs_preds = []
    all_epochs_labels = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        cur_epoch_preds = []
        cur_epoch_labels = []
        for batch in train_loader:
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

            if config['debug']: break #! todo remove this
        
        # Concatenate training predictions and training labels
        cur_epoch_preds = torch.cat(cur_epoch_preds, dim=0)
        cur_epoch_labels = torch.cat(cur_epoch_labels, dim=0)
        all_epochs_preds.append(cur_epoch_preds)
        all_epochs_labels.append(cur_epoch_labels)

        avg_train_loss = total_loss / len(train_loader)
        epoch_loss.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{epochs} | Training Loss: {avg_train_loss}")

        eval_preds, eval_labels, eval_loss = evaluate_data(model, dev_loader, config, device)
        #todo apply early stopping
    

    return model, epoch_loss, all_epochs_preds, all_epochs_labels


dataset_per_mode = {
    'binary': MutualDataset
}

def main(config):

    print('Training')
    print(config)
    base_dir = os.path.join(config['data_dir'], config['dataset_name'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config['seed'])

    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])

    dataset_class = dataset_per_mode[config['mode']]

    train_samples = load_all_samples(base_dir, 'train')
    train_dataset = dataset_class(train_samples, tokenizer, config['max_seq_length'])

    dev_samples = load_all_samples(base_dir, 'dev')
    dev_dataset = dataset_class(dev_samples, tokenizer, config['max_seq_length'])

    # create the model
    model = AutoModelForSequenceClassification.from_pretrained(config['model_name'], num_labels = 2)
    model = model.to(device)

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
    optimizer = AdamW(optimizer_grouped_parameters, lr=config['learning_rate'], eps=config['adam_epsilon'])
    # t_total = len(train_dataloader) * config['epochs']
    # # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=t_total)

    model, avg_loss, all_epochs_preds, all_epochs_labels = train(model, train_loader, dev_loader, optimizer, config, device)

    if config['calculate_probs']: # useful for data maps
        all_epochs_probs = [calculate_probs(current_preds) for current_preds in all_epochs_preds]

    # Save the model
    #! be cautious not to exhaust memory since we save it every time
    out_dir = config['out_dir']
    checkpoint_name = get_checkpoint_name(config)
    save_path = os.path.join(out_dir, checkpoint_name)
    model.save_pretrained(save_path)

    out_path = os.path.join(save_path, "training_loss.png")
    plt.plot(avg_loss)
    plt.savefig(out_path)

def load_config(path):
    # Open and read the JSON file
    with open(path, 'r') as file:
        config = json.load(file)
    return config

if __name__ == "__main__":
    config_path = os.path.join("conf", "config.json")
    config = load_config(config_path)
    main(config)
    