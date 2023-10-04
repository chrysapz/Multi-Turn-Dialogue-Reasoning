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
from evaluate import evaluate_data, calculate_probs
from time import gmtime, strftime
from tqdm import tqdm

def train(model, train_loader, dev_loader, optimizer, config, device):
    print('start training')
    epochs = config['epochs']

    epoch_loss = []
    all_epochs_preds = []
    all_epochs_labels = []
    best_r1 = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        cur_epoch_preds = []
        cur_epoch_labels = []
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

            if config['debug']: break #! todo remove this
        
        # Concatenate training predictions and training labels
        cur_epoch_preds = torch.cat(cur_epoch_preds, dim=0)
        cur_epoch_labels = torch.cat(cur_epoch_labels, dim=0)
        all_epochs_preds.append(cur_epoch_preds)
        all_epochs_labels.append(cur_epoch_labels)

        avg_train_loss = total_loss / len(train_loader)
        epoch_loss.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{epochs} | Training Loss: {avg_train_loss}")

        eval_preds, eval_labels, eval_loss, metrics = evaluate_data(model, dev_loader, config, device)
        if metrics != {} and metrics['r1'] > best_r1:
            best_model = deepcopy(model)
            best_epoch = epoch
            best_optimizer = deepcopy(optimizer)
            
    if config['debug']:
        best_model = model
        best_epoch = epoch
        best_optimizer = optimizer
    
    best_model_info = {'model_state_dict': best_model.state_dict(), 'epoch':epoch,'optimizer_state_dict':best_optimizer.state_dict()}

    return best_model_info, epoch_loss, all_epochs_preds, all_epochs_labels


dataset_per_mode = {
    'binary': MutualDataset
}

def main(config):
    print('Training')
    print(config)
    base_dir = os.path.join(config['data_dir'], config['dataset_name'])

    print('transformers version', transformers.__version__)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config['seed'])

    tokenizer = RobertaTokenizerFast.from_pretrained(config['tokenizer_name'], use_fast=True)
    print('loaded tokenizer', strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    dataset_class = dataset_per_mode[config['mode']]

    train_samples = load_all_samples(base_dir, 'train')
    train_dataset = dataset_class(train_samples, tokenizer, config['max_seq_length'])
    print('tokenized train', strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    dev_samples = load_all_samples(base_dir, 'dev')
    dev_dataset = dataset_class(dev_samples, tokenizer, config['max_seq_length'])
    print('tokenized dev', strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    # create the model
    model = AutoModelForSequenceClassification.from_pretrained(config['model_name'], num_labels = 2)
    model = model.to(device)
    print('loaded model', strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    train_collate_fn = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    print('after init train collate fn', strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config['batch_size'], collate_fn=train_collate_fn)
    print('after init train dataloader', strftime("%Y-%m-%d %H:%M:%S", gmtime()))

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
    print('load optimizer')
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config['learning_rate'], eps=config['adam_epsilon'])
    # t_total = len(train_dataloader) * config['epochs']
    # # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=t_total)

    model_info, avg_loss, all_epochs_preds, all_epochs_labels = train(model, train_loader, dev_loader, optimizer, config, device)

    if config['calculate_probs']: # useful for data maps
        all_epochs_probs = [calculate_probs(current_preds) for current_preds in all_epochs_preds]

    # Save the model
    #! be cautious not to exhaust memory since we save it every time
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
    
    #!check that we use the updated optimizer
    a=1

def load_config(path):
    # Open and read the JSON file
    with open(path, 'r') as file:
        config = json.load(file)
    return config

if __name__ == "__main__":
    config_path = os.path.join("conf", "config.json")
    config = load_config(config_path)
    main(config)
    