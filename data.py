import os
import json
from transformers import AutoTokenizer
from mutual_dataset import MutualDataset

answer_mapping = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3
}

def load_all_samples(base_dir,mode):
    data_dir = os.path.join(base_dir, mode)
    samples = [] 
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            sample = load_sample_from_file(os.path.join(data_dir, filename))
            samples.append(sample)
    return samples

def load_sample_from_file(filename):
    with open(filename, 'r') as file:
        data = json.loads(file.read())
        data['answers'] = answer_mapping[data['answers']]
    return data['answers'], data['options'], data['article']


def tokenize_roberta_data(data, tokenizer, max_seq_length):
    """
    We feed into BERT and do binary classification:
        history, option_1 -> BERT -> (p, 1-p) loss
        history, option_2 -> BERT -> (p, 1-p) loss
        etc. seperately 

    Alternatively, we could 
        history, option_1 -> BERT -> (p_1, 1-p) 
        history, option_2 -> BERT -> (p_2, 1-p) 
        etc. for options 3, 4 

        one loss function (p_1, p_2, p_3, p_4) where first element always the true label

    """

    tokenized_input_ids = []
    tokenized_attention_mask = []
    option_flags = [] # 0 or 1 depending on whether it is the correct choice or not
    sentences_id = []
    options_id = []
    for sentence_id, (label_id, options, context_history) in enumerate(data):
        for option_id, option in enumerate(options):
            '''done similarly here https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_xnli.py#L337'''
            # sep_token_id added between the 2 sentences
            tokenizer_dict = tokenizer(context_history, option, truncation=True, max_length = max_seq_length)
            
            #!todo check whether bert considers 0 or 1 as the correct choice
            option_flag = 1 if label_id == option_id else 0 # check whether the option is correct

            tokenized_input_ids.append(tokenizer_dict['input_ids'])
            tokenized_attention_mask.append(tokenizer_dict['attention_mask'])
            option_flags.append(option_flag)
            sentences_id.append(sentence_id)
            options_id.append(option_id)

    return tokenized_input_ids, tokenized_attention_mask, option_flags, sentences_id, options_id

def create_dataset(base_dir, split, tokenizer, max_seq_length):
    """
        Split argument can take values 'train', 'dev', 'test'
    """
    if split not in ['train', 'dev', 'test']:
        raise ValueError('Split argument can take values train, dev or test')
    split_samples = load_all_samples(base_dir, split)
    tokenized_input_ids, tokenized_attention_mask, option_flags, sentence_ids, option_ids = tokenize_roberta_data(split_samples, tokenizer, max_seq_length)
    dataset = MutualDataset(tokenized_input_ids, tokenized_attention_mask, option_flags, sentence_ids, option_ids)
    return dataset

def main(config):
    print(config)
    base_dir = os.path.join(config['data_dir'], config['dataset_name'])
    max_seq_length = config['max_seq_length']
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    print("Creating training set...")
    train_dataset = create_dataset(base_dir, 'train', tokenizer, max_seq_length)
    print("DONE")
    print("Creating validation set...")
    val_dataset = create_dataset(base_dir, 'dev', tokenizer, max_seq_length)
    print("DONE")
    # test_dataset = create_dataset(base_dir, 'test', tokenizer, max_seq_length) # maybe useless since we don't have the labels

def load_config(path):
    # Open and read the JSON file
    with open(path, 'r') as file:
        config = json.load(file)
    return config

#This is helpful right now. Later on it will be moved to the training script
if __name__=='__main__':
    # Open and read the JSON file
    config_path = os.path.join("conf", "config.json")
    config = load_config(config_path)
    main(config)