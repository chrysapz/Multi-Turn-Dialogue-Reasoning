import os
import json

answer_mapping = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3
}

def load_all_samples(base_dir, split):
    
    if split not in ['train', 'dev', 'test']:
        raise ValueError('Split argument can take values train, dev or test')
    
    data_dir = os.path.join(base_dir, split)
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

def load_config(path):
    # Open and read the JSON file
    with open(path, 'r') as file:
        config = json.load(file)
    return config
