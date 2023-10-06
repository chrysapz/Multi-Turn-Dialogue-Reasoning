import os
import json

answer_mapping = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3
}

def load_all_samples(base_dir, split):
    """
    Load and parse samples from text files in a specified directory.
    
    Args:
        base_dir (str): The base directory containing the split subdirectories.
        split (str): The dataset split to load samples from, should be 'train', 'dev', or 'test'.
        
    Returns:
        samples (list): A list of loaded samples, where each sample is a tuple containing the answer index (0-3),
              a list of answer options, and an article.
    
    Raises:
        ValueError: If the 'split' argument is not one of 'train', 'dev', or 'test'.
    """
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
    """
    Load a single sample from a JSON file.
    
    Args:
        filename (str): The path to the JSON file containing the sample data.
        
    Returns:
        tuple: A tuple containing the answer index (0-3), a list of answer options, and an article.
    """
    with open(filename, 'r') as file:
        data = json.loads(file.read())
        data['answers'] = answer_mapping[data['answers']]
    return data['answers'], data['options'], data['article']

def load_config(path):
    """
    Load a configuration from a JSON file.
    
    Args:
        path (str): The path to the JSON file containing the configuration data.
        
    Returns:
        dict: A dictionary containing the configuration data.
    """
    with open(path, 'r') as file:
        config = json.load(file)
    return config
