from sentence_transformers import SentenceTransformer, util #! pip install -U sentence-transformers
import argparse
import pickle
import numpy as np
import os
import torch
from utils import load_pickle, create_pickle
from manual_filtering import preprocess_augmented_labels
from tqdm import tqdm

MODEL_2_KEY = {
    'all-distilroberta-v1':'dist_rob' # This model is a distilled version of Roberta finetuned on several IR taks.
}


metric_functions = {
    'cosine': util.cos_sim,
    'dot': util.dot_score
}


def extract_lists_from_dict(sentences_info):# Extract sentences and labels from batch_generated_info
    """
    Extract generated sentences and labels from a dictionary.

    Args:
        sentences_info (dict): A dictionary of a dictionary containing sentence information.

    Returns:
        tuple: A tuple containing lists of sentence IDs, generated texts, and labels.

    Note:
        Every dictionary of a dictionary should include the gen_text and true_label keys
    """
    generated_texts = []
    labels = []
    sents_id = []
    for sent_id, info in sentences_info.items():
        generated_texts.append(info['gen_text'])
        labels.append(info['true_label'])
        sents_id.append(sent_id)
    return sents_id, generated_texts, labels

def calculate_similarities(batch_size, sentences_info, model_name, metric):
    """
    Calculate similarities between generated texts and labels using a specific model and metric.

    Args:
        batch_size (int): The batch size for similarity calculations.
        sentences_info (dict): For each sentence_id keep a seperate dictionary containing sentence information including the label (the true label) and the gen_text (the generated text from llama)
        model_name (str): The name of the SentenceTransformer model to use.
        metric (str): The similarity metric to use ('cosine' or 'dot').

    Returns:
        sentences_info (dict): An updated dictionary of a dictionary containing sentence information with similarity scores added.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name, device=device)
    new_info_key = f'{MODEL_2_KEY[model_name]}_{metric}' # name of the new key about similarities

    sents_id, generated_texts, labels = extract_lists_from_dict(sentences_info)

    # Initialize lists to store scores
    scores_list = []
   # Process data in batches
    for i in tqdm(range(0, len(generated_texts), batch_size)):
        batch_generated = generated_texts[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        batch_sent_ids = sents_id[i:i+batch_size]

        #Compute embedding for both lists
        embeddings1 = model.encode(batch_generated, convert_to_tensor=True, device=device)
        embeddings2 = model.encode(batch_labels, convert_to_tensor=True, device=device)
        
        batch_scores = metric_functions[metric](embeddings1, embeddings2)

        # Update batch_generated_info with cosine similarities
        for batch_id, (sent_id, score) in enumerate(zip(batch_sent_ids, batch_scores)):
            cur_score = score[batch_id].item()
            sentences_info[sent_id][new_info_key] = cur_score
            scores_list.append(cur_score)

    # Calculate average and standard deviation
    avg_score = np.mean(scores_list)
    std_score = np.std(scores_list)
    print(f'Average similarity ', avg_score, ' std ', std_score)

    return sentences_info

def main(args):
    path = os.path.join('generated_text',args.pickle_name)
    sentences_info = load_pickle(path) 
    
    preprocessed_sentences_info = preprocess_augmented_labels(sentences_info, train_id2options)

    sentences_info = calculate_similarities(args.batch_size, preprocessed_sentences_info, args.model_name, args.metric)
    # we save in the same pickle
    new_pickle_name = args.pickle_name
    create_pickle(sentences_info, path)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Calculate similarities between sentences using various models and metrics.")
    
    # Argument for the path to the pickle file including the sentence information
    parser.add_argument("--pickle_name", type=str, default="only_inference_colab_greedy_decoding.pkl", help="Name of the pickle file containing sentence information.")
    
    # Argument for batch size
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for similarity calculations.")
    
    # Argument for the model name
    parser.add_argument("--model_name", type=str, default="all-distilroberta-v1", help="Name of the model to use for similarity calculations. \
                        Check also the sentence-transformers library https://www.sbert.net/docs/pretrained_models.html")
    
    # Argument for the metric
    parser.add_argument("--metric", type=str, default="cosine", choices=['cosine', 'dot'], help="Similarity metric to use for calculations (choose from 'cosine' or 'dot').")    
    args = parser.parse_args()
    main(args)