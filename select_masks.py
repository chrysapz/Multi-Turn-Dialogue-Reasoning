from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import argparse
from data import load_all_samples
import os
from utils import load_pickle, create_dicts_from_tuples, concat_history_with_true_label, create_pickle
from transformers import T5ForConditionalGeneration, T5Tokenizer
from Concat_dataset import Concat_Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import torch
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
NUM_TRAIN_EXAMPLES = 6000

def mask_based_on_keywords(sentence_transformer_model, list_sentences, n_gram_start, n_gram_finish):
    '''
    Return:
        all_key_phrases: list of list of tuples
    '''
    sentence_model = SentenceTransformer(sentence_transformer_model)
    kw_model = KeyBERT(model=sentence_model)

    # Define the step size for the loop
    step_size = 128
    all_key_phrases = [] 
    # Iterate through the list of sentences with a step size of 16
    for i in range(0, len(list_sentences), step_size):
        # Extract keywords for the current batch of sentences
        batch_sentences = list_sentences[i:i+step_size]
        key_phrases = kw_model.extract_keywords(batch_sentences, keyphrase_ngram_range=(n_gram_start, n_gram_finish))
        all_key_phrases.extend(key_phrases)
    print('finished finding masks')
    return all_key_phrases

def remove_ngram(big_string, ngram):
    # Use regular expression to match 'ngram' as a separate word
    pattern = r'\b' + re.escape(ngram) + r'\b'
    output_string = re.sub(pattern, '[MASK]', big_string)
    return output_string

def put_extra_ids_in_correct_order(input_text):
    # Extract all <extra_id> placeholders
    placeholders = re.findall(r'<extra_id_\d+>', input_text)

    # Sort the placeholders based on their numeric values
    sorted_placeholders = sorted(placeholders, key=lambda x: int(x.strip('<>extra_id_')))

    # Replace the placeholders in the original text with the sorted ones
    output_text = re.sub(r'<extra_id_\d+>', lambda x: sorted_placeholders.pop(0), input_text)
    return output_text

def remove_consecutive_duplicates(input_string):
    words = input_string.split()
    result = []

    if len(words) > 0:
        result.append(words[0])

    for i in range(1, len(words)):
        if not words[i] == words[i - 1] == '[MASK]':
            result.append(words[i])

    return ' '.join(result)

def replace_mask_with_id(input_string):
    len_mask = len('[MASK]')
    counter = 0
    new_string = ''
    len_ = len(input_string)
    i = 0  # Initialize i outside the loop

    while i < len_:
        if input_string[i:i + len_mask] == '[MASK]':
            new_string += f'<extra_id_{counter}>'
            counter += 1  # Increment the counter
            i += len_mask  # Move i by the length of the '[MASK]' string
        else:
            new_string += input_string[i]
            i += 1  # Move i by 1 character

    # print(new_string)
    return new_string

def preprocess_for_T5(sentences, list_n_grams, ids):
    masked_samples = []
    all_n_gram_scores = []
    count_masked_words_sentence = []
    dicts = {}

    for n_grams_tuples, sample, id in zip(list_n_grams, sentences, ids):
        sentinel_counter = 0  # Initialize the sentinel counter
        for tmp_n_gram, n_gram_score in n_grams_tuples:
            sample = remove_ngram(sample, tmp_n_gram)
            all_n_gram_scores.append(n_gram_score)

        # we don't want to have <extra_id_0> <extra_id_1>
        new_sample = remove_consecutive_duplicates(sample)
        correct_sample = replace_mask_with_id(new_sample)

        masked_samples.append(correct_sample)
        count_masked_words_sentence.append(sentinel_counter)
        dicts[id] = correct_sample

    mean_n_gram_score = np.mean(all_n_gram_scores)
    print(f'{mean_n_gram_score} = mean ngram score')

    return masked_samples, count_masked_words_sentence, dicts


def t5_inference(dataloader, model, tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    all_sentence_ids = []
    generated_dialogues_ids = []
    for batch in dataloader:
        inputs = {key: value.to(device) for key, value in batch.items() if key not in ['sentence_ids','option_id']}
        sequence_ids = model.generate(**inputs)

        all_sentence_ids.extend(batch['sentence_ids'])   
        generated_dialogues_ids.extend(sequence_ids.detach().cpu())

    texts_generated_dialogues = []
    for sequence_ids in generated_dialogues_ids:
        texts_generated_dialogues.append(tokenizer.batch_decode(sequence_ids))
    
    return  all_sentence_ids, texts_generated_dialogues, generated_dialogues_ids

def create_dialogues(args):
    save_folder = 'dialogue_generated'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # gen = os.path.join(save_folder,'t5_generated1_1.pkl')
    # t5 = load_pickle(gen)

    config = args#vars(args) # convert to dict
    # config['debug'] = True
    base_dir = os.path.join(config['data_dir'], config['dataset_name'])
    initial_train_samples = load_all_samples(base_dir, 'train')

    indexed_train_list = load_pickle('random_ids.pkl')

    indexed_train_list=indexed_train_list[:6000] # don't care about validation
    shuffled_samples = [initial_train_samples[i] for i in indexed_train_list]

    path_name = f'{config["n_gram_start"]}_{config["n_gram_finish"]}.pkl'

    
    llama_val_samples = shuffled_samples[1500:6000]
    llama_val_random_indices = indexed_train_list[1500:6000]

    id2history, id2options, id2label_id = create_dicts_from_tuples(llama_val_samples, llama_val_random_indices)

    samples, ids = concat_history_with_true_label(id2history, id2options, id2label_id)

    if config['debug']:
        samples = samples[:120]

    #id2history correct dicts wrong difference between samples[0] and id2history[982]
    keywords = mask_based_on_keywords('all-distilroberta-v1', samples, config['n_gram_start'], config['n_gram_finish'])

    sent_path = 'keywords_'+path_name
    keywords_samples = list(zip(keywords, samples))
    create_pickle(keywords_samples, sent_path)

    new_samples, count_masks, dicts = preprocess_for_T5(samples, keywords, ids)

    dict_path = 'id2t5_' +path_name
    create_pickle(dicts, dict_path)
    tokenizer = T5Tokenizer.from_pretrained(config['model_name'])
    # print('inside select masks ',dicts[982])
    
    # so that we can be sure that it gives the same result with the one below

    model = T5ForConditionalGeneration.from_pretrained(config['model_name'])
    collate_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True,max_length=256,pad_to_multiple_of=8)
    dataset = Concat_Dataset(new_samples, ids, tokenizer)
    data_loader = DataLoader(dataset, shuffle=False, batch_size=config['batch_size'], collate_fn=collate_fn)

    all_sentence_ids, generated_dialogues, generated_dialogues_ids = t5_inference(data_loader, model, tokenizer)
    
    count_path = 'gen_ids_'+path_name
    create_pickle(generated_dialogues_ids, count_path)


    count_path = 'count_'+path_name
    create_pickle(count_masks, count_path)

    sent_path = 'sent_id_'+path_name
    create_pickle(all_sentence_ids, sent_path)

    gen_path = 't5_generated'+path_name
    create_pickle(generated_dialogues, gen_path)

    return dicts, keywords_samples, dicts, generated_dialogues_ids, count_masks, all_sentence_ids, generated_dialogues

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="My Argument Parser")

    # Add arguments with default values manually
    parser.add_argument("--mode", type=str, default="binary")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--dataset_name", type=str, default="mutual")
    parser.add_argument('--debug', action='store_true',help='default is not to debug')
    #! for snellius google/t5-v1_1-xl which is 3B
    parser.add_argument('--model_name',type=str, default="t5-small")
    parser.add_argument('--batch_size',type=int, default=64)
    # Add the n_gram_start argument
    parser.add_argument(
        "--n_gram_start",
        type=int,
        default=1,
        help="The starting value for n-grams (default: 3)."
    )
    
    # Add the n_gram_finish argument
    parser.add_argument(
        "--n_gram_finish",
        type=int,
        default=1,
        help="The finishing value for n-grams (default: 3)."
    )
    args = parser.parse_args()
    create_dialogues(args)