from copy import deepcopy
import re
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
turn_start = [
    'f :',
    'm :',
]
nltk.download('punkt')

DELIMETERS = [".", "!", "?",". ", "! ", "? "]

def truncate_last_sentence(text):
    # Split the text into sentences using the pattern.
    sentences = sent_tokenize(text)

    # Check if the last sentence ends with any of the specified delimiters.
    last_sentence = sentences[-1]
    removed_flag = False
    truncated_text = text
    if not any(last_sentence.endswith(delimiter) for delimiter in DELIMETERS):
        # If not, truncate the last sentence.
        for delimiter in DELIMETERS:
            last_sentence = last_sentence.rsplit(delimiter, 1)[0]
            removed_flag = True

        # Reconstruct the text with truncated last sentence.
        truncated_text = ' '.join(sentences[:-1])
        # if removed_flag:
        #     print(last_sentence)

    return truncated_text, removed_flag
                              

def remove_start_true_labels(generated_info):
    '''
    useful for semantic similarity
    '''
    add_start_generated_info = {} 
    for sent_id in generated_info:
        true_label = generated_info[sent_id]['true_label'] # string 
        # if initial_generated_text == '': 
        #     counter_empty += 1
        #     continue # we won't add this generated txt

        add_start_generated_info[sent_id] = deepcopy(generated_info[sent_id])
        if true_label[:4] == 'm : ' or true_label[:4] == 'f : ':
            add_start_generated_info[sent_id]['true_label'] = true_label[4:]
    
    return add_start_generated_info


def add_start_to_augmented_labels(generated_info, train_id2options):
    '''
    Add 'f :' or 'm :' based on what was used in the given options
    '''
    add_start_generated_info = {} 
    for sent_id in generated_info:
        generated_text = generated_info[sent_id]['gen_text'].strip() # string 
        # if initial_generated_text == '': 
        #     counter_empty += 1
        #     continue # we won't add this generated txt

        add_start_generated_info[sent_id] = deepcopy(generated_info[sent_id])
        # check whether the options have m : or f : and put the same
        existing_options = train_id2options[sent_id]  # list  
        for possible_start in turn_start:
            if existing_options[0].startswith(possible_start):
                matching_start = possible_start
                break

        generated_text_with_start = matching_start + ' '+ generated_text
        add_start_generated_info[sent_id]['gen_text'] = generated_text_with_start
    
    return add_start_generated_info

def preprocess_augmented_labels(generated_info):
    preprocessed_generated_info = {}
    counter_empty = 0
    counter_truncate_last_sentence = 0
    for sent_id in generated_info:
        initial_generated_text = generated_info[sent_id]['gen_text'] # string 
        preprocessed_generated = initial_generated_text.replace('\n', '')

        if preprocessed_generated == '': 
            counter_empty += 1
            continue # we won't add this generated txt
        
        truncated_generated, removed_flag = truncate_last_sentence(preprocessed_generated)
        counter_truncate_last_sentence += removed_flag
        preprocessed_generated_info[sent_id] = deepcopy(generated_info[sent_id])

        # generated_text_with_start = matching_start + ' '+ remove_new_lines
        preprocessed_generated_info[sent_id]['gen_text'] = truncated_generated
    
    print(f'total empty generated responses we removed: {counter_empty}')
    return preprocessed_generated_info

def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity = intersection / union
    return similarity

def data_statistics(tokenizer, generated_info):
    ratios = [] 
    generated_lens = []
    labels_lens = []
    subwords_overlap = []
    for sent_id in generated_info:
        current_generated_info = generated_info[sent_id]
        generated_text = current_generated_info['gen_text']
        true_label = current_generated_info['true_label']

        gen_ids = tokenizer.encode(generated_text,add_special_tokens=False)
        generated_text_len = len(gen_ids)

        true_ids = tokenizer.encode(true_label,add_special_tokens=False)
        true_label_len = len(true_ids)

        jacc = jaccard_similarity(gen_ids, true_ids)

        tmp_ratio = generated_text_len / true_label_len
        ratios.append(tmp_ratio)

        generated_lens.append(generated_text_len)
        labels_lens.append(true_label_len)
        subwords_overlap.append(jacc)

    print(f'generated text mean length {np.mean(generated_lens)} +- {np.std(generated_lens)}')
    print(f'label text mean length {np.mean(labels_lens)} +- {np.std(labels_lens)}')
    print(f'avg jaccard similarity', jacc)

    print('ratio = generated_text_length / true_label_length')
    print(f'ratio mean  {np.mean(ratios)} +- {np.std(ratios)}')
    return ratios

def remove_last_sentence(generated_info):
    remove_end_generated_info = deepcopy(generated_info)
    for sent_id in generated_info:
        # Use regular expressions to split the text into sentences
        generated_text = generated_info[sent_id]['gen_text'] # string 
        sentences = re.split(r'[.!?]', generated_text)

        # Remove any empty strings resulting from the split
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        # Remove the last sentence if there is more than one sentence
        if len(sentences) > 1:
            sentences.pop()

        # Reconstruct the text without the last sentence
        cleaned_text = ' '.join(sentences)

        remove_end_generated_info[sent_id]['gen_text'] = cleaned_text
        print(f'cleaned text {cleaned_text} ')
        print(f'initial text {generated_text}')
        print('*'*12)

    return remove_end_generated_info

def remove_using_similarity(new_generated_info, avg_score, sim_key):
    updated_generated_info = {}
    for sent_id in new_generated_info:
        if new_generated_info[sent_id][sim_key] > avg_score:
            updated_generated_info[sent_id] = deepcopy(new_generated_info[sent_id])
    return updated_generated_info