from copy import deepcopy
import re

turn_start = [
    'f :',
    'm :',
]

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
    add_start_generated_info = {} 
    for sent_id in generated_info:
        generated_text = generated_info[sent_id]['gen_text'] # string 
        # if initial_generated_text == '': 
        #     counter_empty += 1
        #     continue # we won't add this generated txt

        add_start_generated_info[sent_id] = deepcopy(generated_info[sent_id])
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
    for sent_id in generated_info:
        initial_generated_text = generated_info[sent_id]['gen_text'] # string 
        preprocessed_generated = initial_generated_text.replace('\n', '')

        if preprocessed_generated == '': 
            counter_empty += 1
            continue # we won't add this generated txt

        preprocessed_generated_info[sent_id] = deepcopy(generated_info[sent_id])

        # generated_text_with_start = matching_start + ' '+ remove_new_lines
        preprocessed_generated_info[sent_id]['gen_text'] = preprocessed_generated
    
    print(f'total empty responses {counter_empty}')
    return preprocessed_generated_info

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

'''
import nltk

def remove_last_sentence(input_text):
    # Download the Punkt tokenizer if you haven't already
    nltk.download('punkt')

    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(input_text)

    # Remove the last sentence
    if sentences:
        sentences.pop()

    # Reconstruct the text without the last sentence
    cleaned_text = ' '.join(sentences)
    return cleaned_text

'''