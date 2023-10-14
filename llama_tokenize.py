# based on https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L22

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict
import re
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
# from llama.tokenizer import Tokenizer

# these are special tokens used by the tokenizer
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

import torch.utils.data as data


def text2dialogformat(sentences):
        """ 
        Convert context history to LLAMA dialog format. 
        Check also https://github.com/facebookresearch/llama/blob/main/example_chat_completion.py

        Args:
            sentences (list): List of tuples containing the answer index (0-3), a list of answer options, and an article.

        Returns:
            tuple: A tuple containing:
                a. all_dialogs (list): list of lists where each element in the inner list is a dictionary containing role ("user" or "assistant") and content (text)
                b. all_answers (list): list of strings containing the text of the true option
                c. without_dummy_flag (list): list of binary values indicating whether we added a dummy text (e.g. hello) in the beginning (value 0) or not (value 1)
        
        Note:
            We consider that when we find "m :" or "f :" in the context history, the speaker changes.
        """
 
        all_dialogs = []
        all_answers = []
        without_dummy_flag = [] # if we don't include a dummy text in the beginning of the context put 1 otherwise put 0
        for label_id, options, initial_context_history in sentences:
            unprocessed_correct_option = options[label_id]

            correct_option = preprocess(unprocessed_correct_option)

            all_answers.append(correct_option[4:])

            # add new line at the end of each sentence
            context_history = preprocess(initial_context_history)

            m_positions = [m.start() for m in re.finditer('m :', context_history)]
            f_positions = [m.start() for m in re.finditer('f :', context_history)]

            # check that there is always a space in between
            assert('m:' not in context_history)
            assert('f:' not in context_history)

            sorted_positions = sorted(m_positions+f_positions)
            all_turns = [] # list of dicts
            if len(sorted_positions) % 2 != 0: # if odd user assistant user
                without_dummy_flag.append(1)
                for i, pos in enumerate(sorted_positions):
                    role = 'user'  if i % 2==0 else 'assistant'

                    next_pos = sorted_positions[i+1] if i < len(sorted_positions)-1 else len(context_history)
                    # +4 because the strings 'm :' and 'f :' are 3 chars
                    turn_dict = {'role': role, 'content': context_history[pos+4 : next_pos] }
                    all_turns.append(turn_dict)
            else: # if even assistant user assistant user
                without_dummy_flag.append(0)
                dummy_hello = {'role': 'user', 'content': 'Hello . \n' }
                all_turns.append(dummy_hello)
                for i, pos in enumerate(sorted_positions):
                    role = 'user'  if i % 2 !=0 else 'assistant'

                    next_pos = sorted_positions[i+1] if i < len(sorted_positions)-1 else len(context_history)
                    # +4 because the strings 'm :' and 'f :' are 3 chars
                    turn_dict = {'role': role, 'content': context_history[pos+4 : next_pos] }
                    all_turns.append(turn_dict)

            all_dialogs.append(all_turns)
        
        return all_dialogs, all_answers, without_dummy_flag

def preprocess(txt):
    """
    Split the context history text into sentences, put newline symbol in between. Then, join and strip.
    See https://huggingface.co/meta-llama/Llama-2-7b-chat about what kind of preprocessing is needed

    Args:
        txt (str): The text.

    Returns:
        new_text (str): Preprocessed  text.
    """
    split_sentences = re.split(r'(?<=[.!?])\s+', txt)

    new_text = '\n '.join(split_sentences)

    new_text = new_text.strip()
    return new_text

def tokenize_add(tokenizer, prompt, answer):
        """
        Tokenize for a prompt and answer pair.
        The format of every sentence should be: bos_id B_INST text_1 E_INST text_2 eos_id

        Args:
            prompt (dict): The prompt dictionary containing role (user or assistant) and content (text) keys
            answer (dict): The answer dictionary containing role (user or assistant) and content (text) keys

        Returns:
            tokenized_text (dict): The tokenized dictionary including the input_ids and the attention_masks as keys
        """
        tokenized_text = tokenizer.encode_plus(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",add_special_tokens =True)
        tokenized_text['input_ids'].append(tokenizer.eos_token_id)
        tokenized_text['attention_mask'].append(1)
        return tokenized_text

def tokenize_text(
        tokenizer,
        dialogs, labels, do_generate, use_context
    ):
        """
        Tokenize a list of dialogues and corresponding labels for model training or generation.

        Args:
            dialogs (list): list of lists where each element in the inner list is a dictionary containing role ("user" or "assistant") and content (text)
            labels (list): list of strings containing the text of the true option of each dialogue
            do_generate (bool): A boolean flag indicating whether we want to use model.generate()
            use_context (bool): A boolean flag indicating whether we ignore context in cross entropy
        Returns:
            tuple: A tuple containing three elements:
                a. prompt_tokens (list): List of lists containing input_ids i.e. the tokenized dialogue prompts.
                b. attention_masks (list): List of list attention masks for the tokenized dialogue prompts.
                c. label_ids (list): List of tokenized label IDs for the dialogue prompts.

        Note:
            

        """
        prompt_tokens = []
        unsafe_requests = []
        attention_masks = []
        label_ids = []
        c=0
        for label_text, dialog in zip(labels, dialogs):
            unsafe_requests.append(
                any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
            )
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS
                        + dialog[0]["content"]
                        + E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]

            # if the dialog list has odd number of elements then we don't tokenize here the last element
            dialog_tokens: List[int] = sum(
                [
                    #! eos token id should we also add \n text?
                    tokenize_add(tokenizer,prompt, answer)['input_ids']
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )

            # if the context dialog list has odd number of elements then we will have to add bos_id B_INST text_1 E_INST to add the last text

            if len(dialog) % 2 != 0:
                #! we also add bos id but not eos id
                last_context_tokens = tokenizer.encode(
                    f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",add_special_tokens=True
                )

                dialog_tokens += last_context_tokens

            # cross entropy should ignore the context history so we put -100 there
            dummy_label_ids = [-100]*len(dialog_tokens) if not use_context else dialog_tokens.copy()

            cur_label_ids = []
            # 
            if len(dialog) % 2 == 0: # even context history we use special identifiers
                cur_label_ids.append(tokenizer.bos_token_id)
                b_inst_ids = tokenizer.encode(B_INST, add_special_tokens=False)
                cur_label_ids.extend(b_inst_ids)

            # add label
            cur_label_ids.extend(tokenizer.encode(label_text.strip() , add_special_tokens=False)  )
            cur_label_ids.append(tokenizer.eos_token_id)

            # we add the labels to the input to the model since we are doing next word prediction of the whole sentence during training
            if not do_generate:
                dialog_tokens.extend(cur_label_ids)

            # now add to the big lists
            prompt_tokens.append(dialog_tokens)
            attention_masks.append([1] * len(prompt_tokens[-1]) )
            label_ids.append(dummy_label_ids + cur_label_ids)
            # # check that all have the same size
            # assert(len(prompt_tokens[-1]) == len(attention_masks[-1]) == len(label_ids[-1]) )

        return prompt_tokens, attention_masks, label_ids


class Llama_dataset(data.Dataset):
    """
    Custom PyTorch dataset for handling LLAMA-style dialogue data.
    
    Args:
        tokenizer (AutoTokenizer): The tokenizer for tokenizing text data.
        split_samples (list): A list of tuples containing label_id, options, and initial_context_history.
        do_generate (bool): Whether to use model.generate() or not.
        ignore_context

    Attributes:
        tokenizer (AutoTokenizer): The tokenizer for tokenizing text data.
        all_input_ids (list): List of input token IDs for the dataset.
        all_attention_masks (list): List of attention masks for the dataset.
        all_labels (list): List of label IDs for the dataset.
        without_dummy_flag (list): List indicating whether dummy tokens are present in the dataset.
    """
    def __init__(self, tokenizer: AutoTokenizer, split_samples, do_generate, use_context):
        self.tokenizer = tokenizer
        dialogs, all_answers, without_dummy_flag = text2dialogformat(split_samples)
        all_input_ids, all_attention_masks, all_labels = tokenize_text(tokenizer, dialogs, all_answers, do_generate, use_context)
        self.all_input_ids = all_input_ids
        self.all_attention_masks = all_attention_masks
        self.all_labels = all_labels
        self.without_dummy_flag = without_dummy_flag

    def __len__(self):
        """
        Get the number of data points in the dataset.

        Returns:
            int: Number of data points.
        """
        return len(self.all_input_ids)

    def __getitem__(self, idx):
        """
        Get a specific data point from the dataset.

        Args:
            idx (int): Index of the data point to retrieve.

        Returns:
            tuple: A tuple containing input_ids, attention_mask, and label for the data point.
        """
        input_ids = self.all_input_ids[idx]
        attention_mask = self.all_attention_masks[idx]
        label = self.all_labels[idx]
        return input_ids, attention_mask, label

class Llama_next_word_dataset(data.Dataset):
    """
    Custom PyTorch dataset for handling LLAMA-style dialogue data.
    
    Args:
        tokenizer (AutoTokenizer): The tokenizer for tokenizing text data.
        split_samples (list): A list of tuples containing label_id, options, and initial_context_history.
        do_generate (bool): Whether to use model.generate() or not.
        ignore_context

    Attributes:
        tokenizer (AutoTokenizer): The tokenizer for tokenizing text data.
        all_input_ids (list): List of input token IDs for the dataset.
        all_attention_masks (list): List of attention masks for the dataset.
        all_labels (list): List of label IDs for the dataset.
        without_dummy_flag (list): List indicating whether dummy tokens are present in the dataset.
    """
    def __init__(self, tokenizer: AutoTokenizer, split_samples, do_generate, use_context):
        self.tokenizer = tokenizer
        dialogs, all_answers, _ = text2dialogformat(split_samples)
        all_input_ids, _, _ = tokenize_text(tokenizer, dialogs, all_answers, do_generate, use_context)
        self.all_input_ids = all_input_ids
        # self.all_attention_masks = all_attention_masks
        # self.all_labels = all_labels
        # self.without_dummy_flag = without_dummy_flag

    def __len__(self):
        """
        Get the number of data points in the dataset.

        Returns:
            int: Number of data points.
        """
        return len(self.all_input_ids)

    def __getitem__(self, idx):
        """
        Get a specific data point from the dataset.

        Args:
            idx (int): Index of the data point to retrieve.

        Returns:
            tuple: A tuple containing input_ids, attention_mask, and label for the data point.
        """
        input_ids = self.all_input_ids[idx]
        return input_ids
        #return input_ids, attention_mask
        #return {"input_ids": input_ids, 'attention_mask':attention_mask,'labels': label}

# it's not possible to pass sentences_id in the HF trainer so I create a subclass for inference
class Llama_with_sent_ids_dataset(Llama_dataset):
    """
    A dataset class for Llama models with sentence IDs.

    This class extends the base Llama_dataset class and adds support for including sentence IDs
    in the dataset.

    Args:
        tokenizer (AutoTokenizer): The tokenizer to use for tokenizing the input data.
        split_samples (list): The split samples for the dataset.
        do_generate (bool): A flag indicating whether to use model.generate() or not.
        dev_ids: A list of sentence IDs corresponding to the relevant samples in the dataset.

    Attributes:
        all_sentences_id: A list of sentence IDs corresponding to the relevant samples in the dataset.
    """
    def __init__(self, tokenizer: AutoTokenizer, split_samples, do_generate, dev_ids, use_context):
        """
        Initialize the Llama_with_sent_ids_dataset.

        Args:
            tokenizer (AutoTokenizer): The tokenizer to use for tokenizing the input data.
            split_samples: The split samples for the dataset.
            do_generate (bool): A flag indicating whether to use model.generate() or not.
            dev_ids: A list of sentence IDs corresponding to the samples in the dataset.
        
        Note:
            We pass the dev_ids as they are before shuffling!
        """
        super().__init__(tokenizer, split_samples, do_generate, use_context)
        all_sentences_id = dev_ids
        self.all_sentences_id = all_sentences_id

    def __getitem__(self, idx):
        """
        Get an item from the dataset by index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing input_ids, attention_mask, label, sentence_id, and without_dummy flag (1 if we didn't add any dummy text at the beginning).
        """
        input_ids = self.all_input_ids[idx]
        attention_mask = self.all_attention_masks[idx]
        label = self.all_labels[idx]
        sentence_id = self.all_sentences_id[idx]
        without_dummy = self.without_dummy_flag[idx]

        return input_ids, attention_mask, label, sentence_id, without_dummy
