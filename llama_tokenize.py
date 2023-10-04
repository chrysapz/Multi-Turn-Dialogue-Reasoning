# based on https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L22

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
# from llama.tokenizer import Tokenizer

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str



class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

import torch.utils.data as data

class Llama_dataset(data.Dataset):
    
    def __init__(self, tokenizer: AutoTokenizer, split_samples, do_generate):
        self.tokenizer = tokenizer
        dialogs, all_answers = self.text2dialogformat(split_samples)
        all_input_ids, all_attention_masks, all_labels = self.tokenize_text(dialogs, all_answers, do_generate)
        self.all_input_ids = all_input_ids
        self.all_attention_masks = all_attention_masks
        self.all_labels = all_labels

    def __len__(self):
        # Number of data point we have
        return len(self.all_input_ids)

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        input_ids = self.all_input_ids[idx]
        attention_mask = self.all_attention_masks[idx]
        label = self.all_labels[idx]
        
        # return input_ids, attention_mask, label
        return input_ids, attention_mask, label
        # return {"input_ids": input_ids, 'attention_mask':attention_mask,'labels': label}

    def preprocess(self, initial_context_history, is_option=False):

        split_sentences = re.split(r'(?<=[.!?])\s+', initial_context_history)
        # if is_option:
        #     if(len(split_sentences) != 1):
        #         print('more')
        # Add '\n' to the end of each sentence and join them back together
        context_history = ' \n'.join(split_sentences)

        context_history = context_history.strip()
        return context_history

    def text2dialogformat(self, sentences):
        #! we preprocess only the context here
        # see here about how to preprocess https://huggingface.co/meta-llama/Llama-2-7b-chat
        all_dialogs = []
        all_answers = []
        for label_id, options, initial_context_history in sentences:
            unprocessed_correct_option = options[label_id]

            correct_option = self.preprocess(unprocessed_correct_option, True)

            all_answers.append(correct_option)
            # history_with_label = initial_context_history +' '+correct_option

            # add new line at the end of each sentence
            context_history = self.preprocess(initial_context_history)

            # split_sentences = re.split(r'(?<=[.!?])\s+', initial_context_history)

            # # Add '\n' to the end of each sentence and join them back together
            # context_history = ' \n'.join(split_sentences)

            # context_history = context_history.strip()

            m_positions = [m.start() for m in re.finditer('m :', context_history)]
            f_positions = [m.start() for m in re.finditer('f :', context_history)]

            # check that there is always a space in between
            assert('m:' not in context_history)
            assert('f:' not in context_history)

            sorted_positions = sorted(m_positions+f_positions)
            all_turns = []
            if len(sorted_positions) % 2 != 0: # if odd user assistant user
                for i, pos in enumerate(sorted_positions):
                    role = 'user'  if i % 2==0 else 'assistant'

                    next_pos = sorted_positions[i+1] if i < len(sorted_positions)-1 else len(context_history)
                    # +4 because the strings 'm :' and 'f :' are 3 chars
                    turn_dict = {'role': role, 'content': context_history[pos+4 : next_pos] }
                    all_turns.append(turn_dict)
            else: # if even assistant user assistant user
                dummy_hello = {'role': 'user', 'content': 'Hello . \n' }
                all_turns.append(dummy_hello)
                for i, pos in enumerate(sorted_positions):
                    role = 'user'  if i % 2 !=0 else 'assistant'

                    next_pos = sorted_positions[i+1] if i < len(sorted_positions)-1 else len(context_history)
                    # +4 because the strings 'm :' and 'f :' are 3 chars
                    turn_dict = {'role': role, 'content': context_history[pos+4 : next_pos] }
                    all_turns.append(turn_dict)

            all_dialogs.append(all_turns)
        
        return all_dialogs, all_answers

    def tokenize_add(self, prompt, answer):
        #! always pairs
        tokenized_text = self.tokenizer.encode_plus(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",add_special_tokens =True)
        tokenized_text['input_ids'].append(self.tokenizer.eos_token_id)
        tokenized_text['attention_mask'].append(1)
        return tokenized_text

    def tokenize_text(
        self,
        dialogs, labels, do_generate
    ):
 
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
            # if the dialog list has even number of elements 
            dialog_tokens: List[int] = sum(
                [
                    #! eos token id should we also add \n text?
                    self.tokenize_add(prompt, answer)['input_ids']
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            # # assert (
            # #     dialog[-1]["role"] == "user"
            # # ), f"Last message must be from user, got {dialog[-1]['role']}"
            # if dialog[-1]["role"] != "user":
            #     print('Last message must be from user, got assistant')
            #     c+=1
            
            # if the context dialog list has even number of elements then we will end up adding this twice
            if len(dialog) % 2 != 0:
                #! we also add bos id but not eos id
                last_context_tokens = self.tokenizer.encode(
                    f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",add_special_tokens=True
                )

                dialog_tokens += last_context_tokens
            # cross entropy should ignore the context history so we put -100 there
            dummy_label_ids = [-100]*len(dialog_tokens)
            #todo rename to option_ids for more clarity
            cur_label_ids = []
            if len(dialog) % 2 == 0: # even context history we use special identifiers
                cur_label_ids.append(self.tokenizer.bos_token_id)
                b_inst_ids = self.tokenizer.encode(B_INST, add_special_tokens=False)
                cur_label_ids.extend(b_inst_ids)

            # add label
            cur_label_ids.extend(self.tokenizer.encode(label_text.strip() , add_special_tokens=False)  )
            cur_label_ids.append(self.tokenizer.eos_token_id)

            # we add the labels to the input to the model since we are doing next word prediction of the whole sentence during training
            if not do_generate:
                dialog_tokens.extend(cur_label_ids)

            # now add to the big lists
            prompt_tokens.append(dialog_tokens)
            attention_masks.append([1] * len(prompt_tokens[-1]) )
            label_ids.append(dummy_label_ids + cur_label_ids)
            # # check that all have the same size
            # assert(len(prompt_tokens[-1]) == len(attention_masks[-1]) == len(label_ids[-1]) )

        print(f' {c} times last message from assistant')
        return prompt_tokens, attention_masks, label_ids

# it's not possible to pass sentences_id in the HF trainer so I create a subclass for inference
class Llama_with_sent_ids_dataset(Llama_dataset):
    
    def __init__(self, tokenizer: AutoTokenizer, split_samples, do_generate, dev_ids):
        super().__init__(tokenizer, split_samples, do_generate)
        all_sentences_id = dev_ids
        self.all_sentences_id = all_sentences_id

    def __getitem__(self, idx):
        input_ids, attention_mask, label = super().__getitem__(idx)
        sentence_id = self.all_sentences_id[idx]
        return input_ids, attention_mask, label, sentence_id

# if __name__=='__main__':
#     [{'role':'user','content':'first text user1'}, {'role':'assistant','content':'first text user 2'}]
#     tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b')
#     llama_preprocesing = Llama_dataset(tokenizer)
#     llama_preprocesing.tokenize_text('')
