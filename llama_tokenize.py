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


class Llama_dataset:
    
    def __init__(self, tokenizer: AutoTokenizer, split_samples, max_length):
        self.tokenizer = tokenizer
        dialogs = self.text2dialogformat(split_samples)
        tokenized_texts =self.tokenize_text(dialogs, max_length)
        self.tokenized_texts = tokenized_texts
        

    def __len__(self):
        # Number of data point we have
        return len(self.tokenized_text)

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        tokenized_text = self.tokenized_texts[idx]
        attention_mask = self.attention_mask[idx]
        sentence_id = self.sentence_ids[idx]
        # return input_ids, attention_mask, label
        return {"input_ids": input_ids, 'attention_mask':attention_mask,'sentence_id':
                sentence_id}


    def text2dialogformat(self, sentences):
        # see here about how to preprocess https://huggingface.co/meta-llama/Llama-2-7b-chat
        all_dialogs = []
        for _, _, initial_context_history in sentences:
            # add new line at the end of each sentence
            # splittext = initial_context_history.split(".") # every sentence is an element in the list
            # for x in range(len(splittext)):
            #     splittext[x] = splittext[x]+ ". \n"#+splittext[x].lstrip()
            # context_history = "".join(splittext)

            # Split the text into sentences based on '.', '!', or '?'
            split_sentences = re.split(r'(?<=[.!?])\s+', initial_context_history)

            # Add '\n' to the end of each sentence and join them back together
            context_history = ' \n '.join(split_sentences)

            context_history = context_history.strip()

            m_positions = [m.start() for m in re.finditer('m :', context_history)]
            f_positions = [m.start() for m in re.finditer('f :', context_history)]

            # check that there is always a space in between
            assert('m:' not in context_history)
            assert('f:' not in context_history)

            sorted_positions = sorted(m_positions+f_positions)
            all_turns = []
            for i, pos in enumerate(sorted_positions):
                role = 'user'  if i % 2==0 else 'assistant'

                next_pos = sorted_positions[i+1] if i < len(sorted_positions)-1 else len(context_history)
                # +4 because the strings 'm :' and 'f :' are 3 chars
                turn_dict = {'role': role, 'content': context_history[pos+4 : next_pos] }
                all_turns.append(turn_dict)

            all_dialogs.append(all_turns)
        
        return all_dialogs


    def tokenize_text(
        self,
        dialogs, max_length
    ):
 
        max_gen_len = max_length
        prompt_tokens = []
        unsafe_requests = []
        for dialog in dialogs:
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
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens: List[int] = sum(
                [
                    #! eos token id should we also add \n text?
                    self.tokenizer.encode_plus(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        bos=True,
                        eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += self.tokenizer.encode_plus(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )
            prompt_tokens.append(dialog_tokens)
        return prompt_tokens
        # generation_tokens, generation_logprobs = self.generate(
        #     prompt_tokens=prompt_tokens,
        #     max_gen_len=max_gen_len,
        #     temperature=temperature,
        #     top_p=top_p,
        #     logprobs=logprobs,
        # )
        # if logprobs:
        #     return [
        #         {
        #             "generation": {
        #                 "role": "assistant",
        #                 "content": self.tokenizer.decode(t)
        #                 if not unsafe
        #                 else UNSAFE_ERROR,
        #             },
        #             "tokens": [self.tokenizer.decode(x) for x in t],
        #             "logprobs": logprobs_i,
        #         }
        #         for t, logprobs_i, unsafe in zip(
        #             generation_tokens, generation_logprobs, unsafe_requests
        #         )
        #     ]
        # return [
        #     {
        #         "generation": {
        #             "role": "assistant",
        #             "content": self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,
        #         }
        #     }
        #     for t, unsafe in zip(generation_tokens, unsafe_requests)
        # ]

# if __name__=='__main__':
#     [{'role':'user','content':'first text user1'}, {'role':'assistant','content':'first text user 2'}]
#     tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b')
#     llama_preprocesing = Llama_dataset(tokenizer)
#     llama_preprocesing.tokenize_text('')
