from transformers import DataCollatorForLanguageModeling
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections.abc import Mapping
import numpy as np
import torch
# from transformers.data_collator import _torch_collate_batch

@dataclass
class LLama_DataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        mlm (`bool`, *optional*, defaults to `True`):
            Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs
            with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
            tokens and the value to predict for the masked token.
        mlm_probability (`float`, *optional*, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".

    <Tip>

    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.

    </Tip>"""

    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"
    
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = {}
        if len(examples[0]) == 5: # useful only when using model.generate
            input_ids, attention_masks, labels, sentences_id, without_dummy = zip(*examples)
            sentences_id = torch.tensor(sentences_id)
            batch['sentences_id'] = sentences_id
            batch['without_dummy'] = torch.tensor(without_dummy)
        elif len(examples[0]) == 3:
            input_ids, attention_masks, labels = zip(*examples)


        # Pad input_ids
        max_input_length = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        for ids, masks in zip(input_ids, attention_masks):
            padded_ids = ids + [self.tokenizer.pad_token_id] * (max_input_length - len(ids))
            padded_input_ids.append(padded_ids)

            padded_mask = masks + [0] * (max_input_length - len(ids))
            padded_attention_masks.append(padded_mask)

        padded_input_ids = torch.tensor(padded_input_ids)

        # Pad labels
        max_label_length = max(len(lbls) for lbls in labels)
        padded_labels = []
        for lbls in labels:
            padded_lbls = lbls + [-100] * (max_label_length - len(lbls))
            padded_labels.append(padded_lbls)
        padded_labels = torch.tensor(padded_labels)

        # Convert attention_masks to tensor
        padded_attention_masks = torch.tensor(padded_attention_masks)
        batch['input_ids'] = padded_input_ids
        batch['attention_mask']=padded_attention_masks
        batch['labels']=padded_labels
        return batch