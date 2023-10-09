import torch.utils.data as data
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
#! we care only about the correct option in both classes
class Seperate_Context_Option_Dataset(data.Dataset):

    def __init__(self, split_samples, tokenizer, max_seq_length):
        super().__init__()
        input_ids, attention_mask, sentence_ids, labels = self.tokenize_eval_llama_data(split_samples, tokenizer, max_seq_length)
        #! here we set as input the cpntext history only
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.sentence_ids = sentence_ids
        #! here we set as label the correct option
        self.labels = labels
        collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
        self.collate_fn = collate_fn


    def __len__(self):
        # Number of data point we have
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        sentence_id = self.sentence_ids[idx]
        label = self.labels[idx]
        # return input_ids, attention_mask, label
        return {"input_ids": input_ids, 'attention_mask':attention_mask,'sentence_id':
                sentence_id, "labels": label}
    
    def tokenize_eval_llama_data(self, data, tokenizer, max_seq_length):
        """
        tokenize h

        """
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({"pad_token":"<pad>"}) # from hf this line
        tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
        tokenized_input_ids = []
        tokenized_attention_mask = []
        sentences_id = []
        labels = []
        # max_input_ids = -1
        for sentence_id, (label_id, options, context_history) in enumerate(data):
            option = options[label_id]
            # sep_token_id added between the 2 sentences
            input = context_history 
            tokenizer_hist_dict = tokenizer(input, truncation=True, max_length = max_seq_length, add_special_tokens = False)

            tokenizer_label_dict = tokenizer(option, truncation=True, max_length = max_seq_length, add_special_tokens = False)
            
            # tmp_input_ids = 

            tokenized_input_ids.append(tokenizer_hist_dict['input_ids'])
            tokenized_attention_mask.append(tokenizer_hist_dict['attention_mask'])
            sentences_id.append(sentence_id)

            # we want to ignore padding in the loss function
            tokenizer_label_dict["input_ids"] = [(l if l != tokenizer.pad_token_id else -100) for l in tokenizer_label_dict["input_ids"]]
            

            labels.append(tokenizer_label_dict['input_ids'])
        #     if max_input_ids < len(tokenizer_label_dict['input_ids']):
        #         max_input_ids = len(tokenizer_label_dict['input_ids'])

        # print('max label ids ', max_input_ids)
        return tokenized_input_ids, tokenized_attention_mask, sentences_id, labels
    
    

class Concat_History_Option_Dataset(data.Dataset):

    def __init__(self, split_samples, tokenizer, max_seq_length):
        super().__init__()
        input_ids, attention_mask, sentence_ids = self.tokenize_training_llama_data(split_samples, tokenizer, max_seq_length)
        """
         here we set as input the history together with the correct option and the labels will be
         the same as the input using the appropriate collate_fn
        """
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.sentence_ids = sentence_ids

        # #! DataCollatorForLanguageModeling sets as labels the input and puts -100 in the padding token
        # collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)
        # self.collate_fn = collate_fn

    def __len__(self):
        # Number of data point we have
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        sentence_id = self.sentence_ids[idx]
        # return input_ids, attention_mask, label
        return {"input_ids": input_ids, 'attention_mask':attention_mask,'sentence_id':
                sentence_id}
    
    def tokenize_training_llama_data(self, data, tokenizer, max_seq_length):
        """
        tokenize h

        """
        tokenizer.pad_token = tokenizer.eos_token # see https://github.com/facebookresearch/llama-recipes/blob/main/examples/inference.py#L75
        tokenizer.add_special_tokens({"pad_token":"<pad>"}) # from hf this line
        tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

        tokenized_input_ids = []
        tokenized_attention_mask = []
        sentences_id = []
        for sentence_id, (label_id, options, context_history) in enumerate(data):
            option = options[label_id]
            
            # sep_token_id added between the 2 sentences
            input = context_history + " "+ option
            tokenizer_dict = tokenizer(input, truncation=True, max_length = max_seq_length)
            

            tokenized_input_ids.append(tokenizer_dict['input_ids'])
            tokenized_attention_mask.append(tokenizer_dict['attention_mask'])
            sentences_id.append(sentence_id)

        return tokenized_input_ids, tokenized_attention_mask, sentences_id