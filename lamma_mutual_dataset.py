import torch.utils.data as data

class MutualDataset(data.Dataset):

    def __init__(self, split_samples, tokenizer, max_seq_length):
        super().__init__()

        input_ids, attention_mask, labels, sentence_ids, option_ids = self.tokenize_roberta_data(split_samples, tokenizer, max_seq_length)
    
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.sentence_ids = sentence_ids
        self.option_ids = option_ids

    def __len__(self):
        # Number of data point we have
        return len(self.labels)

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        label = self.labels[idx]
        sentence_id = self.sentence_ids[idx]
        option_id = self.option_ids[idx]
        # return input_ids, attention_mask, label
        return {"input_ids": input_ids, "labels": label,'attention_mask':attention_mask,'sentence_id':
                sentence_id, "option_id": option_id}

    def tokenize_roberta_data(self, data, tokenizer, max_seq_length):
        """
        We feed into BERT and do binary classification:
            history, option_1 -> BERT -> (p, 1-p) loss
            history, option_2 -> BERT -> (p, 1-p) loss
            etc. seperately 

        Alternatively, we could 
            history, option_1 -> BERT -> (p_1, 1-p) 
            history, option_2 -> BERT -> (p_2, 1-p) 
            etc. for options 3, 4 

            one loss function (p_1, p_2, p_3, p_4) where first element always the true label

        """

        tokenized_input_ids = []
        tokenized_attention_mask = []
        option_flags = [] # 0 or 1 depending on whether it is the correct choice or not
        sentences_id = []
        options_id = []
        for sentence_id, (label_id, options, context_history) in enumerate(data):
            option = options[label_id]
            '''done similarly here https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_xnli.py#L337'''
            # sep_token_id added between the 2 sentences
            tokenizer_dict = tokenizer(context_history, option, truncation=True, max_length = max_seq_length)
        
            tokenized_input_ids.append(tokenizer_dict['input_ids'])
            tokenized_attention_mask.append(tokenizer_dict['attention_mask'])
            sentences_id.append(sentence_id)

        return tokenized_input_ids, tokenized_attention_mask, sentences_id