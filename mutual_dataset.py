import torch.utils.data as data

class MutualDataset(data.Dataset):
    """
    Custom PyTorch Dataset for tokenizing data for a binary classification task using a RoBERTa-based model.
    This dataset takes a list of samples, tokenizes them, and prepares them for model input.

    Args:
        id2history (dict): A dictionary with sentence_id as key and with context history string as value
        id2options (dict): A dictionary with sentence_id as key and a list of the string options as value
        id2label_id (dict): A dictionary with sentence_id as key and a list of the label label id as value
        tokenizer (transformers.AutoTokenizer): The tokenizer for tokenizing the input text.
        max_seq_length (int): The maximum sequence length for tokenized inputs.

    Attributes:
        input_ids (list): A list of lists of token IDs representing the tokenized inputs.
        attention_mask (list): A list of lists of attention masks for tokenized inputs.
        labels (list): A list of binary flags (0 or 1) indicating whether each option is correct.
        sentence_ids (list): A list of sentence IDs (numbers) corresponding to each tokenized input.
        option_ids (list): A list of option IDs (numbers) corresponding to each tokenized input.

    """
    def __init__(self, id2history, id2options, id2label_id, tokenizer, max_seq_length, repeat=False):
        super().__init__()
        sentences_id, input_ids, attention_mask, labels, option_ids = self.tokenize_roberta_data(id2history, id2options, id2label_id, tokenizer, max_seq_length)


        
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.sentence_ids = sentences_id
        self.option_ids = option_ids

    def __len__(self):
        """
        Returns the number of data points in the dataset.

        Returns:
            int: The number of data points.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns the idx-th data point of the dataset.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
                tuple: A tuple containing the following lists:
                    a. "input_ids" (list): List of token IDs for the input.
                    b. "labels" (int): Binary flag (0 or 1) indicating correctness of the option. 
                    c. "attention_mask" (list): List of attention masks for input tokens. 
                    d. "sentence_id" (int): Sentence ID corresponding to the data point. 
                    e. "option_id" (int): Option ID corresponding to the option which is concatenated to the input. 

        """ 

        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        label = self.labels[idx]
        sentence_id = self.sentence_ids[idx]
        option_id = self.option_ids[idx]
        # return input_ids, attention_mask, label
        return {"input_ids": input_ids, "labels": label,'attention_mask':attention_mask,'sentence_id':
                sentence_id, "option_id": option_id}

    def tokenize_roberta_data(self, id2history, id2options, id2label_id, tokenizer, max_seq_length):
        """
    Tokenizes input data for a binary classification task using a RoBERTa-based model.
    The input to the RoBERTa model consists of the concatenation of context history and a possible option.

    Args:
        id2history (dict): A dictionary mapping sentence IDs to context history strings.
        id2options (dict): A dictionary mapping sentence IDs to lists of answer options (strings).
        id2label_id (dict): A dictionary mapping sentence IDs to the index of the correct option.
        tokenizer (transformers.AutoTokenizer): The tokenizer for tokenizing the input text.
        max_seq_length (int): The maximum sequence length for tokenized inputs.

    Returns:
        tuple: A tuple containing the following lists:
            a. sentences_id (list): A list of sentence IDs (numbers) corresponding to each tokenized input.
            b. tokenized_input_ids (list): A list of lists of token IDs representing the tokenized inputs.
            c. tokenized_attention_mask (list): A list of lists of attention masks for tokenized inputs.
            d. option_flags (list): A list of binary flags (0 or 1) indicating whether each option is correct.
            e. options_id (list): A list of option IDs (numbers) corresponding to each tokenized input.
        """

        tokenized_input_ids = []
        tokenized_attention_mask = []
        option_flags = [] # 0 or 1 depending on whether it is the correct choice or not
        sentences_id = []
        options_id = []
        # for label_id, options, context_history in data:
        for sent_id in id2history:
            context_history = id2history[sent_id]
            options = id2options[sent_id]
            true_labels = id2label_id[sent_id] # list because we will maybe have more than one true option
            for option_id, option in enumerate(options):
                '''done similarly here https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_xnli.py#L337'''
                # sep_token_id added between the 2 sentences
                #tokenizer_dict = tokenizer(context_history, option, truncation=True, max_length = max_seq_length)
                tokenizer_dict = tokenizer.encode_plus(context_history, option, truncation=True, max_length = max_seq_length)
                #!todo check whether bert considers 0 or 1 as the correct choice
                option_flag = 1 if option_id in true_labels else 0 # check whether the option is correct

                tokenized_input_ids.append(tokenizer_dict['input_ids'])
                tokenized_attention_mask.append(tokenizer_dict['attention_mask'])
                option_flags.append(option_flag)
                sentences_id.append(sent_id)
                options_id.append(option_id)

        return sentences_id, tokenized_input_ids, tokenized_attention_mask, option_flags, options_id