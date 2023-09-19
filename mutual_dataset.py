import torch.utils.data as data

class MutualDataset(data.Dataset):

    def __init__(self, input_ids, attention_mask, labels):
        """
        Inputs:
            size - Number of data points we want to generate
            std - Standard deviation of the noise (see generate_continuous_xor function)
        """
        super().__init__()
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        # Number of data point we have
        return len(self.labels)

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        label = self.labels[idx]
        # return input_ids, attention_mask, label
        return {"input_ids": input_ids, "labels": label,'attention_mask':attention_mask}