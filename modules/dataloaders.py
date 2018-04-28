from torch.utils.data import Dataset


class SentenceDataset(Dataset):
    def __init__(self):
        """
        A PyTorch Dataset
        What we have to do is to implement the 2 abstract methods:

            - __len__(self): in order to let the DataLoader know the size
                of our dataset and to perform batching, shuffling and so on...
            - __getitem__(self, index): we have to return the properly
                processed data-item from our dataset with a given index
        """
        pass

    def __len__(self):
        """
        Must return the length of the dataset, so the dataloader can know
        how to split it into batches

        Returns:
            (int): the length of the dataset
        """
        pass

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int):

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training example
                * label (int): the class label
                * length (int): the length (tokens) of the sentence

        Examples:
            For an `index` where:
            ::
                self.data[index] = ['this', 'is', 'really', 'simple']
                self.target[index] = "neutral"

            the function will have to return return:
            ::
                example = [  533  3908  1387   649   0     0     0     0
                             0     0     0     0     0     0     0     0
                             0     0     0     0     0     0     0     0]
                label = 1
        """
        pass
