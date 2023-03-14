import torch
import numpy as np

from torch.utils.data import Dataset


class ExploratoryDaggerDataset(Dataset):
    def __init__(self, X=None, y=None):
        self.X = X if X is not None else []
        self.y = y if y is not None else []

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # print(f'Len: {len(self.X)}. Idx: {idx}.')
        X, y = torch.tensor(self.X), torch.tensor(self.y)
        X, y = torch.squeeze(X), torch.squeeze(y)
        # print(f'X: {X.shape}. Y: {y.shape}.')
        
        return X[idx], y[idx]

    def aggregate(self, observations, actions):
        """

        Args:
            observations (list): list of len T where each element is a numpy array of shape (O,)
                                 where O is the observation space of the environment
            actions (list): list of len T where each element is a numpy array of shape (T,)
                            containing the actions taken for each corresponding observation.
        """
        print(f'Appending Observations: {len(observations)} of type {type(observations[0])}.')
        
        self.X = self.X + observations
        self.y = self.y + actions