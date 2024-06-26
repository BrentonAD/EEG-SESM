"""
Representation of the MIT-BIH ECG Arrhythmia Dataset.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from datasets import BaseDataset


class ArrhythmiaDataset(BaseDataset):
    """
    Parameters
    ----------
    data_dir : str
        Path to directory containing 'mitbih_{train/test}.csv' files
    load_data : bool, optional
        Whether to load data on __init__, or delay until `load_data` call.
    normalize : bool, optional
        Whether to shift the data from range [0, 1] -> range[-1, 1]
    """

    def __init__(self, data_dir, load_data=True, normalize=True):
        self.data_dir = Path(data_dir)
        self.train_path = self.data_dir / "mitbih_train.csv"
        assert self.train_path.exists() and self.train_path.is_file(), "File must exist"

        self.val_path = self.data_dir / "mitbih_val.csv"
        assert self.val_path.exists() and self.val_path.is_file(), "File must exist"

        self.sequence_length = 187
        self.input_shape = (self.sequence_length, 1)

        self.num_classes = 5
        self.output_shape = (self.num_classes,)

        self.normalize = normalize
        if load_data:
            self.load_data(normalize=self.normalize)

    def data_dirname(self):
        return self.data_dir

    def load_data(self, normalize=None):
        """
        Define X/y train/val.
        """
        normalize = normalize if normalize is not None else self.normalize

        train_data = np.loadtxt(self.train_path, delimiter=",")
        val_data = np.loadtxt(self.val_path, delimiter=",")
        
        # Must call .permute(0, 2, 1) to swap the channels and sequence positions
        self.X_train = torch.Tensor(train_data[:, :-1, np.newaxis]).permute(0, 2, 1)
        if normalize:
            self.X_train = self.X_train * 2.0 - 1.0
        self.y_train = torch.Tensor(train_data[:, -1])

        self.X_val = torch.Tensor(val_data[:, :-1, np.newaxis]).permute(0, 2, 1)
        if normalize:
            self.X_val = self.X_val * 2.0 - 1.0
        self.y_val = torch.Tensor(val_data[:, -1])

        self.class_weights = (
            1 - (np.bincount(self.y_train) / self.y_train.shape[0])
        ).tolist()

    def __repr__(self):
        return (
            "MIT-BIH Arrhythmia Dataset\n"
            f"Num classes: {self.num_classes}\n"
            f"Input shape: {self.input_shape}\n"
            f"Train, Test counts: {self.X_train.shape[0]}, {self.X_test.shape[0]}\n"
            f"Class weights: {self.class_weights}\n"
        )
