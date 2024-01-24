import os
import re
from pathlib import Path

import numpy as np
import torch

from datasets import BaseDataset
from datasets.utils import get_balance_class_oversample

class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM"
}

def print_n_samples_each_class(labels):
    import numpy as np
    unique_labels = np.unique(labels)
    for c in unique_labels:
        n_samples = len(np.where(labels == c)[0])
        print("{}: {}".format(class_dict[c], n_samples))

class SleepEdfDataset(BaseDataset):
    """
    Parameters
    ----------
    data_dir : str
        Path to directory containing '*.npz' files
    channel_name : str
        Name of channel to extract.
    val_subject_ids : list(int)
        Indices of subjects used for validation set
    test_subject_ids : list(int), optional
        Indices of subjects used for hold-out test set. If not set then no hold-out (e.g. self.X_test, self.y_test) set will be created
    load_data : bool, optional
        Whether to load data on __init__, or delay until `load_data` call.
    """

    def __init__(self, data_dir, channel_name, val_subject_ids, test_subject_ids=None, load_data=True):
        self.data_dir = Path(os.path.join(data_dir, channel_name))
        assert self.data_dir.exists() and self.data_dir.is_dir(), "Directory must exist and be a folder"

        self.sequence_length = 3000
        self.input_shape = (self.sequence_length, 1)

        self.num_classes = 6
        self.output_shape = (self.sequence_length,)

        self.total_number_folds = 20
        self.val_subject_ids = val_subject_ids
        if test_subject_ids:
            self.test_subject_ids = test_subject_ids
        else:
            self.test_subject_ids = []

        if load_data:
            self.load_data()
        
    def _load_npz_file(self, npz_file):
        """Load data and labels from a npz file."""
        with np.load(npz_file) as f:
            data = f["x"]
            labels = f["y"]
            sampling_rate = f["fs"]
        return data, labels, sampling_rate

    def _load_npz_list_files(self, npz_files):
        """Load data and labels from list of npz files."""
        data = []
        labels = []
        fs = None
        for npz_f in npz_files:
            print("Loading {} ...".format(npz_f))
            tmp_data, tmp_labels, sampling_rate = self._load_npz_file(npz_f)
            if fs is None:
                fs = sampling_rate
            elif fs != sampling_rate:
                raise Exception("Found mismatch in sampling rate.")
            data.append(tmp_data)
            labels.append(tmp_labels)
        data = np.vstack(data)
        labels = np.hstack(labels)
        return data, labels

    def load_data(self, n_files=None):
        """
        Define X/y train/test.
        """
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        npzfiles = []
        for idx, f in enumerate(allfiles):
            if ".npz" in f:
                npzfiles.append(os.path.join(self.data_dir, f))
        npzfiles.sort()

        if n_files is not None:
            npzfiles = npzfiles[:n_files]

        validation_files = []
        test_files = []

        validation_ids_pattern = '|'.join(map(lambda n: str(n).zfill(2), self.val_subject_ids))
        test_ids_pattern = '|'.join(map(lambda n: str(n).zfill(2), self.test_subject_ids))
        
        validation_pattern = re.compile(f".*SC4({validation_ids_pattern})[1-9]E0.npz")
        test_pattern = re.compile(f".*SC4({test_ids_pattern})[1-9]E0.npz")
        for f in npzfiles:
            if validation_pattern.match(f):
                validation_files.append(f)
            elif test_pattern.match(f):
                test_files.append(f)

        train_files = list(set(npzfiles) - set(validation_files) - set(test_files))
        train_files.sort()
        validation_files.sort()
        test_files.sort()

        # Load training and validation sets
        print("Load training set:")
        data_train, label_train = self._load_npz_list_files(npz_files=train_files)
        print(" ")
        print("Load validation set:")
        data_val, label_val = self._load_npz_list_files(npz_files=validation_files)
        print(" ")
        if len(test_files)>0:
            print("Load test set:")
            data_test, label_test = self._load_npz_list_files(npz_files=test_files)
            print(" ")
        else:
            data_test = None
            label_test = None

        # Casting
        data_train = data_train.astype(np.float32)
        label_train = label_train.astype(np.int32)
        data_val = data_val.astype(np.float32)
        label_val = label_val.astype(np.int32)
        if (data_test is not None) and (label_test is not None):
            data_test = data_test.astype(np.float32)
            label_test = label_test.astype(np.int32)

        print("Training set: {}, {}".format(data_train.shape, label_train.shape))
        print_n_samples_each_class(label_train)
        print(" ")
        print("Validation set: {}, {}".format(data_val.shape, label_val.shape))
        print_n_samples_each_class(label_val)
        print(" ")
        if (data_test is not None) and (label_test is not None):
            print("Test set: {}, {}".format(data_test.shape, label_test.shape))
            print_n_samples_each_class(label_test)
            print(" ")

        # Use balanced-class, oversample training set
        x_train, y_train = get_balance_class_oversample(
            x=data_train, y=label_train
        )
        print("Oversampled training set: {}, {}".format(
            x_train.shape, y_train.shape
        ))
        print_n_samples_each_class(y_train)
        print(" ")

        # Must call .permute(0, 2, 1) to swap the channels and sequence positions
        self.X_train = torch.Tensor(x_train).permute(0, 2, 1)
        self.y_train = torch.Tensor(y_train)
        self.X_val = torch.Tensor(data_val).permute(0, 2, 1)
        self.y_val = torch.Tensor(label_val)
        if (data_test is not None) and (label_test is not None):
            self.X_test = torch.Tensor(data_test).permute(0, 2, 1)
            self.y_test = torch.Tensor(label_test)

        self.class_weights = (
            1 - (np.bincount(self.y_train) / self.y_train.shape[0])
        ).tolist()
    
    @property
    def data_dirname(self):
        return self.data_dir