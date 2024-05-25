import os
import re
from pathlib import Path

import numpy as np
import torch

from datasets import BaseDataset

class MotorImageryDataset(BaseDataset):
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

    def __init__(self, data_dir, val_subject_ids, test_subject_ids=None, load_data=True):
        self.data_dir = Path(data_dir)
        assert self.data_dir.exists() and self.data_dir.is_dir(), "Directory must exist and be a folder"

        self.sequence_length = 2560
        self.input_shape = (self.sequence_length, 1)

        self.num_classes = 2
        self.output_shape = (self.sequence_length,)

        self.total_number_folds = 52
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
        Define X/y train/test
        """
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        npzfiles = []
        for f in allfiles:
            if ".npz" in f:
                npzfiles.append(os.path.join(self.data_dir, f))
        npzfiles.sort()

        if n_files is not None:
            npzfiles = npzfiles[:n_files]

        validation_files = []
        test_files = []

        pattern = re.compile('.*s(\d+)\.npz')

        for f in npzfiles:
            # Extract subject IF from file name
            id_match = pattern.match(f)
            if id_match:
                subject_id = int(id_match.group(1))
                # Check if the subject ID is in validation_subject_ids
                if subject_id in self.val_subject_ids:
                    validation_files.append(f)
                # Check if the subject ID is in test_subject_ids
                elif subject_id in self.test_subject_ids:
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

        self.X_train = torch.Tensor(data_train)
        self.y_train = torch.Tensor(label_train)
        self.X_val = torch.Tensor(data_val)
        self.y_val = torch.Tensor(label_val)
        if (data_test is not None) and (label_test is not None):
            self.X_test = torch.Tensor(data_test)
            self.y_test = torch.Tensor(label_test)

        self.class_weights = (
            1 - (np.bincount(self.y_train) / self.y_train.shape[0])
        ).tolist()

    @property
    def data_dirname(self):
        return self.data_dir