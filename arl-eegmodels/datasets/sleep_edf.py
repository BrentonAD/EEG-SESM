import os
import re
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from datasets.utils import get_balance_class_oversample

class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM"
}

def print_n_samples_each_class(labels):
    unique_labels = np.unique(labels)
    for c in unique_labels:
        n_samples = len(np.where(labels == c)[0])
        print("{}: {}".format(class_dict[c], n_samples))

class SleepEdfDataset:
    def __init__(self, data_dir, channel_name, val_subject_ids, test_subject_ids=None, load_data=True):
        self.data_dir = Path(os.path.join(data_dir, channel_name))
        assert self.data_dir.exists() and self.data_dir.is_dir(), "Directory must exist and be a folder"

        self.sequence_length = 3000
        self.input_shape = (self.sequence_length, 1)

        self.num_classes = 6
        self.output_shape = (self.sequence_length,)

        self.total_number_folds = 20
        self.val_subject_ids = val_subject_ids
        self.test_subject_ids = test_subject_ids or []

        if load_data:
            self.load_data()

    def _load_npz_file(self, npz_file):
        with np.load(npz_file) as f:
            data = f["x"]
            labels = f["y"]
            sampling_rate = f["fs"]
        return data, labels, sampling_rate

    def _load_npz_list_files(self, npz_files):
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
        allfiles = os.listdir(self.data_dir)
        npzfiles = [os.path.join(self.data_dir, f) for f in allfiles if ".npz" in f]
        npzfiles.sort()

        if n_files is not None:
            npzfiles = npzfiles[:n_files]

        validation_files = [f for f in npzfiles if re.match(f".*SC4({'|'.join(map(lambda n: str(n).zfill(2), self.val_subject_ids))})[1-9]E0.npz", f)]
        test_files = [f for f in npzfiles if re.match(f".*SC4({'|'.join(map(lambda n: str(n).zfill(2), self.test_subject_ids))})[1-9]E0.npz", f)]

        train_files = list(set(npzfiles) - set(validation_files) - set(test_files))
        train_files.sort()
        validation_files.sort()
        test_files.sort()

        print("Load training set:")
        data_train, label_train = self._load_npz_list_files(npz_files=train_files)
        print(" ")
        print("Load validation set:")
        data_val, label_val = self._load_npz_list_files(npz_files=validation_files)
        print(" ")
        if len(test_files) > 0:
            print("Load test set:")
            data_test, label_test = self._load_npz_list_files(npz_files=test_files)
            print(" ")
        else:
            data_test = None
            label_test = None

        data_train = data_train.astype(np.float32)
        label_train = label_train.astype(np.int32)
        data_val = data_val.astype(np.float32)
        label_val = label_val.astype(np.int32)
        if data_test is not None and label_test is not None:
            data_test = data_test.astype(np.float32)
            label_test = label_test.astype(np.int32)

        print("Training set: {}, {}".format(data_train.shape, label_train.shape))
        print_n_samples_each_class(label_train)
        print(" ")
        print("Validation set: {}, {}".format(data_val.shape, label_val.shape))
        print_n_samples_each_class(label_val)
        print(" ")
        if data_test is not None and label_test is not None:
            print("Test set: {}, {}".format(data_test.shape, label_test.shape))
            print_n_samples_each_class(label_test)
            print(" ")

        x_train, y_train = get_balance_class_oversample(data_train, label_train)
        print("Oversampled training set: {}, {}".format(x_train.shape, y_train.shape))
        print_n_samples_each_class(y_train)
        print(" ")

        self.X_train = self.transform_tensor(tf.convert_to_tensor(x_train, dtype=tf.float32))
        self.y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
        self.X_val = self.transform_tensor(tf.convert_to_tensor(data_val, dtype=tf.float32))
        self.y_val = tf.convert_to_tensor(label_val, dtype=tf.int32)
        if data_test is not None and label_test is not None:
            self.X_test = self.transform_tensor(tf.convert_to_tensor(data_test, dtype=tf.float32))
            self.y_test = tf.convert_to_tensor(label_test, dtype=tf.int32)

        class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train.numpy()), y=self.y_train.numpy())
        self.class_weights = {i: w for i, w in enumerate(class_weights)}

    def transform_tensor(self, X):
        X = tf.transpose(X, perm=[0,2,1])
        X = tf.expand_dims(X, -1)
        return X

    @property
    def data_dirname(self):
        return self.data_dir
