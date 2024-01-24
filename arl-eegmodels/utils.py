import numpy as np

import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from datasets import SleepEdfDataset

def get_data(name="sleep_edf", batch_size=64):
    dataset = SleepEdfDataset('../SESM/datasets/eeg/sleep_edf/prepared', 'fpz_cz', [0, 1], [2])
    
    train_ds = tf.data.Dataset.from_tensor_slices((dataset.X_train, dataset.y_train))
    train_ds = train_ds.shuffle(buffer_size=len(dataset.X_train)).batch(batch_size)

    val_ds = tf.data.Dataset.from_tensor_slices((dataset.X_val, dataset.y_val))
    val_ds = val_ds.batch(batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((dataset.X_test, dataset.y_test))
    test_ds = test_ds.batch(batch_size)

    class_weights = dataset.class_weights
    max_len = dataset.sequence_length
    
    return train_ds, val_ds, test_ds, class_weights, max_len