import numpy as np

import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from datasets import SleepEdfDataset, MotorImageryDataset

def get_data(dataset="sleep_edf", batch_size=64):
    if dataset == "sleep_edf":
        dataset = SleepEdfDataset('../SESM/datasets/eeg/sleep_edf/prepared', 'fpz_cz', [7, 15], [3,17,11])
    elif dataset == "motor_imagery":
        #dataset = MotorImageryDataset('../SESM/datasets/eeg/motor_imagery/prepared', [43,13,4,35,11],[14,3,25,52,30])
        #dataset = MotorImageryDataset('../SESM/datasets/eeg/motor_imagery/prepared', [51,30,5,49,12], [18,2,11,42,39])
        #dataset = MotorImageryDataset('../SESM/datasets/eeg/motor_imagery/prepared', [4,9,13,40,7] ,[52,29,22,35,17])
        #dataset = MotorImageryDataset('../SESM/datasets/eeg/motor_imagery/prepared', [47,51,35,37,39],[2,23,28,42,41])
        #dataset = MotorImageryDataset('../SESM/datasets/eeg/motor_imagery/prepared', [44,16,5,7,15],[51,30,40,52,20])
        #dataset = MotorImageryDataset('../SESM/datasets/eeg/motor_imagery/prepared', [51,42,6,12,5],[31,46,36,2,39])
        #dataset = MotorImageryDataset('../SESM/datasets/eeg/motor_imagery/prepared', [24,32,1,9,19],[39,5,27,25,15])
        #dataset = MotorImageryDataset('../SESM/datasets/eeg/motor_imagery/prepared', [49,34,19,28,44],[13,29,41,16,50])
        #dataset = MotorImageryDataset('../SESM/datasets/eeg/motor_imagery/prepared', [28,34,26,1,29],[35,11,51,21,6])
        dataset = MotorImageryDataset('../SESM/datasets/eeg/motor_imagery/prepared', [11,3,28,41,39],[25,42,6,1,51])
    
    train_ds = tf.data.Dataset.from_tensor_slices((dataset.X_train, dataset.y_train))
    train_ds = train_ds.shuffle(buffer_size=len(dataset.X_train)).batch(batch_size)

    val_ds = tf.data.Dataset.from_tensor_slices((dataset.X_val, dataset.y_val))
    val_ds = val_ds.batch(batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((dataset.X_test, dataset.y_test))
    test_ds = test_ds.batch(batch_size)

    class_weights = dataset.class_weights
    max_len = dataset.sequence_length
    
    return train_ds, val_ds, test_ds, class_weights, max_len