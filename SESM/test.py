from datasets import SleepEdfDataset

test_data = SleepEdfDataset('datasets/eeg/sleep_edf/prepared', 'fpz_cz', [0,1], [2])