import os
import shutil
import argparse
import glob

import numpy as np
from scipy import stats
import scipy.io as sio

from mne import create_info
from mne.io import RawArray

PRE_EVENT_DURATION = 1
EVENT_DURATION = 3
POST_EVENT_DURATION = 1

CHANNELS = [
    "FP1",
    "AF7",
    "AF3",
    "F1",
    "F3",
    "F5",
    "F7",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "C1",
    "C3",
    "C5",
    "T7",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "P1",
    "P3",
    "P5",
    "P7",
    "P9",
    "PO7",
    "PO3",
    "O1",
    "IZ",
    "OZ",
    "POZ",
    "PZ",
    "CPZ",
    "FPZ",
    "FP2",
    "AF8",
    "AF4",
    "AFZ",
    "FZ",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT8",
    "FC6",
    "FC4",
    "FC2",
    "FCZ",
    "CZ",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP8",
    "CP6",
    "CP4",
    "CP2",
    "P2",
    "P4",
    "P6",
    "P8",
    "P10",
    "PO8",
    "PO4",
    "O2",
    "EMG1",
    "EMG2",
    "EMG3",
    "EMG4",
    "STI 014"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./input",
                        help="File path to the raw data files.")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory where to save numpy files outputs.")
    parser.add_argument("--select_channels", type=str, nargs='+', default="C3",
                        help="The selected channel(s)")
    args = parser.parse_args()


    data_dir = args.data_dir
    output_dir = args.output_dir
    select_channels = args.select_channels

    # Get channel index
    channel_indices = []
    for channel in select_channels:
        try:
            channel_index = CHANNELS.index(channel)
            channel_indices.append(channel_index)
        except ValueError:
            raise ValueError(f"{channel} not found in the list of available channels")

    # Output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    
    data_files = glob.glob(os.path.join(data_dir, "*.mat"))
    for subject_file in data_files:
        print(f"Processing {subject_file}")
        raw_mat = sio.loadmat(subject_file, simplify_cells=True)["eeg"]
        
        sample_rate = raw_mat["srate"]
        window_before_event_frames = int(PRE_EVENT_DURATION*sample_rate)
        window_after_event_frames = int(POST_EVENT_DURATION*sample_rate)
        event_window_frames = int(EVENT_DURATION*sample_rate)

        event_onsets = raw_mat["imagery_event"].nonzero()[0]

        # Initialize an array to store the samples
        left_samples = []
        right_samples = []

        # Iterate over each event index
        for idx in event_onsets:
            # Calculate the start and end indices for the sample
            start_idx = idx - window_before_event_frames
            end_idx = idx + event_window_frames + window_after_event_frames
            
            # Ensure the indices are within bounds
            start_idx = max(0, start_idx)
            end_idx = min(len(raw_mat["imagery_event"]), end_idx)
            
            # Extract the sample from the data arrays and convert from microV to V
            left_sample = raw_mat["imagery_left"][channel_indices, start_idx:end_idx]*0.0000001
            right_sample = raw_mat["imagery_right"][channel_indices, start_idx:end_idx]*0.0000001
            
            # Add the sample to the list of samples
            left_samples.append(left_sample)
            right_samples.append(right_sample)

        # Convert the list of samples to a numpy array
        imagery_left = np.array(left_samples)
        imagery_right = np.array(right_samples)

        x = np.concatenate((imagery_left, imagery_right), axis=0)
        y = np.concatenate((np.zeros(imagery_left.shape[0]), np.ones(imagery_right.shape[0])))

        x_standardized = stats.zscore(x, axis=0)

        # Save
        filename = os.path.basename(subject_file).replace(".mat", ".npz")
        save_dict = {
            "x": x_standardized, 
            "y": y, 
            "fs": sample_rate,
            "ch_labels": select_channels
        }
        np.savez(os.path.join(args.output_dir, filename), **save_dict)

        print("\n=======================================\n")

if __name__ == "__main__":
    main()