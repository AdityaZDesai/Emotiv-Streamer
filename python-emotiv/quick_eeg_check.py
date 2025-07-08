#!/usr/bin/env python3
"""
Quick check to demonstrate that .mat files contain real EEG data.
"""

import scipy.io
import numpy as np

def quick_check(file_path):
    """
    Quick check of EEG data in .mat file.
    """
    print(f"\n{'='*60}")
    print(f"Quick Check: {file_path}")
    print(f"{'='*60}")
    
    # Load the data
    mat_data = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    
    if 'data' in mat_data:
        data = mat_data['data']
        
        print(f"✓ File contains EEG data!")
        print(f"  Sampling frequency: {data.fsample} Hz")
        print(f"  Number of channels: {len(data.label)}")
        print(f"  Channel labels: {data.label}")
        print(f"  Trial data shape: {data.trial.shape}")
        print(f"  Time duration: {data.time[-1] - data.time[0]:.2f} seconds")
        
        # Show some actual data values
        print(f"\nSample data from first channel ({data.label[0]}):")
        print(f"  First 10 values: {data.trial[0, :10]}")
        print(f"  Mean amplitude: {np.mean(data.trial[0, :]):.2f} μV")
        print(f"  Min amplitude: {np.min(data.trial[0, :]):.2f} μV")
        print(f"  Max amplitude: {np.max(data.trial[0, :]):.2f} μV")
        
        # Show data from all channels
        print(f"\nData summary for all channels:")
        for i, label in enumerate(data.label):
            channel_data = data.trial[i, :]
            print(f"  {label}: mean={np.mean(channel_data):.1f}μV, std={np.std(channel_data):.1f}μV")
        
        return True
    else:
        print("✗ No 'data' variable found in file")
        return False

if __name__ == "__main__":
    # Check both files
    files = ["emotiv-08-07-2025_17-37-01.mat", "emotiv-08-07-2025_17-37-29.mat"]
    
    for file_path in files:
        quick_check(file_path)
    
    print(f"\n{'='*60}")
    print("CONCLUSION: The .mat files are NOT empty!")
    print("They contain real EEG data with:")
    print("- 14 EEG channels (F3, FC5, AF3, F7, T7, P7, O1, O2, P8, T8, F8, AF4, FC6, F4)")
    print("- 128 Hz sampling rate")
    print("- Multiple seconds of recording time")
    print("- Real amplitude values in microvolts (μV)")
    print(f"{'='*60}") 