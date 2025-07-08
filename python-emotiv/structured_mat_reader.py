#!/usr/bin/env python3
"""
Script to read structured .mat files with complex data types.
This handles MATLAB struct arrays and other complex data structures.
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_structured_mat(file_path):
    """
    Load a structured .mat file and handle complex data types.
    
    Args:
        file_path (str): Path to the .mat file
        
    Returns:
        dict: Dictionary containing the data from the .mat file
    """
    try:
        print(f"Loading file: {file_path}")
        mat_data = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
        return mat_data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None


def explore_structured_data(mat_data, file_name):
    """
    Explore structured data in .mat files.
    
    Args:
        mat_data (dict): Data loaded from .mat file
        file_name (str): Name of the file being analyzed
    """
    print(f"\n{'='*60}")
    print(f"Contents of {file_name}")
    print(f"{'='*60}")
    
    for key in mat_data.keys():
        if not key.startswith('__'):
            value = mat_data[key]
            print(f"\nVariable: {key}")
            print(f"Type: {type(value).__name__}")
            
            if isinstance(value, np.ndarray):
                print(f"Shape: {value.shape}")
                print(f"Data type: {value.dtype}")
                
                # Handle structured arrays
                if value.dtype.names is not None:
                    print("This is a structured array with fields:")
                    for field_name in value.dtype.names:
                        field_data = value[field_name]
                        print(f"  {field_name}: {type(field_data).__name__}")
                        if hasattr(field_data, 'shape'):
                            print(f"    Shape: {field_data.shape}")
                        if hasattr(field_data, 'dtype'):
                            print(f"    Data type: {field_data.dtype}")
                        
                        # Show sample data for simple types
                        if isinstance(field_data, (np.ndarray, list)) and len(field_data) > 0:
                            if isinstance(field_data[0], (int, float, np.number)):
                                print(f"    Sample values: {field_data[:min(5, len(field_data))]}")
                            elif isinstance(field_data[0], str):
                                print(f"    Sample values: {field_data[:min(3, len(field_data))]}")
                
                # Handle regular arrays
                elif value.size > 0:
                    print(f"Min value: {np.min(value)}")
                    print(f"Max value: {np.max(value)}")
                    print(f"Mean value: {np.mean(value):.4f}")
                    
                    if value.ndim == 1:
                        print(f"First 5 values: {value[:min(5, len(value))]}")
                    elif value.ndim == 2:
                        print(f"First row, first 5 values: {value[0, :min(5, value.shape[1])]}")
            
            elif hasattr(value, '__dict__'):
                # Handle MATLAB struct objects
                print("MATLAB struct object with fields:")
                for attr_name in dir(value):
                    if not attr_name.startswith('_'):
                        attr_value = getattr(value, attr_name)
                        print(f"  {attr_name}: {type(attr_value).__name__}")
                        if hasattr(attr_value, 'shape'):
                            print(f"    Shape: {attr_value.shape}")
                        if hasattr(attr_value, 'dtype'):
                            print(f"    Data type: {attr_value.dtype}")
            else:
                print(f"Value: {value}")
    
    print()


def extract_eeg_data(mat_data):
    """
    Extract EEG data from structured .mat file.
    
    Args:
        mat_data (dict): Data loaded from .mat file
        
    Returns:
        dict: Extracted EEG data components
    """
    eeg_data = {}
    
    for key in mat_data.keys():
        if not key.startswith('__'):
            value = mat_data[key]
            
            # Handle structured arrays (common in EEG data)
            if isinstance(value, np.ndarray) and value.dtype.names is not None:
                print(f"Found structured data in '{key}'")
                
                for field_name in value.dtype.names:
                    field_data = value[field_name]
                    eeg_data[f"{key}_{field_name}"] = field_data
                    
                    # Look for trial data (common in EEG)
                    if field_name == 'trial' and hasattr(field_data, 'shape'):
                        print(f"  Trial data shape: {field_data.shape}")
                        if len(field_data) > 0:
                            trial = field_data[0]
                            if hasattr(trial, 'shape'):
                                print(f"  First trial shape: {trial.shape}")
                    
                    # Look for time data
                    elif field_name == 'time' and hasattr(field_data, 'shape'):
                        print(f"  Time data shape: {field_data.shape}")
                    
                    # Look for sample rate
                    elif field_name == 'fsample':
                        print(f"  Sample rate: {field_data}")
                    
                    # Look for channel labels
                    elif field_name == 'label' and hasattr(field_data, 'shape'):
                        print(f"  Channel labels: {field_data}")
    
    return eeg_data


def plot_eeg_trials(eeg_data, file_name):
    """
    Plot EEG trial data if available.
    
    Args:
        eeg_data (dict): Extracted EEG data
        file_name (str): Name of the file being analyzed
    """
    # Look for trial data
    trial_keys = [key for key in eeg_data.keys() if 'trial' in key.lower()]
    
    for trial_key in trial_keys:
        trial_data = eeg_data[trial_key]
        
        if hasattr(trial_data, 'shape') and len(trial_data) > 0:
            print(f"Plotting trial data from {trial_key}")
            
            # Handle different trial data formats
            if isinstance(trial_data, np.ndarray):
                if trial_data.ndim == 3:  # [trials, channels, time]
                    num_trials = min(3, trial_data.shape[0])  # Plot first 3 trials
                    num_channels = min(8, trial_data.shape[1])  # Plot first 8 channels
                    
                    plt.figure(figsize=(15, 10))
                    plt.suptitle(f'EEG Trials from {file_name}', fontsize=16)
                    
                    for trial_idx in range(num_trials):
                        for ch_idx in range(num_channels):
                            plt.subplot(num_trials, num_channels, trial_idx * num_channels + ch_idx + 1)
                            plt.plot(trial_data[trial_idx, ch_idx, :])
                            plt.title(f'Trial {trial_idx+1}, Ch {ch_idx+1}')
                            plt.ylabel('Amplitude')
                            if trial_idx == num_trials - 1:
                                plt.xlabel('Time (samples)')
                    
                    plt.tight_layout()
                    plt.show()
                    return
                
                elif trial_data.ndim == 2:  # [channels, time]
                    num_channels = min(8, trial_data.shape[0])
                    
                    plt.figure(figsize=(12, 8))
                    plt.suptitle(f'EEG Data from {file_name}', fontsize=16)
                    
                    for ch_idx in range(num_channels):
                        plt.subplot(num_channels, 1, ch_idx + 1)
                        plt.plot(trial_data[ch_idx, :])
                        plt.title(f'Channel {ch_idx+1}')
                        plt.ylabel('Amplitude')
                        if ch_idx == num_channels - 1:
                            plt.xlabel('Time (samples)')
                    
                    plt.tight_layout()
                    plt.show()
                    return
    
    print("No plottable trial data found.")


def main():
    """Main function to process structured .mat files."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Find all .mat files in the directory
    mat_files = list(script_dir.glob("*.mat"))
    
    if not mat_files:
        print("No .mat files found in the current directory.")
        return
    
    print(f"Found {len(mat_files)} .mat file(s):")
    for i, file_path in enumerate(mat_files, 1):
        print(f"  {i}. {file_path.name}")
    
    print("\nProcessing each file...")
    
    for file_path in mat_files:
        # Load the .mat file with structured data handling
        mat_data = load_structured_mat(file_path)
        
        if mat_data is not None:
            # Explore the structured contents
            explore_structured_data(mat_data, file_path.name)
            
            # Extract EEG data
            eeg_data = extract_eeg_data(mat_data)
            
            # Try to plot EEG data
            plot_eeg_trials(eeg_data, file_path.name)
        else:
            print(f"Failed to load {file_path.name}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main() 