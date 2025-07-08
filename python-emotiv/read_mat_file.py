#!/usr/bin/env python3
"""
Script to read and analyze .mat files from Emotiv EEG data.
This script can read MATLAB .mat files and display their contents.
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path


def load_mat_file(file_path):
    """
    Load a .mat file and return its contents.
    
    Args:
        file_path (str): Path to the .mat file
        
    Returns:
        dict: Dictionary containing the data from the .mat file
    """
    try:
        print(f"Loading file: {file_path}")
        mat_data = scipy.io.loadmat(file_path)
        return mat_data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None


def explore_mat_contents(mat_data, file_name):
    """
    Explore and display the contents of a .mat file.
    
    Args:
        mat_data (dict): Data loaded from .mat file
        file_name (str): Name of the file being analyzed
    """
    print(f"\n{'='*60}")
    print(f"Contents of {file_name}")
    print(f"{'='*60}")
    
    # List all variables in the .mat file
    print("Variables in the .mat file:")
    for key in mat_data.keys():
        # Skip MATLAB's internal variables (start with __)
        if not key.startswith('__'):
            value = mat_data[key]
            if isinstance(value, np.ndarray):
                print(f"  {key}: {type(value).__name__} with shape {value.shape}")
                if value.size > 0:
                    print(f"    Data type: {value.dtype}")
                    if value.ndim == 1:
                        print(f"    First few values: {value[:min(5, len(value))]}")
                    elif value.ndim == 2:
                        print(f"    Shape: {value.shape[0]} rows x {value.shape[1]} columns")
                        print(f"    Sample values: {value[0, :min(3, value.shape[1])]}")
            else:
                print(f"  {key}: {type(value).__name__} = {value}")
    
    print()


def plot_eeg_data(mat_data, file_name):
    """
    Plot EEG data if it exists in the .mat file.
    
    Args:
        mat_data (dict): Data loaded from .mat file
        file_name (str): Name of the file being analyzed
    """
    # Look for common EEG data variable names
    eeg_vars = ['eeg', 'data', 'EEG', 'Data', 'signal', 'Signal', 'channels', 'Channels']
    
    for var_name in eeg_vars:
        if var_name in mat_data:
            data = mat_data[var_name]
            if isinstance(data, np.ndarray) and data.ndim >= 2:
                print(f"Found EEG data in variable '{var_name}'")
                
                # Plot the data
                plt.figure(figsize=(12, 8))
                plt.suptitle(f'EEG Data from {file_name}', fontsize=16)
                
                if data.ndim == 2:
                    # If 2D, assume rows are channels and columns are time points
                    num_channels = min(data.shape[0], 8)  # Limit to 8 channels for readability
                    
                    for i in range(num_channels):
                        plt.subplot(num_channels, 1, i+1)
                        plt.plot(data[i, :])
                        plt.title(f'Channel {i+1}')
                        plt.ylabel('Amplitude')
                        if i == num_channels - 1:
                            plt.xlabel('Time (samples)')
                
                plt.tight_layout()
                plt.show()
                return
    
    print("No recognizable EEG data found for plotting.")


def save_data_info(mat_data, file_name):
    """
    Save information about the .mat file contents to a text file.
    
    Args:
        mat_data (dict): Data loaded from .mat file
        file_name (str): Name of the file being analyzed
    """
    output_file = f"{Path(file_name).stem}_info.txt"
    
    with open(output_file, 'w') as f:
        f.write(f"Analysis of {file_name}\n")
        f.write("="*50 + "\n\n")
        
        for key in mat_data.keys():
            if not key.startswith('__'):
                value = mat_data[key]
                f.write(f"Variable: {key}\n")
                f.write(f"Type: {type(value).__name__}\n")
                
                if isinstance(value, np.ndarray):
                    f.write(f"Shape: {value.shape}\n")
                    f.write(f"Data type: {value.dtype}\n")
                    if value.size > 0:
                        f.write(f"Min value: {np.min(value)}\n")
                        f.write(f"Max value: {np.max(value)}\n")
                        f.write(f"Mean value: {np.mean(value)}\n")
                        f.write(f"Standard deviation: {np.std(value)}\n")
                
                f.write("\n")
    
    print(f"Data information saved to {output_file}")


def main():
    """Main function to process .mat files."""
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
        # Load the .mat file
        mat_data = load_mat_file(file_path)
        
        if mat_data is not None:
            # Explore the contents
            explore_mat_contents(mat_data, file_path.name)
            
            # Try to plot EEG data
            plot_eeg_data(mat_data, file_path.name)
            
            # Save information to file
            save_data_info(mat_data, file_path.name)
        else:
            print(f"Failed to load {file_path.name}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main() 