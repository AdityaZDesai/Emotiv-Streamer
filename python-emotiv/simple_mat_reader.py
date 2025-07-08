#!/usr/bin/env python3
"""
Simple script to read a .mat file using scipy.io
"""

import scipy.io
import numpy as np

def read_mat_file(file_path):
    """
    Read a .mat file and display its contents.
    
    Args:
        file_path (str): Path to the .mat file
    """
    try:
        # Load the .mat file
        print(f"Loading file: {file_path}")
        mat = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
        
        print(f"\nSuccessfully loaded {file_path}")
        print("="*50)
        
        # Display all variables in the file
        print("Variables found in the .mat file:")
        for key in mat.keys():
            # Skip MATLAB's internal variables (start with __)
            if not key.startswith('__'):
                value = mat[key]
                print(f"\nVariable: {key}")
                print(f"Type: {type(value).__name__}")
                
                if isinstance(value, np.ndarray):
                    print(f"Shape: {value.shape}")
                    print(f"Data type: {value.dtype}")
                    
                    if value.size > 0:
                        print(f"Min value: {np.min(value)}")
                        print(f"Max value: {np.max(value)}")
                        print(f"Mean value: {np.mean(value):.4f}")
                        
                        # Show sample data
                        if value.ndim == 1:
                            print(f"First 5 values: {value[:5]}")
                        elif value.ndim == 2:
                            print(f"First row, first 5 values: {value[0, :5]}")
                else:
                    print(f"Value: {value}")
        
        # Access the EEG data
        eeg_signals = mat['data'].trial  # Shape: (13, 640) or (13, 2560)
        time_vector = mat['data'].time   # Time points
        channel_labels = mat['data'].label  # Channel names
        sampling_rate = mat['data'].fsample  # 128 Hz
        
        return mat
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

if __name__ == "__main__":
    # Example usage - you can change this to your specific file
    file_path = "emotiv-07-08-2025_18-31-33.mat"  # Change this to your file name
    
    # Read the .mat file
    mat_data = read_mat_file(file_path)
    
    if mat_data is not None:
        print(f"\nFile loaded successfully!")
        print("You can now access the data using mat_data['variable_name']")
    else:
        print("Failed to load the file.") 