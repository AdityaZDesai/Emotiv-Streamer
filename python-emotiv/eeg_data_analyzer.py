#!/usr/bin/env python3
"""
EEG Data Analyzer for Emotiv .mat files
This script properly handles the structured EEG data format.
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_eeg_data(file_path):
    """
    Load EEG data from .mat file with proper handling of MATLAB structs.
    
    Args:
        file_path (str): Path to the .mat file
        
    Returns:
        dict: Dictionary containing the EEG data
    """
    try:
        print(f"Loading EEG data from: {file_path}")
        mat_data = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
        
        # Extract the data struct
        if 'data' in mat_data:
            data_struct = mat_data['data']
            
            # Extract individual components
            eeg_data = {
                'fsample': data_struct.fsample,  # Sampling frequency
                'label': data_struct.label,      # Channel labels
                'trial': data_struct.trial,      # EEG trial data
                'time': data_struct.time,        # Time vector
                'sampleinfo': data_struct.sampleinfo  # Sample information
            }
            
            print(f"Successfully loaded EEG data:")
            print(f"  Sampling frequency: {eeg_data['fsample']} Hz")
            print(f"  Number of channels: {len(eeg_data['label'])}")
            print(f"  Trial data shape: {eeg_data['trial'].shape}")
            print(f"  Time points: {len(eeg_data['time'])}")
            print(f"  Channel labels: {eeg_data['label']}")
            
            return eeg_data
        else:
            print("No 'data' variable found in the .mat file")
            return None
            
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None


def plot_eeg_channels(eeg_data, file_name):
    """
    Plot EEG data for all channels.
    
    Args:
        eeg_data (dict): EEG data dictionary
        file_name (str): Name of the file being analyzed
    """
    trial_data = eeg_data['trial']
    time_data = eeg_data['time']
    labels = eeg_data['label']
    
    num_channels = trial_data.shape[0]
    
    # Create subplots for each channel
    fig, axes = plt.subplots(num_channels, 1, figsize=(15, 2*num_channels))
    fig.suptitle(f'EEG Data from {file_name}', fontsize=16)
    
    # If only one channel, make axes iterable
    if num_channels == 1:
        axes = [axes]
    
    for i in range(num_channels):
        axes[i].plot(time_data, trial_data[i, :])
        axes[i].set_title(f'Channel: {labels[i]}')
        axes[i].set_ylabel('Amplitude (μV)')
        axes[i].grid(True, alpha=0.3)
        
        # Only add x-label to the bottom plot
        if i == num_channels - 1:
            axes[i].set_xlabel('Time (seconds)')
    
    plt.tight_layout()
    plt.show()


def plot_power_spectrum(eeg_data, file_name):
    """
    Plot power spectrum for each channel.
    
    Args:
        eeg_data (dict): EEG data dictionary
        file_name (str): Name of the file being analyzed
    """
    trial_data = eeg_data['trial']
    fsample = eeg_data['fsample']
    labels = eeg_data['label']
    
    num_channels = trial_data.shape[0]
    
    # Create subplots for power spectra
    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 2*num_channels))
    fig.suptitle(f'Power Spectrum from {file_name}', fontsize=16)
    
    # If only one channel, make axes iterable
    if num_channels == 1:
        axes = [axes]
    
    for i in range(num_channels):
        # Compute FFT
        signal = trial_data[i, :]
        fft_vals = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(len(signal), 1/fsample)
        
        # Compute power spectrum (positive frequencies only)
        positive_freq_mask = fft_freq > 0
        power_spectrum = np.abs(fft_vals[positive_freq_mask])**2
        frequencies = fft_freq[positive_freq_mask]
        
        # Plot power spectrum
        axes[i].semilogy(frequencies, power_spectrum)
        axes[i].set_title(f'Channel: {labels[i]}')
        axes[i].set_ylabel('Power')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(0, 50)  # Focus on 0-50 Hz range
        
        # Only add x-label to the bottom plot
        if i == num_channels - 1:
            axes[i].set_xlabel('Frequency (Hz)')
    
    plt.tight_layout()
    plt.show()


def analyze_eeg_statistics(eeg_data, file_name):
    """
    Analyze and display EEG statistics.
    
    Args:
        eeg_data (dict): EEG data dictionary
        file_name (str): Name of the file being analyzed
    """
    trial_data = eeg_data['trial']
    labels = eeg_data['label']
    
    print(f"\n{'='*60}")
    print(f"EEG Statistics for {file_name}")
    print(f"{'='*60}")
    
    print(f"Recording duration: {eeg_data['time'][-1] - eeg_data['time'][0]:.2f} seconds")
    print(f"Sampling frequency: {eeg_data['fsample']} Hz")
    print(f"Number of time points: {len(eeg_data['time'])}")
    print(f"Number of channels: {len(labels)}")
    
    print(f"\nChannel Statistics:")
    print(f"{'Channel':<15} {'Mean (μV)':<12} {'Std (μV)':<12} {'Min (μV)':<12} {'Max (μV)':<12}")
    print("-" * 70)
    
    for i, label in enumerate(labels):
        if i < trial_data.shape[0]:  # Handle dimension mismatch
            channel_data = trial_data[i, :]
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            min_val = np.min(channel_data)
            max_val = np.max(channel_data)
            
            print(f"{label:<15} {mean_val:<12.2f} {std_val:<12.2f} {min_val:<12.2f} {max_val:<12.2f}")
        else:
            print(f"{label:<15} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")


def save_eeg_summary(eeg_data, file_name):
    """
    Save EEG data summary to a text file.
    
    Args:
        eeg_data (dict): EEG data dictionary
        file_name (str): Name of the file being analyzed
    """
    output_file = f"{Path(file_name).stem}_eeg_summary.txt"
    
    with open(output_file, 'w') as f:
        f.write(f"EEG Data Summary for {file_name}\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Recording Information:\n")
        f.write(f"  Date: {file_name}\n")
        f.write(f"  Sampling frequency: {eeg_data['fsample']} Hz\n")
        f.write(f"  Duration: {eeg_data['time'][-1] - eeg_data['time'][0]:.2f} seconds\n")
        f.write(f"  Number of time points: {len(eeg_data['time'])}\n")
        f.write(f"  Number of channels: {len(eeg_data['label'])}\n\n")
        
        f.write(f"Channel Information:\n")
        for i, label in enumerate(eeg_data['label']):
            f.write(f"  Channel {i+1}: {label}\n")
        f.write("\n")
        
        f.write(f"Data Statistics:\n")
        trial_data = eeg_data['trial']
        for i, label in enumerate(eeg_data['label']):
            if i < trial_data.shape[0]:  # Handle dimension mismatch
                channel_data = trial_data[i, :]
                f.write(f"  {label}:\n")
                f.write(f"    Mean: {np.mean(channel_data):.4f} μV\n")
                f.write(f"    Std: {np.std(channel_data):.4f} μV\n")
                f.write(f"    Min: {np.min(channel_data):.4f} μV\n")
                f.write(f"    Max: {np.max(channel_data):.4f} μV\n")
                f.write(f"    Range: {np.max(channel_data) - np.min(channel_data):.4f} μV\n\n")
            else:
                f.write(f"  {label}:\n")
                f.write(f"    Data not available (dimension mismatch)\n\n")
    
    print(f"EEG summary saved to {output_file}")


def main():
    """Main function to analyze EEG data from .mat files."""
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
    
    print("\nAnalyzing each file...")
    
    for file_path in mat_files:
        print(f"\n{'='*80}")
        print(f"Processing: {file_path.name}")
        print(f"{'='*80}")
        
        # Load EEG data
        eeg_data = load_eeg_data(file_path)
        
        if eeg_data is not None:
            # Analyze statistics
            analyze_eeg_statistics(eeg_data, file_path.name)
            
            # Save summary
            save_eeg_summary(eeg_data, file_path.name)
            
            # Plot data
            print(f"\nGenerating plots for {file_path.name}...")
            plot_eeg_channels(eeg_data, file_path.name)
            plot_power_spectrum(eeg_data, file_path.name)
        else:
            print(f"Failed to load {file_path.name}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main() 