#!/usr/bin/env python3
"""
Visualize EEG data from .mat files
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def visualize_eeg(file_path):
    """
    Load and visualize EEG data from a .mat file
    """
    print(f"Loading and visualizing: {file_path}")
    
    # Load the data
    mat_data = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    
    # Extract EEG data
    eeg_signals = mat_data['data'].trial
    time_vector = mat_data['data'].time
    channel_labels = mat_data['data'].label
    sampling_rate = mat_data['data'].fsample
    
    print(f"EEG Data Shape: {eeg_signals.shape}")
    print(f"Time duration: {time_vector[-1] - time_vector[0]:.2f} seconds")
    print(f"Channels: {channel_labels}")
    
    # Create visualization
    num_channels = eeg_signals.shape[0]
    
    # Plot 1: Time series for all channels
    plt.figure(figsize=(15, 10))
    plt.suptitle(f'EEG Signals from {file_path}', fontsize=16)
    
    for i in range(num_channels):
        plt.subplot(num_channels, 1, i+1)
        plt.plot(time_vector, eeg_signals[i, :])
        plt.title(f'Channel: {channel_labels[i]}')
        plt.ylabel('Amplitude (μV)')
        plt.grid(True, alpha=0.3)
        
        if i == num_channels - 1:
            plt.xlabel('Time (seconds)')
    
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Power spectrum
    plt.figure(figsize=(12, 8))
    plt.suptitle(f'Power Spectrum from {file_path}', fontsize=16)
    
    for i in range(num_channels):
        plt.subplot(num_channels, 1, i+1)
        
        # Compute FFT
        signal = eeg_signals[i, :]
        fft_vals = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(len(signal), 1/sampling_rate)
        
        # Positive frequencies only
        positive_freq_mask = fft_freq > 0
        power_spectrum = np.abs(fft_vals[positive_freq_mask])**2
        frequencies = fft_freq[positive_freq_mask]
        
        plt.semilogy(frequencies, power_spectrum)
        plt.title(f'Channel: {channel_labels[i]}')
        plt.ylabel('Power')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 50)  # Focus on 0-50 Hz
        
        if i == num_channels - 1:
            plt.xlabel('Frequency (Hz)')
    
    plt.tight_layout()
    plt.show()
    
    return eeg_signals, time_vector, channel_labels, sampling_rate

if __name__ == "__main__":
    # Visualize the first file
    file_path = "Emotiv_Systems_Inc.-EPOC_BCI-07-15-2025_12-29-44.mat"
    eeg_data, time_data, labels, fs = visualize_eeg(file_path)
    
    print(f"\nData loaded successfully!")
    print(f"You can now work with:")
    print(f"- eeg_data: EEG signals (shape: {eeg_data.shape})")
    print(f"- time_data: Time vector (length: {len(time_data)})")
    print(f"- labels: Channel labels: {labels}")
    print(f"- fs: Sampling rate: {fs} Hz") 