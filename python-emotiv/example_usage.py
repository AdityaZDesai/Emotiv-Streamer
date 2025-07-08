#!/usr/bin/env python3
"""
Example usage of EEG data from .mat files
This shows common tasks you can do with the EEG data
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def load_eeg_data(file_path):
    """Load EEG data from .mat file"""
    mat_data = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    
    eeg_signals = mat_data['data'].trial
    time_vector = mat_data['data'].time
    channel_labels = mat_data['data'].label
    sampling_rate = mat_data['data'].fsample
    
    return eeg_signals, time_vector, channel_labels, sampling_rate

def example_1_basic_analysis():
    """Example 1: Basic data exploration"""
    print("=== Example 1: Basic Data Exploration ===")
    
    # Load data
    eeg_signals, time_vector, labels, fs = load_eeg_data("emotiv-08-07-2025_17-37-01.mat")
    
    print(f"Data shape: {eeg_signals.shape}")
    print(f"Time duration: {time_vector[-1] - time_vector[0]:.2f} seconds")
    print(f"Sampling rate: {fs} Hz")
    print(f"Channels: {labels}")
    
    # Show some statistics
    print(f"\nChannel Statistics:")
    for i, label in enumerate(labels):
        if i < eeg_signals.shape[0]:  # Handle dimension mismatch
            channel_data = eeg_signals[i, :]
            print(f"  {label}: mean={np.mean(channel_data):.1f}μV, std={np.std(channel_data):.1f}μV")

def example_2_signal_filtering():
    """Example 2: Filter EEG signals"""
    print("\n=== Example 2: Signal Filtering ===")
    
    # Load data
    eeg_signals, time_vector, labels, fs = load_eeg_data("emotiv-08-07-2025_17-37-01.mat")
    
    # Get first channel
    signal = eeg_signals[0, :]
    
    # Apply bandpass filter (1-40 Hz)
    from scipy import signal as sig
    nyquist = fs / 2
    b, a = sig.butter(4, [1/nyquist, 40/nyquist], btype='band')
    filtered_signal = sig.filtfilt(b, a, signal)
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_vector, signal, label='Original', alpha=0.7)
    plt.plot(time_vector, filtered_signal, label='Filtered (1-40 Hz)', alpha=0.7)
    plt.title(f'Channel {labels[0]} - Time Domain')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (μV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Frequency domain
    plt.subplot(2, 1, 2)
    from scipy.fft import fft, fftfreq
    fft_orig = fft(signal)
    fft_filt = fft(filtered_signal)
    fft_freq = fftfreq(len(signal), 1/fs)
    
    positive_freq_mask = fft_freq > 0
    plt.semilogy(fft_freq[positive_freq_mask], 
                np.abs(fft_orig[positive_freq_mask])**2, 
                label='Original', alpha=0.7)
    plt.semilogy(fft_freq[positive_freq_mask], 
                np.abs(fft_filt[positive_freq_mask])**2, 
                label='Filtered', alpha=0.7)
    plt.title(f'Channel {labels[0]} - Frequency Domain')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 50)
    
    plt.tight_layout()
    plt.show()

def example_3_feature_extraction():
    """Example 3: Extract features from EEG"""
    print("\n=== Example 3: Feature Extraction ===")
    
    # Load data
    eeg_signals, time_vector, labels, fs = load_eeg_data("emotiv-08-07-2025_17-37-01.mat")
    
    # Extract features from first channel
    signal = eeg_signals[0, :]
    
    # Time domain features
    features = {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'variance': np.var(signal),
        'rms': np.sqrt(np.mean(signal**2)),
        'peak_to_peak': np.max(signal) - np.min(signal),
        'zero_crossings': np.sum(np.diff(np.sign(signal - np.mean(signal))) != 0)
    }
    
    print(f"Features for channel {labels[0]}:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
    
    # Frequency domain features
    from scipy.fft import fft, fftfreq
    fft_vals = fft(signal)
    fft_freq = fftfreq(len(signal), 1/fs)
    
    positive_freq_mask = fft_freq > 0
    frequencies = fft_freq[positive_freq_mask]
    power_spectrum = np.abs(fft_vals[positive_freq_mask])**2
    
    # Power in different bands
    delta_power = np.sum(power_spectrum[(frequencies >= 0.5) & (frequencies < 4)])
    theta_power = np.sum(power_spectrum[(frequencies >= 4) & (frequencies < 8)])
    alpha_power = np.sum(power_spectrum[(frequencies >= 8) & (frequencies < 13)])
    beta_power = np.sum(power_spectrum[(frequencies >= 13) & (frequencies < 30)])
    
    print(f"\nFrequency band powers:")
    print(f"  Delta (0.5-4 Hz): {delta_power:.2e}")
    print(f"  Theta (4-8 Hz): {theta_power:.2e}")
    print(f"  Alpha (8-13 Hz): {alpha_power:.2e}")
    print(f"  Beta (13-30 Hz): {beta_power:.2e}")

def example_4_comparison_between_files():
    """Example 4: Compare data between files"""
    print("\n=== Example 4: File Comparison ===")
    
    # Load both files
    eeg1, time1, labels1, fs1 = load_eeg_data("emotiv-08-07-2025_17-37-01.mat")
    eeg2, time2, labels2, fs2 = load_eeg_data("emotiv-08-07-2025_17-37-29.mat")
    
    print(f"File 1: {eeg1.shape}, duration: {time1[-1] - time1[0]:.2f}s")
    print(f"File 2: {eeg2.shape}, duration: {time2[-1] - time2[0]:.2f}s")
    
    # Compare first channel
    signal1 = eeg1[0, :]
    signal2 = eeg2[0, :]
    
    print(f"\nComparison of channel {labels1[0]}:")
    print(f"  File 1 - mean: {np.mean(signal1):.1f}μV, std: {np.std(signal1):.1f}μV")
    print(f"  File 2 - mean: {np.mean(signal2):.1f}μV, std: {np.std(signal2):.1f}μV")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time1, signal1, label='File 1', alpha=0.7)
    plt.title(f'Channel {labels1[0]} - File 1')
    plt.ylabel('Amplitude (μV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(time2, signal2, label='File 2', alpha=0.7, color='orange')
    plt.title(f'Channel {labels1[0]} - File 2')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (μV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Run all examples"""
    print("EEG Data Analysis Examples")
    print("="*50)
    
    example_1_basic_analysis()
    example_2_signal_filtering()
    example_3_feature_extraction()
    example_4_comparison_between_files()
    
    print(f"\n{'='*50}")
    print("Examples complete!")
    print("\nYou can now:")
    print("1. Load your own .mat files using load_eeg_data()")
    print("2. Apply filters to clean the signals")
    print("3. Extract features for analysis")
    print("4. Compare different recordings")
    print("5. Use the EEGAnalyzer class for more advanced analysis")

if __name__ == "__main__":
    main() 