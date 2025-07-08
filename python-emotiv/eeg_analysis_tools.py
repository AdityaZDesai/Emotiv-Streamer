#!/usr/bin/env python3
"""
EEG Analysis Tools
Common functions for EEG signal processing and analysis
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import pandas as pd

class EEGAnalyzer:
    def __init__(self, file_path):
        """
        Initialize EEG analyzer with data from .mat file
        """
        self.file_path = file_path
        self.load_data()
    
    def load_data(self):
        """Load EEG data from .mat file"""
        mat_data = scipy.io.loadmat(self.file_path, squeeze_me=True, struct_as_record=False)
        
        self.eeg_signals = mat_data['data'].trial
        self.time_vector = mat_data['data'].time
        self.channel_labels = mat_data['data'].label
        self.sampling_rate = mat_data['data'].fsample
        
        print(f"Loaded EEG data: {self.eeg_signals.shape}")
        print(f"Duration: {self.time_vector[-1] - self.time_vector[0]:.2f} seconds")
        print(f"Channels: {self.channel_labels}")
    
    def filter_signal(self, data, low_freq=1.0, high_freq=40.0, filter_type='bandpass'):
        """
        Apply bandpass filter to EEG signal
        
        Args:
            data: Input signal
            low_freq: Lower frequency cutoff (Hz)
            high_freq: Upper frequency cutoff (Hz)
            filter_type: 'bandpass', 'lowpass', or 'highpass'
        """
        nyquist = self.sampling_rate / 2
        
        if filter_type == 'bandpass':
            b, a = signal.butter(4, [low_freq/nyquist, high_freq/nyquist], btype='band')
        elif filter_type == 'lowpass':
            b, a = signal.butter(4, high_freq/nyquist, btype='low')
        elif filter_type == 'highpass':
            b, a = signal.butter(4, low_freq/nyquist, btype='high')
        
        filtered_data = signal.filtfilt(b, a, data)
        return filtered_data
    
    def extract_features(self, channel_idx=0):
        """
        Extract basic features from EEG signal
        
        Args:
            channel_idx: Index of channel to analyze
        """
        signal_data = self.eeg_signals[channel_idx, :]
        
        # Time domain features
        features = {
            'mean': np.mean(signal_data),
            'std': np.std(signal_data),
            'variance': np.var(signal_data),
            'rms': np.sqrt(np.mean(signal_data**2)),
            'peak_to_peak': np.max(signal_data) - np.min(signal_data),
            'skewness': self._skewness(signal_data),
            'kurtosis': self._kurtosis(signal_data)
        }
        
        # Frequency domain features
        fft_vals = fft(signal_data)
        fft_freq = fftfreq(len(signal_data), 1/self.sampling_rate)
        
        # Power in different frequency bands
        positive_freq_mask = fft_freq > 0
        frequencies = fft_freq[positive_freq_mask]
        power_spectrum = np.abs(fft_vals[positive_freq_mask])**2
        
        # Define frequency bands
        delta_mask = (frequencies >= 0.5) & (frequencies < 4)
        theta_mask = (frequencies >= 4) & (frequencies < 8)
        alpha_mask = (frequencies >= 8) & (frequencies < 13)
        beta_mask = (frequencies >= 13) & (frequencies < 30)
        gamma_mask = (frequencies >= 30) & (frequencies < 50)
        
        features.update({
            'delta_power': np.sum(power_spectrum[delta_mask]),
            'theta_power': np.sum(power_spectrum[theta_mask]),
            'alpha_power': np.sum(power_spectrum[alpha_mask]),
            'beta_power': np.sum(power_spectrum[beta_mask]),
            'gamma_power': np.sum(power_spectrum[gamma_mask]),
            'total_power': np.sum(power_spectrum)
        })
        
        return features
    
    def _skewness(self, data):
        """Calculate skewness"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3)
    
    def _kurtosis(self, data):
        """Calculate kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def plot_filtered_comparison(self, channel_idx=0, low_freq=1.0, high_freq=40.0):
        """
        Plot original vs filtered signal
        """
        original = self.eeg_signals[channel_idx, :]
        filtered = self.filter_signal(original, low_freq, high_freq)
        
        plt.figure(figsize=(15, 8))
        
        # Time domain
        plt.subplot(2, 1, 1)
        plt.plot(self.time_vector, original, label='Original', alpha=0.7)
        plt.plot(self.time_vector, filtered, label='Filtered', alpha=0.7)
        plt.title(f'Channel {self.channel_labels[channel_idx]} - Time Domain')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude (μV)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Frequency domain
        plt.subplot(2, 1, 2)
        fft_orig = fft(original)
        fft_filt = fft(filtered)
        fft_freq = fftfreq(len(original), 1/self.sampling_rate)
        
        positive_freq_mask = fft_freq > 0
        plt.semilogy(fft_freq[positive_freq_mask], 
                    np.abs(fft_orig[positive_freq_mask])**2, 
                    label='Original', alpha=0.7)
        plt.semilogy(fft_freq[positive_freq_mask], 
                    np.abs(fft_filt[positive_freq_mask])**2, 
                    label='Filtered', alpha=0.7)
        plt.title(f'Channel {self.channel_labels[channel_idx]} - Frequency Domain')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 50)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_all_channels(self):
        """
        Analyze all channels and return summary
        """
        results = []
        
        for i, label in enumerate(self.channel_labels):
            if i < self.eeg_signals.shape[0]:  # Handle dimension mismatch
                features = self.extract_features(i)
                features['channel'] = label
                results.append(features)
        
        return pd.DataFrame(results)
    
    def plot_feature_comparison(self):
        """
        Plot feature comparison across channels
        """
        df = self.analyze_all_channels()
        
        # Select key features to plot
        key_features = ['mean', 'std', 'rms', 'alpha_power', 'beta_power']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('EEG Features Across Channels', fontsize=16)
        
        for i, feature in enumerate(key_features):
            row = i // 3
            col = i % 3
            
            axes[row, col].bar(df['channel'], df[feature])
            axes[row, col].set_title(feature.replace('_', ' ').title())
            axes[row, col].tick_params(axis='x', rotation=45)
            axes[row, col].grid(True, alpha=0.3)
        
        # Remove the last subplot if not needed
        if len(key_features) < 6:
            axes[1, 2].remove()
        
        plt.tight_layout()
        plt.show()
        
        return df

def main():
    """Example usage"""
    # Analyze the first file
    analyzer = EEGAnalyzer("emotiv-08-07-2025_17-37-01.mat")
    
    print("\n=== EEG Analysis Results ===")
    
    # Extract features from first channel
    features = analyzer.extract_features(0)
    print(f"\nFeatures for channel {analyzer.channel_labels[0]}:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
    
    # Analyze all channels
    print(f"\nAnalyzing all channels...")
    df = analyzer.analyze_all_channels()
    print(f"\nChannel Analysis Summary:")
    print(df[['channel', 'mean', 'std', 'alpha_power', 'beta_power']].round(2))
    
    # Plot filtered comparison
    print(f"\nGenerating filtered signal comparison...")
    analyzer.plot_filtered_comparison(0)
    
    # Plot feature comparison
    print(f"\nGenerating feature comparison...")
    analyzer.plot_feature_comparison()
    
    print(f"\nAnalysis complete! You can now:")
    print(f"1. Use analyzer.filter_signal() to filter signals")
    print(f"2. Use analyzer.extract_features() to get features")
    print(f"3. Use analyzer.analyze_all_channels() for full analysis")

if __name__ == "__main__":
    main() 