'''

Objective:
To isolate the EEG signal components related to cognitive processes (such as attention, focus, and thinking) 
by removing noise and filtering the signal to retain only the relevant brainwave frequencies.


Note:
Frequency bands of interest:
- Delta: 0.5 - 4 Hz
- Theta: 4 - 8 Hz
- Alpha: 8 - 12 Hz
- Beta: 12 - 30 Hz
- Gamma: 30 - 100 Hz
- Focused attention: 8 - 12 Hz (Alpha band)
- Cognitive processing: 12 - 30 Hz (Beta band)
- Noise: Frequencies outside the range of interest (e.g., below 0.5 Hz or above 30 Hz)
'''

import scipy.io

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# === CONFIGURATION ===
MAT_FILE_PATH = 'emotiv-07-08-2025_18-31-33.mat'  # Make sure this file exists in the same directory or provide full path
EEG_CHANNEL_KEY = 'data'              # change if your .mat uses a different key
TIME_KEY = 'time'                     # optional, depends on how time is stored
FS = 128                              # sampling rate (Hz) — adjust based on your device
LOWCUT = 13                         # Hz
HIGHCUT = 30                          # Hz
FILTER_ORDER = 4

# === BANDPASS FILTER ===
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def dynamic_artifact_rejection(signal, window=256, z_thresh=3):
    """Remove artifacts using a moving z-score threshold."""
    import numpy as np
    clean = signal.copy()
    for i in range(0, len(signal), window):
        seg = signal[i:i+window]
        if len(seg) == 0:
            continue
        mean = np.nanmean(seg)
        std = np.nanstd(seg)
        z = (seg - mean) / (std if std > 0 else 1)
        seg[np.abs(z) > z_thresh] = np.nan
        clean[i:i+window] = seg
    return clean

# === LOAD .MAT FILE === 
mat = scipy.io.loadmat(MAT_FILE_PATH)
print("Top-level keys in .mat file:", mat.keys())

# You may need to inspect the structure depending on your recording format
raw_data_struct = mat[EEG_CHANNEL_KEY]
print("Raw EEG data struct type:", type(raw_data_struct))
print("Raw EEG data struct content:", repr(raw_data_struct))

eeg_matrix = raw_data_struct['trial'][0, 0]  # shape: (14, N)
time_vector = raw_data_struct['time'][0, 0].squeeze()  # shape: (N,)

# Filter for beta band (thinking)
filtered_beta = bandpass_filter(eeg_matrix, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
filtered_beta_clean = dynamic_artifact_rejection(filtered_beta[0])

# Plot
plt.figure(figsize=(12, 4))
plt.plot(time_vector, filtered_beta_clean)
plt.title(f'Beta Band ({LOWCUT}-{HIGHCUT} Hz) EEG, Dynamic Artifact Rejection')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.grid(True)
plt.tight_layout()
plt.show()


# === APPLY BANDPASS FILTER ===

filtered = bandpass_filter(eeg_matrix, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
ARTIFACT_THRESHOLD = 10  # µV

# Replace large spike values with NaN
filtered_clean = filtered[0].copy()
filtered_clean[abs(filtered_clean) > ARTIFACT_THRESHOLD] = float('nan')

# === PLOT ===
plt.figure(figsize=(12, 4))
plt.plot(time_vector, filtered_clean)  # Plot the cleaned signal for the first channel
plt.title(f'Filtered & Cleaned EEG Signal ({LOWCUT}-{HIGHCUT} Hz, Threshold {ARTIFACT_THRESHOLD} µV)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.grid(True)
plt.tight_layout()
plt.show()

# === PLOT FFT OF FIRST CHANNEL ===
import numpy as np

# Remove NaNs for FFT calculation
clean_segment = filtered_clean[~np.isnan(filtered_clean)]

# Compute FFT
fft_vals = np.fft.rfft(clean_segment)
fft_freq = np.fft.rfftfreq(len(clean_segment), d=1./FS)
power = np.abs(fft_vals)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(fft_freq, power)
plt.title("Frequency Spectrum of Cleaned EEG Signal (First Channel)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.grid(True)
plt.tight_layout()
plt.show()
