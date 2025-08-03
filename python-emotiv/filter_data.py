'''

Objective:
To isolate the EEG signal components related to cognitive processes (such as attention, focus, and thinking) 
by removing noise and filtering the signal to retain only the relevant brainwave frequencies.


Pipeline:
- Load and preprocess EEG data from Emotiv headsets.
- Apply Fourier-based filtering to isolate relevant frequency bands.
- Dynamically remove artifacts (e.g., movement noise).
- Extract and visualize cognitive processing signals (e.g., beta band).
- Plot the results in real-time for analysis.


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


Research:
Decoding the cognitive states of attention and distraction in a real-life setting using EEG
    https://www.nature.com/articles/s41598-022-24417-w

'''

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# === CONFIGURATION ===
MAT_FILE_PATH = 'emotiv-07-08-2025_18-31-33.mat' 
EEG_CHANNEL_KEY = 'data'
FS = 128 # sampling rate (Hz) 
TARGET_FS = 128  # Set to 128 for real-time processing
FILTER_ORDER = 4

# Frequency bands
BETA_LOW = 13
BETA_HIGH = 30


# === 1. LOAD AND PREPROCESS EEG DATA ===
mat = scipy.io.loadmat(MAT_FILE_PATH)
raw_data_struct = mat[EEG_CHANNEL_KEY]
eeg_matrix = raw_data_struct['trial'][0, 0]  # shape: (channels, N)
time_vector = raw_data_struct['time'][0, 0].squeeze()  # shape: (N,)

# Downsample if needed
if FS != TARGET_FS:
    eeg_matrix = resample(eeg_matrix, int(eeg_matrix.shape[1] * TARGET_FS / FS), axis=1)
    time_vector = np.linspace(time_vector[0], time_vector[-1], eeg_matrix.shape[1])
    FS = TARGET_FS

# === 2. FOURIER-BASED FILTERING (Bandpass) ===
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# === 3. DYNAMIC ARTIFACT REMOVAL ===
def dynamic_artifact_rejection(signal, window=256, z_thresh=3):
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

# === 4. EXTRACT & VISUALIZE COGNITIVE PROCESSING SIGNALS (Beta Band) ===
# Select a channel (e.g., channel 0 = AF3, or choose based on your montage)
channel_idx = 0
raw_signal = eeg_matrix[channel_idx]

# Bandpass filter for beta band
beta_signal = bandpass_filter(raw_signal, BETA_LOW, BETA_HIGH, FS, FILTER_ORDER)

# Artifact rejection
beta_clean = dynamic_artifact_rejection(beta_signal, window=int(FS*2), z_thresh=3)


# # === 5. PLOT RESULTS (Simulate Real-Time) ===
# plt.figure(figsize=(12, 4))
# plt.plot(time_vector, raw_signal, label='Raw EEG')
# plt.plot(time_vector, beta_clean, label='Beta Band (Cleaned)', linewidth=2)
# plt.title('EEG Channel {}: Raw vs. Beta Band (Cleaned)'.format(channel_idx))
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude (µV)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # --- Real-time simulation: plot in sliding windows ---
# window_sec = 2
# step_sec = 0.5
# window_samples = int(window_sec * FS)
# step_samples = int(step_sec * FS)

# plt.figure(figsize=(12, 4))
# for start in range(0, len(beta_clean) - window_samples, step_samples):
#     plt.clf()
#     plt.plot(time_vector[start:start+window_samples], beta_clean[start:start+window_samples])
#     plt.title(f'Real-Time Beta Band (Cleaned) [{time_vector[start]:.2f}-{time_vector[start+window_samples]:.2f}s]')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude (µV)')
#     plt.pause(0.05)
# plt.close()