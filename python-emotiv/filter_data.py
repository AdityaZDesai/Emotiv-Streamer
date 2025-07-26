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



Research:
Decoding the cognitive states of attention and distraction in a real-life setting using EEG
    https://www.nature.com/articles/s41598-022-24417-w

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

from scipy.signal import resample

# Downsample data
target_fs = 256
downsample_factor = int(FS / target_fs) if FS > target_fs else 1
if downsample_factor > 1:
    eeg_matrix = resample(eeg_matrix, eeg_matrix.shape[1] // downsample_factor, axis=1)
    time_vector = time_vector[::downsample_factor]
    FS = target_fs

# Broad bandpass filter for preprocessing
PRE_LOW = 0.5
PRE_HIGH = 45
eeg_matrix = bandpass_filter(eeg_matrix, PRE_LOW, PRE_HIGH, FS, FILTER_ORDER)

# ICA for artifact removal (placeholder)
# from mne.preprocessing import ICA
# ica = ICA(n_components=14, random_state=97)
# ica.fit(eeg_matrix)  # EEG matrix should be structured as epochs for MNE
# eeg_matrix = ica.apply(eeg_matrix)

import numpy as np

# --- Hardcoded time window for analysis (17-26 seconds) ---
start_time = 16
end_time = 24
start_idx = int(start_time * FS)
end_idx = int(end_time * FS)
segment = eeg_matrix[:, start_idx:end_idx]

filtered_beta = bandpass_filter(segment, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
filtered_beta_clean = dynamic_artifact_rejection(filtered_beta[0])

plt.figure(figsize=(12, 4))
plt.plot(time_vector, eeg_matrix[0], label='Original EEG')
plt.axvspan(start_time, end_time, color='orange', alpha=0.3, label='Selected Time Window')
plt.title('Full EEG with Highlighted Analysis Window')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Beta band component (cleaned) for the selected window
plt.figure(figsize=(12, 5))
plt.plot(time_vector[start_idx:end_idx], filtered_beta_clean, label='Beta Band (Cleaned)', linewidth=2)
plt.title('Beta Band Component (Cleaned) in Selected Window')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.grid(True)
plt.tight_layout()
plt.show()