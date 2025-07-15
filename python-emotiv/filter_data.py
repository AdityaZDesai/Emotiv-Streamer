'''

Objective:
To isolate the EEG signal components related to cognitive processes (such as attention, focus, and thinking) 
by removing noise and filtering the signal to retain only the relevant brainwave frequencies.

'''

import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# === CONFIGURATION ===
MAT_FILE_PATH = 'emotiv-07-08-2025_18-31-33.mat'  # Make sure this file exists in the same directory or provide full path
EEG_CHANNEL_KEY = 'data'              # change if your .mat uses a different key
TIME_KEY = 'time'                     # optional, depends on how time is stored
FS = 128                              # sampling rate (Hz) — adjust based on your device
LOWCUT = 1                            # Hz
HIGHCUT = 10                          # Hz
FILTER_ORDER = 4

# === BANDPASS FILTER ===
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# === LOAD .MAT FILE ===
mat = scipy.io.loadmat(MAT_FILE_PATH)
print("Top-level keys in .mat file:", mat.keys())

# You may need to inspect the structure depending on your recording format
raw_data_struct = mat[EEG_CHANNEL_KEY]
print("Raw EEG data struct type:", type(raw_data_struct))
print("Raw EEG data struct content:", repr(raw_data_struct))

eeg_matrix = raw_data_struct['trial'][0, 0]  # shape: (14, N)
time_vector = raw_data_struct['time'][0, 0].squeeze()  # shape: (N,)

# === APPLY BANDPASS FILTER ===
filtered = bandpass_filter(eeg_matrix, LOWCUT, HIGHCUT, FS, FILTER_ORDER)

# === PLOT ===
plt.figure(figsize=(12, 4))
plt.plot(time_vector, filtered[0])  # Plot the first channel
plt.title(f'Filtered EEG Signal ({LOWCUT}-{HIGHCUT} Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.grid(True)
plt.tight_layout()
plt.show()
