'''

Objective:
To isolate the EEG signal components related to cognitive processes (such as attention, focus, and thinking) 
by removing noise and filtering the signal to retain only the relevant brainwave frequencies.

'''
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# === CONFIGURATION ===
MAT_FILE_PATH = 'emotiv-07-08-2025_18-31-33.mat'  # e.g., 'emotiv-0708-2025_18-31-33.mat'
EEG_CHANNEL_KEY = 'data'              # change if your .mat uses a different key
TIME_KEY = 'time'                     # optional, depends on how time is stored
FS = 128                              # sampling rate (Hz) — adjust based on your device
LOWCUT = 8                            # Hz
HIGHCUT = 13                          # Hz
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
print("Keys in .mat file:", mat.keys())

# You may need to inspect the structure depending on your recording format
raw_data = mat[EEG_CHANNEL_KEY].squeeze()   # 1D array of EEG values
try:
    time_vector = mat[TIME_KEY].squeeze()
except:
    time_vector = [i / FS for i in range(len(raw_data))]  # fallback if no time stored

# === APPLY BANDPASS FILTER ===
filtered = bandpass_filter(raw_data, LOWCUT, HIGHCUT, FS, FILTER_ORDER)

# === PLOT ===
plt.figure(figsize=(12, 4))
plt.plot(time_vector, filtered)
plt.title(f'Filtered EEG Signal ({LOWCUT}-{HIGHCUT} Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.grid(True)
plt.tight_layout()
plt.show()
