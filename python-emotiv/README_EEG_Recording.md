# Emotiv EPOC EEG Data Recording

This guide explains how to record EEG data from the Emotiv EPOC headset and save it to .mat files.

## Prerequisites

1. **Emotiv EPOC headset** connected via USB
2. **Headset turned ON** (battery or USB power)
3. **Python dependencies** installed (see requirements.txt)
4. **USB permissions** (may need sudo on Linux/macOS)

## Quick Start

### 1. Test Your Headset

First, test if your headset is working:

```bash
# Test without sudo first
python test_headset.py

# If that fails, try with sudo
sudo python test_headset.py
```

### 2. Record EEG Data

To record EEG data and save it to a .mat file:

```bash
# Record with manual stop (Ctrl+C)
python record_eeg_data.py

# Or with sudo if needed
sudo python record_eeg_data.py
```

## Scripts Overview

### `test_headset.py`
- Tests if the Emotiv EPOC headset is working
- Checks data acquisition
- Shows contact quality
- Verifies all channels are functional

### `record_eeg_data.py`
- Records EEG data from the headset
- Saves data to timestamped .mat files
- Supports both manual stop (Ctrl+C) and timed recording
- Shows real-time progress

### `visualize_eeg.py`
- Loads and visualizes recorded .mat files
- Shows time series and power spectrum plots

### `eeg_analysis_tools.py`
- Comprehensive EEG analysis tools
- Feature extraction
- Signal filtering
- Statistical analysis

## Recording Options

When you run `record_eeg_data.py`, you'll be asked to choose:

1. **Manual Stop**: Record until you press Ctrl+C
2. **Timed Recording**: Record for a specific duration (in seconds)

## Output Files

Recorded data is saved as `.mat` files with the format:
```
emotiv-MM-DD-YYYY_HH-MM-SS.mat
```

The .mat file contains:
- `data.trial`: EEG signals (channels × time points)
- `data.time`: Time vector
- `data.label`: Channel labels
- `data.fsample`: Sampling rate (128 Hz)
- `data.sampleinfo`: Sample information
- `date`: Recording timestamp

## Troubleshooting

### "Headset not found" errors:
1. Make sure the headset is connected via USB
2. Make sure the headset is turned ON
3. Try running with `sudo`
4. Check USB recognition: `lsusb | grep Emotiv`

### "Permission denied" errors:
```bash
# On Linux/macOS, you may need to run with sudo
sudo python record_eeg_data.py
```

### "Headset turned off" errors:
1. Check battery level
2. Make sure the headset is properly powered
3. Try reconnecting the USB cable

### Poor signal quality:
1. Check contact quality in the test script
2. Make sure electrodes are properly positioned
3. Add saline solution if needed
4. Check for interference from other devices

## Example Workflow

1. **Test the headset**:
   ```bash
   python test_headset.py
   ```

2. **Record some data**:
   ```bash
   python record_eeg_data.py
   # Choose option 1 (manual stop)
   # Record for 30 seconds, then press Ctrl+C
   ```

3. **Visualize the data**:
   ```bash
   python visualize_eeg.py
   ```

4. **Analyze the data**:
   ```bash
   python eeg_analysis_tools.py
   ```

## Data Format

The recorded .mat files are compatible with:
- MATLAB
- Python (scipy.io.loadmat)
- FieldTrip (MATLAB toolbox)
- MNE-Python

## Channel Information

The Emotiv EPOC has 14 channels:
- **F3, F4**: Frontal electrodes
- **FC5, FC6**: Fronto-central electrodes  
- **AF3, AF4**: Anterior frontal electrodes
- **F7, F8**: Lateral frontal electrodes
- **T7, T8**: Temporal electrodes
- **P7, P8**: Parietal electrodes
- **O1, O2**: Occipital electrodes

## Sampling Rate

- **128 Hz** (128 samples per second per channel)
- Internal processing at 2048 Hz, downsampled to 128 Hz

## Notes

- The headset must be turned ON before running the scripts
- Recording stops automatically when you press Ctrl+C
- Data is saved immediately when recording stops
- Each recording creates a new .mat file with a timestamp
- The scripts handle battery packets automatically (no EEG data in these packets) 