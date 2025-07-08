# MATLAB .mat File Reader

This directory contains Python scripts to read and analyze MATLAB .mat files, specifically designed for Emotiv EEG data.

## Files

1. **`simple_mat_reader.py`** - A simple script to read a single .mat file
2. **`read_mat_file.py`** - A comprehensive script that processes all .mat files in the directory

## Installation

Make sure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

Or install the dependencies manually:

```bash
pip install scipy numpy matplotlib
```

## Usage

### Simple Mat Reader

To read a specific .mat file:

```python
# Edit the file_path variable in simple_mat_reader.py
file_path = "your_file.mat"

# Run the script
python simple_mat_reader.py
```

### Comprehensive Mat Reader

To process all .mat files in the directory:

```bash
python read_mat_file.py
```

This script will:
- Find all .mat files in the directory
- Load each file and display its contents
- Attempt to plot EEG data if found
- Save detailed information to text files

## Example Output

When you run the simple reader, you'll see output like:

```
Loading file: emotiv-08-07-2025_17-37-01.mat

Successfully loaded emotiv-08-07-2025_17-37-01.mat
==================================================
Variables found in the .mat file:

Variable: data
Type: ndarray
Shape: (14, 1024)
Data type: float64
Min value: -1234.5678
Max value: 1234.5678
Mean value: 0.1234
First row, first 5 values: [1.234 2.345 3.456 4.567 5.678]
```

## Accessing Data

After loading a .mat file, you can access the data like this:

```python
import scipy.io

# Load the file
mat = scipy.io.loadmat('your_file.mat')

# Access specific variables
data = mat['data']  # Replace 'data' with the actual variable name
print(f"Data shape: {data.shape}")
```

## Notes

- The scripts automatically skip MATLAB's internal variables (those starting with `__`)
- For EEG data plotting, the script looks for common variable names like 'eeg', 'data', 'EEG', etc.
- The comprehensive script saves detailed information about each file to separate text files
- Make sure your .mat files are in the same directory as the scripts, or provide the full path

## Troubleshooting

If you encounter errors:

1. **File not found**: Make sure the .mat file exists in the specified path
2. **Import errors**: Install the required dependencies using `pip install -r requirements.txt`
3. **Memory errors**: Large .mat files might require significant memory; consider processing smaller chunks 