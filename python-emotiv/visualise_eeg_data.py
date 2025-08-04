@ -0,0 +1,422 @@
#!/usr/bin/env python3
"""
View and analyze saved EEG data from record_eeg_with_live_filter.py
Works with .mat, .npz, and .json files
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob
from datetime import datetime

# Try to import scipy for .mat files
try:
    import scipy.io
    HAS_SCIPY = True
    print("✓ scipy available - can load .mat files")
except ImportError:
    HAS_SCIPY = False
    print("⚠️ scipy not available - .mat files will be skipped")

def find_latest_data_files():
    """Find the most recent EEG data files"""
    print("🔍 Looking for EEG data files...")
    
    # Look for different file types
    mat_files = glob.glob("*filtered*.mat")
    npz_files = glob.glob("*filtered*.npz") 
    json_files = glob.glob("*filtered*.json")
    
    if not mat_files and not npz_files and not json_files:
        print("❌ No EEG data files found!")
        print("Make sure you're in the directory where you saved the data.")
        return None, None, None
    
    # Get the most recent files
    latest_mat = max(mat_files, key=os.path.getctime) if mat_files else None
    latest_npz = max(npz_files, key=os.path.getctime) if npz_files else None  
    latest_json = max(json_files, key=os.path.getctime) if json_files else None
    
    print(f"Found data files:")
    if latest_mat:
        print(f"  📊 MATLAB: {latest_mat}")
    if latest_npz:
        print(f"  🐍 NumPy: {latest_npz}")
    if latest_json:
        print(f"  📄 JSON: {latest_json}")
        
    return latest_mat, latest_npz, latest_json

def load_npz_data(filename):
    """Load data from NumPy .npz file"""
    print(f"\n📂 Loading NumPy data from: {filename}")
    
    try:
        data = np.load(filename)
        
        print("Available data arrays:")
        for key in data.files:
            array = data[key]
            if isinstance(array, np.ndarray):
                print(f"  - {key}: shape {array.shape}, dtype {array.dtype}")
            else:
                print(f"  - {key}: {array}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading NPZ file: {e}")
        return None

def load_mat_data(filename):
    """Load data from MATLAB .mat file"""
    if not HAS_SCIPY:
        print("❌ Cannot load .mat file - scipy not installed")
        return None
        
    print(f"\n📂 Loading MATLAB data from: {filename}")
    
    try:
        data = scipy.io.loadmat(filename)
        
        print("Available data structures:")
        for key, value in data.items():
            if not key.startswith('__'):  # Skip MATLAB metadata
                if isinstance(value, np.ndarray):
                    print(f"  - {key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"  - {key}: {type(value)}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading MAT file: {e}")
        return None

def load_json_metadata(filename):
    """Load metadata from JSON file"""
    print(f"\n📂 Loading JSON metadata from: {filename}")
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        if 'metadata' in data:
            metadata = data['metadata']
            print("📋 Recording Information:")
            print(f"  - Date: {metadata.get('recording_date', 'Unknown')}")
            print(f"  - Duration: {metadata.get('duration_seconds', 'Unknown')} seconds")
            print(f"  - Samples: {metadata.get('total_samples', 'Unknown')}")
            print(f"  - Sampling Rate: {metadata.get('sampling_rate', 'Unknown')} Hz")
            print(f"  - Channels: {metadata.get('channels', 'Unknown')}")
            
            if 'frequency_bands' in metadata:
                print("🌊 Frequency Bands:")
                for band, freq_range in metadata['frequency_bands'].items():
                    print(f"  - {band.capitalize()}: {freq_range}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading JSON file: {e}")
        return None

def plot_eeg_data(data, data_type='npz', start_time=None, end_time=None):
    """Plot EEG data"""
    print(f"\n📈 Creating plots...")
    
    try:
        if data_type == 'npz':
            # Extract data from NPZ
            timestamps = data.get('timestamps', None)
            raw_data = data.get('raw_data', None)
            
            # Get frequency bands
            bands = {}
            for key in data.files:
                if '_filtered' in key:
                    band_name = key.replace('_filtered', '')
                    bands[band_name] = data[key]
        
        elif data_type == 'mat':
            # Extract data from MAT file structure
            # The data is typically stored in data['data'] with substructures
            if 'data' in data:
                raw_structure = data['data']['raw'][0, 0] if 'raw' in data['data'].dtype.names else None
                if raw_structure is not None:
                    timestamps = raw_structure['time'].flatten()
                    raw_data = raw_structure['trial']
                else:
                    print("❌ Could not find raw data structure in MAT file")
                    return
                
                # Get filtered bands
                bands = {}
                for key in data['data'].dtype.names:
                    if '_filtered' in key:
                        band_name = key.replace('_filtered', '')
                        band_data = data['data'][key][0, 0]['trial']
                        bands[band_name] = band_data
            else:
                print("❌ Could not find expected data structure in MAT file")
                return

        if timestamps is None or len(timestamps) == 0:
            print("❌ No timestamp data found")
            return
            
        # Select the desired time range
        if start_time is not None and end_time is not None:
            print(f"📊 Plotting time range: {start_time:.2f}s to {end_time:.2f}s")
            indices = np.where((timestamps >= start_time) & (timestamps <= end_time))[0]
            if len(indices) == 0:
                print(f"❌ No data found in time range {start_time:.2f}s to {end_time:.2f}s")
                return
        else:
            indices = np.arange(len(timestamps))
            print(f"📊 Plotting full recording: 0s to {timestamps[-1]:.2f}s")

        # Apply time selection
        timestamps_plot = timestamps[indices]
        
        if raw_data is not None:
            if len(raw_data.shape) > 1:
                raw_data_plot = raw_data[indices, :]
            else:
                raw_data_plot = raw_data[indices]
        else:
            raw_data_plot = None
            
        bands_plot = {}
        for band_name, band_data in bands.items():
            if band_data is not None and len(band_data) > 0:
                # Handle potential length mismatch between timestamps and filtered data
                band_indices = indices[indices < len(band_data)]
                if len(band_data.shape) > 1:
                    bands_plot[band_name] = band_data[band_indices, :]
                else:
                    bands_plot[band_name] = band_data[band_indices]
        # Create subplots
        num_plots = 1 + len(bands_plot)  # Raw + filtered bands
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 2*num_plots))
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot raw data
        if raw_data_plot is not None:
            if len(raw_data_plot.shape) > 1:
                # Multi-channel: plot first channel
                axes[plot_idx].plot(timestamps_plot, raw_data_plot[:, 0], 'b-', linewidth=0.5)
                axes[plot_idx].set_title('Raw EEG Data (Channel 1: F3)')
            else:
                # Single channel
                axes[plot_idx].plot(timestamps_plot, raw_data_plot, 'b-', linewidth=0.5)
                axes[plot_idx].set_title('Raw EEG Data')
            
            axes[plot_idx].set_ylabel('Amplitude (μV)')
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot filtered bands
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, (band_name, band_data) in enumerate(bands_plot.items()):
            if band_data is not None and len(band_data) > 0:
                color = colors[i % len(colors)]
                
                # Create time array that matches the band data length
                band_timestamps = timestamps_plot[:len(band_data)]
                
                if len(band_data.shape) > 1:
                    # Multi-channel: plot first channel
                    axes[plot_idx].plot(band_timestamps, band_data[:, 0], color=color, linewidth=0.8)
                    axes[plot_idx].set_title(f'{band_name.capitalize()} Band (Channel 1: F3)')
                else:
                    # Single channel
                    axes[plot_idx].plot(band_timestamps, band_data, color=color, linewidth=0.8)
                    axes[plot_idx].set_title(f'{band_name.capitalize()} Band')
                
                axes[plot_idx].set_ylabel('Amplitude (μV)')
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
        
        # Add x-label to bottom plot
        axes[-1].set_xlabel('Time (seconds)')
        
        plt.tight_layout()
        plt.show()
        
        print("✓ Plots created successfully!")
        print("💡 Close the plot window to continue")
        
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        import traceback
        traceback.print_exc()

def analyze_frequency_bands(data, data_type='npz'):
    """Analyze power in different frequency bands"""
    print(f"\n🔬 Analyzing frequency band power...")
    
    try:
        bands = {}
        
        if data_type == 'npz':
            for key in data.files:
                if '_filtered' in key:
                    band_name = key.replace('_filtered', '')
                    bands[band_name] = data[key]
        
        elif data_type == 'mat':
            if 'data' in data:
                for key in data['data'].dtype.names:
                    if '_filtered' in key:
                        band_name = key.replace('_filtered', '')
                        band_data = data['data'][key][0, 0]['trial']
                        bands[band_name] = band_data
        
        if not bands:
            print("❌ No filtered band data found")
            return
        
        print("📊 Average Power by Frequency Band:")
        print("-" * 40)
        
        band_powers = {}
        for band_name, band_data in bands.items():
            if band_data is not None and len(band_data) > 0:
                # Calculate RMS power
                if len(band_data.shape) > 1:
                    # Multi-channel: average across channels
                    power = np.sqrt(np.mean(band_data**2, axis=1)).mean()
                else:
                    # Single channel
                    power = np.sqrt(np.mean(band_data**2))
                
                band_powers[band_name] = power
                print(f"{band_name.capitalize():>8}: {power:8.2f} μV")
        
        # Create bar plot of band powers
        if band_powers:
            plt.figure(figsize=(10, 6))
            bands = list(band_powers.keys())
            powers = list(band_powers.values())
            
            bars = plt.bar(bands, powers, color=['red', 'orange', 'green', 'blue', 'purple'])
            plt.title('Average Power by Frequency Band')
            plt.xlabel('Frequency Band')
            plt.ylabel('RMS Power (μV)')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, power in zip(bars, powers):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(powers)*0.01,
                        f'{power:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
            
            print("✓ Band power analysis complete!")
        
    except Exception as e:
        print(f"❌ Error analyzing frequency bands: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to examine EEG data"""
    print("🧠 EEG Data Viewer")
    print("=" * 50)
    
    # Find data files
    mat_file, npz_file, json_file = find_latest_data_files()
    
    if not mat_file and not npz_file and not json_file:
        return
    
    # Load metadata first if available
    if json_file:
        json_data = load_json_metadata(json_file)
    
    # Choose which format to load
    data = None
    data_type = None
    
    if npz_file:
        data = load_npz_data(npz_file)
        data_type = 'npz'
    elif mat_file and HAS_SCIPY:
        data = load_mat_data(mat_file)
        data_type = 'mat'
    
    if data is None:
        print("❌ Could not load any data files")
        return
    
    # Ask user what they want to do
    print(f"\n🎯 What would you like to do?")
    print("1. View time series plots")
    print("2. Analyze frequency band power")
    print("3. Both")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice in ['1', '3']:
        # Get timestamps for time range selection
        if data_type == 'npz':
            timestamps = data.get('timestamps', None)
        elif data_type == 'mat' and 'data' in data:
            raw_structure = data['data']['raw'][0, 0] if 'raw' in data['data'].dtype.names else None
            if raw_structure is not None:
                timestamps = raw_structure['time'].flatten()
            else:
                timestamps = None
        
        if timestamps is not None:
            print(f"\n⏱️  Recording duration: 0s to {timestamps[-1]:.2f}s ({len(timestamps)} samples)")
            
            # Ask user for time range
            try:
                start_input = input("Enter start time in seconds (or press Enter for beginning): ").strip()
                start_time = float(start_input) if start_input else 0
                
                end_input = input("Enter end time in seconds (or press Enter for end): ").strip()
                end_time = float(end_input) if end_input else timestamps[-1]
                
                # Validate time range
                if start_time < 0:
                    start_time = 0
                if end_time > timestamps[-1]:
                    end_time = timestamps[-1]
                if end_time <= start_time:
                    print("❌ Invalid time range. Using full recording.")
                    start_time, end_time = None, None
                    
            except ValueError:
                print("❌ Invalid input. Using full recording.")
                start_time, end_time = None, None
        else:
            start_time, end_time = None, None
        
        plot_eeg_data(data, data_type, start_time, end_time)
    
    if choice in ['2', '3']:
        analyze_frequency_bands(data, data_type)
    
    if choice == '4':
        print("👋 Goodbye!")
    
    print(f"\n🎉 Analysis complete!")
    print(f"💡 Your EEG data shows brain activity across different frequency bands:")
    print(f"   - Delta: Deep sleep, unconscious processes")
    print(f"   - Theta: Memory, creativity, REM sleep") 
    print(f"   - Alpha: Relaxed awareness, attention")
    print(f"   - Beta: Active thinking, cognitive processing")
    print(f"   - Gamma: High-level cognitive functions")

if __name__ == "__main__":
    main()