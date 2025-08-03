#!/usr/bin/env python3
"""
Record EEG data with live filtering from various EEG headsets and save to .mat file
Integrates real-time bandpass filtering and artifact rejection
"""

import scipy.io
import numpy as np
import time
import datetime
import signal
import sys
from scipy.signal import butter, filtfilt, lfilter
from scipy.signal import butter, filtfilt, lfilter
import matplotlib
# Let matplotlib use its default backend - more reliable
print("✓ Using matplotlib default backend")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from emotiv.epoc import EPOC, EPOCTurnedOffError, EPOCUSBError
from collections import deque

# Import filtering functions from filter_data.py
try:
    from filter_data import bandpass_filter, dynamic_artifact_rejection
    print("✓ Imported filtering functions from filter_data.py")
except ImportError:
    print("⚠️ Could not import from filter_data.py, using built-in filters")
    # Fallback functions if filter_data.py is not available
    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    
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

class LiveEEGFilter:
    """Real-time EEG filtering class using filter_data.py functions"""
    
    def __init__(self, fs=128, filter_order=4):
        self.fs = fs
        self.filter_order = filter_order
        
        # Frequency bands (matching filter_data.py configuration)
        self.DELTA_LOW = 0.5
        self.DELTA_HIGH = 4
        self.THETA_LOW = 4
        self.THETA_HIGH = 8
        self.ALPHA_LOW = 8
        self.ALPHA_HIGH = 12
        self.BETA_LOW = 12  # Changed to match filter_data.py (was 13)
        self.BETA_HIGH = 30
        self.GAMMA_LOW = 30
        self.GAMMA_HIGH = 50  # Low gamma for cognitive functions
        
        # Artifact rejection parameters
        self.artifact_window = int(fs * 2)  # 2-second window
        self.z_threshold = 3.0
        
        # Ring buffers for collecting data for batch processing
        self.filter_buffer_size = max(1024, fs * 4)  # 4 seconds buffer
        self.raw_buffer = deque(maxlen=self.filter_buffer_size)
        self.filtered_buffers = {
            'delta': deque(maxlen=self.filter_buffer_size),
            'theta': deque(maxlen=self.filter_buffer_size),
            'alpha': deque(maxlen=self.filter_buffer_size),
            'beta': deque(maxlen=self.filter_buffer_size),
            'gamma': deque(maxlen=self.filter_buffer_size),
            'broadband': deque(maxlen=self.filter_buffer_size)
        }
        
        # Define frequency bands for filtering
        self.filters = {
            'delta': (self.DELTA_LOW, self.DELTA_HIGH),
            'theta': (self.THETA_LOW, self.THETA_HIGH),
            'alpha': (self.ALPHA_LOW, self.ALPHA_HIGH),
            'beta': (self.BETA_LOW, self.BETA_HIGH),
            'gamma': (self.GAMMA_LOW, 45),
            'broadband': (0.5, min(45, fs/2 * 0.9))
        }
    
    def process_sample(self, raw_sample, channel=0):
        """Process a single EEG sample with movement-specific artifact handling"""
        # Add to raw buffer
        self.raw_buffer.append(raw_sample)
        
        # For real-time processing, we'll return the raw sample
        # and do batch filtering when we have enough data
        filtered_results = {}
        
        # If we have enough data, apply filtering to recent chunk
        if len(self.raw_buffer) >= self.artifact_window:
            recent_data = np.array(list(self.raw_buffer)[-self.artifact_window:])
            # Apply filtering for each band using filter_data.py functions
            for band, (low_freq, high_freq) in self.filters.items():
                try:
                    # Apply bandpass filter from filter_data.py
                    filtered_chunk = bandpass_filter(
                        recent_data, low_freq, high_freq, self.fs, self.filter_order
                    )
                    
                    # Band-specific artifact handling based on movement sensitivity
                    if band == 'alpha':
                        # Alpha: Less sensitive to head movement, more to eye/muscle artifacts
                        # Use gentler artifact rejection (higher threshold)
                        clean_chunk = dynamic_artifact_rejection(
                            filtered_chunk, window=self.artifact_window//3, z_thresh=2.5
                        )
                    
                    elif band == 'beta':
                        # Beta: More sensitive to head movement and muscle artifacts
                        # Use stronger artifact rejection
                        clean_chunk = dynamic_artifact_rejection(
                            filtered_chunk, window=self.artifact_window//4, z_thresh=2.0
                        )
                    
                    else:
                        # Standard processing for other bands
                        clean_chunk = dynamic_artifact_rejection(
                            filtered_chunk, window=self.artifact_window//4, z_thresh=self.z_threshold
                        )
                    
                    # Store the latest processed sample
                    if len(clean_chunk) > 0:
                        # Use the last sample that's not NaN
                        valid_samples = clean_chunk[~np.isnan(clean_chunk)]
                        if len(valid_samples) > 0:
                            filtered_results[band] = valid_samples[-1]
                        else:
                            filtered_results[band] = raw_sample  # Fallback to raw
                    else:
                        filtered_results[band] = raw_sample
                        
                    # Update the buffer with filtered values
                    if len(clean_chunk) > 0:
                        self.filtered_buffers[band].extend(clean_chunk[-10:])  # Keep last 10 samples
                        
                except Exception as e:
                    # Fallback to raw sample if filtering fails
                    filtered_results[band] = raw_sample
                    if 'Warning' not in str(e):  # Avoid spam
                        print(f"Warning: Filtering failed for {band} band: {e}")
        else:
            # Not enough data yet, return raw sample for all bands
            for band in self.filters:
                filtered_results[band] = raw_sample
        
        return filtered_results
    
    def get_power_in_band(self, band='beta', window_seconds=2):
        """Calculate power in a specific frequency band over recent window"""
        if band not in self.filtered_buffers:
            return 0
        
        window_samples = int(self.fs * window_seconds)
        buffer = self.filtered_buffers[band]
        
        if len(buffer) < window_samples:
            data = list(buffer)
        else:
            data = list(buffer)[-window_samples:]
        
        if len(data) == 0:
            return 0
        
        # Calculate RMS power
        power = np.sqrt(np.mean(np.array(data)**2))
        return power

class EEGRecorderWithFilter:
    def __init__(self):
        self.epoc = None
        self.recording = False
        self.data_buffer = []
        self.filtered_data_buffers = {
            'raw': [],
            'delta': [],
            'theta': [],
            'alpha': [],
            'beta': [],
            'gamma': [],
            'broadband': []
        }
        self.time_buffer = []
        self.start_time = None
        self.selected_device = None
        self.verbose_output = True

        # Filtering setup
        self.sampling_rate = 128
        self.live_filter = LiveEEGFilter(fs=self.sampling_rate)
        self.enable_live_filtering = True
        self.active_filter_bands = ['raw', 'alpha', 'beta']  # Default bands to display

        # FFT analysis buffers
        self.fft_buffer = []
        self.fft_frequencies = []
        self.fft_timestamps = []
        self.chunk_size = 128
        self.current_chunk = []
        self.chunk_start_time = None

        # Live plotting variables
        self.live_plot_active = False
        self.plot_data_buffers = {band: [] for band in self.filtered_data_buffers.keys()}
        self.plot_time_buffer = []
        self.plot_window_size = 10
        self.fig = None
        self.axes = None
        self.lines = {}
        self.ani = None

        # Accelerometer buffer for storing accel samples
        self.accel_buffer = []
        
    def list_devices(self):
        """List all available EEG devices"""
        print("🔍 Scanning for EEG devices...")
        try:
            devices = EPOC.list_all_devices()
            
            if not devices:
                print("No USB devices found!")
                return []
            
            eeg_devices = [d for d in devices if d['is_eeg']]
            
            if eeg_devices:
                print(f"\n✓ Found {len(eeg_devices)} EEG device(s):")
                for i, device in enumerate(eeg_devices):
                    print(f"  {i+1}. {device['manufacturer']} - {device['product']}")
                    print(f"     Serial: {device['serial']}")
                    print(f"     VID:PID = {device['vendor_id']}:{device['product_id']}")
                    print()
            else:
                print("No EEG devices found in the automatic scan.")
                print("You can still choose from all USB devices manually.")
            
            return devices
            
        except Exception as e:
            print(f"Error scanning for devices: {e}")
            return []
    
    def select_device(self):
        """Interactive device selection"""
        print("🎧 EEG Device Selection")
        print("="*50)
        
        devices = self.list_devices()
        if not devices:
            return None
        
        print("\nOptions:")
        print("1. Automatically select device")
        print("2. Manually select device") 
        print("3. Use default Emotiv detection")
        print("4. Cancel")
        
        while True:
            try:
                choice = int(input("\nSelect option (1-4): "))
                
                if choice == 1:
                    eeg_devices = [d for d in devices if d['is_eeg']]
                    if eeg_devices:
                        self.selected_device = eeg_devices[0]
                        print(f"✓ Auto-selected: {self.selected_device['manufacturer']} - {self.selected_device['product']}")
                        return self.selected_device
                    else:
                        print("No EEG devices found for auto-selection.")
                        continue
                        
                elif choice == 2:
                    self.selected_device = EPOC.select_device()
                    if self.selected_device:
                        print(f"✓ Selected: {self.selected_device['manufacturer']} - {self.selected_device['product']}")
                        return self.selected_device
                    else:
                        print("No device selected.")
                        return None
                        
                elif choice == 3:
                    self.selected_device = None
                    print("✓ Using default Emotiv device detection")
                    return "default"
                    
                elif choice == 4:
                    return None
                    
                else:
                    print("Invalid choice. Please try again.")
                    
            except ValueError:
                print("Please enter a valid number.")
    
    def setup_headset(self):
        """Initialize the EEG headset"""
        device_choice = self.select_device()
        
        if device_choice is None:
            print("No device selected. Exiting.")
            return False
        
        print(f"\n{'='*60}")
        print("🎧 Initializing EEG Headset...")
        print(f"{'='*60}")
        
        if device_choice == "default":
            print("Using default Emotiv device detection...")
            self.epoc = EPOC()
        else:
            print(f"Connecting to: {device_choice['manufacturer']} - {device_choice['product']}")
            self.epoc = EPOC(device_info=device_choice)
        
        print("✓ Headset initialized successfully!")
        
        # Update sampling rate if available
        if hasattr(self.epoc, 'sampling_rate') and getattr(self.epoc, 'sampling_rate', None):
            self.sampling_rate = getattr(self.epoc, 'sampling_rate', 128)
            # Reinitialize filter with correct sampling rate
            self.live_filter = LiveEEGFilter(fs=self.sampling_rate)
            print(f"✓ Updated sampling rate to: {self.sampling_rate} Hz")
        
        # Display device information
        if hasattr(self.epoc, 'channels') and getattr(self.epoc, 'channels', None):
            print(f"Channels: {getattr(self.epoc, 'channels', [])}")
        print(f"Sampling rate: {self.sampling_rate} Hz")
        
        if self.selected_device:
            print(f"Device: {self.selected_device['manufacturer']} - {self.selected_device['product']}")
            print(f"Serial: {self.selected_device['serial']}")
            print(f"VID:PID: {self.selected_device['vendor_id']}:{self.selected_device['product_id']}")
        
        return True
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C to stop recording gracefully"""
        print("\n\nStopping recording...")
        self.stop_recording()
        sys.exit(0)
    
    def print_eeg_data(self, data, channels, sample_count, elapsed_time):
        """Print formatted EEG data with channel information (same as record_eeg_data.py)"""
        if not data:
            return
            
        # Get channel names
        if channels:
            channel_names = channels
        else:
            # Default channel names if not available
            num_channels = len(data) if isinstance(data, (list, np.ndarray)) else 1
            channel_names = [f"Ch{i+1}" for i in range(num_channels)]
        
        # Format the data for display
        if isinstance(data, (list, np.ndarray)):
            # Convert to list if it's a numpy array
            data_list = data.tolist() if hasattr(data, 'tolist') else data
        else:
            # Single value
            data_list = [data]
        
        # Ensure we have the same number of channels and data points
        if len(channel_names) != len(data_list):
            # Adjust channel names to match data length
            if len(channel_names) < len(data_list):
                channel_names.extend([f"Ch{i+1}" for i in range(len(channel_names), len(data_list))])
            else:
                channel_names = channel_names[:len(data_list)]
        
        # Print header every 10 samples for readability
        if sample_count % 10 == 1:
            print(f"\n{'='*80}")
            print(f"📊 Real-time EEG Data - Sample #{sample_count} | Time: {elapsed_time:.3f}s")
            print(f"{'='*80}")
            print(f"{'Channel':<15} {'Value (μV)':<15} {'Status':<10}")
            print(f"{'-'*15} {'-'*15} {'-'*10}")
        
        # Print each channel's data
        for i, (channel, value) in enumerate(zip(channel_names, data_list)):
            # Determine signal quality/status
            if abs(value) > 1000:  # Very high values might indicate artifacts
                status = "⚠️ HIGH"
            elif abs(value) < 1:  # Very low values might indicate poor contact
                status = "⚠️ LOW"
            else:
                status = "✅ OK"
            
            print(f"{channel:<15} {value:<15.2f} {status:<10}")
        
        # Print summary every 50 samples
        if sample_count % 50 == 0:
            print(f"\n📈 Summary: {sample_count} samples, {elapsed_time:.1f}s elapsed")
            print(f"Average amplitude: {np.mean(np.abs(data_list)):.2f} μV")
            print(f"Max amplitude: {np.max(np.abs(data_list)):.2f} μV")
            print(f"Min amplitude: {np.min(np.abs(data_list)):.2f} μV")
            print(f"Data range: {np.max(data_list) - np.min(data_list):.2f} μV")
            print("-" * 80)
    
    def setup_live_plot_with_filters(self, channels):
        """Setup separate figure windows for each frequency band"""
        try:
            plt.ion()
            
            # Create separate figures for each frequency band
            self.figs = {}
            self.axes = {}
            self.lines = {}
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            # Screen position offsets for each window
            x_positions = [100, 300, 500, 700, 900]  # Horizontal positions
            y_positions = [100, 150, 200, 250, 300]  # Vertical positions (slightly staggered)
            
            for idx, band in enumerate(self.active_filter_bands):
                # Create individual figure for each band
                fig = plt.figure(figsize=(12, 6))
                
                # Position windows in a cascade pattern
                x_pos = x_positions[idx % len(x_positions)]
                y_pos = y_positions[idx % len(y_positions)]
                
                # Set window position (works on most backends)
                try:
                    manager = fig.canvas.manager
                    if hasattr(manager, 'window'):
                        if hasattr(manager.window, 'move'):
                            manager.window.move(x_pos, y_pos)
                        elif hasattr(manager.window, 'SetPosition'):
                            manager.window.SetPosition((x_pos, y_pos))
                except Exception:
                    pass  # Window positioning might not work on all systems
                
                ax = fig.add_subplot(111)
                
                # Store figure and axis
                self.figs[band] = fig
                self.axes[band] = ax
                self.lines[band] = []
                
                # Get channel names
                if channels:
                    channel_names = channels
                else:
                    channel_names = ['EEG']
                
                # Create lines for each channel
                for i, channel in enumerate(channel_names):
                    color = colors[i % len(colors)]
                    line, = ax.plot([], [], color=color, linewidth=1.5, 
                                   label=f'{channel}', alpha=0.9)
                    self.lines[band].append(line)
                
                # Setup individual plot
                ax.set_xlabel('Time (s)', fontsize=12)
                ax.set_ylabel('Amplitude (μV)', fontsize=12)
                ax.set_title(f'{band.capitalize()} Band ({self.get_band_range(band)}) - Live EEG', 
                           fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right')
                ax.set_xlim(0, self.plot_window_size)
                ax.set_ylim(-100, 100)
                
                # Style the plot
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(0.5)
                ax.spines['bottom'].set_linewidth(0.5)
                
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.1)
                
                # Force the window to show
                fig.canvas.draw()
                fig.canvas.flush_events()
            
            self.live_plot_active = True
            print(f"✓ Created {len(self.active_filter_bands)} separate plot windows")
            print(f"✓ Displaying frequency bands: {', '.join(self.active_filter_bands)}")
            print("💡 Each band now has its own window for better visibility!")
            
        except Exception as e:
            print(f"Warning: Could not setup live plotting windows: {e}")
            print("Live plotting will be disabled.")
            self.live_plot_active = False
    
    def get_band_range(self, band):
        """Get frequency range string for a band"""
        ranges = {
            'raw': 'Unfiltered',
            'delta': '0.5-4 Hz',
            'theta': '4-8 Hz', 
            'alpha': '8-12 Hz',
            'beta': '12-30 Hz',
            'gamma': '30-50 Hz (Low Gamma)',
            'broadband': '0.5-50 Hz'
        }
        return ranges.get(band, 'Unknown')
    
    def update_filtered_plot(self):
        """Update the filtered EEG plots in separate windows"""
        if not self.live_plot_active or not self.plot_time_buffer:
            return
        
        current_time = time.time() - self.start_time
        time_window_start = max(0, current_time - self.plot_window_size)
        
        # Filter data within time window
        valid_indices = [i for i, t in enumerate(self.plot_time_buffer) 
                        if time_window_start <= t <= current_time]
        
        if not valid_indices:
            return
        
        window_times = [self.plot_time_buffer[i] for i in valid_indices]
        
        # Update each frequency band in its own window
        for band in self.active_filter_bands:
            if band not in self.figs or band not in self.axes:
                continue
                
            fig = self.figs[band]
            ax = self.axes[band]
            ax.clear()
            
            if band in self.plot_data_buffers and self.plot_data_buffers[band]:
                window_data = [self.plot_data_buffers[band][i] for i in valid_indices]
                
                if window_data:
                    window_times_arr = np.array(window_times)
                    
                    # Check if we have multi-channel data by examining the first sample
                    if len(window_data) > 0 and hasattr(window_data[0], '__len__') and not isinstance(window_data[0], str):
                        # Multi-channel data: convert to proper array format
                        try:
                            window_data_arr = np.array(window_data)
                            if len(window_data_arr.shape) == 2:
                                num_channels = window_data_arr.shape[1]
                                colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
                                
                                for ch in range(min(num_channels, 8)):  # Limit to 8 channels for clarity
                                    color = colors[ch % len(colors)]
                                    ax.plot(window_times_arr, window_data_arr[:, ch], 
                                           color=color, linewidth=1.5, label=f'Ch{ch+1}', alpha=0.9)
                            else:
                                # Fallback for unexpected data structure
                                ax.plot(window_times_arr, window_data_arr, 
                                       color='blue', linewidth=1.5, alpha=0.9)
                        except Exception as e:
                            print(f"Debug: Error plotting multi-channel data for {band}: {e}")
                            # Fallback: try to plot as single channel
                            try:
                                window_data_flat = [item[0] if hasattr(item, '__len__') else item for item in window_data]
                                ax.plot(window_times_arr, window_data_flat, 
                                       color='blue', linewidth=1.5, alpha=0.9)
                            except:
                                print(f"Warning: Could not plot data for {band} band")
                    else:
                        # Single channel data
                        window_data_arr = np.array(window_data)
                        ax.plot(window_times_arr, window_data_arr, 
                               color='blue', linewidth=1.5, alpha=0.9)
            
            # Setup individual plot with enhanced styling
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Amplitude (μV)', fontsize=12)  
            ax.set_title(f'{band.capitalize()} Band ({self.get_band_range(band)}) - {current_time:.1f}s', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            ax.set_xlim(time_window_start, current_time)
            
            # Style the plot
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)
            
            # Auto-adjust y-axis with better scaling
            if band in self.plot_data_buffers and self.plot_data_buffers[band]:
                recent_data = self.plot_data_buffers[band][-100:]  # Last 100 samples
                if recent_data:
                    try:
                        data_arr = np.array(recent_data)
                        if data_arr.size > 0:
                            y_min = np.min(data_arr)
                            y_max = np.max(data_arr) 
                            y_range = y_max - y_min
                            if y_range > 0:
                                y_margin = y_range * 0.15  # Slightly larger margin
                                ax.set_ylim(y_min - y_margin, y_max + y_margin)
                            else:
                                ax.set_ylim(-50, 50)  # Default range if no variation
                    except Exception:
                        ax.set_ylim(-100, 100)  # Fallback range
            
            # Force redraw for this specific figure
            fig.canvas.draw()
            fig.canvas.flush_events()
    
    def add_to_filtered_plot_buffer(self, raw_data, filtered_data, timestamp):
        """Add data to the plotting buffers for each frequency band"""
        if not self.live_plot_active:
            return
        
        # Add to time buffer
        self.plot_time_buffer.append(timestamp)
        
        # Add raw data
        self.plot_data_buffers['raw'].append(raw_data)
        
        # Add filtered data for each band
        for band in filtered_data:
            if band in self.plot_data_buffers:
                # Handle both single and multi-channel data
                if isinstance(filtered_data[band], list):
                    # Multi-channel: store as numpy array for easier plotting
                    self.plot_data_buffers[band].append(np.array(filtered_data[band]))
                else:
                    # Single channel: store as is
                    self.plot_data_buffers[band].append(filtered_data[band])
        
        # Keep only recent data (last window_size + 5 seconds)
        current_time = time.time() - self.start_time
        cutoff_time = current_time - self.plot_window_size - 5
        
        # Remove old data from all buffers
        while self.plot_time_buffer and self.plot_time_buffer[0] < cutoff_time:
            self.plot_time_buffer.pop(0)
            for band in self.plot_data_buffers:
                if self.plot_data_buffers[band]:
                    self.plot_data_buffers[band].pop(0)
    
    def start_recording(self, duration=None):
        """Start recording EEG data with live filtering and accelerometer artifact rejection"""
        if not self.epoc:
            print("Headset not initialized!")
            return False

        # Set up signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)

        self.recording = True
        self.data_buffer = []
        for band in self.filtered_data_buffers:
            self.filtered_data_buffers[band] = []
        self.time_buffer = []
        self.start_time = time.time()
        self.accel_buffer = []

        print(f"\n{'='*80}")
        print("🎧 EEG Recording with Live Filtering Started!")
        print(f"{'='*80}")
        print("Recording EEG data from headset with real-time filtering...")
        print("Press Ctrl+C to stop recording and save data")

        if self.enable_live_filtering:
            print(f"✓ Live filtering enabled:")
            print(f"  - Delta: {self.live_filter.DELTA_LOW}-{self.live_filter.DELTA_HIGH} Hz")
            print(f"  - Theta: {self.live_filter.THETA_LOW}-{self.live_filter.THETA_HIGH} Hz") 
            print(f"  - Alpha: {self.live_filter.ALPHA_LOW}-{self.live_filter.ALPHA_HIGH} Hz")
            print(f"  - Beta: {self.live_filter.BETA_LOW}-{self.live_filter.BETA_HIGH} Hz")
            print(f"  - Gamma: {self.live_filter.GAMMA_LOW}-{self.live_filter.GAMMA_HIGH} Hz")

        if duration:
            print(f"Recording will automatically stop after {duration} seconds")

        # Get channel information
        channels = None
        if hasattr(self.epoc, 'channels') and getattr(self.epoc, 'channels', None):
            channels = getattr(self.epoc, 'channels', None)
            print(f"Channels: {channels}")

        print(f"Sampling rate: {self.sampling_rate} Hz")
        print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")

        # Setup live plotting if enabled
        if hasattr(self, 'enable_live_plot') and self.enable_live_plot:
            self.setup_live_plot_with_filters(channels)

        try:
            sample_count = 0
            while self.recording:
                try:
                    raw_data = self.epoc.get_sample()
                except Exception as e:
                    print(f"Error getting sample: {e}")
                    time.sleep(0.001)
                    continue

                # Get accelerometer data if available
                accel_data = getattr(self.epoc, 'get_accel', lambda: None)()
                self.accel_buffer.append(accel_data)

                # Artifact rejection based on accelerometer
                # If accel_data is available and above threshold, mark EEG as artifact (set to NaN)
                if raw_data:
                    if accel_data is not None:
                        try:
                            # If accel_data is a list/tuple/array of 3 values
                            if isinstance(accel_data, (list, tuple, np.ndarray)) and len(accel_data) == 3:
                                accel_magnitude = np.sqrt(sum([a**2 for a in accel_data]))
                                if accel_magnitude > 1.5:
                                    # Mark EEG sample as artifact (set to NaN)
                                    if isinstance(raw_data, (list, np.ndarray)):
                                        raw_data = [np.nan for _ in raw_data]
                                    else:
                                        raw_data = np.nan
                        except Exception as e:
                            print(f"Warning: Error computing accel magnitude: {e}")

                    current_time = time.time() - self.start_time

                    # Store raw data
                    self.data_buffer.append(raw_data)
                    self.time_buffer.append(current_time)
                    self.filtered_data_buffers['raw'].append(raw_data)

                    # Apply live filtering if enabled
                    filtered_results = {}
                    if self.enable_live_filtering:
                        # Handle single or multi-channel data
                        if isinstance(raw_data, (list, np.ndarray)) and hasattr(raw_data, '__len__'):
                            raw_list = raw_data.tolist() if hasattr(raw_data, 'tolist') else raw_data

                            # Process ALL channels
                            multi_channel_filtered = {}
                            for band in self.live_filter.filters.keys():
                                multi_channel_filtered[band] = []

                            # Process each channel separately
                            for ch_idx, raw_sample in enumerate(raw_list):
                                ch_filtered = self.live_filter.process_sample(raw_sample, channel=ch_idx)
                                for band, value in ch_filtered.items():
                                    multi_channel_filtered[band].append(value)

                            # Store multi-channel filtered results
                            filtered_results = multi_channel_filtered
                            for band, values in filtered_results.items():
                                if band in self.filtered_data_buffers:
                                    self.filtered_data_buffers[band].append(values)
                        else:
                            # Single channel data
                            raw_sample = raw_data
                            filtered_results = self.live_filter.process_sample(raw_sample, channel=0)
                            for band, value in filtered_results.items():
                                if band in self.filtered_data_buffers:
                                    self.filtered_data_buffers[band].append(value)

                    # Add to plot buffers
                    if self.live_plot_active:
                        self.add_to_filtered_plot_buffer(raw_data, filtered_results, current_time)

                        # Update plot every 25 samples
                        if sample_count % 25 == 0:
                            try:
                                self.update_filtered_plot()
                            except Exception as e:
                                print(f"Plot update failed: {e}")

                    sample_count += 1

                    # Print real-time EEG data with channels (same format as record_eeg_data.py)
                    if self.verbose_output:
                        self.print_eeg_data(raw_data, channels, sample_count, current_time)
                    else:
                        # Print progress every 128 samples (approximately 1 second for 128Hz)
                        if sample_count % 128 == 0:
                            elapsed = time.time() - self.start_time
                            print(f"Recording... {elapsed:.1f}s elapsed, {sample_count} samples collected")

                    # Check duration limit
                    if duration and current_time >= duration:
                        print(f"\nRecording duration ({duration}s) reached. Stopping...")
                        break

                # Small delay to prevent system overload
                time.sleep(0.001)

        except EPOCTurnedOffError:
            print("✗ Headset turned off during recording!")
            return False
        except EPOCUSBError as e:
            print(f"✗ USB error during recording: {e}")
            return False
        except KeyboardInterrupt:
            print("\nRecording interrupted by user")

        return True
    
    def stop_recording(self):
        """Stop recording and save filtered data and accelerometer values"""
        self.recording = False

        if not self.data_buffer:
            print("No data recorded!")
            return

        print(f"\n{'='*80}")
        print("📊 Recording Summary with Filtering")
        print(f"{'='*80}")
        print(f"Total samples: {len(self.data_buffer)}")
        print(f"Recording duration: {self.time_buffer[-1]:.2f} seconds")
        print(f"Raw data shape: {np.array(self.data_buffer).shape}")

        if self.enable_live_filtering:
            print(f"Filtered bands processed:")
            for band in self.filtered_data_buffers:
                if self.filtered_data_buffers[band]:
                    print(f"  - {band.capitalize()}: {len(self.filtered_data_buffers[band])} samples")

        # Save data to .mat file
        timestamp = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

        device_name = "eeg_filtered"
        filename = f"{device_name}-{timestamp}.mat"

        # Create comprehensive data structure
        channels = None
        if hasattr(self.epoc, 'channels') and getattr(self.epoc, 'channels', None):
            channels = getattr(self.epoc, 'channels', None)

        mat_data = {
            'data': {
                'raw': {
                    'trial': np.array(self.data_buffer).T,
                    'time': np.array(self.time_buffer),
                    'label': np.array(channels) if channels else np.array(['EEG']),
                    'fsample': self.sampling_rate,
                    'sampleinfo': np.array([1, len(self.time_buffer)])
                }
            },
            'date': timestamp,
            'sampling_rate': self.sampling_rate,
            'filter_settings': {
                'delta_band': f"{self.live_filter.DELTA_LOW}-{self.live_filter.DELTA_HIGH}",
                'theta_band': f"{self.live_filter.THETA_LOW}-{self.live_filter.THETA_HIGH}",
                'alpha_band': f"{self.live_filter.ALPHA_LOW}-{self.live_filter.ALPHA_HIGH}",
                'beta_band': f"{self.live_filter.BETA_LOW}-{self.live_filter.BETA_HIGH}",
                'gamma_band': f"{self.live_filter.GAMMA_LOW}-{self.live_filter.GAMMA_HIGH}",
                'filter_order': self.live_filter.filter_order,
                'artifact_threshold': self.live_filter.z_threshold
            }
        }

        # Add filtered data if available
        if self.enable_live_filtering:
            for band in self.filtered_data_buffers:
                if self.filtered_data_buffers[band] and band != 'raw':
                    mat_data['data'][f'{band}_filtered'] = {
                        'trial': np.array(self.filtered_data_buffers[band]).T,
                        'time': np.array(self.time_buffer),
                        'label': np.array(channels) if channels else np.array(['EEG']),
                        'fsample': self.sampling_rate,
                        'sampleinfo': np.array([1, len(self.time_buffer)])
                    }

        # Add accelerometer data if available
        if self.accel_buffer and any([x is not None for x in self.accel_buffer]):
            # Try to convert to numpy array, fill Nones with np.nan
            accel_arr = []
            for a in self.accel_buffer:
                if a is None:
                    accel_arr.append([np.nan, np.nan, np.nan])
                elif isinstance(a, (list, tuple, np.ndarray)) and len(a) == 3:
                    accel_arr.append(list(a))
                else:
                    # Unknown format, fill with nan
                    accel_arr.append([np.nan, np.nan, np.nan])
            accel_arr = np.array(accel_arr)
            mat_data['data']['accel'] = {
                'trial': accel_arr.T,
                'time': np.array(self.time_buffer),
                'label': np.array(['AccelX','AccelY','AccelZ']),
                'fsample': self.sampling_rate,
                'sampleinfo': np.array([1, len(self.time_buffer)])
            }

        # Add device info if available
        if self.selected_device:
            mat_data['device_info'] = {
                'manufacturer': self.selected_device['manufacturer'],
                'product': self.selected_device['product'],
                'serial': self.selected_device['serial'],
                'vendor_id': self.selected_device['vendor_id'],
                'product_id': self.selected_device['product_id']
            }

        try:
            scipy.io.savemat(filename, mat_data)
            print(f"✓ Filtered data saved to: {filename}")

            # Calculate file size
            num_channels = 1
            if hasattr(self.data_buffer[0], '__len__'):
                num_channels = len(self.data_buffer[0])

            total_samples = len(self.data_buffer) * (len(self.filtered_data_buffers) if self.enable_live_filtering else 1)
            file_size_kb = total_samples * num_channels * 8 / 1024
            print(f"✓ File size: {file_size_kb:.1f} KB")

            if self.enable_live_filtering:
                print(f"✓ Includes {len([b for b in self.filtered_data_buffers if self.filtered_data_buffers[b]])} frequency bands")

            # Save additional formats for easy access
            self.save_additional_formats(timestamp, channels)

        except Exception as e:
            print(f"✗ Failed to save data: {e}")
            return False

        return True
    
    def save_additional_formats(self, timestamp, channels):
        """Save data in multiple formats for easy local access"""
        import json
        import pandas as pd
        
        print(f"\n💾 Saving data in multiple formats for local access...")
        
        device_name = "eeg_filtered"
        if self.selected_device:
            device_name = f"{self.selected_device['manufacturer'].replace(' ', '_')}-{self.selected_device['product'].replace(' ', '_')}_filtered"
        
        try:
            # 1. Save as CSV (easiest for Excel, Python pandas)
            csv_filename = f"{device_name}-{timestamp}.csv"
            
            # Prepare data for CSV
            csv_data = {
                'timestamp': self.time_buffer,
                'raw_data': [str(d) for d in self.data_buffer]  # Convert to string for CSV
            }
            
            # Add filtered bands
            if self.enable_live_filtering:
                for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                    if band in self.filtered_data_buffers and self.filtered_data_buffers[band]:
                        csv_data[f'{band}_filtered'] = [str(d) for d in self.filtered_data_buffers[band]]
            
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_filename, index=False)
            print(f"✓ CSV data saved to: {csv_filename}")
            
        except Exception as e:
            print(f"Warning: Could not save CSV: {e}")
        
        try:
            # 2. Save as JSON (human-readable, easy for web apps)
            json_filename = f"{device_name}-{timestamp}.json"
            
            json_data = {
                'metadata': {
                    'recording_date': timestamp,
                    'sampling_rate': self.sampling_rate,
                    'duration_seconds': self.time_buffer[-1] if self.time_buffer else 0,
                    'total_samples': len(self.data_buffer),
                    'channels': channels if channels else ['EEG'],
                    'frequency_bands': {
                        'delta': '0.5-4 Hz',
                        'theta': '4-8 Hz', 
                        'alpha': '8-12 Hz',
                        'beta': '12-30 Hz',
                        'gamma': '30-50 Hz'
                    }
                },
                'raw_data': {
                    'timestamps': self.time_buffer,
                    'samples': self.data_buffer[:1000]  # First 1000 samples for JSON (size limit)
                }
            }
            
            # Add filtered data samples
            if self.enable_live_filtering:
                json_data['filtered_data'] = {}
                for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                    if band in self.filtered_data_buffers and self.filtered_data_buffers[band]:
                        json_data['filtered_data'][band] = self.filtered_data_buffers[band][:1000]  # First 1000 samples
            
            with open(json_filename, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            print(f"✓ JSON metadata saved to: {json_filename}")
            
        except Exception as e:
            print(f"Warning: Could not save JSON: {e}")
        
        try:
            # 3. Save as NumPy compressed format (best for Python analysis)
            npz_filename = f"{device_name}-{timestamp}.npz"
            
            npz_data = {
                'timestamps': np.array(self.time_buffer),
                'raw_data': np.array(self.data_buffer),
                'sampling_rate': self.sampling_rate
            }
            
            # Add filtered bands
            if self.enable_live_filtering:
                for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                    if band in self.filtered_data_buffers and self.filtered_data_buffers[band]:
                        npz_data[f'{band}_filtered'] = np.array(self.filtered_data_buffers[band])
            
            np.savez_compressed(npz_filename, **npz_data)
            print(f"✓ NumPy compressed data saved to: {npz_filename}")
            
        except Exception as e:
            print(f"Warning: Could not save NPZ: {e}")
        
        
        print(f"\n🎉 All data formats saved! You can now:")
        print(f"  - Open CSV in Excel/Google Sheets")
        print(f"  - Load NPZ in Python for analysis")
        print(f"  - View JSON for quick metadata")
        print(f"  - Use MATLAB .mat file for complete analysis")
    
    def disconnect(self):
        """Disconnect from the headset"""
        if self.epoc:
            try:
                self.epoc.disconnect()
                print("✓ Headset disconnected")
            except Exception as e:
                print(f"Note: Error during disconnect: {e}")
        
        # Clean up live plotting - close all separate figure windows
        if self.live_plot_active:
            if hasattr(self, 'figs') and self.figs:
                for band, fig in self.figs.items():
                    try:
                        plt.close(fig)
                    except Exception:
                        pass
                print(f"✓ Closed {len(self.figs)} live plotting windows")
            elif hasattr(self, 'fig') and self.fig:
                # Fallback for old single-figure setup
                plt.close(self.fig)
                print("✓ Live plotting window closed")

def check_system_compatibility():
    """Check system compatibility for GUI plotting"""
    import platform
    import sys
    
    print(f"\n{'='*60}")
    print("🔍 System Information")
    print(f"{'='*60}")
    
    # System info
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Matplotlib Backend: {matplotlib.get_backend()}")
    
    # Check if interactive plotting is available
    interactive_available = matplotlib.get_backend() not in ['Agg', 'PS', 'PDF', 'SVG']
    if interactive_available:
        print("✓ Interactive plotting is available")
        print("Live plotting should work!")
    else:
        print("⚠️ Non-interactive backend detected")
        print("Live plotting will be disabled, but FFT analysis will still work")
        print("\nTo enable live plotting, you can:")
        print("1. Install Qt: pip install PyQt5")
        print("2. Or install Tkinter: brew install python-tk")
        print("3. Or use a different Python installation with GUI support")
    
    print(f"{'='*60}")

# Run compatibility check at startup
check_system_compatibility()

def main():
    """Main function"""
    print("🎧 Universal EEG Data Recorder")
    print("="*50)
    print("Supports multiple EEG headsets including Emotiv, OpenBCI, Muse, and others")
    print("With real-time cognitive filtering for focus and attention analysis")
    
    recorder = EEGRecorderWithFilter()
    
    # Initialize headset
    if not recorder.setup_headset():
        return 1
    
    try:
        # Ask user for output verbosity
        print("\nOutput options:")
        print("1. Detailed real-time EEG data display")
        print("2. Simple progress display")
        
        verbose_choice = input("\nEnter choice (1 or 2): ").strip()
        recorder.verbose_output = (verbose_choice == "1")
        
        # Enable filtering by default but keep it in background
        recorder.enable_live_filtering = True
        recorder.active_filter_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']  # Plot all cognitive bands
        
        print("\n📊 EEG Frequency Bands Being Plotted:")
        print("Delta (0.5-4 Hz): Deep sleep, unconscious processes")
        print("Theta (4-8 Hz): Memory, creativity, REM sleep")
        print("Alpha (8-12 Hz): Relaxed awareness, attention state")
        print("Beta (12-30 Hz): Active thinking, cognitive processing")
        print("Low Gamma (30-50 Hz): High-level cognitive functions")
        print("\n💾 All data will be saved locally for analysis")
        
        # Check if interactive plotting is available
        interactive_available = matplotlib.get_backend() not in ['Agg', 'PS', 'PDF', 'SVG']
        
        if not interactive_available:
            print("\n⚠️ Live plotting is not available with the current matplotlib backend")
            print("Only FFT analysis will be available after recording")
            print("\nTo fix this, you can:")
            print("1. Install Tkinter: brew install python-tk")
            print("2. Or install Qt: brew install qt5")
            print("3. Or use a different Python installation with GUI support")
            recorder.enable_live_plot = False
        else:
            # Ask user for live plotting
            print("\nLive plotting options:")
            print("1. Enable live graphical display")
            print("2. Disable live plotting")
            
            plot_choice = input("\nEnter choice (1 or 2): ").strip()
            recorder.enable_live_plot = (plot_choice == "1")
        
        if recorder.enable_live_plot:
            print("✓ Live plotting will be enabled")
            print("Note: A new window will open showing real-time EEG data")
            print("You can stop the plot anytime by clicking the 'Stop Plot' button")
            print("On macOS: If the window doesn't appear, check your dock or use Cmd+Tab")
        else:
            print("✓ Live plotting disabled")
        
        # Ask user for recording duration
        print("\nRecording options:")
        print("1. Record until Ctrl+C (manual stop)")
        print("2. Record for specific duration")
        
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        duration = None
        if choice == "2":
            try:
                duration = float(input("Enter recording duration in seconds: "))
                if duration <= 0:
                    print("Invalid duration. Using manual stop.")
                    duration = None
            except ValueError:
                print("Invalid input. Using manual stop.")
                duration = None
        
        # Start recording
        if recorder.start_recording(duration):
            recorder.stop_recording()
        else:
            print("Recording failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\nRecording interrupted")
    finally:
        recorder.disconnect()
    
    print("\n🎉 Recording session complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
