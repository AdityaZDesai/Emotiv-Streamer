#!/usr/bin/env python3
"""
Record EEG data from various EEG headsets and save to .mat file
"""

import scipy.io
import numpy as np
import time
import datetime
import signal
import sys
from scipy.fft import fft, fftfreq
import matplotlib
# Let matplotlib use its default backend - more reliable
print("✓ Using matplotlib default backend")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from emotiv.epoc import EPOC, EPOCTurnedOffError, EPOCUSBError

class EEGRecorder:
    def __init__(self):
        self.epoc = None
        self.recording = False
        self.data_buffer = []
        self.time_buffer = []
        self.start_time = None
        self.selected_device = None
        self.verbose_output = True  # Control real-time data display
        
        # Fourier analysis buffers
        self.fft_buffer = []  # Store FFT results
        self.fft_frequencies = []  # Store frequency arrays
        self.fft_timestamps = []  # Store timestamps for each FFT
        self.sampling_rate = 128  # Default sampling rate
        self.chunk_size = 128  # 1 second chunks (128 samples at 128Hz)
        self.current_chunk = []  # Current 1-second data chunk
        self.chunk_start_time = None
        
        # Live plotting variables
        self.live_plot_active = False
        self.plot_data_buffer = []  # Buffer for plotting (last N seconds)
        self.plot_time_buffer = []
        self.plot_window_size = 10  # Show last 10 seconds
        self.fig = None
        self.ax = None
        self.lines = []
        self.ani = None
        
    def list_devices(self):
        """List all available EEG devices"""
        print("🔍 Scanning for EEG devices...")
        try:
            devices = EPOC.list_all_devices()
            
            if not devices:
                print("No USB devices found!")
                return []
            
            # Filter and display EEG devices
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
                    # Auto-select first EEG device
                    eeg_devices = [d for d in devices if d['is_eeg']]
                    if eeg_devices:
                        self.selected_device = eeg_devices[0]
                        print(f"✓ Auto-selected: {self.selected_device['manufacturer']} - {self.selected_device['product']}")
                        return self.selected_device
                    else:
                        print("No EEG devices found for auto-selection.")
                        continue
                        
                elif choice == 2:
                    # Manual selection
                    self.selected_device = EPOC.select_device()
                    if self.selected_device:
                        print(f"✓ Selected: {self.selected_device['manufacturer']} - {self.selected_device['product']}")
                        return self.selected_device
                    else:
                        print("No device selected.")
                        return None
                        
                elif choice == 3:
                    # Use default Emotiv detection
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

        # Device selection
        device_choice = self.select_device()
        
        if device_choice is None:
            print("No device selected. Exiting.")
            return False
        
        print(f"\n{'='*60}")
        print("🎧 Initializing EEG Headset...")
        print(f"{'='*60}")
        
        if device_choice == "default":
            # Use default Emotiv detection
            print("Using default Emotiv device detection...")
            self.epoc = EPOC()
        else:
            # Use selected device
            print(f"Connecting to: {device_choice['manufacturer']} - {device_choice['product']}")
            self.epoc = EPOC(device_info=device_choice)
        
        print("✓ Headset initialized successfully!")
        
        # Display device information
        if hasattr(self.epoc, 'channels') and getattr(self.epoc, 'channels', None):
            print(f"Channels: {getattr(self.epoc, 'channels', [])}")
        if hasattr(self.epoc, 'sampling_rate') and getattr(self.epoc, 'sampling_rate', None):
            print(f"Sampling rate: {getattr(self.epoc, 'sampling_rate', 'Unknown')} Hz")
        
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
        """Print formatted EEG data with channel information"""
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
    
    def process_chunk_fft(self, chunk_data, timestamp):
        """Process a 1-second chunk of data with FFT"""
        if len(chunk_data) < self.chunk_size:
            return None  # Not enough data for a complete chunk
            
        # Convert to numpy array if it's not already
        chunk_array = np.array(chunk_data)
        
        # Handle multi-channel data
        if len(chunk_array.shape) > 1:
            # Multi-channel data: shape is (samples, channels)
            num_channels = chunk_array.shape[1]
            fft_results = []
            
            for ch in range(num_channels):
                channel_data = chunk_array[:, ch]
                
                # Apply window function to reduce spectral leakage
                window = np.hanning(len(channel_data))
                windowed_data = channel_data * window
                
                # Apply FFT
                fft_result = fft(windowed_data)
                
                # Get positive frequencies only (up to Nyquist frequency)
                positive_fft = np.abs(fft_result[:len(fft_result)//2])
                fft_results.append(positive_fft)
            
            # Calculate frequencies (same for all channels)
            freqs = fftfreq(len(chunk_array), 1/self.sampling_rate)
            positive_freqs = freqs[:len(freqs)//2]
            
            # Store results
            self.fft_buffer.append(np.array(fft_results))
            self.fft_frequencies.append(positive_freqs)
            self.fft_timestamps.append(timestamp)
            
            return np.array(fft_results), positive_freqs
            
        else:
            # Single channel data
            # Apply window function to reduce spectral leakage
            window = np.hanning(len(chunk_array))
            windowed_data = chunk_array * window
            
            # Apply FFT
            fft_result = fft(windowed_data)
            
            # Calculate frequencies
            freqs = fftfreq(len(chunk_array), 1/self.sampling_rate)
            
            # Get positive frequencies only (up to Nyquist frequency)
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft_result[:len(freqs)//2])
            
            # Store results
            self.fft_buffer.append(positive_fft)
            self.fft_frequencies.append(positive_freqs)
            self.fft_timestamps.append(timestamp)
            
            return positive_fft, positive_freqs
    
    def add_to_chunk(self, data, timestamp):
        """Add data point to current chunk and process if chunk is complete"""
        if self.chunk_start_time is None:
            self.chunk_start_time = timestamp
            
        self.current_chunk.append(data)
        
        # Check if we have a complete 1-second chunk
        if len(self.current_chunk) >= self.chunk_size:
            # Process the complete chunk
            fft_result = self.process_chunk_fft(self.current_chunk, self.chunk_start_time)
            
            if fft_result:
                fft_data, freqs = fft_result
                if self.verbose_output:
                    print(f"\n🔬 FFT Analysis - Chunk at {self.chunk_start_time:.1f}s")
                    # Handle multi-channel data
                    if len(fft_data.shape) > 1:
                        for ch_idx in range(fft_data.shape[0]):
                            peak_freq = freqs[np.argmax(fft_data[ch_idx])]
                            max_amp = np.max(fft_data[ch_idx])
                            print(f"  Channel {ch_idx+1}: Peak at {peak_freq:.1f} Hz, Amp: {max_amp:.2f}")
                    else:
                        peak_freq = freqs[np.argmax(fft_data)]
                        max_amp = np.max(fft_data)
                        print(f"  Peak frequency: {peak_freq:.1f} Hz, Max amplitude: {max_amp:.2f}")
            
            # Reset for next chunk
            self.current_chunk = []
            self.chunk_start_time = timestamp
    
    def visualize_fft_results(self):
        """Visualize the FFT results"""
        if not self.fft_buffer:
            print("No FFT data to visualize!")
            return
        
        # Check if we can create plots
        if matplotlib.get_backend() in ['Agg', 'PS', 'PDF', 'SVG']:
            print(f"\n{'='*80}")
            print("📊 FFT Analysis Results (Text Only)")
            print(f"{'='*80}")
            print(f"Total FFT chunks: {len(self.fft_buffer)}")
            print(f"Time span: {self.fft_timestamps[0]:.1f}s to {self.fft_timestamps[-1]:.1f}s")
            print("\n⚠️ Interactive plotting not available. Saving data for later analysis.")
            self.save_fft_data()
            return
            
        print(f"\n{'='*80}")
        print("📊 FFT Analysis Results")
        print(f"{'='*80}")
        print(f"Total FFT chunks: {len(self.fft_buffer)}")
        print(f"Time span: {self.fft_timestamps[0]:.1f}s to {self.fft_timestamps[-1]:.1f}s")
        
        try:
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Real-time EEG FFT Analysis', fontsize=16)
            
            # Check if we have multi-channel data
            is_multi_channel = len(self.fft_buffer[0].shape) > 1
            
            if is_multi_channel:
                num_channels = self.fft_buffer[0].shape[0]
                print(f"Multi-channel data detected: {num_channels} channels")
                
                # 1. Average power spectrum for each channel
                avg_freqs = self.fft_frequencies[0]  # All frequency arrays should be the same
                for ch in range(num_channels):
                    channel_avg = np.mean([fft_data[ch] for fft_data in self.fft_buffer], axis=0)
                    axes[0, 0].plot(avg_freqs, channel_avg, label=f'Channel {ch+1}')
                
                axes[0, 0].set_title('Average Power Spectrum (All Channels)')
                axes[0, 0].set_xlabel('Frequency (Hz)')
                axes[0, 0].set_ylabel('Amplitude')
                axes[0, 0].grid(True)
                axes[0, 0].set_xlim(0, 50)  # Focus on relevant EEG frequencies
                axes[0, 0].legend()
                
                # 2. Time-frequency heatmap for first channel
                fft_matrix = np.array([fft_data[0] for fft_data in self.fft_buffer])  # Use first channel
                im = axes[0, 1].imshow(fft_matrix.T, aspect='auto', origin='lower', 
                                      extent=[self.fft_timestamps[0], self.fft_timestamps[-1], 
                                             avg_freqs[0], avg_freqs[-1]])
                axes[0, 1].set_title('Time-Frequency Heatmap (Channel 1)')
                axes[0, 1].set_xlabel('Time (s)')
                axes[0, 1].set_ylabel('Frequency (Hz)')
                plt.colorbar(im, ax=axes[0, 1], label='Amplitude')
                
                # 3. Peak frequency over time for each channel
                for ch in range(min(num_channels, 3)):  # Show first 3 channels
                    peak_freqs = [freqs[np.argmax(fft_data[ch])] for fft_data, freqs in zip(self.fft_buffer, self.fft_frequencies)]
                    axes[1, 0].plot(self.fft_timestamps, peak_freqs, 'o-', label=f'Channel {ch+1}')
                
                axes[1, 0].set_title('Peak Frequency Over Time')
                axes[1, 0].set_xlabel('Time (s)')
                axes[1, 0].set_ylabel('Peak Frequency (Hz)')
                axes[1, 0].grid(True)
                axes[1, 0].legend()
                
                # 4. Total power over time for each channel
                for ch in range(min(num_channels, 3)):  # Show first 3 channels
                    total_power = [np.sum(fft_data[ch]) for fft_data in self.fft_buffer]
                    axes[1, 1].plot(self.fft_timestamps, total_power, 'o-', label=f'Channel {ch+1}')
                
                axes[1, 1].set_title('Total Power Over Time')
                axes[1, 1].set_xlabel('Time (s)')
                axes[1, 1].set_ylabel('Total Power')
                axes[1, 1].grid(True)
                axes[1, 1].legend()
                
            else:
                # Single channel data
                # 1. Average power spectrum
                avg_fft = np.mean(self.fft_buffer, axis=0)
                avg_freqs = self.fft_frequencies[0]  # All frequency arrays should be the same
                
                axes[0, 0].plot(avg_freqs, avg_fft)
                axes[0, 0].set_title('Average Power Spectrum')
                axes[0, 0].set_xlabel('Frequency (Hz)')
                axes[0, 0].set_ylabel('Amplitude')
                axes[0, 0].grid(True)
                axes[0, 0].set_xlim(0, 50)  # Focus on relevant EEG frequencies
                
                # 2. Time-frequency heatmap
                fft_matrix = np.array(self.fft_buffer)
                im = axes[0, 1].imshow(fft_matrix.T, aspect='auto', origin='lower', 
                                      extent=[self.fft_timestamps[0], self.fft_timestamps[-1], 
                                             avg_freqs[0], avg_freqs[-1]])
                axes[0, 1].set_title('Time-Frequency Heatmap')
                axes[0, 1].set_xlabel('Time (s)')
                axes[0, 1].set_ylabel('Frequency (Hz)')
                plt.colorbar(im, ax=axes[0, 1], label='Amplitude')
                
                # 3. Peak frequency over time
                peak_freqs = [freqs[np.argmax(fft_data)] for fft_data, freqs in zip(self.fft_buffer, self.fft_frequencies)]
                axes[1, 0].plot(self.fft_timestamps, peak_freqs, 'o-')
                axes[1, 0].set_title('Peak Frequency Over Time')
                axes[1, 0].set_xlabel('Time (s)')
                axes[1, 0].set_ylabel('Peak Frequency (Hz)')
                axes[1, 0].grid(True)
                
                # 4. Total power over time
                total_power = [np.sum(fft_data) for fft_data in self.fft_buffer]
                axes[1, 1].plot(self.fft_timestamps, total_power, 'o-')
                axes[1, 1].set_title('Total Power Over Time')
                axes[1, 1].set_xlabel('Time (s)')
                axes[1, 1].set_ylabel('Total Power')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Save the plot
            timestamp = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
            plot_filename = f"fft_analysis_{timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"✓ FFT visualization saved as: {plot_filename}")
            
            # Show the plot
            plt.show()
            
            # Print summary statistics
            print(f"\n📈 FFT Summary Statistics:")
            if is_multi_channel:
                for ch in range(num_channels):
                    ch_peak_freqs = [freqs[np.argmax(fft_data[ch])] for fft_data, freqs in zip(self.fft_buffer, self.fft_frequencies)]
                    ch_total_power = [np.sum(fft_data[ch]) for fft_data in self.fft_buffer]
                    print(f"Channel {ch+1}:")
                    print(f"  Average peak frequency: {np.mean(ch_peak_freqs):.2f} Hz")
                    print(f"  Peak frequency range: {np.min(ch_peak_freqs):.2f} - {np.max(ch_peak_freqs):.2f} Hz")
                    print(f"  Average total power: {np.mean(ch_total_power):.2f}")
                    print(f"  Power range: {np.min(ch_total_power):.2f} - {np.max(ch_total_power):.2f}")
            else:
                peak_freqs = [freqs[np.argmax(fft_data)] for fft_data, freqs in zip(self.fft_buffer, self.fft_frequencies)]
                total_power = [np.sum(fft_data) for fft_data in self.fft_buffer]
                print(f"Average peak frequency: {np.mean(peak_freqs):.2f} Hz")
                print(f"Peak frequency range: {np.min(peak_freqs):.2f} - {np.max(peak_freqs):.2f} Hz")
                print(f"Average total power: {np.mean(total_power):.2f}")
                print(f"Power range: {np.min(total_power):.2f} - {np.max(total_power):.2f}")
                
        except Exception as e:
            print(f"Error creating plot: {e}")
            print("Saving FFT data for later analysis...")
            self.save_fft_data()
            return
    
    def save_fft_data(self):
        """Save the FFT data to a .mat file"""
        if not self.fft_buffer:
            print("No FFT data to save.")
            return
        
        timestamp = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        filename = f"fft_data_{timestamp}.mat"
        
        try:
            scipy.io.savemat(filename, {'fft_data': self.fft_buffer,
                                       'frequencies': self.fft_frequencies,
                                       'timestamps': self.fft_timestamps})
            print(f"✓ FFT data saved to: {filename}")
        except Exception as e:
            print(f"✗ Failed to save FFT data: {e}")
    
    def setup_live_plot(self, channels):
        """Setup the live plotting window"""
        try:
            plt.ion()  # Turn on interactive mode
            
            # Create figure and axis
            self.fig, self.ax = plt.subplots(figsize=(15, 8))
            self.fig.suptitle('Live EEG Data Stream', fontsize=16)
            
            # Setup plot elements
            self._setup_plot_elements(channels)
            
            # Show the plot and keep it open
            plt.show(block=False)
            plt.pause(0.1)  # Small pause to ensure window appears
            
            # Add some test data to verify plot is working
            test_times = np.linspace(0, 5, 50)
            test_data = np.sin(test_times * 2 * np.pi * 2) * 50  # 2Hz sine wave
            self.plot_data_buffer = test_data.tolist()
            self.plot_time_buffer = test_times.tolist()
            
            # Ensure the figure is properly displayed
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            self.live_plot_active = True
            print("✓ Live plotting window opened")
            print("Note: If you don't see the window, check your dock or use Cmd+Tab to switch to it")
            
        except Exception as e:
            print(f"Warning: Could not setup live plotting window: {e}")
            print("Live plotting will be disabled.")
            self.live_plot_active = False
    
    def _setup_plot_elements(self, channels):
        """Setup plot elements"""
        # Get channel names
        if channels:
            channel_names = channels
        else:
            num_channels = 1  # Will be updated when we get first data
            channel_names = ['EEG']
        
        # Create line objects for each channel
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        self.lines = []
        
        for i, channel in enumerate(channel_names):
            color = colors[i % len(colors)]
            line, = self.ax.plot([], [], color=color, linewidth=1, label=channel, alpha=0.8)
            self.lines.append(line)
        
        # Setup the plot
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Amplitude (μV)')
        self.ax.set_title('Real-time EEG Signals')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # Add control buttons
        ax_stop = plt.axes([0.7, 0.02, 0.1, 0.04])
        self.stop_button = Button(ax_stop, 'Stop Plot')
        self.stop_button.on_clicked(self.stop_live_plot)
        
        ax_save = plt.axes([0.82, 0.02, 0.1, 0.04])
        self.save_button = Button(ax_save, 'Save Plot')
        self.save_button.on_clicked(lambda event: self.save_live_plot())
        
        # Add status text
        self.status_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                       verticalalignment='top', fontsize=10,
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Initialize plot limits
        self.ax.set_xlim(0, self.plot_window_size)
        self.ax.set_ylim(-100, 100)  # Will be auto-adjusted
        
        plt.tight_layout()
        
        # Force the window to show and update
        plt.show(block=False)
        plt.pause(0.1)  # Small pause to ensure window appears
        
        # Additional macOS-specific window management
        try:
            if hasattr(self.fig.canvas.manager, 'window'):
                self.fig.canvas.manager.window.raise_()
                self.fig.canvas.manager.window.focus_force()
        except:
            pass
        
        self.live_plot_active = True
        print("✓ Live plotting window opened")
        print("Note: If you don't see the window, check your dock or use Cmd+Tab to switch to it")
        
        # Test if window is actually visible
        try:
            plt.pause(0.5)  # Give time for window to appear
            if hasattr(self.fig.canvas.manager, 'window'):
                window_visible = self.fig.canvas.manager.window.winfo_exists()
                if window_visible:
                    print("✓ Plot window is visible and active")
                else:
                    print("⚠️ Plot window may not be visible")
            else:
                print("✓ Plot window created successfully")
        except:
            print("✓ Plot window created (visibility check skipped)")
        
        print("Note: If you don't see the window, check your dock or use Cmd+Tab to switch to it")
        
        # Provide additional help for macOS users
        import platform
        if platform.system() == 'Darwin':  # macOS
            print("\n📱 macOS Tips:")
            print("- Check the dock for a Python/Matplotlib icon")
            print("- Use Cmd+Tab to cycle through open applications")
            print("- Look for the window in Mission Control (F3)")
            print("- If still not visible, try clicking on the Python icon in the dock")
            print("- You can also try: Cmd+Space and search for 'Python'")
    
    def stop_live_plot(self, event=None):
        """Stop the live plotting"""
        self.live_plot_active = False
        if self.ani:
            self.ani.event_source.stop()
        print("Live plotting stopped")
    
    def update_live_plot(self, frame):
        """Update function for the live plot animation"""
        if not self.live_plot_active or not self.plot_data_buffer:
            return self.lines
        
        # Debug: Show animation is running
        if frame % 10 == 0:  # Every 10 frames (1 second)
            print(f"Animation frame: {frame}, Buffer size: {len(self.plot_data_buffer)}")
        
        # Get current time window
        current_time = time.time() - self.start_time
        time_window_start = max(0, current_time - self.plot_window_size)
        
        # Filter data within the time window
        valid_indices = [i for i, t in enumerate(self.plot_time_buffer) 
                        if time_window_start <= t <= current_time]
        
        if not valid_indices:
            return self.lines
        
        # Get data within window
        window_times = [self.plot_time_buffer[i] for i in valid_indices]
        window_data = [self.plot_data_buffer[i] for i in valid_indices]
        
        # Convert to numpy arrays
        window_times = np.array(window_times)
        window_data = np.array(window_data)
        
        # Debug: Print data info
        if frame % 50 == 0:  # Print every 50 frames (5 seconds)
            print(f"Debug: {len(window_data)} samples, time range: {window_times[0]:.1f}s - {window_times[-1]:.1f}s")
            print(f"Debug: Data shape: {window_data.shape}, Data range: {np.min(window_data):.2f} - {np.max(window_data):.2f}")
        
        # Clear the plot and redraw
        self.ax.clear()
        
        # Handle multi-channel data
        if len(window_data.shape) > 1:
            num_channels = window_data.shape[1]
            
            # Plot each channel
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            for ch in range(num_channels):
                color = colors[ch % len(colors)]
                self.ax.plot(window_times, window_data[:, ch], color=color, linewidth=1, 
                           label=f'Channel {ch+1}', alpha=0.8)
        else:
            # Single channel data
            self.ax.plot(window_times, window_data, color='blue', linewidth=1, alpha=0.8)
        
        # Setup the plot
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Amplitude (μV)')
        self.ax.set_title(f'Real-time EEG Signals - {current_time:.1f}s')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # Update plot limits
        self.ax.set_xlim(time_window_start, current_time)
        
        # Auto-adjust y-axis based on data range
        if len(window_data.shape) > 1:
            y_min = np.min(window_data)
            y_max = np.max(window_data)
        else:
            y_min = np.min(window_data)
            y_max = np.max(window_data)
        
        y_range = y_max - y_min
        if y_range > 0:
            y_margin = y_range * 0.1
            self.ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # Update status text
        if hasattr(self, 'status_text'):
            num_samples = len(self.plot_data_buffer)
            if len(window_data.shape) > 1:
                num_channels = window_data.shape[1]
                status = f'Samples: {num_samples} | Channels: {num_channels} | Window: {self.plot_window_size}s'
            else:
                status = f'Samples: {num_samples} | Window: {self.plot_window_size}s'
            self.status_text.set_text(status)
        
        # Force redraw
        self.fig.canvas.draw()
        
        return self.lines
    
    def add_to_plot_buffer(self, data, timestamp):
        """Add data to the plotting buffer"""
        if not self.live_plot_active:
            return
        
        self.plot_data_buffer.append(data)
        self.plot_time_buffer.append(timestamp)
        
        # Debug: Print buffer info occasionally
        if len(self.plot_data_buffer) % 50 == 0:  # Every 50 samples
            print(f"Plot buffer: {len(self.plot_data_buffer)} samples, latest data: {data}")
        
        # Keep only the last N seconds of data
        current_time = time.time() - self.start_time
        cutoff_time = current_time - self.plot_window_size - 5  # Keep extra 5 seconds
        
        # Remove old data
        while self.plot_time_buffer and self.plot_time_buffer[0] < cutoff_time:
            self.plot_time_buffer.pop(0)
            self.plot_data_buffer.pop(0)
    
    def save_live_plot(self):
        """Save the current live plot as an image"""
        if not self.live_plot_active or not self.fig:
            return
        
        timestamp = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        plot_filename = f"live_eeg_plot_{timestamp}.png"
        
        try:
            self.fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"✓ Live plot saved as: {plot_filename}")
        except Exception as e:
            print(f"✗ Failed to save live plot: {e}")
    
    def simple_plot_update(self):
        """Simple plot update without animation framework"""
        if not self.live_plot_active or not self.plot_data_buffer:
            return
        
        # Get current time window
        current_time = time.time() - self.start_time
        time_window_start = max(0, current_time - self.plot_window_size)
        
        # Filter data within the time window
        valid_indices = [i for i, t in enumerate(self.plot_time_buffer) 
                        if time_window_start <= t <= current_time]
        
        if not valid_indices:
            return
        
        # Get data within window
        window_times = [self.plot_time_buffer[i] for i in valid_indices]
        window_data = [self.plot_data_buffer[i] for i in valid_indices]
        
        # Convert to numpy arrays
        window_times = np.array(window_times)
        window_data = np.array(window_data)
        
        # Clear the plot and redraw
        self.ax.clear()
        
        # Handle multi-channel data
        if len(window_data.shape) > 1:
            num_channels = window_data.shape[1]
            
            # Plot each channel
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            for ch in range(num_channels):
                color = colors[ch % len(colors)]
                self.ax.plot(window_times, window_data[:, ch], color=color, linewidth=1, 
                           label=f'Channel {ch+1}', alpha=0.8)
        else:
            # Single channel data
            self.ax.plot(window_times, window_data, color='blue', linewidth=1, alpha=0.8)
        
        # Setup the plot
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Amplitude (μV)')
        self.ax.set_title(f'Real-time EEG Signals - {current_time:.1f}s')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # Update plot limits
        self.ax.set_xlim(time_window_start, current_time)
        
        # Auto-adjust y-axis based on data range
        if len(window_data.shape) > 1:
            y_min = np.min(window_data)
            y_max = np.max(window_data)
        else:
            y_min = np.min(window_data)
            y_max = np.max(window_data)
        
        y_range = y_max - y_min
        if y_range > 0:
            y_margin = y_range * 0.1
            self.ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # Force redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        print(f"Plot updated: {len(window_data)} samples, time: {current_time:.1f}s")
    
    def keep_plot_alive(self):
        """Keep the plot window alive and handle events"""
        if self.live_plot_active and self.fig:
            try:
                # Process any pending matplotlib events
                self.fig.canvas.flush_events()
                plt.pause(0.001)  # Small pause to allow event processing
            except Exception as e:
                print(f"Plot event handling error: {e}")
    
    def start_recording(self, duration=None):
        """Start recording EEG data"""
        if not self.epoc:
            print("Headset not initialized!")
            return False
        
        # Set up signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)
        
        self.recording = True
        self.data_buffer = []
        self.time_buffer = []
        self.start_time = time.time()
        
        print(f"\n{'='*60}")
        print("🎧 EEG Recording Started!")
        print(f"{'='*60}")
        print("Recording EEG data from headset...")
        print("Press Ctrl+C to stop recording and save data")
        
        if duration:
            print(f"Recording will automatically stop after {duration} seconds")
        
        # Display channel info if available
        if hasattr(self.epoc, 'channels') and getattr(self.epoc, 'channels', None):
            print(f"\nChannels: {getattr(self.epoc, 'channels', [])}")
        if hasattr(self.epoc, 'sampling_rate') and getattr(self.epoc, 'sampling_rate', None):
            print(f"Sampling rate: {getattr(self.epoc, 'sampling_rate', 'Unknown')} Hz")
        
        print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # Get channel information and sampling rate
        channels = None
        if hasattr(self.epoc, 'channels') and getattr(self.epoc, 'channels', None):
            channels = getattr(self.epoc, 'channels', None)
        
        # Get sampling rate and update chunk size
        if hasattr(self.epoc, 'sampling_rate') and getattr(self.epoc, 'sampling_rate', None):
            self.sampling_rate = getattr(self.epoc, 'sampling_rate', 128)
            self.chunk_size = self.sampling_rate  # 1 second chunks
            print(f"✓ Using sampling rate: {self.sampling_rate} Hz")
            print(f"✓ Chunk size: {self.chunk_size} samples (1 second)")
        else:
            print(f"⚠️ Using default sampling rate: {self.sampling_rate} Hz")
        
        # Setup live plotting if enabled
        if hasattr(self, 'enable_live_plot') and self.enable_live_plot:
            self.setup_live_plot(channels)
            if self.live_plot_active:
                # Use manual updates instead of FuncAnimation to avoid deletion issues
                print("✓ Live plotting setup complete")
                print("Note: Plot will update every 50 samples")
            else:
                print("⚠️ Live plotting setup failed, continuing without live plot")
        
        try:
            sample_count = 0
            while self.recording:
                # Get EEG sample
                try:
                    data = self.epoc.get_sample()
                except Exception as e:
                    print(f"Error getting sample: {e}")
                    # For non-Emotiv devices, we might need different handling
                    time.sleep(0.001)
                    continue
                
                if data:  # Skip battery packets (empty data)
                    current_time = time.time() - self.start_time
                    
                    # Store data
                    self.data_buffer.append(data)
                    self.time_buffer.append(current_time)
                    
                    # Add to FFT chunk processing
                    self.add_to_chunk(data, current_time)
                    
                    # Add to live plot buffer
                    self.add_to_plot_buffer(data, current_time)
                    
                    # Manual plot update every 50 samples (backup to animation)
                    if self.live_plot_active and sample_count % 50 == 0:
                        try:
                            self.simple_plot_update()
                        except Exception as e:
                            print(f"Manual plot update failed: {e}")
                    
                    # Keep plot window alive
                    if self.live_plot_active and sample_count % 10 == 0:
                        self.keep_plot_alive()
                    
                    sample_count += 1
                    
                    # Print real-time EEG data with channels
                    if self.verbose_output:
                        self.print_eeg_data(data, channels, sample_count, current_time)
                    else:
                        # Print progress every 128 samples (approximately 1 second for 128Hz)
                        if sample_count % 128 == 0:
                            elapsed = time.time() - self.start_time
                            print(f"Recording... {elapsed:.1f}s elapsed, {sample_count} samples collected")
                    
                    # Check if duration limit reached
                    if duration and current_time >= duration:
                        print(f"\nRecording duration ({duration}s) reached. Stopping...")
                        break
                
                # Small delay to prevent overwhelming the system
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
        """Stop recording and save data"""
        self.recording = False
        
        if not self.data_buffer:
            print("No data recorded!")
            return
        
        # Convert to numpy arrays
        eeg_data = np.array(self.data_buffer)
        time_data = np.array(self.time_buffer)
        
        print(f"\n{'='*60}")
        print("📊 Recording Summary")
        print(f"{'='*60}")
        print(f"Total samples: {len(self.data_buffer)}")
        print(f"Recording duration: {time_data[-1]:.2f} seconds")
        print(f"Data shape: {eeg_data.shape}")
        print(f"FFT chunks processed: {len(self.fft_buffer)}")
        
        # Get channel info if available
        channels = None
        sampling_rate = 128  # Default
        
        if hasattr(self.epoc, 'channels') and getattr(self.epoc, 'channels', None):
            channels = getattr(self.epoc, 'channels', None)
            print(f"Channels: {channels}")
        
        if hasattr(self.epoc, 'sampling_rate') and getattr(self.epoc, 'sampling_rate', None):
            sampling_rate = getattr(self.epoc, 'sampling_rate', 128)
        
        # Save to .mat file
        timestamp = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        
        # Include device info in filename if available
        device_name = "eeg"
        if self.selected_device:
            device_name = f"{self.selected_device['manufacturer'].replace(' ', '_')}-{self.selected_device['product'].replace(' ', '_')}"
        
        filename = f"{device_name}-{timestamp}.mat"
        
        # Create MATLAB-compatible data structure
        mat_data = {
            'data': {
                'trial': eeg_data.T,  # Transpose to match expected format
                'time': time_data,
                'label': np.array(channels) if channels else np.array([f"Ch{i+1}" for i in range(eeg_data.shape[1])]),
                'fsample': sampling_rate,
                'sampleinfo': np.array([1, len(time_data)])
            },
            'date': timestamp
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
            print(f"✓ Data saved to: {filename}")
            
            # Calculate file size
            num_channels = eeg_data.shape[1] if len(eeg_data.shape) > 1 else 1
            file_size_kb = len(self.data_buffer) * num_channels * 8 / 1024
            print(f"✓ File size: {file_size_kb:.1f} KB")
            
        except Exception as e:
            print(f"✗ Failed to save data: {e}")
            return False
        
        # Save live plot if active
        if self.live_plot_active:
            print(f"\n{'='*60}")
            print("📊 Saving Live Plot...")
            print(f"{'='*60}")
            self.save_live_plot()
        
        # Visualize FFT results
        if self.fft_buffer:
            print(f"\n{'='*60}")
            print("🔬 Processing FFT Analysis...")
            print(f"{'='*60}")
            self.visualize_fft_results()
        
        return True
    
    def disconnect(self):
        """Disconnect from the headset"""
        if self.epoc:
            try:
                self.epoc.disconnect()
                print("✓ Headset disconnected")
            except Exception as e:
                print(f"Note: Error during disconnect: {e}")
        
        # Clean up live plotting
        if self.live_plot_active:
            self.stop_live_plot()
            if self.fig:
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
    
    recorder = EEGRecorder()
    
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