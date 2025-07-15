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
from emotiv.epoc import EPOC, EPOCTurnedOffError, EPOCUSBError

class EEGRecorder:
    def __init__(self):
        self.epoc = None
        self.recording = False
        self.data_buffer = []
        self.time_buffer = []
        self.start_time = None
        self.selected_device = None
        
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
        try:
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
            
        except Exception as e:
            print(f"✗ Failed to initialize headset: {e}")
            print("\nTroubleshooting tips:")
            print("1. Make sure the EEG headset is connected via USB")
            print("2. Make sure the headset is turned ON (if applicable)")
            print("3. Try running with sudo: sudo python record_eeg_data.py")
            print("4. Check if the headset is recognized: lsusb")
            print("5. Try different USB ports")
            return False
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C to stop recording gracefully"""
        print("\n\nStopping recording...")
        self.stop_recording()
        sys.exit(0)
    
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
                    
                    sample_count += 1
                    
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
        
        return True
    
    def disconnect(self):
        """Disconnect from the headset"""
        if self.epoc:
            try:
                self.epoc.disconnect()
                print("✓ Headset disconnected")
            except Exception as e:
                print(f"Note: Error during disconnect: {e}")

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