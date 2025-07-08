#!/usr/bin/env python3
"""
Record EEG data from Emotiv EPOC headset and save to .mat file
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
        
    def setup_headset(self):
        """Initialize the Emotiv EPOC headset"""
        try:
            print("Initializing Emotiv EPOC headset...")
            self.epoc = EPOC()
            print("✓ Headset initialized successfully!")
            print(f"Channels: {self.epoc.channels}")
            print(f"Sampling rate: {self.epoc.sampling_rate} Hz")
            return True
        except Exception as e:
            print(f"✗ Failed to initialize headset: {e}")
            print("\nTroubleshooting tips:")
            print("1. Make sure the Emotiv EPOC headset is connected via USB")
            print("2. Make sure the headset is turned ON")
            print("3. Try running with sudo: sudo python record_eeg_data.py")
            print("4. Check if the headset is recognized: lsusb | grep Emotiv")
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
        print("Recording EEG data from Emotiv EPOC headset...")
        print("Press Ctrl+C to stop recording and save data")
        
        if duration:
            print(f"Recording will automatically stop after {duration} seconds")
        
        print(f"\nChannels: {self.epoc.channels}")
        print(f"Sampling rate: {self.epoc.sampling_rate} Hz")
        print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        try:
            sample_count = 0
            while self.recording:
                # Get EEG sample
                data = self.epoc.get_sample()
                
                if data:  # Skip battery packets (empty data)
                    current_time = time.time() - self.start_time
                    
                    # Store data
                    self.data_buffer.append(data)
                    self.time_buffer.append(current_time)
                    
                    sample_count += 1
                    
                    # Print progress every 128 samples (1 second)
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
        print(f"Channels: {self.epoc.channels}")
        
        # Save to .mat file
        timestamp = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        filename = f"emotiv-{timestamp}.mat"
        
        # Create MATLAB-compatible data structure
        mat_data = {
            'data': {
                'trial': eeg_data.T,  # Transpose to match expected format
                'time': time_data,
                'label': np.array(self.epoc.channels),
                'fsample': self.epoc.sampling_rate,
                'sampleinfo': np.array([1, len(time_data)])
            },
            'date': timestamp
        }
        
        try:
            scipy.io.savemat(filename, mat_data)
            print(f"✓ Data saved to: {filename}")
            print(f"✓ File size: {len(self.data_buffer) * len(self.epoc.channels) * 8 / 1024:.1f} KB")
        except Exception as e:
            print(f"✗ Failed to save data: {e}")
            return False
        
        return True
    
    def disconnect(self):
        """Disconnect from the headset"""
        if self.epoc:
            self.epoc.disconnect()
            print("✓ Headset disconnected")

def main():
    """Main function"""
    print("🎧 Emotiv EPOC EEG Data Recorder")
    print("="*50)
    
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