#!/usr/bin/env python3
"""
Test script to check if Emotiv EPOC headset is working
"""

import sys
import time
from emotiv.epoc import EPOC, EPOCTurnedOffError, EPOCUSBError

def test_headset():
    """Test the Emotiv EPOC headset"""
    print("🎧 Testing Emotiv EPOC Headset")
    print("="*40)
    
    try:
        # Initialize headset
        print("1. Initializing headset...")
        epoc = EPOC()
        print("✓ Headset initialized successfully!")
        
        # Print headset info
        print(f"2. Headset information:")
        print(f"   Channels: {epoc.channels}")
        print(f"   Sampling rate: {epoc.sampling_rate} Hz")
        print(f"   Battery level: {epoc.battery}%")
        
        # Test data acquisition
        print("3. Testing data acquisition...")
        print("   Collecting 10 samples...")
        
        samples = []
        for i in range(10):
            try:
                data = epoc.get_sample()
                if data:
                    samples.append(data)
                    print(f"   Sample {i+1}: {[f'{x:.2f}' for x in data[:3]]}... (first 3 channels)")
                else:
                    print(f"   Sample {i+1}: Battery packet (no EEG data)")
                time.sleep(0.1)  # Small delay
            except EPOCTurnedOffError:
                print("✗ Headset turned off!")
                return False
            except EPOCUSBError as e:
                print(f"✗ USB error: {e}")
                return False
        
        if samples:
            print(f"✓ Successfully collected {len(samples)} EEG samples!")
            print(f"   Data shape: {len(samples)} samples × {len(samples[0])} channels")
            
            # Show sample statistics
            data_array = np.array(samples)
            print(f"   Mean amplitude: {np.mean(data_array):.2f} μV")
            print(f"   Amplitude range: {np.min(data_array):.2f} to {np.max(data_array):.2f} μV")
        else:
            print("✗ No EEG data collected!")
            return False
        
        # Test contact quality
        print("4. Testing contact quality...")
        for channel in epoc.channels[:5]:  # Show first 5 channels
            quality = epoc.get_quality(channel)
            if quality is not None:
                print(f"   {channel}: {quality:.3f}")
            else:
                print(f"   {channel}: No quality data")
        
        # Disconnect
        print("5. Disconnecting...")
        epoc.disconnect()
        print("✓ Headset disconnected successfully!")
        
        print("\n🎉 All tests passed! Headset is working correctly.")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure the Emotiv EPOC headset is connected via USB")
        print("2. Make sure the headset is turned ON")
        print("3. Try running with sudo: sudo python test_headset.py")
        print("4. Check if the headset is recognized: lsusb | grep Emotiv")
        return False

if __name__ == "__main__":
    import numpy as np
    success = test_headset()
    sys.exit(0 if success else 1) 