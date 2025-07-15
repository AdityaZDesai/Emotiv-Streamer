#!/usr/bin/env python3
"""
Test script to demonstrate EEG device detection functionality
"""

from emotiv.epoc import EPOC

def main():
    print("🎧 EEG Device Detection Test")
    print("="*50)
    
    # List all devices
    print("Scanning for all USB devices...")
    devices = EPOC.list_all_devices()
    
    if not devices:
        print("No USB devices found!")
        return
    
    print(f"Found {len(devices)} USB devices total")
    
    # Show EEG devices
    eeg_devices = [d for d in devices if d['is_eeg']]
    print(f"Found {len(eeg_devices)} potential EEG devices")
    
    if eeg_devices:
        print("\n🎧 EEG Devices:")
        print("-" * 40)
        for i, device in enumerate(eeg_devices):
            print(f"{i+1}. {device['manufacturer']} - {device['product']}")
            print(f"   Serial: {device['serial']}")
            print(f"   VID:PID: {device['vendor_id']}:{device['product_id']}")
            print()
    
    # Show first few non-EEG devices
    other_devices = [d for d in devices if not d['is_eeg']]
    if other_devices:
        print(f"\n📱 Other USB devices (first 5 of {len(other_devices)}):")
        print("-" * 40)
        for i, device in enumerate(other_devices[:5]):
            print(f"{i+1}. {device['manufacturer']} - {device['product']}")
            print(f"   Serial: {device['serial']}")
            print(f"   VID:PID: {device['vendor_id']}:{device['product_id']}")
            print()
    
    print("Device detection test completed!")

if __name__ == "__main__":
    main() 