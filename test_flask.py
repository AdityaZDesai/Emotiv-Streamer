#!/usr/bin/env python3
"""
Test script for Flask EEG Recording Interface
"""

import sys
import os

# Add the python-emotiv directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'python-emotiv'))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import flask
        print("✓ Flask imported successfully")
    except ImportError as e:
        print(f"✗ Flask import failed: {e}")
        return False
    
    try:
        from record_eeg_with_live_filter import EEGRecorder
        print("✓ EEGRecorder imported successfully")
    except ImportError as e:
        print(f"✗ EEGRecorder import failed: {e}")
        return False
    
    try:
        import matplotlib
        print("✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    return True

def test_eeg_recorder():
    """Test EEG recorder initialization"""
    print("\nTesting EEG recorder...")
    
    try:
        from record_eeg_with_live_filter import EEGRecorder
        
        # Test basic initialization
        recorder = EEGRecorder()
        print("✓ EEGRecorder initialized successfully")
        
        # Test device listing
        devices = recorder.list_devices()
        print(f"✓ Found {len(devices)} devices")
        
        return True
        
    except Exception as e:
        print(f"✗ EEGRecorder test failed: {e}")
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        'app.py',
        'templates/index.html',
        'static/css/style.css',
        'static/js/app.js',
        'requirements_flask.txt',
        'README_Flask.md'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("🧪 Testing Flask EEG Recording Interface")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please install required dependencies.")
        return False
    
    # Test file structure
    if not test_file_structure():
        print("\n❌ File structure test failed. Some files are missing.")
        return False
    
    # Test EEG recorder
    if not test_eeg_recorder():
        print("\n❌ EEG recorder test failed.")
        return False
    
    print("\n✅ All tests passed!")
    print("\n🚀 To run the Flask application:")
    print("1. Install Flask: pip install Flask")
    print("2. Run: python app.py")
    print("3. Open browser to: http://localhost:5000")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 