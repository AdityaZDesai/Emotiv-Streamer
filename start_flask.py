#!/usr/bin/env python3
"""
Startup script for Flask EEG Recording Interface
"""

import sys
import os
import subprocess
import importlib.util

def check_package(package_name):
    """Check if a package is installed"""
    return importlib.util.find_spec(package_name) is not None

def install_package(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install_dependencies():
    """Check and install required dependencies"""
    print("🔍 Checking dependencies...")
    
    required_packages = {
        'flask': 'Flask',
        'matplotlib': 'matplotlib',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'pandas': 'pandas'
    }
    
    missing_packages = []
    
    for package, pip_name in required_packages.items():
        if check_package(package):
            print(f"✓ {package} is installed")
        else:
            print(f"✗ {package} is missing")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        
        for package in missing_packages:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✓ {package} installed successfully")
            else:
                print(f"✗ Failed to install {package}")
                return False
    
    return True

def main():
    """Main startup function"""
    print("🚀 Starting Flask EEG Recording Interface")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ Error: app.py not found in current directory")
        print("Please run this script from the project root directory")
        return False
    
    # Check and install dependencies
    if not check_and_install_dependencies():
        print("❌ Failed to install required dependencies")
        return False
    
    # Add python-emotiv to path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'python-emotiv'))
    
    # Try to import the EEG recorder
    try:
        from record_eeg_with_live_filter import EEGRecorderWithFilter
        print("✓ EEG recorder module imported successfully")
    except ImportError as e:
        print(f"⚠️ Warning: Could not import EEG recorder: {e}")
        print("The web interface will still work, but EEG recording functionality may be limited")
    
    # Start Flask app
    print("\n🌐 Starting Flask web server...")
    print("📱 Open your browser to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Import and run Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting Flask app: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 