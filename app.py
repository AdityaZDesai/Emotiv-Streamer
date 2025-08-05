#!/usr/bin/env python3
"""
Flask API for EEG Recording Interface
Backend API only - frontend will be Next.js
"""

import sys
import os
sys.path.append('python-emotiv')

from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import queue
import time
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from record_eeg_with_live_filter import EEGRecorderWithFilter, EEGEmulator, LiveEEGFilter

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Global variables for EEG recorder state
eeg_recorder = None
recording_active = False
data_queue = queue.Queue(maxsize=1000)
plot_data = {'raw': [], 'alpha': [], 'beta': [], 'delta': [], 'theta': [], 'gamma': []}
timestamps = []
recording_start_time = None

class FlaskEEGRecorder(EEGRecorderWithFilter):
    """Flask-optimized EEG recorder with simplified processing"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_callback = None
        self.enable_live_plot = False  # Disable matplotlib plotting for web
        self.verbose_output = False  # Reduce console output
        
    def set_data_callback(self, callback):
        """Set callback function for data streaming"""
        self.data_callback = callback
    
    def add_to_filtered_plot_buffer(self, raw_data, filtered_data, timestamp):
        """Override to send data to Flask - simplified version"""
        # Don't call parent method to avoid any unwanted data sending
        # super().add_to_filtered_plot_buffer(raw_data, filtered_data, timestamp)
        
        # Send data to Flask if callback is set
        if self.data_callback:
            # Send raw data
            self.data_callback('raw', raw_data, timestamp)
            
            # Send filtered data for each band only if it has actual data
            for band, data in filtered_data.items():
                if band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                    # Only send data if it's not empty
                    if data and len(data) > 0:
                        print(f"🔍 SENDING {band} data: {len(data)} values")
                        self.data_callback(band, data, timestamp)
                    else:
                        print(f"🔍 SKIPPING {band} data - empty")
                    # Don't send anything if data is empty - this will keep the chart empty
    
    def start_recording(self, duration=None):
        """Optimized recording - no heavy filtering or plotting"""
        if not self.epoc:
            print("Headset not initialized!")
            return False

        # Don't set up signal handler in background thread
        self.recording = True
        self.data_buffer = []
        for band in self.filtered_data_buffers:
            self.filtered_data_buffers[band] = []
        self.time_buffer = []
        self.start_time = time.time()
        self.accel_buffer = []

        print(f"\n{'='*80}")
        print("🎧 EEG Recording Started (Web Optimized)!")
        print(f"{'='*80}")
        
        if hasattr(self, 'emulator_mode') and self.emulator_mode:
            print("🤖 Recording from EEG Emulator...")
        else:
            print("Recording EEG data from headset...")
        
        print("Recording will continue until stopped via web interface")

        # Get channel information
        channels = None
        if hasattr(self.epoc, 'channels') and getattr(self.epoc, 'channels', None):
            channels = getattr(self.epoc, 'channels', None)

        print(f"Sampling rate: {self.sampling_rate} Hz")
        print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")

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
                if hasattr(self.epoc, 'get_accel'):
                    accel_data = self.epoc.get_accel()
                else:
                    accel_data = None
                self.accel_buffer.append(accel_data)

                if raw_data:
                    current_time = time.time() - self.start_time

                    # Store raw data
                    self.data_buffer.append(raw_data)
                    self.time_buffer.append(current_time)
                    self.filtered_data_buffers['raw'].append(raw_data)

                    # Apply proper live filtering
                    filtered_results = {}
                    print(f"🔍 FILTER DEBUG: enable_live_filtering={self.enable_live_filtering}, hasattr live_filter={hasattr(self, 'live_filter')}")
                    
                    # Ensure filtering is enabled
                    if not hasattr(self, 'live_filter'):
                        print("🔧 Setting up live filter...")
                        self.live_filter = LiveEEGFilter(fs=self.sampling_rate)
                        # Reduce buffer size for faster response
                        self.live_filter.artifact_window = int(self.sampling_rate * 0.5)  # 0.5 seconds instead of 2
                        print(f"🔧 Filter artifact window set to: {self.live_filter.artifact_window} samples")
                        self.enable_live_filtering = True
                    else:
                        # Ensure the existing filter has the correct window size
                        if self.live_filter.artifact_window != int(self.sampling_rate * 0.5):
                            self.live_filter.artifact_window = int(self.sampling_rate * 0.5)
                            print(f"🔧 Updated filter artifact window to: {self.live_filter.artifact_window} samples")
                    
                    if self.enable_live_filtering and hasattr(self, 'live_filter'):
                        print(f"🔍 DEBUG: Live filtering enabled, processing {len(raw_data) if isinstance(raw_data, (list, np.ndarray)) else 1} channels")
                        print(f"🔍 DEBUG: Raw data sample: {raw_data[:3] if isinstance(raw_data, (list, np.ndarray)) else raw_data}")
                        print(f"🔍 DEBUG: Buffer has {len(self.live_filter.raw_buffer)} samples, need {self.live_filter.artifact_window}")
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
                                if ch_idx == 0:  # Debug first channel only
                                    print(f"🔍 DEBUG: Channel {ch_idx} filtered results: {ch_filtered}")
                                    
                                    # Debug: Check if filtering actually worked
                                    if 'delta' in ch_filtered:
                                        raw_val = raw_sample
                                        delta_val = ch_filtered['delta']
                                        # Handle empty list case
                                        if isinstance(delta_val, list) and len(delta_val) == 0:
                                            print(f"🔧 DEBUG: Channel {ch_idx} delta band - buffer not full yet (empty list)")
                                        elif abs(raw_val - delta_val) < 0.001:
                                            print(f"⚠️ WARNING: Channel {ch_idx} filtering may not be working - raw={raw_val:.6f}, delta={delta_val:.6f}")
                                        else:
                                            print(f"✅ SUCCESS: Channel {ch_idx} filtering working - raw={raw_val:.6f}, delta={delta_val:.6f}")
                                
                                for band, value in ch_filtered.items():
                                    # Handle empty lists (buffer not full yet)
                                    if isinstance(value, list) and len(value) == 0:
                                        # Use 0.0 for empty data to maintain channel count
                                        multi_channel_filtered[band].append(0.0)
                                    else:
                                        multi_channel_filtered[band].append(value)

                            # Store multi-channel filtered results
                            filtered_results = multi_channel_filtered
                            for band, values in filtered_results.items():
                                if band in self.filtered_data_buffers:
                                    self.filtered_data_buffers[band].append(values)
                        else:
                            # Not enough data for filtering yet - provide empty data
                            print(f"🔍 DEBUG: Not enough data for filtering ({len(raw_list)} samples), providing empty data")
                            # Create empty filtered bands until filtering is ready
                            filtered_results = {
                                'delta': [],
                                'theta': [],
                                'alpha': [],
                                'beta': [],
                                'gamma': []
                            }
                            # Store empty results
                            for band, values in filtered_results.items():
                                if band in self.filtered_data_buffers:
                                    self.filtered_data_buffers[band].append(values)
                            
                            # Debug: Check if filtering is working
                            if sample_count % 100 == 0:  # Every 100 samples
                                print(f"🔍 DEBUG: Sample {sample_count} - Raw vs Filtered comparison:")
                                print(f"  Raw: {raw_list[:3]}")
                                for band in ['alpha', 'beta', 'delta', 'theta', 'gamma']:
                                    if band in filtered_results:
                                        print(f"  {band}: {filtered_results[band][:3]}")
                                
                                # Check if filtering is actually producing different values
                                if 'alpha' in filtered_results and 'beta' in filtered_results:
                                    alpha_avg = sum(filtered_results['alpha'][:3]) / 3
                                    beta_avg = sum(filtered_results['beta'][:3]) / 3
                                    raw_avg = sum(raw_list[:3]) / 3
                                    print(f"  Averages - Raw: {raw_avg:.3f}, Alpha: {alpha_avg:.3f}, Beta: {beta_avg:.3f}")
                                    if abs(alpha_avg - beta_avg) < 0.1:
                                        print("  ⚠️ WARNING: Alpha and Beta values are too similar - filtering may not be working!")
                                    else:
                                        print("  ✅ Filtering is working - bands have different values!")
                                print("---")
                    else:
                        # Single channel data
                        raw_sample = raw_data
                        filtered_results = self.live_filter.process_sample(raw_sample, channel=0)
                        for band, value in filtered_results.items():
                            if band in self.filtered_data_buffers:
                                # Handle empty lists (buffer not full yet)
                                if isinstance(value, list) and len(value) == 0:
                                    # Use 0.0 for empty data to maintain consistency
                                    self.filtered_data_buffers[band].append(0.0)
                                else:
                                    self.filtered_data_buffers[band].append(value)

                    # Add to plot buffers for web streaming
                    if self.data_callback:
                        self.add_to_filtered_plot_buffer(raw_data, filtered_results, current_time)

                    sample_count += 1

                    # Print progress every 128 samples (approximately 1 second for 128Hz)
                    if sample_count % 128 == 0:
                        elapsed = time.time() - self.start_time
                        print(f"Recording... {elapsed:.1f}s elapsed, {sample_count} samples collected")

                    # Check duration limit
                    if duration and current_time >= duration:
                        print(f"\nRecording duration ({duration}s) reached. Stopping...")
                        break

                # Minimal delay for maximum responsiveness
                time.sleep(0.00001)  # 10 microseconds for ultra-smooth updates

        except Exception as e:
            print(f"Error during recording: {e}")
            return False

        return True

def data_callback(band, data, timestamp):
    """Optimized callback function to send EEG data to Flask"""
    global plot_data, timestamps, recording_start_time
    
    # Debug: Print data structure for first few samples
    if len(plot_data.get(band, [])) < 5:
        print(f"🔍 DEBUG {band}: data type={type(data)}, length={len(data) if hasattr(data, '__len__') else 'N/A'}")
        if isinstance(data, (list, np.ndarray)) and len(data) > 0:
            print(f"🔍 DEBUG {band}: data[0] type={type(data[0])}, value={data[0]}")
    
    # Convert data to a format that can be serialized
    if isinstance(data, (list, np.ndarray)):
        if len(data) > 0:
            # For multi-channel data, store all channels
            if isinstance(data[0], (list, np.ndarray)):
                # Multi-channel data - store as list of values
                data_values = []
                for ch in data:
                    if isinstance(ch, (list, np.ndarray)) and len(ch) > 0:
                        data_values.append(float(ch))  # Use the channel value directly
                    else:
                        data_values.append(float(ch) if ch is not None else 0.0)
            else:
                # Multi-channel data (list of numbers) - store all channels
                data_values = []
                for value in data:
                    data_values.append(float(value) if value is not None else 0.0)
        else:
            # Don't send any data if the input is empty
            return
    else:
        data_values = [float(data) if data is not None else 0.0]
    
    # Only print every 500th sample to reduce console spam
    if len(plot_data.get(band, [])) % 500 == 0:
        print(f"📊 Data callback: {band} = {data_values[:3]}... (channels: {len(data_values)})")
    
    if band in plot_data:
        plot_data[band].append(data_values)
        if len(plot_data[band]) > 500:  # Keep last 500 samples for faster processing
            plot_data[band].pop(0)
    
    # Add relative timestamp (seconds since recording started)
    if recording_start_time is None:
        recording_start_time = time.time()
        timestamps.append(0.0)
    else:
        relative_time = time.time() - recording_start_time
        timestamps.append(relative_time)
    
    if len(timestamps) > 500:  # Keep last 500 timestamps
        timestamps.pop(0)
    
    # Put data in queue for streaming (every sample for maximum smoothness)
    try:
        data_queue.put_nowait({
            'band': band,
            'data': data_values,
            'timestamp': timestamps[-1]  # Use relative timestamp
        })
    except queue.Full:
        pass  # Queue is full, skip this data point

def recording_worker():
    """Worker thread for EEG recording"""
    global eeg_recorder, recording_active
    
    try:
        print("🎧 Starting EEG recording in background thread...")
        if eeg_recorder and eeg_recorder.epoc:
            eeg_recorder.start_recording()
        else:
            print("❌ No EEG recorder or device available")
    except KeyboardInterrupt:
        print("Recording stopped by user")
    except Exception as e:
        print(f"❌ Error in recording thread: {e}")
        import traceback
        traceback.print_exc()
    finally:
        recording_active = False
        if eeg_recorder:
            try:
                eeg_recorder.disconnect()
            except Exception as e:
                print(f"Warning: Error during disconnect: {e}")

# API Routes
@app.route('/api/test')
def test():
    """Test endpoint to check if Flask API is working"""
    return jsonify({
        'status': 'success',
        'message': 'Flask API is running correctly',
        'timestamp': datetime.datetime.now().isoformat()
    })

@app.route('/api/debug')
def debug():
    """Debug endpoint to check data collection"""
    global plot_data, timestamps, recording_active, eeg_recorder
    
    debug_info = {
        'recording_active': recording_active,
        'eeg_recorder_exists': eeg_recorder is not None,
        'emulator_mode': getattr(eeg_recorder, 'emulator_mode', False) if eeg_recorder else False,
        'data_lengths': {band: len(values) for band, values in plot_data.items()},
        'timestamps_length': len(timestamps),
        'latest_data': {}
    }
    
    # Add latest data sample for each band
    for band, values in plot_data.items():
        if values:
            debug_info['latest_data'][band] = values[-1]
    
    return jsonify(debug_info)

@app.route('/api/test_data')
def test_data():
    """Generate test data for frontend development"""
    import random
    
    # Generate fake EEG data
    test_data = {}
    bands = ['raw', 'alpha', 'beta', 'delta', 'theta', 'gamma']
    
    for band in bands:
        # Generate 100 samples of 14-channel data
        values = []
        for _ in range(100):
            channel_data = []
            for ch in range(14):
                # Generate realistic EEG-like values
                value = random.uniform(-50, 50) + random.uniform(-10, 10) * random.random()
                channel_data.append(value)
            values.append(channel_data)
        
        test_data[band] = {
            'values': values,
            'timestamps': [i * 0.1 for i in range(100)],  # 0.1s intervals
            'channels': 14
        }
    
    return jsonify(test_data)

@app.route('/api/start_recording', methods=['POST'])
def start_recording():
    """Start EEG recording"""
    global eeg_recorder, recording_active, plot_data, timestamps, recording_start_time
    
    if recording_active:
        return jsonify({'status': 'error', 'message': 'Recording already active'})
    
    try:
        # Reset data buffers
        for band in plot_data:
            plot_data[band] = []
        timestamps = []
        recording_start_time = None
        
        # Initialize EEG recorder if not exists
        if eeg_recorder is None:
            print("🎧 Initializing EEG recorder...")
            eeg_recorder = FlaskEEGRecorder()
            eeg_recorder.set_data_callback(data_callback)
            
            # Setup headset with timeout and fallback
            try:
                if not eeg_recorder.setup_headset():
                    print("⚠️ Headset setup failed, using emulator")
                    # Force emulator mode
                    eeg_recorder.epoc = EEGEmulator(sampling_rate=128, num_channels=14)
                    eeg_recorder.emulator_mode = True
                    eeg_recorder.channels = eeg_recorder.epoc.channels
            except Exception as e:
                print(f"⚠️ Error during headset setup: {e}")
                print("🤖 Falling back to emulator mode")
                eeg_recorder.epoc = EEGEmulator(sampling_rate=128, num_channels=14)
                eeg_recorder.emulator_mode = True
                eeg_recorder.channels = eeg_recorder.epoc.channels
        
        # Enable live plotting for data streaming
        eeg_recorder.enable_live_plot = True
        
        # Start recording in background thread
        recording_active = True
        recording_thread = threading.Thread(target=recording_worker, daemon=True)
        recording_thread.start()
        
        mode = "Emulator" if getattr(eeg_recorder, 'emulator_mode', False) else "Real Device"
        print(f"✅ Recording started successfully in {mode} mode")
        
        response_data = {
            'status': 'success',
            'message': f'Recording started successfully in {mode} mode',
            'emulator_mode': getattr(eeg_recorder, 'emulator_mode', False)
        }
        
        print(f"🔄 Returning response: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error starting recording: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/stop_recording', methods=['POST'])
def stop_recording():
    """Stop EEG recording"""
    global eeg_recorder, recording_active
    
    if not recording_active:
        return jsonify({'status': 'error', 'message': 'No recording active'})
    
    try:
        recording_active = False
        
        if eeg_recorder:
            eeg_recorder.stop_recording()
            eeg_recorder.disconnect()
        
        return jsonify({'status': 'success', 'message': 'Recording stopped and data saved'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/status')
def get_status():
    """Get current recording status"""
    global recording_active, eeg_recorder
    
    status = {
        'recording': recording_active,
        'emulator_mode': getattr(eeg_recorder, 'emulator_mode', False) if eeg_recorder else False,
        'channels': getattr(eeg_recorder, 'channels', []) if eeg_recorder else [],
        'sampling_rate': getattr(eeg_recorder, 'sampling_rate', 0) if eeg_recorder else 0
    }
    
    return jsonify(status)

@app.route('/api/data')
def get_data():
    """Get current EEG data for plotting - optimized for smooth updates"""
    global plot_data, timestamps
    
    # Convert data to format suitable for JavaScript
    data = {}
    for band, values in plot_data.items():
        if values:
            # Send all samples for maximum smoothness
            sampled_values = values  # Send every sample
            sampled_timestamps = timestamps if timestamps else []
            
            # For multi-channel data, we need to transpose the data
            if sampled_values and isinstance(sampled_values[0], list):
                # Multi-channel data - transpose to get channels as separate arrays
                num_channels = len(sampled_values[0]) if sampled_values[0] else 1
                channel_data = [[] for _ in range(num_channels)]
                
                for sample in sampled_values[-500:]:  # Last 500 samples for faster processing
                    if isinstance(sample, list):
                        for ch_idx, ch_value in enumerate(sample):
                            if ch_idx < num_channels:
                                channel_data[ch_idx].append(ch_value)
                    else:
                        # Fallback for single value
                        channel_data[0].append(sample)
                
                data[band] = {
                    'values': channel_data,  # Array of arrays, one per channel
                    'timestamps': sampled_timestamps[-500:] if sampled_timestamps else [],
                    'channels': num_channels
                }
            else:
                # Single channel data
                data[band] = {
                    'values': [sampled_values[-500:]],  # Wrap in array for consistency
                    'timestamps': sampled_timestamps[-500:] if sampled_timestamps else [],
                    'channels': 1
                }
    
    return jsonify(data)

if __name__ == '__main__':
    print("🚀 Starting Flask EEG API Server...")
    print("📱 API will be available at http://localhost:5000")
    print("🎧 The interface will automatically detect EEG devices or use emulator")
    print("⚡ Optimized for web streaming - no heavy filtering or plotting")
    app.run(host='0.0.0.0', port=5000, debug=True) 