from flask import Flask, render_template, jsonify, request, Response, stream_template
import threading
import time
import json
import os
import sys
from datetime import datetime
import datetime as dt
import queue
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Add the python-emotiv directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'python-emotiv'))

# Import the EEG recorder
from record_eeg_with_live_filter import EEGRecorderWithFilter, EEGEmulator

app = Flask(__name__)

# Global variables for EEG recording
eeg_recorder = None
recording_thread = None
recording_active = False
data_queue = queue.Queue()
plot_data = {
    'raw': [],
    'delta': [],
    'theta': [],
    'alpha': [],
    'beta': [],
    'gamma': []
}
timestamps = []
recording_start_time = None

class FlaskEEGRecorder(EEGRecorderWithFilter):
    """Optimized EEG Recorder for Flask - No heavy filtering or plotting"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_callback = None
        # Disable heavy operations for web streaming
        self.enable_live_plot = False  # No matplotlib plotting
        self.verbose_output = False    # No console spam
        
    def set_data_callback(self, callback):
        """Set callback function to send data to Flask"""
        self.data_callback = callback
        
    def add_to_filtered_plot_buffer(self, raw_data, filtered_data, timestamp):
        """Override to send data to Flask - simplified version"""
        # Call the parent method first
        super().add_to_filtered_plot_buffer(raw_data, filtered_data, timestamp)
        
        # Send data to Flask if callback is set
        if self.data_callback:
            # Send raw data
            self.data_callback('raw', raw_data, timestamp)
            
            # Send filtered data for each band
            for band, data in filtered_data.items():
                if band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                    self.data_callback(band, data, timestamp)
    
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
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

                    # Apply lightweight filtering (simplified)
                    filtered_results = {}
                    if self.enable_live_filtering:
                        # Simplified filtering - just store raw data for now
                        # We'll do proper filtering on the frontend or in post-processing
                        if isinstance(raw_data, (list, np.ndarray)) and hasattr(raw_data, '__len__'):
                            # Multi-channel: use ALL channels
                            raw_sample = raw_data  # Keep all channels
                        else:
                            raw_sample = raw_data
                        
                        # Create simple filtered bands (just raw data for now)
                        filtered_results = {
                            'delta': raw_sample,
                            'theta': raw_sample,
                            'alpha': raw_sample,
                            'beta': raw_sample,
                            'gamma': raw_sample
                        }
                        
                        # Store filtered results
                        for band, value in filtered_results.items():
                            if band in self.filtered_data_buffers:
                                self.filtered_data_buffers[band].append(value)

                    # Send to Flask callback
                    if self.data_callback:
                        self.add_to_filtered_plot_buffer(raw_data, filtered_results, current_time)

                    sample_count += 1

                    # Print progress every 128 samples (1 second)
                    if sample_count % 128 == 0:
                        elapsed = time.time() - self.start_time
                        print(f"Recording... {elapsed:.1f}s elapsed, {sample_count} samples collected")

                    # Check duration limit
                    if duration and current_time >= duration:
                        print(f"\nRecording duration ({duration}s) reached. Stopping...")
                        break

                # Minimal delay - removed heavy sleep
                time.sleep(0.0001)  # 0.1ms instead of 1ms

        except Exception as e:
            print(f"Recording error: {e}")
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
            data_values = [0.0]
    else:
        data_values = [float(data) if data is not None else 0.0]
    
    # Only print every 100th sample to reduce console spam
    if len(plot_data.get(band, [])) % 100 == 0:
        print(f"📊 Data callback: {band} = {data_values[:3]}... (channels: {len(data_values)})")
    
    if band in plot_data:
        plot_data[band].append(data_values)
        if len(plot_data[band]) > 1000:  # Keep last 1000 samples (about 20 seconds at 50Hz effective rate)
            plot_data[band].pop(0)
    
    # Add relative timestamp (seconds since recording started)
    if recording_start_time is None:
        recording_start_time = time.time()
        timestamps.append(0.0)
    else:
        relative_time = time.time() - recording_start_time
        timestamps.append(relative_time)
    
    if len(timestamps) > 1000:  # Keep last 1000 timestamps
        timestamps.pop(0)
    
    # Put data in queue for streaming (more frequently for smoother updates)
    if len(plot_data.get(band, [])) % 3 == 0:  # Queue every 3rd sample for smoother updates
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

@app.route('/')
def index():
    """Main page with EEG recording interface"""
    return render_template('index.html')

@app.route('/api/test')
def test():
    """Test endpoint to check if Flask is working"""
    return jsonify({
        'status': 'success',
        'message': 'Flask app is running correctly',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/debug')
def debug():
    """Debug endpoint to check data collection"""
    global plot_data, timestamps, recording_active, eeg_recorder
    
    debug_info = {
        'recording_active': recording_active,
        'data_lengths': {band: len(data) for band, data in plot_data.items()},
        'timestamps_length': len(timestamps),
        'eeg_recorder_exists': eeg_recorder is not None,
        'emulator_mode': getattr(eeg_recorder, 'emulator_mode', False) if eeg_recorder else False,
        'latest_data': {}
    }
    
    # Add latest data samples for each band
    for band, data in plot_data.items():
        if data:
            debug_info['latest_data'][band] = data[-1] if data else 0.0
    
    return jsonify(debug_info)

@app.route('/api/test_data')
def test_data():
    """Generate test data for UI testing"""
    global plot_data, timestamps
    
    import random
    import math
    
    # Generate some fake EEG data
    current_time = time.time()
    
    for band in ['raw', 'delta', 'theta', 'alpha', 'beta', 'gamma']:
        # Generate 20 samples of fake data
        for i in range(20):
            # Create realistic-looking EEG data
            if band == 'raw':
                value = 20 * math.sin(i * 0.5) + random.uniform(-5, 5)
            elif band == 'alpha':
                value = 15 * math.sin(i * 0.3) + random.uniform(-3, 3)
            elif band == 'beta':
                value = 10 * math.sin(i * 0.7) + random.uniform(-2, 2)
            else:
                value = 5 * math.sin(i * 0.4) + random.uniform(-1, 1)
            
            plot_data[band].append(value)
            timestamps.append(current_time + i * 0.1)  # 100ms intervals
            
            # Keep only last 100 samples
            if len(plot_data[band]) > 100:
                plot_data[band].pop(0)
            if len(timestamps) > 100:
                timestamps.pop(0)
    
    return jsonify({
        'status': 'success',
        'message': 'Test data generated',
        'samples_added': 20
    })

@app.route('/api/start_recording', methods=['POST'])
def start_recording():
    """Start EEG recording"""
    global eeg_recorder, recording_thread, recording_active, recording_start_time
    
    if recording_active:
        return jsonify({'status': 'error', 'message': 'Recording already active'})
    
    try:
        print("🔄 Initializing EEG recorder...")
        
        # Initialize EEG recorder
        eeg_recorder = FlaskEEGRecorder()
        eeg_recorder.set_data_callback(data_callback)
        print(f"✅ Data callback set: {eeg_recorder.data_callback is not None}")
        
        print("🔄 Setting up headset...")
        
        # Setup headset with error handling
        try:
            print("🔄 Attempting device detection...")
            success = eeg_recorder.setup_headset()
            
            if not success:
                print("⚠️ Device setup failed, switching to emulator mode")
                # Force emulator mode if device setup fails
                eeg_recorder.epoc = EEGEmulator(sampling_rate=128, num_channels=14)
                eeg_recorder.emulator_mode = True
                eeg_recorder.channels = eeg_recorder.epoc.channels
                success = True
                
        except Exception as setup_error:
            print(f"⚠️ Setup error: {setup_error}")
            print("🔄 Falling back to emulator mode...")
            # Fallback to emulator mode
            try:
                from record_eeg_with_live_filter import EEGEmulator
                eeg_recorder.epoc = EEGEmulator(sampling_rate=128, num_channels=14)
                eeg_recorder.emulator_mode = True
                eeg_recorder.channels = eeg_recorder.epoc.channels
                success = True
            except Exception as emu_error:
                return jsonify({'status': 'error', 'message': f'Failed to initialize emulator: {emu_error}'})
        
        print("🔄 Configuring recording...")
        
        # Configure recording - simplified for web
        eeg_recorder.enable_live_filtering = True
        eeg_recorder.active_filter_bands = ['raw', 'delta', 'theta', 'alpha', 'beta', 'gamma']
        eeg_recorder.enable_live_plot = False  # Disable matplotlib plotting
        print(f"✅ Live filtering enabled: {eeg_recorder.enable_live_filtering}")
        print(f"✅ Live plotting disabled for web optimization")
        print(f"✅ Active bands: {eeg_recorder.active_filter_bands}")
        
        print("🔄 Starting recording thread...")
        
        # Reset recording start time
        recording_start_time = None
        
        # Start recording in background thread
        recording_thread = threading.Thread(target=recording_worker, daemon=True)
        recording_thread.start()
        recording_active = True
        
        # Wait a moment to ensure the thread starts
        time.sleep(0.5)
        
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
            # Send more samples for smoother updates
            sampled_values = values[::1]  # Send every sample for maximum smoothness
            sampled_timestamps = timestamps[::1] if timestamps else []
            
            # For multi-channel data, we need to transpose the data
            if sampled_values and isinstance(sampled_values[0], list):
                # Multi-channel data - transpose to get channels as separate arrays
                num_channels = len(sampled_values[0]) if sampled_values[0] else 1
                channel_data = [[] for _ in range(num_channels)]
                
                for sample in sampled_values[-1000:]:  # Last 1000 samples
                    if isinstance(sample, list):
                        for ch_idx, ch_value in enumerate(sample):
                            if ch_idx < num_channels:
                                channel_data[ch_idx].append(ch_value)
                    else:
                        # Fallback for single value
                        channel_data[0].append(sample)
                
                data[band] = {
                    'values': channel_data,  # Array of arrays, one per channel
                    'timestamps': sampled_timestamps[-1000:] if sampled_timestamps else [],
                    'channels': num_channels
                }
            else:
                # Single channel data
                data[band] = {
                    'values': [sampled_values[-1000:]],  # Wrap in array for consistency
                    'timestamps': sampled_timestamps[-1000:] if sampled_timestamps else [],
                    'channels': 1
                }
    
    return jsonify(data)

@app.route('/api/plot/<band>')
def get_plot(band):
    """Generate and return a plot image for a specific frequency band"""
    global plot_data, timestamps
    
    if band not in plot_data or not plot_data[band]:
        return jsonify({'error': 'No data available for this band'})
    
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Get data
        values = plot_data[band][-100:]  # Last 100 samples
        times = timestamps[-100:] if timestamps else list(range(len(values)))
        
        # Convert timestamps to relative time
        if times and len(times) > 1:
            relative_times = [t - times[0] for t in times]
        else:
            relative_times = list(range(len(values)))
        
        # Plot data
        ax.plot(relative_times, values, linewidth=1)
        ax.set_title(f'{band.upper()} Band EEG Activity')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude (μV)')
        ax.grid(True, alpha=0.3)
        
        # Convert plot to image
        canvas = FigureCanvas(fig)
        canvas.draw()
        
        # Convert to base64 string
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', bbox_inches='tight', dpi=100)
        img_data.seek(0)
        img_base64 = base64.b64encode(img_data.getvalue()).decode()
        
        plt.close(fig)
        
        return jsonify({'image': img_base64})
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/stream')
def stream():
    """Server-sent events stream for real-time data updates"""
    def generate():
        while True:
            try:
                # Wait for new data
                data = data_queue.get(timeout=1)
                
                # Send data as SSE
                yield f"data: {json.dumps(data)}\n\n"
                
            except queue.Empty:
                # Send heartbeat
                yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': time.time()})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/plain')

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    print("🚀 Starting Flask EEG Recording Interface (Optimized)...")
    print("📱 Open your browser to http://localhost:5000")
    print("🎧 The interface will automatically detect EEG devices or use emulator")
    print("⚡ Optimized for web streaming - no heavy filtering or plotting")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 