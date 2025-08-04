# EEG Recording Flask Interface

A modern web-based interface for controlling EEG recording and viewing live plots through a browser.

## Features

- 🎧 **Automatic Device Detection**: Automatically detects EEG devices or switches to emulator mode
- 📊 **Real-time Visualization**: Live plotting of EEG data with Chart.js
- 🎛️ **Web Controls**: Start/stop recording through a beautiful web interface
- 📱 **Responsive Design**: Works on desktop and mobile devices
- 🔄 **Multiple Frequency Bands**: View raw, delta, theta, alpha, beta, and gamma bands
- 💾 **Data Logging**: Real-time log of all activities and events
- 🚀 **Server-Sent Events**: Real-time data streaming to the browser

## Quick Start

### 1. Install Dependencies

```bash
# Install Flask dependencies
pip install -r requirements_flask.txt

# Install EEG recording dependencies (if not already installed)
pip install -r python-emotiv/requirements.txt
```

### 2. Run the Application

```bash
python app.py
```

### 3. Open Your Browser

Navigate to: `http://localhost:5000`

## Interface Overview

### Control Panel
- **Start Recording**: Initialize EEG device/emulator and begin recording
- **Stop Recording**: Stop recording and save data to files
- **Device Status**: Shows current device mode, channels, and sampling rate

### Live Plots
- **Band Selection**: Switch between different frequency bands using the buttons
- **Real-time Chart**: Live updating chart showing EEG activity
- **Current Values**: Real-time numerical values for each frequency band

### Data Log
- **Activity Log**: Timestamped log of all system activities
- **Error Reporting**: Clear error messages and status updates

## Frequency Bands

- **Raw**: Unfiltered EEG activity
- **Delta (0.5-4 Hz)**: Deep sleep, unconscious processes
- **Theta (4-8 Hz)**: Memory, creativity, REM sleep
- **Alpha (8-13 Hz)**: Relaxation, closed eyes
- **Beta (13-30 Hz)**: Active thinking, focus
- **Gamma (30-100 Hz)**: High-level processing

## Device Modes

### Physical Device Mode
When an EEG headset is detected:
- Connects to the physical device
- Records real EEG data
- Shows actual channel information

### Emulator Mode
When no physical device is detected:
- Automatically switches to emulator
- Generates realistic EEG data
- Simulates different brain states
- Perfect for testing and development

## API Endpoints

- `GET /` - Main interface
- `POST /api/start_recording` - Start EEG recording
- `POST /api/stop_recording` - Stop EEG recording
- `GET /api/status` - Get current recording status
- `GET /api/data` - Get current EEG data
- `GET /api/plot/<band>` - Get plot image for specific band
- `GET /stream` - Server-sent events stream

## File Structure

```
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Main web interface
├── static/
│   ├── css/
│   │   └── style.css     # Custom styling
│   └── js/
│       └── app.js        # Frontend JavaScript
├── python-emotiv/        # EEG recording backend
├── requirements_flask.txt # Flask dependencies
└── README_Flask.md       # This file
```

## Troubleshooting

### No EEG Device Detected
- The system will automatically switch to emulator mode
- This is normal behavior for testing without physical hardware

### Connection Issues
- Check that the Flask server is running on port 5000
- Ensure no firewall is blocking the connection
- Try refreshing the browser page

### Chart Not Updating
- Check the browser console for JavaScript errors
- Ensure the data stream is connected (check log messages)
- Try switching between different frequency bands

## Data Storage

When recording stops, data is automatically saved to:
- `.mat` files (MATLAB format)
- `.csv` files (comma-separated values)
- `.json` files (JSON format)
- `.npz` files (NumPy compressed format)

Files are saved in the current directory with timestamps.

## Browser Compatibility

- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

## Development

### Adding New Features
1. Modify `app.py` for backend changes
2. Update `templates/index.html` for UI changes
3. Edit `static/js/app.js` for frontend functionality
4. Customize `static/css/style.css` for styling

### Customizing the Interface
- Colors and styling can be modified in `style.css`
- Chart appearance can be adjusted in `app.js`
- Layout changes can be made in `index.html`

## Security Notes

- The application runs in debug mode for development
- For production use, disable debug mode and use proper WSGI server
- Consider adding authentication for multi-user environments
- Ensure proper firewall rules for network access

## License

This interface is part of the Emotiv Streamer project and follows the same licensing terms. 