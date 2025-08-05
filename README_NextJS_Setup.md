# EEG Recording Interface - Flask + Next.js Setup

This project uses Flask for the backend API and Next.js for the frontend, providing a modern, responsive EEG recording interface.

## Architecture

- **Backend**: Flask API (Python) - handles EEG data processing and device communication
- **Frontend**: Next.js (React + TypeScript) - provides modern UI with real-time charts

## Setup Instructions

### 1. Backend Setup (Flask API)

1. **Activate the virtual environment:**
   ```bash
   cd python-emotiv
   .\venv\Scripts\Activate.ps1
   cd ..
   ```

2. **Install Flask-CORS:**
   ```bash
   pip install flask-cors
   ```

3. **Start the Flask API server:**
   ```bash
   python app.py
   ```
   
   The API will be available at `http://localhost:5000`

### 2. Frontend Setup (Next.js)

1. **Navigate to the frontend directory:**
   ```bash
   cd eeg-frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the Next.js development server:**
   ```bash
   npm run dev
   ```
   
   The frontend will be available at `http://localhost:3000`

## Usage

1. **Open your browser** and go to `http://localhost:3000`
2. **Click "Start Recording"** to begin EEG data collection
3. **Switch between frequency bands** (Raw, Alpha, Beta, etc.) to view different data
4. **View multi-channel data** - Raw EEG shows all 14 channels with different colors

## Features

### Backend (Flask API)
- ✅ EEG device detection and emulator fallback
- ✅ Real-time data processing
- ✅ Multi-channel support (14 channels)
- ✅ RESTful API endpoints
- ✅ CORS enabled for frontend communication

### Frontend (Next.js)
- ✅ Modern React with TypeScript
- ✅ Real-time Chart.js integration
- ✅ Responsive design with Tailwind CSS
- ✅ Multi-channel visualization
- ✅ Band selection (Raw, Alpha, Beta, Delta, Theta, Gamma)
- ✅ Live value displays

## API Endpoints

- `GET /api/test` - Test API connectivity
- `GET /api/status` - Get recording status
- `GET /api/debug` - Debug data collection
- `GET /api/data` - Get current EEG data
- `POST /api/start_recording` - Start EEG recording
- `POST /api/stop_recording` - Stop EEG recording

## Data Structure

The API returns EEG data in this format:
```json
{
  "raw": {
    "values": [[ch1_data], [ch2_data], ...], // 14 channels
    "timestamps": [...],
    "channels": 14
  },
  "alpha": { ... },
  "beta": { ... }
}
```

## Development

### Backend Development
- Modify `app.py` for API changes
- Add new endpoints as needed
- Test with `curl http://localhost:5000/api/test`

### Frontend Development
- Modify `eeg-frontend/src/components/EEGInterface.tsx` for UI changes
- Add new components in `eeg-frontend/src/components/`
- Hot reload available at `http://localhost:3000`

## Troubleshooting

### CORS Issues
- Ensure Flask-CORS is installed: `pip install flask-cors`
- Check that CORS is enabled in `app.py`

### Connection Issues
- Verify Flask API is running on port 5000
- Check browser console for API errors
- Test API directly: `curl http://localhost:5000/api/test`

### Data Not Showing
- Check browser console for errors
- Verify recording is active: `curl http://localhost:5000/api/status`
- Check debug endpoint: `curl http://localhost:5000/api/debug`

## Benefits of This Setup

1. **Separation of Concerns**: Backend handles data, frontend handles UI
2. **Modern Frontend**: React with TypeScript for better development experience
3. **Better Performance**: Next.js optimizations and server-side rendering
4. **Scalability**: Easy to add new features to either backend or frontend
5. **Maintainability**: Clean code structure and modern tooling

## Next Steps

- Add authentication
- Implement data export features
- Add more visualization options
- Create mobile-responsive design
- Add real-time filtering options 