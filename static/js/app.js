// EEG Recording Interface JavaScript

class EEGInterface {
    constructor() {
        this.chart = null;
        this.currentBand = 'raw';
        this.isRecording = false;
        this.eventSource = null;
        this.updateInterval = null;
        
        this.initializeChart();
        this.bindEvents();
        this.updateStatus();
    }
    
    initializeChart() {
        const ctx = document.getElementById('eegChart').getContext('2d');
        
        // Channel colors for multi-channel display
        const channelColors = [
            '#667eea',  // Blue
            '#f093fb',  // Pink
            '#4facfe',  // Light Blue
            '#43e97b',  // Green
            '#fa709a',  // Rose
            '#fee140',  // Yellow
            '#a8edea',  // Cyan
            '#fed6e3'   // Light Pink
        ];
        
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Channel 1',
                    data: [],
                    borderColor: channelColors[0],
                    backgroundColor: channelColors[0] + '20',
                    borderWidth: 1.5,
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 0
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Time (seconds)'
                        },
                        grid: {
                            color: 'rgba(0,0,0,0.1)'
                        },
                        ticks: {
                            maxTicksLimit: 25  // More ticks to show 20 seconds properly
                        }
                        // Remove fixed min/max to allow auto-scaling based on data
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Amplitude (μV)'
                        },
                        grid: {
                            color: 'rgba(0,0,0,0.1)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            usePointStyle: true,
                            padding: 10,
                            font: {
                                size: 11
                            }
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }
    
    bindEvents() {
        // Start recording button
        document.getElementById('startBtn').addEventListener('click', () => {
            this.startRecording();
        });
        
        // Stop recording button
        document.getElementById('stopBtn').addEventListener('click', () => {
            this.stopRecording();
        });
        
        // Band selection buttons
        document.querySelectorAll('.btn-group .btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const band = e.target.textContent.toLowerCase();
                this.showBand(band);
            });
        });
    }
    
    async startRecording() {
        try {
            this.showLoading('Initializing EEG recording...');
            
            // Reset chart data for new recording
            this.resetChart();
            
            const response = await fetch('/api/start_recording', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.isRecording = true;
                this.updateUI();
                this.startDataStream();
                this.logMessage(`Recording started successfully. ${result.emulator_mode ? 'Using EEG Emulator mode.' : 'Using physical EEG device.'}`, 'success');
            } else {
                this.logMessage(`Failed to start recording: ${result.message}`, 'error');
            }
            
        } catch (error) {
            this.logMessage(`Error starting recording: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    async stopRecording() {
        try {
            this.showLoading('Stopping recording and saving data...');
            
            const response = await fetch('/api/stop_recording', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.isRecording = false;
                this.updateUI();
                this.stopDataStream();
                this.logMessage('Recording stopped and data saved successfully.', 'success');
            } else {
                this.logMessage(`Failed to stop recording: ${result.message}`, 'error');
            }
            
        } catch (error) {
            this.logMessage(`Error stopping recording: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    startDataStream() {
        // Use polling for data updates (more reliable than SSE)
        this.updateInterval = setInterval(() => {
            this.updateData();
        }, 100); // Update every 100ms for smoother 10 FPS
    }
    
    stopDataStream() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }
    
    async updateData() {
        try {
            const response = await fetch('/api/data');
            const data = await response.json();
            
            console.log('Received data:', data);
            console.log('Current band:', this.currentBand);
            
            // Update current values display
            Object.keys(data).forEach(band => {
                const valueElement = document.getElementById(`${band}Value`);
                if (valueElement && data[band].values.length > 0) {
                    const latestValue = data[band].values[data[band].values.length - 1];
                    // Handle multi-channel data - show average of all channels
                    if (Array.isArray(latestValue)) {
                        const avgValue = latestValue.reduce((sum, val) => sum + val, 0) / latestValue.length;
                        valueElement.textContent = avgValue.toFixed(2);
                    } else {
                        valueElement.textContent = latestValue.toFixed(2);
                    }
                }
            });
            
            // Update chart if current band has data
            if (data[this.currentBand] && data[this.currentBand].values.length > 0) {
                console.log('Updating chart with data:', data[this.currentBand]);
                this.updateChart(data[this.currentBand]);
            } else {
                console.log('No data for current band or no values');
            }
            
        } catch (error) {
            console.error('Error updating data:', error);
        }
    }
    
    processData(data) {
        // Update current values
        const valueElement = document.getElementById(`${data.band}Value`);
        if (valueElement) {
            valueElement.textContent = data.data.toFixed(2);
        }
        
        // Update chart if this is the current band
        if (data.band === this.currentBand) {
            this.addDataPoint(data.data, data.timestamp);
        }
    }
    
    updateChart(data) {
        console.log('updateChart called with data:', data);
        // Update chart with new data from polling - optimized for multi-channel display
        if (data.values && data.values.length > 0) {
            console.log('Data has values, length:', data.values.length);
            // Clear existing data
            this.chart.data.labels = [];
            
            // Handle multi-channel data
            if (Array.isArray(data.values[0])) {
                // Multi-channel data - each element in data.values is a channel array
                const numChannels = data.values.length;
                
                // Ensure we have enough datasets
                while (this.chart.data.datasets.length < numChannels) {
                    const channelIndex = this.chart.data.datasets.length;
                    const channelColors = [
                        '#667eea', '#f093fb', '#4facfe', '#43e97b', 
                        '#fa709a', '#fee140', '#a8edea', '#fed6e3'
                    ];
                    
                    this.chart.data.datasets.push({
                        label: `Channel ${channelIndex + 1}`,
                        data: [],
                        borderColor: channelColors[channelIndex % channelColors.length],
                        backgroundColor: channelColors[channelIndex % channelColors.length] + '20',
                        borderWidth: 1.5,
                        fill: false,
                        tension: 0.1,
                        pointRadius: 0,
                        pointHoverRadius: 3
                    });
                }
                
                // Remove extra datasets if we have fewer channels now
                this.chart.data.datasets.splice(numChannels);
                
                // Add data for each channel
                for (let ch = 0; ch < numChannels; ch++) {
                    this.chart.data.datasets[ch].data = [];
                    
                    data.values[ch].forEach((value, index) => {
                        const time = data.timestamps[index] || index;
                        const timeValue = typeof time === 'number' ? time : parseFloat(time) || index;
                        
                        if (ch === 0) { // Only add time labels once
                            this.chart.data.labels.push(timeValue);
                        }
                        this.chart.data.datasets[ch].data.push(value);
                    });
                }
            } else {
                // Single channel data
                this.chart.data.datasets[0].data = [];
                data.values.forEach((value, index) => {
                    const time = data.timestamps[index] || index;
                    const timeValue = typeof time === 'number' ? time : parseFloat(time) || index;
                    this.chart.data.labels.push(timeValue);
                    this.chart.data.datasets[0].data.push(value);
                });
            }
            
            this.chart.update('none');
        }
    }
    
    addDataPoint(value, timestamp) {
        // Handle multi-channel data
        const timeValue = typeof timestamp === 'number' ? timestamp : parseFloat(timestamp) || 0;
        
        // Add time label
        this.chart.data.labels.push(timeValue);
        
        // Handle multi-channel values
        if (Array.isArray(value)) {
            // Multi-channel data
            const numChannels = value.length;
            
            // Ensure we have enough datasets
            while (this.chart.data.datasets.length < numChannels) {
                const channelIndex = this.chart.data.datasets.length;
                const channelColors = [
                    '#667eea', '#f093fb', '#4facfe', '#43e97b', 
                    '#fa709a', '#fee140', '#a8edea', '#fed6e3'
                ];
                
                this.chart.data.datasets.push({
                    label: `Channel ${channelIndex + 1}`,
                    data: [],
                    borderColor: channelColors[channelIndex % channelColors.length],
                    backgroundColor: channelColors[channelIndex % channelColors.length] + '20',
                    borderWidth: 1.5,
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 3
                });
            }
            
            // Add data for each channel
            for (let ch = 0; ch < numChannels; ch++) {
                if (this.chart.data.datasets[ch]) {
                    this.chart.data.datasets[ch].data.push(value[ch]);
                    
                    // Keep only last 500 points per channel
                    if (this.chart.data.datasets[ch].data.length > 500) {
                        this.chart.data.datasets[ch].data.shift();
                    }
                }
            }
        } else {
            // Single channel data
            this.chart.data.datasets[0].data.push(value);
            
            // Keep only last 500 points
            if (this.chart.data.datasets[0].data.length > 500) {
                this.chart.data.datasets[0].data.shift();
            }
        }
        
        // Keep only last 500 time labels
        if (this.chart.data.labels.length > 500) {
            this.chart.data.labels.shift();
        }
        
        this.chart.update('none');
    }
    
    showBand(band) {
        this.currentBand = band;
        
        // Update button states
        document.querySelectorAll('.btn-group .btn').forEach(btn => {
            btn.classList.remove('active');
            if (btn.textContent.toLowerCase() === band) {
                btn.classList.add('active');
            }
        });
        
        // Update chart title
        this.chart.options.plugins.title = {
            display: true,
            text: `${band.toUpperCase()} Band EEG Activity - Multi-Channel View`
        };
        
        // Update legend visibility based on band
        if (band === 'raw') {
            this.chart.options.plugins.legend.display = true; // Show legend for multi-channel raw data
        } else {
            this.chart.options.plugins.legend.display = false; // Hide legend for filtered bands
        }
        
        // Update chart
        this.chart.update();
        
        // Fetch data for this band
        this.updateData();
        
        this.logMessage(`Switched to ${band.toUpperCase()} band view`, 'info');
    }
    
    updateUI() {
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const statusAlert = document.getElementById('statusAlert');
        
        if (this.isRecording) {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            document.body.classList.add('recording');
            
            statusAlert.className = 'alert alert-success';
            statusAlert.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> Recording in progress...';
        } else {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            document.body.classList.remove('recording');
            
            statusAlert.className = 'alert alert-info';
            statusAlert.innerHTML = '<i class="fas fa-info-circle"></i> Ready to start recording';
        }
    }
    
    async updateStatus() {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();
            
            // Update status display
            const recordingStatus = document.getElementById('recordingStatus');
            const deviceMode = document.getElementById('deviceMode');
            const channelCount = document.getElementById('channelCount');
            const sampleRate = document.getElementById('sampleRate');
            const channelList = document.getElementById('channelList');
            
            recordingStatus.textContent = status.recording ? 'Recording' : 'Stopped';
            recordingStatus.className = `badge ${status.recording ? 'bg-success' : 'bg-secondary'}`;
            
            deviceMode.textContent = status.emulator_mode ? 'Emulator' : 'Physical Device';
            deviceMode.className = `badge ${status.emulator_mode ? 'bg-warning' : 'bg-info'}`;
            
            channelCount.textContent = status.channels.length;
            sampleRate.textContent = status.sampling_rate;
            channelList.textContent = status.channels.join(', ') || 'None';
            
        } catch (error) {
            console.error('Error updating status:', error);
        }
    }
    
    logMessage(message, type = 'info') {
        const logContainer = document.getElementById('dataLog');
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${type}`;
        
        const timestamp = new Date().toLocaleTimeString();
        logEntry.textContent = `[${timestamp}] ${message}`;
        
        logContainer.appendChild(logEntry);
        logContainer.scrollTop = logContainer.scrollHeight;
        
        // Keep only last 100 log entries
        while (logContainer.children.length > 100) {
            logContainer.removeChild(logContainer.firstChild);
        }
    }
    
    showLoading(message) {
        document.getElementById('loadingMessage').textContent = message;
        const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
        modal.show();
    }
    
    hideLoading() {
        const modal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
        if (modal) {
            modal.hide();
        }
    }
    
    resetChart() {
        // Clear chart data and reset for new recording
        this.chart.data.labels = [];
        
        // Clear all datasets (for multi-channel support)
        this.chart.data.datasets.forEach(dataset => {
            dataset.data = [];
        });
        
        this.chart.update('none');
    }
}

// Global functions
function showBand(band) {
    if (window.eegInterface) {
        window.eegInterface.showBand(band);
    }
}

function clearLog() {
    const logContainer = document.getElementById('dataLog');
    logContainer.innerHTML = '<div class="log-entry">Log cleared.</div>';
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.eegInterface = new EEGInterface();
    
    // Update status periodically
    setInterval(() => {
        if (window.eegInterface) {
            window.eegInterface.updateStatus();
        }
    }, 5000);
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (window.eegInterface && window.eegInterface.isRecording) {
        window.eegInterface.stopRecording();
    }
}); 