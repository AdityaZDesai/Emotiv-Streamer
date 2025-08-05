'use client';

import React, { useEffect, useRef, useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions,
  ChartData,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface EEGData {
  values: number[][];
  timestamps: number[];
  channels: number;
}

interface EEGResponse {
  raw?: EEGData;
  alpha?: EEGData;
  beta?: EEGData;
  delta?: EEGData;
  theta?: EEGData;
  gamma?: EEGData;
}

const EEGInterface: React.FC = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [currentBand, setCurrentBand] = useState<'raw' | 'alpha' | 'beta' | 'delta' | 'theta' | 'gamma'>('raw');
  
  // Debug: Log initial state
  console.log('Initial currentBand:', currentBand);
  const [chartData, setChartData] = useState<{
    [key: string]: ChartData<'line'>
  }>({
    raw: { labels: [], datasets: [] },
    alpha: { labels: [], datasets: [] },
    beta: { labels: [], datasets: [] },
    delta: { labels: [], datasets: [] },
    theta: { labels: [], datasets: [] },
    gamma: { labels: [], datasets: [] }
  });
  const [currentValues, setCurrentValues] = useState<{[key: string]: number}>({});
  const [status, setStatus] = useState<string>('Ready');
  const [fps, setFps] = useState<number>(0);
  const [logs, setLogs] = useState<string[]>([]);
  const updateIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const frameCountRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(Date.now());
  const isRecordingRef = useRef<boolean>(false);
  const animationFrameRef = useRef<number | null>(null);
  const currentDataRef = useRef<EEGResponse | null>(null);

  // Channel colors for multi-channel display
  const channelColors = [
    '#667eea', '#f093fb', '#4facfe', '#43e97b', 
    '#fa709a', '#fee140', '#a8edea', '#fed6e3',
    '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4',
    '#ffeaa7', '#dda0dd'
  ];

  const initializeChart = () => {
    addLog('Initializing charts with empty data...');
    const initialChartData = {
      raw: { labels: [], datasets: [] },
      alpha: { labels: [], datasets: [] },
      beta: { labels: [], datasets: [] },
      delta: { labels: [], datasets: [] },
      theta: { labels: [], datasets: [] },
      gamma: { labels: [], datasets: [] }
    };
    addLog('Setting initial chart data to empty');
    setChartData(initialChartData);
  };

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = `[${timestamp}] ${message}`;
    setLogs(prev => [...prev.slice(-99), logEntry]); // Keep last 100 logs
    console.log(message);
  };

  const updateAllCharts = (data: EEGResponse) => {
    addLog('updateAllCharts called with data');
    
    const newChartData = { ...chartData };
    
    // Update each band's chart
    const bands = ['raw', 'alpha', 'beta', 'delta', 'theta', 'gamma'] as const;
    
    // Check if this is the first time we're getting real filtered data (different from raw)
    let hasRealFilteredData = false;
    let rawFirstValue = null;
    
    // Get the first value from raw data for comparison
    if (data.raw && data.raw.values && data.raw.values.length > 0) {
      const rawFirstChannel = data.raw.values[0] || [];
      if (rawFirstChannel.length > 0) {
        rawFirstValue = rawFirstChannel[0];
      }
    }
    
    // Check if any filtered band has different values from raw
    bands.forEach(band => {
      if (band !== 'raw') {
        const bandData = data[band];
        if (bandData && bandData.values && bandData.values.length > 0) {
          const firstChannelData = bandData.values[0] || [];
          if (firstChannelData.length > 0) {
            const filteredFirstValue = firstChannelData[0];
            // If the filtered value is different from raw, we have real filtering
            if (rawFirstValue !== null && Math.abs(filteredFirstValue - rawFirstValue) > 0.001) {
              hasRealFilteredData = true;
              addLog(`🎯 Real filtering detected! ${band}: raw=${rawFirstValue.toFixed(3)}, filtered=${filteredFirstValue.toFixed(3)}`);
            }
          }
        }
      }
    });
    
    // If we have real filtered data for the first time, clear all filtered charts
    if (hasRealFilteredData) {
      addLog('🎉 First real filtered data detected! Clearing old filtered charts...');
      bands.forEach(band => {
        if (band !== 'raw') {
          newChartData[band] = {
            labels: [],
            datasets: []
          };
        }
      });
    }
    
    bands.forEach(band => {
      const bandData = data[band];
      if (bandData && bandData.values && bandData.values.length > 0) {
        // Check if this is empty data (for filtered bands)
        const firstChannelData = bandData.values[0] || [];
        if (band === 'raw' || firstChannelData.length > 0) {
          addLog(`Updating ${band} chart with ${bandData.channels} channels`);
          
          const datasets = [];
          // Create datasets for each channel
          for (let ch = 0; ch < bandData.channels; ch++) {
            const channelData = bandData.values[ch] || [];
            datasets.push({
              label: `Channel ${ch + 1}`,
              data: channelData,
              borderColor: channelColors[ch % channelColors.length],
              backgroundColor: channelColors[ch % channelColors.length] + '20',
              borderWidth: 1.5,
              fill: false,
              tension: 0.1,
              pointRadius: 0,
              pointHoverRadius: 3,
              stepped: false,
              cubicInterpolationMode: 'monotone' as const
            });
          }

          newChartData[band] = {
            labels: bandData.timestamps.map(t => t.toFixed(1)),
            datasets
          };
        } else {
          addLog(`Skipping ${band} chart - no data yet`);
        }
      }
    });
    
    setChartData(newChartData);
  };

  const updateData = async () => {
    // Don't fetch data if not recording
    if (!isRecordingRef.current) {
      console.log('Not recording, skipping data fetch');
      return;
    }
    
    try {
      console.log('Fetching data from API...');
      const response = await fetch('http://localhost:5000/api/data');
      console.log('Response status:', response.status);
      console.log('Response headers:', response.headers);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data: EEGResponse = await response.json();
      console.log('Received data keys:', Object.keys(data));
      console.log('Received data:', data);
      
      // Store current data for band switching
      currentDataRef.current = data;
      console.log(`💾 STORED DATA DEBUG: Stored data with keys:`, Object.keys(data));
      console.log(`  Data structure check:`, {
        raw: data.raw ? 'exists' : 'missing',
        alpha: data.alpha ? 'exists' : 'missing',
        beta: data.beta ? 'exists' : 'missing',
        delta: data.delta ? 'exists' : 'missing',
        theta: data.theta ? 'exists' : 'missing',
        gamma: data.gamma ? 'exists' : 'missing'
      });
      
      // Debug: Compare data between bands
      if (Object.keys(data).length > 1) {
        console.log('🔍 BAND COMPARISON DEBUG:');
        const bands = ['raw', 'alpha', 'beta', 'delta', 'theta', 'gamma'];
        for (let i = 0; i < bands.length - 1; i++) {
          const band1 = bands[i];
          const band2 = bands[i + 1];
          const data1 = data[band1 as keyof EEGResponse];
          const data2 = data[band2 as keyof EEGResponse];
          
          if (data1 && data2 && data1.values.length > 0 && data2.values.length > 0) {
            const val1 = data1.values[0][0]; // First value of first channel
            const val2 = data2.values[0][0]; // First value of first channel
            console.log(`  ${band1} vs ${band2}: ${val1} vs ${val2} (diff: ${Math.abs(val1 - val2).toFixed(3)})`);
          }
        }
      }

      // Update FPS counter
      frameCountRef.current++;
      const currentTime = Date.now();
      if (currentTime - lastTimeRef.current >= 1000) {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
        lastTimeRef.current = currentTime;
      }

      // Update current values display
      Object.keys(data).forEach(band => {
        const bandData = data[band as keyof EEGResponse];
        console.log(`Processing band ${band}:`, bandData);
        if (bandData && bandData.values && bandData.values.length > 0) {
          console.log(`Band ${band} has ${bandData.values.length} channels`);
          console.log(`First channel has ${bandData.values[0]?.length || 0} values`);
          
          const latestValues = bandData.values.map(channel => 
            channel[channel.length - 1] || 0
          );
          console.log(`Latest values for ${band}:`, latestValues);
          const avgValue = latestValues.reduce((sum, val) => sum + val, 0) / latestValues.length;
          console.log(`Average value for ${band}:`, avgValue);
          setCurrentValues(prev => ({ ...prev, [band]: avgValue }));
        } else {
          console.log(`No data for band ${band}`);
        }
      });

      // Update all charts with the new data
      addLog('Updating all charts with data');
      updateAllCharts(data);

    } catch (error) {
      console.error('Error updating data:', error);
    }
  };

  const startDataLoop = () => {
    // Cancel any existing animation frame first
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    
    console.log('Starting data loop with requestAnimationFrame');
    let frameCount = 0;
    const loop = () => {
      if (isRecordingRef.current) {
        frameCount++;
        if (frameCount % 60 === 0) { // Log every 60 frames (about once per second)
          console.log(`Data loop frame ${frameCount}`);
        }
        updateData();
        animationFrameRef.current = requestAnimationFrame(loop);
      } else {
        console.log('Data loop stopped - recording inactive');
        animationFrameRef.current = null;
      }
    };
    animationFrameRef.current = requestAnimationFrame(loop);
  };

  const startRecording = async () => {
    try {
      setLogs([]); // Clear logs when starting new recording
      addLog('Starting recording...');
      setStatus('Starting recording...');
      const response = await fetch('http://localhost:5000/api/start_recording', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const result = await response.json();
      addLog(`Start recording result: ${result.status} - ${result.message}`);
      
      if (result.status === 'success') {
        setIsRecording(true);
        isRecordingRef.current = true;
        setStatus('Recording active');
        initializeChart();
        addLog('Starting data loop...');
        
        // Start high-frequency data loop using requestAnimationFrame
        // Only start if not already running
        if (!animationFrameRef.current) {
          startDataLoop();
        }
      } else {
        setStatus(`Error: ${result.message}`);
      }
    } catch (error) {
      addLog(`Error starting recording: ${error}`);
      setStatus('Error starting recording');
    }
  };

  const stopRecording = async () => {
    try {
      addLog('Stopping recording...');
      setStatus('Stopping recording...');
      const response = await fetch('http://localhost:5000/api/stop_recording', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const result = await response.json();
      addLog(`Stop recording result: ${result.status} - ${result.message}`);
      
      if (result.status === 'success') {
        setIsRecording(false);
        isRecordingRef.current = false;
        setStatus('Recording stopped');
        
        // Stop the animation frame loop
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
          animationFrameRef.current = null;
          console.log('Animation frame loop stopped');
        }
        
        // Stop polling (requestAnimationFrame stops automatically when isRecording becomes false)
        if (updateIntervalRef.current) {
          clearInterval(updateIntervalRef.current);
          updateIntervalRef.current = null;
        }
      } else {
        setStatus(`Error: ${result.message}`);
      }
    } catch (error) {
      addLog(`Error stopping recording: ${error}`);
      setStatus('Error stopping recording');
    }
  };

  const changeBand = (band: typeof currentBand) => {
    console.log(`🎯 CHANGE BAND CLICKED: ${band}`);
    console.log(`Current band before change: ${currentBand}`);
    console.log(`Current data ref exists: ${!!currentDataRef.current}`);
    
    setCurrentBand(band);
    
    // Immediately update chart with data for the new band using stored data
    if (currentDataRef.current) {
      console.log(`🔄 BAND CHANGE DEBUG: Switching to ${band}`);
      console.log(`  Available bands:`, Object.keys(currentDataRef.current));
      console.log(`  Full current data:`, currentDataRef.current);
      
      const newBandData = currentDataRef.current[band];
      console.log(`  New band data for ${band}:`, newBandData);
      
                    if (newBandData && newBandData.values.length > 0) {
        addLog(`✅ Band ${band} has data, updating all charts`);
        addLog(`Data structure: channels=${newBandData.channels}, timestamps=${newBandData.timestamps.length}, values=${newBandData.values.length}`);
        updateAllCharts(currentDataRef.current!);
      } else {
        addLog(`❌ No data available for band ${band}`);
      }
    } else {
      console.log('❌ No current data available for band change');
    }
  };

  useEffect(() => {
    // Cleanup interval and animation frame on unmount
    return () => {
      if (updateIntervalRef.current) {
        clearInterval(updateIntervalRef.current);
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  const chartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 0
    },
    elements: {
      point: {
        radius: 0, // Hide points for better performance
        hoverRadius: 3
      },
      line: {
        tension: 0.1 // Smooth line curves
      }
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
          maxTicksLimit: 15, // Fewer ticks for better performance
          autoSkip: true,
          autoSkipPadding: 10
        }
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
        display: currentBand === 'raw',
        position: 'top' as const,
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
        intersect: false,
        enabled: false // Disable tooltips for better performance
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            EEG Recording Interface
          </h1>
          <p className="text-gray-600">
            Real-time EEG data visualization with multi-channel support
          </p>
          <div className="mt-4 flex items-center space-x-4">
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${
              isRecording 
                ? 'bg-green-100 text-green-800' 
                : 'bg-gray-100 text-gray-800'
            }`}>
              {isRecording ? 'Recording' : 'Stopped'}
            </div>
            <div className="text-sm text-gray-600">
              Status: {status}
            </div>
            <div className="text-sm text-gray-600">
              FPS: {fps}
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <div className="flex flex-wrap gap-4 items-center">
            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={`px-6 py-3 rounded-lg font-medium transition-colors ${
                isRecording
                  ? 'bg-red-500 hover:bg-red-600 text-white'
                  : 'bg-blue-500 hover:bg-blue-600 text-white'
              }`}
            >
              {isRecording ? 'Stop Recording' : 'Start Recording'}
            </button>


            {/* All bands are now displayed simultaneously */}
            <div className="text-sm text-gray-600">
              All 6 frequency bands are displayed in the charts below
            </div>
          </div>
        </div>

        {/* Current Values */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Current Values</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {(['raw', 'alpha', 'beta', 'delta', 'theta', 'gamma'] as const).map((band) => (
              <div key={band} className="text-center">
                <div className="text-sm font-medium text-gray-600 mb-1">
                  {band.toUpperCase()}
                </div>
                <div className="text-2xl font-bold text-gray-900">
                  {currentValues[band]?.toFixed(2) || '0.00'}
                </div>
                <div className="text-xs text-gray-500">μV</div>
              </div>
            ))}
          </div>
        </div>

        {/* Charts Stack */}
        <div className="space-y-6">
          {(['raw', 'alpha', 'beta', 'delta', 'theta', 'gamma'] as const).map((band) => (
            <div key={band} className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                {band === 'raw' ? 'Multi-Channel Raw EEG' : `${band.toUpperCase()} Band`}
              </h2>
              <div className="h-96">
                <Line 
                  data={chartData[band]} 
                  options={{
                    ...chartOptions,
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      ...chartOptions.plugins,
                      legend: {
                        display: true, // Show legend to see channel colors
                        position: 'top' as const,
                        labels: {
                          usePointStyle: true,
                          padding: 10,
                          font: {
                            size: 11
                          }
                        }
                      }
                    }
                  }} 
                />
              </div>
            </div>
          ))}
        </div>

        {/* Logs Section */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold text-gray-900">System Logs</h2>
            <button
              onClick={() => setLogs([])}
              className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-md transition-colors"
            >
              Clear Logs
            </button>
          </div>
          <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm h-64 overflow-y-auto">
            {logs.length === 0 ? (
              <div className="text-gray-500">No logs yet. Start recording to see activity...</div>
            ) : (
              logs.map((log, index) => (
                <div key={index} className="mb-1">
                  {log}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default EEGInterface; 