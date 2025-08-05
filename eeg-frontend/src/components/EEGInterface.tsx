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
  const [chartData, setChartData] = useState<any>({
    labels: [],
    datasets: []
  });
  const [currentValues, setCurrentValues] = useState<{[key: string]: number}>({});
  const [status, setStatus] = useState<string>('Ready');
  const [fps, setFps] = useState<number>(0);
  const updateIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const frameCountRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(Date.now());
  const isRecordingRef = useRef<boolean>(false);
  const animationIdRef = useRef<number | null>(null);

  // Channel colors for multi-channel display
  const channelColors = [
    '#667eea', '#f093fb', '#4facfe', '#43e97b', 
    '#fa709a', '#fee140', '#a8edea', '#fed6e3',
    '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4',
    '#ffeaa7', '#dda0dd'
  ];

  const initializeChart = () => {
    console.log('Initializing chart...');
    const datasets = [{
      label: 'Channel 1',
      data: [],
      borderColor: channelColors[0],
      backgroundColor: channelColors[0] + '20',
      borderWidth: 1.5,
      fill: false,
      tension: 0.1,
      pointRadius: 0,
      pointHoverRadius: 3
    }];

    const initialData = {
      labels: [],
      datasets
    };
    console.log('Setting initial chart data:', initialData);
    setChartData(initialData);
  };

  const updateChart = (data: EEGData) => {
    console.log('updateChart called with data:', data);
    if (!data.values || data.values.length === 0) {
      console.log('No values in data');
      return;
    }

    const numChannels = data.channels;
    console.log('Number of channels:', numChannels);
    const datasets = [];

    // Create datasets for each channel
    for (let ch = 0; ch < numChannels; ch++) {
      const channelData = data.values[ch] || [];
      console.log(`Channel ${ch + 1} data length:`, channelData.length);
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
        stepped: false, // Smooth lines
        cubicInterpolationMode: 'monotone' // Better curve interpolation
      });
    }

    const newChartData = {
      labels: data.timestamps.map(t => t.toFixed(1)),
      datasets
    };
    console.log('Setting chart data:', newChartData);
    console.log('Chart data labels length:', newChartData.labels.length);
    console.log('Chart data datasets length:', newChartData.datasets.length);
    console.log('First few labels:', newChartData.labels.slice(0, 5));
    console.log('First dataset first few values:', newChartData.datasets[0]?.data?.slice(0, 5));
    setChartData(newChartData);
  };

  const updateData = async () => {
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

      // Update chart if current band has data
      const currentBandData = data[currentBand];
      console.log('Current band data:', currentBandData);
      if (currentBandData && currentBandData.values.length > 0) {
        console.log('Updating chart with data');
        updateChart(currentBandData);
      } else {
        console.log('No data for current band or no values');
        console.log('Available bands:', Object.keys(data));
        console.log('Current band:', currentBand);
        
        // If no data and we're supposed to be recording, something is wrong
        if (isRecordingRef.current && Object.keys(data).length === 0) {
          console.log('Recording but no data available, stopping data loop');
          setIsRecording(false);
          isRecordingRef.current = false;
          if (animationIdRef.current) {
            cancelAnimationFrame(animationIdRef.current);
            animationIdRef.current = null;
          }
        }
      }

    } catch (error) {
      console.error('Error updating data:', error);
    }
  };

  const startDataLoop = () => {
    console.log('Starting data loop with requestAnimationFrame');
    let frameCount = 0;
    
    const loop = () => {
      if (isRecordingRef.current) {
        frameCount++;
        if (frameCount % 60 === 0) { // Log every 60 frames (about once per second)
          console.log(`Data loop frame ${frameCount}`);
        }
        updateData();
        animationIdRef.current = requestAnimationFrame(loop);
      } else {
        console.log('Data loop stopped - recording inactive');
        if (animationIdRef.current) {
          cancelAnimationFrame(animationIdRef.current);
          animationIdRef.current = null;
        }
      }
    };
    animationIdRef.current = requestAnimationFrame(loop);
  };

  const startRecording = async () => {
    try {
      console.log('Starting recording...');
      setStatus('Starting recording...');
      
      // Reset any previous state
      setIsRecording(false);
      isRecordingRef.current = false;
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
        animationIdRef.current = null;
      }
      
      const response = await fetch('http://localhost:5000/api/start_recording', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const result = await response.json();
      console.log('Start recording result:', result);
      
      if (result.status === 'success') {
        setIsRecording(true);
        isRecordingRef.current = true;
        setStatus('Recording active');
        initializeChart();
        console.log('Starting data loop...');
        
        // Start high-frequency data loop using requestAnimationFrame
        startDataLoop();
      } else {
        setStatus(`Error: ${result.message}`);
      }
    } catch (error) {
      console.error('Error starting recording:', error);
      setStatus('Error starting recording');
    }
  };

  const stopRecording = async () => {
    try {
      console.log('🛑 Frontend: Stopping recording...');
      setStatus('Stopping recording...');
      
      // Immediately stop the data loop
      setIsRecording(false);
      isRecordingRef.current = false;
      
      // Immediately cancel the animation frame to stop the data loop
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
        animationIdRef.current = null;
        console.log('🛑 Frontend: Animation frame cancelled');
      }
      
      // Quick health check first
      try {
        const healthResponse = await fetch('http://localhost:5000/api/test', { 
          signal: AbortSignal.timeout(1000) // 1 second timeout
        });
        if (!healthResponse.ok) {
          throw new Error('Server health check failed');
        }
        console.log('🛑 Frontend: Server health check passed');
      } catch (healthError) {
        console.log('🛑 Frontend: Server health check failed:', healthError);
        setStatus('Server not responding - recording stopped locally');
        return;
      }
      
      // Create a timeout for the fetch request
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000); // 3 second timeout
      
      console.log('🛑 Frontend: Making stop request to API...');
      const response = await fetch('http://localhost:5000/api/stop_recording', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        signal: controller.signal
      });

      clearTimeout(timeoutId);
      console.log('🛑 Frontend: Stop request completed, status:', response.status);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      console.log('🛑 Frontend: Stop response:', result);
      
      if (result.status === 'success') {
        setStatus('Recording stopped');
        console.log('🛑 Frontend: Recording stopped successfully');
        
        // Stop polling (requestAnimationFrame stops automatically when isRecording becomes false)
        if (updateIntervalRef.current) {
          clearInterval(updateIntervalRef.current);
          updateIntervalRef.current = null;
        }
      } else {
        setStatus(`Error: ${result.message}`);
      }
    } catch (error) {
      console.error('Error stopping recording:', error);
      if (error instanceof Error && error.name === 'AbortError') {
        setStatus('Stop request timed out, but recording stopped locally');
      } else if (error instanceof Error && error.message.includes('Failed to fetch')) {
        setStatus('Cannot connect to server - recording stopped locally');
        console.log('🛑 Frontend: Server connection failed, but recording stopped locally');
      } else {
        setStatus('Error stopping recording');
      }
    }
  };

  const changeBand = (band: typeof currentBand) => {
    setCurrentBand(band);
  };

  useEffect(() => {
    // Cleanup on unmount
    return () => {
      if (updateIntervalRef.current) {
        clearInterval(updateIntervalRef.current);
      }
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
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
            <button 
              onClick={updateData}
              className="px-4 py-2 rounded-lg font-medium transition-colors bg-green-500 hover:bg-green-600 text-white"
            >
              Test Data Fetch
            </button>

            {/* Band Selection */}
            <div className="flex flex-wrap gap-2">
              {(['raw', 'alpha', 'beta', 'delta', 'theta', 'gamma'] as const).map((band) => (
                <button
                  key={band}
                  onClick={() => changeBand(band)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    currentBand === band
                      ? 'bg-blue-100 text-blue-800 border-2 border-blue-300'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {band.toUpperCase()}
                </button>
              ))}
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

        {/* Chart */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            {currentBand === 'raw' ? 'Multi-Channel Raw EEG' : `${currentBand.toUpperCase()} Band`}
          </h2>
                  <div className="h-96">
          <div className="text-sm text-gray-500 mb-2">
            Chart data: {chartData.labels.length} labels, {chartData.datasets.length} datasets
            {chartData.labels.length > 0 && (
              <span> | First label: {chartData.labels[0]} | Last label: {chartData.labels[chartData.labels.length - 1]}</span>
            )}
          </div>
          <Line data={chartData} options={chartOptions} />
        </div>
        </div>
      </div>
    </div>
  );
};

export default EEGInterface; 