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
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';

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

interface DeviceInfo {
  type: 'none' | 'emulator' | 'real';
  name: string;
  is_emulator: boolean;
}

interface EEGInterfaceProps {
  externalIsRecording?: boolean;
  onRecordingChange?: (isRecording: boolean) => void;
  onStatusChange?: (status: string) => void;
  onFpsChange?: (fps: number) => void;
  onDeviceInfoChange?: (deviceInfo: DeviceInfo) => void;
  onCurrentValuesChange?: (currentValues: {[key: string]: number}) => void;
}

const EEGInterface: React.FC<EEGInterfaceProps> = ({
  externalIsRecording,
  onRecordingChange,
  onStatusChange,
  onFpsChange,
  onDeviceInfoChange,
  onCurrentValuesChange
}) => {
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
  const [deviceInfo, setDeviceInfo] = useState<DeviceInfo>({
    type: 'none',
    name: 'No device connected',
    is_emulator: false
  });
  const updateIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  // Maximum number of data points to display on charts (smaller = faster filling)
  const MAX_DATA_POINTS = 100; // Adjust this value to change time window size
  // Function to check device status
  const checkDeviceStatus = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/status');
      const status = await response.json();
      if (status.device_info) {
        setDeviceInfo(status.device_info);
        onDeviceInfoChange?.(status.device_info);
      }
    } catch (error) {
      console.log('Error checking device status:', error);
    }
  };
  const frameCountRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(Date.now());
  const isRecordingRef = useRef<boolean>(false);
  const animationFrameRef = useRef<number | null>(null);
  const currentDataRef = useRef<EEGResponse | null>(null);

  // Check device status on component mount
  useEffect(() => {
    checkDeviceStatus();
  }, []);

  // Sync with external recording state
  useEffect(() => {
    if (externalIsRecording !== undefined) {
      const wasRecording = isRecordingRef.current;
      setIsRecording(externalIsRecording);
      isRecordingRef.current = externalIsRecording;
      
      if (externalIsRecording && !wasRecording) {
        // External recording started
        console.log('External recording started, initializing charts and starting data loop');
        initializeChart();
        startDataLoop();
      } else if (!externalIsRecording && wasRecording) {
        // External recording stopped
        console.log('External recording stopped, stopping data loop');
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
          animationFrameRef.current = null;
        }
      }
    }
  }, [externalIsRecording]);

  // Update external state when internal state changes
  useEffect(() => {
    onRecordingChange?.(isRecording);
  }, [isRecording, onRecordingChange]);

  useEffect(() => {
    onStatusChange?.(status);
  }, [status, onStatusChange]);

  useEffect(() => {
    onFpsChange?.(fps);
  }, [fps, onFpsChange]);

  useEffect(() => {
    onCurrentValuesChange?.(currentValues);
  }, [currentValues, onCurrentValuesChange]);

  // Channel colors for multi-channel display - minimalistic colors
  const channelColors = [
    '#3b82f6', '#8b5cf6', '#06b6d4', '#10b981', 
    '#f59e0b', '#ef4444', '#ec4899', '#8b5cf6',
    '#06b6d4', '#10b981', '#f59e0b', '#ef4444',
    '#ec4899', '#8b5cf6'
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
            // Limit data points to show smaller time window
            const limitedChannelData = channelData.slice(-MAX_DATA_POINTS);
            datasets.push({
              label: `Channel ${ch + 1}`,
              data: limitedChannelData,
              borderColor: channelColors[ch % channelColors.length],
              backgroundColor: channelColors[ch % channelColors.length] + '20',
              borderWidth: 1,
              fill: false,
              tension: 0.1,
              pointRadius: 0,
              pointHoverRadius: 3,
              stepped: false,
              cubicInterpolationMode: 'monotone' as const
            });
          }

          // Limit timestamps to match the data points
          const limitedTimestamps = bandData.timestamps.slice(-MAX_DATA_POINTS);
          newChartData[band] = {
            labels: limitedTimestamps.map(t => t.toFixed(1)),
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
        
        // Update device information if provided
        if (result.device_info) {
          setDeviceInfo(result.device_info);
          addLog(`Device: ${result.device_info.name} (${result.device_info.is_emulator ? 'Emulator' : 'Real Device'})`);
        }
        
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
          text: 'Time (seconds)',
          font: {
            size: 12,
            weight: 'bold'
          }
        },
        grid: {
          color: 'rgba(148, 163, 184, 0.1)',
          lineWidth: 1
        },
        ticks: {
          maxTicksLimit: 15, // Fewer ticks for better performance
          autoSkip: true,
          autoSkipPadding: 10,
          font: {
            size: 10
          }
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Amplitude (μV)',
          font: {
            size: 12,
            weight: 'bold'
          }
        },
        grid: {
          color: 'rgba(148, 163, 184, 0.1)',
          lineWidth: 1
        },
        ticks: {
          font: {
            size: 10
          }
        }
      }
    },
    plugins: {
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
    <div className="min-h-screen bg-slate-950 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Graph Navigation Labels */}
        <div className="mb-4">
          <div className="flex space-x-1 bg-slate-800/50 rounded-lg p-1">
            {(['raw', 'alpha', 'beta', 'delta', 'theta', 'gamma'] as const).map((band) => (
              <button
                key={band}
                onClick={() => {
                  const element = document.getElementById(`chart-${band}`);
                  if (element) {
                    element.scrollIntoView({ 
                      behavior: 'smooth', 
                      block: 'nearest',
                      inline: 'center'
                    });
                  }
                }}
                className="flex-1 px-4 py-2 text-sm font-medium rounded-md transition-all duration-200 hover:bg-slate-700/50 text-slate-300 hover:text-slate-100"
              >
                {band === 'raw' ? 'Raw EEG' : `${band.toUpperCase()} Band`}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Horizontal Charts Container - Full Width */}
      <div className="w-full overflow-x-auto scrollbar-hide">
        <div className="flex space-x-6 min-w-max px-[10vw]">
          {(['raw', 'alpha', 'beta', 'delta', 'theta', 'gamma'] as const).map((band) => (
            <div key={band} id={`chart-${band}`} className="w-[80vw] flex-shrink-0">
              <Card className="bg-slate-900/50 border-slate-700 h-full">
                <CardHeader className="pb-3">
                  <CardTitle className="text-xl font-medium text-slate-100">
                    {band === 'raw' ? 'Raw EEG' : `${band.toUpperCase()} Band`}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[calc(100vh-320px)] p-4 rounded-lg bg-slate-800/30 border border-slate-700">
                    <Line 
                      data={chartData[band]} 
                      options={{
                        ...chartOptions,
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                          ...chartOptions.plugins,
                          legend: {
                            display: true,
                            position: 'top' as const,
                            labels: {
                              usePointStyle: true,
                              padding: 12,
                              font: {
                                size: 12
                              }
                            }
                          }
                        }
                      }} 
                    />
                  </div>
                </CardContent>
              </Card>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default EEGInterface; 