'use client';

import { useState } from 'react';
import EEGInterface from '@/components/EEGInterface';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

export default function Home() {
  const [isRecording, setIsRecording] = useState(false);
  const [status, setStatus] = useState('Ready');
  const [fps, setFps] = useState(0);
  const [deviceInfo, setDeviceInfo] = useState({
    type: 'none' as 'none' | 'emulator' | 'real',
    name: 'No device connected',
    is_emulator: false
  });

  const startRecording = async () => {
    try {
      setStatus('Starting recording...');
      const response = await fetch('http://localhost:5000/api/start_recording', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const result = await response.json();
      
      if (result.status === 'success') {
        setIsRecording(true);
        setStatus('Recording active');
        
        if (result.device_info) {
          setDeviceInfo(result.device_info);
        }
      } else {
        setStatus(`Error: ${result.message}`);
      }
    } catch (error) {
      setStatus('Error starting recording');
    }
  };

  const stopRecording = async () => {
    try {
      setStatus('Stopping recording...');
      const response = await fetch('http://localhost:5000/api/stop_recording', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const result = await response.json();
      
      if (result.status === 'success') {
        setIsRecording(false);
        setStatus('Recording stopped');
      } else {
        setStatus(`Error: ${result.message}`);
      }
    } catch (error) {
      setStatus('Error stopping recording');
    }
  };

  return (
    <div className="bg-slate-950">
      {/* Navigation Bar - Always at Top */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-slate-900/80 backdrop-blur-sm border-b border-slate-700/50">
        <div className="max-w-7xl mx-auto px-6 py-3">
          <div className="flex items-center justify-between">
            {/* Left Side - Title and Status */}
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-3">
                <h2 className="text-lg font-medium text-slate-100">Neural Interface</h2>
                <span className="text-sm text-slate-400">Real-time EEG data visualization</span>
              </div>
              
              <div className="flex items-center space-x-3">
                <Badge variant={isRecording ? "destructive" : "secondary"} className="text-xs">
                  {isRecording ? 'Recording' : 'Stopped'}
                </Badge>
                <Badge variant="outline" className="text-xs border-slate-600 text-slate-300">
                  {status}
                </Badge>
                <Badge variant="outline" className="text-xs border-slate-600 text-slate-300">
                  {fps} FPS
                </Badge>
                <Badge variant="outline" className={`text-xs ${
                  deviceInfo.is_emulator 
                    ? 'border-amber-600 text-amber-300' 
                    : deviceInfo.type === 'real'
                    ? 'border-blue-600 text-blue-300'
                    : 'border-slate-600 text-slate-300'
                }`}>
                  {deviceInfo.is_emulator ? 'Emulator' : deviceInfo.type === 'real' ? 'Real Device' : 'No Device'}
                </Badge>
              </div>
            </div>

            {/* Right Side - Recording Button */}
            <div className="flex items-center space-x-4">
              <Button
                onClick={isRecording ? stopRecording : startRecording}
                className={`px-4 py-2 text-sm font-medium transition-colors ${
                  isRecording
                    ? 'bg-red-600 hover:bg-red-700 text-white'
                    : 'bg-blue-600 hover:bg-blue-700 text-white'
                }`}
              >
                {isRecording ? 'Stop Recording' : 'Start Recording'}
              </Button>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section - Full Screen */}
      <div className="h-screen relative overflow-hidden flex items-center justify-center">
        {/* Background Pattern */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#4f4f4f2e_1px,transparent_1px),linear-gradient(to_bottom,#4f4f4f2e_1px,transparent_1px)] bg-[size:14px_24px]"></div>
        
        {/* Hero Content */}
        <div className="relative z-10 max-w-7xl mx-auto px-6 text-center">
          <Badge className="mb-6 bg-blue-600/20 text-blue-300 border-blue-500/30">
            Real-time Brain-Computer Interface
          </Badge>
          
          <h1 className="text-5xl md:text-7xl font-bold text-slate-100 mb-6">
            Neural
            <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent"> Interface</span>
          </h1>
          
          <p className="text-xl text-slate-400 mb-8 max-w-3xl mx-auto">
            Advanced EEG data visualization and analysis platform for real-time brain wave monitoring and research applications.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-16">
            <Button 
              onClick={() => document.getElementById('eeg-interface')?.scrollIntoView({ behavior: 'smooth' })}
              className="px-8 py-4 text-lg bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white"
            >
              Start Monitoring
            </Button>
            <Button 
              variant="outline" 
              className="px-8 py-4 text-lg border-slate-600 text-slate-300 hover:bg-slate-800"
            >
              Learn More
            </Button>
          </div>

          {/* Stats Section */}
          <div className="grid md:grid-cols-4 gap-8 max-w-4xl mx-auto">
            <div>
              <div className="text-3xl font-bold text-slate-100 mb-2">6</div>
              <div className="text-slate-400">Frequency Bands</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-slate-100 mb-2">14+</div>
              <div className="text-slate-400">EEG Channels</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-slate-100 mb-2">60+</div>
              <div className="text-slate-400">FPS Update Rate</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-slate-100 mb-2">Real-time</div>
              <div className="text-slate-400">Data Processing</div>
            </div>
          </div>
        </div>
      </div>

      {/* EEG Interface Section - Sticky */}
      <div id="eeg-interface" className="sticky top-0 z-30">
        <div className="absolute top-6 left-6 z-40">
          <Button 
            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
            variant="outline"
            className="bg-slate-900/80 border-slate-600 text-slate-300 hover:bg-slate-800"
          >
            ↑ Back to Top
          </Button>
        </div>
        <EEGInterface 
          externalIsRecording={isRecording}
          onRecordingChange={setIsRecording}
          onStatusChange={setStatus}
          onFpsChange={setFps}
          onDeviceInfoChange={setDeviceInfo}
        />
      </div>
    </div>
  );
}
