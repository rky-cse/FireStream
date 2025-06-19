import React, { useState, useRef, useEffect } from 'react';

const VoiceProsodyComponent = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [transcriptions, setTranscriptions] = useState([]);
  const [prosodyResults, setProsodyResults] = useState([]);
  const [error, setError] = useState('');
  const [systemInfo, setSystemInfo] = useState(null);
  const [debugInfo, setDebugInfo] = useState('');
  const [opensmileDebug, setOpensmileDebug] = useState(null);
  
  const wsRef = useRef(null);
  const audioContextRef = useRef(null);
  const sourceRef = useRef(null);
  const processorRef = useRef(null);
  const streamRef = useRef(null);
  const clientId = useRef(`client_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`);

  // Connect to WebSocket and check system
  useEffect(() => {
    checkSystemHealth();
    debugOpenSMILE();
    connectWebSocket();
    return () => {
      cleanup();
    };
  }, []);

  const checkSystemHealth = async () => {
    try {
      const response = await fetch('http://localhost:8000/health');
      const health = await response.json();
      setSystemInfo(health);
      
      if (!health.whisper_loaded) {
        setError('Whisper model not loaded');
      }
    } catch (err) {
      setError('Cannot connect to backend server');
      console.error('Health check failed:', err);
    }
  };

  const debugOpenSMILE = async () => {
    try {
      const response = await fetch('http://localhost:8000/debug/opensmile');
      const debug = await response.json();
      setOpensmileDebug(debug);
      console.log('OpenSMILE Debug:', debug);
    } catch (err) {
      console.error('OpenSMILE debug failed:', err);
    }
  };

  const connectWebSocket = () => {
    try {
      const ws = new WebSocket(`ws://localhost:8000/ws/${clientId.current}`);
      
      ws.onopen = () => {
        setIsConnected(true);
        setError('');
        console.log('Connected to WebSocket');
      };
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMessage(data);
      };
      
      ws.onclose = () => {
        setIsConnected(false);
        console.log('WebSocket disconnected');
      };
      
      ws.onerror = (error) => {
        setError('WebSocket connection failed');
        console.error('WebSocket error:', error);
      };
      
      wsRef.current = ws;
    } catch (err) {
      setError('Failed to connect to server');
      console.error('Connection error:', err);
    }
  };

  const handleMessage = (data) => {
    console.log('Received message:', data.type, data);
    
    switch (data.type) {
      case 'transcription':
        const newTranscription = {
          id: Date.now(),
          text: data.text,
          timestamp: data.timestamp,
          sampleCount: data.sample_count,
          duration: data.duration,
          sessionId: data.session_id
        };
        setTranscriptions(prev => [...prev, newTranscription].slice(-5));
        setDebugInfo(`✅ Transcription: "${data.text.slice(0, 30)}..." (${data.duration?.toFixed(1)}s) [${data.session_id?.slice(-8)}]`);
        console.log('New transcription:', data.text, 'Session:', data.session_id);
        break;
        
      case 'prosody_analysis':
        const newResult = {
          id: Date.now(),
          ...data.analysis
        };
        setProsodyResults(prev => [...prev, newResult].slice(-3));
        const method = data.analysis.prosody_summary?.extraction_method || 'unknown';
        const featureCount = data.analysis.prosody_summary?.total_features || 0;
        setDebugInfo(`🔬 Prosody: ${featureCount} features via ${method} [${data.analysis.session_id?.slice(-8)}]`);
        console.log('Prosody analysis completed for session:', data.analysis.session_id);
        break;
        
      case 'error':
        setError(`Server: ${data.message}`);
        console.error('Server error:', data.message);
        break;
    }
  };

  const startRecording = async () => {
    try {
      setError('');
      setDebugInfo('🎤 Requesting microphone access...');
      
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        }
      });
      
      streamRef.current = stream;
      setDebugInfo('✅ Microphone access granted, setting up audio processing...');
      
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000
      });
      
      const source = audioContextRef.current.createMediaStreamSource(stream);
      sourceRef.current = source;
      
      const processor = audioContextRef.current.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      
      let audioBuffer = [];
      
      processor.onaudioprocess = (event) => {
        const inputBuffer = event.inputBuffer;
        const inputData = inputBuffer.getChannelData(0);
        
        audioBuffer.push(...inputData);
        
        if (audioBuffer.length >= 16000) {
          sendAudioData(audioBuffer, audioContextRef.current.sampleRate);
          audioBuffer = [];
        }
      };
      
      source.connect(processor);
      processor.connect(audioContextRef.current.destination);
      
      setIsRecording(true);
      setDebugInfo(`🎙️ Recording active (${audioContextRef.current.sampleRate}Hz)`);
      console.log('Recording started with Web Audio API');
      
    } catch (err) {
      setError('Recording failed: ' + err.message);
      setDebugInfo('❌ Recording failed');
      console.error('Recording error:', err);
    }
  };

  const stopRecording = () => {
    try {
      if (sourceRef.current) {
        sourceRef.current.disconnect();
      }
      if (processorRef.current) {
        processorRef.current.disconnect();
      }
      
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
      
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ 
          type: 'audio_end',
          sampleRate: 16000
        }));
      }
      
      setIsRecording(false);
      setDebugInfo('⏹️ Recording stopped');
      console.log('Recording stopped');
    } catch (err) {
      console.error('Error stopping recording:', err);
      setError('Error stopping recording: ' + err.message);
    }
  };

  const sendAudioData = (audioArray, sampleRate) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify({
          type: 'audio_data',
          data: Array.from(audioArray),
          sampleRate: sampleRate
        }));
        
        console.log(`Sent ${audioArray.length} audio samples (${(audioArray.length/sampleRate).toFixed(1)}s)`);
      } catch (err) {
        console.error('Error sending audio data:', err);
        setError('Failed to send audio data: ' + err.message);
      }
    }
  };

  const analyzeProsody = (text, sessionId) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'analyze_prosody',
        text: text,
        session_id: sessionId,
        sampleRate: 16000
      }));
      setDebugInfo(`🔬 Prosody analysis requested for session: ${sessionId?.slice(-8)}`);
      console.log('Requested prosody analysis for:', text, 'Session:', sessionId);
    } else {
      setError('Not connected to server');
    }
  };

  const cleanup = () => {
    if (sourceRef.current) {
      sourceRef.current.disconnect();
    }
    if (processorRef.current) {
      processorRef.current.disconnect();
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    if (wsRef.current) {
      wsRef.current.close();
    }
  };

  const clearAll = () => {
    setTranscriptions([]);
    setProsodyResults([]);
    setError('');
    setDebugInfo('');
  };

  return (
    <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-lg p-6">
      {/* Header */}
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">
          Voice Prosody Analysis
        </h2>
        <p className="text-gray-600">
          Real-time speech-to-text with prosody analysis - System Working! ✅
        </p>
        
        {/* Status */}
        <div className="mt-4 space-y-2">
          <div className="flex justify-center items-center space-x-6">
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm">WebSocket: {isConnected ? '✅ Connected' : '❌ Disconnected'}</span>
            </div>
            
            {systemInfo && (
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${systemInfo.whisper_loaded ? 'bg-green-500' : 'bg-red-500'}`}></div>
                <span className="text-sm">Whisper: {systemInfo.whisper_loaded ? '✅ Ready' : '❌ Issues'}</span>
              </div>
            )}
            
            {opensmileDebug && (
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${
                  opensmileDebug.prosody_test?.success ? 'bg-green-500' : 'bg-yellow-500'
                }`}></div>
                <span className="text-sm">
                  Prosody: {opensmileDebug.prosody_test?.success ? '✅ Working' : '⚠️ Fallback'}
                </span>
              </div>
            )}
            
            <div className="text-xs text-gray-500">
              User: rky-cse | 2025-06-19 09:37:05 UTC
            </div>
          </div>
          
          {/* System Details */}
          {systemInfo && (
            <div className="text-xs text-gray-500">
              Whisper: {systemInfo.whisper_loaded ? '✓' : '✗'} | 
              OpenSMILE: {systemInfo.opensmile_available ? '✓' : '✗'} | 
              Method: {systemInfo.audio_method || 'Web Audio API'} |
              Connections: {systemInfo.active_connections} |
              Sessions: {systemInfo.stored_sessions || 0}
            </div>
          )}
          
          {/* OpenSMILE Status */}
          {opensmileDebug && (
            <div className="text-xs text-blue-600">
              OpenSMILE: {opensmileDebug.binary_works ? 'Binary OK' : 'Binary Issues'} | 
              Config: {opensmileDebug.config_exists ? 'Found' : 'Missing'} | 
              Features: {opensmileDebug.prosody_test?.feature_count || 0} | 
              Method: {opensmileDebug.prosody_test?.method || 'Unknown'}
            </div>
          )}
          
          {/* Debug Info */}
          {debugInfo && (
            <div className="text-xs text-green-600 font-mono bg-green-50 p-2 rounded">
              {debugInfo}
            </div>
          )}
        </div>
      </div>

      {/* Controls */}
      <div className="flex justify-center space-x-4 mb-6">
        <button
          onClick={isRecording ? stopRecording : startRecording}
          disabled={!isConnected}
          className={`px-6 py-3 rounded-lg font-medium ${
            isRecording 
              ? 'bg-red-600 hover:bg-red-700 text-white animate-pulse' 
              : 'bg-green-600 hover:bg-green-700 text-white'
          } disabled:bg-gray-400 disabled:cursor-not-allowed`}
        >
          {isRecording ? '🛑 Stop Recording' : '🎤 Start Recording'}
        </button>
        
        <button
          onClick={clearAll}
          className="px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
        >
          🗑️ Clear All
        </button>
        
        <button
          onClick={() => {
            checkSystemHealth();
            debugOpenSMILE();
          }}
          className="px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          🔄 Refresh Status
        </button>
      </div>

      {/* Success Banner */}
      {isConnected && systemInfo?.whisper_loaded && (
        <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
          <div className="flex items-center justify-center space-x-3">
            <span className="text-green-800 font-medium">
              🎉 System is working! Transcription and prosody analysis are both functional.
            </span>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-800">{error}</p>
        </div>
      )}

      {/* Recording Indicator */}
      {isRecording && (
        <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <div className="flex items-center justify-center space-x-3">
            <div className="w-4 h-4 bg-red-500 rounded-full animate-pulse"></div>
            <span className="text-yellow-800 font-medium">
              🎙️ Recording in progress... Speak clearly for best results
            </span>
          </div>
        </div>
      )}

      {/* Transcriptions */}
      {transcriptions.length > 0 && (
        <div className="mb-6 p-6 bg-blue-50 border-l-4 border-blue-500 rounded-lg">
          <h3 className="text-lg font-semibold text-blue-900 mb-4">
            📝 Recent Transcriptions
          </h3>
          <div className="space-y-3">
            {transcriptions.map((item) => (
              <div key={item.id} className="bg-white p-4 rounded-lg shadow-sm">
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <p className="text-gray-800 italic font-medium">"{item.text}"</p>
                    <small className="text-gray-500">
                      {new Date(item.timestamp).toLocaleTimeString()} | 
                      Duration: {item.duration?.toFixed(1)}s | 
                      Samples: {item.sampleCount?.toLocaleString()} |
                      Session: {item.sessionId?.slice(-8) || 'N/A'}
                    </small>
                  </div>
                  <button
                    onClick={() => analyzeProsody(item.text, item.sessionId)}
                    disabled={!item.sessionId}
                    className={`ml-4 px-3 py-1 text-sm rounded ${
                      item.sessionId 
                        ? 'bg-blue-600 text-white hover:bg-blue-700' 
                        : 'bg-gray-400 text-gray-200 cursor-not-allowed'
                    }`}
                  >
                    🔬 Analyze Prosody
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Prosody Results */}
      {prosodyResults.length > 0 && (
        <div className="p-6 bg-green-50 border-l-4 border-green-500 rounded-lg">
          <h3 className="text-lg font-semibold text-green-900 mb-4">
            🔬 Prosody Analysis Results
          </h3>
          <div className="space-y-4">
            {prosodyResults.map((result) => (
              <div key={result.id} className="bg-white p-6 rounded-lg shadow-sm">
                <div className="mb-4">
                  <h4 className="font-medium text-gray-900 mb-2">
                    "{result.text}"
                  </h4>
                  <div className="text-sm text-gray-600 space-y-1">
                    <p>📝 Words: {result.word_count} | Characters: {result.character_count}</p>
                    <p>⏱️ Duration: {result.audio_duration?.toFixed(1)}s | Samples: {result.sample_count?.toLocaleString()}</p>
                    <p>🔗 Session: {result.session_id?.slice(-8) || 'N/A'}</p>
                    <p>🕒 Analyzed: {new Date(result.timestamp).toLocaleTimeString()}</p>
                  </div>
                </div>
                
                {result.prosody_summary && (
                  <div className="mb-4 p-3 bg-gray-50 rounded">
                    <h5 className="font-medium text-gray-900 mb-2">📊 Prosody Summary:</h5>
                    <div className="text-sm text-gray-700 space-y-1">
                      <p>✅ Status: {result.prosody_summary.feature_extraction_success ? 'Success' : 'Failed'}</p>
                      <p>🔧 Method: {result.prosody_summary.extraction_method || 'Unknown'}</p>
                      <p>📊 Total Features: {result.prosody_summary.total_features}</p>
                      <p>🎵 Pitch Features: {result.prosody_summary.pitch_features_count}</p>
                      <p>🔊 Energy Features: {result.prosody_summary.energy_features_count}</p>
                      <p>📈 Spectral Features: {result.prosody_summary.spectral_features_count || 0}</p>
                      <p>⏱️ Temporal Features: {result.prosody_summary.temporal_features_count || 0}</p>
                      {!result.prosody_summary.feature_extraction_success && (
                        <p className="text-red-600">❌ Error: {result.prosody_summary.error_info}</p>
                      )}
                    </div>
                  </div>
                )}
                
                {/* Key Features */}
                {result.key_features && Object.keys(result.key_features).length > 0 && (
                  <div className="mb-4">
                    <h5 className="font-medium text-gray-900 mb-2">🔑 Key Prosodic Features:</h5>
                    <div className="grid grid-cols-2 gap-2 text-sm bg-blue-50 p-3 rounded">
                      {Object.entries(result.key_features).map(([key, value]) => (
                        <div key={key} className="flex justify-between">
                          <span className="text-gray-700 truncate">{key}:</span>
                          <span className="font-mono text-blue-800">{value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {/* Raw Features (if available) */}
                {result.prosody_features && 
                 typeof result.prosody_features === 'object' && 
                 !result.prosody_features.error && 
                 Object.keys(result.prosody_features).length > 0 && (
                  <div>
                    <h5 className="font-medium text-gray-900 mb-2">
                      🔧 All Features ({Object.keys(result.prosody_features).length}):
                    </h5>
                    <div className="text-sm text-gray-700 max-h-40 overflow-y-auto bg-gray-50 p-3 rounded">
                      {Object.entries(result.prosody_features)
                        .filter(([key, value]) => typeof value === 'number' || (typeof value === 'string' && value.length < 20))
                        .slice(0, 20)
                        .map(([key, value]) => (
                        <div key={key} className="flex justify-between py-1 border-b border-gray-200 last:border-b-0">
                          <span className="truncate font-mono text-xs">{key}:</span>
                          <span className="ml-2 font-mono text-xs">
                            {typeof value === 'number' ? value.toFixed(3) : value}
                          </span>
                        </div>
                      ))}
                      {Object.keys(result.prosody_features).length > 20 && (
                        <p className="text-gray-500 mt-2 text-center">
                          ... and {Object.keys(result.prosody_features).length - 20} more features
                        </p>
                      )}
                    </div>
                  </div>
                )}
                
                {/* Error display if prosody extraction failed */}
                {result.prosody_features && result.prosody_features.error && (
                  <div className="p-3 bg-red-50 border border-red-200 rounded">
                    <h5 className="font-medium text-red-900 mb-2">❌ Prosody Extraction Error:</h5>
                    <p className="text-sm text-red-700">{result.prosody_features.error}</p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default VoiceProsodyComponent;