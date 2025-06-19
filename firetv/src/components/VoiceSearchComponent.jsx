import React, { useState, useRef, useEffect } from 'react';

const VoiceSearchComponent = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [searchResults, setSearchResults] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const [voiceSession, setVoiceSession] = useState(null);
  const [systemStatus, setSystemStatus] = useState(null);
  const [error, setError] = useState('');
  const [statusMessage, setStatusMessage] = useState('');
  
  // Audio refs - same pattern as working prosody code
  const wsRef = useRef(null);
  const audioContextRef = useRef(null);
  const sourceRef = useRef(null);
  const processorRef = useRef(null);
  const streamRef = useRef(null);
  const searchInputRef = useRef(null);
  const clientId = useRef(`search_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`);

  useEffect(() => {
    checkSystemHealth();
    connectWebSocket();
    return () => {
      cleanup();
    };
  }, []);

  const checkSystemHealth = async () => {
    try {
      const response = await fetch('http://localhost:8001/api/health');
      const health = await response.json();
      setSystemStatus(health);
      
      if (!health.whisper_loaded) {
        setError('Whisper model not loaded');
      }
    } catch (err) {
      setError('Cannot connect to voice search service on port 8001');
      console.error('Health check failed:', err);
    }
  };

  const connectWebSocket = () => {
    try {
      const ws = new WebSocket(`ws://localhost:8001/ws/voice-search/${clientId.current}`);
      
      ws.onopen = () => {
        setIsConnected(true);
        setError('');
        setStatusMessage('✅ Connected to voice search service');
        console.log('Connected to voice search WebSocket');
      };
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
      };
      
      ws.onclose = () => {
        setIsConnected(false);
        setStatusMessage('❌ Disconnected from voice search service');
        console.log('Voice search WebSocket disconnected');
      };
      
      ws.onerror = (error) => {
        setError('Voice search connection failed');
        console.error('WebSocket error:', error);
      };
      
      wsRef.current = ws;
    } catch (err) {
      setError('Failed to connect to voice search service');
      console.error('Connection error:', err);
    }
  };

  const handleWebSocketMessage = (data) => {
    console.log('Voice search message:', data.type, data);
    
    switch (data.type) {
      case 'recording_started':
        setStatusMessage('🎤 Recording voice search...');
        break;
        
      case 'voice_transcription':
        setSearchQuery(data.text);
        setVoiceSession(data.session_id);
        setStatusMessage(`✅ Voice recognized: "${data.text}" ${data.prosody_available ? '(with prosody)' : ''}`);
        setIsRecording(false);
        // Auto-focus search input
        if (searchInputRef.current) {
          searchInputRef.current.focus();
        }
        break;
        
      case 'search_results':
        setSearchResults(data.results);
        setIsSearching(false);
        setStatusMessage(`🔍 Found ${data.results.total_results} results`);
        break;
        
      case 'error':
        setError(data.message);
        setIsRecording(false);
        setIsSearching(false);
        setStatusMessage('❌ Operation failed');
        break;
    }
  };

  const startVoiceSearch = async () => {
    if (!isConnected) {
      setError('Not connected to voice search service');
      return;
    }

    try {
      setError('');
      setStatusMessage('🎤 Requesting microphone access...');
      
      // Get microphone stream - same pattern as working code
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        }
      });
      
      streamRef.current = stream;
      setStatusMessage('🎤 Microphone access granted, setting up audio...');
      
      // Create audio context - same as working code
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000
      });
      
      const source = audioContextRef.current.createMediaStreamSource(stream);
      sourceRef.current = source;
      
      const processor = audioContextRef.current.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      
      let audioBuffer = [];
      
      // Start recording session
      wsRef.current.send(JSON.stringify({
        type: 'start_voice_recording'
      }));
      
      processor.onaudioprocess = (event) => {
        const inputBuffer = event.inputBuffer;
        const inputData = inputBuffer.getChannelData(0);
        
        // Accumulate audio data
        audioBuffer.push(...inputData);
        
        // Send data every ~1 second (same pattern as working code)
        if (audioBuffer.length >= 16000) {
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({
              type: 'audio_data',
              data: Array.from(audioBuffer),
              sampleRate: 16000
            }));
          }
          audioBuffer = [];
        }
      };
      
      // Connect audio graph
      source.connect(processor);
      processor.connect(audioContextRef.current.destination);
      
      setIsRecording(true);
      setStatusMessage('🎙️ Recording voice search... Click stop when done');
      
    } catch (err) {
      setError('Voice search failed: ' + err.message);
      setStatusMessage('❌ Microphone access denied');
      console.error('Voice search error:', err);
    }
  };

  const stopVoiceSearch = () => {
    try {
      // Send stop signal
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'stop_voice_recording'
        }));
      }
      
      // Cleanup audio - same pattern as working code
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
      
      setIsRecording(false);
      setStatusMessage('⏹️ Processing voice recording...');
      
    } catch (err) {
      console.error('Error stopping voice search:', err);
      setIsRecording(false);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    
    if (!searchQuery.trim()) {
      setError('Please enter a search query');
      return;
    }
    
    setIsSearching(true);
    setError('');
    setStatusMessage('🔍 Searching...');
    
    try {
      const searchPayload = {
        query: searchQuery.trim(),
        type: voiceSession ? 'voice' : 'text',
        session_id: voiceSession
      };
      
      console.log('Sending search request:', searchPayload);
      
      const response = await fetch('http://localhost:8001/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(searchPayload)
      });
      
      const results = await response.json();
      
      if (results.error) {
        setError(results.error);
        setIsSearching(false);
        setStatusMessage('❌ Search failed');
      } else {
        setSearchResults(results);
        setIsSearching(false);
        
        const prosodyInfo = results.search_metadata?.prosody_summary;
        if (prosodyInfo && prosodyInfo.method !== 'text_only') {
          setStatusMessage(`🔍 Found ${results.total_results} results (Voice-enhanced: ${prosodyInfo.method})`);
        } else {
          setStatusMessage(`🔍 Found ${results.total_results} results`);
        }
      }
      
    } catch (err) {
      setError('Search request failed: ' + err.message);
      setIsSearching(false);
      setStatusMessage('❌ Search request failed');
      console.error('Search error:', err);
    }
  };

  const clearSearch = () => {
    setSearchQuery('');
    setSearchResults(null);
    setVoiceSession(null);
    setError('');
    setStatusMessage('');
    if (searchInputRef.current) {
      searchInputRef.current.focus();
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

  return (
    <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-lg p-6">
      {/* Header */}
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">
          🎤 Voice Search
        </h2>
        <p className="text-gray-600">
          Search using voice or text - Enhanced with prosody analysis
        </p>
        
        {/* System Status */}
        <div className="mt-4 space-y-2">
          <div className="flex justify-center items-center space-x-6">
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm">WebSocket: {isConnected ? '✅ Connected' : '❌ Disconnected'}</span>
            </div>
            
            {systemStatus && (
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${systemStatus.whisper_loaded ? 'bg-green-500' : 'bg-red-500'}`}></div>
                <span className="text-sm">Whisper: {systemStatus.whisper_loaded ? '✅ Ready' : '❌ Not Available'}</span>
              </div>
            )}
            
            <div className="text-xs text-gray-500">
              rky-cse | 2025-06-19 11:32:44 UTC
            </div>
          </div>
          
          {/* Status Message */}
          {statusMessage && (
            <div className="text-sm text-blue-600 bg-blue-50 p-2 rounded">
              {statusMessage}
            </div>
          )}
        </div>
      </div>

      {/* Search Box */}
      <div className="mb-6">
        <form onSubmit={handleSearch} className="relative">
          <div className="flex items-center border-2 border-gray-300 rounded-lg focus-within:border-blue-500">
            <input
              ref={searchInputRef}
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Enter your search query or use voice search..."
              className="flex-1 px-4 py-3 text-lg border-none outline-none rounded-l-lg"
              disabled={isSearching || isRecording}
            />
            
            {/* Voice Button */}
            <button
              type="button"
              onClick={isRecording ? stopVoiceSearch : startVoiceSearch}
              disabled={!isConnected || isSearching}
              className={`px-4 py-3 border-l border-gray-300 ${
                isRecording 
                  ? 'bg-red-100 text-red-700 hover:bg-red-200 animate-pulse' 
                  : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
              } disabled:bg-gray-100 disabled:text-gray-400 disabled:cursor-not-allowed`}
              title={isRecording ? 'Stop voice recording' : 'Start voice search'}
            >
              {isRecording ? '🛑' : '🎤'}
            </button>
            
            {/* Search Button */}
            <button
              type="submit"
              disabled={!searchQuery.trim() || isSearching || isRecording}
              className="px-6 py-3 bg-green-600 text-white rounded-r-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              {isSearching ? '⏳' : '🔍'}
            </button>
          </div>
        </form>
        
        {/* Controls */}
        <div className="flex justify-between items-center mt-2">
          <div className="text-sm text-gray-500">
            {voiceSession && (
              <span className="bg-green-100 text-green-800 px-2 py-1 rounded">
                🎤 Voice session: {voiceSession.slice(-8)}
              </span>
            )}
          </div>
          
          <button
            onClick={clearSearch}
            className="text-sm text-gray-500 hover:text-gray-700"
          >
            🗑️ Clear
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-800">❌ {error}</p>
        </div>
      )}

      {/* Recording Indicator */}
      {isRecording && (
        <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <div className="flex items-center justify-center space-x-3">
            <div className="w-4 h-4 bg-red-500 rounded-full animate-pulse"></div>
            <span className="text-yellow-800 font-medium">
              🎤 Recording voice search... Click the red button to stop
            </span>
          </div>
        </div>
      )}

      {/* Search Results */}
      {searchResults && (
        <div className="space-y-6">
          {/* Search Summary */}
          <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded-lg">
            <div className="flex justify-between items-start">
              <div>
                <h3 className="text-lg font-semibold text-blue-900 mb-2">
                  Search Results for: "{searchResults.query}"
                </h3>
                <p className="text-blue-700 text-sm">
                  Found {searchResults.total_results} results • 
                  Search type: {searchResults.search_metadata?.prosody_summary?.method === 'text_only' ? '📝 Text' : '🎤 Voice'} • 
                  Context: {searchResults.search_context}
                  {searchResults.prosody_influenced && (
                    <span className="ml-2 bg-blue-200 text-blue-800 px-2 py-1 rounded text-xs">
                      ✨ Prosody Enhanced
                    </span>
                  )}
                </p>
              </div>
            </div>
            
            {/* Prosody Summary */}
            {searchResults.search_metadata?.prosody_summary && 
             searchResults.search_metadata.prosody_summary.method !== 'text_only' && (
              <div className="mt-3 p-3 bg-white rounded border text-sm">
                <h4 className="font-medium text-gray-900 mb-2">🔬 Voice Analysis:</h4>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>Method: {searchResults.search_metadata.prosody_summary.method}</div>
                  <div>Duration: {searchResults.search_metadata.prosody_summary.duration}s</div>
                  <div>Intensity: {searchResults.search_metadata.prosody_summary.intensity}</div>
                  <div>Tempo: {searchResults.search_metadata.prosody_summary.tempo}</div>
                  <div>Pitch: {searchResults.search_metadata.prosody_summary.pitch}</div>
                  <div>Energy: {searchResults.search_metadata.prosody_summary.energy?.toFixed(4) || 'N/A'}</div>
                </div>
              </div>
            )}
          </div>
          
          {/* Results List */}
          <div className="space-y-4">
            {searchResults.results.map((result, index) => (
              <div key={index} className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                <h4 className="text-lg font-medium text-blue-600 hover:text-blue-800 mb-2">
                  <a href={result.url} target="_blank" rel="noopener noreferrer">
                    {result.title}
                  </a>
                </h4>
                <p className="text-gray-700 text-sm mb-2">{result.snippet}</p>
                <a 
                  href={result.url} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-green-600 text-sm hover:text-green-800"
                >
                  {result.url}
                </a>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Connection Help */}
      {!isConnected && (
        <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <h4 className="font-medium text-yellow-900 mb-2">🔧 Connection Issue</h4>
          <p className="text-yellow-800 text-sm mb-2">
            Make sure the voice search service is running on port 8001:
          </p>
          <code className="text-xs bg-yellow-100 p-1 rounded">python voice_search.py</code>
        </div>
      )}
    </div>
  );
};

export default VoiceSearchComponent;