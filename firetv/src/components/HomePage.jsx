import React, { useState, useRef, useEffect } from 'react';
import { Search, Mic, Play, Star, Clock, Loader2 } from 'lucide-react';

const HomePage = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [isConnected, setIsConnected] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState('');
  const [prosodyInfluenced, setProsodyInfluenced] = useState(false);
  
  const wsRef = useRef(null);
  const audioContextRef = useRef(null);
  const sourceRef = useRef(null);
  const processorRef = useRef(null);
  const streamRef = useRef(null);
  const audioBufferRef = useRef([]); 
  const clientId = useRef(`home_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`);

  // Mock video data (unchanged)
  const videos = [ /* ... your 8 videos ... */ ];
  const categories = ['All', 'Prime Video', 'Netflix', 'Disney+', 'HBO Max', 'Hulu'];

  // Filter based on searchQuery and selectedCategory
  const filteredVideos = videos.filter(video => {
    const matchesText = video.title.toLowerCase().includes(searchQuery.toLowerCase())
      || video.keywords.some(k => k.includes(searchQuery.toLowerCase()));
    const matchesCat = selectedCategory === 'All' || video.category === selectedCategory;
    return matchesText && matchesCat;
  });

  // Connect websocket on mount
  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/ws/${clientId.current}`);
    ws.onopen = () => {
      setIsConnected(true);
      setVoiceStatus('Voice service online');
    };
    ws.onmessage = ev => handleVoiceMessage(JSON.parse(ev.data));
    ws.onclose = () => {
      setIsConnected(false);
      setVoiceStatus('Voice service offline');
    };
    ws.onerror = err => {
      console.error('WS error:', err);
      setIsConnected(false);
      setVoiceStatus('Voice service error');
    };
    wsRef.current = ws;
    return () => {
      cleanup();
    };
  }, []);

  const handleVoiceMessage = (msg) => {
    switch (msg.type) {
      case 'transcription':
        setIsProcessing(false);
        setSearchQuery(msg.text);
        setVoiceStatus(`🎤 Heard: "${msg.text}"`);
        // Immediately trigger prosody analysis
        wsRef.current.send(JSON.stringify({
          type: 'analyze_prosody',
          text: msg.text,
          session_id: msg.session_id,
          sampleRate: 16000
        }));
        break;
      case 'prosody_analysis':
        setProsodyInfluenced(msg.analysis.prosody_summary.feature_extraction_success);
        setVoiceStatus('✨ Voice characteristics applied');
        break;
      case 'error':
        setIsProcessing(false);
        setIsRecording(false);
        setVoiceStatus(`❌ ${msg.message}`);
        break;
      default:
        console.warn('Unknown msg type:', msg.type);
    }
  };

  const startRecording = async () => {
    if (!isConnected) {
      setVoiceStatus('❌ Voice service not connected');
      return;
    }
    setVoiceStatus('🎙️ Requesting mic...');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1 }
      });
      streamRef.current = stream;
      const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      audioContextRef.current = ctx;
      const src = ctx.createMediaStreamSource(stream);
      sourceRef.current = src;
      const proc = ctx.createScriptProcessor(4096, 1, 1);
      processorRef.current = proc;
      audioBufferRef.current = [];

      proc.onaudioprocess = ({ inputBuffer }) => {
        const data = inputBuffer.getChannelData(0);
        audioBufferRef.current.push(...data);
        if (audioBufferRef.current.length >= 16000) {
          wsRef.current.send(JSON.stringify({
            type: 'audio_data',
            data: Array.from(audioBufferRef.current),
            sampleRate: 16000
          }));
          audioBufferRef.current = [];
        }
      };

      src.connect(proc);
      proc.connect(ctx.destination);
      setIsRecording(true);
      setIsProcessing(true);
      setVoiceStatus('🔴 Recording...');
    } catch (e) {
      console.error(e);
      setVoiceStatus('❌ Mic access denied');
    }
  };

  const stopRecording = () => {
    // send any leftover frames
    if (audioBufferRef.current.length && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'audio_data',
        data: Array.from(audioBufferRef.current),
        sampleRate: 16000
      }));
    }
    // finalise
    wsRef.current.send(JSON.stringify({ type: 'audio_end', sampleRate: 16000 }));
    // clean up audio nodes
    processorRef.current.disconnect();
    sourceRef.current.disconnect();
    audioContextRef.current.close();
    streamRef.current.getTracks().forEach(t => t.stop());
    audioBufferRef.current = [];
    setIsRecording(false);
    setVoiceStatus('⏳ Processing voice...');
  };

  const handleMicClick = () => {
    isRecording ? stopRecording() : startRecording();
  };

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    setProsodyInfluenced(false);
    // simply filter via searchQuery
  };

  const cleanup = () => {
    if (processorRef.current) processorRef.current.disconnect();
    if (sourceRef.current) sourceRef.current.disconnect();
    if (audioContextRef.current) audioContextRef.current.close();
    if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    if (wsRef.current) wsRef.current.close();
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-black to-gray-900 text-white">
      {/* HEADER */}
      <header className="flex justify-between items-center p-6 bg-black bg-opacity-50">
        <h1 className="text-2xl font-bold text-orange-500">fire tv</h1>
        <div className="flex items-center space-x-4">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}/>
          <span className="text-xs text-gray-400">
            {isConnected ? 'Voice Ready' : 'Voice Offline'}
          </span>
        </div>
      </header>

      {/* SEARCH BAR */}
      <div className="px-6 py-8">
        <form onSubmit={handleSearchSubmit} className="max-w-2xl mx-auto">
          <div className="relative flex items-center bg-gray-800 rounded-lg border border-gray-700 focus-within:border-orange-500">
            <Search className="w-5 h-5 text-gray-400 ml-4"/>
            <input
              type="text"
              className="flex-1 bg-transparent px-4 py-3 placeholder-gray-400 focus:outline-none"
              placeholder="Search movies, TV shows..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              disabled={isRecording || isProcessing}
            />
            <button
              type="button"
              onClick={handleMicClick}
              disabled={!isConnected}
              className={`p-3 mr-2 rounded-lg transition-all ${
                isRecording
                  ? 'bg-red-600 animate-pulse'
                  : 'bg-orange-600 hover:bg-orange-700'
              }`}
            >
              {isProcessing
                ? <Loader2 className="w-5 h-5 animate-spin"/>
                : <Mic className="w-5 h-5"/>}
            </button>
          </div>
          {voiceStatus && (
            <p className="mt-2 text-sm text-center">
              {voiceStatus}
            </p>
          )}
        </form>
      </div>

      {/* CATEGORIES */}
      <div className="px-6 mb-6">
        <div className="flex space-x-4 overflow-x-auto pb-2">
          {categories.map(cat => (
            <button
              key={cat}
              className={`px-4 py-2 rounded-full ${
                selectedCategory === cat
                  ? 'bg-orange-500 text-white'
                  : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
              }`}
              onClick={() => setSelectedCategory(cat)}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      {/* VIDEO GRID */}
      <div className="px-6 pb-12 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {filteredVideos.length
          ? filteredVideos.map(video => (
            <div key={video.id} className="group cursor-pointer transform hover:scale-105 transition">
              <div className="relative overflow-hidden rounded-lg bg-gray-800">
                <img src={video.thumbnail} alt={video.title}
                     className="w-full h-48 object-cover group-hover:scale-110 transition"/>
                <div className="absolute inset-0 bg-black bg-opacity-40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition">
                  <Play className="w-8 h-8 text-white"/>
                </div>
                <div className="absolute top-2 right-2 bg-black bg-opacity-70 px-2 py-1 rounded text-xs flex items-center">
                  <Clock className="w-3 h-3 mr-1"/> {video.duration}
                </div>
                {prosodyInfluenced && (
                  <div className="absolute top-2 left-2 bg-orange-600 px-2 py-1 rounded text-xs">
                    🎤 Voice Match
                  </div>
                )}
              </div>
              <div className="mt-3">
                <h3 className="font-semibold group-hover:text-orange-400 transition">
                  {video.title}
                </h3>
                <div className="flex justify-between items-center mt-1 text-sm text-gray-400">
                  <span>{video.category}</span>
                  <span className="flex items-center">
                    <Star className="w-4 h-4 text-yellow-500 mr-1"/> {video.rating}
                  </span>
                </div>
                <span className="text-xs text-gray-500">{video.year}</span>
              </div>
            </div>
          ))
          : (
            <div className="col-span-full text-center py-12 text-gray-400">
              No results found. Try a different search or voice command.
            </div>
          )
        }
      </div>
    </div>
  );
};

export default HomePage;
