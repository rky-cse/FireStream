import React, { useState, useRef, useEffect } from 'react';
import { Search, Mic, Play, Star, Clock, Loader2 } from 'lucide-react';

const HomePage = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [isConnected, setIsConnected] = useState(false);
  const [voiceSession, setVoiceSession] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState('');
  const [searchResults, setSearchResults] = useState(null);
  
  // Enhanced voice search refs
  const wsRef = useRef(null);
  const audioContextRef = useRef(null);
  const sourceRef = useRef(null);
  const processorRef = useRef(null);
  const streamRef = useRef(null);
  const audioBufferRef = useRef([]); // Add buffer ref to accumulate audio
  const clientId = useRef(`homepage_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`);

  // Mock video data
  const videos = [
    {
      id: 1,
      title: "The Marvelous Mrs. Maisel",
      thumbnail: "https://images.unsplash.com/photo-1489599735188-252d3fa5c85e?w=400&h=225&fit=crop",
      duration: "45 min",
      rating: 4.8,
      category: "Prime Video",
      year: "2023",
      keywords: ["comedy", "period", "drama", "amazon", "marvelous", "maisel"]
    },
    {
      id: 2,
      title: "The Boys",
      thumbnail: "https://images.unsplash.com/photo-1440404653325-ab127d49abc1?w=400&h=225&fit=crop",
      duration: "1h 2min",
      rating: 4.9,
      category: "Prime Video",
      year: "2024",
      keywords: ["superhero", "action", "dark", "comic", "boys", "violence"]
    },
    {
      id: 3,
      title: "Stranger Things",
      thumbnail: "https://images.unsplash.com/photo-1518676590629-3dcbd9c5a5c9?w=400&h=225&fit=crop",
      duration: "52 min",
      rating: 4.7,
      category: "Netflix",
      year: "2023",
      keywords: ["sci-fi", "horror", "80s", "kids", "stranger", "things", "upside down"]
    },
    {
      id: 4,
      title: "The Mandalorian",
      thumbnail: "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=225&fit=crop",
      duration: "38 min",
      rating: 4.6,
      category: "Disney+",
      year: "2024",
      keywords: ["star wars", "mandalorian", "baby yoda", "space", "bounty hunter"]
    },
    {
      id: 5,
      title: "House of Dragons",
      thumbnail: "https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=400&h=225&fit=crop",
      duration: "1h 8min",
      rating: 4.5,
      category: "HBO Max",
      year: "2024",
      keywords: ["fantasy", "dragons", "game of thrones", "medieval", "hbo"]
    },
    {
      id: 6,
      title: "Wednesday",
      thumbnail: "https://images.unsplash.com/photo-1509909756405-be0199881695?w=400&h=225&fit=crop",
      duration: "47 min",
      rating: 4.4,
      category: "Netflix",
      year: "2023",
      keywords: ["horror", "comedy", "addams family", "wednesday", "school"]
    },
    {
      id: 7,
      title: "The Bear",
      thumbnail: "https://images.unsplash.com/photo-1414235077428-338989a2e8c0?w=400&h=225&fit=crop",
      duration: "32 min",
      rating: 4.9,
      category: "Hulu",
      year: "2024",
      keywords: ["cooking", "drama", "chef", "restaurant", "kitchen", "bear"]
    },
    {
      id: 8,
      title: "Avatar: The Last Airbender",
      thumbnail: "https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=400&h=225&fit=crop",
      duration: "55 min",
      rating: 4.3,
      category: "Netflix",
      year: "2024",
      keywords: ["animation", "adventure", "elements", "avatar", "airbender", "kids"]
    }
  ];

  const categories = ['All', 'Prime Video', 'Netflix', 'Disney+', 'HBO Max', 'Hulu'];

  // Enhanced search function using prosody-enhanced results or local filtering
  const getFilteredVideos = () => {
    if (searchResults && searchResults.results) {
      // If we have voice search results, try to match them with videos
      const voiceResults = searchResults.results;
      const matchedVideos = [];
      
      voiceResults.forEach(result => {
        const matchingVideos = videos.filter(video => {
          const titleMatch = video.title.toLowerCase().includes(searchQuery.toLowerCase());
          const keywordMatch = video.keywords.some(keyword => 
            result.title.toLowerCase().includes(keyword) || 
            result.snippet.toLowerCase().includes(keyword)
          );
          return titleMatch || keywordMatch;
        });
        matchedVideos.push(...matchingVideos);
      });
      
      // Remove duplicates and apply category filter
      const uniqueVideos = [...new Map(matchedVideos.map(v => [v.id, v])).values()];
      return uniqueVideos.filter(video => 
        selectedCategory === 'All' || video.category === selectedCategory
      );
    }
    
    // Fallback to normal search
    return videos.filter(video => {
      const matchesSearch = video.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           video.keywords.some(keyword => keyword.includes(searchQuery.toLowerCase()));
      const matchesCategory = selectedCategory === 'All' || video.category === selectedCategory;
      return matchesSearch && matchesCategory;
    });
  };

  const filteredVideos = getFilteredVideos();

  // Enhanced voice search setup
  useEffect(() => {
    connectToVoiceSearch();
    return () => {
      cleanup();
    };
  }, []);

  const connectToVoiceSearch = () => {
    try {
      const ws = new WebSocket(`ws://localhost:8001/ws/voice-search/${clientId.current}`);
      
      ws.onopen = () => {
        setIsConnected(true);
        console.log('Connected to enhanced voice search');
      };
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleVoiceSearchMessage(data);
      };
      
      ws.onclose = () => {
        setIsConnected(false);
        console.log('Voice search disconnected');
      };
      
      ws.onerror = (error) => {
        setIsConnected(false);
        console.error('Voice search error:', error);
      };
      
      wsRef.current = ws;
    } catch (err) {
      console.error('Failed to connect to voice search:', err);
      setIsConnected(false);
    }
  };

  const handleVoiceSearchMessage = (data) => {
    console.log('Voice search message:', data.type, data);
    
    switch (data.type) {
      case 'recording_started':
        setVoiceStatus('🎤 Recording your voice...');
        break;
        
      case 'voice_transcription':
        setSearchQuery(data.text);
        setVoiceSession(data.session_id);
        setIsRecording(false);
        setIsProcessing(true);
        setVoiceStatus(`✅ Voice recognized: "${data.text}"`);
        
        // Automatically search with the transcribed text
        performEnhancedSearch(data.text, data.session_id);
        break;
        
      case 'search_results':
        setSearchResults(data.results);
        setIsProcessing(false);
        const prosodyNote = data.results.prosody_influenced ? ' (Voice-enhanced)' : '';
        setVoiceStatus(`🔍 Found ${data.results.total_results} results${prosodyNote}`);
        break;
        
      case 'error':
        setIsRecording(false);
        setIsProcessing(false);
        setVoiceStatus(`❌ ${data.message}`);
        console.error('Voice search error:', data.message);
        break;
    }
  };

  const performEnhancedSearch = async (query, sessionId) => {
    try {
      const searchPayload = {
        query: query.trim(),
        type: sessionId ? 'voice' : 'text',
        session_id: sessionId
      };
      
      const response = await fetch('http://localhost:8001/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(searchPayload)
      });
      
      const results = await response.json();
      
      if (!results.error) {
        setSearchResults(results);
        setIsProcessing(false);
        const prosodyNote = results.prosody_influenced ? ' (Voice-enhanced)' : '';
        setVoiceStatus(`🔍 Found ${results.total_results} results${prosodyNote}`);
      } else {
        setIsProcessing(false);
        setVoiceStatus(`❌ Search failed: ${results.error}`);
      }
    } catch (err) {
      setIsProcessing(false);
      setVoiceStatus(`❌ Search request failed`);
      console.error('Search error:', err);
    }
  };

  const startVoiceSearch = async () => {
    if (!isConnected) {
      setVoiceStatus('❌ Voice search service not available');
      return;
    }

    try {
      setVoiceStatus('🎤 Requesting microphone access...');
      
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        }
      });
      
      streamRef.current = stream;
      setVoiceStatus('🎤 Microphone access granted, setting up audio...');
      
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000
      });
      
      const source = audioContextRef.current.createMediaStreamSource(stream);
      sourceRef.current = source;
      
      const processor = audioContextRef.current.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      
      // Initialize audio buffer - FIXED: Reset buffer at start
      audioBufferRef.current = [];
      
      // Start recording session
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'start_voice_recording'
        }));
      }
      
      // FIXED: Proper audio processing like working VoiceSearchComponent
      processor.onaudioprocess = (event) => {
        const inputBuffer = event.inputBuffer;
        const inputData = inputBuffer.getChannelData(0);
        
        // Accumulate audio data in buffer
        audioBufferRef.current.push(...inputData);
        
        // Send data every ~1 second (16000 samples at 16kHz) - same as working code
        if (audioBufferRef.current.length >= 16000) {
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({
              type: 'audio_data',
              data: Array.from(audioBufferRef.current), // Send all accumulated data
              sampleRate: 16000
            }));
            
            console.log(`Sent ${audioBufferRef.current.length} audio samples (${(audioBufferRef.current.length/16000).toFixed(1)}s)`);
          }
          audioBufferRef.current = []; // Clear buffer after sending
        }
      };
      
      source.connect(processor);
      processor.connect(audioContextRef.current.destination);
      
      setIsRecording(true);
      setVoiceStatus('🎙️ Recording... Click again to stop');
      console.log('Recording started with Web Audio API');
      
    } catch (err) {
      setVoiceStatus('❌ Microphone access denied');
      console.error('Voice search error:', err);
    }
  };

  const stopVoiceSearch = () => {
    try {
      // FIXED: Send any remaining audio data before stopping
      if (audioBufferRef.current.length > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'audio_data',
          data: Array.from(audioBufferRef.current),
          sampleRate: 16000
        }));
        console.log(`Sent final ${audioBufferRef.current.length} audio samples`);
      }
      
      // Send stop signal
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'stop_voice_recording'
        }));
      }
      
      // Cleanup audio - same as working code
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
      setVoiceStatus('⏹️ Processing voice...');
      console.log('Recording stopped');
      
      // Clear buffer
      audioBufferRef.current = [];
      
    } catch (err) {
      console.error('Error stopping voice search:', err);
      setIsRecording(false);
    }
  };

  const handleVoiceClick = () => {
    if (isRecording) {
      stopVoiceSearch();
    } else {
      startVoiceSearch();
    }
  };

  const handleSearch = (e) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      performEnhancedSearch(searchQuery.trim(), null);
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
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-black to-gray-900 text-white">
      {/* Header */}
      <header className="flex items-center justify-between p-6 bg-black bg-opacity-50">
        <div className="flex items-center space-x-4">
          <div className="text-2xl font-bold text-orange-500">fire tv</div>
          {/* Voice Search Status Indicator */}
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className="text-xs text-gray-400">
              {isConnected ? 'Voice Ready' : 'Voice Offline'}
            </span>
          </div>
        </div>
        <div className="flex items-center space-x-4">
          <div className="text-sm">Good evening, rky-cse</div>
          <div className="text-xs text-gray-400">2025-06-19 11:52:43 UTC</div>
          <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-sm font-semibold">
            R
          </div>
        </div>
      </header>

      {/* Enhanced Search Section */}
      <div className="px-6 py-8">
        <div className="max-w-2xl mx-auto">
          <form onSubmit={handleSearch}>
            <div className="relative">
              <div className="flex items-center bg-gray-800 rounded-lg border border-gray-700 focus-within:border-orange-500">
                <Search className="w-5 h-5 text-gray-400 ml-4" />
                <input
                  type="text"
                  placeholder="Search movies, TV shows, actors..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="flex-1 bg-transparent px-4 py-3 text-white placeholder-gray-400 focus:outline-none"
                  disabled={isRecording || isProcessing}
                />
                <button
                  type="button"
                  onClick={handleVoiceClick}
                  disabled={!isConnected || isProcessing}
                  className={`p-3 mr-2 rounded-lg transition-all ${
                    isRecording 
                      ? 'bg-red-600 text-white animate-pulse' 
                      : isConnected
                      ? 'bg-orange-600 text-white hover:bg-orange-700'
                      : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                  }`}
                  title={isRecording ? 'Stop recording' : 'Start voice search'}
                >
                  {isProcessing ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Mic className="w-5 h-5" />
                  )}
                </button>
              </div>
              
              {/* Voice Status Display */}
              {voiceStatus && (
                <div className={`absolute top-full left-0 right-0 mt-2 p-3 rounded-lg text-center text-sm z-10 ${
                  isRecording ? 'bg-red-600' : 
                  voiceStatus.includes('✅') ? 'bg-green-600' :
                  voiceStatus.includes('❌') ? 'bg-red-600' :
                  'bg-blue-600'
                }`}>
                  {voiceStatus}
                </div>
              )}
              
              {/* Voice Session Info */}
              {voiceSession && (
                <div className="absolute top-full right-0 mt-2 p-2 bg-orange-600 rounded text-xs z-10">
                  🎤 Voice session: {voiceSession.slice(-8)}
                </div>
              )}
            </div>
          </form>
        </div>
      </div>

      {/* Categories */}
      <div className="px-6 mb-6">
        <div className="flex space-x-4 overflow-x-auto pb-2">
          {categories.map((category) => (
            <button
              key={category}
              onClick={() => setSelectedCategory(category)}
              className={`px-4 py-2 rounded-full whitespace-nowrap transition-all ${
                selectedCategory === category
                  ? 'bg-orange-500 text-white'
                  : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
              }`}
            >
              {category}
            </button>
          ))}
        </div>
      </div>

      {/* Enhanced Search Results Info */}
      {searchResults && searchResults.prosody_influenced && (
        <div className="px-6 mb-4">
          <div className="max-w-2xl mx-auto bg-orange-600 bg-opacity-20 border border-orange-500 rounded-lg p-3">
            <div className="text-sm text-orange-300">
              ✨ <strong>Voice-Enhanced Search:</strong> Results influenced by your voice characteristics
              <div className="text-xs text-orange-400 mt-1">
                {searchResults.search_metadata?.prosody_summary && (
                  <>
                    Intensity: {searchResults.search_metadata.prosody_summary.intensity} • 
                    Tempo: {searchResults.search_metadata.prosody_summary.tempo} • 
                    Context: {searchResults.search_context}
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Videos Grid */}
      <div className="px-6 pb-12">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {filteredVideos.map((video) => (
            <div
              key={video.id}
              className="group cursor-pointer transform transition-all duration-300 hover:scale-105"
            >
              <div className="relative overflow-hidden rounded-lg bg-gray-800">
                <img
                  src={video.thumbnail}
                  alt={video.title}
                  className="w-full h-48 object-cover transition-transform duration-300 group-hover:scale-110"
                />
                <div className="absolute inset-0 bg-black bg-opacity-40 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
                  <div className="bg-white bg-opacity-20 rounded-full p-3 backdrop-blur-sm">
                    <Play className="w-8 h-8 text-white" fill="white" />
                  </div>
                </div>
                <div className="absolute top-2 right-2 bg-black bg-opacity-70 px-2 py-1 rounded text-xs flex items-center">
                  <Clock className="w-3 h-3 mr-1" />
                  {video.duration}
                </div>
                {/* Voice Search Match Indicator */}
                {searchResults && searchResults.prosody_influenced && searchQuery && (
                  <div className="absolute top-2 left-2 bg-orange-600 bg-opacity-90 px-2 py-1 rounded text-xs">
                    🎤 Voice Match
                  </div>
                )}
              </div>
              <div className="mt-3">
                <h3 className="font-semibold text-white group-hover:text-orange-400 transition-colors">
                  {video.title}
                </h3>
                <div className="flex items-center justify-between mt-1">
                  <span className="text-sm text-gray-400">{video.category}</span>
                  <div className="flex items-center text-sm text-gray-400">
                    <Star className="w-4 h-4 text-yellow-500 mr-1" fill="currentColor" />
                    {video.rating}
                  </div>
                </div>
                <span className="text-xs text-gray-500">{video.year}</span>
              </div>
            </div>
          ))}
        </div>

        {filteredVideos.length === 0 && (
          <div className="text-center py-12">
            <div className="text-gray-400 text-lg">No results found</div>
            <div className="text-gray-500 text-sm mt-2">
              Try searching for something else or use voice search
            </div>
          </div>
        )}
      </div>

      {/* Enhanced Bottom Navigation */}
      <div className="fixed bottom-4 left-1/2 transform -translate-x-1/2 bg-gray-800 bg-opacity-90 backdrop-blur-sm px-4 py-2 rounded-full text-sm text-gray-300">
        {isConnected ? (
          <span>🎤 Enhanced voice search ready • Browse by category</span>
        ) : (
          <span>Voice search offline • Browse by category</span>
        )}
      </div>
    </div>
  );
};

export default HomePage;