// File: src/components/SearchSection.js
import React, { useState, useRef, useEffect } from 'react';
import { Search, Mic } from 'lucide-react';

const SearchSection = ({ searchQuery, setSearchQuery }) => {
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef(null);

  // Voice recognition setup
  useEffect(() => {
    if ('webkitSpeechRecognition' in window) {
      const recognition = new window.webkitSpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';

      recognition.onstart = () => {
        setIsListening(true);
      };

      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setSearchQuery(transcript);
        setIsListening(false);
      };

      recognition.onerror = () => {
        setIsListening(false);
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current = recognition;
    }
  }, [setSearchQuery]);

  const startVoiceSearch = () => {
    if (recognitionRef.current) {
      recognitionRef.current.start();
    }
  };

  return (
    <div className="px-6 py-8">
      <div className="max-w-2xl mx-auto">
        <div className="relative">
          <div className="flex items-center bg-gray-800 rounded-lg border border-gray-700 focus-within:border-orange-500">
            <Search className="w-5 h-5 text-gray-400 ml-4" />
            <input
              type="text"
              placeholder="Search movies, TV shows, actors..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="flex-1 bg-transparent px-4 py-3 text-white placeholder-gray-400 focus:outline-none"
            />
            <button
              onClick={startVoiceSearch}
              className={`p-3 mr-2 rounded-lg transition-all ${
                isListening 
                  ? 'bg-red-600 text-white animate-pulse' 
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              <Mic className="w-5 h-5" />
            </button>
          </div>
          {isListening && (
            <div className="absolute top-full left-0 right-0 mt-2 p-3 bg-red-600 rounded-lg text-center">
              <span className="text-sm">Listening...</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SearchSection;