import React, { useState, useRef, useEffect } from 'react';
import { Search, Mic, Play, Star, Clock } from 'lucide-react';

const HomePage = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('All');
  const recognitionRef = useRef(null);

  // Mock video data
  const videos = [
    {
      id: 1,
      title: "The Marvelous Mrs. Maisel",
      thumbnail: "https://images.unsplash.com/photo-1489599735188-252d3fa5c85e?w=400&h=225&fit=crop",
      duration: "45 min",
      rating: 4.8,
      category: "Prime Video",
      year: "2023"
    },
    {
      id: 2,
      title: "The Boys",
      thumbnail: "https://images.unsplash.com/photo-1440404653325-ab127d49abc1?w=400&h=225&fit=crop",
      duration: "1h 2min",
      rating: 4.9,
      category: "Prime Video",
      year: "2024"
    },
    {
      id: 3,
      title: "Stranger Things",
      thumbnail: "https://images.unsplash.com/photo-1518676590629-3dcbd9c5a5c9?w=400&h=225&fit=crop",
      duration: "52 min",
      rating: 4.7,
      category: "Netflix",
      year: "2023"
    },
    {
      id: 4,
      title: "The Mandalorian",
      thumbnail: "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=225&fit=crop",
      duration: "38 min",
      rating: 4.6,
      category: "Disney+",
      year: "2024"
    },
    {
      id: 5,
      title: "House of Dragons",
      thumbnail: "https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=400&h=225&fit=crop",
      duration: "1h 8min",
      rating: 4.5,
      category: "HBO Max",
      year: "2024"
    },
    {
      id: 6,
      title: "Wednesday",
      thumbnail: "https://images.unsplash.com/photo-1509909756405-be0199881695?w=400&h=225&fit=crop",
      duration: "47 min",
      rating: 4.4,
      category: "Netflix",
      year: "2023"
    },
    {
      id: 7,
      title: "The Bear",
      thumbnail: "https://images.unsplash.com/photo-1414235077428-338989a2e8c0?w=400&h=225&fit=crop",
      duration: "32 min",
      rating: 4.9,
      category: "Hulu",
      year: "2024"
    },
    {
      id: 8,
      title: "Avatar: The Last Airbender",
      thumbnail: "https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=400&h=225&fit=crop",
      duration: "55 min",
      rating: 4.3,
      category: "Netflix",
      year: "2024"
    }
  ];

  const categories = ['All', 'Prime Video', 'Netflix', 'Disney+', 'HBO Max', 'Hulu'];

  const filteredVideos = videos.filter(video => {
    const matchesSearch = video.title.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = selectedCategory === 'All' || video.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

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
  }, []);

  const startVoiceSearch = () => {
    if (recognitionRef.current) {
      recognitionRef.current.start();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-black to-gray-900 text-white">
      {/* Header */}
      <header className="flex items-center justify-between p-6 bg-black bg-opacity-50">
        <div className="flex items-center space-x-4">
          <div className="text-2xl font-bold text-orange-500">fire tv</div>
        </div>
        <div className="flex items-center space-x-4">
          <div className="text-sm">Good evening</div>
          <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-sm font-semibold">
            U
          </div>
        </div>
      </header>

      {/* Search Section */}
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
              Try searching for something else or check your spelling
            </div>
          </div>
        )}
      </div>

      {/* Bottom Navigation Hint */}
      <div className="fixed bottom-4 left-1/2 transform -translate-x-1/2 bg-gray-800 bg-opacity-90 backdrop-blur-sm px-4 py-2 rounded-full text-sm text-gray-300">
        Use voice search or browse by category
      </div>
    </div>
  );
};

export default HomePage;