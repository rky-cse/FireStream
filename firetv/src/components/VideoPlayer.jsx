// src/components/VideoPlayer.jsx
import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, Volume2, VolumeX, Maximize, Settings } from 'lucide-react';
import { mockStreamData } from '../utils/mockData';

const VideoPlayer = ({ showChat, onToggleChat }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [volume, setVolume] = useState(100);
  const [showControls, setShowControls] = useState(true);
  const [currentTime, setCurrentTime] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const videoRef = useRef(null);
  const controlsTimeoutRef = useRef(null);

  const { title, description, duration } = mockStreamData;

  useEffect(() => {
    const timer = setInterval(() => {
      if (isPlaying) {
        setCurrentTime(prev => Math.min(prev + 1, duration));
      }
    }, 1000);

    return () => clearInterval(timer);
  }, [isPlaying, duration]);

  // Auto-hide controls
  useEffect(() => {
    if (controlsTimeoutRef.current) {
      clearTimeout(controlsTimeoutRef.current);
    }
    
    if (showControls) {
      controlsTimeoutRef.current = setTimeout(() => {
        if (isPlaying) {
          setShowControls(false);
        }
      }, 3000);
    }

    return () => {
      if (controlsTimeoutRef.current) {
        clearTimeout(controlsTimeoutRef.current);
      }
    };
  }, [showControls, isPlaying]);

  const formatTime = (seconds) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    if (hrs > 0) {
      return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handleProgressClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const width = rect.width;
    const newTime = Math.floor((clickX / width) * duration);
    setCurrentTime(newTime);
  };

  const handleVolumeChange = (e) => {
    const newVolume = parseInt(e.target.value);
    setVolume(newVolume);
    setIsMuted(newVolume === 0);
  };

  const handleMuteToggle = () => {
    setIsMuted(!isMuted);
  };

  const handleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };

  const handleMouseMove = () => {
    setShowControls(true);
  };

  return (
    <div 
      className="flex-1 relative bg-black"
      onMouseMove={handleMouseMove}
    >
      {/* Video Background */}
      <div className="w-full h-full bg-gradient-to-br from-gray-900 to-black flex items-center justify-center">
        <div className="text-center">
          <div className="w-32 h-32 bg-gray-800 rounded-full flex items-center justify-center mb-4 mx-auto">
            {isPlaying ? (
              <div className="w-16 h-16 bg-red-600 rounded-full flex items-center justify-center">
                <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
              </div>
            ) : (
              <Play className="w-16 h-16 text-white" />
            )}
          </div>
          <h2 className="text-2xl font-bold text-white mb-2">{title}</h2>
          <p className="text-gray-400">{description}</p>
          {isPlaying && (
            <div className="mt-4 text-red-500 font-semibold animate-pulse">
              🔴 LIVE
            </div>
          )}
        </div>
      </div>

      {/* Video Controls Overlay */}
      <div className={`absolute inset-0 transition-opacity duration-300 ${
        showControls ? 'opacity-100' : 'opacity-0'
      }`}>
        {/* Center Play Button */}
        <div className="absolute inset-0 flex items-center justify-center">
          <button
            onClick={handlePlayPause}
            className="bg-black/50 hover:bg-black/70 rounded-full p-4 transition-colors"
          >
            {isPlaying ? (
              <Pause className="w-12 h-12 text-white" />
            ) : (
              <Play className="w-12 h-12 text-white" />
            )}
          </button>
        </div>

        {/* Bottom Controls */}
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-6">
          {/* Progress Bar */}
          <div className="mb-4">
            <div
              className="w-full h-2 bg-gray-700 rounded cursor-pointer hover:h-3 transition-all"
              onClick={handleProgressClick}
            >
              <div
                className="h-full bg-red-600 rounded transition-all duration-300"
                style={{ width: `${(currentTime / duration) * 100}%` }}
              ></div>
            </div>
            <div className="flex justify-between text-sm text-gray-300 mt-1">
              <span>{formatTime(currentTime)}</span>
              <span>{formatTime(duration)}</span>
            </div>
          </div>

          {/* Control Buttons */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={handlePlayPause}
                className="bg-white/20 hover:bg-white/30 rounded-full p-3 transition-colors"
              >
                {isPlaying ? (
                  <Pause className="w-6 h-6 text-white" />
                ) : (
                  <Play className="w-6 h-6 text-white" />
                )}
              </button>
              
              <div className="flex items-center space-x-2">
                <button
                  onClick={handleMuteToggle}
                  className="bg-white/20 hover:bg-white/30 rounded-full p-2 transition-colors"
                >
                  {isMuted || volume === 0 ? (
                    <VolumeX className="w-5 h-5 text-white" />
                  ) : (
                    <Volume2 className="w-5 h-5 text-white" />
                  )}
                </button>
                
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={isMuted ? 0 : volume}
                  onChange={handleVolumeChange}
                  className="w-20 h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                />
              </div>

              <div className="text-white">
                <h3 className="font-semibold">{title}</h3>
                <p className="text-sm text-gray-300">{description}</p>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              <button className="bg-white/20 hover:bg-white/30 rounded-full p-2 transition-colors">
                <Settings className="w-5 h-5 text-white" />
              </button>
              
              <button
                onClick={onToggleChat}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  showChat 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-white/20 text-white hover:bg-white/30'
                }`}
              >
                Chat
              </button>
              
              <button 
                onClick={handleFullscreen}
                className="bg-white/20 hover:bg-white/30 rounded-full p-2 transition-colors"
              >
                <Maximize className="w-5 h-5 text-white" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoPlayer;