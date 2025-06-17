import React, { createContext, useContext, useState, useEffect } from 'react';
import { useVideoPlayer } from '../hooks/useVideoPlayer';
import { getRandomViewerCount } from '../utils/helpers';
import { VIEWER_COUNT_UPDATE_INTERVAL } from '../utils/constants';

const VideoPlayerContext = createContext();

export const useVideoPlayerContext = () => {
  const context = useContext(VideoPlayerContext);
  if (!context) {
    throw new Error('useVideoPlayerContext must be used within a VideoPlayerProvider');
  }
  return context;
};

export const VideoPlayerProvider = ({ children }) => {
  const videoPlayer = useVideoPlayer();
  const [viewerCount, setViewerCount] = useState(1247);
  const [isLive, setIsLive] = useState(true);
  const [streamTitle, setStreamTitle] = useState('Fire TV Live Stream');
  const [streamCategory, setStreamCategory] = useState('Entertainment');
  const [streamQuality, setStreamQuality] = useState('1080p');

  // Simulate viewer count changes
  useEffect(() => {
    const interval = setInterval(() => {
      setViewerCount(prev => getRandomViewerCount(prev, 50));
    }, VIEWER_COUNT_UPDATE_INTERVAL);

    return () => clearInterval(interval);
  }, []);

  // Update stream stats based on playing state
  useEffect(() => {
    if (videoPlayer.isPlaying) {
      // Slightly increase viewer count when playing
      setViewerCount(prev => prev + Math.floor(Math.random() * 10));
    }
  }, [videoPlayer.isPlaying]);

  const contextValue = {
    // Video player state and actions
    ...videoPlayer,
    
    // Stream metadata
    viewerCount,
    isLive,
    streamTitle,
    streamCategory,
    streamQuality,
    
    // Stream actions
    setViewerCount,
    setIsLive,
    setStreamTitle,
    setStreamCategory,
    setStreamQuality
  };

  return (
    <VideoPlayerContext.Provider value={contextValue}>
      {children}
    </VideoPlayerContext.Provider>
  );
};