import { useState, useEffect, useCallback, useRef } from 'react';
import { VIDEO_STATES, CONTROL_HIDE_DELAY } from '../utils/constants';

export const useVideoPlayer = () => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [volume, setVolume] = useState(1);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(300); // 5 minutes default
  const [showControls, setShowControls] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [videoState, setVideoState] = useState(VIDEO_STATES.PAUSED);
  const [isBuffering, setIsBuffering] = useState(false);

  const videoRef = useRef(null);
  const controlsTimeoutRef = useRef(null);

  // Toggle play/pause
  const togglePlay = useCallback(() => {
    setIsPlaying(prev => !prev);
    setVideoState(prev => 
      prev === VIDEO_STATES.PLAYING ? VIDEO_STATES.PAUSED : VIDEO_STATES.PLAYING
    );
  }, []);

  // Toggle mute
  const toggleMute = useCallback(() => {
    setIsMuted(prev => !prev);
  }, []);

  // Set volume
  const handleVolumeChange = useCallback((newVolume) => {
    setVolume(newVolume);
    if (newVolume === 0) {
      setIsMuted(true);
    } else if (isMuted) {
      setIsMuted(false);
    }
  }, [isMuted]);

  // Seek to specific time
  const seekTo = useCallback((time) => {
    const clampedTime = Math.max(0, Math.min(time, duration));
    setCurrentTime(clampedTime);
  }, [duration]);

  // Handle progress update
  const handleProgress = useCallback((time) => {
    setCurrentTime(time);
  }, []);

  // Show controls
  const showVideoControls = useCallback(() => {
    setShowControls(true);
    
    // Clear existing timeout
    if (controlsTimeoutRef.current) {
      clearTimeout(controlsTimeoutRef.current);
    }

    // Hide controls after delay if playing
    if (isPlaying) {
      controlsTimeoutRef.current = setTimeout(() => {
        setShowControls(false);
      }, CONTROL_HIDE_DELAY);
    }
  }, [isPlaying]);

  // Hide controls
  const hideVideoControls = useCallback(() => {
    if (controlsTimeoutRef.current) {
      clearTimeout(controlsTimeoutRef.current);
    }
    setShowControls(false);
  }, []);

  // Toggle fullscreen
  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen?.();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen?.();
      setIsFullscreen(false);
    }
  }, []);

  // Handle keyboard shortcuts
  const handleKeyPress = useCallback((event) => {
    switch (event.code) {
      case 'Space':
        event.preventDefault();
        togglePlay();
        break;
      case 'KeyM':
        toggleMute();
        break;
      case 'KeyF':
        toggleFullscreen();
        break;
      case 'ArrowLeft':
        seekTo(currentTime - 10);
        break;
      case 'ArrowRight':
        seekTo(currentTime + 10);
        break;
      case 'ArrowUp':
        event.preventDefault();
        handleVolumeChange(Math.min(1, volume + 0.1));
        break;
      case 'ArrowDown':
        event.preventDefault();
        handleVolumeChange(Math.max(0, volume - 0.1));
        break;
      default:
        break;
    }
  }, [togglePlay, toggleMute, toggleFullscreen, seekTo, currentTime, handleVolumeChange, volume]);

  // Update current time when playing
  useEffect(() => {
    let interval;
    if (isPlaying && !isBuffering) {
      interval = setInterval(() => {
        setCurrentTime(prev => {
          if (prev >= duration) {
            setIsPlaying(false);
            setVideoState(VIDEO_STATES.PAUSED);
            return duration;
          }
          return prev + 1;
        });
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isPlaying, isBuffering, duration]);

  // Auto-hide controls when playing
  useEffect(() => {
    if (isPlaying && showControls) {
      const timeout = setTimeout(() => {
        setShowControls(false);
      }, CONTROL_HIDE_DELAY);
      return () => clearTimeout(timeout);
    }
  }, [isPlaying, showControls]);

  // Listen for fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  // Keyboard event listeners
  useEffect(() => {
    document.addEventListener('keydown', handleKeyPress);
    return () => document.removeEventListener('keydown', handleKeyPress);
  }, [handleKeyPress]);

  // Calculate progress percentage
  const progressPercentage = (currentTime / duration) * 100;

  return {
    // State
    isPlaying,
    isMuted,
    volume,
    currentTime,
    duration,
    showControls,
    isFullscreen,
    videoState,
    isBuffering,
    progressPercentage,
    videoRef,

    // Actions
    togglePlay,
    toggleMute,
    handleVolumeChange,
    seekTo,
    handleProgress,
    showVideoControls,
    hideVideoControls,
    toggleFullscreen,
    setDuration,
    setIsBuffering
  };
};