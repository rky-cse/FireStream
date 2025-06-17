// src/App.jsx
import React, { useState } from 'react';
import VideoPlayer from './components/VideoPlayer';
import ChatPanel from './components/ChatPanel';
import HomePage from './components/HomePage';

function App() {
  const [showChat, setShowChat] = useState(true);
  const [currentView, setCurrentView] = useState('home'); // 'home' or 'player'
  const [selectedVideo, setSelectedVideo] = useState(null);

  const handleToggleChat = () => {
    setShowChat(!showChat);
  };

  const handleVideoSelect = (video) => {
    setSelectedVideo(video);
    setCurrentView('player');
  };

  const handleBackToHome = () => {
    setCurrentView('home');
    setSelectedVideo(null);
  };

  return (
    <div className="w-full h-screen bg-black">
      {currentView === 'home' ? (
        <HomePage onVideoSelect={handleVideoSelect} />
      ) : (
        <div className="flex h-full">
          <VideoPlayer 
            showChat={showChat} 
            onToggleChat={handleToggleChat}
            selectedVideo={selectedVideo}
            onBackToHome={handleBackToHome}
          />
          <ChatPanel isVisible={showChat} />
        </div>
      )}
    </div>
  );
}

export default App;