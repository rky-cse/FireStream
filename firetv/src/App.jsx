// src/App.jsx
import React, { useState } from 'react';
import VideoPlayer from './components/VideoPlayer';
import ChatPanel from './components/ChatPanel';

function App() {
  const [showChat, setShowChat] = useState(true);

  const handleToggleChat = () => {
    setShowChat(!showChat);
  };

  return (
    <div className="w-full h-screen bg-black">
      <div className="flex h-full">
        <VideoPlayer 
          showChat={showChat} 
          onToggleChat={handleToggleChat} 
        />
        <ChatPanel isVisible={showChat} />
      </div>
    </div>
  );
}

export default App;