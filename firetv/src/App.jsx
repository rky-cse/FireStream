import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './components/HomePage';
import VideoPlayer from './components/VideoPlayer';
import VoiceProsodyComponent from './components/VoiceProsodyComponent';
import VoiceSearchComponent from './components/VoiceSearchComponent';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/video/:contentId" element={<VideoPlayer />} />
        <Route path="/search" element={<VoiceProsodyComponent />} />
        <Route path="/voice-search" element={<VoiceSearchComponent />} />
        
        {/* Optional: Redirect or 404 handling */}
        <Route path="*" element={<HomePage />} />
      </Routes>
    </Router>
  );
};

export default App;