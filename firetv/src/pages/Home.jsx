import React from 'react';
import Sidebar from '../components/Sidebar';
import RecommendationCarousel from '../components/RecommendationCarousel';
import ChatOverlay from '../components/ChatOverlay';
import ShoppableItemsPanel from '../components/ShoppableItemsPanel';

export default function Home() {
  return (
    <div className="flex h-screen bg-gray-900">
      {/* Sidebar */}
      <Sidebar />

      {/* Main content area */}
      <main className="relative flex-1 p-6 overflow-hidden">
        {/* Header */}
        <header className="flex justify-between items-center mb-6">
          <h1 className="text-2xl text-white font-bold">FireTV AI Recommender</h1>
          <div className="text-sm text-gray-400">Group Mood: 😊 +5</div>
        </header>

        {/* Recommendations */}
        <RecommendationCarousel />

        {/* Overlays */}
        <ChatOverlay />
        <ShoppableItemsPanel />
      </main>
    </div>
  );
}
