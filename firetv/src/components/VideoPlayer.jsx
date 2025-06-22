import React, { useState, useEffect, useRef } from 'react';
import { MessageCircle } from 'lucide-react';
import { useParams } from 'react-router-dom';
import LiveChat from './LiveChat'; // Adjust the path as needed

const VideoPlayer = () => {
  const userId = localStorage.getItem('user_id');
  const { contentId } = useParams();

  const [videoMetadata, setVideoMetadata] = useState(null);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const videoRef = useRef(null);

  // Fetch video metadata
  useEffect(() => {
    if (!contentId) return;

    const fetchMetadata = async () => {
      try {
        const res = await fetch(`http://localhost:8000/api/video/metadata/${contentId}`);
        if (!res.ok) throw new Error('Failed to fetch metadata');
        const data = await res.json();
        setVideoMetadata(data);
      } catch (err) {
        console.error(err);
        setError('Failed to load video metadata');
      } finally {
        setLoading(false);
      }
    };

    fetchMetadata();
  }, [contentId]);

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <span className="text-white text-xl">Loading video...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <span className="text-red-500 text-xl">{error}</span>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black text-white relative">
      {/* Video */}
      <div className="relative">
        <video
          ref={videoRef}
          src={`http://localhost:8000/api/video/stream/${contentId}`}
          poster="/api/placeholder/1920/1080"
          controls
          className="w-full h-[60vh] object-contain bg-black"
        >
          Your browser does not support video.
        </video>
        <button
          onClick={() => setIsChatOpen((o) => !o)}
          className="absolute top-4 right-4 bg-blue-600 hover:bg-blue-700 p-3 rounded-full shadow-lg z-10"
        >
          <MessageCircle size={24} />
        </button>
      </div>

      {/* Metadata */}
      {videoMetadata && (
        <div className="p-6 max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold mb-2">{videoMetadata.title}</h1>
          <div className="flex flex-wrap gap-4 text-gray-300 mb-4">
            <span>{videoMetadata.release_year}</span>
            <span>•</span>
            <span>{videoMetadata.duration} min</span>
            <span>•</span>
            <span>{videoMetadata.genre}</span>
            <span>•</span>
            <span>Rating: {videoMetadata.rating}/10</span>
          </div>
          <p className="text-gray-300 mb-4">{videoMetadata.description}</p>
          <div className="text-sm text-gray-400">
            <p>
              <strong>Director:</strong> {videoMetadata.director}
            </p>
            <p>
              <strong>Cast:</strong> {videoMetadata.actors}
            </p>
          </div>
        </div>
      )}

      {/* Live Chat Component */}
      <LiveChat
        contentId={contentId}
        userId={userId}
        isOpen={isChatOpen}
        onClose={() => setIsChatOpen(false)}
      />

      {/* Overlay for mobile */}
      {isChatOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={() => setIsChatOpen(false)}
        />
      )}
    </div>
  );
};

export default VideoPlayer;