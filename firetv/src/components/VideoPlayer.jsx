import React, { useState, useEffect, useRef } from 'react';
import { Send, MessageCircle, Users, X } from 'lucide-react';
import { useParams } from 'react-router-dom';

const VideoPlayer = () => {
  // Get userId from localStorage (string or null)
  const userId = localStorage.getItem('user_id');
  // Destructure contentId from URL params
  const { contentId } = useParams();

  const [videoMetadata, setVideoMetadata] = useState(null);
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [onlineUsers, setOnlineUsers] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const wsRef = useRef(null);
  const messagesEndRef = useRef(null);
  const videoRef = useRef(null);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

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

  // WebSocket chat connection
  useEffect(() => {
    if (!userId || !contentId) return;

    let retryTimer;

    const connect = () => {
      const ws = new WebSocket(`ws://localhost:8000/ws/chat/${contentId}/${userId}`);
      wsRef.current = ws;

      ws.onopen = () => setIsConnected(true);
      ws.onclose = () => {
        setIsConnected(false);
        retryTimer = setTimeout(connect, 3000);
      };
      ws.onerror = (e) => {
        console.error('WebSocket error:', e);
        ws.close();
      };
      ws.onmessage = ({ data }) => {
        const msg = JSON.parse(data);
        if (msg.type === 'message') {
          setMessages((m) => [
            ...m,
            {
              id: msg.id ?? Date.now(),
              type: 'user',
              userId: msg.user_id,
              username: msg.username,
              message: msg.message,
              timestamp: msg.timestamp,
            },
          ]);
        } else if (msg.type === 'user_joined' || msg.type === 'user_left') {
          setMessages((m) => [
            ...m,
            {
              id: Date.now(),
              type: 'system',
              message: `${msg.username} ${msg.type === 'user_joined' ? 'joined' : 'left'} the chat`,
              timestamp: new Date().toISOString(),
            },
          ]);
        } else if (msg.type === 'online_users') {
          setOnlineUsers(msg.users || []);
        }
      };
    };

    connect();
    return () => {
      clearTimeout(retryTimer);
      wsRef.current?.close();
    };
  }, [userId, contentId]);

  const sendMessage = (e) => {
    e.preventDefault();
    const text = newMessage.trim();
    if (!text || wsRef.current?.readyState !== WebSocket.OPEN) return;

    wsRef.current.send(
      JSON.stringify({
        type: 'message',
        message: text,
        user_id: userId,
        content_id: contentId,
      })
    );
    setNewMessage('');
  };

  const formatTime = (iso) =>
    new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

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

      {/* Chat Sidebar */}
      <div
        className={`fixed top-0 right-0 h-full bg-gray-900 border-l border-gray-700 transition-transform duration-300 z-50 w-80 ${
          isChatOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        <div className="p-4 border-b border-gray-700 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <MessageCircle size={20} />
            <h3 className="font-semibold">Live Chat</h3>
            <span
              className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-green-500' : 'bg-red-500'
              }`}
            />
          </div>
          <button onClick={() => setIsChatOpen(false)} className="text-gray-400 hover:text-white">
            <X size={20} />
          </button>
        </div>

        <div className="p-3 border-b border-gray-700">
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <Users size={16} />
            <span>{onlineUsers.length} online</span>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto h-[calc(100vh-200px)] p-4 space-y-3">
          {messages.map((msg) =>
            msg.type === 'system' ? (
              <div key={msg.id} className="text-center text-gray-500 text-sm italic">
                {msg.message}
              </div>
            ) : (
              <div key={msg.id} className="bg-gray-800 rounded-lg p-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-semibold text-blue-400 text-sm">
                    {msg.userId === userId ? 'You' : msg.username || `User ${msg.userId.slice(-4)}`}
                  </span>
                  <span className="text-xs text-gray-500">{formatTime(msg.timestamp)}</span>
                </div>
                <p className="text-sm">{msg.message}</p>
              </div>
            )
          )}
          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={sendMessage} className="p-4 border-t border-gray-700">
          <div className="flex gap-2">
            <input
              type="text"
              value={newMessage}
              onChange={(e) => setNewMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendMessage(e)}
              placeholder={isConnected ? 'Type a message...' : 'Connecting...'}
              disabled={!isConnected}
              maxLength={500}
              className="flex-1 bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500 disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={!newMessage.trim() || !isConnected}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed p-2 rounded-lg transition-colors"
            >
              <Send size={16} />
            </button>
          </div>
        </form>
      </div>

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
