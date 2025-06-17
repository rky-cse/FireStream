// src/components/ChatPanel.jsx
import React, { useState, useEffect, useRef } from 'react';
import { Send, Smile, Heart, ThumbsUp, Star } from 'lucide-react';
import ChatMessage from './ChatMessage';
import EmojiPicker from './EmojiPicker';
import { mockMessages, mockStreamData } from '../utils/mockData';

const ChatPanel = ({ isVisible }) => {
  const [messages, setMessages] = useState(mockMessages);
  const [newMessage, setNewMessage] = useState('');
  const [showEmojiPicker, setShowEmojiPicker] = useState(false);
  const [reactions, setReactions] = useState({
    hearts: 23,
    thumbsUp: 45,
    stars: 12
  });
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Simulate new messages coming in
  useEffect(() => {
    const interval = setInterval(() => {
      if (Math.random() > 0.7) { // 30% chance every 5 seconds
        const randomMessages = [
          'This is epic! 🚀',
          'Amazing stream quality',
          'Love this part! ❤️',
          'Best movie ever 🎬',
          'HD looks fantastic',
          'Great sound quality 🔊'
        ];
        const randomUsers = ['StreamLover', 'MovieBuff', 'ActionFan', 'CinemaGeek', 'HDWatcher'];
        
        const newMsg = {
          id: Date.now(),
          user: randomUsers[Math.floor(Math.random() * randomUsers.length)],
          message: randomMessages[Math.floor(Math.random() * randomMessages.length)],
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          isOwn: false
        };
        
        setMessages(prev => [...prev, newMsg]);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const handleSendMessage = () => {
    if (newMessage.trim()) {
      const message = {
        id: Date.now(),
        user: 'You',
        message: newMessage,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        isOwn: true
      };
      setMessages([...messages, message]);
      setNewMessage('');
    }
  };

  const handleEmojiSelect = (emoji) => {
    setNewMessage(prev => prev + emoji);
    setShowEmojiPicker(false);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };

  const handleReaction = (type) => {
    setReactions(prev => ({
      ...prev,
      [type]: prev[type] + 1
    }));
  };

  if (!isVisible) return null;

  return (
    <div className="w-80 bg-gray-900 border-l border-gray-700 flex flex-col">
      {/* Chat Header */}
      <div className="p-4 border-b border-gray-700">
        <h3 className="text-white font-semibold flex items-center">
          <div className="w-2 h-2 bg-red-500 rounded-full mr-2 animate-pulse"></div>
          Live Chat
        </h3>
        <p className="text-gray-400 text-sm">{mockStreamData.viewerCount.toLocaleString()} viewers</p>
      </div>

      {/* Messages */}
      <div className="flex-1 p-4 overflow-y-auto">
        {messages.map((message) => (
          <ChatMessage key={message.id} message={message} isOwn={message.isOwn} />
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Quick Reactions */}
      <div className="p-3 border-t border-gray-700">
        <div className="flex space-x-2 mb-3">
          <button 
            onClick={() => handleReaction('hearts')}
            className="flex items-center space-x-1 bg-gray-800 hover:bg-gray-700 px-3 py-1 rounded-full text-sm text-white transition-colors"
          >
            <Heart className="w-4 h-4 text-red-500" />
            <span>{reactions.hearts}</span>
          </button>
          <button 
            onClick={() => handleReaction('thumbsUp')}
            className="flex items-center space-x-1 bg-gray-800 hover:bg-gray-700 px-3 py-1 rounded-full text-sm text-white transition-colors"
          >
            <ThumbsUp className="w-4 h-4 text-blue-500" />
            <span>{reactions.thumbsUp}</span>
          </button>
          <button 
            onClick={() => handleReaction('stars')}
            className="flex items-center space-x-1 bg-gray-800 hover:bg-gray-700 px-3 py-1 rounded-full text-sm text-white transition-colors"
          >
            <Star className="w-4 h-4 text-yellow-500" />
            <span>{reactions.stars}</span>
          </button>
        </div>

        {/* Message Input */}
        <div className="relative">
          <div className="flex space-x-2">
            <div className="relative flex-1">
              <input
                type="text"
                value={newMessage}
                onChange={(e) => setNewMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type a message..."
                className="w-full bg-gray-800 text-white px-3 py-2 rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none text-sm"
              />
              <button
                onClick={() => setShowEmojiPicker(!showEmojiPicker)}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white"
              >
                <Smile className="w-4 h-4" />
              </button>
              <EmojiPicker
                isOpen={showEmojiPicker}
                onEmojiSelect={handleEmojiSelect}
                onClose={() => setShowEmojiPicker(false)}
              />
            </div>
            <button
              onClick={handleSendMessage}
              className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-2 rounded-lg transition-colors"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatPanel;