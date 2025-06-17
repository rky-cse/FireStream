// src/components/ChatMessage.jsx
import React from 'react';

const ChatMessage = ({ message, isOwn }) => {
  return (
    <div className={`mb-3 ${isOwn ? 'text-right' : 'text-left'}`}>
      <div className={`inline-block max-w-xs px-3 py-2 rounded-lg ${
        isOwn 
          ? 'bg-blue-600 text-white' 
          : 'bg-gray-700 text-gray-100'
      }`}>
        {!isOwn && <div className="text-xs text-blue-400 mb-1">{message.user}</div>}
        <div className="text-sm">{message.message}</div>
        <div className="text-xs opacity-70 mt-1">{message.timestamp}</div>
      </div>
    </div>
  );
};

export default ChatMessage;