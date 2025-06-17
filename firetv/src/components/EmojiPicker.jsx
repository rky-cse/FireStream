// src/components/EmojiPicker.jsx
import React from 'react';
import { emojis } from '../utils/mockData';

const EmojiPicker = ({ onEmojiSelect, isOpen, onClose }) => {
  if (!isOpen) return null;
  
  return (
    <div className="absolute bottom-full left-0 mb-2 bg-gray-800 rounded-lg p-3 shadow-xl border border-gray-600 z-50">
      <div className="grid grid-cols-6 gap-2">
        {emojis.map((emoji, index) => (
          <button
            key={index}
            onClick={() => onEmojiSelect(emoji)}
            className="text-2xl hover:bg-gray-700 rounded p-1 transition-colors"
          >
            {emoji}
          </button>
        ))}
      </div>
    </div>
  );
};

export default EmojiPicker;