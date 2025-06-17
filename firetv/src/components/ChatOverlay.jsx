import React from 'react';

const mockMessages = [
  { user: 'Alice', text: 'This scene is hilarious!' },
  { user: 'Bob', text: 'LOL 😂' },
  { user: 'Eve', text: 'That cliffhanger though…' }
];

export default function ChatOverlay() {
  return (
    <div className="absolute bottom-4 left-24 w-2/3 p-4 bg-black bg-opacity-50 rounded-lg">
      <div className="max-h-36 overflow-y-auto space-y-1">
        {mockMessages.map((msg, idx) => (
          <div key={idx} className="text-sm text-white">
            <span className="font-semibold text-yellow-300">{msg.user}:</span>{' '}
            {msg.text}
          </div>
        ))}
      </div>
    </div>
  );
}
