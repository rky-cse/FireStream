import React, { useState, useEffect, useRef } from 'react';
import { X } from 'lucide-react'; // Import Lucide icon

const Notification = ({ userId }) => {
  const [notifications, setNotifications] = useState([]);
  const ws = useRef(null);

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    ws.current = new WebSocket(`${protocol}//${host}/api/video/ws/${userId}`);

    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setNotifications(prev => [
        ...prev,
        {
          id: Date.now(),
          imagePath: data.image_path,
          imageName: data.image_name.replace(/\.[^/.]+$/, "")
        }
      ]);
    };

    return () => ws.current?.close();
  }, [userId]);

  useEffect(() => {
    if (notifications.length > 0) {
      const timer = setTimeout(() => {
        setNotifications(prev => prev.slice(1));
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [notifications]);

  const handleClose = (id) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  return (
    <div className="fixed bottom-4 right-4 z-50 space-y-2">
      {notifications.map((notification) => (
        <div 
          key={notification.id}
          className="w-72 bg-white rounded-lg shadow-lg overflow-hidden animate-fade-in"
        >
          <div className="relative">
            <img 
              src={notification.imagePath}
              alt={notification.imageName}
              className="w-full h-40 object-cover"
            />
            <button
              onClick={() => handleClose(notification.id)}
              className="absolute top-1 right-1 p-1 rounded-full bg-black/50 hover:bg-black/70"
            >
              <X className="w-4 h-4 text-white" /> {/* Lucide icon */}
            </button>
          </div>
          <div className="p-3">
            <h3 className="font-medium text-gray-900 truncate">
              {notification.imageName}
            </h3>
            <p className="text-sm text-gray-500">Product detected</p>
          </div>
        </div>
      ))}
    </div>
  );
};

// Add to your global CSS
const styleElement = document.createElement('style');
styleElement.innerHTML = `
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .animate-fade-in {
    animation: fadeIn 0.3s ease-out forwards;
  }
`;
document.head.appendChild(styleElement);

export default Notification;