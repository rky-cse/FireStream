import React, { useState, useEffect, useRef } from 'react';
import { Send, Users, X, Sparkles, MessageCircle } from 'lucide-react';

const LiveChat = ({ contentId, userId, isOpen, onClose }) => {
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [onlineUsers, setOnlineUsers] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [chatSummary, setChatSummary] = useState(null);
  const [highlightSummary, setHighlightSummary] = useState(false);
  const [connectionAttempts, setConnectionAttempts] = useState(0);

  const wsRef = useRef(null);
  const messagesEndRef = useRef(null);
  const summaryRef = useRef(null);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Highlight new summary and scroll to it
  useEffect(() => {
    if (chatSummary) {
      setHighlightSummary(true);
      summaryRef.current?.scrollIntoView({ behavior: 'smooth' });
      const timer = setTimeout(() => setHighlightSummary(false), 1500);
      return () => clearTimeout(timer);
    }
  }, [chatSummary]);

  // WebSocket chat connection with reconnect logic
  useEffect(() => {
    if (!userId || !contentId || !isOpen) return;

    let retryTimer;
    const maxRetries = 5;
    const retryDelay = 3000;

    const connect = () => {
      try {
        const ws = new WebSocket(`ws://localhost:8000/ws/chat/${contentId}/${userId}`);
        wsRef.current = ws;

        ws.onopen = () => {
          setIsConnected(true);
          setConnectionAttempts(0); // Reset attempts on successful connection
        };

        ws.onclose = () => {
          setIsConnected(false);
          if (connectionAttempts < maxRetries) {
            retryTimer = setTimeout(() => {
              setConnectionAttempts(prev => prev + 1);
              connect();
            }, retryDelay);
          }
        };

        ws.onerror = (e) => {
          console.error('WebSocket error:', e);
          ws.close();
        };

        ws.onmessage = ({ data }) => {
          try {
            const msg = JSON.parse(data);
            switch (msg.type) {
              case 'message':
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
                break;
              case 'user_joined':
              case 'user_left':
                setMessages((m) => [
                  ...m,
                  {
                    id: Date.now(),
                    type: 'system',
                    message: `${msg.username} ${msg.type === 'user_joined' ? 'joined' : 'left'} the chat`,
                    timestamp: new Date().toISOString(),
                  },
                ]);
                break;
              case 'online_users':
                setOnlineUsers(msg.users || []);
                break;
              case 'chat_summary':
                setChatSummary({
                  text: msg.summary,
                  messageCount: msg.message_count,
                  timestamp: msg.timestamp
                });
                break;
              default:
                console.log('Unhandled message type:', msg.type);
            }
          } catch (parseError) {
            console.error('Error parsing WebSocket message:', parseError);
          }
        };
      } catch (wsError) {
        console.error('WebSocket connection error:', wsError);
      }
    };

    connect();

    return () => {
      clearTimeout(retryTimer);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [userId, contentId, isOpen, connectionAttempts]);

  const sendMessage = (e) => {
    e.preventDefault();
    const text = newMessage.trim();
    if (!text || wsRef.current?.readyState !== WebSocket.OPEN) return;

    try {
      wsRef.current.send(
        JSON.stringify({
          type: 'message',
          message: text,
          user_id: userId,
          content_id: contentId,
        })
      );
      setNewMessage('');
    } catch (sendError) {
      console.error('Error sending message:', sendError);
    }
  };

  const refreshSummary = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify({ type: 'get_summary' }));
      } catch (error) {
        console.error('Error refreshing summary:', error);
      }
    }
  };

  const formatTime = (iso) =>
    new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  if (!isOpen) return null;

   return (
    <div className="fixed top-0 right-0 h-full bg-gray-900 border-l border-gray-700 transition-transform duration-300 z-50 w-80 flex flex-col">
      {/* Header */}
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
        <button onClick={onClose} className="text-gray-400 hover:text-white">
          <X size={20} />
        </button>
      </div>

      {/* Online Users */}
      <div className="p-3 border-b border-gray-700 flex justify-between items-center">
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <Users size={16} />
          <span>{onlineUsers.length} online</span>
        </div>
        <button 
          onClick={refreshSummary}
          className="text-xs bg-gray-700 hover:bg-gray-600 px-2 py-1 rounded flex items-center gap-1"
          disabled={!isConnected}
        >
          <Sparkles size={12} />
          Refresh Summary
        </button>
      </div>

      {/* Chat Summary Section */}
      {chatSummary && (
        <div 
          ref={summaryRef}
          className={`p-4 border-b border-gray-700 bg-gray-800/50 ${
            highlightSummary ? 'animate-highlight' : ''
          }`}
        >
          <div className="flex items-center gap-2 text-yellow-400 mb-1">
            <Sparkles size={16} />
            <span className="font-medium text-sm">Chat Summary</span>
          </div>
          <p className="text-sm text-gray-100 mb-1">{chatSummary.text}</p>
          <div className="flex justify-between items-center">
            <span className="text-xs text-gray-400">
              Based on {chatSummary.messageCount} recent messages
            </span>
            <span className="text-xs text-gray-500">
              {formatTime(chatSummary.timestamp)}
            </span>
          </div>
        </div>
      )}

      {/* Messages Container - now uses flex-grow */}
      <div className="flex-grow overflow-y-auto p-4 space-y-3">
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

      {/* Input Form - fixed at bottom */}
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
  );
};

export default LiveChat;