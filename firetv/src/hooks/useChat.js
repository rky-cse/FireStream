import { useState, useCallback, useRef, useEffect } from 'react';
import { initialMessages, generateRandomMessage } from '../data/mockMessages';
import { validateMessage, generateId, extractFirstEmoji, getCurrentTimestamp } from '../utils/helpers';
import { MAX_CHAT_MESSAGES, CHAT_TYPES } from '../utils/constants';

export const useChat = () => {
  const [messages, setMessages] = useState(initialMessages);
  const [isVisible, setIsVisible] = useState(true);
  const [unreadCount, setUnreadCount] = useState(0);
  const [isAutoScroll, setIsAutoScroll] = useState(true);
  
  const chatEndRef = useRef(null);
  const messageCountRef = useRef(initialMessages.length);

  // Add new message
  const addMessage = useCallback((messageData) => {
    if (!validateMessage(messageData.message)) return false;

    const newMessage = {
      id: generateId(),
      user: messageData.user || 'Anonymous',
      message: messageData.message,
      timestamp: getCurrentTimestamp(),
      emoji: messageData.emoji || extractFirstEmoji(messageData.message),
      type: messageData.type || CHAT_TYPES.MESSAGE
    };

    setMessages(prev => {
      const updated = [...prev, newMessage];
      // Limit messages to prevent memory issues
      if (updated.length > MAX_CHAT_MESSAGES) {
        return updated.slice(-MAX_CHAT_MESSAGES);
      }
      return updated;
    });

    // Update unread count if chat is hidden
    if (!isVisible) {
      setUnreadCount(prev => prev + 1);
    }

    messageCountRef.current += 1;
    return true;
  }, [isVisible]);

  // Send user message
  const sendMessage = useCallback((content, user = 'You') => {
    return addMessage({
      user,
      message: content,
      type: CHAT_TYPES.MESSAGE
    });
  }, [addMessage]);

  // Send emoji
  const sendEmoji = useCallback((emoji, user = 'You') => {
    return addMessage({
      user,
      message: emoji,
      emoji,
      type: CHAT_TYPES.EMOJI
    });
  }, [addMessage]);

  // Clear chat
  const clearChat = useCallback(() => {
    setMessages([]);
    setUnreadCount(0);
    messageCountRef.current = 0;
  }, []);

  // Toggle chat visibility
  const toggleChatVisibility = useCallback(() => {
    setIsVisible(prev => {
      if (!prev) {
        setUnreadCount(0); // Clear unread when opening
      }
      return !prev;
    });
  }, []);

  // Show chat
  const showChat = useCallback(() => {
    setIsVisible(true);
    setUnreadCount(0);
  }, []);

  // Hide chat
  const hideChat = useCallback(() => {
    setIsVisible(false);
  }, []);

  // Handle scroll behavior
  const handleScroll = useCallback((event) => {
    const { scrollTop, scrollHeight, clientHeight } = event.target;
    const isAtBottom = scrollHeight - scrollTop === clientHeight;
    setIsAutoScroll(isAtBottom);
  }, []);

  // Scroll to bottom
  const scrollToBottom = useCallback(() => {
    if (isAutoScroll && chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [isAutoScroll]);

  // Auto scroll when new messages arrive
  useEffect(() => {
    if (isAutoScroll) {
      scrollToBottom();
    }
  }, [messages, isAutoScroll, scrollToBottom]);

  // Simulate random messages for demo
  useEffect(() => {
    const interval = setInterval(() => {
      // Random chance to add a message (20% every 10 seconds)
      if (Math.random() < 0.2) {
        const randomMessage = generateRandomMessage();
        addMessage(randomMessage);
      }
    }, 10000);

    return () => clearInterval(interval);
  }, [addMessage]);

  // Get latest messages (for notifications, etc.)
  const getLatestMessages = useCallback((count = 5) => {
    return messages.slice(-count);
  }, [messages]);

  // Get message stats
  const getMessageStats = useCallback(() => {
    return {
      total: messages.length,
      unread: unreadCount,
      users: new Set(messages.map(msg => msg.user)).size
    };
  }, [messages, unreadCount]);

  return {
    // State
    messages,
    isVisible,
    unreadCount,
    isAutoScroll,
    chatEndRef,
    messageCount: messageCountRef.current,

    // Actions
    addMessage,
    sendMessage,
    sendEmoji,
    clearChat,
    toggleChatVisibility,
    showChat,
    hideChat,
    handleScroll,
    scrollToBottom,

    // Getters
    getLatestMessages,
    getMessageStats
  };
};