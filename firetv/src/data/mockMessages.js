import { getCurrentTimestamp, extractFirstEmoji } from '../utils/helpers';

export const initialMessages = [
  {
    id: 1,
    user: 'StreamFan42',
    message: 'This is incredible! 🔥🔥',
    timestamp: '2:35 PM',
    emoji: '🔥',
    type: 'message'
  },
  {
    id: 2,
    user: 'TechWatcher',
    message: 'Amazing quality stream!',
    timestamp: '2:36 PM',
    emoji: '⭐',
    type: 'message'
  },
  {
    id: 3,
    user: 'LiveViewer99',
    message: '❤️❤️❤️',
    timestamp: '2:37 PM',
    emoji: '❤️',
    type: 'emoji'
  },
  {
    id: 4,
    user: 'CoolUser123',
    message: 'Love the Fire TV vibes! 🚀',
    timestamp: '2:38 PM',
    emoji: '🚀',
    type: 'message'
  },
  {
    id: 5,
    user: 'StreamMaster',
    message: 'Best stream of the day 💯',
    timestamp: '2:39 PM',
    emoji: '💯',
    type: 'message'
  },
  {
    id: 6,
    user: 'EpicViewer',
    message: '👏👏👏',
    timestamp: '2:40 PM',
    emoji: '👏',
    type: 'emoji'
  },
  {
    id: 7,
    user: 'FireFan88',
    message: 'This interface is so smooth!',
    timestamp: '2:41 PM',
    emoji: '✨',
    type: 'message'
  }
];

// Generate random messages for demo
export const generateRandomMessage = () => {
  const users = [
    'StreamLover', 'TechFan', 'VideoWatcher', 'LiveUser', 'ChatMaster',
    'ViewerPro', 'StreamKing', 'MediaFan', 'TechViewer', 'LiveStream99'
  ];

  const messages = [
    'This is amazing! 🔥',
    'Great content! 👍',
    'Love this stream ❤️',
    'Awesome quality! ⭐',
    'Keep it up! 💪',
    'Perfect! 💯',
    'So good! 😍',
    'Epic stream! 🚀',
    'Incredible! ⚡',
    'Best stream ever! 🎉'
  ];

  const randomUser = users[Math.floor(Math.random() * users.length)];
  const randomMessage = messages[Math.floor(Math.random() * messages.length)];

  return {
    id: Date.now() + Math.random(),
    user: randomUser,
    message: randomMessage,
    timestamp: getCurrentTimestamp(),
    emoji: extractFirstEmoji(randomMessage),
    type: 'message'
  };
};

// Generate system messages
export const createSystemMessage = (content) => ({
  id: Date.now() + Math.random(),
  user: 'System',
  message: content,
  timestamp: getCurrentTimestamp(),
  emoji: '🔔',
  type: 'system'
});

// Welcome message
export const welcomeMessage = createSystemMessage('Welcome to the Fire TV Live Stream! Enjoy the show! 🎬');