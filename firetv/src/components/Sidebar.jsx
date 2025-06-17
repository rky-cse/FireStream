// src/components/Sidebar.jsx

import React from 'react';
import {
  HomeIcon,
  FlameIcon,
  FilmIcon,
  ShoppingCartIcon,
  MessageCircleIcon
} from 'lucide-react';

const navItems = [
  { name: 'Home',     icon: HomeIcon },
  { name: 'Trending', icon: FlameIcon },
  { name: 'Movies',   icon: FilmIcon },
  { name: 'Shop',     icon: ShoppingCartIcon },
  { name: 'Chat',     icon: MessageCircleIcon }
];

export default function Sidebar() {
  return (
    <aside className="w-20 bg-gray-900 text-gray-300 flex flex-col items-center py-4 space-y-6">
      {navItems.map((item) => (
        <button
          key={item.name}
          className="flex flex-col items-center focus:outline-none hover:text-white"
        >
          <item.icon size={24} />
          <span className="text-xs mt-1">{item.name}</span>
        </button>
      ))}
    </aside>
  );
}
