// File: src/components/Header.js
import React from 'react';

const Header = () => {
  return (
    <header className="flex items-center justify-between p-6 bg-black bg-opacity-50">
      <div className="flex items-center space-x-4">
        <div className="text-2xl font-bold text-orange-500">fire tv</div>
      </div>
      <div className="flex items-center space-x-4">
        <div className="text-sm">Good evening</div>
        <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-sm font-semibold">
          U
        </div>
      </div>
    </header>
  );
};

export default Header;