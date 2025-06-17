// File: src/components/VideoCard.js
import React from 'react';
import { Play, Star, Clock } from 'lucide-react';

const VideoCard = ({ video }) => {
  return (
    <div className="group cursor-pointer transform transition-all duration-300 hover:scale-105">
      <div className="relative overflow-hidden rounded-lg bg-gray-800">
        <img
          src={video.thumbnail}
          alt={video.title}
          className="w-full h-48 object-cover transition-transform duration-300 group-hover:scale-110"
        />
        <div className="absolute inset-0 bg-black bg-opacity-40 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
          <div className="bg-white bg-opacity-20 rounded-full p-3 backdrop-blur-sm">
            <Play className="w-8 h-8 text-white" fill="white" />
          </div>
        </div>
        <div className="absolute top-2 right-2 bg-black bg-opacity-70 px-2 py-1 rounded text-xs flex items-center">
          <Clock className="w-3 h-3 mr-1" />
          {video.duration}
        </div>
      </div>
      <div className="mt-3">
        <h3 className="font-semibold text-white group-hover:text-orange-400 transition-colors">
          {video.title}
        </h3>
        <div className="flex items-center justify-between mt-1">
          <span className="text-sm text-gray-400">{video.category}</span>
          <div className="flex items-center text-sm text-gray-400">
            <Star className="w-4 h-4 text-yellow-500 mr-1" fill="currentColor" />
            {video.rating}
          </div>
        </div>
        <span className="text-xs text-gray-500">{video.year}</span>
      </div>
    </div>
  );
};

export default VideoCard;