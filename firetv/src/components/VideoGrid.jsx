// File: src/components/VideoGrid.js
import React from 'react';
import VideoCard from './VideoCard';

const VideoGrid = ({ videos }) => {
  return (
    <div className="px-6 pb-12">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {videos.map((video) => (
          <VideoCard key={video.id} video={video} />
        ))}
      </div>

      {videos.length === 0 && (
        <div className="text-center py-12">
          <div className="text-gray-400 text-lg">No results found</div>
          <div className="text-gray-500 text-sm mt-2">
            Try searching for something else or check your spelling
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoGrid;