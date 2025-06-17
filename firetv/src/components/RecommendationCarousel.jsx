import React from 'react';

const recommendations = [
  {
    title: 'Feel‑Good Fiesta',
    cover: '/images/fiesta.jpg',
    reason: 'Group positivity soared'
  },
  {
    title: 'Comedy Blast',
    cover: '/images/comedy.jpg',
    reason: 'LOL spikes in chat'
  },
  {
    title: 'Cozy Mystery',
    cover: '/images/mystery.jpg',
    reason: 'Rainy‑day vibes'
  }
];

export default function RecommendationCarousel() {
  return (
    <section className="mt-6">
      <h2 className="text-xl text-white font-semibold mb-2">Recommended for You</h2>
      <div className="flex space-x-4 overflow-x-auto pb-4">
        {recommendations.map((rec) => (
          <div
            key={rec.title}
            className="min-w-[200px] bg-gray-800 rounded-lg overflow-hidden shadow-lg"
          >
            <img
              src={rec.cover}
              alt={rec.title}
              className="w-full h-32 object-cover"
            />
            <div className="p-2">
              <h3 className="text-md text-white font-medium">{rec.title}</h3>
              <p className="text-xs text-gray-400">{rec.reason}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
