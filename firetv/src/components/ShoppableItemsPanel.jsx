import React from 'react';

const products = [
  {
    name: "Actor's Sunglasses",
    image: '/images/sunglasses.png',
    link: '#'
  },
  {
    name: 'Hero Jacket',
    image: '/images/jacket.png',
    link: '#'
  }
];

export default function ShoppableItemsPanel() {
  return (
    <aside className="absolute top-1/4 right-6 w-56 bg-gray-800 bg-opacity-75 p-4 rounded-lg">
      <h4 className="text-white text-md font-semibold mb-3">Shoppable Items</h4>
      <div className="space-y-3">
        {products.map((item) => (
          <a
            key={item.name}
            href={item.link}
            className="flex items-center space-x-2 hover:bg-gray-700 p-2 rounded"
          >
            <img
              src={item.image}
              alt={item.name}
              className="w-10 h-10 object-cover rounded"
            />
            <span className="text-sm text-white">{item.name}</span>
          </a>
        ))}
      </div>
    </aside>
  );
}
