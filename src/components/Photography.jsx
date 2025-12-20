import React, { useState } from 'react';
import { Camera, MapPin, Calendar, X } from 'lucide-react';

const Photography = ({ photography }) => {
  const [selectedPhoto, setSelectedPhoto] = useState(null);

  const formatDate = (dateString) => {
    const [year, month] = dateString.split('-');
    const date = new Date(year, month - 1);
    return date.toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
  };

  return (
    <section id="photography" className="section-padding">
      <div className="container-width">
        <h2 className="text-4xl md:text-5xl font-bold text-dark-text-primary mb-4">
          Photography & Exploration
        </h2>
        <div className="w-20 h-1 bg-dark-border mb-12"></div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {photography.map((photo, index) => (
            <div
              key={index}
              onClick={() => setSelectedPhoto(photo)}
              className="group relative aspect-square overflow-hidden rounded-lg cursor-pointer bg-dark-surface border border-dark-border hover:border-dark-text-muted transition-all"
            >
              <img
                src={photo.image}
                alt={photo.title}
                className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
                <div className="absolute bottom-0 left-0 right-0 p-4">
                  <h3 className="text-white font-semibold mb-1">{photo.title}</h3>
                  <div className="flex items-center text-gray-300 text-xs">
                    <MapPin size={12} className="mr-1" />
                    <span>{photo.location}</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Lightbox Modal */}
        {selectedPhoto && (
          <div
            className="fixed inset-0 bg-black/95 z-50 flex items-center justify-center p-4"
            onClick={() => setSelectedPhoto(null)}
          >
            <button
              onClick={() => setSelectedPhoto(null)}
              className="absolute top-6 right-6 text-white hover:text-gray-300 transition-colors"
            >
              <X size={32} />
            </button>

            <div className="max-w-6xl w-full" onClick={(e) => e.stopPropagation()}>
              <img
                src={selectedPhoto.image}
                alt={selectedPhoto.title}
                className="w-full h-auto rounded-lg"
              />
              <div className="mt-6 text-center">
                <h3 className="text-2xl font-bold text-white mb-2">
                  {selectedPhoto.title}
                </h3>
                <div className="flex items-center justify-center gap-6 text-gray-300 mb-3">
                  <div className="flex items-center">
                    <MapPin size={16} className="mr-2" />
                    {selectedPhoto.location}
                  </div>
                  <div className="flex items-center">
                    <Calendar size={16} className="mr-2" />
                    {formatDate(selectedPhoto.date)}
                  </div>
                </div>
                <p className="text-gray-400">{selectedPhoto.description}</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
};

export default Photography;
