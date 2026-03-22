import React, { useState } from 'react';
import { Hotel, MapPin } from 'lucide-react';
import './Block.css';

export default function HotelListBlock({ data }) {
  const hotels = data?.items || data?.hotels || [];
  const [show, setShow] = useState(5);

  if (data?.error) {
    return (
      <div className="block-card">
        <div className="block-header">
          <Hotel size={15} strokeWidth={1.5} />
          <span>Hotels</span>
        </div>
        <p className="block-error">Could not load hotels: {data.error?.message || JSON.stringify(data.error)}</p>
      </div>
    );
  }

  return (
    <div className="block-card">
      <div className="block-header">
        <Hotel size={15} strokeWidth={1.5} />
        <span>Hotels Found — {hotels.length} results</span>
      </div>
      {hotels.length === 0 ? (
        <p className="no-data">No hotels found matching your criteria.</p>
      ) : (
        <>
          <div className="hotel-list">
            {hotels.slice(0, show).map((h, i) => (
              <div className="hotel-item" key={h.id || i}>
                <div className="hotel-info">
                  <p className="hotel-name">{h.name}</p>
                  <p className="hotel-meta">
                    <MapPin size={11} />
                    {h.city}, {h.country}
                  </p>
                  <div className="hotel-amenities">
                    {(h.amenities || []).slice(0, 4).map((a) => (
                      <span className="amenity-tag" key={a}>
                        {a.replace(/_/g, ' ')}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="hotel-right">
                  <div className="hotel-rating">
                    {Array.from({ length: 5 }).map((_, idx) => (
                      <span
                        key={idx}
                        style={{ color: idx < h.rating ? 'var(--accent)' : 'var(--text-muted)' }}
                      >
                        ★
                      </span>
                    ))}
                  </div>
                  <p className="hotel-stars-label">{h.rating}-star</p>
                </div>
              </div>
            ))}
          </div>
          {hotels.length > show && (
            <button className="show-more-btn" onClick={() => setShow((s) => s + 5)}>
              Show {Math.min(5, hotels.length - show)} more hotels
            </button>
          )}
        </>
      )}
    </div>
  );
}