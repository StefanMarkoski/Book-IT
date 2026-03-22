import React from 'react';
import { Sparkles, Globe } from 'lucide-react';
import './Block.css';

export default function DestinationBlock({ data }) {
  const suggestions = data?.suggestions || [];
  return (
    <div className="block-card">
      <div className="block-header">
        <Sparkles size={15} strokeWidth={1.5} />
        <span>Destination Suggestions</span>
      </div>
      <div className="destination-grid">
        {suggestions.map((s, i) => (
          <div className="destination-card" key={i}>
            <div className="dest-flag"><Globe size={18} strokeWidth={1.5} /></div>
            <div>
              <p className="dest-name">{s.name || s.city || s.destination || s}</p>
              {s.reason && <p className="dest-reason">{s.reason}</p>}
              {s.country && <p className="dest-country">{s.country}</p>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}