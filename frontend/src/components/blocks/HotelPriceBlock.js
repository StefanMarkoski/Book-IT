import React from 'react';
import { DollarSign, ExternalLink } from 'lucide-react';
import './Block.css';

export default function HotelPriceBlock({ data }) {
  const allResults = data?.price_results || [];

  const results = allResults.filter((r) => {
    if (!r.snippet) return false;
    if (r.snippet.length > 300) return false;
    if (r.snippet.includes('%3C')) return false;
    if (r.snippet.includes('%3E')) return false;
    return true;
  });

  return (
    <div className="block-card block-card--accent">
      <div className="block-header">
        <DollarSign size={15} strokeWidth={1.5} />
        <span>Pricing — {data?.hotel}</span>
        <span className="block-header-sub">{data?.city}</span>
      </div>
      <div className="price-list">
        {results.length === 0 ? (
          <p className="no-data">No price data available.</p>
        ) : (
          results.map((r, i) => (
            <div className="price-item" key={i}>
              <div className="price-item-header">
                <p className="price-source">{r.title}</p>
                {r.url && (
                  <a href={r.url} target="_blank" rel="noopener noreferrer" className="price-link">
                    <ExternalLink size={11} strokeWidth={1.5} />
                  </a>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}