import React from 'react';
import { TrendingDown } from 'lucide-react';
import './Block.css';

export default function PriceAnalysisBlock({ content }) {
  return (
    <div className="block-card block-card--gold">
      <div className="block-header">
        <TrendingDown size={15} strokeWidth={1.5} />
        <span>Price Analysis</span>
      </div>
      <p className="analysis-text">{content}</p>
    </div>
  );
}