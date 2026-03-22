import React from 'react';
import { Compass, User, AlertTriangle } from 'lucide-react';
import WeatherBlock from './blocks/WeatherBlock';
import HotelListBlock from './blocks/HotelListBlock';
import HotelPriceBlock from './blocks/HotelPriceBlock';
import DestinationBlock from './blocks/DestinationBlock';
import PriceAnalysisBlock from './blocks/PriceAnalysisBlock';
import './MessageBubble.css';

export default function MessageBubble({ message }) {
  const isUser = message.role === 'user';

  if (isUser) {
    return (
      <div className="bubble-row bubble-row--user">
        <div className="bubble bubble--user">
          <p>{message.content}</p>
        </div>
        <div className="avatar avatar--user">
          <User size={14} strokeWidth={1.5} />
        </div>
      </div>
    );
  }

  return (
    <div className="bubble-row bubble-row--assistant">
      <div className="avatar avatar--assistant">
        <Compass size={14} strokeWidth={1.5} />
      </div>
      <div className="assistant-content">
        {message.error && (
          <div className="error-card">
            <AlertTriangle size={15} strokeWidth={1.5} />
            <span>{message.error}</span>
          </div>
        )}
        {message.content && (
          <div className="bubble bubble--assistant">
            <p>{message.content}</p>
          </div>
        )}
        {message.blocks && message.blocks.length > 0 && (
          <div className="blocks-container">
            {message.blocks.map((block, i) => renderBlock(block, i))}
          </div>
        )}
        {message.priceAnalysis && (
          <PriceAnalysisBlock content={message.priceAnalysis} />
        )}
      </div>
    </div>
  );
}

function renderBlock(block, index) {
  const key = `block-${index}`;
  switch (block.type) {
    case 'weather_forecast':    return <WeatherBlock key={key} data={block.data} />;
    case 'hotel_list':          return <HotelListBlock key={key} data={block.data} />;
    case 'hotel_price':         return <HotelPriceBlock key={key} data={block.data} />;
    case 'destination_suggestions': return <DestinationBlock key={key} data={block.data} />;
    default: return null;
  }
}