import React from 'react';
import { Compass } from 'lucide-react';
import './TypingIndicator.css';

export default function TypingIndicator() {
  return (
    <div className="typing-row">
      <div className="avatar avatar--assistant">
        <Compass size={14} strokeWidth={1.5} />
      </div>
      <div className="typing-bubble">
        <div className="typing-loader">
          <span /><span /><span />
        </div>
        <p className="typing-label">Researching your trip...</p>
      </div>
    </div>
  );
}