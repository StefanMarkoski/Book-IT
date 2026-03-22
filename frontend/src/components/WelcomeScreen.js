import React, { useState } from 'react';
import { Send, MapPin, CloudSun, Hotel, Sparkles } from 'lucide-react';
import './WelcomeScreen.css';

const SUGGESTIONS = [
  { icon: <MapPin size={16} strokeWidth={1.5} />, label: 'Suggest me destinations', prompt: 'Suggest me some beautiful travel destinations in Europe for a summer trip' },
  { icon: <Hotel size={16} strokeWidth={1.5} />, label: 'Find hotels in Paris', prompt: 'Find me top-rated hotels in Paris with a spa and good reviews' },
  { icon: <CloudSun size={16} strokeWidth={1.5} />, label: 'Weather + hotels combo', prompt: "What's the weather like in Barcelona this week? Also show me some good hotels there" },
  { icon: <Sparkles size={16} strokeWidth={1.5} />, label: 'Plan a weekend trip', prompt: 'Help me plan a weekend trip to Rome — weather, hotels, and highlights' },
];

export default function WelcomeScreen({ onSend }) {
  const [input, setInput] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    onSend(input);
    setInput('');
  };

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSend(input);
      setInput('');
    }
  };

  return (
    <div className="welcome">
      <div className="welcome-content">
        <div className="welcome-hero">
          <div className="hero-glyph">✦</div>
          <h1 className="hero-title">Where do you want<br />to go?</h1>
          <p className="hero-sub">
            Ask me about destinations, hotels, weather forecasts,<br />
            or let me plan your next adventure.
          </p>
        </div>
        <div className="welcome-input-wrap">
          <form className="input-form" onSubmit={handleSubmit}>
            <textarea
              className="main-input"
              placeholder="Paris for a week, budget hotels, good weather..."
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKey}
              rows={1}
              autoFocus
            />
            <button className="send-btn" type="submit" disabled={!input.trim()} aria-label="Send">
              <Send size={16} strokeWidth={2} />
            </button>
          </form>
        </div>
        <div className="suggestions-grid">
          {SUGGESTIONS.map((s) => (
            <button key={s.label} className="suggestion-pill" onClick={() => onSend(s.prompt)}>
              <span className="pill-icon">{s.icon}</span>
              <span>{s.label}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}