import React from 'react';
import { Compass, RotateCcw } from 'lucide-react';
import './Header.css';

export default function Header({ onClear, hasMessages }) {
  return (
    <header className="header">
      <div className="header-inner">
        <div className="header-brand">
          <div className="brand-icon">
            <Compass size={18} strokeWidth={1.5} />
          </div>
          <span className="brand-name">BookIT</span>
          <span className="brand-tag">AI Travel Advisor</span>
        </div>
        <div className="header-actions">
          <div className="status-dot" title="API connected" />
          {hasMessages && (
            <button className="btn-ghost" onClick={onClear} title="New conversation">
              <RotateCcw size={15} strokeWidth={1.5} />
              <span>New chat</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
}