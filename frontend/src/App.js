import React, { useState, useRef, useEffect } from 'react';
import Header from './components/Header';
import ChatWindow from './components/ChatWindow';
import WelcomeScreen from './components/WelcomeScreen';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);
  const abortRef = useRef(null);

  const sendMessage = async (text) => {
    if (!text.trim() || isLoading) return;

    const userMsg = { role: 'user', content: text, id: Date.now() };
    setMessages(prev => [...prev, userMsg]);
    setHasStarted(true);
    setIsLoading(true);

    try {
      const res = await fetch('/chat_agentic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Server error' }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      const assistantMsg = {
        role: 'assistant',
        id: Date.now() + 1,
        content: data.message,
        blocks: data.blocks || [],
        priceAnalysis: data.price_analysis || null,
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (err) {
      const errMsg = {
        role: 'assistant',
        id: Date.now() + 1,
        content: null,
        error: err.message || 'Something went wrong. Please try again.',
        blocks: [],
      };
      setMessages(prev => [...prev, errMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setHasStarted(false);
  };

  return (
    <div className="app-shell">
      <Header onClear={clearChat} hasMessages={messages.length > 0} />
      <main className="app-main">
        {!hasStarted ? (
          <WelcomeScreen onSend={sendMessage} />
        ) : (
          <ChatWindow
            messages={messages}
            isLoading={isLoading}
            onSend={sendMessage}
          />
        )}
      </main>
    </div>
  );
}

export default App;