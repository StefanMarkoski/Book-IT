import React from 'react';
import { CloudSun } from 'lucide-react';
import './Block.css';

const CONDITION_ICON = {
  Clear: '☀️', Clouds: '☁️', Rain: '🌧️', Drizzle: '🌦️',
  Snow: '❄️', Thunderstorm: '⛈️', Mist: '🌫️', Fog: '🌫️',
};

export default function WeatherBlock({ data }) {
  const forecast = data?.forecast || [];
  const daily = {};
  forecast.forEach(p => {
    const day = p.date_time?.split(' ')[0];
    if (day && !daily[day]) daily[day] = p;
  });
  const days = Object.values(daily).slice(0, 5);

  return (
    <div className="block-card">
      <div className="block-header">
        <CloudSun size={15} strokeWidth={1.5} />
        <span>5-Day Forecast — {data?.city}</span>
      </div>
      <div className="weather-grid">
        {days.map((d, i) => (
          <div className="weather-day" key={i}>
            <p className="weather-date">{new Date(d.date_time).toLocaleDateString('en', { weekday: 'short', month: 'short', day: 'numeric' })}</p>
            <div className="weather-icon">{CONDITION_ICON[d.condition] || '🌡️'}</div>
            <p className="weather-temp">{Math.round(d.temp_c)}°C</p>
            <p className="weather-cond">{d.condition}</p>
          </div>
        ))}
      </div>
    </div>
  );
}