from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

@dataclass(frozen=True)
class WeatherPoint:
    date_time_iso: str
    temp_c: float
    condition: str


class OpenWeatherTool:
    """
    A tool for fetching weather data from the OpenWeather API.
    Attributes:
        api_key (str): The API key for accessing the OpenWeather API.
        timeout_s (int): The timeout in seconds for API requests.
    """
    def __init__(self, api_key: Optional[str] = None, timeout_s: int = 15):
        """
        Initializes the OpenWeatherTool with the provided API key.
        If no API key is provided, it attempts to read it from the environment variable 'OPENWEATHER_API_KEY'.
        Args:
            api_key (Optional[str]): The API key for accessing the OpenWeather API.
        """
        self._api_key = api_key or os.getenv("WEATHER_API_KEY")
        if not self._api_key:
            raise RuntimeError("WEATHER_API_KEY is missing. Add it to your .env.")
        self._timeout_s = timeout_s


    def _geo_city_to_lat_lon(self, city: str) -> Dict[str, float]:
        """
        Converts a city name to its corresponding latitude and longitude using the OpenWeather Geocoding API.
        Args:
            city (str): The name of the city to geocode.
        """
        url="https://api.openweathermap.org/geo/1.0/direct"
        params = {
            "q": city,
            "limit": 1,
            "appid": self._api_key
        }
        response = requests.get(url, params=params, timeout=self._timeout_s)
        response.raise_for_status()
        data = response.json()

        if not data:
            raise ValueError(f"City '{city}' not found.")
        return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"])}
    
    def get_forecast_5day(self, city: str) -> Dict[str, Any]:
        """
        Fetches the 5-day weather forecast for a given city using the OpenWeather API.
        Args:
            city (str): The name of the city for which to fetch the weather forecast.
        """    
        coords = self._geo_city_to_lat_lon(city)
        lat,lon = coords["lat"], coords["lon"]
        url = "https://api.openweathermap.org/data/2.5/forecast"

        params = {
            "lat": lat,
            "lon": lon,
            "appid": self._api_key,
            "units": "metric"
        }

        response = requests.get(url, params=params, timeout=self._timeout_s)
        response.raise_for_status()
        data = response.json()

        points : List[WeatherPoint] = []
        for item in data.get("list", []):
            dt_iso = item.get("dt_txt")
            main = item.get("main", {})
            weather_arr = item.get("weather", [])
            temp = float(main.get("temp")) if main.get("temp") is not None else None
            condition = weather_arr[0].get("main") if weather_arr else "Unknown"

            if dt_iso is None or temp is None:
                continue
            points.append(WeatherPoint(date_time_iso=dt_iso, temp_c=temp, condition=condition))

        return {
            "city":city,
            "lat":lat,
            "lon":lon,
            "units":"metric",
            "forecast": [
                {"date_time":p.date_time_iso,"temp_c":p.temp_c,"condition":p.condition} 
                for p in points
            ]
        }