from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from app.tools.hotels_tool import HotelsApiTool
from app.tools.weather_tool import OpenWeatherTool
from app.tools.web_search_tool import WebSearchTool

# Singletons (pure tools; no memory/state stored per request)
_web = WebSearchTool()
_weather = OpenWeatherTool()
_hotels = HotelsApiTool()


@tool
def web_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search the web and return a compact list of results (title/url/snippet)."""
    return _web.search(query, max_results=max_results, max_sentences=2, include_raw=False)


@tool
def get_weather_forecast(city: str) -> Dict[str, Any]:
    """Get ~5 day weather forecast for a city (OpenWeather)."""
    return _weather.get_forecast_5day(city)


@tool
def search_hotels(
    city: str,
    min_rating: int = 4,
    limit: int = 10,
    amenities: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Search hotels for a city with optional amenities filter."""
    return _hotels.search_hotels(city=city, min_rating=min_rating, limit=limit, amenities=amenities)