from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass(frozen=True)
class Hotel:
    id: int
    name: str
    city: str
    country: str
    country_code: str
    rating: int
    lat: float
    lng: float
    amenities: List[str]


class HotelsApiTool:
    _AMENITY_ALIASES: Dict[str, List[str]] = {
        "wifi": ["free_wifi", "wifi", "internet"],
        "pool": ["pool"],
        "parking": ["parking"],
        "spa": ["spa"],
        "gym": ["gym", "fitness"],
        "restaurant": ["restaurant"],
        "bar": ["bar"],  
    }

    def __init__(self, api_key: Optional[str] = None, timeout_s: int = 15):
        self._api_key = api_key or os.getenv("HOTELS_API_KEY")
        if not self._api_key:
            raise RuntimeError("HOTELS_API_KEY is missing. Add it to your .env.")
        self._timeout_s = timeout_s
        self._base_url = "https://api.hotels-api.com/v1"

    def _normalize_requested_amenities(self, amenities: List[str]) -> List[str]:
        return [a.strip().lower().replace(" ", "_") for a in amenities if a and a.strip()]

    def _hotel_has_amenities(self, hotel_amenities: List[str], requested: List[str]) -> bool:
        hotel_set = {a.strip().lower() for a in hotel_amenities if a}

        for req in requested:
            aliases = self._AMENITY_ALIASES.get(req, [req])

            if any(alias in hotel_set for alias in aliases):
                continue

            # allow substring match (e.g. "fitness_center" contains "fitness")
            if any(req in ha for ha in hotel_set):
                continue

            return False

        return True

    def _error_payload(
        self,
        *,
        params: Dict[str, Any],
        err_type: str,
        message: str,
        status_code: Optional[int] = None,
        body_preview: Optional[str] = None,
    ) -> Dict[str, Any]:
        err: Dict[str, Any] = {"type": err_type, "message": message}
        if status_code is not None:
            err["status_code"] = status_code
        if body_preview:
            err["body_preview"] = body_preview
        return {"query": params, "hotels": [], "error": err}

    def search_hotels(
        self,
        *,
        city: Optional[str] = None,
        country: Optional[str] = None,
        country_code: Optional[str] = None,
        name: Optional[str] = None,
        rating: Optional[int] = None,
        min_rating: Optional[int] = None,
        amenities: Optional[List[str]] = None,
        limit: int = 20,
        page: int = 1,
    ) -> Dict[str, Any]:
        """
        NOTE:
        We intentionally DO NOT send rating/min_rating to the upstream API because it is unstable
        (can return 500 for valid inputs). We filter rating locally instead.
        """

        headers = {"X-API-KEY": self._api_key}

        # Params we send to upstream API (NO rating filters)
        params: Dict[str, Any] = {"limit": limit, "page": page}
        if city:
            params["city"] = city
        if country:
            params["country"] = country
        if country_code:
            params["country_code"] = country_code
        if name:
            params["name"] = name

        url = f"{self._base_url}/hotels/search"

        try:
            r = requests.get(url, headers=headers, params=params, timeout=self._timeout_s)
        except requests.RequestException as e:
            return self._error_payload(
                params=params,
                err_type="hotels_api_request_error",
                message=str(e),
            )

        if not r.ok:
            body = (r.text or "")[:500]
            return self._error_payload(
                params=params,
                err_type="hotels_api_http_error",
                message="Hotels API request failed",
                status_code=r.status_code,
                body_preview=body if body else None,
            )

        try:
            payload = r.json()
        except Exception:
            body = (r.text or "")[:500]
            return self._error_payload(
                params=params,
                err_type="hotels_api_bad_json",
                message="Hotels API returned non-JSON response",
                status_code=r.status_code,
                body_preview=body if body else None,
            )

        if not payload.get("success", False):
            return self._error_payload(
                params=params,
                err_type="hotels_api_error",
                message=payload.get("error", "Hotels API error"),
                status_code=r.status_code,
            )

        raw_hotels = payload.get("data", []) or []

        # Local rating filter (exact or minimum)
        if rating is not None:
            raw_hotels = [h for h in raw_hotels if int(h.get("rating", 0) or 0) == int(rating)]
        if min_rating is not None:
            raw_hotels = [h for h in raw_hotels if int(h.get("rating", 0) or 0) >= int(min_rating)]

        # Local amenities filter
        requested_amenities: List[str] = []
        if amenities:
            requested_amenities = self._normalize_requested_amenities(amenities)
            raw_hotels = [
                h
                for h in raw_hotels
                if self._hotel_has_amenities(h.get("amenities", []) or [], requested_amenities)
            ]

        hotels: List[Hotel] = []
        for h in raw_hotels:
            hotels.append(
                Hotel(
                    id=int(h.get("id", 0) or 0),
                    name=str(h.get("name", "") or ""),
                    city=str(h.get("city", "") or ""),
                    country=str(h.get("country", "") or ""),
                    country_code=str(h.get("country_code", "") or ""),
                    rating=int(h.get("rating", 0) or 0),
                    lat=float(h.get("lat", 0.0) or 0.0),
                    lng=float(h.get("lng", 0.0) or 0.0),
                    amenities=list(h.get("amenities", []) or []),
                )
            )

        # Echo what the user asked for (including filters we applied locally)
        query_echo = dict(params)
        if rating is not None:
            query_echo["rating"] = int(rating)
        if min_rating is not None:
            query_echo["min_rating"] = int(min_rating)
        if requested_amenities:
            query_echo["amenities"] = requested_amenities

        return {
            "query": query_echo,
            "hotels": [
                {
                    "id": ht.id,
                    "name": ht.name,
                    "city": ht.city,
                    "country": ht.country,
                    "country_code": ht.country_code,
                    "rating": ht.rating,
                    "lat": ht.lat,
                    "lng": ht.lng,
                    "amenities": ht.amenities,
                }
                for ht in hotels
            ],
        }