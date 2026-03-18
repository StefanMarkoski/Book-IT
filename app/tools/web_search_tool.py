from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from tavily import TavilyClient


class WebSearchTool:
    _BLOCKED_DOMAINS = (
        "facebook.com",
        "pinterest.",
        "tiktok.",
        "instagram.com",
    )

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("TAVILY_API_KEY")
        self._client = TavilyClient(api_key=self._api_key) if self._api_key else None

    def _is_blocked(self, url: str) -> bool:
        u = (url or "").lower()
        return any(d in u for d in self._BLOCKED_DOMAINS)

    def _compress_snippet(self, text: str, max_sentences: int = 2) -> str:
        if not text:
            return ""
        text = re.sub(r"\s*#+\s*", " ", text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = text.replace("max_bytes", " ").replace("strip_icc", " ")
        text = " ".join(text.replace("\n", " ").split()).strip()
        parts = re.split(r"(?<=[.!?])\s+", text)
        parts = [p for p in parts if p]
        compressed = " ".join(parts[:max_sentences]).strip()
        if compressed and not compressed.endswith((".", "!", "?", "…")):
            compressed += "…"
        return compressed

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        max_sentences: int = 2,
        include_raw: bool = False,
    ) -> Dict[str, Any]:
        if not self._client:
            return {
                "query": query,
                "results": [],
                "error": {"type": "missing_api_key", "message": "TAVILY_API_KEY missing in .env"},
            }

        resp = self._client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_answer=False,
            include_images=False,
            include_raw_content=False,
        )

        results: List[Dict[str, Any]] = []
        raw_results: List[Dict[str, Any]] = []

        for r in resp.get("results", []):
            title = r.get("title") or ""
            url = r.get("url") or ""
            if self._is_blocked(url):
                continue

            snippet_raw = r.get("content") or ""
            snippet = self._compress_snippet(snippet_raw, max_sentences=max_sentences)

            results.append({"title": title, "url": url, "snippet": snippet})

            if include_raw:
                raw_results.append({"title": title, "url": url, "snippet_raw": snippet_raw})

        out: Dict[str, Any] = {"query": query, "results": results}
        if include_raw:
            out["raw_results"] = raw_results
        return out