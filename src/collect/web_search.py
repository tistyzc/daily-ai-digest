"""Web search fetcher using Tavily API for supplementary news discovery."""

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime

import requests

from src.collect.rss_fetcher import Article

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result from Tavily or Brave Search."""

    title: str
    url: str
    snippet: str
    source: str = "web-search"

    def to_article(self) -> Article:
        return Article(
            title=self.title,
            url=self.url,
            source_name=self.source,
            source_category="web-search",
            authority_score=4,
            language="en",
            published_at=datetime.now(UTC),
            summary=self.snippet,
            content=self.snippet,
            content_length=len(self.snippet),
        )


class WebSearchFetcher:
    """Fetches news via Tavily Search API (primary) with Brave Search fallback."""

    TAVILY_API = "https://api.tavily.com/search"
    TIMEOUT = 20

    def __init__(self, queries_en: list[str], queries_zh: list[str], max_results: int = 5) -> None:
        self.queries_en = queries_en
        self.queries_zh = queries_zh
        self.max_results = max_results
        self.api_key = os.environ.get("TAVILY_API_KEY", "")

    def fetch_all(self) -> list[SearchResult]:
        """Run all search queries and aggregate results."""
        all_results: list[SearchResult] = []

        for query in self.queries_en + self.queries_zh:
            try:
                results = self._search(query)
                all_results.extend(results)
                logger.info(f"[Search] '{query}' -> {len(results)} results")
            except Exception as e:
                logger.warning(f"[Search] '{query}' failed: {e}")

        # Deduplicate by URL
        seen: set[str] = set()
        unique = []
        for r in all_results:
            if r.url not in seen:
                seen.add(r.url)
                unique.append(r)

        return unique

    def _search(self, query: str) -> list[SearchResult]:
        """Execute a single search query."""
        if self.api_key:
            return self._search_tavily(query)
        return self._search_public(query)

    def _search_tavily(self, query: str) -> list[SearchResult]:
        """Search via Tavily API (requires API key)."""
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": self.max_results,
            "search_depth": "basic",
            "include_domains": [],
            "exclude_domains": [],
        }

        resp = requests.post(self.TAVILY_API, json=payload, timeout=self.TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
                source="tavily",
            )
            for r in data.get("results", [])
        ]

    def _search_public(self, query: str) -> list[SearchResult]:
        """Fallback: scrape DuckDuckGo lite (no API key needed)."""
        url = "https://lite.duckduckgo.com/lite/"
        try:
            resp = requests.post(
                url,
                data={"q": query},
                headers={"User-Agent": "DailyAIDigest/1.0", "Content-Type": "application/x-www-form-urlencoded"},
                timeout=self.TIMEOUT,
            )
            resp.raise_for_status()

            from bs4 import BeautifulSoup

            soup = BeautifulSoup(resp.text, "lxml")
            results = []
            for row in soup.select("table tr")[1:]:  # skip header
                cols = row.find_all("td")
                if len(cols) >= 2:
                    link = cols[1].find("a")
                    snippet = cols[1].get_text(strip=True)
                    if link:
                        results.append(
                            SearchResult(
                                title=link.get_text(strip=True),
                                url=link.get("href", ""),
                                snippet=snippet,
                                source="duckduckgo",
                            )
                        )
            return results[: self.max_results]
        except Exception as e:
            logger.warning(f"DuckDuckGo fallback failed: {e}")
            return []
