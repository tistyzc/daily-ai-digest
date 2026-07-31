"""RSS feed fetcher with retry, timeout, and graceful degradation."""

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

import feedparser
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class Article:
    """Normalized article representation across all sources."""

    title: str
    url: str
    source_name: str
    source_category: str
    authority_score: int
    language: str  # "en" or "zh"
    published_at: datetime | None = None
    summary: str = ""
    content: str = ""
    content_length: int = 0
    tags: list[str] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.url)


class RSSFetcher:
    """Fetches and parses RSS/Atom feeds with resilience."""

    TIMEOUT = 30  # seconds
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    USER_AGENT = "DailyAIDigest/1.0 (RSS Reader; +https://github.com/daily-ai-digest)"

    def __init__(self, sources: list[dict]) -> None:
        """
        Args:
            sources: List of source configs with name, url, category, authority_score, language.
        """
        self.sources = sources

    def fetch_all(self) -> list[Article]:
        """Fetch articles from all configured RSS sources."""
        all_articles: list[Article] = []
        for source in self.sources:
            try:
                articles = self._fetch_one(source)
                all_articles.extend(articles)
                logger.info(f"[{source['name']}] Fetched {len(articles)} articles")
            except Exception as e:
                logger.warning(f"[{source['name']}] Failed: {e}")
        return all_articles

    def _fetch_one(self, source: dict) -> list[Article]:
        """Fetch articles from a single RSS feed with retries."""
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                resp = requests.get(
                    source["url"],
                    timeout=self.TIMEOUT,
                    headers={"User-Agent": self.USER_AGENT},
                )
                resp.raise_for_status()
                feed = feedparser.parse(resp.content)

                if feed.bozo and not feed.entries:
                    raise ValueError(f"Malformed feed: {feed.bozo_exception}")

                return [
                    self._parse_entry(entry, source)
                    for entry in feed.entries
                    if self._is_valid_entry(entry)
                ]

            except (requests.RequestException, ValueError) as e:
                if attempt < self.MAX_RETRIES:
                    logger.debug(f"[{source['name']}] Retry {attempt}/{self.MAX_RETRIES}: {e}")
                    time.sleep(self.RETRY_DELAY)
                else:
                    raise

        return []

    def _is_valid_entry(self, entry: dict) -> bool:
        """Check if feed entry has minimum required fields."""
        return bool(entry.get("title") and (entry.get("link") or entry.get("id")))

    def _parse_entry(self, entry: dict, source: dict) -> Article:
        """Parse a feed entry into a normalized Article."""
        url = entry.get("link") or entry.get("id", "")
        title = entry.get("title", "").strip()
        summary = self._clean_html(entry.get("summary", "") or entry.get("description", ""))
        content = self._extract_content(entry)
        published = self._parse_date(entry)

        return Article(
            title=title,
            url=url,
            source_name=source["name"],
            source_category=source.get("category", "general"),
            authority_score=source.get("authority_score", 5),
            language=source.get("language", "en"),
            published_at=published,
            summary=summary,
            content=content,
            content_length=len(content),
            tags=entry.get("tags", []),
        )

    @staticmethod
    def _clean_html(html: str) -> str:
        """Strip HTML tags, return plain text."""
        try:
            return BeautifulSoup(html, "lxml").get_text(separator=" ", strip=True)
        except Exception:
            return html

    @staticmethod
    def _extract_content(entry: dict) -> str:
        """Extract the richest available text content from an entry."""
        candidates = []
        for key in ("content", "summary", "summary_detail", "description"):
            val = entry.get(key)
            if isinstance(val, str):
                candidates.append(val)
            elif isinstance(val, list) and val:
                candidates.append(val[0].get("value", ""))
            elif isinstance(val, dict):
                candidates.append(val.get("value", ""))
        combined = " ".join(candidates)
        return RSSFetcher._clean_html(combined)

    @staticmethod
    def _parse_date(entry: dict) -> datetime | None:
        """Parse published date from feed entry."""
        for date_field in ("published_parsed", "updated_parsed"):
            tp = entry.get(date_field)
            if tp and len(tp) >= 6:
                try:
                    return datetime(*tp[:6], tzinfo=UTC)
                except (TypeError, ValueError):
                    pass
        return None
