"""Content quality gate — drops articles that are too short to be useful."""

import logging

from src.collect.rss_fetcher import Article

logger = logging.getLogger(__name__)


class ContentGate:
    """Filters out articles with insufficient content."""

    def __init__(self, min_content_length: int = 600) -> None:
        self.min_length = min_content_length

    def filter(self, articles: list[Article]) -> list[Article]:
        """Return only articles meeting the minimum content length."""
        passed = []
        for a in articles:
            if a.content_length >= self.min_length:
                passed.append(a)
            else:
                logger.debug(f"GATE DROP [{a.content_length} chars]: {a.title[:80]}")
        logger.info(f"Content gate: {len(passed)}/{len(articles)} passed (min {self.min_length} chars)")
        return passed
