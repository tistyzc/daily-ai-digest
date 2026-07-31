"""Cross-source deduplication based on title similarity."""

import logging
from difflib import SequenceMatcher

from src.collect.rss_fetcher import Article

logger = logging.getLogger(__name__)


class Deduplicator:
    """Removes duplicate articles based on title similarity."""

    def __init__(self, threshold: float = 0.85) -> None:
        """
        Args:
            threshold: Similarity ratio (0.0-1.0) above which articles are considered duplicates.
        """
        self.threshold = threshold

    def deduplicate(self, articles: list[Article]) -> list[Article]:
        """Return deduplicated articles, keeping the highest authority_score version."""
        if not articles:
            return []

        # Sort by authority_score descending — keep the better source
        sorted_articles = sorted(articles, key=lambda a: (a.title, a.authority_score), reverse=True)
        unique: list[Article] = []

        for article in sorted_articles:
            is_dup = False
            for existing in unique:
                sim = self._similarity(article.title, existing.title)
                if sim >= self.threshold:
                    is_dup = True
                    logger.debug(f"DEDUP: '{article.title[:60]}' ~ '{existing.title[:60]}' ({sim:.2f})")
                    break

            if not is_dup:
                unique.append(article)

        removed = len(articles) - len(unique)
        logger.info(f"Deduplication: removed {removed} duplicates, {len(unique)} remaining")
        return unique

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Calculate title similarity ratio."""
        # Normalize: lowercase, strip common prefixes
        a = a.lower().strip()
        b = b.lower().strip()

        # Remove common noise prefixes
        for prefix in ["[github trending] ", "show hn: ", "ask hn: "]:
            a = a.removeprefix(prefix)
            b = b.removeprefix(prefix)

        return SequenceMatcher(None, a, b).ratio()
