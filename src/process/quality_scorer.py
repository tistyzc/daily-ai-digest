"""Multi-dimensional quality scoring for articles."""

import logging
from datetime import UTC, datetime, timedelta

from src.collect.rss_fetcher import Article

logger = logging.getLogger(__name__)


class QualityScorer:
    """Scores articles on authority, recall, recency, and cross-source validation."""

    def __init__(
        self,
        authority_boost: int = 3,
        cross_source_boost: int = 5,
        recency_boost: int = 2,
        already_reported_penalty: int = -5,
        lookback_days: int = 2,
    ) -> None:
        self.authority_boost = authority_boost
        self.cross_source_boost = cross_source_boost
        self.recency_boost = recency_boost
        self.already_reported_penalty = already_reported_penalty
        self.lookback_days = lookback_days

    def score(self, articles: list[Article], previous_titles: set[str] | None = None) -> list[tuple[Article, int]]:
        """Score all articles, return (article, score) pairs sorted descending."""
        # Detect cross-source coverage (same story from multiple sources)
        title_groups = self._group_by_similarity(articles)

        scored: list[tuple[Article, int]] = []
        now = datetime.now(UTC)
        cutoff = now - timedelta(days=self.lookback_days)

        for article in articles:
            score = 0

            # Authority: high-authority sources get a boost
            if article.authority_score >= 8:
                score += self.authority_boost

            # Cross-source: if 2+ sources cover the same story
            group = title_groups.get(article.title, 1)
            if group >= 2:
                score += self.cross_source_boost

            # Recency: bonus for last 24 hours
            if article.published_at and article.published_at > cutoff:
                score += self.recency_boost

            # Already reported: penalty for stories that appeared in previous digest
            if previous_titles and self._is_already_reported(article.title, previous_titles):
                score += self.already_reported_penalty

            scored.append((article, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    @staticmethod
    def _group_by_similarity(articles: list[Article]) -> dict[str, int]:
        """Count how many sources cover each unique story (by title similarity)."""
        from difflib import SequenceMatcher

        groups: dict[int, list[int]] = {}  # index -> [indices in same group]
        assigned: set[int] = set()

        for i, a in enumerate(articles):
            if i in assigned:
                continue
            groups[i] = [i]
            assigned.add(i)
            for j, b in enumerate(articles):
                if j in assigned:
                    continue
                sim = SequenceMatcher(None, a.title.lower(), b.title.lower()).ratio()
                if sim >= 0.80:
                    groups[i].append(j)
                    assigned.add(j)

        result: dict[str, int] = {}
        for indices in groups.values():
            count = len(indices)
            for idx in indices:
                result[articles[idx].title] = count

        return result

    @staticmethod
    def _is_already_reported(title: str, previous_titles: set[str]) -> bool:
        """Check if a very similar title appeared previously."""
        from difflib import SequenceMatcher

        return any(
            SequenceMatcher(None, title.lower(), prev.lower()).ratio() >= 0.85
            for prev in previous_titles
        )
