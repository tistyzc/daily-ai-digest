"""Tests for content gate."""

from src.collect.rss_fetcher import Article
from src.process.content_gate import ContentGate


class TestContentGate:
    def test_drops_short_content(self, sample_articles):
        """Articles below min_content_length should be dropped."""
        gate = ContentGate(min_content_length=600)
        passed = gate.filter(sample_articles)

        # The "Short post" article has content_length=16, should be dropped
        titles = [a.title for a in passed]
        assert "Short post" not in titles
        assert len(passed) < len(sample_articles)

    def test_passes_long_content(self, sample_articles):
        """Articles at or above min_content_length should pass."""
        gate = ContentGate(min_content_length=600)
        passed = gate.filter(sample_articles)

        assert all(a.content_length >= 600 for a in passed)

    def test_empty_input(self):
        """Should handle empty list gracefully."""
        gate = ContentGate(min_content_length=600)
        result = gate.filter([])
        assert result == []

    def test_zero_threshold_passes_all(self, sample_articles):
        """Setting min_content_length=0 should pass all articles."""
        gate = ContentGate(min_content_length=0)
        passed = gate.filter(sample_articles)
        assert len(passed) == len(sample_articles)

    def test_very_high_threshold_drops_all(self):
        """Impossibly high threshold should drop everything."""
        article = Article(
            title="Test", url="https://x.com", source_name="X", source_category="x",
            authority_score=5, language="en", content="Hello world", content_length=11,
        )
        gate = ContentGate(min_content_length=10000)
        passed = gate.filter([article])
        assert len(passed) == 0
