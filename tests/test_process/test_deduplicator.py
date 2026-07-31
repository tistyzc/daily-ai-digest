"""Tests for deduplicator."""

from src.collect.rss_fetcher import Article
from src.process.deduplicator import Deduplicator


class TestDeduplicator:
    def test_removes_near_duplicate_titles(self):
        """Similar titles should be deduplicated."""
        from src.collect.rss_fetcher import Article

        # Create two articles with very similar titles for dedup testing
        articles = [
            Article(title="GPT-5 Released with Major Improvements", url="https://a.com", source_name="OpenAI Blog",
                    source_category="ai", authority_score=9, language="en", content="x" * 700, content_length=700),
            Article(title="GPT-5 Released with Major Improvements", url="https://b.com", source_name="TechCrunch",
                    source_category="business", authority_score=7, language="en", content="x" * 700, content_length=700),
            Article(title="Rust 2026 Edition", url="https://c.com", source_name="Ars Technica",
                    source_category="tech", authority_score=8, language="en", content="x" * 700, content_length=700),
        ]

        dedup = Deduplicator(threshold=0.85)
        unique = dedup.deduplicate(articles)

        # Same title should be deduplicated, keeping the higher-authority one
        titles = [a.title for a in unique]
        gpt5_titles = [t for t in titles if "GPT-5" in t]
        assert len(gpt5_titles) == 1  # Only one GPT-5 article should remain (higher authority_score)

    def test_different_titles_kept(self):
        """Dissimilar titles should both be kept."""
        articles = [
            Article(title="Python 4.0 Released", url="https://a.com", source_name="A", source_category="x", authority_score=7, language="en", content="x" * 100, content_length=100),
            Article(title="Rust 2026 Edition Announced", url="https://b.com", source_name="B", source_category="x", authority_score=7, language="en", content="x" * 100, content_length=100),
        ]
        dedup = Deduplicator(threshold=0.85)
        unique = dedup.deduplicate(articles)
        assert len(unique) == 2

    def test_exact_same_title(self):
        """Exact same titles should deduplicate, keeping higher authority."""
        articles = [
            Article(title="Same Title", url="https://a.com", source_name="TechCrunch", source_category="x", authority_score=7, language="en", content="x" * 100, content_length=100),
            Article(title="Same Title", url="https://b.com", source_name="MIT Tech Review", source_category="x", authority_score=9, language="en", content="x" * 100, content_length=100),
        ]
        dedup = Deduplicator(threshold=0.85)
        unique = dedup.deduplicate(articles)
        assert len(unique) == 1
        assert unique[0].authority_score == 9

    def test_empty_input(self):
        dedup = Deduplicator()
        assert dedup.deduplicate([]) == []

    def test_single_article(self):
        article = Article(title="Only One", url="https://x.com", source_name="X", source_category="x", authority_score=5, language="en", content="x" * 100, content_length=100)
        dedup = Deduplicator(threshold=0.85)
        result = dedup.deduplicate([article])
        assert len(result) == 1
