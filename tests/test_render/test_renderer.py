"""Tests for the Jinja2 renderer."""

import tempfile
from pathlib import Path

import pytest

from src.render.renderer import DigestRenderer


class TestDigestRenderer:
    @pytest.fixture
    def renderer(self):
        """Create a renderer with the real templates dir."""
        template_dir = Path(__file__).resolve().parent.parent.parent / "src" / "render" / "templates"
        return DigestRenderer(template_dir=str(template_dir))

    def test_render_index_produces_html(self, renderer):
        """Index rendering should produce valid HTML with expected content."""
        from unittest.mock import MagicMock

        article = MagicMock()
        article.title = "GPT-5 Released"
        article.url = "https://example.com/gpt5"
        article.source_name = "OpenAI Blog"
        article.language = "en"

        topics = {
            "Large Language Models": [
                {
                    "article": article,
                    "en_summary": "GPT-5 has improved reasoning.",
                    "zh_summary": "GPT-5推理能力提升。",
                    "key_points": ["Better reasoning"],
                    "key_points_zh": ["推理更好"],
                    "impact": "Important for AI developers.",
                    "impact_zh": "对AI开发者很重要。",
                    "critic_flags": [],
                }
            ],
        }

        stats = {"total_articles": 1, "sources_count": 1, "topics_count": 1}

        html = renderer.render_index(
            date_str="2026-07-31",
            topics=topics,
            stats=stats,
            critical_flags=[],
        )

        assert "<!DOCTYPE html>" in html
        assert "GPT-5 Released" in html
        assert "OpenAI Blog" in html
        assert "GPT-5推理能力提升" in html
        assert "2026" in html or "July" in html

    def test_render_index_with_search_and_filter(self, renderer):
        """Index should include search box and topic filter chips."""
        topics = {
            "AI Research": [],
            "Developer Tools": [],
        }

        html = renderer.render_index(
            date_str="2026-07-31",
            topics=topics,
            stats={"total_articles": 0, "sources_count": 0, "topics_count": 2},
            critical_flags=[],
        )

        assert 'id="search"' in html
        assert 'id="filter-bar"' in html

    def test_render_index_dark_mode(self, renderer):
        """Index should support dark/light theme toggle."""
        html = renderer.render_index(
            date_str="2026-07-31",
            topics={},
            stats={"total_articles": 0, "sources_count": 0, "topics_count": 0},
            critical_flags=[],
        )

        assert "data-theme" in html
        assert "toggleTheme" in html
        assert "[data-theme=\"dark\"]" in html

    def test_render_index_critic_flags(self, renderer):
        """Index should display critic flags when present."""
        topics = {}

        flags = [
            {"article_title": "Test Article", "reason": "Bias detected"},
        ]

        html = renderer.render_index(
            date_str="2026-07-31",
            topics=topics,
            stats={"total_articles": 0, "sources_count": 0, "topics_count": 0},
            critical_flags=flags,
        )

        assert "Editorial Notes" in html or "critic" in html.lower()
        assert "Bias detected" in html

    def test_render_archive(self, renderer):
        """Archive page should list past digests."""
        archives = [
            {"date": "2026-07-31", "title": "Daily AI Digest", "article_count": 25, "url": "archive/2026-07-31.html"},
            {"date": "2026-07-30", "title": "Daily AI Digest", "article_count": 22, "url": "archive/2026-07-30.html"},
        ]

        html = renderer.render_archive(archives)

        assert "2026-07-31" in html
        assert "2026-07-30" in html
        assert "25" in html
        assert "22" in html

    def test_render_archive_empty(self, renderer):
        """Empty archive should show a message, not crash."""
        html = renderer.render_archive([])
        assert "No archives yet" in html or "first" in html.lower()

    def test_write_file_creates_directory(self, renderer):
        """write_file should create parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/subdir/test.html"
            renderer.write_file("<html></html>", path)
            assert Path(path).exists()
            assert Path(path).read_text() == "<html></html>"


def _mock_article(title: str, source_name: str):
    """Helper to create a mock Article for template rendering."""
    from unittest.mock import MagicMock

    article = MagicMock()
    article.title = title
    article.url = "https://example.com"
    article.source_name = source_name
    article.language = "en"
    return article
