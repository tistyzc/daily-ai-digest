"""Tests for RSS fetcher."""

from unittest.mock import MagicMock, patch

from src.collect.rss_fetcher import RSSFetcher


class TestRSSFetcher:
    def test_fetch_all_with_valid_feed(self):
        """Fetch should parse feed entries correctly."""
        sources = [
            {
                "name": "Test Blog",
                "url": "https://example.com/feed.xml",
                "category": "technology",
                "authority_score": 7,
                "language": "en",
            }
        ]

        mock_entry = {
            "title": "Test Article Title",
            "link": "https://example.com/post/1",
            "summary": "<p>This is a test summary with <b>HTML</b> tags.</p>",
            "published_parsed": (2026, 7, 31, 8, 0, 0, 4, 212, 0),
            "tags": [{"term": "AI"}],
        }

        mock_feed = MagicMock()
        mock_feed.bozo = False
        mock_feed.entries = [mock_entry]

        with patch("feedparser.parse", return_value=mock_feed), patch("requests.get") as mock_get:
            mock_get.return_value.content = b"<rss>...</rss>"
            mock_get.return_value.raise_for_status = MagicMock()

            fetcher = RSSFetcher(sources)
            articles = fetcher.fetch_all()

        assert len(articles) == 1
        assert articles[0].title == "Test Article Title"
        assert articles[0].url == "https://example.com/post/1"
        assert articles[0].source_name == "Test Blog"
        assert articles[0].language == "en"
        assert articles[0].authority_score == 7
        # HTML should be stripped
        assert "HTML" not in articles[0].summary or "<b>" not in articles[0].summary

    def test_empty_feed_returns_empty_list(self):
        """Empty feeds should return empty list, not crash."""
        sources = [{"name": "Empty", "url": "https://empty.com", "category": "general", "authority_score": 5, "language": "en"}]

        mock_feed = MagicMock()
        mock_feed.bozo = False
        mock_feed.entries = []

        with patch("feedparser.parse", return_value=mock_feed), patch("requests.get") as mock_get:
            mock_get.return_value.content = b"<rss/>"
            mock_get.return_value.raise_for_status = MagicMock()

            fetcher = RSSFetcher(sources)
            articles = fetcher.fetch_all()

        assert len(articles) == 0

    def test_failing_source_is_logged_but_doesnt_crash(self):
        """When one source fails, others should still be processed."""
        sources = [
            {"name": "Broken", "url": "https://broken.com", "category": "general", "authority_score": 5, "language": "en"},
            {"name": "Good", "url": "https://good.com", "category": "general", "authority_score": 5, "language": "en"},
        ]

        mock_entry = {"title": "Good Article", "link": "https://good.com/1", "summary": "Content here"}
        mock_good_feed = MagicMock()
        mock_good_feed.bozo = False
        mock_good_feed.entries = [mock_entry]

        call_count = 0

        def side_effect_feedparse(content):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Connection refused")
            return mock_good_feed

        with patch("requests.get") as mock_get:
            mock_get.return_value.content = b"<rss/>"
            mock_get.return_value.raise_for_status = MagicMock()
            with patch("feedparser.parse", side_effect=side_effect_feedparse):
                fetcher = RSSFetcher(sources)
                articles = fetcher.fetch_all()

        # Second source should have succeeded
        assert len(articles) == 1
        assert articles[0].title == "Good Article"

    def test_clean_html_strips_tags(self):
        """HTML cleaning should strip all tags."""
        html = "<div><p>Hello <b>world</b></p></div>"
        result = RSSFetcher._clean_html(html)
        assert "Hello" in result
        assert "world" in result
        assert "<div>" not in result
        assert "<b>" not in result

    def test_article_content_length(self):
        """Article should track content length."""
        sources = [{"name": "Test", "url": "https://test.com", "category": "tech", "authority_score": 5, "language": "en"}]
        mock_entry = {
            "title": "Long Post",
            "link": "https://test.com/1",
            "summary": "A" * 2000,
            "content": [{"value": "B" * 500}],
        }

        mock_feed = MagicMock()
        mock_feed.bozo = False
        mock_feed.entries = [mock_entry]

        with patch("feedparser.parse", return_value=mock_feed), patch("requests.get") as mock_get:
            mock_get.return_value.content = b"<rss/>"
            mock_get.return_value.raise_for_status = MagicMock()

            fetcher = RSSFetcher(sources)
            articles = fetcher.fetch_all()

        assert articles[0].content_length > 0
