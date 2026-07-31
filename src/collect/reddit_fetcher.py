"""Reddit post fetcher via PRAW (Python Reddit API Wrapper)."""

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


@dataclass
class RedditPost:
    """Normalized Reddit post."""

    title: str
    url: str
    subreddit: str
    score: int
    num_comments: int
    selftext: str
    permalink: str
    created_at: datetime
    flair: str = ""

    def to_article(self) -> "Article":
        """Convert to the shared Article format. Lazy import to avoid circular dependency."""
        from src.collect.rss_fetcher import Article

        content = self.selftext
        return Article(
            title=self.title,
            url=self.url if self.url.startswith("http") else f"https://reddit.com{self.permalink}",
            source_name=f"r/{self.subreddit}",
            source_category="community",
            authority_score=4,
            language="en",
            published_at=self.created_at,
            summary=self.selftext[:500] if self.selftext else "",
            content=content,
            content_length=len(content),
            tags=[self.flair] if self.flair else [],
        )


class RedditFetcher:
    """Fetches top posts from configured subreddits."""

    def __init__(
        self,
        subreddits: list[str],
        post_limit: int = 15,
        min_score: int = 10,
    ) -> None:
        self.subreddits = subreddits
        self.post_limit = post_limit
        self.min_score = min_score

        # PRAW is optional — gracefully degrade if not configured
        self.reddit = None
        client_id = os.environ.get("REDDIT_CLIENT_ID")
        client_secret = os.environ.get("REDDIT_CLIENT_SECRET")

        if client_id and client_secret:
            try:
                import praw

                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent="DailyAIDigest/1.0",
                )
                logger.info("Reddit client initialized via PRAW")
            except Exception as e:
                logger.warning(f"Reddit client initialization failed: {e}")
        else:
            logger.info("Reddit credentials not set — using public JSON API fallback")

    def fetch_all(self) -> list[RedditPost]:
        """Fetch posts from all configured subreddits."""
        all_posts: list[RedditPost] = []
        for sub in self.subreddits:
            try:
                posts = self._fetch_subreddit(sub)
                all_posts.extend(posts)
                logger.info(f"[r/{sub}] Fetched {len(posts)} posts")
            except Exception as e:
                logger.warning(f"[r/{sub}] Failed: {e}")
        return all_posts

    def _fetch_subreddit(self, subreddit: str) -> list[RedditPost]:
        """Fetch posts from a single subreddit."""
        if self.reddit:
            return self._fetch_via_praw(subreddit)
        return self._fetch_via_json(subreddit)

    def _fetch_via_praw(self, subreddit: str) -> list[RedditPost]:
        """Fetch using authenticated PRAW client."""
        sub = self.reddit.subreddit(subreddit)
        posts = []
        for post in sub.hot(limit=self.post_limit):
            if post.score >= self.min_score:
                posts.append(self._parse_praw_post(post))
        return posts

    def _fetch_via_json(self, subreddit: str) -> list[RedditPost]:
        """Fetch using public .json API (no auth needed, rate-limited)."""
        import requests

        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={self.post_limit}"
        resp = requests.get(
            url,
            headers={"User-Agent": "DailyAIDigest/1.0"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        posts = []
        for child in data.get("data", {}).get("children", []):
            post_data = child["data"]
            if post_data.get("score", 0) >= self.min_score:
                posts.append(self._parse_json_post(post_data))
        return posts

    def _parse_praw_post(self, post) -> RedditPost:
        return RedditPost(
            title=post.title,
            url=post.url,
            subreddit=str(post.subreddit),
            score=post.score,
            num_comments=post.num_comments,
            selftext=post.selftext,
            permalink=post.permalink,
            created_at=datetime.fromtimestamp(post.created_utc, tz=UTC),
            flair=post.link_flair_text or "",
        )

    def _parse_json_post(self, data: dict) -> RedditPost:
        return RedditPost(
            title=data["title"],
            url=data.get("url", ""),
            subreddit=data.get("subreddit", ""),
            score=data.get("score", 0),
            num_comments=data.get("num_comments", 0),
            selftext=data.get("selftext", ""),
            permalink=data.get("permalink", ""),
            created_at=datetime.fromtimestamp(data["created_utc"], tz=UTC),
            flair=data.get("link_flair_text", ""),
        )
