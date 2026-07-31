"""GitHub Trending repos fetcher — surfaces popular AI/ML repositories."""

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime

import requests

logger = logging.getLogger(__name__)

# Keywords that indicate an AI/ML/IT-related repo
_AI_KEYWORDS = [
    "ai", "llm", "ml", "machine-learning", "deep-learning", "gpt", "transformer",
    "neural", "nlp", "computer-vision", "diffusion", "rag", "agent",
    "langchain", "pytorch", "tensorflow", "jax", "cuda", "inference",
    "fine-tune", "embedding", "vector", "prompt", "tool", "cli",
    "rust", "go", "typescript", "python", "compiler", "database",
    "kubernetes", "docker", "serverless", "api", "framework",
]


@dataclass
class TrendingRepo:
    """Normalized GitHub trending repository."""

    name: str
    full_name: str
    url: str
    description: str
    language: str
    stars_today: int
    total_stars: int
    forks: int

    def to_article(self) -> "Article":
        from src.collect.rss_fetcher import Article

        return Article(
            title=f"[GitHub Trending] {self.full_name} — {self.description[:100] if self.description else 'No description'}",
            url=self.url,
            source_name="GitHub Trending",
            source_category="devtools",
            authority_score=5,
            language="en",
            published_at=datetime.now(UTC),
            summary=f"⭐ {self.stars_today} stars today · {self.total_stars} total · Language: {self.language} · {self.description}",
            content=self.description,
            content_length=len(self.description),
            tags=["github-trending", self.language.lower()] if self.language else ["github-trending"],
        )


class GitHubTrendingFetcher:
    """Fetches trending repositories using the GitHub API or scraping."""

    GITHUB_API = "https://api.github.com"
    SEARCH_ENDPOINT = "/search/repositories"

    def __init__(
        self,
        languages: list[str] | None = None,
        min_stars_today: int = 20,
        max_repos: int = 10,
    ) -> None:
        self.languages = languages or ["python", "typescript", "rust", "go"]
        self.min_stars_today = min_stars_today
        self.max_repos = max_repos

        self.token = os.environ.get("GITHUB_TOKEN", "")
        self.headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "DailyAIDigest/1.0",
        }
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"

    def fetch_all(self) -> list[TrendingRepo]:
        """Fetch trending repos across all configured languages."""
        all_repos: list[TrendingRepo] = []
        for lang in self.languages:
            try:
                repos = self._fetch_trending(lang)
                # Filter for AI/ML relevance
                repos = [r for r in repos if self._is_ai_related(r)]
                all_repos.extend(repos[:5])  # top 5 per language
                logger.info(f"[GitHub/{lang}] Found {len(repos)} AI-related repos")
            except Exception as e:
                logger.warning(f"[GitHub/{lang}] Failed: {e}")

        # Sort by stars_today descending, take top max_repos
        all_repos.sort(key=lambda r: r.stars_today, reverse=True)
        return all_repos[: self.max_repos]

    def _fetch_trending(self, language: str) -> list[TrendingRepo]:
        """Fetch trending repos for a language from GitHub search API."""
        yesterday = datetime.now(UTC).strftime("%Y-%m-%d")
        query = f"language:{language} created:>={yesterday}"
        url = f"{self.GITHUB_API}{self.SEARCH_ENDPOINT}"
        params = {"q": query, "sort": "stars", "order": "desc", "per_page": 20}

        resp = requests.get(url, headers=self.headers, params=params, timeout=15)

        if resp.status_code == 403:
            logger.warning("GitHub API rate limit hit — returning empty")
            return []

        resp.raise_for_status()
        data = resp.json()

        repos = []
        for item in data.get("items", []):
            repos.append(
                TrendingRepo(
                    name=item["name"],
                    full_name=item["full_name"],
                    url=item["html_url"],
                    description=item.get("description", "") or "",
                    language=item.get("language", language) or language,
                    stars_today=item.get("stargazers_count", 0),  # approximation
                    total_stars=item.get("stargazers_count", 0),
                    forks=item.get("forks_count", 0),
                )
            )
        return repos

    @staticmethod
    def _is_ai_related(repo: TrendingRepo) -> bool:
        """Check if a repo is AI/ML-related by scanning name and description."""
        text = f"{repo.name} {repo.description}".lower()
        return any(kw in text for kw in _AI_KEYWORDS)
