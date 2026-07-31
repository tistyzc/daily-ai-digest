"""Relevance filter — uses DeepSeek to score articles for IT/AI relevance."""

import json
import logging

from openai import OpenAI

from src.collect.rss_fetcher import Article

logger = logging.getLogger(__name__)

_RELEVANCE_PROMPT = """You are an AI/IT industry news curator. Score the following article on its relevance to IT and AI professionals on a scale of 0.0 to 1.0.

Scoring guidelines:
- 0.9-1.0: Directly about AI/ML/LLMs, major infrastructure changes, critical security vulnerabilities, groundbreaking research
- 0.7-0.9: Important tech industry news, new developer tools, significant product launches
- 0.5-0.7: General tech news with some relevance to developers
- 0.3-0.5: Tangentially tech-related
- 0.0-0.3: Not relevant to IT/AI professionals

Article Title: {title}
Article Source: {source}
Article Summary: {summary}

Respond with ONLY a JSON object: {{"score": <float 0.0-1.0>, "reason": "<one short sentence in English>"}}"""


class RelevanceFilter:
    """Filters articles by AI/IT relevance using DeepSeek scoring."""

    def __init__(self, client: OpenAI, min_score: float = 0.7, model: str = "deepseek-chat") -> None:
        self.client = client
        self.min_score = min_score
        self.model = model

    def filter(self, articles: list[Article]) -> list[Article]:
        """Score all articles and return those above the relevance threshold."""
        if not articles:
            return []

        passed: list[Article] = []
        for article in articles:
            try:
                score = self._score_article(article)
                if score >= self.min_score:
                    passed.append(article)
                    logger.debug(f"RELEVANCE {score:.2f} PASS: {article.title[:80]}")
                else:
                    logger.debug(f"RELEVANCE {score:.2f} DROP: {article.title[:80]}")
            except Exception as e:
                logger.warning(f"Relevance scoring failed for '{article.title[:60]}': {e}")
                # Include article if scoring fails (fail-open)
                passed.append(article)

        logger.info(f"Relevance filter: {len(passed)}/{len(articles)} passed (min {self.min_score})")
        return passed

    def _score_article(self, article: Article) -> float:
        """Score a single article's relevance via DeepSeek."""
        prompt = _RELEVANCE_PROMPT.format(
            title=article.title,
            source=article.source_name,
            summary=article.summary[:800],
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1,
        )

        raw = response.choices[0].message.content.strip()
        # Remove markdown code fences if present
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw)
        return float(data["score"])
