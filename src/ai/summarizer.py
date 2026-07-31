"""Bilingual article summarizer — English originals + Chinese translation/insights."""

import json
import logging

from src.ai.deepseek_client import DeepSeekClient
from src.collect.rss_fetcher import Article

logger = logging.getLogger(__name__)

_SUMMARY_SYSTEM = """You are a senior IT/AI industry analyst writing for a daily bilingual (English + Chinese) tech digest.

Your task: given an article, produce a structured summary.

Respond with ONLY a JSON object:
{
  "en_summary": "2-3 sentence English summary, factual and precise. Capture the key finding or announcement.",
  "zh_summary": "2-3 sentence Chinese translation/adaptation. NOT a word-for-word translation — make it natural Chinese that Chinese developers would enjoy reading.",
  "key_points": ["key point 1 in English", "key point 2 in English"],
  "key_points_zh": ["要点1", "要点2"],
  "impact": "one sentence on why this matters to developers/AI practitioners, in English",
  "impact_zh": "对开发者/AI从业者有什么影响，一句中文"
}"""

_SUMMARY_USER = """Article Title: {title}
Source: {source}
Original Summary: {summary}

Full Content:
{content}"""


class BilingualSummarizer:
    """Summarizes articles in both English and Chinese using DeepSeek."""

    def __init__(self, client: DeepSeekClient, model: str | None = None) -> None:
        self.client = client
        self.model = model or client.default_model

    def summarize(self, article: Article) -> dict:
        """Generate bilingual summary for a single article.

        Returns:
            dict with en_summary, zh_summary, key_points, key_points_zh, impact, impact_zh.
        """
        content_preview = article.content[:3000]  # limit context window cost

        prompt = _SUMMARY_USER.format(
            title=article.title,
            source=article.source_name,
            summary=article.summary[:500],
            content=content_preview,
        )

        try:
            raw = self.client.chat(
                messages=[
                    {"role": "system", "content": _SUMMARY_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                max_tokens=800,
                temperature=0.3,
            )

            # Strip markdown code fences
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(raw)

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Summarizer parse error for '{article.title[:60]}': {e}")
            return {
                "en_summary": article.summary[:300],
                "zh_summary": article.summary[:300],
                "key_points": [],
                "key_points_zh": [],
                "impact": "",
                "impact_zh": "",
            }

    def summarize_batch(self, articles: list[Article]) -> list[dict]:
        """Summarize a batch of articles."""
        results: list[dict] = []
        for i, article in enumerate(articles):
            logger.info(f"Summarizing [{i + 1}/{len(articles)}]: {article.title[:80]}")
            result = self.summarize(article)
            result["_article"] = article
            results.append(result)
        return results
