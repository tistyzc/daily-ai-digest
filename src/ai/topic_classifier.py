"""Topic classifier — assigns articles to categories using DeepSeek."""

import json
import logging

from src.ai.deepseek_client import DeepSeekClient
from src.collect.rss_fetcher import Article

logger = logging.getLogger(__name__)

# Taxonomy of topics for the digest
TOPICS = [
    "Large Language Models",
    "AI Research & Breakthroughs",
    "Developer Tools & Platforms",
    "Open Source",
    "AI Business & Funding",
    "Security & Safety",
    "Infrastructure & Cloud",
    "Programming Languages",
    "Industry Trends & Analysis",
    "AI Ethics & Policy",
]

_CLASSIFY_SYSTEM = f"""You are a tech news classifier. Assign each article to the most appropriate topic from this list:

{chr(10).join(f'- {t}' for t in TOPICS)}

Respond with ONLY a JSON array: [{{"title": "article title", "topic": "topic name"}}]"""


class TopicClassifier:
    """Classifies articles into predefined topics."""

    def __init__(self, client: DeepSeekClient, model: str | None = None) -> None:
        self.client = client
        self.model = model or client.default_model

    def classify(self, articles: list[Article]) -> dict[str, list[Article]]:
        """Classify articles into topic buckets.

        Returns:
            Dict mapping topic_name -> list of articles.
        """
        if not articles:
            return {}

        # Batch classify: send titles in groups of 10
        buckets: dict[str, list[Article]] = {t: [] for t in TOPICS}

        for i in range(0, len(articles), 10):
            batch = articles[i : i + 10]
            try:
                assignments = self._classify_batch(batch)
                for article, topic in zip(batch, assignments, strict=False):
                    if topic in buckets:
                        buckets[topic].append(article)
                    else:
                        buckets["Industry Trends & Analysis"].append(article)
            except Exception as e:
                logger.warning(f"Classification batch failed: {e}")
                # Put all in default bucket
                for article in batch:
                    buckets["Industry Trends & Analysis"].append(article)

        # Remove empty buckets
        return {k: v for k, v in buckets.items() if v}

    def _classify_batch(self, articles: list[Article]) -> list[str]:
        """Classify a small batch of articles."""
        titles = "\n".join(f"- {a.title[:120]}" for a in articles)

        raw = self.client.chat_json(
            messages=[
                {"role": "system", "content": _CLASSIFY_SYSTEM},
                {"role": "user", "content": f"Classify these articles:\n\n{titles}"},
            ],
            model=self.model,
            max_tokens=500,
            temperature=0.1,
        )

        data = json.loads(raw)
        return [item.get("topic", "Industry Trends & Analysis") for item in data]
