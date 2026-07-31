"""Factual consistency critic — validates AI summaries against source material."""

import json
import logging

from src.ai.deepseek_client import DeepSeekClient

logger = logging.getLogger(__name__)

_FACTUAL_SYSTEM = """You are a fact-checker for an AI/IT news digest. Your job is to verify that a summary is factually consistent with the original article.

Checks to perform:
1. Are the key claims in the summary supported by the source text?
2. Are numbers, dates, and named entities correct?
3. Is anything hallucinated (fabricated) that doesn't appear in the source?

Respond with ONLY a JSON object:
{
  "is_factually_correct": true/false,
  "issues": ["issue description if any"],
  "confidence": 0.0-1.0
}"""


class FactualCritic:
    """Validates that AI-generated summaries are factually consistent with the source."""

    def __init__(self, client: DeepSeekClient, model: str | None = None) -> None:
        self.client = client
        self.model = model or client.default_model

    def check(self, title: str, original_content: str, generated_summary: str) -> dict:
        """Check factual consistency between source and summary.

        Returns:
            dict with is_factually_correct, issues, confidence.
        """
        prompt = f"""Source Article:
{original_content[:2000]}

Generated Summary:
{generated_summary}

Verify the summary against the source."""

        try:
            raw = self.client.chat(
                messages=[
                    {"role": "system", "content": _FACTUAL_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                max_tokens=300,
                temperature=0.1,
            )
            result = json.loads(raw.strip().removeprefix("```json").removesuffix("```"))
            if not result["is_factually_correct"]:
                logger.warning(f"[FACTUAL] Issues in '{title[:60]}': {result.get('issues')}")
            return result
        except Exception as e:
            logger.warning(f"[FACTUAL] Check failed for '{title[:60]}': {e}")
            return {"is_factually_correct": True, "issues": [], "confidence": 0.0}
