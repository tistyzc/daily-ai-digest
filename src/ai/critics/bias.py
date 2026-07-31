"""Hype and commercial bias detector — flags overly promotional or biased language."""

import json
import logging

from src.ai.deepseek_client import DeepSeekClient

logger = logging.getLogger(__name__)

_BIAS_SYSTEM = """You are an editorial reviewer checking for bias and excessive hype in a tech news digest.

Look for:
1. Commercial/promotional language: "revolutionary", "game-changing", "the best", "unprecedented" without evidence
2. Uncritical product launch coverage: reads like a press release
3. Missing counterpoints: claims without acknowledging limitations or competing approaches
4. Vendor lock-in bias: overly promoting one company's ecosystem
5. Hype inflation: describing incremental improvements as breakthroughs

A LITTLE enthusiasm is fine — this is a tech digest, not an academic journal. Flag only clearly excessive or misleading language.

Respond with ONLY a JSON object:
{
  "has_bias_issues": true/false,
  "bias_level": "none" | "mild" | "moderate" | "severe",
  "issues": ["description of issues found"],
  "suggested_fix": "how to rephrase to be more balanced, or null"
}"""


class BiasCritic:
    """Detects commercial bias and hype in article summaries."""

    def __init__(self, client: DeepSeekClient, model: str | None = None) -> None:
        self.client = client
        self.model = model or client.default_model

    def check(self, title: str, summary_en: str, summary_zh: str) -> dict:
        """Check summaries for bias and hype.

        Returns:
            dict with has_bias_issues, bias_level, issues, suggested_fix.
        """
        text = f"Title: {title}\n\nEnglish summary: {summary_en}\n\nChinese summary: {summary_zh}"

        try:
            raw = self.client.chat(
                messages=[
                    {"role": "system", "content": _BIAS_SYSTEM},
                    {"role": "user", "content": text},
                ],
                model=self.model,
                max_tokens=300,
                temperature=0.1,
            )
            result = json.loads(raw.strip().removeprefix("```json").removesuffix("```"))
            if result["has_bias_issues"]:
                logger.warning(
                    f"[BIAS] {result.get('bias_level', 'unknown')} bias in '{title[:60]}': {result.get('issues')}"
                )
            return result
        except Exception as e:
            logger.warning(f"[BIAS] Check failed for '{title[:60]}': {e}")
            return {"has_bias_issues": False, "bias_level": "none", "issues": [], "suggested_fix": None}
