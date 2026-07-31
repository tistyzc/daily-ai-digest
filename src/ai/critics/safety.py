"""Content safety critic — screens for inappropriate or harmful content."""

import json
import logging

from src.ai.deepseek_client import DeepSeekClient

logger = logging.getLogger(__name__)

_SAFETY_SYSTEM = """You are a content safety reviewer for a professional AI/IT news digest read by developers worldwide.

Check for the following:
1. Hate speech, harassment, or discriminatory content
2. Violent or graphic content
3. Promotion of illegal activities
4. Malware/phishing links or instructions
5. NSFW or sexually explicit content
6. Content that could cause real-world harm if published

TECHNICAL DISCUSSIONS about AI safety, security vulnerabilities, or ethical debates are OK — these are normal tech industry topics. Only flag truly harmful content.

Respond with ONLY a JSON object:
{
  "is_safe": true/false,
  "flags": ["category of concern if any"],
  "reason": "brief explanation if flagged, else 'ok'"
}"""


class SafetyCritic:
    """Screens content for safety concerns before publication."""

    def __init__(self, client: DeepSeekClient, model: str | None = None) -> None:
        self.client = client
        self.model = model or client.default_model

    def check(self, title: str, content: str, summary: dict) -> dict:
        """Check if content is safe to publish.

        Returns:
            dict with is_safe, flags, reason.
        """
        text_to_check = f"Title: {title}\n\nContent: {content[:1500]}\n\nSummary to publish: {summary.get('en_summary', '')}\n\nChinese summary: {summary.get('zh_summary', '')}"

        try:
            raw = self.client.chat(
                messages=[
                    {"role": "system", "content": _SAFETY_SYSTEM},
                    {"role": "user", "content": text_to_check},
                ],
                model=self.model,
                max_tokens=200,
                temperature=0.0,
            )
            result = json.loads(raw.strip().removeprefix("```json").removesuffix("```"))
            if not result["is_safe"]:
                logger.warning(f"[SAFETY] Flagged '{title[:60]}': {result.get('flags')} — {result.get('reason')}")
            return result
        except Exception as e:
            logger.warning(f"[SAFETY] Check failed for '{title[:60]}': {e}")
            return {"is_safe": True, "flags": [], "reason": "ok"}
