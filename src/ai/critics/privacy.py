"""Privacy screener — detects and flags personally identifiable information (PII)."""

import json
import logging
import re

from src.ai.deepseek_client import DeepSeekClient

logger = logging.getLogger(__name__)

# Regex patterns for common PII types (fast pre-screening)
_PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
}


_PRIVACY_SYSTEM = """You are a privacy reviewer for a public tech news digest.

Check the content for:
1. Email addresses
2. Phone numbers
3. Physical addresses
4. Personal identification numbers
5. Private API keys or tokens
6. Internal company URLs or endpoints
7. Non-public personal data

Public figures' names, company names, and public URLs are FINE. Only flag truly private/sensitive data.

Respond with ONLY a JSON object:
{
  "has_pii": true/false,
  "pii_types": ["type of PII found"],
  "action": "remove" | "redact" | "ok"
}"""


class PrivacyCritic:
    """Screens content for PII and sensitive data before publication."""

    def __init__(self, client: DeepSeekClient, model: str | None = None) -> None:
        self.client = client
        self.model = model or client.default_model

    def check(self, title: str, content: str, summary: dict) -> dict:
        """Check for PII in article and summary.

        Returns:
            dict with has_pii, pii_types, action.
        """
        combined = f"Title: {title}\nContent: {content[:1500]}\nSummary: {summary.get('en_summary', '')}"

        # Fast pre-screen with regex
        regex_hits = self._regex_scan(combined)
        if not regex_hits:
            # No regex matches — likely clean, but do a quick LLM pass for nuanced cases
            pass

        try:
            raw = self.client.chat(
                messages=[
                    {"role": "system", "content": _PRIVACY_SYSTEM},
                    {"role": "user", "content": combined},
                ],
                model=self.model,
                max_tokens=200,
                temperature=0.0,
            )
            result = json.loads(raw.strip().removeprefix("```json").removesuffix("```"))
            if result["has_pii"]:
                logger.warning(f"[PRIVACY] PII found in '{title[:60]}': {result.get('pii_types')}")
            return result
        except Exception as e:
            logger.warning(f"[PRIVACY] Check failed for '{title[:60]}': {e}")
            return {"has_pii": False, "pii_types": [], "action": "ok"}

    def _regex_scan(self, text: str) -> list[str]:
        """Fast pre-scan with regex patterns. Returns list of PII types found."""
        hits = []
        for pii_type, pattern in _PII_PATTERNS.items():
            if pattern.search(text):
                hits.append(pii_type)
        return hits
