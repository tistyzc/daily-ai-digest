"""Tests for the 4 critic modules."""

import json
from unittest.mock import MagicMock

from src.ai.critics.bias import BiasCritic
from src.ai.critics.factual import FactualCritic
from src.ai.critics.privacy import PrivacyCritic
from src.ai.critics.safety import SafetyCritic


class TestFactualCritic:
    def test_passes_when_consistent(self, mock_deepseek_client):
        """Should pass when summary matches source."""
        mock_deepseek_client.chat.return_value = json.dumps(
            {"is_factually_correct": True, "issues": [], "confidence": 0.95}
        )

        critic = FactualCritic(mock_deepseek_client)
        result = critic.check("AI News", "Source content about AI.", "Summary about AI.")

        assert result["is_factually_correct"] is True
        assert result["confidence"] > 0.9

    def test_flags_inconsistency(self, mock_deepseek_client):
        """Should flag when summary doesn't match source."""
        mock_deepseek_client.chat.return_value = json.dumps(
            {"is_factually_correct": False, "issues": ["Claim about revenues not in source"], "confidence": 0.3}
        )

        critic = FactualCritic(mock_deepseek_client)
        result = critic.check("AI News", "Source about LLMs.", "Claims $1B revenue.")

        assert result["is_factually_correct"] is False
        assert len(result["issues"]) > 0

    def test_handles_parse_error(self, mock_deepseek_client):
        """Should return safe default on parse error."""
        mock_deepseek_client.chat.return_value = "not json"

        critic = FactualCritic(mock_deepseek_client)
        result = critic.check("Title", "Content", "Summary")

        # Fail-safe: assume correct if we can't verify
        assert result["is_factually_correct"] is True


class TestSafetyCritic:
    def test_passes_safe_content(self, mock_deepseek_client):
        """Normal tech content should pass safety check."""
        mock_deepseek_client.chat.return_value = json.dumps(
            {"is_safe": True, "flags": [], "reason": "ok"}
        )

        critic = SafetyCritic(mock_deepseek_client)
        result = critic.check("Kubernetes Update", "K8s 2.0 released.", {"en_summary": "K8s update."})

        assert result["is_safe"] is True

    def test_flags_harmful_content(self, mock_deepseek_client):
        """Harmful content should be flagged."""
        mock_deepseek_client.chat.return_value = json.dumps(
            {"is_safe": False, "flags": ["hate-speech"], "reason": "Contains discriminatory language"}
        )

        critic = SafetyCritic(mock_deepseek_client)
        result = critic.check("Bad Post", "Harmful content here.", {"en_summary": "Bad stuff."})

        assert result["is_safe"] is False
        assert "hate-speech" in result["flags"]


class TestBiasCritic:
    def test_passes_balanced_content(self, mock_deepseek_client):
        """Balanced content should pass bias check."""
        mock_deepseek_client.chat.return_value = json.dumps(
            {"has_bias_issues": False, "bias_level": "none", "issues": [], "suggested_fix": None}
        )

        critic = BiasCritic(mock_deepseek_client)
        result = critic.check("GPT-5 Release", "OpenAI released GPT-5 with some improvements.", "OpenAI发布GPT-5。")

        assert result["has_bias_issues"] is False

    def test_flags_hype(self, mock_deepseek_client):
        """Overly promotional content should be flagged."""
        mock_deepseek_client.chat.return_value = json.dumps(
            {
                "has_bias_issues": True,
                "bias_level": "severe",
                "issues": ["Uses revolutionary without evidence"],
                "suggested_fix": "Remove 'revolutionary', describe specific improvements",
            }
        )

        critic = BiasCritic(mock_deepseek_client)
        result = critic.check(
            "GPT-5",
            "GPT-5 is a revolutionary, game-changing breakthrough that will change everything forever!",
            "GPT-5是革命性的、改变游戏规则的突破！",
        )

        assert result["has_bias_issues"] is True
        assert result["bias_level"] == "severe"


class TestPrivacyCritic:
    def test_passes_clean_content(self, mock_deepseek_client):
        """Content without PII should pass."""
        mock_deepseek_client.chat.return_value = json.dumps(
            {"has_pii": False, "pii_types": [], "action": "ok"}
        )

        critic = PrivacyCritic(mock_deepseek_client)
        result = critic.check("Tech News", "A new framework was released.", {"en_summary": "Framework release."})

        assert result["has_pii"] is False

    def test_flags_email_addresses(self, mock_deepseek_client):
        """Content with email addresses should be flagged."""
        mock_deepseek_client.chat.return_value = json.dumps(
            {"has_pii": True, "pii_types": ["email"], "action": "redact"}
        )

        critic = PrivacyCritic(mock_deepseek_client)
        result = critic.check("Contact Info", "Contact john@example.com for details.", {"en_summary": "Contact info."})

        assert result["has_pii"] is True
        assert "email" in result["pii_types"]

    def test_regex_pre_screen_finds_emails(self):
        """The regex-based fast pre-screen should detect email patterns."""
        critic = PrivacyCritic(MagicMock())
        hits = critic._regex_scan("Contact me at test@example.com or call 555-123-4567")
        assert "email" in hits
        assert "phone" in hits

    def test_regex_pre_screen_clean_text(self):
        """Clean text should produce no regex hits."""
        critic = PrivacyCritic(MagicMock())
        hits = critic._regex_scan("Kubernetes 2.0 was released today with new features.")
        assert len(hits) == 0
