"""Tests for bilingual summarizer."""

import json

from src.ai.summarizer import BilingualSummarizer


class TestBilingualSummarizer:
    def test_summarize_returns_expected_keys(self, mock_deepseek_client, sample_articles):
        """Summarizer should return all expected fields."""
        mock_deepseek_client.chat.return_value = json.dumps(
            {
                "en_summary": "OpenAI released GPT-5 with improved reasoning.",
                "zh_summary": "OpenAI发布了GPT-5，推理能力大幅提升。",
                "key_points": ["Reasoning improved", "Coding better"],
                "key_points_zh": ["推理提升", "编程增强"],
                "impact": "Major step forward for LLM capabilities.",
                "impact_zh": "对AI应用开发有重大影响。",
            }
        )

        summarizer = BilingualSummarizer(mock_deepseek_client)
        result = summarizer.summarize(sample_articles[0])

        assert "en_summary" in result
        assert "zh_summary" in result
        assert "key_points" in result
        assert "key_points_zh" in result
        assert "impact" in result
        assert "impact_zh" in result
        assert len(result["en_summary"]) > 0
        assert len(result["zh_summary"]) > 0

    def test_summarize_handles_malformed_json(self, mock_deepseek_client, sample_articles):
        """If DeepSeek returns invalid JSON, should return fallback."""
        mock_deepseek_client.chat.return_value = "not valid json {{{"

        summarizer = BilingualSummarizer(mock_deepseek_client)
        result = summarizer.summarize(sample_articles[0])

        # Should return fallback structure
        assert "en_summary" in result
        assert "zh_summary" in result

    def test_summarize_batch(self, mock_deepseek_client, sample_articles):
        """Batch summarize should process all articles."""
        mock_deepseek_client.chat.return_value = json.dumps(
            {
                "en_summary": "Summary here.",
                "zh_summary": "这里是摘要。",
                "key_points": ["point 1"],
                "key_points_zh": ["要点1"],
                "impact": "Important.",
                "impact_zh": "很重要。",
            }
        )

        summarizer = BilingualSummarizer(mock_deepseek_client)
        results = summarizer.summarize_batch(sample_articles[:3])

        assert len(results) == 3
        for r in results:
            assert "_article" in r
            assert "en_summary" in r
