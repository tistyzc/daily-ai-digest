"""Shared test fixtures and mocks for daily-ai-digest tests."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from src.collect.rss_fetcher import Article


@pytest.fixture
def sample_articles() -> list[Article]:
    """A set of sample articles for testing."""
    now = datetime.now(UTC)
    return [
        Article(
            title="GPT-5 Released with Major Improvements",
            url="https://example.com/gpt5",
            source_name="OpenAI Blog",
            source_category="ai-research",
            authority_score=9,
            language="en",
            published_at=now,
            summary="OpenAI announced GPT-5 with significant improvements in reasoning and coding.",
            content="OpenAI announced GPT-5 today. The new model features breakthrough reasoning capabilities... " * 20,
            content_length=1500,
        ),
        Article(
            title="GPT-5 Released — A New Era of AI",  # Similar title for dedup testing
            url="https://example.com/gpt5-techcrunch",
            source_name="TechCrunch AI",
            source_category="business",
            authority_score=7,
            language="en",
            published_at=now,
            summary="OpenAI released GPT-5. The new model is a game changer.",
            content="TechCrunch reports on the GPT-5 launch... " * 20,
            content_length=1400,
        ),
        Article(
            title="Short post",  # Too short — should be gated
            url="https://example.com/short",
            source_name="Hacker News",
            source_category="community",
            authority_score=8,
            language="en",
            published_at=now,
            summary="Short.",
            content="Just a brief note.",
            content_length=16,
        ),
        Article(
            title="大模型在金融领域的应用",
            url="https://example.com/finance-llm",
            source_name="机器之心",
            source_category="ai-business",
            authority_score=8,
            language="zh",
            published_at=now,
            summary="大模型在金融风控、投研、客服等场景的应用越来越广泛...",
            content="近年来，大语言模型在金融行业的应用不断深入..." * 20,
            content_length=1200,
        ),
        Article(
            title="Rust 2026 Edition Released",
            url="https://example.com/rust2026",
            source_name="Ars Technica",
            source_category="technology",
            authority_score=8,
            language="en",
            published_at=now,
            summary="Rust 2026 edition brings new features including...",
            content="The Rust team announced the 2026 edition today..." * 20,
            content_length=1300,
        ),
        Article(
            title="Kubernetes 2.0 Announced at KubeCon",
            url="https://example.com/k8s-2",
            source_name="The Verge Tech",
            source_category="technology",
            authority_score=7,
            language="en",
            published_at=now,
            summary="Kubernetes 2.0 brings major changes...",
            content="At KubeCon this week, the Kubernetes team announced..." * 30,
            content_length=1800,
        ),
    ]


@pytest.fixture
def mock_deepseek_client():
    """A mock DeepSeek client that returns controlled responses."""
    client = MagicMock()
    client.chat.return_value = '{"score": 0.85, "reason": "Highly relevant AI news."}'
    client.chat_json.return_value = '[{"topic": "Large Language Models"}, {"topic": "AI Research & Breakthroughs"}]'
    client.default_model = "deepseek-chat"
    return client


@pytest.fixture
def sample_config() -> dict:
    """Minimal pipeline config for testing."""
    return {
        "pipeline": {
            "min_content_length": 600,
            "dedup_similarity_threshold": 85,
            "scoring": {
                "authority_boost": 3,
                "cross_source_boost": 5,
                "recency_boost": 2,
                "already_reported_penalty": -5,
            },
            "min_relevance_score": 0.7,
            "max_articles": 30,
            "lookback_days": 2,
        },
        "deepseek": {
            "summary_model": "deepseek-chat",
            "critic_model": "deepseek-chat",
            "max_tokens_summary": 500,
            "max_tokens_critic": 300,
            "temperature": 0.3,
        },
    }
