"""Tests for DeepSeek client."""

from unittest.mock import MagicMock, patch

import pytest

from src.ai.deepseek_client import DeepSeekClient


class TestDeepSeekClient:
    def test_requires_api_key(self):
        """Should raise ValueError if no API key is provided."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
                DeepSeekClient(api_key="")  # Explicit empty string triggers env var check
            # But explicit api_key works:
            client = DeepSeekClient(api_key="sk-test-123")
            assert client.api_key == "sk-test-123"

    def test_env_var_api_key(self):
        """Should read API key from environment."""
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-env-test"}, clear=True):
            client = DeepSeekClient()
            assert client.api_key == "sk-env-test"

    def test_custom_base_url_and_model(self):
        """Should accept custom base URL and model."""
        client = DeepSeekClient(api_key="sk-test", base_url="https://custom.api.com", default_model="deepseek-v4-pro")
        assert client.base_url == "https://custom.api.com"
        assert client.default_model == "deepseek-v4-pro"

    def test_chat_calls_openai_sdk(self):
        """chat() should call the OpenAI SDK and return the message content."""
        client = DeepSeekClient(api_key="sk-test")

        mock_choice = MagicMock()
        mock_choice.message.content = "Hello, world!"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch.object(client.client.chat.completions, "create", return_value=mock_response) as mock_create:
            result = client.chat(messages=[{"role": "user", "content": "Hi"}])

            mock_create.assert_called_once()
            assert result == "Hello, world!"

    def test_chat_json_adds_response_format(self):
        """chat_json() should add JSON response_format."""
        client = DeepSeekClient(api_key="sk-test")

        mock_choice = MagicMock()
        mock_choice.message.content = '{"key": "value"}'

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch.object(client.client.chat.completions, "create", return_value=mock_response) as mock_create:
            result = client.chat_json(messages=[{"role": "user", "content": "Give me JSON"}])

            call_kwargs = mock_create.call_args[1]
            assert call_kwargs.get("response_format") == {"type": "json_object"}
            assert result == '{"key": "value"}'

    def test_model_override(self):
        """Should use override model when specified."""
        client = DeepSeekClient(api_key="sk-test", default_model="deepseek-chat")

        mock_choice = MagicMock()
        mock_choice.message.content = "ok"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch.object(client.client.chat.completions, "create", return_value=mock_response) as mock_create:
            client.chat(messages=[{"role": "user", "content": "Hi"}], model="deepseek-reasoner")

            assert mock_create.call_args[1]["model"] == "deepseek-reasoner"
