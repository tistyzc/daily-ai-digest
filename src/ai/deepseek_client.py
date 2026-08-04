"""DeepSeek API client wrapper — OpenAI SDK compatible."""

import logging
import os
import time

from openai import InternalServerError, OpenAI

logger = logging.getLogger(__name__)


class DeepSeekClient:
    """Thin wrapper around the DeepSeek API via OpenAI SDK.

    Handles 503 (service busy) with automatic retries.
    """

    DEFAULT_BASE_URL = "https://api.deepseek.com"
    DEFAULT_MODEL = "deepseek-chat"
    RETRY_DELAYS = [10, 30, 60]  # seconds between retries for 503 errors

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not set. Set the environment variable or pass api_key explicitly."
            )

        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.default_model = default_model or self.DEFAULT_MODEL

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=0,  # We handle retries ourselves for 503
        )
        logger.info(f"DeepSeek client initialized: model={self.default_model} base={self.base_url}")

    def _call_with_retry(self, **kwargs) -> str:
        """Call the API with retries for 503 Service Unavailable errors."""
        last_error = None
        for attempt in range(len(self.RETRY_DELAYS) + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except InternalServerError as e:
                last_error = e
                if attempt < len(self.RETRY_DELAYS):
                    delay = self.RETRY_DELAYS[attempt]
                    logger.warning(f"DeepSeek 503 — retrying in {delay}s (attempt {attempt + 1}/{len(self.RETRY_DELAYS) + 1})")
                    time.sleep(delay)
                else:
                    raise
        raise last_error  # type: ignore[return-value]

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int = 500,
        temperature: float = 0.3,
        response_format: dict | None = None,
    ) -> str:
        """Send a chat completion request and return the text response.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts.
            model: Override the default model.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0-2.0).
            response_format: Optional {"type": "json_object"} for structured output.

        Returns:
            The model's text response. Retries on 503 errors.
        """
        kwargs: dict = {
            "model": model or self.default_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        return self._call_with_retry(**kwargs)

    def chat_json(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int = 500,
        temperature: float = 0.1,
    ) -> str:
        """Same as chat() but requests JSON output. Retries on 503 errors."""
        return self.chat(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
