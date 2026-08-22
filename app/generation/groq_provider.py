"""
Groq LLM provider — primary generation provider.

Uses Groq's OpenAI-compatible API for fast inference on open-source models.
"""

from __future__ import annotations

import time

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.generation.base import BaseGenerator
from app.models.generation import GenerationResult, ProviderType
from app.observability.logging import get_logger

logger = get_logger(__name__)


class GroqGenerator(BaseGenerator):
    """Groq API generation provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        max_retries: int = 2,
        timeout: int = 30,
    ):
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self.api_key, timeout=self.timeout)
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> GenerationResult:
        """Generate text using Groq API."""
        start = time.perf_counter()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            elapsed = (time.perf_counter() - start) * 1000
            text = response.choices[0].message.content or ""
            # Strip <think>...</think> blocks from reasoning models
            import re
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            usage = response.usage

            logger.info(
                "groq_generation",
                model=self.model,
                latency_ms=round(elapsed, 2),
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
            )

            return GenerationResult(
                text=text,
                provider=ProviderType.GROQ,
                model=self.model,
                latency_ms=elapsed,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("groq_generation_failed", error=str(e), model=self.model)
            return GenerationResult(
                text="",
                provider=ProviderType.GROQ,
                model=self.model,
                latency_ms=elapsed,
                error=str(e),
            )

    def is_available(self) -> bool:
        return bool(self.api_key)

    @property
    def provider_name(self) -> str:
        return "groq"
