"""
Gemini LLM provider — fallback generation provider.

Uses the google-genai SDK for Gemini API access.
"""

from __future__ import annotations

import time

from app.generation.base import BaseGenerator
from app.models.generation import GenerationResult, ProviderType
from app.observability.logging import get_logger

logger = get_logger(__name__)


class GeminiGenerator(BaseGenerator):
    """Gemini API generation provider (fallback)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        timeout: int = 30,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> GenerationResult:
        """Generate text using Gemini API."""
        start = time.perf_counter()

        try:
            client = self._get_client()

            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            from google.genai import types

            response = client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )

            elapsed = (time.perf_counter() - start) * 1000
            text = response.text or ""

            logger.info(
                "gemini_generation",
                model=self.model,
                latency_ms=round(elapsed, 2),
            )

            return GenerationResult(
                text=text,
                provider=ProviderType.GEMINI,
                model=self.model,
                latency_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("gemini_generation_failed", error=str(e), model=self.model)
            return GenerationResult(
                text="",
                provider=ProviderType.GEMINI,
                model=self.model,
                latency_ms=elapsed,
                error=str(e),
            )

    def is_available(self) -> bool:
        return bool(self.api_key)

    @property
    def provider_name(self) -> str:
        return "gemini"
