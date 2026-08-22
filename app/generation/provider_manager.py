"""
Provider manager with automatic fallback.

Routes generation requests to Groq (primary) and falls back to Gemini
on rate limit, timeout, or provider failure. Provider-specific logic
is fully encapsulated — the rest of the application sees a single interface.
"""

from __future__ import annotations

from app.generation.base import BaseGenerator
from app.generation.groq_provider import GroqGenerator
from app.generation.gemini_provider import GeminiGenerator
from app.models.generation import GenerationResult
from app.observability.logging import get_logger

logger = get_logger(__name__)


class ProviderManager:
    """
    Manages LLM providers with automatic fallback.

    Primary: Groq → Fallback: Gemini

    Falls back when the primary returns an error (rate limit,
    timeout, or any failure).
    """

    def __init__(
        self,
        groq_api_key: str = "",
        gemini_api_key: str = "",
        groq_model: str = "llama-3.3-70b-versatile",
        gemini_model: str = "gemini-2.0-flash",
        timeout: int = 30,
    ):
        self._providers: list[BaseGenerator] = []

        if groq_api_key:
            self._providers.append(
                GroqGenerator(api_key=groq_api_key, model=groq_model, timeout=timeout)
            )
        if gemini_api_key:
            self._providers.append(
                GeminiGenerator(api_key=gemini_api_key, model=gemini_model, timeout=timeout)
            )

        if not self._providers:
            logger.warning("no_llm_providers_configured")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> GenerationResult:
        """
        Generate text, falling back through providers on failure.

        Returns the result from the first provider that succeeds.
        If all providers fail, returns the last error.
        """
        last_result: GenerationResult | None = None

        for i, provider in enumerate(self._providers):
            if not provider.is_available():
                continue

            result = provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if result.error:
                logger.warning(
                    "provider_failed",
                    provider=provider.provider_name,
                    error=result.error,
                    will_fallback=i < len(self._providers) - 1,
                )
                last_result = result
                continue

            if i > 0:
                result.fallback_used = True
                logger.info(
                    "fallback_used",
                    provider=provider.provider_name,
                    model=result.model,
                )

            return result

        # All providers failed
        if last_result:
            return last_result

        return GenerationResult(
            text="",
            error="No LLM providers available. Check API keys.",
        )

    @property
    def available_providers(self) -> list[str]:
        return [p.provider_name for p in self._providers if p.is_available()]
