"""
Abstract base for LLM generation providers.

All providers implement this interface so they can be swapped transparently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.generation import GenerationResult


class BaseGenerator(ABC):
    """Abstract interface for LLM generation providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> GenerationResult:
        """
        Generate text from a prompt.

        Args:
            prompt: The user/context prompt.
            system_prompt: System instructions for grounding.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            GenerationResult with text, provider info, and metrics.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is currently available."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier."""
        ...
