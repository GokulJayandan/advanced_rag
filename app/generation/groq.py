"""Groq implementation of the generation provider interface."""
from __future__ import annotations
import time
from app.generation.base import GenerationProvider
from app.models.generation import GenerationResult, ProviderType

class GroqProvider(GenerationProvider):
    def __init__(self, api_key: str, model: str, temperature: float = 0.1, max_tokens: int = 2048):
        if not api_key:
            raise ValueError("GROQ_API_KEY is required for Groq generation")
        from groq import Groq
        self.client, self.model = Groq(api_key=api_key), model
        self.temperature, self.max_tokens = temperature, max_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        started = time.perf_counter()
        response = self.client.chat.completions.create(model=self.model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=self.temperature, max_tokens=self.max_tokens)
        usage = response.usage
        return GenerationResult(text=response.choices[0].message.content or "", provider=ProviderType.GROQ,
            model=self.model, latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=getattr(usage, "prompt_tokens", None), completion_tokens=getattr(usage, "completion_tokens", None))
