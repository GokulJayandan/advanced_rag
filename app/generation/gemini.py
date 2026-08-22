"""Gemini fallback implementation of the generation provider interface."""
from __future__ import annotations
import time
from app.generation.base import GenerationProvider
from app.models.generation import GenerationResult, ProviderType

class GeminiProvider(GenerationProvider):
    def __init__(self, api_key: str, model: str, temperature: float = 0.1, max_tokens: int = 2048):
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for Gemini generation")
        from google import genai
        self.client, self.model = genai.Client(api_key=api_key), model
        self.temperature, self.max_tokens = temperature, max_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        from google.genai import types
        started = time.perf_counter()
        response = self.client.models.generate_content(model=self.model, contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=self.temperature,
                                               max_output_tokens=self.max_tokens))
        return GenerationResult(text=response.text or "", provider=ProviderType.GEMINI, model=self.model,
                                latency_ms=(time.perf_counter() - started) * 1000)
