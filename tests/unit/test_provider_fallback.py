"""Unit tests for ProviderManager and Fallback mechanism."""

from unittest.mock import MagicMock
from app.generation.provider_manager import ProviderManager
from app.models.generation import GenerationResult, ProviderType


def test_provider_manager_groq_success():
    mgr = ProviderManager(groq_api_key="valid-groq", gemini_api_key="valid-gemini")
    mgr._providers[0].generate = MagicMock(return_value=GenerationResult(
        text="Groq answer", provider=ProviderType.GROQ, model="llama-3.3-70b-versatile"
    ))
    res = mgr.generate("test prompt")
    assert res.text == "Groq answer"
    assert res.provider == ProviderType.GROQ
    assert res.fallback_used is False


def test_provider_manager_fallback_to_gemini():
    mgr = ProviderManager(groq_api_key="valid-groq", gemini_api_key="valid-gemini")
    # Simulate Groq rate limit / failure
    mgr._providers[0].generate = MagicMock(return_value=GenerationResult(
        text="", provider=ProviderType.GROQ, error="Rate limit reached"
    ))
    # Simulate Gemini fallback success
    mgr._providers[1].generate = MagicMock(return_value=GenerationResult(
        text="Gemini answer", provider=ProviderType.GEMINI, model="gemini-2.0-flash"
    ))

    res = mgr.generate("test prompt")
    assert res.text == "Gemini answer"
    assert res.provider == ProviderType.GEMINI
    assert res.fallback_used is True
