"""
Query analyzer — classifies queries to determine optimal retrieval strategy.

Avoids unnecessary expensive transformations for simple queries.
"""

from __future__ import annotations

import time

from app.generation.prompts import QUERY_CLASSIFICATION_PROMPT
from app.generation.provider_manager import ProviderManager
from app.models.generation import QueryType
from app.observability.logging import get_logger

logger = get_logger(__name__)

# Quick heuristic patterns to avoid LLM calls for obvious cases
_SIMPLE_FACTUAL_PATTERNS = [
    "what is", "what are", "who is", "who are", "when was", "when did",
    "where is", "where are", "define", "how many",
]
_COMPARISON_PATTERNS = [
    "compare", "difference between", "versus", " vs ", " vs. ",
    "how does * differ", "contrast",
]
_MULTI_PART_INDICATORS = ["and also", "additionally", "furthermore", "?.*?"]


class QueryAnalyzer:
    """
    Classifies queries to determine retrieval strategy.

    Uses fast heuristics first, then LLM classification for ambiguous cases.
    Simple queries skip expensive transformations.
    """

    def __init__(self, provider_manager: ProviderManager):
        self.provider_manager = provider_manager

    def classify(self, query: str) -> QueryType:
        """
        Classify a query into a QueryType.

        Uses heuristics first to avoid unnecessary LLM calls.
        Falls back to LLM classification for ambiguous queries.
        """
        start = time.perf_counter()

        # Try heuristic classification first
        heuristic_result = self._heuristic_classify(query)
        if heuristic_result is not None:
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(
                "query_classified",
                query_type=heuristic_result.value,
                method="heuristic",
                latency_ms=round(elapsed, 2),
            )
            return heuristic_result

        # Fall back to LLM classification
        return self._llm_classify(query)

    def _heuristic_classify(self, query: str) -> QueryType | None:
        """Fast heuristic classification — returns None if uncertain."""
        q = query.lower().strip()

        # Check for multi-part (multiple question marks)
        if q.count("?") > 1:
            return QueryType.MULTI_PART

        # Check for comparison
        for pattern in _COMPARISON_PATTERNS:
            if pattern in q:
                return QueryType.COMPARISON

        # Check for procedural
        if q.startswith(("how to ", "how do ", "how can ", "steps to ")):
            return QueryType.PROCEDURAL

        # Check for summarization
        if q.startswith(("summarize", "summary of", "give me an overview")):
            return QueryType.SUMMARIZATION

        # Check for simple factual
        for pattern in _SIMPLE_FACTUAL_PATTERNS:
            if q.startswith(pattern):
                return QueryType.FACTUAL

        return None  # Ambiguous — need LLM

    def _llm_classify(self, query: str) -> QueryType:
        """Classify using LLM."""
        start = time.perf_counter()

        prompt = QUERY_CLASSIFICATION_PROMPT.format(query=query)
        result = self.provider_manager.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=20,
        )

        elapsed = (time.perf_counter() - start) * 1000

        if result.error:
            logger.warning("query_classification_failed", error=result.error)
            return QueryType.FACTUAL  # Default fallback

        # Parse response
        response = result.text.strip().lower().replace(" ", "_")

        try:
            query_type = QueryType(response)
        except ValueError:
            # Try to match partial
            for qt in QueryType:
                if qt.value in response:
                    query_type = qt
                    break
            else:
                query_type = QueryType.FACTUAL

        logger.info(
            "query_classified",
            query_type=query_type.value,
            method="llm",
            latency_ms=round(elapsed, 2),
        )

        return query_type
