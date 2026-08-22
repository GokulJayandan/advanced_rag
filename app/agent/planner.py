"""
Query planner — decides retrieval strategy, rewrites queries, decomposes complex queries.

Handles: simple passthrough, query rewriting for ambiguous/poor queries,
query decomposition for multi-part/multi-hop questions, and history-aware
rewriting for conversational context.
"""

from __future__ import annotations

import time

from app.generation.prompts import QUERY_REWRITING_PROMPT, QUERY_DECOMPOSITION_PROMPT
from app.generation.provider_manager import ProviderManager
from app.models.generation import QueryType
from app.observability.logging import get_logger

logger = get_logger(__name__)

# Query types that typically need decomposition
DECOMPOSE_TYPES = {QueryType.MULTI_HOP, QueryType.MULTI_PART, QueryType.COMPARISON}

# Query types that may benefit from rewriting
REWRITE_TYPES = {QueryType.AMBIGUOUS}


class QueryPlanner:
    """
    Plans the retrieval strategy based on query classification.

    Decides whether to:
    - Use the query as-is (simple factual)
    - Rewrite the query (ambiguous)
    - Decompose into sub-queries (multi-hop, multi-part)
    - Generate multiple retrieval queries (comparison)
    """

    def __init__(self, provider_manager: ProviderManager):
        self.provider_manager = provider_manager

    def plan(
        self,
        query: str,
        query_type: QueryType,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict:
        """
        Plan retrieval strategy for a query.

        Returns a dict with:
        - strategy: "simple" | "rewrite" | "decompose"
        - queries: list of queries to execute
        - reasoning: why this strategy was chosen
        """
        start = time.perf_counter()

        # History-aware rewriting if conversational context exists
        effective_query = query
        if conversation_history:
            effective_query = self._history_aware_rewrite(query, conversation_history)

        # Decide strategy based on query type
        if query_type in DECOMPOSE_TYPES:
            result = self._decompose(effective_query, query_type)
        elif query_type in REWRITE_TYPES:
            result = self._rewrite(effective_query)
        else:
            result = {
                "strategy": "simple",
                "queries": [effective_query],
                "reasoning": f"Query type '{query_type.value}' is straightforward — using direct retrieval",
            }

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "query_planned",
            strategy=result["strategy"],
            num_queries=len(result["queries"]),
            latency_ms=round(elapsed, 2),
        )

        return result

    def rewrite_for_retry(self, original_query: str, attempt_number: int) -> list[str]:
        """
        Generate rewritten queries for a retry attempt.

        Used when initial retrieval didn't find sufficient evidence.
        """
        prompt = QUERY_REWRITING_PROMPT.format(
            query=original_query,
            context=f"Previous retrieval attempt #{attempt_number} did not find sufficient evidence. Generate broader or alternative search queries.",
        )

        result = self.provider_manager.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=200,
        )

        if result.error:
            logger.warning("rewrite_for_retry_failed", error=result.error)
            return [original_query]

        queries = [q.strip() for q in result.text.strip().split("\n") if q.strip()]
        if not queries:
            return [original_query]

        logger.info("retry_queries_generated", count=len(queries))
        return queries[:3]  # Cap at 3

    def _rewrite(self, query: str) -> dict:
        """Rewrite an ambiguous query into better search queries."""
        prompt = QUERY_REWRITING_PROMPT.format(query=query, context="")

        result = self.provider_manager.generate(
            prompt=prompt,
            temperature=0.2,
            max_tokens=200,
        )

        if result.error:
            return {
                "strategy": "simple",
                "queries": [query],
                "reasoning": f"Query rewriting failed: {result.error}. Using original query.",
            }

        queries = [q.strip() for q in result.text.strip().split("\n") if q.strip()]
        if not queries:
            queries = [query]

        return {
            "strategy": "rewrite",
            "queries": queries[:3],
            "reasoning": "Query was ambiguous — generated improved search queries",
        }

    def _decompose(self, query: str, query_type: QueryType) -> dict:
        """Decompose a complex query into sub-questions."""
        prompt = QUERY_DECOMPOSITION_PROMPT.format(query=query)

        result = self.provider_manager.generate(
            prompt=prompt,
            temperature=0.1,
            max_tokens=300,
        )

        if result.error:
            return {
                "strategy": "simple",
                "queries": [query],
                "reasoning": f"Query decomposition failed: {result.error}. Using original query.",
            }

        sub_queries = [q.strip() for q in result.text.strip().split("\n") if q.strip()]
        if not sub_queries:
            sub_queries = [query]

        return {
            "strategy": "decompose",
            "queries": sub_queries[:4],
            "reasoning": f"Query type '{query_type.value}' requires decomposition into sub-questions",
        }

    def _history_aware_rewrite(
        self,
        query: str,
        history: list[dict[str, str]],
    ) -> str:
        """Rewrite query using conversation history for context."""
        if not history:
            return query

        # Only use last 3 turns to avoid context overflow
        recent = history[-3:]
        history_text = "\n".join(
            f"{turn.get('role', 'user')}: {turn.get('content', '')}"
            for turn in recent
        )

        prompt = (
            f"Given this conversation history:\n{history_text}\n\n"
            f"And this follow-up question: {query}\n\n"
            f"Rewrite the follow-up question as a standalone question that "
            f"captures the full intent. Return ONLY the rewritten question."
        )

        result = self.provider_manager.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=100,
        )

        if result.error or not result.text.strip():
            return query

        rewritten = result.text.strip()
        logger.info(
            "history_aware_rewrite",
            original=query[:80],
            rewritten=rewritten[:80],
        )
        return rewritten
