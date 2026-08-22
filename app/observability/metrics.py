"""
Per-query metrics collection for observability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class QueryMetrics:
    """Timing and diagnostic metrics for a single query."""
    query_id: str = ""
    total_start: float = 0.0

    # Latencies in milliseconds
    query_analysis_ms: float = 0.0
    query_planning_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    generation_ms: float = 0.0
    citation_validation_ms: float = 0.0
    answer_verification_ms: float = 0.0
    total_ms: float = 0.0

    # Counts
    retrieval_iterations: int = 0
    dense_results: int = 0
    bm25_results: int = 0
    rrf_candidates: int = 0
    reranked_results: int = 0
    api_calls: int = 0
    cache_hits: int = 0

    # Provider
    generation_provider: str = ""
    fallback_used: bool = False

    def start_timer(self) -> None:
        self.total_start = time.perf_counter()

    def finalize(self) -> None:
        if self.total_start > 0:
            self.total_ms = (time.perf_counter() - self.total_start) * 1000

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "total_ms": round(self.total_ms, 2),
            "query_analysis_ms": round(self.query_analysis_ms, 2),
            "query_planning_ms": round(self.query_planning_ms, 2),
            "retrieval_ms": round(self.retrieval_ms, 2),
            "reranking_ms": round(self.reranking_ms, 2),
            "generation_ms": round(self.generation_ms, 2),
            "citation_validation_ms": round(self.citation_validation_ms, 2),
            "answer_verification_ms": round(self.answer_verification_ms, 2),
            "retrieval_iterations": self.retrieval_iterations,
            "dense_results": self.dense_results,
            "bm25_results": self.bm25_results,
            "rrf_candidates": self.rrf_candidates,
            "reranked_results": self.reranked_results,
            "api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
            "generation_provider": self.generation_provider,
            "fallback_used": self.fallback_used,
        }
