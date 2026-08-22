"""
Pydantic models for retrieval results.

Captures full diagnostics at every stage: dense scores, BM25 scores,
RRF fusion scores, and reranker scores for observability and evaluation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.documents import ChunkMetadata


class ScoredChunk(BaseModel):
    """A chunk with scores from various retrieval/ranking stages."""
    chunk_id: str
    text: str
    metadata: ChunkMetadata

    # Dense retrieval
    dense_rank: int | None = None
    dense_score: float | None = None

    # BM25
    bm25_rank: int | None = None
    bm25_score: float | None = None

    # RRF fusion
    rrf_score: float | None = None

    # Reranker
    reranker_score: float | None = None
    final_rank: int | None = None
    original_rank: int | None = None


class RetrievalDiagnostics(BaseModel):
    """Diagnostics for a single retrieval operation."""
    query: str
    dense_results_count: int = 0
    bm25_results_count: int = 0
    rrf_candidates_count: int = 0
    reranked_count: int = 0
    retrieval_latency_ms: float = 0.0
    reranking_latency_ms: float = 0.0
    strategy: str = "hybrid"


class HybridRetrievalResult(BaseModel):
    """Complete result from hybrid retrieval + reranking."""
    query: str
    chunks: list[ScoredChunk] = Field(default_factory=list)
    diagnostics: RetrievalDiagnostics = Field(default_factory=lambda: RetrievalDiagnostics(query=""))
