"""
Agent state — tracks the full lifecycle of a query through the RAG pipeline.

Maintains the complete execution trace including all decisions, retrieval
attempts, scores, and transformations for observability and debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.models.generation import (
    AgentDecision,
    Citation,
    CitationValidationResult,
    EvidenceQuality,
    QueryType,
    VerificationResult,
)
from app.models.retrieval import ScoredChunk, HybridRetrievalResult


@dataclass
class RetrievalAttempt:
    """Record of a single retrieval attempt."""
    attempt_number: int = 0
    query_used: str = ""
    strategy: str = "hybrid"
    dense_count: int = 0
    bm25_count: int = 0
    fused_count: int = 0
    reranked_count: int = 0
    top_reranker_score: float = 0.0
    evidence_sufficient: bool = False


@dataclass
class AgentState:
    """
    Complete state of the RAG agent for a single query.

    Tracks every transformation, decision, and retrieval attempt
    so the full execution can be audited and replayed.
    """

    # Query identity
    query_id: str = ""
    original_query: str = ""
    query_type: QueryType = QueryType.FACTUAL

    # Query transformations
    rewritten_queries: list[str] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    active_query: str = ""  # The current query being used for retrieval

    # Retrieval state
    retrieval_attempts: list[RetrievalAttempt] = field(default_factory=list)
    all_retrieved_chunks: list[ScoredChunk] = field(default_factory=list)
    reranked_chunks: list[ScoredChunk] = field(default_factory=list)
    context_chunks: list[ScoredChunk] = field(default_factory=list)

    # Evidence assessment
    evidence_quality: EvidenceQuality = field(default_factory=EvidenceQuality)
    evidence_sufficient: bool = False

    # Generation
    generated_answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    citation_validation: CitationValidationResult | None = None
    verification: VerificationResult | None = None

    # Agent decisions (audit trail)
    decisions: list[AgentDecision] = field(default_factory=list)

    # Conversation context
    conversation_history: list[dict[str, str]] = field(default_factory=list)

    # Error state
    error: str | None = None

    def add_decision(self, step: str, decision: str, reasoning: str) -> None:
        """Record an agent decision."""
        import time
        self.decisions.append(AgentDecision(
            step=step,
            decision=decision,
            reasoning=reasoning,
            timestamp_ms=time.perf_counter() * 1000,
        ))

    def add_retrieval_attempt(self, attempt: RetrievalAttempt) -> None:
        """Record a retrieval attempt."""
        self.retrieval_attempts.append(attempt)

    @property
    def num_attempts(self) -> int:
        return len(self.retrieval_attempts)
