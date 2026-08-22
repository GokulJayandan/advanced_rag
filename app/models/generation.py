"""
Pydantic models for generation, citations, and verification.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ProviderType(str, Enum):
    """LLM provider types."""
    GROQ = "groq"
    GEMINI = "gemini"


class GenerationResult(BaseModel):
    """Result of LLM generation."""
    text: str = ""
    provider: ProviderType = ProviderType.GROQ
    model: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    fallback_used: bool = False
    error: str | None = None


class Citation(BaseModel):
    """A single citation extracted from the generated answer."""
    citation_text: str = Field(description="The citation as it appears in text")
    filename: str = ""
    page_number: int | None = None
    section: str | None = None
    chunk_id: str | None = None
    is_valid: bool = True
    validation_error: str | None = None


class CitationValidationResult(BaseModel):
    """Result of validating all citations in an answer."""
    total_citations: int = 0
    valid_citations: int = 0
    invalid_citations: int = 0
    citations: list[Citation] = Field(default_factory=list)
    all_valid: bool = True


class VerificationResult(BaseModel):
    """Result of verifying answer against retrieved evidence."""
    is_grounded: bool = True
    evidence_coverage: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of claims supported")
    unsupported_claims: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    verification_details: str = ""


class EvidenceQuality(BaseModel):
    """Composite evidence quality signal (not a calibrated probability)."""
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Evidence quality 0-1")
    num_relevant_chunks: int = 0
    avg_reranker_score: float = 0.0
    evidence_coverage: float = 0.0
    citation_coverage: float = 0.0
    is_sufficient: bool = False
    explanation: str = ""


class QueryType(str, Enum):
    """Query classification types."""
    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    COMPARISON = "comparison"
    MULTI_HOP = "multi_hop"
    MULTI_PART = "multi_part"
    AMBIGUOUS = "ambiguous"
    SUMMARIZATION = "summarization"
    PROCEDURAL = "procedural"


class AgentDecision(BaseModel):
    """A single decision made by the RAG agent."""
    step: str
    decision: str
    reasoning: str
    timestamp_ms: float = 0.0


class QueryResult(BaseModel):
    """Final result returned to the user."""
    query_id: str
    query: str
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    evidence_quality: EvidenceQuality = Field(default_factory=EvidenceQuality)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    query_type: QueryType = QueryType.FACTUAL
    verification: VerificationResult | None = None
    agent_decisions: list[AgentDecision] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    retrieval_iterations: int = 0
    error: str | None = None
