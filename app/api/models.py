"""
Pydantic request/response models for the FastAPI endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    document_id: str = ""
    filename: str = ""
    document_type: str | None = None
    total_chunks: int = 0
    total_elements: int = 0
    already_exists: bool = False
    success: bool = True
    error: str | None = None
    duration_seconds: float = 0.0


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    include_diagnostics: bool = False


class CitationResponse(BaseModel):
    citation_text: str = ""
    filename: str = ""
    page_number: int | None = None
    section: str | None = None
    is_valid: bool = True


class EvidenceQualityResponse(BaseModel):
    score: float = 0.0
    num_relevant_chunks: int = 0
    is_sufficient: bool = False
    explanation: str = ""


class QueryResponse(BaseModel):
    query_id: str = ""
    query: str = ""
    answer: str = ""
    citations: list[CitationResponse] = Field(default_factory=list)
    evidence_quality: EvidenceQualityResponse = Field(default_factory=EvidenceQualityResponse)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    query_type: str = ""
    total_latency_ms: float = 0.0
    retrieval_iterations: int = 0
    agent_decisions: list[dict[str, str]] | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    providers: list[str] = Field(default_factory=list)


class StatsResponse(BaseModel):
    total_chunks: int = 0
    total_documents: int = 0
    bm25_indexed: int = 0
    embedding_cache: dict[str, Any] = Field(default_factory=dict)


class DeleteResponse(BaseModel):
    document_id: str = ""
    chroma_chunks_deleted: int = 0
    bm25_chunks_deleted: int = 0
