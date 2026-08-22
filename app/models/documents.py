"""
Pydantic models for document processing.

These models preserve metadata through every stage of the pipeline,
ensuring traceability from final answer back to source chunk.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "md"


class ChunkMetadata(BaseModel):
    """Metadata preserved for every chunk through all pipeline stages."""
    document_id: str = Field(description="Unique ID for the source document")
    chunk_id: str = Field(description="Deterministic hash-based ID for this chunk")
    filename: str = Field(description="Original filename")
    source_path: str = Field(description="Path to the source file")
    page_number: int | None = Field(default=None, description="Page number if available")
    section: str | None = Field(default=None, description="Section heading if available")
    document_type: DocumentType = Field(description="Type of source document")
    ingestion_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of ingestion",
    )
    chunk_index: int = Field(default=0, description="Position of chunk within document")
    total_chunks: int = Field(default=0, description="Total chunks in the document")
    token_count: int = Field(default=0, description="Approximate token count of chunk")


class DocumentChunk(BaseModel):
    """A single chunk of text with full metadata."""
    text: str = Field(description="The chunk text content")
    metadata: ChunkMetadata = Field(description="Full metadata for traceability")
    embedding: list[float] | None = Field(default=None, exclude=True, description="Embedding vector")

    @property
    def chunk_id(self) -> str:
        return self.metadata.chunk_id

    @property
    def document_id(self) -> str:
        return self.metadata.document_id


class ParsedElement(BaseModel):
    """A single element extracted from a document by the parser."""
    text: str
    element_type: str = Field(default="NarrativeText", description="e.g., Title, NarrativeText, ListItem")
    page_number: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """Result of parsing a single document."""
    document_id: str
    filename: str
    source_path: str
    document_type: DocumentType
    elements: list[ParsedElement] = Field(default_factory=list)
    total_elements: int = 0
    parse_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def model_post_init(self, __context: Any) -> None:
        if self.total_elements == 0:
            self.total_elements = len(self.elements)


class IngestionResult(BaseModel):
    """Result of ingesting a document through the full pipeline."""
    document_id: str
    filename: str
    document_type: DocumentType
    total_chunks: int = 0
    total_elements: int = 0
    already_exists: bool = False
    success: bool = True
    error: str | None = None
    duration_seconds: float = 0.0
    ingestion_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
