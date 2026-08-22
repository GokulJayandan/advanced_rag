"""
Shared Pytest Fixtures and Mocks.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from app.config.settings import Settings
from app.models.documents import ChunkMetadata, DocumentChunk, DocumentType, ParsedDocument, ParsedElement
from app.models.generation import GenerationResult, ProviderType
from app.models.retrieval import ScoredChunk


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def mock_settings(temp_dir):
    return Settings(
        google_api_key="test-google-key",
        groq_api_key="test-groq-key",
        chroma_persist_dir=str(temp_dir / "chroma"),
        bm25_index_dir=str(temp_dir / "bm25"),
        embedding_cache_dir=str(temp_dir / "cache"),
        upload_dir=str(temp_dir / "uploads"),
    )


@pytest.fixture
def sample_parsed_document():
    elements = [
        ParsedElement(text="System Architecture", element_type="Title", page_number=1, section="System Architecture"),
        ParsedElement(text="The system uses a hybrid retrieval pipeline combining dense embeddings and BM25.", element_type="NarrativeText", page_number=1, section="System Architecture"),
        ParsedElement(text="Reranking Stage", element_type="Title", page_number=2, section="Reranking Stage"),
        ParsedElement(text="Candidates are reranked using BGE-Reranker-v2-m3 on CPU.", element_type="NarrativeText", page_number=2, section="Reranking Stage"),
    ]
    return ParsedDocument(
        document_id="doc-12345",
        filename="test_doc.md",
        source_path="/tmp/test_doc.md",
        document_type=DocumentType.MARKDOWN,
        elements=elements,
    )


@pytest.fixture
def sample_chunks():
    chunks = []
    for i in range(5):
        meta = ChunkMetadata(
            document_id="doc-1",
            chunk_id=f"chunk-{i}",
            filename="sample_guide.pdf",
            source_path="/docs/sample_guide.pdf",
            page_number=i + 1,
            section=f"Section {i+1}",
            document_type=DocumentType.PDF,
            chunk_index=i,
            total_chunks=5,
            token_count=100,
        )
        chunks.append(DocumentChunk(text=f"This is the content of chunk {i} discussing topic {i}.", metadata=meta))
    return chunks


@pytest.fixture
def sample_scored_chunks(sample_chunks):
    scored = []
    for i, c in enumerate(sample_chunks):
        sc = ScoredChunk(
            chunk_id=c.chunk_id,
            text=c.text,
            metadata=c.metadata,
            dense_rank=i + 1,
            dense_score=0.9 - (i * 0.1),
            bm25_rank=5 - i,
            bm25_score=10.0 - i,
            rrf_score=0.03 - (i * 0.005),
            reranker_score=0.85 - (i * 0.1),
            final_rank=i + 1,
            original_rank=i + 1,
        )
        scored.append(sc)
    return scored


@pytest.fixture
def mock_provider_manager():
    mgr = MagicMock()
    mgr.generate.return_value = GenerationResult(
        text="The system uses a hybrid retrieval pipeline [Source: test_doc.md, p.1].",
        provider=ProviderType.GROQ,
        model="llama-3.3-70b-versatile",
        latency_ms=120.0,
    )
    mgr.available_providers = ["groq", "gemini"]
    return mgr
