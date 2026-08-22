"""Unit tests for semantic-aware chunker."""

from app.ingestion.chunker import SemanticChunker
from app.models.documents import DocumentType, ParsedDocument, ParsedElement


def test_semantic_chunker_basic(sample_parsed_document):
    chunker = SemanticChunker(chunk_size_tokens=100, chunk_overlap_tokens=15)
    chunks = chunker.chunk(sample_parsed_document)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.metadata.document_id == "doc-12345"
        assert chunk.metadata.filename == "test_doc.md"
        assert chunk.metadata.total_chunks == len(chunks)
        assert chunk.metadata.token_count > 0
        assert chunk.chunk_id is not None


def test_semantic_chunker_empty_document():
    chunker = SemanticChunker()
    doc = ParsedDocument(
        document_id="empty-1",
        filename="empty.txt",
        source_path="/empty.txt",
        document_type=DocumentType.TXT,
        elements=[],
    )
    chunks = chunker.chunk(doc)
    assert chunks == []
