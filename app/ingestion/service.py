"""End-to-end ingestion orchestration kept separate from HTTP concerns."""

from __future__ import annotations

import time
from pathlib import Path

from app.embeddings.embedder import GeminiEmbedder
from app.ingestion.chunker import SemanticChunker
from app.ingestion.parser import DocumentParser
from app.models.documents import IngestionResult
from app.observability.logging import get_logger
from app.retrieval.bm25 import BM25Retriever
from app.storage.vector_store import ChromaVectorStore

logger = get_logger(__name__)


class IngestionService:
    """Parse, chunk, embed and persist a document while keeping indexes aligned."""

    def __init__(self, parser: DocumentParser, chunker: SemanticChunker,
                 embedder: GeminiEmbedder, vector_store: ChromaVectorStore,
                 bm25: BM25Retriever):
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = bm25

    def ingest(self, filepath: str | Path) -> IngestionResult:
        started = time.perf_counter()
        parsed = self.parser.parse(filepath)
        chunks = self.chunker.chunk(parsed)
        if not chunks:
            return IngestionResult(document_id=parsed.document_id, filename=parsed.filename,
                                   document_type=parsed.document_type, success=False,
                                   error="No readable text was extracted from the document.")

        try:
            embeddings = self.embedder.embed_texts([chunk.text for chunk in chunks])
            added = self.vector_store.add_chunks(chunks, embeddings)
            # A rebuild makes this operation idempotent and repairs an interrupted prior run.
            self.bm25.build_from_store(self.vector_store.get_all_chunks())
        except Exception as exc:
            logger.exception("ingestion_failed", document_id=parsed.document_id, error=str(exc))
            return IngestionResult(document_id=parsed.document_id, filename=parsed.filename,
                                   document_type=parsed.document_type, total_elements=parsed.total_elements,
                                   success=False, error=str(exc),
                                   duration_seconds=time.perf_counter() - started)

        return IngestionResult(document_id=parsed.document_id, filename=parsed.filename,
                               document_type=parsed.document_type, total_elements=parsed.total_elements,
                               total_chunks=len(chunks), already_exists=added == 0,
                               duration_seconds=time.perf_counter() - started)

    def delete_document(self, document_id: str) -> int:
        deleted = self.vector_store.delete_document(document_id)
        self.bm25.delete_document(document_id)
        return deleted
