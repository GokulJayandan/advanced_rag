"""
Ingestion pipeline — orchestrates: parse → chunk → embed → store.

Handles duplicate detection, BM25 index synchronization, and progress tracking.
"""

from __future__ import annotations

import time
from pathlib import Path

from app.config.settings import Settings
from app.embeddings.embedder import GeminiEmbedder
from app.ingestion.chunker import SemanticChunker
from app.ingestion.parser import DocumentParser
from app.models.documents import DocumentChunk, IngestionResult
from app.retrieval.bm25 import BM25Retriever
from app.storage.vector_store import ChromaVectorStore, _serialize_metadata
from app.observability.logging import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """
    End-to-end document ingestion: parse → chunk → embed → store.

    Also keeps the BM25 index synchronized with the vector store.
    """

    def __init__(
        self,
        parser: DocumentParser,
        chunker: SemanticChunker,
        embedder: GeminiEmbedder,
        vector_store: ChromaVectorStore,
        bm25_retriever: BM25Retriever,
    ):
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever

    def ingest(self, filepath: str | Path) -> IngestionResult:
        """
        Ingest a single document through the full pipeline.

        Args:
            filepath: Path to the document file.

        Returns:
            IngestionResult with status and metrics.
        """
        filepath = Path(filepath)
        start = time.perf_counter()

        try:
            # Step 1: Parse
            parsed = self.parser.parse(filepath)

            logger.info(
                "ingestion_started",
                filename=parsed.filename,
                document_id=parsed.document_id,
                elements=parsed.total_elements,
            )

            # Step 2: Chunk
            chunks = self.chunker.chunk(parsed)

            if not chunks:
                return IngestionResult(
                    document_id=parsed.document_id,
                    filename=parsed.filename,
                    document_type=parsed.document_type,
                    success=True,
                    error="No chunks produced (document may be empty)",
                    duration_seconds=time.perf_counter() - start,
                )

            # Step 3: Embed
            texts = [c.text for c in chunks]
            embeddings = self.embedder.embed_texts(texts)

            # Step 4: Store in ChromaDB
            added = self.vector_store.add_chunks(chunks, embeddings)

            # Step 5: Update BM25 index
            chunk_ids = [c.metadata.chunk_id for c in chunks]
            metadatas = [_serialize_metadata(c.metadata) for c in chunks]
            self.bm25_retriever.add_chunks(chunk_ids, texts, metadatas)

            duration = time.perf_counter() - start

            result = IngestionResult(
                document_id=parsed.document_id,
                filename=parsed.filename,
                document_type=parsed.document_type,
                total_chunks=len(chunks),
                total_elements=parsed.total_elements,
                already_exists=added == 0,
                success=True,
                duration_seconds=round(duration, 3),
            )

            logger.info(
                "ingestion_complete",
                document_id=parsed.document_id,
                filename=parsed.filename,
                chunks=len(chunks),
                new_chunks=added,
                duration_s=round(duration, 3),
            )

            return result

        except FileNotFoundError as e:
            return IngestionResult(
                document_id="",
                filename=filepath.name,
                document_type=None,
                success=False,
                error=str(e),
                duration_seconds=time.perf_counter() - start,
            )
        except ValueError as e:
            return IngestionResult(
                document_id="",
                filename=filepath.name,
                document_type=None,
                success=False,
                error=str(e),
                duration_seconds=time.perf_counter() - start,
            )
        except Exception as e:
            logger.error("ingestion_failed", filename=filepath.name, error=str(e))
            return IngestionResult(
                document_id="",
                filename=filepath.name,
                document_type=None,
                success=False,
                error=f"Ingestion failed: {str(e)}",
                duration_seconds=time.perf_counter() - start,
            )

    def delete_document(self, document_id: str) -> dict:
        """Delete a document and its chunks from all stores."""
        chroma_deleted = self.vector_store.delete_document(document_id)
        bm25_deleted = self.bm25_retriever.delete_document(document_id)

        logger.info(
            "document_deleted",
            document_id=document_id,
            chroma_deleted=chroma_deleted,
            bm25_deleted=bm25_deleted,
        )

        return {
            "document_id": document_id,
            "chroma_chunks_deleted": chroma_deleted,
            "bm25_chunks_deleted": bm25_deleted,
        }
