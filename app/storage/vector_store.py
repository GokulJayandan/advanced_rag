"""
Persistent ChromaDB vector store.

Independent from retrieval logic — provides a clean CRUD interface
for chunk storage. Uses deterministic chunk IDs to prevent duplicates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from app.models.documents import ChunkMetadata, DocumentChunk
from app.observability.logging import get_logger

logger = get_logger(__name__)

# ChromaDB metadata values must be str, int, float, or bool
def _serialize_metadata(meta: ChunkMetadata) -> dict[str, Any]:
    """Convert ChunkMetadata to a flat dict compatible with ChromaDB."""
    return {
        "document_id": meta.document_id,
        "chunk_id": meta.chunk_id,
        "filename": meta.filename,
        "source_path": meta.source_path,
        "page_number": meta.page_number if meta.page_number is not None else -1,
        "section": meta.section or "",
        "document_type": meta.document_type.value,
        "ingestion_timestamp": meta.ingestion_timestamp,
        "chunk_index": meta.chunk_index,
        "total_chunks": meta.total_chunks,
        "token_count": meta.token_count,
    }


def _deserialize_metadata(data: dict[str, Any]) -> ChunkMetadata:
    """Reconstruct ChunkMetadata from a ChromaDB metadata dict."""
    from app.models.documents import DocumentType

    page = data.get("page_number")
    if page == -1:
        page = None

    section = data.get("section")
    if section == "":
        section = None

    return ChunkMetadata(
        document_id=data["document_id"],
        chunk_id=data["chunk_id"],
        filename=data["filename"],
        source_path=data.get("source_path", ""),
        page_number=page,
        section=section,
        document_type=DocumentType(data["document_type"]),
        ingestion_timestamp=data.get("ingestion_timestamp", ""),
        chunk_index=data.get("chunk_index", 0),
        total_chunks=data.get("total_chunks", 0),
        token_count=data.get("token_count", 0),
    )


class ChromaVectorStore:
    """
    Persistent local ChromaDB vector store.

    Provides CRUD operations for chunks with embeddings.
    Uses deterministic chunk IDs to prevent duplicate indexing.
    """

    def __init__(
        self,
        persist_dir: str | Path = "./chroma_data",
        collection_name: str = "rag_chunks",
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "chroma_initialized",
            persist_dir=str(self.persist_dir),
            collection=collection_name,
            count=self._collection.count(),
        )

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> int:
        """
        Add chunks with embeddings. Skips duplicates by chunk ID.

        Returns the number of new chunks added.
        """
        if not chunks or not embeddings:
            return 0

        # Filter out existing chunk IDs
        ids = [c.metadata.chunk_id for c in chunks]
        existing = set()
        try:
            result = self._collection.get(ids=ids)
            if result and result["ids"]:
                existing = set(result["ids"])
        except Exception:
            pass

        new_ids = []
        new_embeddings = []
        new_documents = []
        new_metadatas = []

        for chunk, embedding in zip(chunks, embeddings):
            if chunk.metadata.chunk_id in existing:
                continue
            new_ids.append(chunk.metadata.chunk_id)
            new_embeddings.append(embedding)
            new_documents.append(chunk.text)
            new_metadatas.append(_serialize_metadata(chunk.metadata))

        if not new_ids:
            logger.info("no_new_chunks", total=len(chunks), duplicates=len(existing))
            return 0

        self._collection.add(
            ids=new_ids,
            embeddings=new_embeddings,
            documents=new_documents,
            metadatas=new_metadatas,
        )

        logger.info(
            "chunks_added",
            new=len(new_ids),
            duplicates=len(existing),
            total=self._collection.count(),
        )

        return len(new_ids)

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 50,
        where: dict | None = None,
    ) -> list[tuple[str, str, ChunkMetadata, float]]:
        """
        Query the vector store by embedding similarity.

        Returns list of (chunk_id, text, metadata, distance) tuples.
        """
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self._collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        output = []
        if results and results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                text = results["documents"][0][i]
                meta_dict = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                metadata = _deserialize_metadata(meta_dict)
                output.append((chunk_id, text, metadata, distance))

        return output

    def get_all_chunks(self) -> list[tuple[str, str, ChunkMetadata]]:
        """Get all chunks in the store (for BM25 index building)."""
        count = self._collection.count()
        if count == 0:
            return []

        result = self._collection.get(
            include=["documents", "metadatas"],
            limit=count,
        )

        output = []
        for i, chunk_id in enumerate(result["ids"]):
            text = result["documents"][i]
            meta = _deserialize_metadata(result["metadatas"][i])
            output.append((chunk_id, text, meta))

        return output

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document. Returns count deleted."""
        # Get chunks for this document
        result = self._collection.get(
            where={"document_id": document_id},
            include=[],
        )

        if not result or not result["ids"]:
            return 0

        ids_to_delete = result["ids"]
        self._collection.delete(ids=ids_to_delete)

        logger.info(
            "document_deleted",
            document_id=document_id,
            chunks_deleted=len(ids_to_delete),
        )

        return len(ids_to_delete)

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Delete specific chunks by ID."""
        if chunk_ids:
            self._collection.delete(ids=chunk_ids)

    @property
    def count(self) -> int:
        return self._collection.count()

    @property
    def stats(self) -> dict[str, Any]:
        """Collection statistics."""
        count = self._collection.count()
        # Get unique document count
        doc_ids = set()
        if count > 0:
            result = self._collection.get(include=["metadatas"], limit=count)
            for meta in result["metadatas"]:
                doc_ids.add(meta.get("document_id", ""))

        return {
            "total_chunks": count,
            "total_documents": len(doc_ids),
            "persist_dir": str(self.persist_dir),
        }
