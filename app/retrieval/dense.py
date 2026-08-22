"""
Dense retrieval using ChromaDB vector similarity search.

Embeds the query via GeminiEmbedder and queries the vector store.
"""

from __future__ import annotations

import time

from app.embeddings.embedder import GeminiEmbedder
from app.models.documents import ChunkMetadata
from app.models.retrieval import ScoredChunk
from app.storage.vector_store import ChromaVectorStore
from app.observability.logging import get_logger

logger = get_logger(__name__)


class DenseRetriever:
    """
    Dense retrieval via embedding similarity search.

    Embeds the query using Gemini and queries ChromaDB for nearest neighbors.
    """

    def __init__(
        self,
        embedder: GeminiEmbedder,
        vector_store: ChromaVectorStore,
        top_k: int = 50,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """
        Retrieve chunks by dense embedding similarity.

        Args:
            query: The search query.
            top_k: Override for number of results.

        Returns:
            List of ScoredChunks with dense_rank and dense_score.
        """
        k = top_k or self.top_k
        start = time.perf_counter()

        # Embed query
        query_embedding = self.embedder.embed_query(query)

        # Query vector store
        results = self.vector_store.query(query_embedding, top_k=k)

        chunks = []
        for rank, (chunk_id, text, metadata, distance) in enumerate(results):
            # ChromaDB returns cosine distance; convert to similarity
            similarity = 1.0 - distance

            chunks.append(ScoredChunk(
                chunk_id=chunk_id,
                text=text,
                metadata=metadata,
                dense_rank=rank + 1,
                dense_score=similarity,
            ))

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "dense_retrieval",
            query=query[:80],
            results=len(chunks),
            top_k=k,
            latency_ms=round(elapsed, 2),
        )

        return chunks
