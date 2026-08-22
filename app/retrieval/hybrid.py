"""
Hybrid retrieval orchestrating dense + BM25 + RRF fusion.

Runs both retrieval methods in sequence and fuses with RRF.
"""

from __future__ import annotations

import time

from app.models.documents import ChunkMetadata
from app.models.retrieval import (
    HybridRetrievalResult,
    RetrievalDiagnostics,
    ScoredChunk,
)
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.rrf import ReciprocalRankFusion
from app.observability.logging import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining dense and sparse retrieval with RRF fusion.

    For every query:
    1. Dense retrieval → top_k candidates
    2. BM25 retrieval → top_k candidates
    3. RRF fusion → combined ranked list
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        rrf: ReciprocalRankFusion,
        dense_top_k: int = 50,
        bm25_top_k: int = 50,
        hybrid_candidates: int = 50,
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf = rrf
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.hybrid_candidates = hybrid_candidates

    def retrieve(
        self,
        query: str,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
        hybrid_candidates: int | None = None,
    ) -> HybridRetrievalResult:
        """
        Execute hybrid retrieval: dense + BM25 + RRF.

        Returns a HybridRetrievalResult with full diagnostics.
        """
        d_top_k = dense_top_k or self.dense_top_k
        b_top_k = bm25_top_k or self.bm25_top_k
        h_candidates = hybrid_candidates or self.hybrid_candidates

        start = time.perf_counter()

        # 1. Dense retrieval
        dense_results = self.dense_retriever.retrieve(query, top_k=d_top_k)

        # 2. BM25 retrieval
        bm25_raw = self.bm25_retriever.query(query, top_k=b_top_k)
        bm25_results: list[ScoredChunk] = []
        for rank, (chunk_id, text, meta_dict, score) in enumerate(bm25_raw):
            # Reconstruct metadata
            from app.storage.vector_store import _deserialize_metadata
            if isinstance(meta_dict, ChunkMetadata):
                metadata = meta_dict
            elif isinstance(meta_dict, dict):
                try:
                    metadata = _deserialize_metadata(meta_dict)
                except Exception:
                    from app.models.documents import DocumentType
                    metadata = ChunkMetadata(
                        document_id=meta_dict.get("document_id", ""),
                        chunk_id=chunk_id,
                        filename=meta_dict.get("filename", ""),
                        source_path=meta_dict.get("source_path", ""),
                        page_number=meta_dict.get("page_number"),
                        section=meta_dict.get("section"),
                        document_type=DocumentType(meta_dict.get("document_type", "txt")),
                    )
            else:
                continue

            bm25_results.append(ScoredChunk(
                chunk_id=chunk_id,
                text=text,
                metadata=metadata,
                bm25_rank=rank + 1,
                bm25_score=score,
            ))

        # 3. RRF fusion
        fused = self.rrf.fuse(dense_results, bm25_results, top_k=h_candidates)

        elapsed = (time.perf_counter() - start) * 1000

        diagnostics = RetrievalDiagnostics(
            query=query,
            dense_results_count=len(dense_results),
            bm25_results_count=len(bm25_results),
            rrf_candidates_count=len(fused),
            retrieval_latency_ms=round(elapsed, 2),
            strategy="hybrid",
        )

        logger.info(
            "hybrid_retrieval",
            query=query[:80],
            dense=len(dense_results),
            bm25=len(bm25_results),
            fused=len(fused),
            latency_ms=round(elapsed, 2),
        )

        return HybridRetrievalResult(
            query=query,
            chunks=fused,
            diagnostics=diagnostics,
        )

    def retrieve_dense_only(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        """Dense retrieval only (for ablation study)."""
        return self.dense_retriever.retrieve(query, top_k=top_k)

    def retrieve_bm25_only(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        """BM25 retrieval only (for ablation study)."""
        k = top_k or self.bm25_top_k
        bm25_raw = self.bm25_retriever.query(query, top_k=k)
        results = []
        for rank, (chunk_id, text, meta_dict, score) in enumerate(bm25_raw):
            from app.models.documents import DocumentType
            if isinstance(meta_dict, dict):
                metadata = ChunkMetadata(
                    document_id=meta_dict.get("document_id", ""),
                    chunk_id=chunk_id,
                    filename=meta_dict.get("filename", ""),
                    source_path=meta_dict.get("source_path", ""),
                    page_number=meta_dict.get("page_number"),
                    section=meta_dict.get("section"),
                    document_type=DocumentType(meta_dict.get("document_type", "txt")),
                )
            else:
                metadata = meta_dict
            results.append(ScoredChunk(
                chunk_id=chunk_id,
                text=text,
                metadata=metadata,
                bm25_rank=rank + 1,
                bm25_score=score,
            ))
        return results
