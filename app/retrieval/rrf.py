"""
Reciprocal Rank Fusion (RRF) for combining dense and sparse rankings.

RRF formula: score(d) = Σ 1 / (k + rank_i(d))

where k is a constant (default 60) and rank_i is the rank of document d
in ranking i.
"""

from __future__ import annotations

from app.models.retrieval import ScoredChunk
from app.observability.logging import get_logger

logger = get_logger(__name__)


class ReciprocalRankFusion:
    """
    Combines multiple ranked lists using Reciprocal Rank Fusion.

    This is the standard RRF algorithm that properly fuses rankings
    rather than simply concatenating or averaging scores.
    """

    def __init__(self, k: int = 60):
        """
        Args:
            k: The RRF constant. Higher k reduces the influence of
               high-ranked documents. Default 60 per the original paper.
        """
        self.k = k

    def fuse(
        self,
        dense_results: list[ScoredChunk],
        bm25_results: list[ScoredChunk],
        top_k: int = 50,
    ) -> list[ScoredChunk]:
        """
        Fuse dense and BM25 rankings using RRF.

        Args:
            dense_results: Chunks from dense retrieval with dense_rank set.
            bm25_results: Chunks from BM25 with bm25_rank set.
            top_k: Number of fused results to return.

        Returns:
            List of ScoredChunks with rrf_score set, sorted by RRF score descending.
        """
        # Build a map: chunk_id → ScoredChunk (merge scores)
        chunk_map: dict[str, ScoredChunk] = {}

        for chunk in dense_results:
            chunk_map[chunk.chunk_id] = chunk.model_copy()

        for chunk in bm25_results:
            if chunk.chunk_id in chunk_map:
                # Merge BM25 scores into existing entry
                existing = chunk_map[chunk.chunk_id]
                existing.bm25_rank = chunk.bm25_rank
                existing.bm25_score = chunk.bm25_score
            else:
                chunk_map[chunk.chunk_id] = chunk.model_copy()

        # Compute RRF scores
        for chunk_id, chunk in chunk_map.items():
            rrf_score = 0.0

            if chunk.dense_rank is not None:
                rrf_score += 1.0 / (self.k + chunk.dense_rank)

            if chunk.bm25_rank is not None:
                rrf_score += 1.0 / (self.k + chunk.bm25_rank)

            chunk.rrf_score = rrf_score

        # Sort by RRF score descending and take top_k
        fused = sorted(
            chunk_map.values(),
            key=lambda c: c.rrf_score or 0.0,
            reverse=True,
        )[:top_k]

        logger.info(
            "rrf_fusion",
            dense_count=len(dense_results),
            bm25_count=len(bm25_results),
            unique_candidates=len(chunk_map),
            fused_count=len(fused),
            rrf_k=self.k,
        )

        return fused
