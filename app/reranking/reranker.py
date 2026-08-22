"""
BGE Reranker for neural reranking of hybrid retrieval candidates.

Uses BGE reranker locally on CPU with graceful fallback to RRF rankings
if the model is unavailable or takes too long to load.
"""

from __future__ import annotations

import time
from typing import Sequence

from app.models.retrieval import ScoredChunk
from app.observability.logging import get_logger

logger = get_logger(__name__)


class BGEReranker:
    """
    Neural reranker using BGE-Reranker.

    Reranks query-chunk pairs with robust fallback to RRF order
    if model loading or execution encounters any issues.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        top_k: int = 10,
        threshold: float = 0.0,
    ):
        self.model_name = model_name
        self.device = device
        self.top_k = top_k
        self.threshold = threshold
        self._reranker = None
        self._load_failed = False

    def _load_model(self) -> None:
        """Lazy-load the reranker model safely."""
        if self.model_name.lower() in ("none", "rrf", "off", ""):
            return

        if self._reranker is not None or self._load_failed:
            return

        logger.info("loading_reranker", model=self.model_name, device=self.device)
        start = time.perf_counter()

        try:
            from FlagEmbedding import FlagReranker

            self._reranker = FlagReranker(
                self.model_name,
                device=self.device,
                use_fp16=False,
            )
            elapsed = time.perf_counter() - start
            logger.info("reranker_loaded", model=self.model_name, load_time_s=round(elapsed, 2))
        except Exception as exc:
            self._load_failed = True
            logger.warning("reranker_load_failed_using_fallback", error=str(exc))

    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """
        Rerank chunks using the BGE reranker. Falls back to RRF ranking if model unavailable.

        Args:
            query: The search query.
            chunks: Candidate chunks from hybrid retrieval.
            top_k: Number of top results to return after reranking.

        Returns:
            Reranked list of ScoredChunks with reranker_score and final_rank set.
        """
        if not chunks:
            return []

        k = top_k or self.top_k
        self._load_model()

        # Fallback: if reranker isn't loaded, return top-k by existing RRF score
        if self._reranker is None:
            logger.info("rerank_using_rrf_fallback", candidates=len(chunks), top_k=k)
            fallback_chunks = []
            for rank, chunk in enumerate(chunks[:k]):
                c = chunk.model_copy()
                c.reranker_score = c.rrf_score or 1.0 / (rank + 1)
                c.final_rank = rank + 1
                c.original_rank = rank + 1
                fallback_chunks.append(c)
            return fallback_chunks

        start = time.perf_counter()
        try:
            # Prepare query-document pairs
            pairs = [[query, chunk.text] for chunk in chunks]

            # Compute reranker scores
            scores = self._reranker.compute_score(pairs, normalize=True)

            if isinstance(scores, (int, float)):
                scores = [scores]

            # Assign scores and track original rank
            scored_chunks = []
            for i, (chunk, score) in enumerate(zip(chunks, scores)):
                reranked = chunk.model_copy()
                reranked.reranker_score = float(score)
                reranked.original_rank = i + 1
                scored_chunks.append(reranked)

            # Sort by reranker score descending
            scored_chunks.sort(key=lambda c: c.reranker_score or 0.0, reverse=True)

            # Apply threshold and top-k
            filtered = [
                c for c in scored_chunks
                if (c.reranker_score or 0.0) >= self.threshold
            ][:k]

            # Assign final ranks
            for rank, chunk in enumerate(filtered):
                chunk.final_rank = rank + 1

            elapsed = (time.perf_counter() - start) * 1000
            logger.info(
                "reranking_complete",
                query=query[:80],
                candidates=len(chunks),
                reranked=len(filtered),
                top_k=k,
                latency_ms=round(elapsed, 2),
            )
            return filtered

        except Exception as exc:
            logger.warning("reranking_computation_failed_using_fallback", error=str(exc))
            fallback_chunks = []
            for rank, chunk in enumerate(chunks[:k]):
                c = chunk.model_copy()
                c.reranker_score = c.rrf_score or 1.0 / (rank + 1)
                c.final_rank = rank + 1
                c.original_rank = rank + 1
                fallback_chunks.append(c)
            return fallback_chunks

    @property
    def is_loaded(self) -> bool:
        return self._reranker is not None
