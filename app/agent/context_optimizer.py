"""
Context optimizer — prepares retrieved chunks for LLM generation.

Removes duplicates, prioritizes high-score chunks, preserves source
diversity, respects context limits, and orders logically.
"""

from __future__ import annotations

from collections import defaultdict

import tiktoken

from app.models.retrieval import ScoredChunk
from app.observability.logging import get_logger

logger = get_logger(__name__)


class ContextOptimizer:
    """
    Optimizes retrieved context before sending to the LLM.

    Applies: deduplication, near-duplicate removal, score-based
    prioritization, source diversity, and context window management.
    """

    def __init__(
        self,
        max_context_tokens: int = 6000,
        max_chunks: int = 10,
        diversity_factor: float = 0.3,
    ):
        self.max_context_tokens = max_context_tokens
        self.max_chunks = max_chunks
        self.diversity_factor = diversity_factor
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def optimize(self, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        """
        Optimize chunks for LLM context.

        Steps:
        1. Deduplicate by chunk_id
        2. Remove near-duplicate texts
        3. Prioritize by reranker score
        4. Ensure source diversity
        5. Fit within token budget
        6. Order logically (by document, then position)
        """
        if not chunks:
            return []

        # 1. Deduplicate by chunk_id
        seen_ids = set()
        deduped = []
        for chunk in chunks:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                deduped.append(chunk)

        # 2. Remove near-duplicate texts (Jaccard similarity > 0.8)
        deduped = self._remove_near_duplicates(deduped)

        # 3. Sort by best score (reranker > rrf > dense)
        deduped.sort(
            key=lambda c: (c.reranker_score or c.rrf_score or c.dense_score or 0.0),
            reverse=True,
        )

        # 4. Source diversity — ensure we don't over-represent one document
        selected = self._apply_diversity(deduped)

        # 5. Fit within token budget
        selected = self._fit_token_budget(selected)

        # 6. Order logically: by document, then chunk_index
        selected.sort(
            key=lambda c: (c.metadata.filename, c.metadata.chunk_index)
        )

        logger.info(
            "context_optimized",
            input_chunks=len(chunks),
            output_chunks=len(selected),
            total_tokens=sum(c.metadata.token_count for c in selected),
        )

        return selected

    def _remove_near_duplicates(
        self, chunks: list[ScoredChunk], threshold: float = 0.8
    ) -> list[ScoredChunk]:
        """Remove chunks with very similar text (Jaccard similarity)."""
        result = []
        for chunk in chunks:
            is_duplicate = False
            chunk_words = set(chunk.text.lower().split())

            for existing in result:
                existing_words = set(existing.text.lower().split())
                if not chunk_words or not existing_words:
                    continue

                intersection = chunk_words & existing_words
                union = chunk_words | existing_words
                jaccard = len(intersection) / len(union) if union else 0

                if jaccard > threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                result.append(chunk)

        return result

    def _apply_diversity(self, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        """Ensure source diversity in selected chunks."""
        if not chunks or self.diversity_factor <= 0:
            return chunks[:self.max_chunks]

        # Group by document
        by_doc: dict[str, list[ScoredChunk]] = defaultdict(list)
        for chunk in chunks:
            by_doc[chunk.metadata.filename].append(chunk)

        # Calculate max per document
        num_docs = len(by_doc)
        if num_docs <= 1:
            return chunks[:self.max_chunks]

        max_per_doc = max(2, int(self.max_chunks / num_docs * (1 + self.diversity_factor)))

        # Round-robin selection
        selected = []
        doc_counts: dict[str, int] = defaultdict(int)

        for chunk in chunks:
            doc = chunk.metadata.filename
            if doc_counts[doc] < max_per_doc and len(selected) < self.max_chunks:
                selected.append(chunk)
                doc_counts[doc] += 1

        return selected

    def _fit_token_budget(self, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        """Trim chunks to fit within token budget."""
        result = []
        total_tokens = 0

        for chunk in chunks:
            token_count = chunk.metadata.token_count
            if token_count == 0:
                token_count = len(self._tokenizer.encode(chunk.text, disallowed_special=()))

            if total_tokens + token_count > self.max_context_tokens:
                break

            result.append(chunk)
            total_tokens += token_count

        return result
