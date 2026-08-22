"""Unit tests for ContextOptimizer."""

from app.agent.context_optimizer import ContextOptimizer
from app.models.documents import ChunkMetadata, DocumentType
from app.models.retrieval import ScoredChunk


def test_context_optimizer_deduplication(sample_scored_chunks):
    optimizer = ContextOptimizer(max_context_tokens=4000, max_chunks=10)

    # Add duplicate chunk
    duplicated = list(sample_scored_chunks) + [sample_scored_chunks[0].model_copy()]
    optimized = optimizer.optimize(duplicated)

    chunk_ids = [c.chunk_id for c in optimized]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert len(optimized) == len(sample_scored_chunks)
