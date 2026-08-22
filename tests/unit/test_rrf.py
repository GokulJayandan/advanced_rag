"""Unit tests for Reciprocal Rank Fusion (RRF)."""

from app.models.retrieval import ScoredChunk
from app.retrieval.rrf import ReciprocalRankFusion


def test_rrf_fusion(sample_chunks):
    rrf = ReciprocalRankFusion(k=60)

    # Dense results (c0 rank 1, c1 rank 2)
    dense_results = [
        ScoredChunk(chunk_id=sample_chunks[0].chunk_id, text=sample_chunks[0].text, metadata=sample_chunks[0].metadata, dense_rank=1),
        ScoredChunk(chunk_id=sample_chunks[1].chunk_id, text=sample_chunks[1].text, metadata=sample_chunks[1].metadata, dense_rank=2),
    ]

    # BM25 results (c1 rank 1, c2 rank 2)
    bm25_results = [
        ScoredChunk(chunk_id=sample_chunks[1].chunk_id, text=sample_chunks[1].text, metadata=sample_chunks[1].metadata, bm25_rank=1),
        ScoredChunk(chunk_id=sample_chunks[2].chunk_id, text=sample_chunks[2].text, metadata=sample_chunks[2].metadata, bm25_rank=2),
    ]

    fused = rrf.fuse(dense_results, bm25_results, top_k=5)

    assert len(fused) == 3
    # c1 is ranked in both lists -> should have highest RRF score: 1/(60+2) + 1/(60+1)
    assert fused[0].chunk_id == sample_chunks[1].chunk_id
    assert fused[0].dense_rank == 2
    assert fused[0].bm25_rank == 1
    assert fused[0].rrf_score is not None
