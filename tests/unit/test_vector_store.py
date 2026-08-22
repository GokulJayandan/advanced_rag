"""Unit tests for ChromaVectorStore."""

from app.storage.vector_store import ChromaVectorStore


def test_vector_store_operations(temp_dir, sample_chunks):
    store = ChromaVectorStore(persist_dir=temp_dir / "chroma", collection_name="test_col")
    assert store.count == 0

    # Add chunks with dummy embeddings
    embeddings = [[0.1 * i] * 768 for i in range(len(sample_chunks))]
    added = store.add_chunks(sample_chunks, embeddings)
    assert added == len(sample_chunks)
    assert store.count == len(sample_chunks)

    # Adding duplicate chunks should return 0
    added_dup = store.add_chunks(sample_chunks, embeddings)
    assert added_dup == 0

    # Query
    query_emb = [0.1] * 768
    results = store.query(query_emb, top_k=3)
    assert len(results) == 3

    # Delete document
    deleted = store.delete_document("doc-1")
    assert deleted == len(sample_chunks)
    assert store.count == 0
