"""Unit tests for embedding cache."""

from app.embeddings.cache import EmbeddingCache


def test_embedding_cache_lifecycle(temp_dir):
    cache = EmbeddingCache(cache_dir=temp_dir / "cache")
    assert cache.size == 0

    text = "Hello world embedding test"
    vec = [0.1, 0.2, 0.3, 0.4]

    # Miss
    assert cache.get(text) is None
    assert cache.stats["misses"] == 1

    # Put and Hit
    cache.put(text, vec)
    assert cache.get(text) == vec
    assert cache.stats["hits"] == 1
    assert cache.size == 1

    # Save and reload
    cache.save()
    reloaded = EmbeddingCache(cache_dir=temp_dir / "cache")
    assert reloaded.size == 1
    assert reloaded.get(text) == vec
