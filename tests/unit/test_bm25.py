"""Unit tests for BM25Retriever."""

from app.models.documents import ChunkMetadata, DocumentType
from app.retrieval.bm25 import BM25Retriever, tokenize


def test_bm25_tokenize():
    tokens = tokenize("Hello, World! This is a BM25 test...")
    assert "hello" in tokens
    assert "world" in tokens
    assert "bm25" in tokens
    assert "test" in tokens


def test_bm25_retriever_operations(temp_dir):
    bm25 = BM25Retriever(index_dir=temp_dir / "bm25")
    assert bm25.count == 0

    ids = ["c1", "c2", "c3"]
    texts = [
        "Python is a popular programming language for data science and AI.",
        "FastAPI is a modern web framework for building APIs with Python.",
        "Docker allows developers to package applications into lightweight containers.",
    ]
    metas = [
        {"document_id": "d1", "filename": "f1.txt"},
        {"document_id": "d1", "filename": "f1.txt"},
        {"document_id": "d2", "filename": "f2.txt"},
    ]

    bm25.add_chunks(ids, texts, metas)
    assert bm25.count == 3

    # Query match
    results = bm25.query("FastAPI Python framework", top_k=2)
    assert len(results) > 0
    assert results[0][0] == "c2"  # c2 should score highest

    # Delete document
    deleted = bm25.delete_document("d1")
    assert deleted == 2
    assert bm25.count == 1
