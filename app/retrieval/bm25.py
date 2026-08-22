"""
BM25 sparse retrieval — independent from the vector store.

Operates over the same chunks stored in Chroma, but maintains its own
index. Independently replaceable from dense retrieval.
"""

from __future__ import annotations

import json
import pickle
import re
import string
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.models.documents import ChunkMetadata
from app.observability.logging import get_logger

logger = get_logger(__name__)


def tokenize(text: str) -> list[str]:
    """Simple tokenization: lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    # Remove very short tokens
    return [t for t in tokens if len(t) > 1]


class BM25Retriever:
    """
    Independent BM25 sparse retrieval over document chunks.

    Maintains its own index that mirrors the chunks in ChromaDB.
    Supports indexing, updating, deletion, querying, and persistence.
    """

    def __init__(self, index_dir: str | Path = "./bm25_index"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self._chunk_ids: list[str] = []
        self._chunk_texts: list[str] = []
        self._chunk_metadatas: list[dict] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

        self._load()

    def _load(self) -> None:
        """Load the BM25 index from disk."""
        index_file = self.index_dir / "bm25_data.pkl"
        if index_file.exists():
            try:
                with open(index_file, "rb") as f:
                    data = pickle.load(f)
                self._chunk_ids = data["chunk_ids"]
                self._chunk_texts = data["chunk_texts"]
                self._chunk_metadatas = data["chunk_metadatas"]
                self._tokenized_corpus = data["tokenized_corpus"]
                if self._tokenized_corpus:
                    self._bm25 = BM25Okapi(self._tokenized_corpus)
                logger.info("bm25_index_loaded", chunks=len(self._chunk_ids))
            except Exception as e:
                logger.warning("bm25_load_failed", error=str(e))
                self._reset()

    def _save(self) -> None:
        """Persist BM25 index to disk."""
        index_file = self.index_dir / "bm25_data.pkl"
        tmp_file = index_file.with_suffix(".tmp")
        try:
            data = {
                "chunk_ids": self._chunk_ids,
                "chunk_texts": self._chunk_texts,
                "chunk_metadatas": self._chunk_metadatas,
                "tokenized_corpus": self._tokenized_corpus,
            }
            with open(tmp_file, "wb") as f:
                pickle.dump(data, f)
            tmp_file.replace(index_file)
        except OSError as e:
            logger.error("bm25_save_failed", error=str(e))

    def _reset(self) -> None:
        """Reset the index."""
        self._chunk_ids = []
        self._chunk_texts = []
        self._chunk_metadatas = []
        self._tokenized_corpus = []
        self._bm25 = None

    def _rebuild_bm25(self) -> None:
        """Rebuild the BM25 model from the tokenized corpus."""
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self._bm25 = None

    def add_chunks(
        self,
        chunk_ids: list[str],
        texts: list[str],
        metadatas: list[dict],
    ) -> int:
        """
        Add chunks to the BM25 index. Skips duplicates.

        Returns number of new chunks added.
        """
        existing = set(self._chunk_ids)
        added = 0

        for cid, text, meta in zip(chunk_ids, texts, metadatas):
            if cid in existing:
                continue
            self._chunk_ids.append(cid)
            self._chunk_texts.append(text)
            self._chunk_metadatas.append(meta)
            self._tokenized_corpus.append(tokenize(text))
            existing.add(cid)
            added += 1

        if added > 0:
            self._rebuild_bm25()
            self._save()
            logger.info("bm25_chunks_added", new=added, total=len(self._chunk_ids))

        return added

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document. Returns count deleted."""
        indices_to_remove = [
            i for i, meta in enumerate(self._chunk_metadatas)
            if meta.get("document_id") == document_id
        ]

        if not indices_to_remove:
            return 0

        # Remove in reverse order to preserve indices
        for i in sorted(indices_to_remove, reverse=True):
            del self._chunk_ids[i]
            del self._chunk_texts[i]
            del self._chunk_metadatas[i]
            del self._tokenized_corpus[i]

        self._rebuild_bm25()
        self._save()

        logger.info(
            "bm25_document_deleted",
            document_id=document_id,
            chunks_deleted=len(indices_to_remove),
        )

        return len(indices_to_remove)

    def query(
        self,
        query_text: str,
        top_k: int = 50,
    ) -> list[tuple[str, str, dict, float]]:
        """
        Query the BM25 index.

        Returns list of (chunk_id, text, metadata, score) tuples,
        sorted by score descending.
        """
        if self._bm25 is None or not self._chunk_ids:
            return []

        tokenized_query = tokenize(query_text)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score
        scored_indices = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results = []
        for idx, score in scored_indices:
            if score > 0:
                results.append((
                    self._chunk_ids[idx],
                    self._chunk_texts[idx],
                    self._chunk_metadatas[idx],
                    float(score),
                ))

        return results

    @property
    def count(self) -> int:
        return len(self._chunk_ids)

    def build_from_store(
        self,
        chunks: list[tuple[str, str, ChunkMetadata]],
    ) -> None:
        """Rebuild BM25 index from vector store contents."""
        self._reset()
        for chunk_id, text, meta in chunks:
            self._chunk_ids.append(chunk_id)
            self._chunk_texts.append(text)
            self._chunk_metadatas.append({
                "document_id": meta.document_id,
                "chunk_id": meta.chunk_id,
                "filename": meta.filename,
                "source_path": meta.source_path,
                "page_number": meta.page_number,
                "section": meta.section,
                "document_type": meta.document_type.value if hasattr(meta.document_type, 'value') else str(meta.document_type),
                "chunk_index": meta.chunk_index,
                "token_count": meta.token_count,
            })
            self._tokenized_corpus.append(tokenize(text))

        self._rebuild_bm25()
        self._save()

        logger.info("bm25_rebuilt_from_store", total_chunks=len(self._chunk_ids))
