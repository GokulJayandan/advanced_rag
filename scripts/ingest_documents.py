"""
CLI script to ingest documents into the Advanced Agentic RAG Platform.

Usage:
    python scripts/ingest_documents.py --dir ./data/docs
    python scripts/ingest_documents.py --file ./data/sample.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config.settings import get_settings
from app.embeddings.embedder import GeminiEmbedder
from app.ingestion.chunker import SemanticChunker
from app.ingestion.parser import DocumentParser
from app.ingestion.pipeline import IngestionPipeline
from app.observability.logging import get_logger, setup_logging
from app.retrieval.bm25 import BM25Retriever
from app.storage.vector_store import ChromaVectorStore

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into RAG database")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", type=str, help="Directory containing documents to ingest")
    group.add_argument("--file", type=str, help="Single file to ingest")

    args = parser.parse_args()

    settings = get_settings()
    setup_logging(log_level=settings.log_level, log_format="console")

    if not settings.google_api_key:
        print("ERROR: GOOGLE_API_KEY is not set in environment or .env file.")
        return

    # Initialize components
    doc_parser = DocumentParser(max_file_size_bytes=settings.max_upload_size_bytes)
    chunker = SemanticChunker(
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )
    embedder = GeminiEmbedder(
        api_key=settings.google_api_key,
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        cache_dir=settings.embedding_cache_dir,
    )
    vector_store = ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name="rag_chunks",
    )
    bm25 = BM25Retriever(index_dir=settings.bm25_index_dir)

    pipeline = IngestionPipeline(
        parser=doc_parser,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        bm25_retriever=bm25,
    )

    files_to_process: list[Path] = []
    if args.file:
        files_to_process.append(Path(args.file))
    elif args.dir:
        dir_path = Path(args.dir)
        for ext in [".pdf", ".docx", ".txt", ".md", ".markdown"]:
            files_to_process.extend(dir_path.glob(f"**/*{ext}"))

    print(f"\n🚀 Found {len(files_to_process)} document(s) to process...\n")

    for f in files_to_process:
        print(f"Ingesting: {f.name} ...")
        res = pipeline.ingest(f)
        if res.success:
            print(f"  ✅ Done: {res.total_chunks} chunks ({res.duration_seconds:.2f}s)")
        else:
            print(f"  ❌ Failed: {res.error}")

    print(f"\n🎉 Ingestion complete! Total chunks in vector store: {vector_store.count}")
    print(f"   BM25 index count: {bm25.count}")


if __name__ == "__main__":
    main()
