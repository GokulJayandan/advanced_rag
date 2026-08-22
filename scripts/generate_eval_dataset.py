"""
CLI script to dynamically generate evaluation datasets from your ingested documents.

Usage:
    python scripts/generate_eval_dataset.py --samples 10 --per-chunk 2 --output ./data/dynamic_eval.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config.settings import get_settings
from app.evaluation.generator import DynamicEvalGenerator
from app.generation.provider_manager import ProviderManager
from app.observability.logging import setup_logging
from app.storage.vector_store import ChromaVectorStore


def main():
    parser = argparse.ArgumentParser(description="Dynamically generate evaluation dataset from ingested documents")
    parser.add_argument("--samples", type=int, default=10, help="Number of document chunks to sample")
    parser.add_argument("--per-chunk", type=int, default=2, help="Number of questions per chunk")
    parser.add_argument("--output", type=str, default="./data/dynamic_eval.json", help="Output JSON path")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(log_level="INFO", log_format="console")

    vector_store = ChromaVectorStore(persist_dir=settings.chroma_persist_dir)
    if vector_store.count == 0:
        print("❌ Error: Vector store is empty. Ingest some documents first with:")
        print("   python scripts/ingest_documents.py --dir ./your_docs_folder")
        return

    provider_mgr = ProviderManager(
        groq_api_key=settings.groq_api_key,
        gemini_api_key=settings.google_api_key,
    )

    generator = DynamicEvalGenerator(
        vector_store=vector_store,
        provider_manager=provider_mgr,
    )

    print(f"\n🧠 Generating dynamic evaluation dataset from {vector_store.count} indexed chunks...")
    dataset = generator.generate_from_store(
        num_chunks_to_sample=args.samples,
        questions_per_chunk=args.per_chunk,
        output_filepath=args.output,
    )

    print(f"\n✅ Successfully generated {len(dataset)} dynamic evaluation questions!")
    print(f"📁 Saved to: {args.output}")


if __name__ == "__main__":
    main()
