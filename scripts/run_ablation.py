"""
CLI script to run ablation studies comparing retrieval and ranking configurations.

Usage:
    python scripts/run_ablation.py --dataset ./data/eval_dataset.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config.settings import get_settings
from app.embeddings.embedder import GeminiEmbedder
from app.evaluation.ablation import AblationStudy
from app.evaluation.dataset import EvaluationDataset
from app.generation.provider_manager import ProviderManager
from app.observability.logging import setup_logging
from app.reranking.reranker import BGEReranker
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.rrf import ReciprocalRankFusion
from app.storage.vector_store import ChromaVectorStore


def main():
    parser = argparse.ArgumentParser(description="Run RAG ablation study")
    parser.add_argument("--dataset", type=str, default="./data/eval_dataset.json", help="Path to evaluation dataset")
    parser.add_argument("--output", type=str, default="./eval_results/ablation_report.json", help="Path for ablation output")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(log_level="INFO", log_format="console")

    dataset = EvaluationDataset(args.dataset)
    print(f"Loaded {len(dataset)} queries for ablation study")

    embedder = GeminiEmbedder(
        api_key=settings.google_api_key,
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        cache_dir=settings.embedding_cache_dir,
    )
    vector_store = ChromaVectorStore(persist_dir=settings.chroma_persist_dir)
    bm25 = BM25Retriever(index_dir=settings.bm25_index_dir)
    dense = DenseRetriever(embedder=embedder, vector_store=vector_store, top_k=settings.dense_top_k)
    rrf = ReciprocalRankFusion(k=settings.rrf_k)
    hybrid = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25, rrf=rrf)
    reranker = BGEReranker(model_name=settings.reranker_model, device=settings.reranker_device)
    provider_mgr = ProviderManager(
        groq_api_key=settings.groq_api_key,
        gemini_api_key=settings.google_api_key,
    )

    ablation = AblationStudy(
        hybrid_retriever=hybrid,
        reranker=reranker,
        provider_manager=provider_mgr,
    )

    print("\n🔬 Running Ablation Study (Configurations A -> E)...")
    results = ablation.run(dataset)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📊 Ablation Comparison Table (saved to {out_path}):\n")
    print(f"{'Configuration':<25} | {'Total Latency (ms)':<18}")
    print("-" * 50)
    for row in results.get("comparison_table", []):
        print(f"{row.get('config', ''):<25} | {row.get('total_ms', 0):<18.2f}")


if __name__ == "__main__":
    main()
