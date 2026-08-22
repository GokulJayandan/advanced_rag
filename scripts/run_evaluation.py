"""
CLI script to run full evaluation on curated dataset.

Usage:
    python scripts/run_evaluation.py --dataset ./data/eval_dataset.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.agent.analyzer import QueryAnalyzer
from app.agent.context_optimizer import ContextOptimizer
from app.agent.evidence_checker import EvidenceChecker
from app.agent.orchestrator import RAGOrchestrator
from app.agent.planner import QueryPlanner
from app.citations.extractor import CitationExtractor
from app.citations.validator import CitationValidator
from app.citations.verifier import AnswerVerifier
from app.config.settings import get_settings
from app.embeddings.embedder import GeminiEmbedder
from app.evaluation.dataset import EvaluationDataset
from app.evaluation.evaluator import RAGEvaluator
from app.generation.provider_manager import ProviderManager
from app.observability.logging import setup_logging
from app.reranking.reranker import BGEReranker
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.rrf import ReciprocalRankFusion
from app.storage.vector_store import ChromaVectorStore


def main():
    parser = argparse.ArgumentParser(description="Run RAG system evaluation")
    parser.add_argument("--dataset", type=str, default="./data/eval_dataset.json", help="Path to evaluation dataset")
    parser.add_argument("--output", type=str, default="./eval_results/eval_report.json", help="Path for results output")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(log_level="INFO", log_format="console")

    dataset = EvaluationDataset(args.dataset)
    print(f"Loaded {len(dataset)} evaluation queries from {args.dataset}")

    # Build components
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
        groq_model=settings.groq_model,
        gemini_model=settings.gemini_generation_model,
    )

    orchestrator = RAGOrchestrator(
        hybrid_retriever=hybrid,
        reranker=reranker,
        provider_manager=provider_mgr,
        query_analyzer=QueryAnalyzer(provider_mgr),
        query_planner=QueryPlanner(provider_mgr),
        evidence_checker=EvidenceChecker(provider_mgr),
        context_optimizer=ContextOptimizer(),
        citation_extractor=CitationExtractor(),
        citation_validator=CitationValidator(),
        answer_verifier=AnswerVerifier(provider_mgr),
    )

    evaluator = RAGEvaluator(orchestrator=orchestrator)
    print("\n🚀 Executing evaluation suite...")
    results = evaluator.evaluate(dataset)

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📊 Evaluation Summary (saved to {out_path}):")
    print(json.dumps(results["aggregate"], indent=2))


if __name__ == "__main__":
    main()
