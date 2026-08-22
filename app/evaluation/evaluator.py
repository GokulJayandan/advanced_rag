"""
RAG evaluator — runs evaluation over a test dataset and produces metrics.
"""

from __future__ import annotations

import time
from typing import Any

from app.agent.orchestrator import RAGOrchestrator
from app.evaluation.dataset import EvaluationDataset, EvalItem
from app.evaluation.metrics import MetricsCalculator
from app.observability.logging import get_logger

logger = get_logger(__name__)


class RAGEvaluator:
    """
    Evaluates the RAG system on a curated dataset.

    Produces per-query and aggregate metrics for retrieval quality,
    generation quality, and system performance.
    """

    def __init__(self, orchestrator: RAGOrchestrator):
        self.orchestrator = orchestrator
        self.metrics_calc = MetricsCalculator()

    def evaluate(
        self,
        dataset: EvaluationDataset,
    ) -> dict[str, Any]:
        """
        Run evaluation on the entire dataset.

        Returns structured results with per-query and aggregate metrics.
        """
        per_query_results = []
        total_start = time.perf_counter()

        for i, item in enumerate(dataset):
            logger.info("evaluating_query", index=i, query=item.query[:80])

            try:
                result = self.orchestrator.query(item.query)

                # Retrieval metrics
                retrieved_ids = []
                for src in result.sources:
                    retrieved_ids.append(src.get("document_id", ""))

                retrieval_metrics = {}
                if item.expected_chunk_ids:
                    retrieved_chunk_ids = [c.chunk_id for c in
                        self.orchestrator.hybrid_retriever.retrieve(item.query).chunks[:10]
                    ] if hasattr(self.orchestrator, 'hybrid_retriever') else []
                    retrieval_metrics = self.metrics_calc.compute_retrieval_metrics(
                        retrieved_chunk_ids, item.expected_chunk_ids
                    )

                # Generation metrics
                has_answer = bool(result.answer and "do not contain" not in result.answer.lower())
                has_citations = len(result.citations) > 0
                is_grounded = result.verification.is_grounded if result.verification else True

                query_result = {
                    "query": item.query,
                    "answer": result.answer[:200],
                    "expected_answer": item.expected_answer[:200],
                    "query_type": result.query_type.value,
                    "has_answer": has_answer,
                    "has_citations": has_citations,
                    "num_citations": len(result.citations),
                    "is_grounded": is_grounded,
                    "evidence_score": result.evidence_quality.score,
                    "total_latency_ms": result.total_latency_ms,
                    "retrieval_iterations": result.retrieval_iterations,
                    "retrieval_metrics": retrieval_metrics,
                    "error": result.error,
                }

                per_query_results.append(query_result)

            except Exception as e:
                logger.error("evaluation_query_failed", query=item.query[:80], error=str(e))
                per_query_results.append({
                    "query": item.query,
                    "error": str(e),
                })

        total_elapsed = (time.perf_counter() - total_start) * 1000

        # Aggregate
        valid_results = [r for r in per_query_results if "answer" in r]
        aggregate = {
            "total_queries": len(per_query_results),
            "successful_queries": len(valid_results),
            "failed_queries": len(per_query_results) - len(valid_results),
            "avg_evidence_score": sum(r.get("evidence_score", 0) for r in valid_results) / max(len(valid_results), 1),
            "answer_rate": sum(1 for r in valid_results if r.get("has_answer")) / max(len(valid_results), 1),
            "citation_rate": sum(1 for r in valid_results if r.get("has_citations")) / max(len(valid_results), 1),
            "grounded_rate": sum(1 for r in valid_results if r.get("is_grounded")) / max(len(valid_results), 1),
            "avg_latency_ms": sum(r.get("total_latency_ms", 0) for r in valid_results) / max(len(valid_results), 1),
            "total_evaluation_ms": round(total_elapsed, 2),
        }

        return {
            "aggregate": aggregate,
            "per_query": per_query_results,
        }
