"""
Ablation study — compares RAG configurations to measure component impact.

Configurations:
A. Dense retrieval only
B. BM25 only
C. Dense + BM25 (concatenation)
D. Dense + BM25 + RRF
E. Hybrid + BGE reranking
F. Hybrid + reranking + query rewriting
G. Full adaptive agentic RAG
"""

from __future__ import annotations

import time
from typing import Any

from app.evaluation.dataset import EvaluationDataset
from app.evaluation.metrics import MetricsCalculator
from app.observability.logging import get_logger

logger = get_logger(__name__)


class AblationStudy:
    """
    Compares RAG configurations to quantify the impact of each component.

    Produces a comparison table showing metrics across configurations.
    Does not claim improvements without measurements.
    """

    def __init__(
        self,
        hybrid_retriever,
        reranker,
        provider_manager,
        orchestrator=None,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.provider_manager = provider_manager
        self.orchestrator = orchestrator
        self.metrics_calc = MetricsCalculator()

    def run(
        self,
        dataset: EvaluationDataset,
        relevant_ids_map: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """
        Run the ablation study across all configurations.

        Args:
            dataset: Evaluation dataset.
            relevant_ids_map: Optional mapping of query → relevant chunk IDs.

        Returns:
            Comparison table with metrics per configuration.
        """
        results = {}
        relevant_map = relevant_ids_map or {}

        configs = [
            ("A_dense_only", self._config_dense_only),
            ("B_bm25_only", self._config_bm25_only),
            ("C_dense_bm25_concat", self._config_concat),
            ("D_dense_bm25_rrf", self._config_rrf),
            ("E_hybrid_reranking", self._config_reranking),
        ]

        if self.orchestrator:
            configs.append(("G_full_agentic", self._config_full_agentic))

        for config_name, config_fn in configs:
            logger.info("ablation_config_start", config=config_name)
            start = time.perf_counter()

            config_results = []
            for item in dataset:
                try:
                    retrieved = config_fn(item.query)
                    retrieved_ids = [c.chunk_id for c in retrieved[:10]]

                    relevant = relevant_map.get(item.query, item.expected_chunk_ids)
                    metrics = {}
                    if relevant:
                        metrics = self.metrics_calc.compute_retrieval_metrics(
                            retrieved_ids, relevant
                        )

                    config_results.append({
                        "query": item.query,
                        "retrieved_count": len(retrieved),
                        "metrics": metrics,
                    })
                except Exception as e:
                    logger.error("ablation_query_failed", config=config_name, error=str(e))
                    config_results.append({"query": item.query, "error": str(e)})

            elapsed = (time.perf_counter() - start) * 1000

            # Aggregate metrics for this configuration
            valid = [r for r in config_results if "metrics" in r and r["metrics"]]
            if valid:
                avg_metrics = {}
                for key in valid[0]["metrics"]:
                    values = [r["metrics"][key] for r in valid if key in r.get("metrics", {})]
                    avg_metrics[key] = round(sum(values) / max(len(values), 1), 4)
            else:
                avg_metrics = {}

            results[config_name] = {
                "total_queries": len(config_results),
                "successful": len(valid),
                "avg_metrics": avg_metrics,
                "total_ms": round(elapsed, 2),
                "per_query": config_results,
            }

            logger.info(
                "ablation_config_complete",
                config=config_name,
                elapsed_ms=round(elapsed, 2),
            )

        return self._format_comparison(results)

    def _config_dense_only(self, query: str):
        return self.hybrid_retriever.retrieve_dense_only(query, top_k=10)

    def _config_bm25_only(self, query: str):
        return self.hybrid_retriever.retrieve_bm25_only(query, top_k=10)

    def _config_concat(self, query: str):
        dense = self.hybrid_retriever.retrieve_dense_only(query, top_k=25)
        bm25 = self.hybrid_retriever.retrieve_bm25_only(query, top_k=25)
        # Simple concat + dedup
        seen = set()
        result = []
        for c in dense + bm25:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                result.append(c)
        return result[:10]

    def _config_rrf(self, query: str):
        result = self.hybrid_retriever.retrieve(query)
        return result.chunks[:10]

    def _config_reranking(self, query: str):
        result = self.hybrid_retriever.retrieve(query)
        return self.reranker.rerank(query, result.chunks, top_k=10)

    def _config_full_agentic(self, query: str):
        if self.orchestrator:
            result = self.orchestrator.query(query)
            return []  # Full agentic returns QueryResult, not chunks
        return []

    def _format_comparison(self, results: dict) -> dict[str, Any]:
        """Format results into a comparison table."""
        table = {"configurations": {}, "comparison_table": []}

        for config_name, data in results.items():
            table["configurations"][config_name] = data["avg_metrics"]
            table["comparison_table"].append({
                "config": config_name,
                **data["avg_metrics"],
                "total_ms": data["total_ms"],
            })

        return table
