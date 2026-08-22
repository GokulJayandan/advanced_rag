"""
Evaluation metrics for retrieval, reranking, and generation quality.
"""

from __future__ import annotations

import math
from typing import Any


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of relevant documents found in the top-k retrieved."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(top_k & relevant) / len(relevant)


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of top-k retrieved documents that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant = set(relevant_ids)
    return sum(1 for doc_id in top_k if doc_id in relevant) / k


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Mean Reciprocal Rank — reciprocal of the rank of the first relevant result."""
    relevant = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    relevant = set(relevant_ids)

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = 1.0 if doc_id in relevant else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because rank starts at 1

    # Ideal DCG
    ideal_rels = sorted([1.0] * min(len(relevant), k) + [0.0] * max(k - len(relevant), 0), reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels[:k]))

    return dcg / idcg if idcg > 0 else 0.0


def ranking_improvement(
    pre_rerank_positions: list[int],
    post_rerank_positions: list[int],
) -> float:
    """Average improvement in rank position after reranking (positive = improvement)."""
    if not pre_rerank_positions or not post_rerank_positions:
        return 0.0
    improvements = [pre - post for pre, post in zip(pre_rerank_positions, post_rerank_positions)]
    return sum(improvements) / len(improvements)


class MetricsCalculator:
    """Calculates retrieval and generation metrics for evaluation."""

    @staticmethod
    def compute_retrieval_metrics(
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k_values: list[int] | None = None,
    ) -> dict[str, float]:
        """Compute all retrieval metrics."""
        if k_values is None:
            k_values = [1, 3, 5, 10]

        metrics: dict[str, float] = {}
        for k in k_values:
            metrics[f"recall@{k}"] = round(recall_at_k(retrieved_ids, relevant_ids, k), 4)
            metrics[f"precision@{k}"] = round(precision_at_k(retrieved_ids, relevant_ids, k), 4)
            metrics[f"ndcg@{k}"] = round(ndcg_at_k(retrieved_ids, relevant_ids, k), 4)

        metrics["mrr"] = round(mrr(retrieved_ids, relevant_ids), 4)

        return metrics

    @staticmethod
    def compute_latency_metrics(
        retrieval_ms: float,
        reranking_ms: float,
        generation_ms: float,
        total_ms: float,
    ) -> dict[str, float]:
        """Compute latency metrics."""
        return {
            "retrieval_ms": round(retrieval_ms, 2),
            "reranking_ms": round(reranking_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "total_ms": round(total_ms, 2),
        }
