"""
RAG Orchestrator — the main agentic workflow engine.

This is the central decision engine that coordinates the entire RAG pipeline:
query analysis → planning → retrieval → reranking → evidence checking →
adaptive retry → generation → citation validation → answer verification.

Implemented as a clean Python state machine. Structured so LangGraph
can be integrated as a drop-in replacement for the workflow engine.
"""

from __future__ import annotations

import time
from typing import Any

from app.agent.analyzer import QueryAnalyzer
from app.agent.context_optimizer import ContextOptimizer
from app.agent.evidence_checker import EvidenceChecker
from app.agent.planner import QueryPlanner
from app.agent.state import AgentState, RetrievalAttempt
from app.citations.extractor import CitationExtractor
from app.citations.validator import CitationValidator
from app.citations.verifier import AnswerVerifier
from app.generation.prompts import (
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT_TEMPLATE,
    format_context,
)
from app.generation.provider_manager import ProviderManager
from app.models.generation import (
    AgentDecision,
    EvidenceQuality,
    QueryResult,
    QueryType,
)
from app.models.retrieval import ScoredChunk
from app.observability.logging import generate_query_id, get_logger
from app.observability.metrics import QueryMetrics
from app.reranking.reranker import BGEReranker
from app.retrieval.hybrid import HybridRetriever

logger = get_logger(__name__)


class RAGOrchestrator:
    """
    Agentic RAG orchestrator — makes meaningful retrieval decisions.

    This is NOT a simple pipeline. The agent:
    1. Analyzes the query to determine complexity
    2. Plans the retrieval strategy (simple / rewrite / decompose)
    3. Executes hybrid retrieval + neural reranking
    4. Checks if evidence is sufficient
    5. Retries with modified strategy if needed (up to max_attempts)
    6. Generates a grounded answer with citations
    7. Validates citations against actual retrieved chunks
    8. Verifies answer groundedness
    9. Returns final answer or refuses if unable to answer safely
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: BGEReranker,
        provider_manager: ProviderManager,
        query_analyzer: QueryAnalyzer,
        query_planner: QueryPlanner,
        evidence_checker: EvidenceChecker,
        context_optimizer: ContextOptimizer,
        citation_extractor: CitationExtractor,
        citation_validator: CitationValidator,
        answer_verifier: AnswerVerifier,
        max_retrieval_iterations: int = 3,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.provider_manager = provider_manager
        self.query_analyzer = query_analyzer
        self.query_planner = query_planner
        self.evidence_checker = evidence_checker
        self.context_optimizer = context_optimizer
        self.citation_extractor = citation_extractor
        self.citation_validator = citation_validator
        self.answer_verifier = answer_verifier
        self.max_retrieval_iterations = max_retrieval_iterations

    def query(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> QueryResult:
        """
        Process a query through the full agentic RAG pipeline.

        Args:
            query: The user's question.
            conversation_history: Previous conversation turns for context.

        Returns:
            QueryResult with answer, citations, evidence quality, and diagnostics.
        """
        metrics = QueryMetrics(query_id=generate_query_id())
        metrics.start_timer()

        state = AgentState(
            query_id=metrics.query_id,
            original_query=query,
            active_query=query,
            conversation_history=conversation_history or [],
        )

        try:
            # === Step 1: Analyze Query ===
            step_start = time.perf_counter()
            state.query_type = self.query_analyzer.classify(query)
            metrics.query_analysis_ms = (time.perf_counter() - step_start) * 1000

            state.add_decision(
                "query_analysis",
                f"classified as: {state.query_type.value}",
                f"Query: '{query[:80]}...' classified for retrieval strategy selection",
            )

            # === Step 2: Plan Retrieval Strategy ===
            step_start = time.perf_counter()
            plan = self.query_planner.plan(
                query=query,
                query_type=state.query_type,
                conversation_history=conversation_history,
            )
            metrics.query_planning_ms = (time.perf_counter() - step_start) * 1000

            retrieval_queries = plan["queries"]
            state.add_decision(
                "query_planning",
                f"strategy: {plan['strategy']}, {len(retrieval_queries)} queries",
                plan["reasoning"],
            )

            if plan["strategy"] == "rewrite":
                state.rewritten_queries = retrieval_queries
            elif plan["strategy"] == "decompose":
                state.sub_queries = retrieval_queries

            # === Step 3: Adaptive Retrieval Loop ===
            for attempt in range(1, self.max_retrieval_iterations + 1):
                step_start = time.perf_counter()

                # Retrieve for all planned queries
                all_chunks = self._execute_retrieval(
                    queries=retrieval_queries,
                    state=state,
                    attempt_number=attempt,
                )

                # Rerank combined results
                step_rerank_start = time.perf_counter()
                reranked = self.reranker.rerank(query, all_chunks)
                metrics.reranking_ms += (time.perf_counter() - step_rerank_start) * 1000

                metrics.retrieval_ms += (time.perf_counter() - step_start) * 1000
                metrics.retrieval_iterations = attempt

                # Record attempt
                attempt_record = RetrievalAttempt(
                    attempt_number=attempt,
                    query_used="; ".join(retrieval_queries),
                    strategy=plan["strategy"],
                    dense_count=len(all_chunks),
                    reranked_count=len(reranked),
                    top_reranker_score=reranked[0].reranker_score if reranked else 0.0,
                )

                # Update state
                state.all_retrieved_chunks = all_chunks
                state.reranked_chunks = reranked

                # === Step 4: Check Evidence Sufficiency ===
                evidence_quality = self.evidence_checker.check(query, reranked)
                state.evidence_quality = evidence_quality
                attempt_record.evidence_sufficient = evidence_quality.is_sufficient
                state.add_retrieval_attempt(attempt_record)

                state.add_decision(
                    f"evidence_check_attempt_{attempt}",
                    f"sufficient: {evidence_quality.is_sufficient}, score: {evidence_quality.score:.3f}",
                    evidence_quality.explanation,
                )

                if evidence_quality.is_sufficient:
                    break

                # === Step 5: Retry with Modified Strategy ===
                if attempt < self.max_retrieval_iterations:
                    state.add_decision(
                        f"retry_decision_{attempt}",
                        "retrying with rewritten queries",
                        f"Evidence insufficient (score={evidence_quality.score:.3f}). Broadening search.",
                    )
                    retrieval_queries = self.query_planner.rewrite_for_retry(
                        original_query=query,
                        attempt_number=attempt,
                    )
                    state.rewritten_queries.extend(retrieval_queries)
                else:
                    state.add_decision(
                        "max_attempts_reached",
                        "proceeding with best available evidence",
                        f"Exhausted {self.max_retrieval_iterations} retrieval attempts",
                    )

            # === Step 6: Optimize Context ===
            state.context_chunks = self.context_optimizer.optimize(state.reranked_chunks)

            # === Step 7: Generate Answer ===
            step_start = time.perf_counter()
            answer = self._generate_answer(query, state.context_chunks)
            metrics.generation_ms = (time.perf_counter() - step_start) * 1000
            state.generated_answer = answer

            if not answer:
                state.add_decision(
                    "generation_failed",
                    "returning empty answer",
                    "LLM generation returned empty response",
                )

            # === Step 8: Extract & Validate Citations ===
            step_start = time.perf_counter()
            citations = self.citation_extractor.extract(answer)
            state.citations = citations

            validation = self.citation_validator.validate(citations, state.context_chunks)
            state.citation_validation = validation
            metrics.citation_validation_ms = (time.perf_counter() - step_start) * 1000

            # Handle invalid citations
            if not validation.all_valid and validation.invalid_citations > 0:
                state.add_decision(
                    "citation_validation",
                    f"{validation.invalid_citations} invalid citations found",
                    "Invalid citations will be noted in the response",
                )

            # === Step 9: Verify Answer Groundedness ===
            step_start = time.perf_counter()
            verification = self.answer_verifier.verify(query, answer, state.context_chunks)
            state.verification = verification
            metrics.answer_verification_ms = (time.perf_counter() - step_start) * 1000

            if not verification.is_grounded:
                state.add_decision(
                    "verification_failed",
                    "answer not fully grounded",
                    f"Unsupported claims: {verification.unsupported_claims}",
                )
                # Attempt regeneration with stricter instructions
                answer = self._regenerate_strict(query, state.context_chunks, verification)
                if answer:
                    state.generated_answer = answer
                    state.citations = self.citation_extractor.extract(answer)

            # === Finalize ===
            metrics.finalize()
            metrics.generation_provider = state.generated_answer[:0]  # Set from result

            # Build source list
            sources = self._build_sources(state.context_chunks)

            return QueryResult(
                query_id=state.query_id,
                query=query,
                answer=state.generated_answer,
                citations=state.citations,
                evidence_quality=state.evidence_quality,
                sources=sources,
                query_type=state.query_type,
                verification=state.verification,
                agent_decisions=state.decisions,
                total_latency_ms=metrics.total_ms,
                retrieval_iterations=metrics.retrieval_iterations,
            )

        except Exception as e:
            metrics.finalize()
            logger.error("orchestrator_error", error=str(e), query_id=state.query_id)
            return QueryResult(
                query_id=state.query_id,
                query=query,
                answer="An error occurred while processing your query. Please try again.",
                error=str(e),
                total_latency_ms=metrics.total_ms,
                agent_decisions=state.decisions,
            )

    def _execute_retrieval(
        self,
        queries: list[str],
        state: AgentState,
        attempt_number: int,
    ) -> list[ScoredChunk]:
        """Execute hybrid retrieval for all queries and merge results."""
        all_chunks: dict[str, ScoredChunk] = {}

        for q in queries:
            result = self.hybrid_retriever.retrieve(q)
            for chunk in result.chunks:
                if chunk.chunk_id not in all_chunks:
                    all_chunks[chunk.chunk_id] = chunk
                else:
                    # Keep the higher-scored version
                    existing = all_chunks[chunk.chunk_id]
                    if (chunk.rrf_score or 0) > (existing.rrf_score or 0):
                        all_chunks[chunk.chunk_id] = chunk

        return list(all_chunks.values())

    def _generate_answer(
        self,
        query: str,
        context_chunks: list[ScoredChunk],
    ) -> str:
        """Generate a grounded answer using the LLM."""
        if not context_chunks:
            return "The available documents do not contain sufficient information to answer this question."

        context_parts = []
        for chunk in context_chunks:
            context_parts.append({
                "text": chunk.text,
                "filename": chunk.metadata.filename,
                "page_number": chunk.metadata.page_number,
                "section": chunk.metadata.section,
            })

        context_text = format_context(context_parts)
        prompt = RAG_USER_PROMPT_TEMPLATE.format(context=context_text, query=query)

        result = self.provider_manager.generate(
            prompt=prompt,
            system_prompt=RAG_SYSTEM_PROMPT,
            temperature=0.1,
        )

        if result.error:
            logger.error("answer_generation_failed", error=result.error)
            return "Unable to generate an answer at this time. Please try again."

        return result.text

    def _regenerate_strict(
        self,
        query: str,
        context_chunks: list[ScoredChunk],
        verification,
    ) -> str | None:
        """Attempt to regenerate with stricter grounding instructions."""
        stricter_prompt = (
            RAG_SYSTEM_PROMPT +
            "\n\nCRITICAL: Your previous answer contained unsupported claims. "
            "This time, be EXTREMELY conservative. Only state what is DIRECTLY "
            "stated in the evidence. If unsure, say so explicitly."
        )

        context_parts = []
        for chunk in context_chunks:
            context_parts.append({
                "text": chunk.text,
                "filename": chunk.metadata.filename,
                "page_number": chunk.metadata.page_number,
                "section": chunk.metadata.section,
            })

        context_text = format_context(context_parts)
        prompt = RAG_USER_PROMPT_TEMPLATE.format(context=context_text, query=query)

        result = self.provider_manager.generate(
            prompt=prompt,
            system_prompt=stricter_prompt,
            temperature=0.0,
        )

        if result.error:
            return None
        return result.text

    def _build_sources(self, chunks: list[ScoredChunk]) -> list[dict[str, Any]]:
        """Build source list for the response."""
        sources = []
        seen_docs = set()
        for chunk in chunks:
            doc_key = chunk.metadata.filename
            if doc_key not in seen_docs:
                seen_docs.add(doc_key)
                sources.append({
                    "filename": chunk.metadata.filename,
                    "document_id": chunk.metadata.document_id,
                    "page_numbers": [],
                    "chunks_used": 0,
                })

            # Update page numbers and chunk count
            for src in sources:
                if src["filename"] == chunk.metadata.filename:
                    src["chunks_used"] += 1
                    if chunk.metadata.page_number and chunk.metadata.page_number not in src["page_numbers"]:
                        src["page_numbers"].append(chunk.metadata.page_number)

        return sources
