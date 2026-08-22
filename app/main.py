"""
Main FastAPI Application Entrypoint.

Initializes configuration, dependencies, and lifecycle management for the
Advanced Agentic RAG Platform.
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent.analyzer import QueryAnalyzer
from app.agent.context_optimizer import ContextOptimizer
from app.agent.evidence_checker import EvidenceChecker
from app.agent.orchestrator import RAGOrchestrator
from app.agent.planner import QueryPlanner
from app.api.routes import configure_routes, router as api_router
from app.citations.extractor import CitationExtractor
from app.citations.validator import CitationValidator
from app.citations.verifier import AnswerVerifier
from app.config.settings import get_settings
from app.embeddings.embedder import GeminiEmbedder
from app.generation.provider_manager import ProviderManager
from app.ingestion.chunker import SemanticChunker
from app.ingestion.parser import DocumentParser
from app.ingestion.pipeline import IngestionPipeline
from app.observability.logging import get_logger, setup_logging
from app.reranking.reranker import BGEReranker
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.rrf import ReciprocalRankFusion
from app.storage.vector_store import ChromaVectorStore

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Factory function to build and configure the FastAPI application."""
    settings = get_settings()
    settings.ensure_directories()
    setup_logging(log_level=settings.log_level, log_format=settings.log_format)

    # Initialize components
    parser = DocumentParser(max_file_size_bytes=settings.max_upload_size_bytes)
    chunker = SemanticChunker(
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )

    embedder = None
    if settings.google_api_key:
        embedder = GeminiEmbedder(
            api_key=settings.google_api_key,
            model=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            cache_dir=settings.embedding_cache_dir,
            max_retries=settings.max_retries,
        )

    vector_store = ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name="rag_chunks",
    )

    bm25_retriever = BM25Retriever(index_dir=settings.bm25_index_dir)

    # Sync BM25 if Chroma has documents but BM25 is empty
    if vector_store.count > 0 and bm25_retriever.count == 0:
        logger.info("syncing_bm25_from_chroma")
        all_chunks = vector_store.get_all_chunks()
        bm25_retriever.build_from_store(all_chunks)

    pipeline = None
    if embedder:
        pipeline = IngestionPipeline(
            parser=parser,
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
        )

    dense_retriever = None
    if embedder:
        dense_retriever = DenseRetriever(
            embedder=embedder,
            vector_store=vector_store,
            top_k=settings.dense_top_k,
        )

    rrf = ReciprocalRankFusion(k=settings.rrf_k)

    hybrid_retriever = None
    if dense_retriever:
        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            rrf=rrf,
            dense_top_k=settings.dense_top_k,
            bm25_top_k=settings.bm25_top_k,
            hybrid_candidates=settings.hybrid_candidates,
        )

    reranker = BGEReranker(
        model_name=settings.reranker_model,
        device=settings.reranker_device,
        top_k=settings.reranker_top_k,
        threshold=settings.reranker_threshold,
    )

    provider_manager = ProviderManager(
        groq_api_key=settings.groq_api_key,
        gemini_api_key=settings.google_api_key,
        groq_model=settings.groq_model,
        gemini_model=settings.gemini_generation_model,
        timeout=settings.api_timeout_seconds,
    )

    query_analyzer = QueryAnalyzer(provider_manager=provider_manager)
    query_planner = QueryPlanner(provider_manager=provider_manager)
    evidence_checker = EvidenceChecker(
        provider_manager=provider_manager,
        sufficiency_threshold=settings.evidence_sufficiency_threshold,
    )
    context_optimizer = ContextOptimizer(
        max_context_tokens=settings.max_context_tokens,
        max_chunks=settings.reranker_top_k,
    )
    citation_extractor = CitationExtractor()
    citation_validator = CitationValidator()
    answer_verifier = AnswerVerifier(provider_manager=provider_manager)

    orchestrator = None
    if hybrid_retriever:
        orchestrator = RAGOrchestrator(
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            provider_manager=provider_manager,
            query_analyzer=query_analyzer,
            query_planner=query_planner,
            evidence_checker=evidence_checker,
            context_optimizer=context_optimizer,
            citation_extractor=citation_extractor,
            citation_validator=citation_validator,
            answer_verifier=answer_verifier,
            max_retrieval_iterations=settings.max_retrieval_iterations,
        )

    configure_routes(
        pipeline=pipeline,
        orchestrator=orchestrator,
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        embedder=embedder,
        provider_manager=provider_manager,
        settings=settings,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("rag_api_starting", version="1.0.0")
        warnings = settings.validate_api_keys()
        for w in warnings:
            logger.warning("api_key_missing", warning=w)

        # Preload the BGE reranker in a background thread so the first query
        # doesn't block for several minutes downloading the model.
        def _preload_reranker():
            try:
                logger.info("preloading_reranker_start")
                reranker._load_model()
                logger.info("preloading_reranker_done")
            except Exception as exc:  # noqa: BLE001
                logger.warning("preloading_reranker_failed", error=str(exc))

        threading.Thread(target=_preload_reranker, daemon=True, name="reranker-preload").start()

        yield
        logger.info("rag_api_shutting_down")

    app = FastAPI(
        title="Advanced Agentic RAG Platform API",
        description="Production-grade Agentic RAG with Hybrid Retrieval, BGE Reranking, Grounded Generation, and Citation Validation.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port, reload=True)
