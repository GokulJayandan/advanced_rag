"""
FastAPI routes for the RAG platform.

Endpoints:
- POST /ingest          — queue a document for async ingestion (returns job_id immediately)
- GET  /ingest/status/{job_id} — poll ingestion job status
- POST /query           — submit queries
- GET  /health          — health check
- GET  /stats           — collection statistics
- DELETE /documents/{document_id} — remove a document
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.api.models import (
    CitationResponse,
    DeleteResponse,
    EvidenceQualityResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    StatsResponse,
)
from app.observability.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

# These will be set during app startup via dependency injection
_pipeline = None
_orchestrator = None
_vector_store = None
_bm25_retriever = None
_embedder = None
_provider_manager = None
_settings = None

# ---------------------------------------------------------------------------
# Background ingestion job store
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class IngestJob(BaseModel):
    job_id: str
    filename: str
    status: JobStatus = JobStatus.QUEUED
    queued_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


# In-memory job store (sufficient for single-process deployment)
_jobs: dict[str, IngestJob] = {}
_job_lock = threading.Lock()


def _run_ingest_job(job_id: str, filepath: Path) -> None:
    """Worker function executed in a thread-pool thread."""
    with _job_lock:
        job = _jobs[job_id]
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc).isoformat()

    try:
        result = _pipeline.ingest(filepath)
        with _job_lock:
            job = _jobs[job_id]
            job.status = JobStatus.DONE
            job.completed_at = datetime.now(timezone.utc).isoformat()
            job.result = {
                "document_id": result.document_id,
                "filename": result.filename,
                "document_type": result.document_type.value if result.document_type else None,
                "total_chunks": result.total_chunks,
                "total_elements": result.total_elements,
                "already_exists": result.already_exists,
                "success": result.success,
                "duration_seconds": result.duration_seconds,
            }
    except Exception as exc:  # noqa: BLE001
        logger.error("ingest_job_failed", job_id=job_id, error=str(exc))
        with _job_lock:
            job = _jobs[job_id]
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now(timezone.utc).isoformat()
            job.error = str(exc)


# ---------------------------------------------------------------------------
# Route configuration
# ---------------------------------------------------------------------------

def configure_routes(pipeline, orchestrator, vector_store, bm25_retriever, embedder, provider_manager, settings):
    """Configure route dependencies. Called during app startup."""
    global _pipeline, _orchestrator, _vector_store, _bm25_retriever, _embedder, _provider_manager, _settings
    _pipeline = pipeline
    _orchestrator = orchestrator
    _vector_store = vector_store
    _bm25_retriever = bm25_retriever
    _embedder = embedder
    _provider_manager = provider_manager
    _settings = settings


# ---------------------------------------------------------------------------
# Ingestion endpoints
# ---------------------------------------------------------------------------

@router.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Queue a document for ingestion.

    Returns immediately with a job_id. Poll GET /ingest/status/{job_id} to
    track progress. This prevents Streamlit from timing out on large files.
    """
    if _pipeline is None or _settings is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(file.filename).suffix.lower()
    allowed = {".pdf", ".docx", ".txt", ".md", ".markdown"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {', '.join(sorted(allowed))}",
        )

    # Read and validate size
    content = await file.read()
    if len(content) > _settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max: {_settings.max_upload_size_mb}MB",
        )

    # Save to disk
    upload_dir = _settings.upload_path
    upload_dir.mkdir(parents=True, exist_ok=True)

    from app.ingestion.parser import sanitize_filename
    safe_name = sanitize_filename(file.filename)
    filepath = upload_dir / safe_name

    with open(filepath, "wb") as f:
        f.write(content)

    # Create job record
    job_id = uuid.uuid4().hex
    job = IngestJob(
        job_id=job_id,
        filename=file.filename,
        status=JobStatus.QUEUED,
        queued_at=datetime.now(timezone.utc).isoformat(),
    )
    with _job_lock:
        _jobs[job_id] = job

    # Dispatch to thread pool (non-blocking)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_ingest_job, job_id, filepath)

    logger.info("ingest_job_queued", job_id=job_id, filename=file.filename)
    return {"job_id": job_id, "filename": file.filename, "status": JobStatus.QUEUED}


@router.get("/ingest/status/{job_id}")
async def ingest_status(job_id: str):
    """Poll the status of an ingestion job."""
    with _job_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return job.model_dump()


# ---------------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------------

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Submit a query to the RAG system."""
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Run query orchestration in thread pool (non-blocking for event loop)
        result = await asyncio.to_thread(
            _orchestrator.query,
            query=request.query,
            conversation_history=request.conversation_history,
        )

        citations = [
            CitationResponse(
                citation_text=c.citation_text,
                filename=c.filename,
                page_number=c.page_number,
                section=c.section,
                is_valid=c.is_valid,
            )
            for c in result.citations
        ]

        evidence = EvidenceQualityResponse(
            score=result.evidence_quality.score,
            num_relevant_chunks=result.evidence_quality.num_relevant_chunks,
            is_sufficient=result.evidence_quality.is_sufficient,
            explanation=result.evidence_quality.explanation,
        )

        decisions = None
        if request.include_diagnostics:
            decisions = [
                {"step": d.step, "decision": d.decision, "reasoning": d.reasoning}
                for d in result.agent_decisions
            ]

        return QueryResponse(
            query_id=result.query_id,
            query=result.query,
            answer=result.answer,
            citations=citations,
            evidence_quality=evidence,
            sources=result.sources,
            query_type=result.query_type.value,
            total_latency_ms=round(result.total_latency_ms, 2),
            retrieval_iterations=result.retrieval_iterations,
            agent_decisions=decisions,
            error=result.error,
        )
    except Exception as e:
        logger.error("query_endpoint_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Health / Stats / Delete
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    providers = _provider_manager.available_providers if _provider_manager else []
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        providers=providers,
    )


@router.get("/stats", response_model=StatsResponse)
async def stats():
    """Collection statistics."""
    if _vector_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    vs_stats = _vector_store.stats
    cache_stats = _embedder.cache_stats if _embedder else {}

    return StatsResponse(
        total_chunks=vs_stats.get("total_chunks", 0),
        total_documents=vs_stats.get("total_documents", 0),
        bm25_indexed=_bm25_retriever.count if _bm25_retriever else 0,
        embedding_cache=cache_stats,
    )


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str):
    """Delete a document and its chunks."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    result = await asyncio.to_thread(_pipeline.delete_document, document_id)

    return DeleteResponse(
        document_id=result["document_id"],
        chroma_chunks_deleted=result["chroma_chunks_deleted"],
        bm25_chunks_deleted=result["bm25_chunks_deleted"],
    )
