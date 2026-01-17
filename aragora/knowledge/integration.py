"""
Knowledge Pipeline Integration with Document Upload System.

Provides hooks to process uploaded documents through the knowledge pipeline
for automatic embedding and fact extraction.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor

from aragora.knowledge.pipeline import KnowledgePipeline, PipelineConfig, ProcessingResult

logger = logging.getLogger(__name__)

# Global pipeline instance (lazily initialized)
_pipeline: Optional[KnowledgePipeline] = None
_pipeline_lock = asyncio.Lock()

# Thread pool for running async code from sync context
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="knowledge-")


@dataclass
class KnowledgeProcessingConfig:
    """Configuration for knowledge processing on upload."""

    # Enable/disable processing
    enabled: bool = True

    # Processing options
    extract_facts: bool = True
    min_fact_confidence: float = 0.5

    # Async options
    process_async: bool = True  # Process in background

    # Callbacks
    on_complete: Optional[Callable[[ProcessingResult], None]] = None
    on_error: Optional[Callable[[str, Exception], None]] = None


@dataclass
class ProcessingJob:
    """Tracks a knowledge processing job."""

    job_id: str
    document_id: str
    filename: str
    workspace_id: str
    status: str = "pending"  # pending, processing, completed, failed
    result: Optional[ProcessingResult] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


# In-memory job tracking (replace with Redis/DB in production)
_jobs: dict[str, ProcessingJob] = {}


async def get_pipeline(workspace_id: str = "default") -> KnowledgePipeline:
    """Get or create the knowledge pipeline for a workspace.

    Args:
        workspace_id: Workspace identifier

    Returns:
        Initialized KnowledgePipeline
    """
    global _pipeline

    async with _pipeline_lock:
        if _pipeline is None:
            config = PipelineConfig(
                workspace_id=workspace_id,
                use_weaviate=_should_use_weaviate(),
                extract_facts=True,
            )
            _pipeline = KnowledgePipeline(config)
            await _pipeline.start()
            logger.info(f"Knowledge pipeline initialized for workspace {workspace_id}")

        return _pipeline


def _should_use_weaviate() -> bool:
    """Check if Weaviate should be used based on environment."""
    import os

    return os.environ.get("ARAGORA_WEAVIATE_ENABLED", "false").lower() == "true"


async def process_document_async(
    content: bytes,
    filename: str,
    workspace_id: str = "default",
    document_id: Optional[str] = None,
    config: Optional[KnowledgeProcessingConfig] = None,
) -> ProcessingResult:
    """Process a document through the knowledge pipeline.

    Args:
        content: Raw file content
        filename: Original filename
        workspace_id: Workspace identifier
        document_id: Optional existing document ID
        config: Processing configuration

    Returns:
        ProcessingResult with chunks and facts
    """
    config = config or KnowledgeProcessingConfig()

    pipeline = await get_pipeline(workspace_id)

    result = await pipeline.process_document(
        content=content,
        filename=filename,
        extract_facts=config.extract_facts,
    )

    if config.on_complete and result.success:
        try:
            config.on_complete(result)
        except Exception as e:
            logger.warning(f"on_complete callback failed: {e}")

    if config.on_error and not result.success:
        try:
            config.on_error(result.document_id, Exception(result.error or "Unknown error"))
        except Exception as e:
            logger.warning(f"on_error callback failed: {e}")

    return result


def process_document_sync(
    content: bytes,
    filename: str,
    workspace_id: str = "default",
    document_id: Optional[str] = None,
    config: Optional[KnowledgeProcessingConfig] = None,
) -> ProcessingResult:
    """Synchronous wrapper for process_document_async.

    Runs the async processing in a thread pool.

    Args:
        content: Raw file content
        filename: Original filename
        workspace_id: Workspace identifier
        document_id: Optional existing document ID
        config: Processing configuration

    Returns:
        ProcessingResult with chunks and facts
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            process_document_async(
                content=content,
                filename=filename,
                workspace_id=workspace_id,
                document_id=document_id,
                config=config,
            )
        )
    finally:
        loop.close()


def queue_document_processing(
    content: bytes,
    filename: str,
    workspace_id: str = "default",
    document_id: Optional[str] = None,
    config: Optional[KnowledgeProcessingConfig] = None,
) -> str:
    """Queue a document for background knowledge processing.

    Returns immediately with a job ID that can be used to check status.

    Args:
        content: Raw file content
        filename: Original filename
        workspace_id: Workspace identifier
        document_id: Optional existing document ID
        config: Processing configuration

    Returns:
        Job ID for tracking
    """
    import uuid

    job_id = f"kp_{uuid.uuid4().hex[:12]}"
    doc_id = document_id or f"doc_{uuid.uuid4().hex[:12]}"

    job = ProcessingJob(
        job_id=job_id,
        document_id=doc_id,
        filename=filename,
        workspace_id=workspace_id,
    )
    _jobs[job_id] = job

    def run_processing():
        try:
            job.status = "processing"
            result = process_document_sync(
                content=content,
                filename=filename,
                workspace_id=workspace_id,
                document_id=doc_id,
                config=config,
            )
            job.result = result
            job.status = "completed" if result.success else "failed"
            job.error = result.error
            job.completed_at = datetime.utcnow()

            logger.info(
                f"Knowledge processing completed: {job_id} "
                f"chunks={result.chunk_count} facts={result.fact_count}"
            )

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.completed_at = datetime.utcnow()
            logger.error(f"Knowledge processing failed: {job_id} - {e}")

    _executor.submit(run_processing)

    logger.info(f"Queued knowledge processing: {job_id} for {filename}")
    return job_id


def get_job_status(job_id: str) -> Optional[dict[str, Any]]:
    """Get the status of a knowledge processing job.

    Args:
        job_id: Job identifier

    Returns:
        Job status dict or None if not found
    """
    job = _jobs.get(job_id)
    if not job:
        return None

    return {
        "job_id": job.job_id,
        "document_id": job.document_id,
        "filename": job.filename,
        "workspace_id": job.workspace_id,
        "status": job.status,
        "error": job.error,
        "created_at": job.created_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "result": (
            {
                "chunk_count": job.result.chunk_count,
                "fact_count": job.result.fact_count,
                "embedded_count": job.result.embedded_count,
                "duration_ms": job.result.duration_ms,
            }
            if job.result
            else None
        ),
    }


def get_all_jobs(
    workspace_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Get all knowledge processing jobs with optional filtering.

    Args:
        workspace_id: Filter by workspace
        status: Filter by status (pending, processing, completed, failed)
        limit: Maximum jobs to return

    Returns:
        List of job status dicts
    """
    jobs = list(_jobs.values())

    if workspace_id:
        jobs = [j for j in jobs if j.workspace_id == workspace_id]

    if status:
        jobs = [j for j in jobs if j.status == status]

    # Sort by created_at descending
    jobs.sort(key=lambda j: j.created_at, reverse=True)

    return [get_job_status(j.job_id) for j in jobs[:limit] if get_job_status(j.job_id)]


async def shutdown_pipeline() -> None:
    """Shutdown the knowledge pipeline gracefully."""
    global _pipeline

    async with _pipeline_lock:
        if _pipeline:
            await _pipeline.stop()
            _pipeline = None
            logger.info("Knowledge pipeline shutdown complete")

    _executor.shutdown(wait=True)


# Convenience function for document handlers
def process_uploaded_document(
    content: bytes,
    filename: str,
    workspace_id: str = "default",
    document_id: Optional[str] = None,
    async_processing: bool = True,
) -> dict[str, Any]:
    """Process an uploaded document through the knowledge pipeline.

    This is the main entry point for document handlers.

    Args:
        content: Raw file content
        filename: Original filename
        workspace_id: Workspace identifier
        document_id: Document ID from storage
        async_processing: If True, process in background and return job_id

    Returns:
        Dict with processing info:
        - If async: {"job_id": "...", "status": "queued"}
        - If sync: {"success": True, "chunks": N, "facts": N}
    """
    if async_processing:
        job_id = queue_document_processing(
            content=content,
            filename=filename,
            workspace_id=workspace_id,
            document_id=document_id,
        )
        return {
            "knowledge_processing": {
                "job_id": job_id,
                "status": "queued",
            }
        }
    else:
        result = process_document_sync(
            content=content,
            filename=filename,
            workspace_id=workspace_id,
            document_id=document_id,
        )
        return {
            "knowledge_processing": {
                "success": result.success,
                "chunks": result.chunk_count,
                "facts": result.fact_count,
                "embedded": result.embedded_count,
                "error": result.error,
            }
        }
