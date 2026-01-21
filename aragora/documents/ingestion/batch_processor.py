"""
Async batch processor for document ingestion queue.

Handles concurrent document parsing, chunking, and indexing with:
- Priority queue management
- Progress tracking and callbacks
- Error handling and retry logic
- Rate limiting for external APIs

Usage:
    from aragora.documents.ingestion.batch_processor import (
        BatchProcessor,
        DocumentJob,
        JobStatus,
    )

    processor = BatchProcessor(max_workers=4)

    # Add jobs
    job_id = await processor.submit(content, filename, workspace_id)

    # Check status
    status = await processor.get_status(job_id)

    # Get results
    result = await processor.get_result(job_id)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional, cast
from uuid import uuid4

from aragora.documents.chunking.strategies import (
    ChunkingConfig,
    ChunkingStrategyType,
    auto_select_strategy,
    get_chunking_strategy,
)
from aragora.documents.chunking.token_counter import get_token_counter
from aragora.documents.ingestion.unstructured_adapter import UnstructuredParser
from aragora.documents.models import (
    DocumentChunk,
    DocumentStatus,
    IngestedDocument,
)
from aragora.exceptions import DocumentChunkError, DocumentParseError

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status of a processing job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    CHUNKING = "chunking"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(Enum):
    """Job priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class DocumentJob:
    """A document processing job."""

    id: str = field(default_factory=lambda: str(uuid4()))

    # Input
    content: bytes = b""
    filename: str = ""
    workspace_id: str = ""
    uploaded_by: str = ""
    tags: list[str] = field(default_factory=list)

    # Configuration
    priority: JobPriority = JobPriority.NORMAL
    chunking_strategy: Optional[str] = None  # Auto-select if None
    chunk_size: int = 512
    chunk_overlap: int = 50
    enable_ocr: bool = True

    # Status
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0  # 0.0 to 1.0
    error_message: str = ""

    # Results
    document: Optional[IngestedDocument] = None
    chunks: list[DocumentChunk] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Callbacks
    on_progress: Optional[Callable[[float, str], None]] = None
    on_complete: Optional[Callable[["DocumentJob"], None]] = None
    on_error: Optional[Callable[["DocumentJob", Exception], None]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excluding callbacks and content)."""
        return {
            "id": self.id,
            "filename": self.filename,
            "workspace_id": self.workspace_id,
            "uploaded_by": self.uploaded_by,
            "tags": self.tags,
            "priority": self.priority.name,
            "chunking_strategy": self.chunking_strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "status": self.status.value,
            "progress": self.progress,
            "error_message": self.error_message,
            "document_id": self.document.id if self.document else None,
            "chunk_count": len(self.chunks),
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class BatchResult:
    """Result of batch processing."""

    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    total_documents: int
    total_chunks: int
    total_tokens: int
    duration_ms: int
    jobs: list[DocumentJob]


class BatchProcessor:
    """
    Async batch processor for document ingestion.

    Manages a queue of document processing jobs with concurrent execution,
    progress tracking, and error handling.
    """

    def __init__(
        self,
        max_workers: int = 4,
        max_queue_size: int = 1000,
        default_chunk_size: int = 512,
        default_chunk_overlap: int = 50,
    ):
        """
        Initialize batch processor.

        Args:
            max_workers: Maximum concurrent processing tasks
            max_queue_size: Maximum queue size
            default_chunk_size: Default chunk size in tokens
            default_chunk_overlap: Default overlap between chunks
        """
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap

        # Job storage
        self._jobs: dict[str, DocumentJob] = {}
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._active_workers = 0
        self._running = False
        self._worker_tasks: list[asyncio.Task] = []

        # Components
        self._parser = UnstructuredParser()
        self._token_counter = get_token_counter()

        # Statistics
        self._stats = {
            "total_processed": 0,
            "total_failed": 0,
            "total_chunks": 0,
            "total_tokens": 0,
        }

    async def start(self) -> None:
        """Start the batch processor workers."""
        if self._running:
            return

        self._running = True
        logger.info(f"Starting batch processor with {self.max_workers} workers")

        # Start worker tasks
        for i in range(self.max_workers):
            task = asyncio.create_task(self._worker(i))
            self._worker_tasks.append(task)

    async def stop(self, wait: bool = True) -> None:
        """
        Stop the batch processor.

        Args:
            wait: Wait for current jobs to complete
        """
        self._running = False

        if wait:
            # Wait for queue to empty
            while not self._queue.empty():
                await asyncio.sleep(0.1)

            # Wait for active workers
            while self._active_workers > 0:
                await asyncio.sleep(0.1)

        # Cancel worker tasks
        for task in self._worker_tasks:
            task.cancel()

        self._worker_tasks.clear()
        logger.info("Batch processor stopped")

    async def submit(
        self,
        content: bytes,
        filename: str,
        workspace_id: str = "",
        uploaded_by: str = "",
        tags: Optional[list[str]] = None,
        priority: JobPriority = JobPriority.NORMAL,
        chunking_strategy: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        enable_ocr: bool = True,
        on_progress: Optional[Callable[[float, str], None]] = None,
        on_complete: Optional[Callable[[DocumentJob], None]] = None,
        on_error: Optional[Callable[[DocumentJob, Exception], None]] = None,
    ) -> str:
        """
        Submit a document for processing.

        Args:
            content: Raw file content
            filename: Original filename
            workspace_id: Workspace ID
            uploaded_by: User ID
            tags: Optional tags
            priority: Job priority
            chunking_strategy: Strategy name or None for auto
            chunk_size: Chunk size in tokens
            chunk_overlap: Overlap between chunks
            enable_ocr: Enable OCR for images
            on_progress: Progress callback
            on_complete: Completion callback
            on_error: Error callback

        Returns:
            Job ID
        """
        job = DocumentJob(
            content=content,
            filename=filename,
            workspace_id=workspace_id,
            uploaded_by=uploaded_by,
            tags=tags or [],
            priority=priority,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size or self.default_chunk_size,
            chunk_overlap=chunk_overlap or self.default_chunk_overlap,
            enable_ocr=enable_ocr,
            on_progress=on_progress,
            on_complete=on_complete,
            on_error=on_error,
        )

        self._jobs[job.id] = job

        # Add to priority queue (lower number = higher priority)
        priority_value = -priority.value  # Negate so higher priority comes first
        await self._queue.put((priority_value, job.created_at.timestamp(), job.id))

        logger.debug(f"Job {job.id} submitted: {filename}")
        return job.id

    async def submit_batch(
        self,
        files: list[tuple[bytes, str]],
        workspace_id: str = "",
        uploaded_by: str = "",
        priority: JobPriority = JobPriority.NORMAL,
    ) -> list[str]:
        """
        Submit multiple documents for processing.

        Args:
            files: List of (content, filename) tuples
            workspace_id: Workspace ID
            uploaded_by: User ID
            priority: Job priority

        Returns:
            List of job IDs
        """
        job_ids = []
        for content, filename in files:
            job_id = await self.submit(
                content=content,
                filename=filename,
                workspace_id=workspace_id,
                uploaded_by=uploaded_by,
                priority=priority,
            )
            job_ids.append(job_id)
        return job_ids

    async def get_status(self, job_id: str) -> Optional[dict[str, Any]]:
        """Get job status."""
        job = self._jobs.get(job_id)
        if not job:
            return None
        return job.to_dict()

    async def get_result(self, job_id: str) -> Optional[DocumentJob]:
        """Get job result (including document and chunks)."""
        return self._jobs.get(job_id)

    async def cancel(self, job_id: str) -> bool:
        """Cancel a queued job."""
        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.status == JobStatus.QUEUED:
            job.status = JobStatus.CANCELLED
            return True

        return False

    async def wait_for_job(
        self, job_id: str, timeout: Optional[float] = None
    ) -> Optional[DocumentJob]:
        """
        Wait for a job to complete.

        Args:
            job_id: Job ID
            timeout: Optional timeout in seconds

        Returns:
            Completed job or None if not found/timeout
        """
        start = time.monotonic()

        while True:
            job = self._jobs.get(job_id)
            if not job:
                return None

            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return job

            if timeout and (time.monotonic() - start) > timeout:
                return None

            await asyncio.sleep(0.1)

    async def wait_for_batch(
        self, job_ids: list[str], timeout: Optional[float] = None
    ) -> BatchResult:
        """
        Wait for multiple jobs to complete.

        Args:
            job_ids: List of job IDs
            timeout: Optional timeout in seconds

        Returns:
            BatchResult with all job results
        """
        start = time.monotonic()
        jobs = []

        for job_id in job_ids:
            remaining_timeout = None
            if timeout:
                elapsed = time.monotonic() - start
                remaining_timeout = max(0, timeout - elapsed)

            job = await self.wait_for_job(job_id, remaining_timeout)
            if job:
                jobs.append(job)

        # Calculate statistics
        completed = [j for j in jobs if j.status == JobStatus.COMPLETED]
        failed = [j for j in jobs if j.status == JobStatus.FAILED]

        total_chunks = sum(len(j.chunks) for j in completed)
        total_tokens = sum(c.token_count for j in completed for c in j.chunks)

        return BatchResult(
            total_jobs=len(job_ids),
            completed_jobs=len(completed),
            failed_jobs=len(failed),
            total_documents=len(completed),
            total_chunks=total_chunks,
            total_tokens=total_tokens,
            duration_ms=int((time.monotonic() - start) * 1000),
            jobs=jobs,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get processor statistics."""
        queued = sum(1 for j in self._jobs.values() if j.status == JobStatus.QUEUED)
        processing = sum(1 for j in self._jobs.values() if j.status == JobStatus.PROCESSING)

        return {
            "running": self._running,
            "max_workers": self.max_workers,
            "active_workers": self._active_workers,
            "queued_jobs": queued,
            "processing_jobs": processing,
            "total_processed": self._stats["total_processed"],
            "total_failed": self._stats["total_failed"],
            "total_chunks": self._stats["total_chunks"],
            "total_tokens": self._stats["total_tokens"],
        }

    async def _worker(self, worker_id: int) -> None:
        """Worker coroutine that processes jobs from queue."""
        logger.debug(f"Worker {worker_id} started")

        while self._running:
            try:
                # Get job from queue with timeout
                try:
                    _, _, job_id = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                job = self._jobs.get(job_id)
                if not job or job.status == JobStatus.CANCELLED:
                    self._queue.task_done()
                    continue

                self._active_workers += 1

                try:
                    await self._process_job(job)
                except Exception as e:
                    logger.exception(f"Worker {worker_id} error processing {job_id}")
                    job.status = JobStatus.FAILED
                    job.error_message = str(e)
                    self._stats["total_failed"] += 1

                    if job.on_error:
                        try:
                            job.on_error(job, e)
                        except (TypeError, ValueError, AttributeError) as callback_err:
                            logger.debug(
                                f"Job {job_id} error callback raised expected error: {callback_err}"
                            )
                        except Exception as callback_err:
                            logger.warning(
                                f"Job {job_id} error callback raised unexpected error: {callback_err}"
                            )
                finally:
                    self._active_workers -= 1
                    self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Worker {worker_id} unexpected error: {e}")

        logger.debug(f"Worker {worker_id} stopped")

    async def _process_job(self, job: DocumentJob) -> None:
        """Process a single document job."""
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.now(timezone.utc)
        job.progress = 0.0

        self._update_progress(job, 0.0, "Starting document parsing...")

        # Step 1: Parse document (30% of progress)
        try:
            parser = UnstructuredParser(enable_ocr=job.enable_ocr)
            document = parser.parse_to_document(
                content=job.content,
                filename=job.filename,
                workspace_id=job.workspace_id,
                uploaded_by=job.uploaded_by,
                tags=job.tags,
            )
        except Exception as e:
            raise DocumentParseError(
                document_id=None,
                reason=str(e),
                original_error=e,
            ) from e

        self._update_progress(job, 0.3, "Document parsed, starting chunking...")
        job.status = JobStatus.CHUNKING

        # Step 2: Chunk document (60% of progress)
        try:
            # Auto-select strategy if not specified
            strategy_name = job.chunking_strategy
            if not strategy_name:
                strategy_name = auto_select_strategy(document.text, job.filename)

            config = ChunkingConfig(
                chunk_size=job.chunk_size,
                overlap=job.chunk_overlap,
            )
            # Ensure strategy_name is a valid type (default to semantic)
            if strategy_name not in ("semantic", "sliding", "recursive", "fixed"):
                strategy_name = "semantic"
            strategy = get_chunking_strategy(
                cast(ChunkingStrategyType, strategy_name), **config.__dict__
            )

            chunks = strategy.chunk(
                text=document.text,
                document_id=document.id,
            )

            # Update document with chunking info
            document.chunk_count = len(chunks)
            document.chunk_ids = [c.id for c in chunks]
            document.chunking_strategy = strategy_name
            document.chunk_size = job.chunk_size
            document.chunk_overlap = job.chunk_overlap
            document.total_tokens = sum(c.token_count for c in chunks)

        except Exception as e:
            raise DocumentChunkError(
                document_id=document.id if document else None,
                reason=str(e),
                original_error=e,
            ) from e

        self._update_progress(job, 0.6, "Chunking complete, finalizing...")
        job.status = JobStatus.INDEXING

        # Step 3: Finalize (100% of progress)
        document.status = DocumentStatus.INDEXED
        document.processed_at = datetime.now(timezone.utc)
        document.indexed_at = datetime.now(timezone.utc)

        # Store results
        job.document = document
        job.chunks = chunks
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now(timezone.utc)
        job.progress = 1.0

        # Update stats
        self._stats["total_processed"] += 1
        self._stats["total_chunks"] += len(chunks)
        self._stats["total_tokens"] += document.total_tokens

        self._update_progress(job, 1.0, "Processing complete")

        # Call completion callback
        if job.on_complete:
            try:
                job.on_complete(job)
            except Exception as e:
                logger.warning(f"Error in completion callback: {e}")

        logger.info(
            f"Job {job.id} completed: {job.filename} -> "
            f"{len(chunks)} chunks, {document.total_tokens} tokens"
        )

    def _update_progress(self, job: DocumentJob, progress: float, message: str) -> None:
        """Update job progress and notify callback."""
        job.progress = progress

        if job.on_progress:
            try:
                job.on_progress(progress, message)
            except Exception as e:
                logger.warning(f"Error in progress callback: {e}")


# Global processor instance
_batch_processor: Optional[BatchProcessor] = None


async def get_batch_processor() -> BatchProcessor:
    """Get or create the global batch processor."""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor()
        await _batch_processor.start()
    return _batch_processor


__all__ = [
    "BatchProcessor",
    "DocumentJob",
    "JobStatus",
    "JobPriority",
    "BatchResult",
    "get_batch_processor",
]
