"""
Export format operations handler mixin.

Extracted from handler.py for modularity. Provides export formatting methods
for debates in various formats (CSV, HTML, TXT, MD, LaTeX).

Also provides batch export with SSE progress streaming.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional, Protocol

from ..base import (
    HandlerResult,
    error_response,
    json_response,
    require_storage,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# =============================================================================
# Batch Export Types
# =============================================================================


class BatchExportStatus(Enum):
    """Status of a batch export job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchExportItem:
    """Single item in a batch export."""

    debate_id: str
    format: str
    status: BatchExportStatus = BatchExportStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass
class BatchExportJob:
    """Batch export job tracking."""

    job_id: str
    items: List[BatchExportItem]
    status: BatchExportStatus = BatchExportStatus.PENDING
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    processed_count: int = 0
    success_count: int = 0
    error_count: int = 0

    @property
    def total_count(self) -> int:
        return len(self.items)

    @property
    def progress_percent(self) -> float:
        if self.total_count == 0:
            return 100.0
        return (self.processed_count / self.total_count) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "total_count": self.total_count,
            "processed_count": self.processed_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "progress_percent": round(self.progress_percent, 1),
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


# In-memory job storage (production would use Redis)
_batch_export_jobs: Dict[str, BatchExportJob] = {}
_batch_export_events: Dict[str, asyncio.Queue] = {}


class _DebatesHandlerProtocol(Protocol):
    """Protocol defining the interface expected by ExportOperationsMixin.

    This protocol enables proper type checking for mixin classes that
    expect to be mixed into a class providing these methods/attributes.
    """

    ctx: Dict[str, Any]

    def get_storage(self) -> Optional[Any]:
        """Get debate storage instance."""
        ...


class ExportOperationsMixin:
    """Mixin providing export formatting operations for DebatesHandler."""

    def _start_batch_export(
        self: _DebatesHandlerProtocol,
        handler: Any,
        debate_ids: List[str],
        format: str,
    ) -> HandlerResult:
        """
        Start a batch export job.

        POST body:
        {
            "debate_ids": ["id1", "id2", ...],
            "format": "json" | "csv" | "md" | "html" | "txt"
        }

        Returns:
        {
            "job_id": "export_abc123",
            "total_count": 10,
            "status": "pending",
            "stream_url": "/api/debates/export/batch/export_abc123/stream"
        }
        """
        valid_formats = {"json", "csv", "html", "txt", "md"}
        if format not in valid_formats:
            return error_response(f"Invalid format: {format}. Valid: {valid_formats}", 400)

        if not debate_ids:
            return error_response("debate_ids is required and cannot be empty", 400)

        if len(debate_ids) > 100:
            return error_response("Batch cannot exceed 100 debates", 400)

        # Create job
        job_id = f"export_{uuid.uuid4().hex[:12]}"
        items = [BatchExportItem(debate_id=did, format=format) for did in debate_ids]
        job = BatchExportJob(job_id=job_id, items=items)
        _batch_export_jobs[job_id] = job

        # Create event queue for SSE streaming
        _batch_export_events[job_id] = asyncio.Queue()

        # Start processing in background
        asyncio.create_task(self._process_batch_export(job))

        logger.info(f"Batch export {job_id} started with {len(items)} items")

        return json_response(
            {
                "job_id": job_id,
                "total_count": len(items),
                "status": job.status.value,
                "stream_url": f"/api/debates/export/batch/{job_id}/stream",
                "status_url": f"/api/debates/export/batch/{job_id}/status",
            }
        )

    async def _process_batch_export(self: _DebatesHandlerProtocol, job: BatchExportJob) -> None:
        """Process batch export items and emit progress events."""
        job.status = BatchExportStatus.PROCESSING
        await self._emit_export_event(job.job_id, "started", job.to_dict())

        storage = self.get_storage()
        if not storage:
            job.status = BatchExportStatus.FAILED
            await self._emit_export_event(
                job.job_id,
                "error",
                {
                    "message": "Storage not available",
                },
            )
            return

        for i, item in enumerate(job.items):
            try:
                item.status = BatchExportStatus.PROCESSING
                item.started_at = time.time()

                # Export the debate
                debate = storage.get_debate(item.debate_id)
                if not debate:
                    item.status = BatchExportStatus.FAILED
                    item.error = f"Debate not found: {item.debate_id}"
                    job.error_count += 1
                else:
                    # Generate export content
                    content = self._generate_export_content(debate, item.format)
                    item.result = content
                    item.status = BatchExportStatus.COMPLETED
                    job.success_count += 1

                item.completed_at = time.time()

            except Exception as e:
                item.status = BatchExportStatus.FAILED
                item.error = str(e)
                item.completed_at = time.time()
                job.error_count += 1
                logger.warning(f"Batch export item failed: {item.debate_id}: {e}")

            job.processed_count += 1

            # Emit progress event
            await self._emit_export_event(
                job.job_id,
                "progress",
                {
                    "debate_id": item.debate_id,
                    "index": i + 1,
                    "total": job.total_count,
                    "percent": job.progress_percent,
                    "status": item.status.value,
                    "error": item.error,
                },
            )

            # Small delay to prevent overwhelming clients
            await asyncio.sleep(0.05)

        # Mark job complete
        job.status = BatchExportStatus.COMPLETED
        job.completed_at = time.time()

        await self._emit_export_event(job.job_id, "completed", job.to_dict())

    def _generate_export_content(self: _DebatesHandlerProtocol, debate: dict, format: str) -> str:
        """Generate export content for a debate."""
        from aragora.server.debate_export import (
            format_debate_csv,
            format_debate_html,
            format_debate_md,
            format_debate_txt,
        )

        if format == "json":
            return json.dumps(debate, indent=2, default=str)
        elif format == "csv":
            result = format_debate_csv(debate, "messages")
            return result.content
        elif format == "html":
            result = format_debate_html(debate)
            return result.content
        elif format == "txt":
            result = format_debate_txt(debate)
            return result.content
        elif format == "md":
            result = format_debate_md(debate)
            return result.content
        else:
            return json.dumps(debate, indent=2, default=str)

    async def _emit_export_event(
        self: _DebatesHandlerProtocol, job_id: str, event_type: str, data: Dict[str, Any]
    ) -> None:
        """Emit an event to all connected SSE clients."""
        queue = _batch_export_events.get(job_id)
        if queue:
            await queue.put(
                {
                    "type": event_type,
                    "timestamp": time.time(),
                    **data,
                }
            )

    def _get_batch_export_status(self: _DebatesHandlerProtocol, job_id: str) -> HandlerResult:
        """Get status of a batch export job."""
        job = _batch_export_jobs.get(job_id)
        if not job:
            return error_response(f"Export job not found: {job_id}", 404)

        items_summary = []
        for item in job.items:
            items_summary.append(
                {
                    "debate_id": item.debate_id,
                    "status": item.status.value,
                    "error": item.error,
                    "has_result": item.result is not None,
                }
            )

        return json_response(
            {
                **job.to_dict(),
                "items": items_summary,
            }
        )

    def _get_batch_export_results(self: _DebatesHandlerProtocol, job_id: str) -> HandlerResult:
        """Get results of a completed batch export."""
        job = _batch_export_jobs.get(job_id)
        if not job:
            return error_response(f"Export job not found: {job_id}", 404)

        if job.status != BatchExportStatus.COMPLETED:
            return error_response(f"Export job not complete (status: {job.status.value})", 400)

        results = []
        for item in job.items:
            results.append(
                {
                    "debate_id": item.debate_id,
                    "format": item.format,
                    "status": item.status.value,
                    "content": item.result,
                    "error": item.error,
                }
            )

        return json_response(
            {
                "job_id": job_id,
                "status": job.status.value,
                "results": results,
            }
        )

    async def _stream_batch_export_progress(
        self: _DebatesHandlerProtocol, job_id: str
    ) -> AsyncGenerator[str, None]:
        """Stream batch export progress via SSE."""
        job = _batch_export_jobs.get(job_id)
        if not job:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Job not found'})}\n\n"
            return

        queue = _batch_export_events.get(job_id)
        if not queue:
            queue = asyncio.Queue()
            _batch_export_events[job_id] = queue

        # Send initial status
        yield f"data: {json.dumps({'type': 'connected', 'job_id': job_id, **job.to_dict()})}\n\n"

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(event)}\n\n"

                    # Stop streaming after completion
                    if event.get("type") in ("completed", "failed", "cancelled"):
                        break

                except asyncio.TimeoutError:
                    # Send keepalive
                    yield ": keepalive\n\n"

                    # Check if job is already complete
                    if job.status in (
                        BatchExportStatus.COMPLETED,
                        BatchExportStatus.FAILED,
                        BatchExportStatus.CANCELLED,
                    ):
                        yield f"data: {json.dumps({'type': 'completed', **job.to_dict()})}\n\n"
                        break

        except Exception as e:
            logger.warning(f"SSE stream error for {job_id}: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    def _list_batch_exports(self: _DebatesHandlerProtocol, limit: int = 50) -> HandlerResult:
        """List batch export jobs."""
        jobs = sorted(
            _batch_export_jobs.values(),
            key=lambda j: j.created_at,
            reverse=True,
        )[:limit]

        return json_response(
            {
                "jobs": [job.to_dict() for job in jobs],
                "count": len(jobs),
            }
        )

    @require_storage
    def _export_debate(
        self: _DebatesHandlerProtocol,
        handler: Any,
        debate_id: str,
        format: str,
        table: str,
    ) -> HandlerResult:
        """Export debate in specified format."""
        from aragora.exceptions import (
            DatabaseError,
            RecordNotFoundError,
            StorageError,
        )

        valid_formats = {"json", "csv", "html", "txt", "md"}
        if format not in valid_formats:
            return error_response(f"Invalid format: {format}. Valid: {valid_formats}", 400)

        storage = self.get_storage()
        try:
            debate = storage.get_debate(debate_id)
            if not debate:
                return error_response(f"Debate not found: {debate_id}", 404)

            if format == "json":
                return json_response(debate)
            elif format == "csv":
                return _format_csv(debate, table)
            elif format == "txt":
                return _format_txt(debate)
            elif format == "md":
                return _format_md(debate)
            elif format in ("latex", "tex"):
                return _format_latex(debate)
            else:  # format == "html"
                return _format_html(debate)

        except RecordNotFoundError:
            logger.info("Export failed - debate not found: %s", debate_id)
            return error_response(f"Debate not found: {debate_id}", 404)
        except (StorageError, DatabaseError) as e:
            logger.error(
                "Export failed for %s (format=%s): %s: %s",
                debate_id,
                format,
                type(e).__name__,
                e,
                exc_info=True,
            )
            return error_response("Database error during export", 500)
        except ValueError as e:
            logger.warning("Export failed for %s - invalid format: %s", debate_id, e)
            return error_response(f"Invalid export format: {e}", 400)


def _format_csv(debate: dict, table: str) -> HandlerResult:
    """Format debate as CSV for the specified table type."""
    from aragora.server.debate_export import format_debate_csv

    result = format_debate_csv(debate, table)
    return HandlerResult(
        status_code=200,
        content_type=result.content_type,
        body=result.content,
        headers={"Content-Disposition": f'attachment; filename="{result.filename}"'},
    )


def _format_html(debate: dict) -> HandlerResult:
    """Format debate as standalone HTML page."""
    from aragora.server.debate_export import format_debate_html

    result = format_debate_html(debate)
    return HandlerResult(
        status_code=200,
        content_type=result.content_type,
        body=result.content,
        headers={"Content-Disposition": f'attachment; filename="{result.filename}"'},
    )


def _format_txt(debate: dict) -> HandlerResult:
    """Format debate as plain text transcript."""
    from aragora.server.debate_export import format_debate_txt

    result = format_debate_txt(debate)
    return HandlerResult(
        status_code=200,
        content_type=result.content_type,
        body=result.content,
        headers={"Content-Disposition": f'attachment; filename="{result.filename}"'},
    )


def _format_md(debate: dict) -> HandlerResult:
    """Format debate as Markdown transcript."""
    from aragora.server.debate_export import format_debate_md

    result = format_debate_md(debate)
    return HandlerResult(
        status_code=200,
        content_type=result.content_type,
        body=result.content,
        headers={"Content-Disposition": f'attachment; filename="{result.filename}"'},
    )


def _format_latex(debate: dict) -> HandlerResult:
    """Format debate as LaTeX document."""
    from aragora.server.debate_export import format_debate_latex

    result = format_debate_latex(debate)
    return HandlerResult(
        status_code=200,
        content_type=result.content_type,
        body=result.content,
        headers={"Content-Disposition": f'attachment; filename="{result.filename}"'},
    )


__all__ = ["ExportOperationsMixin"]
