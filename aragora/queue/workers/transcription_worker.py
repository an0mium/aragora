"""
Transcription Job Queue Worker.

Processes transcription jobs from the durable queue, enabling:
- Restart recovery after server crashes
- Retry logic on transient failures
- Priority-based scheduling

Usage:
    from aragora.queue.workers.transcription_worker import TranscriptionWorker

    worker = TranscriptionWorker()
    await worker.start()  # Starts processing loop
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

from aragora.storage.job_queue_store import (
    QueuedJob,
    get_job_store,
)

logger = logging.getLogger(__name__)

# Job type constants
JOB_TYPE_TRANSCRIPTION = "transcription"
JOB_TYPE_TRANSCRIPTION_AUDIO = "transcription_audio"
JOB_TYPE_TRANSCRIPTION_VIDEO = "transcription_video"
JOB_TYPE_TRANSCRIPTION_YOUTUBE = "transcription_youtube"


class TranscriptionWorker:
    """
    Worker that processes transcription jobs from the durable queue.

    Features:
    - Polls job queue for pending transcription jobs
    - Executes transcription with progress tracking
    - Handles failures with automatic retry
    - Supports graceful shutdown
    """

    def __init__(
        self,
        worker_id: Optional[str] = None,
        poll_interval: float = 2.0,
        max_concurrent: int = 2,
        broadcast_fn: Optional[Callable[..., Any]] = None,
    ):
        """
        Initialize transcription worker.

        Args:
            worker_id: Unique worker identifier (auto-generated if not provided)
            poll_interval: Seconds between queue polls when idle
            max_concurrent: Maximum concurrent transcription jobs
            broadcast_fn: Optional WebSocket broadcast function for streaming
        """
        self.worker_id = worker_id or f"transcription-worker-{os.getpid()}"
        self.poll_interval = poll_interval
        self.max_concurrent = max_concurrent
        self.broadcast_fn = broadcast_fn

        self._running = False
        self._active_jobs: dict[str, asyncio.Task] = {}
        self._store = get_job_store()

    async def start(self) -> None:
        """Start the worker processing loop."""
        self._running = True
        logger.info(f"[{self.worker_id}] Starting transcription worker")

        job_types = [
            JOB_TYPE_TRANSCRIPTION,
            JOB_TYPE_TRANSCRIPTION_AUDIO,
            JOB_TYPE_TRANSCRIPTION_VIDEO,
            JOB_TYPE_TRANSCRIPTION_YOUTUBE,
        ]

        while self._running:
            try:
                # Clean up completed tasks
                self._cleanup_completed_tasks()

                # Check if we can take more jobs
                if len(self._active_jobs) >= self.max_concurrent:
                    await asyncio.sleep(0.5)
                    continue

                # Dequeue next job
                job = await self._store.dequeue(
                    worker_id=self.worker_id,
                    job_types=job_types,
                )

                if job:
                    # Start processing job
                    task = asyncio.create_task(
                        self._process_job(job),
                        name=f"transcription-{job.id}",
                    )
                    self._active_jobs[job.id] = task
                    logger.info(f"[{self.worker_id}] Started processing job {job.id}")
                else:
                    # No jobs available, wait before polling again
                    await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                logger.info(f"[{self.worker_id}] Worker cancelled")
                break
            except Exception as e:
                logger.error(f"[{self.worker_id}] Worker error: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)

        # Wait for active jobs to complete on shutdown
        if self._active_jobs:
            logger.info(f"[{self.worker_id}] Waiting for {len(self._active_jobs)} active jobs")
            await asyncio.gather(*self._active_jobs.values(), return_exceptions=True)

        logger.info(f"[{self.worker_id}] Worker stopped")

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        logger.info(f"[{self.worker_id}] Stopping worker")
        self._running = False

    def _cleanup_completed_tasks(self) -> None:
        """Remove completed tasks from tracking dict."""
        completed = [job_id for job_id, task in self._active_jobs.items() if task.done()]
        for job_id in completed:
            task = self._active_jobs.pop(job_id)
            if task.exception():
                logger.warning(f"[{self.worker_id}] Job {job_id} failed: {task.exception()}")

    async def _process_job(self, job: QueuedJob) -> None:
        """Process a single transcription job."""
        start_time = time.time()

        try:
            job_type = job.job_type
            logger.info(f"[{self.worker_id}] Processing {job_type} job {job.id}")

            # Route to appropriate handler
            if job_type in (JOB_TYPE_TRANSCRIPTION_AUDIO, JOB_TYPE_TRANSCRIPTION):
                result = await self._process_audio_job(job)
            elif job_type == JOB_TYPE_TRANSCRIPTION_VIDEO:
                result = await self._process_video_job(job)
            elif job_type == JOB_TYPE_TRANSCRIPTION_YOUTUBE:
                result = await self._process_youtube_job(job)
            else:
                raise ValueError(f"Unknown job type: {job_type}")

            # Mark job as completed
            duration = time.time() - start_time
            await self._store.complete(
                job.id,
                result={
                    **result,
                    "duration_seconds": duration,
                },
            )

            logger.info(f"[{self.worker_id}] Completed job {job.id} in {duration:.1f}s")

        except Exception as e:
            logger.error(
                f"[{self.worker_id}] Job {job.id} failed: {e}",
                exc_info=True,
            )

            # Check if we should retry
            should_retry = job.attempts < job.max_attempts
            await self._store.fail(
                job.id,
                error=str(e),
                should_retry=should_retry,
            )

            if should_retry:
                logger.info(
                    f"[{self.worker_id}] Job {job.id} will retry "
                    f"(attempt {job.attempts}/{job.max_attempts})"
                )

    async def _process_audio_job(self, job: QueuedJob) -> dict:
        """Process an audio transcription job."""
        from aragora.transcription import transcribe_audio

        payload = job.payload
        file_path = payload.get("file_path")
        file_data = payload.get("file_data")  # Base64 encoded if provided
        language = payload.get("language")
        backend = payload.get("backend")

        temp_path = None
        try:
            # Get or create temp file
            if file_path and Path(file_path).exists():
                audio_path = Path(file_path)
            elif file_data:
                import base64

                suffix = payload.get("file_extension", ".mp3")
                temp_path = Path(tempfile.mktemp(suffix=suffix))
                temp_path.write_bytes(base64.b64decode(file_data))
                audio_path = temp_path
            else:
                raise ValueError("No audio file provided")

            # Run transcription
            result = await transcribe_audio(
                audio_path,
                language=language,
                backend=backend,
            )

            return {
                "status": "completed",
                "text": result.text,
                "language": result.language,
                "duration": result.duration,
                "segments": [
                    {"start": s.start, "end": s.end, "text": s.text} for s in result.segments
                ],
                "backend": result.backend,
                "processing_time": result.processing_time,
            }

        finally:
            # Cleanup temp file
            if temp_path and temp_path.exists():
                temp_path.unlink()

    async def _process_video_job(self, job: QueuedJob) -> dict:
        """Process a video transcription job."""
        from aragora.transcription import transcribe_video

        payload = job.payload
        file_path = payload.get("file_path")
        file_data = payload.get("file_data")
        language = payload.get("language")
        backend = payload.get("backend")

        temp_path = None
        try:
            if file_path and Path(file_path).exists():
                video_path = Path(file_path)
            elif file_data:
                import base64

                suffix = payload.get("file_extension", ".mp4")
                temp_path = Path(tempfile.mktemp(suffix=suffix))
                temp_path.write_bytes(base64.b64decode(file_data))
                video_path = temp_path
            else:
                raise ValueError("No video file provided")

            result = await transcribe_video(
                video_path,
                language=language,
                backend=backend,
            )

            return {
                "status": "completed",
                "text": result.text,
                "language": result.language,
                "duration": result.duration,
                "segments": [
                    {"start": s.start, "end": s.end, "text": s.text} for s in result.segments
                ],
                "backend": result.backend,
                "processing_time": result.processing_time,
            }

        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink()

    async def _process_youtube_job(self, job: QueuedJob) -> dict:
        """Process a YouTube transcription job."""
        from aragora.transcription import transcribe_youtube

        payload = job.payload
        url = payload.get("url")
        if not url:
            raise ValueError("No YouTube URL provided")

        language = payload.get("language")
        backend = payload.get("backend")
        use_cache = payload.get("use_cache", True)

        result = await transcribe_youtube(
            url,
            language=language,
            backend=backend,
            use_cache=use_cache,
        )

        return {
            "status": "completed",
            "text": result.text,
            "language": result.language,
            "duration": result.duration,
            "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in result.segments],
            "backend": result.backend,
            "processing_time": result.processing_time,
        }


async def enqueue_transcription_job(
    job_id: str,
    job_type: str,
    payload: dict,
    user_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    priority: int = 0,
) -> QueuedJob:
    """
    Enqueue a transcription job for durable processing.

    Args:
        job_id: Unique job identifier
        job_type: Type of transcription job
        payload: Job payload (file_path, url, options, etc.)
        user_id: Optional user ID
        workspace_id: Optional workspace ID
        priority: Job priority (higher = more urgent)

    Returns:
        The queued job
    """
    job = QueuedJob(
        id=job_id,
        job_type=job_type,
        payload=payload,
        priority=priority,
        user_id=user_id,
        workspace_id=workspace_id,
    )

    store = get_job_store()
    await store.enqueue(job)

    logger.info(f"Enqueued transcription job: {job_id} ({job_type})")
    return job


async def recover_interrupted_transcriptions() -> int:
    """
    Recover interrupted transcription jobs after server restart.

    Returns:
        Number of jobs recovered
    """
    store = get_job_store()
    recovered = 0

    try:
        # Recover any stale jobs in the job queue
        job_types = [
            JOB_TYPE_TRANSCRIPTION,
            JOB_TYPE_TRANSCRIPTION_AUDIO,
            JOB_TYPE_TRANSCRIPTION_VIDEO,
            JOB_TYPE_TRANSCRIPTION_YOUTUBE,
        ]

        for job_type in job_types:
            stale_recovered = await store.recover_stale_jobs(  # type: ignore[call-arg]
                stale_threshold_seconds=300.0,
                job_types=[job_type],
            )
            if stale_recovered:
                logger.info(f"Recovered {stale_recovered} stale {job_type} jobs")
                recovered += stale_recovered

    except Exception as e:
        logger.error(f"Error recovering transcription jobs: {e}", exc_info=True)

    if recovered:
        logger.info(f"Recovered {recovered} interrupted transcription jobs")

    return recovered


__all__ = [
    "TranscriptionWorker",
    "enqueue_transcription_job",
    "recover_interrupted_transcriptions",
    "JOB_TYPE_TRANSCRIPTION",
    "JOB_TYPE_TRANSCRIPTION_AUDIO",
    "JOB_TYPE_TRANSCRIPTION_VIDEO",
    "JOB_TYPE_TRANSCRIPTION_YOUTUBE",
]
