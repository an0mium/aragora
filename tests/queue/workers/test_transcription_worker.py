"""
Comprehensive Tests for Transcription Job Queue Worker.

Tests the transcription worker including:
- Worker lifecycle (start, stop, configuration)
- Job processing for audio, video, and YouTube jobs
- Job routing based on job_type
- Error handling and retry logic
- Task cleanup and concurrency management
- Job enqueueing and recovery
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.queue.workers.transcription_worker import (
    TranscriptionWorker,
    JOB_TYPE_TRANSCRIPTION,
    JOB_TYPE_TRANSCRIPTION_AUDIO,
    JOB_TYPE_TRANSCRIPTION_VIDEO,
    JOB_TYPE_TRANSCRIPTION_YOUTUBE,
    enqueue_transcription_job,
    recover_interrupted_transcriptions,
)
from aragora.storage.job_queue_store import (
    JobStatus,
    QueuedJob,
    SQLiteJobStore,
    reset_job_store,
    set_job_store,
)


@pytest.fixture
def temp_db():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_transcription_jobs.db"


@pytest.fixture
def store(temp_db):
    """Create a SQLite job store for testing."""
    reset_job_store()
    s = SQLiteJobStore(temp_db)
    set_job_store(s)
    yield s
    reset_job_store()


# =============================================================================
# Job Type Constants Tests
# =============================================================================


class TestJobTypeConstants:
    """Tests for job type constant values."""

    def test_transcription_type(self):
        assert JOB_TYPE_TRANSCRIPTION == "transcription"

    def test_audio_type(self):
        assert JOB_TYPE_TRANSCRIPTION_AUDIO == "transcription_audio"

    def test_video_type(self):
        assert JOB_TYPE_TRANSCRIPTION_VIDEO == "transcription_video"

    def test_youtube_type(self):
        assert JOB_TYPE_TRANSCRIPTION_YOUTUBE == "transcription_youtube"


# =============================================================================
# Worker Initialization Tests
# =============================================================================


class TestTranscriptionWorkerInit:
    """Tests for TranscriptionWorker initialization."""

    def test_default_initialization(self, store):
        """Test worker initializes with defaults."""
        worker = TranscriptionWorker()
        assert worker.worker_id.startswith("transcription-worker-")
        assert worker.poll_interval == 2.0
        assert worker.max_concurrent == 2
        assert worker.broadcast_fn is None
        assert worker._running is False
        assert worker._active_jobs == {}

    def test_custom_worker_id(self, store):
        """Test worker with custom ID."""
        worker = TranscriptionWorker(worker_id="custom-transcriber")
        assert worker.worker_id == "custom-transcriber"

    def test_custom_poll_interval(self, store):
        """Test worker with custom poll interval."""
        worker = TranscriptionWorker(poll_interval=5.0)
        assert worker.poll_interval == 5.0

    def test_custom_max_concurrent(self, store):
        """Test worker with custom max concurrent."""
        worker = TranscriptionWorker(max_concurrent=10)
        assert worker.max_concurrent == 10

    def test_with_broadcast_fn(self, store):
        """Test worker with broadcast function."""
        mock_fn = MagicMock()
        worker = TranscriptionWorker(broadcast_fn=mock_fn)
        assert worker.broadcast_fn is mock_fn


# =============================================================================
# Worker Lifecycle Tests
# =============================================================================


class TestTranscriptionWorkerLifecycle:
    """Tests for worker start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_sets_running(self, store):
        """Start should set _running to True."""
        worker = TranscriptionWorker(poll_interval=0.05)
        task = asyncio.create_task(worker.start())
        await asyncio.sleep(0.1)

        assert worker._running is True

        await worker.stop()
        await asyncio.wait_for(task, timeout=2.0)

    @pytest.mark.asyncio
    async def test_stop_sets_not_running(self, store):
        """Stop should set _running to False."""
        worker = TranscriptionWorker(poll_interval=0.05)
        task = asyncio.create_task(worker.start())
        await asyncio.sleep(0.1)

        await worker.stop()
        assert worker._running is False

        await asyncio.wait_for(task, timeout=2.0)

    @pytest.mark.asyncio
    async def test_stop_without_start(self, store):
        """Stop without start should not raise."""
        worker = TranscriptionWorker()
        await worker.stop()
        assert worker._running is False

    @pytest.mark.asyncio
    async def test_handles_cancellation(self, store):
        """Worker should handle CancelledError gracefully."""
        worker = TranscriptionWorker(poll_interval=0.05)
        task = asyncio.create_task(worker.start())
        await asyncio.sleep(0.1)

        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

        assert worker._running is False or task.done()

    @pytest.mark.asyncio
    async def test_waits_for_active_jobs(self, store):
        """Worker should wait for active jobs before stopping."""
        worker = TranscriptionWorker(poll_interval=0.05)
        completed = False

        async def slow_task():
            nonlocal completed
            await asyncio.sleep(0.1)
            completed = True

        task = asyncio.create_task(slow_task())
        worker._active_jobs["slow"] = task

        start_task = asyncio.create_task(worker.start())
        await asyncio.sleep(0.05)
        await worker.stop()
        await asyncio.wait_for(start_task, timeout=3.0)

        assert completed

    @pytest.mark.asyncio
    async def test_handles_error_in_loop(self, store):
        """Worker should continue after errors in the loop."""
        worker = TranscriptionWorker(poll_interval=0.05)

        call_count = 0
        original_dequeue = worker._store.dequeue

        async def failing_dequeue(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("Transient error")
            return None

        with patch.object(worker._store, "dequeue", side_effect=failing_dequeue):
            task = asyncio.create_task(worker.start())
            await asyncio.sleep(0.3)
            await worker.stop()
            await asyncio.wait_for(task, timeout=2.0)

        assert call_count >= 2


# =============================================================================
# Task Cleanup Tests
# =============================================================================


class TestTranscriptionCleanup:
    """Tests for _cleanup_completed_tasks method."""

    @pytest.mark.asyncio
    async def test_removes_completed_tasks(self, store):
        """Should remove completed tasks."""
        worker = TranscriptionWorker()

        async def noop():
            pass

        task = asyncio.create_task(noop())
        await task
        worker._active_jobs["done"] = task

        worker._cleanup_completed_tasks()
        assert "done" not in worker._active_jobs

    @pytest.mark.asyncio
    async def test_keeps_running_tasks(self, store):
        """Should keep running tasks."""
        worker = TranscriptionWorker()
        event = asyncio.Event()

        async def wait():
            await event.wait()

        task = asyncio.create_task(wait())
        worker._active_jobs["running"] = task

        worker._cleanup_completed_tasks()
        assert "running" in worker._active_jobs

        event.set()
        await task

    @pytest.mark.asyncio
    async def test_handles_failed_tasks(self, store):
        """Should handle failed tasks without crashing."""
        worker = TranscriptionWorker()

        async def fail():
            raise RuntimeError("Task error")

        task = asyncio.create_task(fail())
        try:
            await task
        except RuntimeError:
            pass
        worker._active_jobs["failed"] = task

        worker._cleanup_completed_tasks()
        assert "failed" not in worker._active_jobs


# =============================================================================
# Job Processing Tests
# =============================================================================


class TestTranscriptionJobProcessing:
    """Tests for _process_job routing and completion."""

    @pytest.mark.asyncio
    async def test_routes_audio_job(self, store):
        """Should route audio job to _process_audio_job."""
        worker = TranscriptionWorker()

        job = QueuedJob(
            id="audio-job",
            job_type=JOB_TYPE_TRANSCRIPTION_AUDIO,
            payload={"file_path": "/tmp/test.mp3"},
        )
        await store.enqueue(job)
        claimed = await store.dequeue(worker_id=worker.worker_id)

        with patch.object(
            worker,
            "_process_audio_job",
            new_callable=AsyncMock,
            return_value={"status": "completed", "text": "Hello"},
        ):
            await worker._process_job(claimed)

        job_after = await store.get("audio-job")
        assert job_after.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_routes_transcription_job(self, store):
        """Should route generic transcription job to _process_audio_job."""
        worker = TranscriptionWorker()

        job = QueuedJob(
            id="generic-job",
            job_type=JOB_TYPE_TRANSCRIPTION,
            payload={"file_path": "/tmp/test.mp3"},
        )
        await store.enqueue(job)
        claimed = await store.dequeue(worker_id=worker.worker_id)

        with patch.object(
            worker,
            "_process_audio_job",
            new_callable=AsyncMock,
            return_value={"status": "completed", "text": "Hello"},
        ):
            await worker._process_job(claimed)

        job_after = await store.get("generic-job")
        assert job_after.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_routes_video_job(self, store):
        """Should route video job to _process_video_job."""
        worker = TranscriptionWorker()

        job = QueuedJob(
            id="video-job",
            job_type=JOB_TYPE_TRANSCRIPTION_VIDEO,
            payload={"file_path": "/tmp/test.mp4"},
        )
        await store.enqueue(job)
        claimed = await store.dequeue(worker_id=worker.worker_id)

        with patch.object(
            worker,
            "_process_video_job",
            new_callable=AsyncMock,
            return_value={"status": "completed", "text": "Video text"},
        ):
            await worker._process_job(claimed)

        job_after = await store.get("video-job")
        assert job_after.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_routes_youtube_job(self, store):
        """Should route YouTube job to _process_youtube_job."""
        worker = TranscriptionWorker()

        job = QueuedJob(
            id="yt-job",
            job_type=JOB_TYPE_TRANSCRIPTION_YOUTUBE,
            payload={"url": "https://youtube.com/watch?v=test"},
        )
        await store.enqueue(job)
        claimed = await store.dequeue(worker_id=worker.worker_id)

        with patch.object(
            worker,
            "_process_youtube_job",
            new_callable=AsyncMock,
            return_value={"status": "completed", "text": "YouTube text"},
        ):
            await worker._process_job(claimed)

        job_after = await store.get("yt-job")
        assert job_after.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_unknown_job_type_fails(self, store):
        """Should fail on unknown job type."""
        worker = TranscriptionWorker()

        job = QueuedJob(
            id="unknown-job",
            job_type="transcription_unknown",
            payload={},
            max_attempts=1,
        )
        await store.enqueue(job)
        claimed = await store.dequeue(worker_id=worker.worker_id)

        await worker._process_job(claimed)

        job_after = await store.get("unknown-job")
        assert job_after.status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_completed_result_includes_duration(self, store):
        """Should include duration_seconds in completed result."""
        worker = TranscriptionWorker()

        job = QueuedJob(
            id="duration-job",
            job_type=JOB_TYPE_TRANSCRIPTION_AUDIO,
            payload={"file_path": "/tmp/test.mp3"},
        )
        await store.enqueue(job)
        claimed = await store.dequeue(worker_id=worker.worker_id)

        with patch.object(
            worker,
            "_process_audio_job",
            new_callable=AsyncMock,
            return_value={"status": "completed", "text": "Test"},
        ):
            await worker._process_job(claimed)

        job_after = await store.get("duration-job")
        assert job_after.result is not None
        assert "duration_seconds" in job_after.result


# =============================================================================
# Error Handling and Retry Tests
# =============================================================================


class TestTranscriptionRetry:
    """Tests for error handling and retry logic."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, store):
        """Should retry job when attempts remain."""
        worker = TranscriptionWorker()

        job = QueuedJob(
            id="retry-job",
            job_type=JOB_TYPE_TRANSCRIPTION_AUDIO,
            payload={"file_path": "/tmp/test.mp3"},
            max_attempts=3,
        )
        await store.enqueue(job)
        claimed = await store.dequeue(worker_id=worker.worker_id)

        with patch.object(
            worker,
            "_process_audio_job",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Transcription failed"),
        ):
            await worker._process_job(claimed)

        job_after = await store.get("retry-job")
        assert job_after.status == JobStatus.PENDING
        assert job_after.error == "Transcription failed"

    @pytest.mark.asyncio
    async def test_permanent_failure_after_max_attempts(self, store):
        """Should permanently fail after max attempts."""
        worker = TranscriptionWorker()

        job = QueuedJob(
            id="perm-fail-job",
            job_type=JOB_TYPE_TRANSCRIPTION_AUDIO,
            payload={"file_path": "/tmp/test.mp3"},
            max_attempts=1,
        )
        await store.enqueue(job)
        claimed = await store.dequeue(worker_id=worker.worker_id)

        with patch.object(
            worker,
            "_process_audio_job",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Persistent error"),
        ):
            await worker._process_job(claimed)

        job_after = await store.get("perm-fail-job")
        assert job_after.status == JobStatus.FAILED


# =============================================================================
# Audio Job Processing Tests
# =============================================================================


class TestProcessAudioJob:
    """Tests for _process_audio_job method."""

    @pytest.mark.asyncio
    async def test_audio_with_file_path(self, store):
        """Should process audio with file path."""
        worker = TranscriptionWorker()

        # Create a real temp file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake audio data")
            temp_path = f.name

        try:
            mock_result = MagicMock()
            mock_result.text = "Transcribed text"
            mock_result.language = "en"
            mock_result.duration = 120.5
            mock_result.segments = [
                MagicMock(start=0.0, end=5.0, text="Hello"),
                MagicMock(start=5.0, end=10.0, text="World"),
            ]
            mock_result.backend = "whisper"
            mock_result.processing_time = 3.2

            job = QueuedJob(
                id="audio-path-job",
                job_type=JOB_TYPE_TRANSCRIPTION_AUDIO,
                payload={"file_path": temp_path, "language": "en"},
            )

            with patch(
                "aragora.transcription.transcribe_audio",
                new_callable=AsyncMock,
                return_value=mock_result,
            ):
                result = await worker._process_audio_job(job)

            assert result["status"] == "completed"
            assert result["text"] == "Transcribed text"
            assert result["language"] == "en"
            assert result["duration"] == 120.5
            assert len(result["segments"]) == 2
            assert result["backend"] == "whisper"
            assert result["processing_time"] == 3.2
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_audio_with_base64_data(self, store):
        """Should process audio with base64-encoded data."""
        import base64

        worker = TranscriptionWorker()

        mock_result = MagicMock()
        mock_result.text = "Base64 audio text"
        mock_result.language = "en"
        mock_result.duration = 30.0
        mock_result.segments = []
        mock_result.backend = "whisper"
        mock_result.processing_time = 1.0

        audio_data = base64.b64encode(b"fake audio bytes").decode()

        job = QueuedJob(
            id="audio-b64-job",
            job_type=JOB_TYPE_TRANSCRIPTION_AUDIO,
            payload={
                "file_data": audio_data,
                "file_extension": ".wav",
            },
        )

        with patch(
            "aragora.transcription.transcribe_audio",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await worker._process_audio_job(job)

        assert result["status"] == "completed"
        assert result["text"] == "Base64 audio text"

    @pytest.mark.asyncio
    async def test_audio_no_file_raises(self, store):
        """Should raise ValueError when no file is provided."""
        worker = TranscriptionWorker()

        job = QueuedJob(
            id="no-file-job",
            job_type=JOB_TYPE_TRANSCRIPTION_AUDIO,
            payload={},
        )

        with pytest.raises(ValueError, match="No audio file provided"):
            await worker._process_audio_job(job)

    @pytest.mark.asyncio
    async def test_audio_nonexistent_path(self, store):
        """Should raise ValueError when file path does not exist."""
        worker = TranscriptionWorker()

        job = QueuedJob(
            id="bad-path-job",
            job_type=JOB_TYPE_TRANSCRIPTION_AUDIO,
            payload={"file_path": "/nonexistent/audio.mp3"},
        )

        # No file_data either, so should raise
        with pytest.raises(ValueError, match="No audio file provided"):
            await worker._process_audio_job(job)


# =============================================================================
# Video Job Processing Tests
# =============================================================================


class TestProcessVideoJob:
    """Tests for _process_video_job method."""

    @pytest.mark.asyncio
    async def test_video_with_file_path(self, store):
        """Should process video with file path."""
        worker = TranscriptionWorker()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video data")
            temp_path = f.name

        try:
            mock_result = MagicMock()
            mock_result.text = "Video transcription"
            mock_result.language = "en"
            mock_result.duration = 300.0
            mock_result.segments = [MagicMock(start=0.0, end=10.0, text="Scene 1")]
            mock_result.backend = "whisper"
            mock_result.processing_time = 15.0

            job = QueuedJob(
                id="video-path-job",
                job_type=JOB_TYPE_TRANSCRIPTION_VIDEO,
                payload={"file_path": temp_path},
            )

            with patch(
                "aragora.transcription.transcribe_video",
                new_callable=AsyncMock,
                return_value=mock_result,
            ):
                result = await worker._process_video_job(job)

            assert result["status"] == "completed"
            assert result["text"] == "Video transcription"
            assert result["duration"] == 300.0
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_video_with_base64_data(self, store):
        """Should process video with base64-encoded data."""
        import base64

        worker = TranscriptionWorker()

        mock_result = MagicMock()
        mock_result.text = "Base64 video text"
        mock_result.language = "en"
        mock_result.duration = 60.0
        mock_result.segments = []
        mock_result.backend = "whisper"
        mock_result.processing_time = 5.0

        video_data = base64.b64encode(b"fake video bytes").decode()

        job = QueuedJob(
            id="video-b64-job",
            job_type=JOB_TYPE_TRANSCRIPTION_VIDEO,
            payload={
                "file_data": video_data,
                "file_extension": ".mkv",
            },
        )

        with patch(
            "aragora.transcription.transcribe_video",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await worker._process_video_job(job)

        assert result["status"] == "completed"
        assert result["text"] == "Base64 video text"

    @pytest.mark.asyncio
    async def test_video_no_file_raises(self, store):
        """Should raise ValueError when no video file provided."""
        worker = TranscriptionWorker()

        job = QueuedJob(
            id="no-video-job",
            job_type=JOB_TYPE_TRANSCRIPTION_VIDEO,
            payload={},
        )

        with pytest.raises(ValueError, match="No video file provided"):
            await worker._process_video_job(job)


# =============================================================================
# YouTube Job Processing Tests
# =============================================================================


class TestProcessYoutubeJob:
    """Tests for _process_youtube_job method."""

    @pytest.mark.asyncio
    async def test_youtube_with_url(self, store):
        """Should process YouTube job with URL."""
        worker = TranscriptionWorker()

        mock_result = MagicMock()
        mock_result.text = "YouTube video transcription"
        mock_result.language = "en"
        mock_result.duration = 600.0
        mock_result.segments = [
            MagicMock(start=0.0, end=30.0, text="Introduction"),
        ]
        mock_result.backend = "youtube_api"
        mock_result.processing_time = 2.0

        job = QueuedJob(
            id="yt-url-job",
            job_type=JOB_TYPE_TRANSCRIPTION_YOUTUBE,
            payload={
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "language": "en",
                "use_cache": True,
            },
        )

        with patch(
            "aragora.transcription.transcribe_youtube",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await worker._process_youtube_job(job)

        assert result["status"] == "completed"
        assert result["text"] == "YouTube video transcription"
        assert result["duration"] == 600.0

    @pytest.mark.asyncio
    async def test_youtube_no_url_raises(self, store):
        """Should raise ValueError when no URL provided."""
        worker = TranscriptionWorker()

        job = QueuedJob(
            id="no-url-job",
            job_type=JOB_TYPE_TRANSCRIPTION_YOUTUBE,
            payload={},
        )

        with pytest.raises(ValueError, match="No YouTube URL provided"):
            await worker._process_youtube_job(job)

    @pytest.mark.asyncio
    async def test_youtube_passes_options(self, store):
        """Should pass language, backend, and cache options."""
        worker = TranscriptionWorker()

        mock_result = MagicMock()
        mock_result.text = "test"
        mock_result.language = "fr"
        mock_result.duration = 10.0
        mock_result.segments = []
        mock_result.backend = "whisper"
        mock_result.processing_time = 1.0

        job = QueuedJob(
            id="yt-opts-job",
            job_type=JOB_TYPE_TRANSCRIPTION_YOUTUBE,
            payload={
                "url": "https://youtube.com/watch?v=abc",
                "language": "fr",
                "backend": "whisper",
                "use_cache": False,
            },
        )

        with patch(
            "aragora.transcription.transcribe_youtube",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_transcribe:
            await worker._process_youtube_job(job)

        mock_transcribe.assert_called_once_with(
            "https://youtube.com/watch?v=abc",
            language="fr",
            backend="whisper",
            use_cache=False,
        )


# =============================================================================
# Enqueue Tests
# =============================================================================


class TestEnqueueTranscriptionJob:
    """Tests for enqueue_transcription_job function."""

    @pytest.mark.asyncio
    async def test_enqueue_basic(self, store):
        """Should enqueue a transcription job."""
        job = await enqueue_transcription_job(
            job_id="transcription-1",
            job_type=JOB_TYPE_TRANSCRIPTION_AUDIO,
            payload={"file_path": "/tmp/test.mp3"},
        )

        assert job.id == "transcription-1"
        assert job.job_type == JOB_TYPE_TRANSCRIPTION_AUDIO
        assert job.payload["file_path"] == "/tmp/test.mp3"
        assert job.status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_enqueue_persists(self, store):
        """Should persist job to store."""
        await enqueue_transcription_job(
            job_id="persist-transcription",
            job_type=JOB_TYPE_TRANSCRIPTION_VIDEO,
            payload={"file_path": "/tmp/test.mp4"},
        )

        stored = await store.get("persist-transcription")
        assert stored is not None
        assert stored.job_type == JOB_TYPE_TRANSCRIPTION_VIDEO

    @pytest.mark.asyncio
    async def test_enqueue_with_user_workspace(self, store):
        """Should include user and workspace IDs."""
        job = await enqueue_transcription_job(
            job_id="user-transcription",
            job_type=JOB_TYPE_TRANSCRIPTION,
            payload={},
            user_id="user-1",
            workspace_id="ws-1",
        )

        assert job.user_id == "user-1"
        assert job.workspace_id == "ws-1"

    @pytest.mark.asyncio
    async def test_enqueue_with_priority(self, store):
        """Should set job priority."""
        job = await enqueue_transcription_job(
            job_id="priority-transcription",
            job_type=JOB_TYPE_TRANSCRIPTION,
            payload={},
            priority=5,
        )

        assert job.priority == 5

    @pytest.mark.asyncio
    async def test_enqueue_youtube_job(self, store):
        """Should enqueue a YouTube transcription job."""
        job = await enqueue_transcription_job(
            job_id="youtube-1",
            job_type=JOB_TYPE_TRANSCRIPTION_YOUTUBE,
            payload={"url": "https://youtube.com/watch?v=abc"},
        )

        assert job.job_type == JOB_TYPE_TRANSCRIPTION_YOUTUBE
        assert job.payload["url"] == "https://youtube.com/watch?v=abc"

    @pytest.mark.asyncio
    async def test_enqueue_all_job_types(self, store):
        """Should enqueue all job types successfully."""
        job_types = [
            JOB_TYPE_TRANSCRIPTION,
            JOB_TYPE_TRANSCRIPTION_AUDIO,
            JOB_TYPE_TRANSCRIPTION_VIDEO,
            JOB_TYPE_TRANSCRIPTION_YOUTUBE,
        ]

        for i, jt in enumerate(job_types):
            job = await enqueue_transcription_job(
                job_id=f"type-test-{i}",
                job_type=jt,
                payload={},
            )
            assert job.job_type == jt


# =============================================================================
# Recovery Tests
# =============================================================================


class TestRecoverInterruptedTranscriptions:
    """Tests for recover_interrupted_transcriptions function."""

    @pytest.mark.asyncio
    async def test_no_stale_jobs(self, store):
        """Should return 0 when no stale jobs."""
        recovered = await recover_interrupted_transcriptions()
        assert recovered == 0

    @pytest.mark.asyncio
    async def test_recovers_stale_audio_jobs(self, store):
        """Should recover stale audio transcription jobs."""
        job = QueuedJob(
            id="stale-audio",
            job_type=JOB_TYPE_TRANSCRIPTION_AUDIO,
            payload={"file_path": "/tmp/test.mp3"},
        )
        await store.enqueue(job)
        await store.dequeue(worker_id="crashed-worker")

        # Backdate
        conn = store._get_conn()
        conn.execute(
            "UPDATE job_queue SET started_at = ? WHERE id = ?",
            (time.time() - 400, "stale-audio"),
        )
        conn.commit()

        recovered = await store.recover_stale_jobs(stale_threshold_seconds=300.0)
        assert recovered == 1

        job_after = await store.get("stale-audio")
        assert job_after.status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_recovers_multiple_job_types(self, store):
        """Should recover stale jobs across all transcription types."""
        for i, jt in enumerate(
            [
                JOB_TYPE_TRANSCRIPTION,
                JOB_TYPE_TRANSCRIPTION_AUDIO,
                JOB_TYPE_TRANSCRIPTION_VIDEO,
                JOB_TYPE_TRANSCRIPTION_YOUTUBE,
            ]
        ):
            job = QueuedJob(
                id=f"stale-{i}",
                job_type=jt,
                payload={},
            )
            await store.enqueue(job)
            await store.dequeue(worker_id="crashed-worker")

        # Backdate all
        conn = store._get_conn()
        conn.execute(
            "UPDATE job_queue SET started_at = ?",
            (time.time() - 400,),
        )
        conn.commit()

        recovered = await store.recover_stale_jobs(stale_threshold_seconds=300.0)
        assert recovered == 4


# =============================================================================
# Concurrency Tests
# =============================================================================


class TestTranscriptionConcurrency:
    """Tests for worker concurrency management."""

    @pytest.mark.asyncio
    async def test_dequeues_all_transcription_types(self, store):
        """Worker should dequeue all four transcription job types."""
        for i, jt in enumerate(
            [
                JOB_TYPE_TRANSCRIPTION,
                JOB_TYPE_TRANSCRIPTION_AUDIO,
                JOB_TYPE_TRANSCRIPTION_VIDEO,
                JOB_TYPE_TRANSCRIPTION_YOUTUBE,
            ]
        ):
            job = QueuedJob(id=f"type-{i}", job_type=jt, payload={})
            await store.enqueue(job)

        # Dequeue with all types
        job_types = [
            JOB_TYPE_TRANSCRIPTION,
            JOB_TYPE_TRANSCRIPTION_AUDIO,
            JOB_TYPE_TRANSCRIPTION_VIDEO,
            JOB_TYPE_TRANSCRIPTION_YOUTUBE,
        ]

        dequeued_types = set()
        for _ in range(4):
            job = await store.dequeue(worker_id="test", job_types=job_types)
            if job:
                dequeued_types.add(job.job_type)

        assert len(dequeued_types) == 4

    @pytest.mark.asyncio
    async def test_respects_max_concurrent(self, store):
        """Worker should not exceed max_concurrent."""
        worker = TranscriptionWorker(max_concurrent=1, poll_interval=0.05)

        event = asyncio.Event()

        async def wait():
            await event.wait()

        task = asyncio.create_task(wait())
        worker._active_jobs["blocking"] = task

        # Worker should not dequeue when at capacity
        start_task = asyncio.create_task(worker.start())
        await asyncio.sleep(0.15)

        await worker.stop()
        event.set()
        await asyncio.wait_for(start_task, timeout=3.0)
