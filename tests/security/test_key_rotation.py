"""
Tests for Key Rotation Scheduler.
"""

import asyncio
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, AsyncMock, patch

from aragora.security.key_rotation import (
    KeyRotationScheduler,
    KeyRotationConfig,
    KeyRotationJob,
    KeyInfo,
    RotationStatus,
    SchedulerStatus,
    get_key_rotation_scheduler,
    set_key_rotation_scheduler,
    start_key_rotation_scheduler,
    stop_key_rotation_scheduler,
)


class TestKeyRotationConfig:
    """Tests for KeyRotationConfig."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = KeyRotationConfig()

        assert config.rotation_interval_days == 90
        assert config.check_interval_hours == 6
        assert config.auto_rotate_kms_keys is True
        assert config.key_overlap_days == 7
        assert config.max_retries == 3

    def test_custom_values(self):
        """Should accept custom values."""
        config = KeyRotationConfig(
            rotation_interval_days=30,
            check_interval_hours=1,
            auto_rotate_kms_keys=False,
            key_overlap_days=14,
        )

        assert config.rotation_interval_days == 30
        assert config.check_interval_hours == 1
        assert config.auto_rotate_kms_keys is False
        assert config.key_overlap_days == 14


class TestKeyInfo:
    """Tests for KeyInfo dataclass."""

    def test_create_key_info(self):
        """Should create key info with required fields."""
        now = datetime.now(timezone.utc)
        key_info = KeyInfo(
            key_id="test-key",
            provider="aws",
            version=1,
            created_at=now,
        )

        assert key_info.key_id == "test-key"
        assert key_info.provider == "aws"
        assert key_info.version == 1
        assert key_info.is_active is True
        assert key_info.tenant_id is None

    def test_key_info_with_tenant(self):
        """Should support tenant-specific keys."""
        now = datetime.now(timezone.utc)
        key_info = KeyInfo(
            key_id="tenant-abc-key",
            provider="vault",
            version=3,
            created_at=now,
            tenant_id="tenant-abc",
            last_rotated_at=now - timedelta(days=30),
        )

        assert key_info.tenant_id == "tenant-abc"
        assert key_info.last_rotated_at is not None


class TestKeyRotationJob:
    """Tests for KeyRotationJob dataclass."""

    def test_create_job(self):
        """Should create rotation job."""
        job = KeyRotationJob(
            id="job-123",
            key_id="test-key",
            provider="aws",
        )

        assert job.id == "job-123"
        assert job.key_id == "test-key"
        assert job.status == RotationStatus.PENDING
        assert job.retries == 0

    def test_job_to_dict(self):
        """Should serialize to dictionary."""
        job = KeyRotationJob(
            id="job-123",
            key_id="test-key",
            provider="vault",
            old_version=1,
            new_version=2,
            status=RotationStatus.COMPLETED,
        )

        data = job.to_dict()

        assert data["id"] == "job-123"
        assert data["key_id"] == "test-key"
        assert data["provider"] == "vault"
        assert data["old_version"] == 1
        assert data["new_version"] == 2
        assert data["status"] == "completed"


class TestKeyRotationScheduler:
    """Tests for KeyRotationScheduler."""

    @pytest.fixture
    def scheduler(self):
        """Create a scheduler for testing."""
        config = KeyRotationConfig(
            rotation_interval_days=90,
            check_interval_hours=6,
        )
        return KeyRotationScheduler(config=config)

    def test_initial_status(self, scheduler):
        """Should start in stopped state."""
        assert scheduler.status == SchedulerStatus.STOPPED

    @pytest.mark.asyncio
    async def test_start_stop(self, scheduler):
        """Should start and stop correctly."""
        await scheduler.start()
        assert scheduler.status == SchedulerStatus.RUNNING

        await scheduler.stop()
        assert scheduler.status == SchedulerStatus.STOPPED

    @pytest.mark.asyncio
    async def test_pause_resume(self, scheduler):
        """Should pause and resume correctly."""
        await scheduler.start()

        await scheduler.pause()
        assert scheduler.status == SchedulerStatus.PAUSED

        await scheduler.resume()
        assert scheduler.status == SchedulerStatus.RUNNING

        await scheduler.stop()

    def test_track_key(self, scheduler):
        """Should track keys for rotation."""
        now = datetime.now(timezone.utc)
        key_info = KeyInfo(
            key_id="test-key",
            provider="local",
            version=1,
            created_at=now,
        )

        scheduler.track_key(key_info)

        tracked = scheduler.get_tracked_keys()
        assert len(tracked) == 1
        assert tracked[0].key_id == "test-key"
        assert tracked[0].next_rotation_at is not None

    def test_untrack_key(self, scheduler):
        """Should remove key from tracking."""
        now = datetime.now(timezone.utc)
        key_info = KeyInfo(
            key_id="test-key",
            provider="local",
            version=1,
            created_at=now,
        )

        scheduler.track_key(key_info)
        assert len(scheduler.get_tracked_keys()) == 1

        scheduler.untrack_key("test-key")
        assert len(scheduler.get_tracked_keys()) == 0

    def test_stats(self, scheduler):
        """Should provide scheduler statistics."""
        stats = scheduler.get_stats()

        assert stats.status == SchedulerStatus.STOPPED
        assert stats.total_rotations == 0
        assert stats.keys_tracked == 0

    @pytest.mark.asyncio
    async def test_rotate_now_local(self, scheduler):
        """Should execute immediate rotation for local keys."""
        with patch(
            "aragora.security.key_rotation.KeyRotationScheduler._rotate_local_key"
        ) as mock_rotate:
            mock_rotate.return_value = None

            job = await scheduler.rotate_now(key_id="test-key")

            assert job.key_id == "test-key"
            # The job should have been processed (either completed or attempted)
            assert job.started_at is not None

    @pytest.mark.asyncio
    async def test_rotate_all_due(self, scheduler):
        """Should rotate all keys that are due."""
        now = datetime.now(timezone.utc)

        # Add a key that's due for rotation
        due_key = KeyInfo(
            key_id="due-key",
            provider="local",
            version=1,
            created_at=now - timedelta(days=100),
            last_rotated_at=now - timedelta(days=100),
        )
        scheduler.track_key(due_key)

        # Add a key that's not due yet
        not_due_key = KeyInfo(
            key_id="not-due-key",
            provider="local",
            version=1,
            created_at=now - timedelta(days=10),
            last_rotated_at=now - timedelta(days=10),
        )
        scheduler.track_key(not_due_key)

        with patch(
            "aragora.security.key_rotation.KeyRotationScheduler._rotate_local_key"
        ) as mock_rotate:
            mock_rotate.return_value = None

            jobs = await scheduler.rotate_all_due()

            # Only the due key should be rotated
            assert len(jobs) == 1
            assert jobs[0].key_id == "due-key"

    def test_job_history(self, scheduler):
        """Should maintain job history."""
        history = scheduler.get_job_history()
        assert history == []

        # History grows as jobs complete
        job = KeyRotationJob(
            id="job-1",
            key_id="test-key",
            status=RotationStatus.COMPLETED,
        )
        scheduler._job_history.append(job)

        history = scheduler.get_job_history()
        assert len(history) == 1

    @pytest.mark.asyncio
    async def test_event_callback(self, scheduler):
        """Should emit events via callback."""
        events = []

        def callback(event_type, data):
            events.append((event_type, data))

        scheduler.set_event_callback(callback)

        await scheduler.start()
        await scheduler.stop()

        # Should have start and stop events
        event_types = [e[0] for e in events]
        assert "key_rotation_scheduler_started" in event_types
        assert "key_rotation_scheduler_stopped" in event_types


class TestKeyRotationSchedulerVault:
    """Tests for Vault-specific key rotation."""

    @pytest.mark.asyncio
    async def test_rotate_vault_key(self):
        """Should rotate key via Vault provider."""
        from aragora.security.kms_provider import HashiCorpVaultProvider

        scheduler = KeyRotationScheduler()

        mock_provider = MagicMock(spec=HashiCorpVaultProvider)
        mock_provider.get_key_metadata = AsyncMock(return_value=MagicMock(version="1"))
        mock_provider.rotate_key = AsyncMock(return_value=MagicMock(version="2"))

        job = KeyRotationJob(
            id="vault-job",
            key_id="vault-key",
            provider="vault",
        )

        with patch(
            "aragora.security.kms_provider.get_kms_provider",
            return_value=mock_provider,
        ):
            # Make isinstance check pass
            with patch.object(HashiCorpVaultProvider, "__instancecheck__", return_value=True):
                await scheduler._rotate_vault_key(job)

        assert job.old_version == 1
        assert job.new_version == 2


class TestGlobalSchedulerFunctions:
    """Tests for global scheduler management functions."""

    def setup_method(self):
        """Reset global scheduler before each test."""
        set_key_rotation_scheduler(None)

    def teardown_method(self):
        """Clean up after each test."""
        set_key_rotation_scheduler(None)

    def test_get_scheduler_none(self):
        """Should return None when not initialized."""
        assert get_key_rotation_scheduler() is None

    def test_set_scheduler(self):
        """Should set global scheduler."""
        scheduler = KeyRotationScheduler()
        set_key_rotation_scheduler(scheduler)

        assert get_key_rotation_scheduler() is scheduler

    @pytest.mark.asyncio
    async def test_start_scheduler(self):
        """Should start global scheduler."""
        scheduler = await start_key_rotation_scheduler()

        assert scheduler is not None
        assert scheduler.status == SchedulerStatus.RUNNING

        await stop_key_rotation_scheduler()
        assert get_key_rotation_scheduler() is None

    @pytest.mark.asyncio
    async def test_start_scheduler_with_config(self):
        """Should start scheduler with custom config."""
        config = KeyRotationConfig(rotation_interval_days=30)
        scheduler = await start_key_rotation_scheduler(config=config)

        assert scheduler.config.rotation_interval_days == 30

        await stop_key_rotation_scheduler()


class TestKeyRotationIntegration:
    """Integration tests for key rotation with encryption service."""

    @pytest.mark.asyncio
    async def test_rotate_local_updates_encryption_service(self):
        """Should update encryption service when rotating local keys."""
        scheduler = KeyRotationScheduler()

        mock_service = MagicMock()
        mock_service.get_active_key.return_value = MagicMock(version=1)
        mock_service.rotate_key.return_value = MagicMock(key_id="default", version=2)

        with patch(
            "aragora.security.encryption.get_encryption_service",
            return_value=mock_service,
        ):
            job = KeyRotationJob(
                id="local-job",
                key_id="default",
                provider="local",
            )

            await scheduler._rotate_local_key(job)

            assert job.old_version == 1
            assert job.new_version == 2
            mock_service.rotate_key.assert_called_once()
