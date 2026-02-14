"""
Tests for secrets rotation scheduler module.

Tests cover:
- SecretType enum
- RotationStatus enum
- RotationTrigger enum
- SecretMetadata dataclass
- RotationResult dataclass
- SecretsRotationConfig dataclass
- SecretsRotationStorage class
- SecretsRotationScheduler class
- Global scheduler singleton
"""

import asyncio
import contextvars
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.scheduler.secrets_rotation_scheduler import (
    RotationResult,
    RotationStatus,
    RotationTrigger,
    SecretMetadata,
    SecretsRotationConfig,
    SecretsRotationScheduler,
    SecretsRotationStorage,
    SecretType,
    get_secrets_rotation_scheduler,
    rotate_secret,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestSecretType:
    """Tests for SecretType enum."""

    def test_has_all_secret_types(self):
        """Enum has all expected secret types."""
        assert SecretType.API_KEY.value == "api_key"
        assert SecretType.DATABASE_CREDENTIAL.value == "database_credential"
        assert SecretType.JWT_SECRET.value == "jwt_secret"
        assert SecretType.OAUTH_TOKEN.value == "oauth_token"
        assert SecretType.ENCRYPTION_KEY.value == "encryption_key"
        assert SecretType.WEBHOOK_SECRET.value == "webhook_secret"
        assert SecretType.SERVICE_ACCOUNT.value == "service_account"

    def test_secret_type_count(self):
        """Enum has exactly 7 secret types."""
        assert len(SecretType) == 7

    def test_is_string_enum(self):
        """SecretType values are strings."""
        for secret_type in SecretType:
            assert isinstance(secret_type.value, str)


class TestRotationStatus:
    """Tests for RotationStatus enum."""

    def test_has_all_statuses(self):
        """Enum has all expected statuses."""
        assert RotationStatus.SCHEDULED.value == "scheduled"
        assert RotationStatus.IN_PROGRESS.value == "in_progress"
        assert RotationStatus.VERIFYING.value == "verifying"
        assert RotationStatus.COMPLETED.value == "completed"
        assert RotationStatus.FAILED.value == "failed"
        assert RotationStatus.ROLLED_BACK.value == "rolled_back"

    def test_status_count(self):
        """Enum has exactly 6 statuses."""
        assert len(RotationStatus) == 6


class TestRotationTrigger:
    """Tests for RotationTrigger enum."""

    def test_has_all_triggers(self):
        """Enum has all expected triggers."""
        assert RotationTrigger.SCHEDULED.value == "scheduled"
        assert RotationTrigger.MANUAL.value == "manual"
        assert RotationTrigger.EXPIRATION.value == "expiration"
        assert RotationTrigger.COMPROMISE.value == "compromise"
        assert RotationTrigger.POLICY.value == "policy"

    def test_trigger_count(self):
        """Enum has exactly 5 triggers."""
        assert len(RotationTrigger) == 5


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestSecretMetadata:
    """Tests for SecretMetadata dataclass."""

    def test_create_with_required_fields(self):
        """Creates metadata with required fields only."""
        metadata = SecretMetadata(
            secret_id="secret_123",
            secret_type=SecretType.API_KEY,
            name="My API Key",
        )

        assert metadata.secret_id == "secret_123"
        assert metadata.secret_type == SecretType.API_KEY
        assert metadata.name == "My API Key"

    def test_default_values(self):
        """Default values are set correctly."""
        metadata = SecretMetadata(
            secret_id="secret_123",
            secret_type=SecretType.API_KEY,
            name="My API Key",
        )

        assert metadata.description == ""
        assert metadata.last_rotated_at is None
        assert metadata.next_rotation_at is None
        assert metadata.rotation_interval_days == 90
        assert metadata.owner is None
        assert metadata.tags == {}
        assert metadata.is_active is True

    def test_custom_values(self):
        """Custom values are set correctly."""
        now = datetime.now(timezone.utc)
        next_rotation = now + timedelta(days=30)

        metadata = SecretMetadata(
            secret_id="secret_123",
            secret_type=SecretType.DATABASE_CREDENTIAL,
            name="DB Password",
            description="Production database password",
            last_rotated_at=now,
            next_rotation_at=next_rotation,
            rotation_interval_days=30,
            owner="infra-team",
            tags={"env": "production"},
            is_active=True,
        )

        assert metadata.description == "Production database password"
        assert metadata.rotation_interval_days == 30
        assert metadata.owner == "infra-team"
        assert metadata.tags == {"env": "production"}


class TestRotationResult:
    """Tests for RotationResult dataclass."""

    def test_create_with_required_fields(self):
        """Creates result with required fields only."""
        result = RotationResult(
            rotation_id="rot_123",
            secret_id="secret_456",
            secret_type=SecretType.API_KEY,
        )

        assert result.rotation_id == "rot_123"
        assert result.secret_id == "secret_456"
        assert result.secret_type == SecretType.API_KEY

    def test_default_values(self):
        """Default values are set correctly."""
        result = RotationResult(
            rotation_id="rot_123",
            secret_id="secret_456",
            secret_type=SecretType.API_KEY,
        )

        assert result.status == RotationStatus.SCHEDULED
        assert result.trigger == RotationTrigger.SCHEDULED
        assert result.started_at is None
        assert result.completed_at is None
        assert result.initiated_by == "system"
        assert result.old_version is None
        assert result.new_version is None
        assert result.grace_period_ends is None
        assert result.duration_seconds == 0.0
        assert result.verification_passed is False
        assert result.error_message is None
        assert result.rolled_back is False
        assert result.notes == ""

    def test_to_dict(self):
        """to_dict returns proper dictionary."""
        now = datetime.now(timezone.utc)
        result = RotationResult(
            rotation_id="rot_123",
            secret_id="secret_456",
            secret_type=SecretType.API_KEY,
            status=RotationStatus.COMPLETED,
            trigger=RotationTrigger.MANUAL,
            started_at=now,
            completed_at=now + timedelta(seconds=5),
            verification_passed=True,
        )

        d = result.to_dict()

        assert d["rotation_id"] == "rot_123"
        assert d["secret_id"] == "secret_456"
        assert d["secret_type"] == "api_key"
        assert d["status"] == "completed"
        assert d["trigger"] == "manual"
        assert d["verification_passed"] is True

    def test_to_dict_with_none_values(self):
        """to_dict handles None values correctly."""
        result = RotationResult(
            rotation_id="rot_123",
            secret_id="secret_456",
            secret_type=SecretType.API_KEY,
        )

        d = result.to_dict()

        assert d["started_at"] is None
        assert d["completed_at"] is None
        assert d["grace_period_ends"] is None


class TestSecretsRotationConfig:
    """Tests for SecretsRotationConfig dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        config = SecretsRotationConfig()

        assert config.api_key_rotation_days == 90
        assert config.database_rotation_days == 90
        assert config.jwt_rotation_days == 30
        assert config.oauth_rotation_days == 30
        assert config.encryption_key_rotation_days == 365
        assert config.default_grace_period_hours == 24
        assert config.verify_after_rotation is True
        assert config.rollback_on_verification_failure is True
        assert config.notify_days_before == 7

    def test_custom_values(self):
        """Custom values are set correctly."""
        config = SecretsRotationConfig(
            api_key_rotation_days=60,
            jwt_rotation_days=14,
            default_grace_period_hours=48,
            notify_days_before=14,
        )

        assert config.api_key_rotation_days == 60
        assert config.jwt_rotation_days == 14
        assert config.default_grace_period_hours == 48
        assert config.notify_days_before == 14


# =============================================================================
# Storage Tests
# =============================================================================


class TestSecretsRotationStorage:
    """Tests for SecretsRotationStorage class."""

    @pytest.fixture
    def storage(self):
        """Create in-memory storage for testing."""
        SecretsRotationStorage._conn_var = contextvars.ContextVar(
            "secrets_rotation_conn", default=None
        )
        return SecretsRotationStorage()

    def test_init_creates_schema(self, storage):
        """Initializes with required tables."""
        conn = storage._get_conn()
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        assert "managed_secrets" in tables
        assert "rotation_history" in tables
        assert "rotation_schedule" in tables

    def test_save_and_get_secret(self, storage):
        """Saves and retrieves a secret."""
        metadata = SecretMetadata(
            secret_id="secret_123",
            secret_type=SecretType.API_KEY,
            name="Test API Key",
            description="Test description",
        )

        storage.save_secret(metadata)
        retrieved = storage.get_secret("secret_123")

        assert retrieved is not None
        assert retrieved.secret_id == "secret_123"
        assert retrieved.name == "Test API Key"

    def test_get_nonexistent_secret(self, storage):
        """Returns None for nonexistent secret."""
        result = storage.get_secret("nonexistent")

        assert result is None

    def test_update_existing_secret(self, storage):
        """Updates an existing secret."""
        metadata = SecretMetadata(
            secret_id="secret_123",
            secret_type=SecretType.API_KEY,
            name="Original Name",
        )
        storage.save_secret(metadata)

        metadata.name = "Updated Name"
        storage.save_secret(metadata)

        retrieved = storage.get_secret("secret_123")
        assert retrieved.name == "Updated Name"

    def test_get_all_secrets(self, storage):
        """Gets all secrets."""
        for i in range(3):
            storage.save_secret(
                SecretMetadata(
                    secret_id=f"secret_{i}",
                    secret_type=SecretType.API_KEY,
                    name=f"Secret {i}",
                )
            )

        secrets = storage.get_all_secrets()

        assert len(secrets) == 3

    def test_get_all_secrets_active_only(self, storage):
        """Gets only active secrets."""
        storage.save_secret(
            SecretMetadata(
                secret_id="active",
                secret_type=SecretType.API_KEY,
                name="Active",
                is_active=True,
            )
        )
        inactive = SecretMetadata(
            secret_id="inactive",
            secret_type=SecretType.API_KEY,
            name="Inactive",
            is_active=False,
        )
        storage.save_secret(inactive)

        active_secrets = storage.get_all_secrets(active_only=True)
        all_secrets = storage.get_all_secrets(active_only=False)

        assert len(active_secrets) == 1
        assert len(all_secrets) == 2

    def test_get_secrets_due_for_rotation(self, storage):
        """Gets secrets due for rotation."""
        past = datetime.now(timezone.utc) - timedelta(days=1)
        future = datetime.now(timezone.utc) + timedelta(days=30)

        storage.save_secret(
            SecretMetadata(
                secret_id="overdue",
                secret_type=SecretType.API_KEY,
                name="Overdue",
                next_rotation_at=past,
            )
        )
        storage.save_secret(
            SecretMetadata(
                secret_id="future",
                secret_type=SecretType.API_KEY,
                name="Future",
                next_rotation_at=future,
            )
        )

        due = storage.get_secrets_due_for_rotation()

        assert len(due) == 1
        assert due[0].secret_id == "overdue"

    def test_get_secrets_expiring_soon(self, storage):
        """Gets secrets expiring within N days."""
        now = datetime.now(timezone.utc)
        soon = now + timedelta(days=3)
        later = now + timedelta(days=30)

        storage.save_secret(
            SecretMetadata(
                secret_id="soon",
                secret_type=SecretType.API_KEY,
                name="Soon",
                next_rotation_at=soon,
            )
        )
        storage.save_secret(
            SecretMetadata(
                secret_id="later",
                secret_type=SecretType.API_KEY,
                name="Later",
                next_rotation_at=later,
            )
        )

        expiring = storage.get_secrets_expiring_soon(days=7)

        assert len(expiring) == 1
        assert expiring[0].secret_id == "soon"

    def test_save_and_get_rotation(self, storage):
        """Saves and retrieves rotation history."""
        rotation = RotationResult(
            rotation_id="rot_123",
            secret_id="secret_456",
            secret_type=SecretType.API_KEY,
            status=RotationStatus.COMPLETED,
            started_at=datetime.now(timezone.utc),
        )

        storage.save_rotation(rotation)
        history = storage.get_rotation_history(secret_id="secret_456")

        assert len(history) == 1
        assert history[0].rotation_id == "rot_123"

    def test_get_rotation_history_limit(self, storage):
        """Limits rotation history results."""
        for i in range(15):
            storage.save_rotation(
                RotationResult(
                    rotation_id=f"rot_{i}",
                    secret_id="secret_123",
                    secret_type=SecretType.API_KEY,
                    started_at=datetime.now(timezone.utc) - timedelta(hours=i),
                )
            )

        history = storage.get_rotation_history(limit=10)

        assert len(history) == 10

    def test_get_rotation_history_sorted(self, storage):
        """Rotation history is sorted by started_at descending."""
        for i in range(5):
            storage.save_rotation(
                RotationResult(
                    rotation_id=f"rot_{i}",
                    secret_id="secret_123",
                    secret_type=SecretType.API_KEY,
                    started_at=datetime.now(timezone.utc) - timedelta(hours=i),
                )
            )

        history = storage.get_rotation_history()

        for i in range(len(history) - 1):
            assert history[i].started_at >= history[i + 1].started_at


# =============================================================================
# Scheduler Tests
# =============================================================================


class TestSecretsRotationScheduler:
    """Tests for SecretsRotationScheduler class."""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler with in-memory storage."""
        SecretsRotationStorage._conn_var = contextvars.ContextVar(
            "secrets_rotation_conn", default=None
        )
        config = SecretsRotationConfig(storage_path=None)
        return SecretsRotationScheduler(config)

    def test_init(self, scheduler):
        """Initializes with default state."""
        assert scheduler._running is False
        assert scheduler._task is None
        assert scheduler._rotation_handlers == {}
        assert scheduler._verification_handlers == {}
        assert scheduler._notification_handlers == []

    def test_register_secret(self, scheduler):
        """Registers a secret for rotation."""
        metadata = scheduler.register_secret(
            secret_id="secret_123",
            secret_type=SecretType.API_KEY,
            name="My API Key",
            description="Test API key",
        )

        assert metadata.secret_id == "secret_123"
        assert metadata.next_rotation_at is not None
        assert metadata.rotation_interval_days == 90

    def test_register_secret_custom_interval(self, scheduler):
        """Registers secret with custom interval."""
        metadata = scheduler.register_secret(
            secret_id="secret_123",
            secret_type=SecretType.JWT_SECRET,
            name="JWT Secret",
            rotation_interval_days=14,
        )

        assert metadata.rotation_interval_days == 14

    def test_register_secret_with_tags(self, scheduler):
        """Registers secret with tags."""
        metadata = scheduler.register_secret(
            secret_id="secret_123",
            secret_type=SecretType.API_KEY,
            name="Test",
            tags={"env": "prod", "team": "platform"},
        )

        assert metadata.tags == {"env": "prod", "team": "platform"}

    def test_get_secret(self, scheduler):
        """Gets a registered secret."""
        scheduler.register_secret(
            secret_id="secret_123",
            secret_type=SecretType.API_KEY,
            name="Test",
        )

        metadata = scheduler.get_secret("secret_123")

        assert metadata is not None
        assert metadata.secret_id == "secret_123"

    def test_get_nonexistent_secret(self, scheduler):
        """Returns None for nonexistent secret."""
        result = scheduler.get_secret("nonexistent")

        assert result is None

    def test_get_all_secrets(self, scheduler):
        """Gets all registered secrets."""
        for i in range(3):
            scheduler.register_secret(
                secret_id=f"secret_{i}",
                secret_type=SecretType.API_KEY,
                name=f"Secret {i}",
            )

        secrets = scheduler.get_all_secrets()

        assert len(secrets) == 3

    def test_register_rotation_handler(self, scheduler):
        """Registers a rotation handler."""
        handler = MagicMock(return_value={"success": True})

        scheduler.register_rotation_handler(SecretType.API_KEY, handler)

        assert SecretType.API_KEY in scheduler._rotation_handlers

    def test_register_verification_handler(self, scheduler):
        """Registers a verification handler."""
        handler = MagicMock(return_value=True)

        scheduler.register_verification_handler(SecretType.API_KEY, handler)

        assert SecretType.API_KEY in scheduler._verification_handlers

    def test_register_notification_handler(self, scheduler):
        """Registers a notification handler."""
        handler = MagicMock()

        scheduler.register_notification_handler(handler)

        assert handler in scheduler._notification_handlers

    @pytest.mark.asyncio
    async def test_rotate_secret_no_handler(self, scheduler):
        """Rotates secret without handler (simulated)."""
        scheduler.register_secret(
            secret_id="secret_123",
            secret_type=SecretType.API_KEY,
            name="Test",
        )

        result = await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="secret_123",
        )

        assert result.status == RotationStatus.COMPLETED
        assert result.new_version is not None

    @pytest.mark.asyncio
    async def test_rotate_secret_with_handler(self, scheduler):
        """Rotates secret with custom handler."""
        scheduler.register_secret(
            secret_id="secret_123",
            secret_type=SecretType.API_KEY,
            name="Test",
        )
        handler = MagicMock(return_value={"success": True, "new_version": "v2"})
        scheduler.register_rotation_handler(SecretType.API_KEY, handler)

        result = await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="secret_123",
        )

        handler.assert_called_once_with("secret_123")
        assert result.status == RotationStatus.COMPLETED
        assert result.new_version == "v2"

    @pytest.mark.asyncio
    async def test_rotate_secret_handler_failure(self, scheduler):
        """Handles rotation handler failure."""
        scheduler.register_secret(
            secret_id="secret_123",
            secret_type=SecretType.API_KEY,
            name="Test",
        )
        handler = MagicMock(return_value={"success": False, "error": "Auth failed"})
        scheduler.register_rotation_handler(SecretType.API_KEY, handler)

        result = await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="secret_123",
        )

        assert result.status == RotationStatus.FAILED
        assert "Auth failed" in result.error_message

    @pytest.mark.asyncio
    async def test_rotate_secret_handler_exception(self, scheduler):
        """Handles rotation handler exception."""
        scheduler.register_secret(
            secret_id="secret_123",
            secret_type=SecretType.API_KEY,
            name="Test",
        )
        handler = MagicMock(side_effect=RuntimeError("Connection refused"))
        scheduler.register_rotation_handler(SecretType.API_KEY, handler)

        result = await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="secret_123",
        )

        assert result.status == RotationStatus.FAILED
        assert "Connection refused" in result.error_message

    @pytest.mark.asyncio
    async def test_rotate_secret_with_verification(self, scheduler):
        """Rotates with verification handler."""
        scheduler.register_secret(
            secret_id="secret_123",
            secret_type=SecretType.API_KEY,
            name="Test",
        )
        rotation_handler = MagicMock(return_value={"success": True})
        verification_handler = MagicMock(return_value=True)
        scheduler.register_rotation_handler(SecretType.API_KEY, rotation_handler)
        scheduler.register_verification_handler(SecretType.API_KEY, verification_handler)

        result = await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="secret_123",
        )

        verification_handler.assert_called_once()
        assert result.verification_passed is True
        assert result.status == RotationStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_rotate_secret_verification_failure_rollback(self, scheduler):
        """Rolls back on verification failure."""
        scheduler.register_secret(
            secret_id="secret_123",
            secret_type=SecretType.API_KEY,
            name="Test",
        )
        rotation_handler = MagicMock(return_value={"success": True})
        verification_handler = MagicMock(return_value=False)
        rollback_handler = MagicMock(return_value=True)
        scheduler.register_rotation_handler(SecretType.API_KEY, rotation_handler)
        scheduler.register_verification_handler(SecretType.API_KEY, verification_handler)
        scheduler.register_rollback_handler(SecretType.API_KEY, rollback_handler)

        result = await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="secret_123",
        )

        rollback_handler.assert_called_once()
        assert result.status == RotationStatus.ROLLED_BACK
        assert result.rolled_back is True

    @pytest.mark.asyncio
    async def test_rotate_secret_updates_metadata(self, scheduler):
        """Rotation updates secret metadata."""
        metadata = scheduler.register_secret(
            secret_id="secret_123",
            secret_type=SecretType.API_KEY,
            name="Test",
        )
        original_next_rotation = metadata.next_rotation_at

        result = await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="secret_123",
        )

        updated = scheduler.get_secret("secret_123")
        assert updated.last_rotated_at is not None
        assert updated.next_rotation_at > original_next_rotation

    @pytest.mark.asyncio
    async def test_rotate_secret_with_trigger(self, scheduler):
        """Rotation tracks trigger type."""
        scheduler.register_secret(
            secret_id="secret_123",
            secret_type=SecretType.API_KEY,
            name="Test",
        )

        result = await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="secret_123",
            trigger=RotationTrigger.COMPROMISE,
        )

        assert result.trigger == RotationTrigger.COMPROMISE

    @pytest.mark.asyncio
    async def test_rotate_secret_with_initiator(self, scheduler):
        """Rotation tracks initiator."""
        scheduler.register_secret(
            secret_id="secret_123",
            secret_type=SecretType.API_KEY,
            name="Test",
        )

        result = await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="secret_123",
            initiated_by="admin@example.com",
        )

        assert result.initiated_by == "admin@example.com"

    @pytest.mark.asyncio
    async def test_rotate_secret_sends_notification(self, scheduler):
        """Rotation sends notification."""
        scheduler.register_secret(
            secret_id="secret_123",
            secret_type=SecretType.API_KEY,
            name="Test",
        )
        notification_handler = MagicMock()
        scheduler.register_notification_handler(notification_handler)

        await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="secret_123",
        )

        notification_handler.assert_called_once()
        notification = notification_handler.call_args[0][0]
        assert notification["type"] == "rotation_completed"
        assert notification["secret_id"] == "secret_123"

    def test_get_rotation_history(self, scheduler):
        """Gets rotation history."""
        # Use sync method to add rotation directly
        rotation = RotationResult(
            rotation_id="rot_123",
            secret_id="secret_456",
            secret_type=SecretType.API_KEY,
            started_at=datetime.now(timezone.utc),
        )
        scheduler._storage.save_rotation(rotation)

        history = scheduler.get_rotation_history(secret_id="secret_456")

        assert len(history) == 1

    def test_get_compliance_report(self, scheduler):
        """Gets compliance report."""
        # Add some secrets - categories are mutually exclusive:
        # overdue (days_until < 0), expiring_soon (0 <= days_until <= 7), compliant (days_until > 7)
        now = datetime.now(timezone.utc)
        past = now - timedelta(days=10)  # overdue
        future = now + timedelta(days=30)  # compliant
        soon = now + timedelta(days=3)  # expiring_soon

        for i, (next_rot, name) in enumerate(
            [
                (past, "Overdue"),
                (future, "Compliant"),
                (soon, "Expiring Soon"),
            ]
        ):
            metadata = SecretMetadata(
                secret_id=f"secret_{i}",
                secret_type=SecretType.API_KEY,
                name=name,
                next_rotation_at=next_rot,
            )
            scheduler._storage.save_secret(metadata)

        report = scheduler.get_compliance_report()

        assert report["total_secrets"] == 3
        assert report["overdue"] == 1
        assert report["compliant"] == 1
        assert report["expiring_soon"] == 1  # Only the "soon" secret


class TestSchedulerLifecycle:
    """Tests for scheduler lifecycle methods."""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler with in-memory storage."""
        SecretsRotationStorage._conn_var = contextvars.ContextVar(
            "secrets_rotation_conn", default=None
        )
        config = SecretsRotationConfig(storage_path=None)
        return SecretsRotationScheduler(config)

    @pytest.mark.asyncio
    async def test_start(self, scheduler):
        """Starts the scheduler."""
        await scheduler.start()

        assert scheduler._running is True
        assert scheduler._task is not None

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_start_idempotent(self, scheduler):
        """Start is idempotent."""
        await scheduler.start()
        task1 = scheduler._task

        await scheduler.start()
        task2 = scheduler._task

        assert task1 is task2

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop(self, scheduler):
        """Stops the scheduler."""
        await scheduler.start()
        await scheduler.stop()

        assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, scheduler):
        """Stop when not running doesn't raise."""
        await scheduler.stop()  # Should not raise

        assert scheduler._running is False


class TestDefaultIntervals:
    """Tests for default rotation intervals."""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler with custom config."""
        SecretsRotationStorage._conn_var = contextvars.ContextVar(
            "secrets_rotation_conn", default=None
        )
        config = SecretsRotationConfig(
            api_key_rotation_days=60,
            database_rotation_days=90,
            jwt_rotation_days=14,
            oauth_rotation_days=30,
            encryption_key_rotation_days=365,
        )
        return SecretsRotationScheduler(config)

    def test_api_key_interval(self, scheduler):
        """API key uses configured interval."""
        metadata = scheduler.register_secret(
            secret_id="test",
            secret_type=SecretType.API_KEY,
            name="Test",
        )

        assert metadata.rotation_interval_days == 60

    def test_database_interval(self, scheduler):
        """Database credential uses configured interval."""
        metadata = scheduler.register_secret(
            secret_id="test",
            secret_type=SecretType.DATABASE_CREDENTIAL,
            name="Test",
        )

        assert metadata.rotation_interval_days == 90

    def test_jwt_interval(self, scheduler):
        """JWT secret uses configured interval."""
        metadata = scheduler.register_secret(
            secret_id="test",
            secret_type=SecretType.JWT_SECRET,
            name="Test",
        )

        assert metadata.rotation_interval_days == 14


class TestGlobalScheduler:
    """Tests for global scheduler singleton."""

    def test_get_scheduler_creates_instance(self):
        """Creates scheduler on first call."""
        import aragora.scheduler.secrets_rotation_scheduler as module

        module._scheduler = None

        scheduler = get_secrets_rotation_scheduler()

        assert isinstance(scheduler, SecretsRotationScheduler)

    def test_get_scheduler_returns_same_instance(self):
        """Returns same instance on subsequent calls."""
        import aragora.scheduler.secrets_rotation_scheduler as module

        module._scheduler = None

        scheduler1 = get_secrets_rotation_scheduler()
        scheduler2 = get_secrets_rotation_scheduler()

        assert scheduler1 is scheduler2


class TestConvenienceFunction:
    """Tests for rotate_secret convenience function."""

    @pytest.mark.asyncio
    async def test_rotate_secret_function(self):
        """rotate_secret convenience function works."""
        import aragora.scheduler.secrets_rotation_scheduler as module

        module._scheduler = None

        scheduler = get_secrets_rotation_scheduler()
        scheduler.register_secret(
            secret_id="convenience_test",
            secret_type=SecretType.API_KEY,
            name="Test",
        )

        result = await rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="convenience_test",
        )

        assert result.status == RotationStatus.COMPLETED
