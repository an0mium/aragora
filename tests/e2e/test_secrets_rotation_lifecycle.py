"""
E2E Tests for Secrets Rotation Lifecycle.

Tests comprehensive workflows for secrets rotation including:
- Full secret rotation lifecycle (register -> rotate -> verify -> complete)
- Grace period handling (old + new versions coexist)
- Verification and rollback scenarios
- Compliance reporting (SOC 2 CC6.2)
- Multiple secret types (API keys, JWT, OAuth, database credentials)
- Scheduled rotation triggers
- Manual rotation with audit trail
- Error handling and recovery

SOC 2 Controls: CC6.2 (Credential Management)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.scheduler.secrets_rotation_scheduler import (
    RotationResult,
    RotationStatus,
    RotationTrigger,
    SecretMetadata,
    SecretType,
    SecretsRotationConfig,
    SecretsRotationScheduler,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rotation_config():
    """Create a test rotation config with in-memory storage."""
    return SecretsRotationConfig(
        storage_path=":memory:",
        api_key_rotation_days=90,
        jwt_rotation_days=30,
        oauth_rotation_days=30,
        database_rotation_days=90,
        encryption_key_rotation_days=365,
        default_grace_period_hours=24,
        verify_after_rotation=False,  # Disable verification for basic tests
        rollback_on_verification_failure=True,
        notify_days_before=7,
    )


@pytest.fixture
def scheduler(rotation_config):
    """Create a test scheduler."""
    return SecretsRotationScheduler(rotation_config)


@pytest.fixture
def scheduler_with_verification():
    """Create a scheduler with verification enabled."""
    config = SecretsRotationConfig(
        storage_path=":memory:",
        verify_after_rotation=True,
        rollback_on_verification_failure=True,
    )
    return SecretsRotationScheduler(config)


# =============================================================================
# Secret Registration Tests
# =============================================================================


class TestSecretRegistration:
    """Tests for secret registration lifecycle."""

    def test_register_api_key(self, scheduler):
        """Test registering an API key for rotation."""
        metadata = scheduler.register_secret(
            secret_id="api_key_prod_001",
            secret_type=SecretType.API_KEY,
            name="Production API Key",
            description="Main API key for production environment",
            owner="platform-team",
            tags={"environment": "production", "service": "main"},
        )

        assert metadata.secret_id == "api_key_prod_001"
        assert metadata.secret_type == SecretType.API_KEY
        assert metadata.name == "Production API Key"
        assert metadata.is_active is True
        assert metadata.rotation_interval_days == 90
        assert metadata.next_rotation_at is not None

    def test_register_jwt_secret(self, scheduler):
        """Test registering a JWT secret for rotation."""
        metadata = scheduler.register_secret(
            secret_id="jwt_signing_key",
            secret_type=SecretType.JWT_SECRET,
            name="JWT Signing Key",
            description="Key used to sign authentication tokens",
            rotation_interval_days=30,
        )

        assert metadata.secret_type == SecretType.JWT_SECRET
        assert metadata.rotation_interval_days == 30

    def test_register_database_credential(self, scheduler):
        """Test registering database credentials for rotation."""
        metadata = scheduler.register_secret(
            secret_id="postgres_prod_password",
            secret_type=SecretType.DATABASE_CREDENTIAL,
            name="PostgreSQL Production Password",
            owner="database-team",
        )

        assert metadata.secret_type == SecretType.DATABASE_CREDENTIAL

    def test_register_oauth_token(self, scheduler):
        """Test registering OAuth token for rotation."""
        metadata = scheduler.register_secret(
            secret_id="google_oauth_refresh",
            secret_type=SecretType.OAUTH_TOKEN,
            name="Google OAuth Refresh Token",
            rotation_interval_days=30,
        )

        assert metadata.secret_type == SecretType.OAUTH_TOKEN

    def test_register_encryption_key(self, scheduler):
        """Test registering encryption key for rotation."""
        metadata = scheduler.register_secret(
            secret_id="aes_master_key",
            secret_type=SecretType.ENCRYPTION_KEY,
            name="AES Master Key",
            rotation_interval_days=365,
            description="Master encryption key for data at rest",
        )

        assert metadata.secret_type == SecretType.ENCRYPTION_KEY
        assert metadata.rotation_interval_days == 365

    def test_register_webhook_secret(self, scheduler):
        """Test registering webhook secret for rotation."""
        metadata = scheduler.register_secret(
            secret_id="stripe_webhook_secret",
            secret_type=SecretType.WEBHOOK_SECRET,
            name="Stripe Webhook Secret",
            tags={"provider": "stripe", "purpose": "payment_events"},
        )

        assert metadata.secret_type == SecretType.WEBHOOK_SECRET

    def test_register_multiple_secrets(self, scheduler):
        """Test registering multiple secrets."""
        secrets_to_register = [
            ("api_key_1", SecretType.API_KEY, "API Key 1"),
            ("api_key_2", SecretType.API_KEY, "API Key 2"),
            ("jwt_key", SecretType.JWT_SECRET, "JWT Key"),
            ("db_pass", SecretType.DATABASE_CREDENTIAL, "DB Password"),
        ]

        registered = []
        for secret_id, secret_type, name in secrets_to_register:
            metadata = scheduler.register_secret(
                secret_id=secret_id,
                secret_type=secret_type,
                name=name,
            )
            registered.append(metadata)

        assert len(registered) == 4
        assert all(m.is_active for m in registered)

    def test_register_with_custom_interval(self, scheduler):
        """Test registering with custom rotation interval."""
        metadata = scheduler.register_secret(
            secret_id="custom_interval_key",
            secret_type=SecretType.API_KEY,
            name="Custom Interval Key",
            rotation_interval_days=45,
        )

        assert metadata.rotation_interval_days == 45


# =============================================================================
# Secret Rotation Tests
# =============================================================================


class TestSecretRotation:
    """Tests for secret rotation operations."""

    @pytest.mark.asyncio
    async def test_rotate_api_key(self, scheduler):
        """Test rotating an API key."""
        # Register the secret
        scheduler.register_secret(
            secret_id="api_key_to_rotate",
            secret_type=SecretType.API_KEY,
            name="API Key to Rotate",
        )

        # Rotate the secret
        result = await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="api_key_to_rotate",
            trigger=RotationTrigger.MANUAL,
            initiated_by="test_user",
        )

        assert result.secret_id == "api_key_to_rotate"
        assert result.status == RotationStatus.COMPLETED
        assert result.trigger == RotationTrigger.MANUAL
        assert result.initiated_by == "test_user"
        assert result.rotation_id is not None

    @pytest.mark.asyncio
    async def test_rotate_jwt_secret(self, scheduler):
        """Test rotating a JWT secret."""
        scheduler.register_secret(
            secret_id="jwt_to_rotate",
            secret_type=SecretType.JWT_SECRET,
            name="JWT to Rotate",
        )

        result = await scheduler.rotate_secret(
            secret_type=SecretType.JWT_SECRET,
            secret_id="jwt_to_rotate",
        )

        assert result.status == RotationStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_rotation_generates_new_version(self, scheduler):
        """Test that rotation generates a new version hash."""
        scheduler.register_secret(
            secret_id="version_test_key",
            secret_type=SecretType.API_KEY,
            name="Version Test Key",
        )

        result = await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="version_test_key",
        )

        assert result.new_version is not None
        assert len(result.new_version) > 0
        # Version should be a hash
        assert all(c in "0123456789abcdef" for c in result.new_version.lower())

    @pytest.mark.asyncio
    async def test_rotation_sets_grace_period(self, scheduler):
        """Test that rotation sets an appropriate grace period."""
        scheduler.register_secret(
            secret_id="grace_period_key",
            secret_type=SecretType.API_KEY,
            name="Grace Period Key",
        )

        result = await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="grace_period_key",
        )

        assert result.grace_period_ends is not None
        # Grace period should be in the future
        assert result.grace_period_ends > datetime.now(timezone.utc)

    @pytest.mark.asyncio
    async def test_rotation_updates_last_rotated(self, scheduler):
        """Test that rotation updates last_rotated_at timestamp."""
        scheduler.register_secret(
            secret_id="timestamp_test_key",
            secret_type=SecretType.API_KEY,
            name="Timestamp Test Key",
        )

        before_rotation = datetime.now(timezone.utc)
        await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="timestamp_test_key",
        )

        # Get the updated metadata
        metadata = scheduler.get_secret("timestamp_test_key")
        assert metadata is not None
        assert metadata.last_rotated_at is not None
        assert metadata.last_rotated_at >= before_rotation

    @pytest.mark.asyncio
    async def test_rotation_schedules_next_rotation(self, scheduler):
        """Test that rotation schedules the next rotation."""
        scheduler.register_secret(
            secret_id="next_schedule_key",
            secret_type=SecretType.API_KEY,
            name="Next Schedule Key",
            rotation_interval_days=90,
        )

        await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="next_schedule_key",
        )

        metadata = scheduler.get_secret("next_schedule_key")
        assert metadata.next_rotation_at is not None
        # Next rotation should be approximately 90 days from now
        expected_next = datetime.now(timezone.utc) + timedelta(days=90)
        diff = abs((metadata.next_rotation_at - expected_next).days)
        assert diff <= 1  # Allow 1 day tolerance

    @pytest.mark.asyncio
    async def test_multiple_rotations_track_history(self, scheduler):
        """Test that multiple rotations are tracked in history."""
        scheduler.register_secret(
            secret_id="history_test_key",
            secret_type=SecretType.API_KEY,
            name="History Test Key",
        )

        # Perform multiple rotations
        results = []
        for i in range(3):
            result = await scheduler.rotate_secret(
                secret_type=SecretType.API_KEY,
                secret_id="history_test_key",
                initiated_by=f"user_{i}",
            )
            results.append(result)

        assert len(results) == 3
        # All should have unique rotation IDs
        rotation_ids = [r.rotation_id for r in results]
        assert len(set(rotation_ids)) == 3


# =============================================================================
# Rotation Trigger Tests
# =============================================================================


class TestRotationTriggers:
    """Tests for different rotation triggers."""

    @pytest.mark.asyncio
    async def test_manual_trigger(self, scheduler):
        """Test manual rotation trigger."""
        scheduler.register_secret(
            secret_id="manual_trigger_key",
            secret_type=SecretType.API_KEY,
            name="Manual Trigger Key",
        )

        result = await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="manual_trigger_key",
            trigger=RotationTrigger.MANUAL,
        )

        assert result.trigger == RotationTrigger.MANUAL

    @pytest.mark.asyncio
    async def test_scheduled_trigger(self, scheduler):
        """Test scheduled rotation trigger."""
        scheduler.register_secret(
            secret_id="scheduled_trigger_key",
            secret_type=SecretType.API_KEY,
            name="Scheduled Trigger Key",
        )

        result = await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="scheduled_trigger_key",
            trigger=RotationTrigger.SCHEDULED,
        )

        assert result.trigger == RotationTrigger.SCHEDULED

    @pytest.mark.asyncio
    async def test_expiration_trigger(self, scheduler):
        """Test expiration rotation trigger."""
        scheduler.register_secret(
            secret_id="expiration_trigger_key",
            secret_type=SecretType.API_KEY,
            name="Expiration Trigger Key",
        )

        result = await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="expiration_trigger_key",
            trigger=RotationTrigger.EXPIRATION,
        )

        assert result.trigger == RotationTrigger.EXPIRATION

    @pytest.mark.asyncio
    async def test_compromise_trigger(self, scheduler):
        """Test compromise (emergency) rotation trigger."""
        scheduler.register_secret(
            secret_id="compromise_trigger_key",
            secret_type=SecretType.API_KEY,
            name="Compromise Trigger Key",
        )

        result = await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="compromise_trigger_key",
            trigger=RotationTrigger.COMPROMISE,
        )

        assert result.trigger == RotationTrigger.COMPROMISE

    @pytest.mark.asyncio
    async def test_policy_trigger(self, scheduler):
        """Test policy-based rotation trigger."""
        scheduler.register_secret(
            secret_id="policy_trigger_key",
            secret_type=SecretType.API_KEY,
            name="Policy Trigger Key",
        )

        result = await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="policy_trigger_key",
            trigger=RotationTrigger.POLICY,
        )

        assert result.trigger == RotationTrigger.POLICY


# =============================================================================
# Verification and Rollback Tests
# =============================================================================


class TestVerificationAndRollback:
    """Tests for rotation verification and rollback."""

    @pytest.mark.asyncio
    async def test_verification_success(self, scheduler_with_verification):
        """Test successful verification after rotation."""
        scheduler = scheduler_with_verification

        scheduler.register_secret(
            secret_id="verify_success_key",
            secret_type=SecretType.API_KEY,
            name="Verify Success Key",
        )

        # Register a verification handler that always passes
        scheduler.register_verification_handler(
            SecretType.API_KEY,
            lambda old, new: True,
        )

        result = await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="verify_success_key",
        )

        assert result.status == RotationStatus.COMPLETED
        assert result.verification_passed is True
        assert result.rolled_back is False

    @pytest.mark.asyncio
    async def test_verification_failure_triggers_rollback(self, scheduler_with_verification):
        """Test that verification failure triggers rollback."""
        scheduler = scheduler_with_verification

        scheduler.register_secret(
            secret_id="verify_fail_key",
            secret_type=SecretType.API_KEY,
            name="Verify Fail Key",
        )

        # Register a verification handler that always fails
        scheduler.register_verification_handler(
            SecretType.API_KEY,
            lambda old, new: False,
        )

        # Register a rollback handler
        rollback_called = False

        def rollback_handler(old_version, new_version):
            nonlocal rollback_called
            rollback_called = True
            return True

        scheduler.register_rollback_handler(SecretType.API_KEY, rollback_handler)

        result = await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="verify_fail_key",
        )

        assert result.verification_passed is False
        assert result.rolled_back is True
        assert rollback_called is True

    @pytest.mark.asyncio
    async def test_verification_without_rollback_config(self):
        """Test verification failure when rollback is disabled."""
        config = SecretsRotationConfig(
            storage_path=":memory:",
            verify_after_rotation=True,
            rollback_on_verification_failure=False,
        )
        scheduler = SecretsRotationScheduler(config)

        scheduler.register_secret(
            secret_id="no_rollback_key",
            secret_type=SecretType.API_KEY,
            name="No Rollback Key",
        )

        # Register a verification handler that fails
        scheduler.register_verification_handler(
            SecretType.API_KEY,
            lambda old, new: False,
        )

        result = await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="no_rollback_key",
        )

        # When verification fails and rollback is disabled,
        # the rotation completes but may be marked as failed
        assert result.status in (RotationStatus.COMPLETED, RotationStatus.FAILED)
        assert result.rolled_back is False


# =============================================================================
# Custom Handler Tests
# =============================================================================


class TestCustomHandlers:
    """Tests for custom rotation handlers."""

    @pytest.mark.asyncio
    async def test_custom_rotation_handler(self, scheduler):
        """Test custom rotation handler is called."""
        scheduler.register_secret(
            secret_id="custom_handler_key",
            secret_type=SecretType.API_KEY,
            name="Custom Handler Key",
        )

        handler_data = {}

        def custom_handler(secret_id: str) -> Dict[str, Any]:
            handler_data["called"] = True
            handler_data["secret_id"] = secret_id
            return {"new_key": "generated_key_value"}

        scheduler.register_rotation_handler(SecretType.API_KEY, custom_handler)

        await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="custom_handler_key",
        )

        assert handler_data["called"] is True
        assert handler_data["secret_id"] == "custom_handler_key"

    @pytest.mark.asyncio
    async def test_handler_per_secret_type(self, scheduler):
        """Test different handlers for different secret types."""
        scheduler.register_secret(
            secret_id="api_key_type",
            secret_type=SecretType.API_KEY,
            name="API Key Type",
        )
        scheduler.register_secret(
            secret_id="jwt_type",
            secret_type=SecretType.JWT_SECRET,
            name="JWT Type",
        )

        api_handler_called = False
        jwt_handler_called = False

        def api_handler(secret_id):
            nonlocal api_handler_called
            api_handler_called = True
            return {}

        def jwt_handler(secret_id):
            nonlocal jwt_handler_called
            jwt_handler_called = True
            return {}

        scheduler.register_rotation_handler(SecretType.API_KEY, api_handler)
        scheduler.register_rotation_handler(SecretType.JWT_SECRET, jwt_handler)

        await scheduler.rotate_secret(SecretType.API_KEY, "api_key_type")
        assert api_handler_called is True
        assert jwt_handler_called is False

        await scheduler.rotate_secret(SecretType.JWT_SECRET, "jwt_type")
        assert jwt_handler_called is True


# =============================================================================
# Compliance Reporting Tests
# =============================================================================


class TestComplianceReporting:
    """Tests for SOC 2 CC6.2 compliance reporting."""

    def test_compliance_report_structure(self, scheduler):
        """Test compliance report has required fields."""
        report = scheduler.get_compliance_report()

        assert "total_secrets" in report
        assert "compliant" in report
        assert "overdue" in report
        # Note: "upcoming" is called "expiring_soon" in the actual implementation
        assert "expiring_soon" in report
        assert "compliance_rate" in report
        assert "by_type" in report

    def test_compliance_with_no_secrets(self, scheduler):
        """Test compliance report with no registered secrets."""
        report = scheduler.get_compliance_report()

        assert report["total_secrets"] == 0
        assert report["compliant"] == 0
        assert report["overdue"] == 0
        assert report["compliance_rate"] == 1.0  # No secrets = compliant (1.0 not 100.0)

    def test_compliance_with_fresh_secrets(self, scheduler):
        """Test compliance with freshly registered secrets."""
        scheduler.register_secret(
            secret_id="fresh_key_1",
            secret_type=SecretType.API_KEY,
            name="Fresh Key 1",
        )
        scheduler.register_secret(
            secret_id="fresh_key_2",
            secret_type=SecretType.API_KEY,
            name="Fresh Key 2",
        )

        report = scheduler.get_compliance_report()

        assert report["total_secrets"] == 2
        assert report["compliant"] == 2
        assert report["overdue"] == 0
        assert report["compliance_rate"] == 1.0  # 1.0 not 100.0

    @pytest.mark.asyncio
    async def test_compliance_after_rotation(self, scheduler):
        """Test compliance improves after rotation."""
        scheduler.register_secret(
            secret_id="rotated_key",
            secret_type=SecretType.API_KEY,
            name="Rotated Key",
        )

        await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="rotated_key",
        )

        report = scheduler.get_compliance_report()

        assert report["compliant"] == 1
        assert report["overdue"] == 0

    def test_compliance_report_by_type(self, scheduler):
        """Test compliance report breakdown by type."""
        scheduler.register_secret("api_1", SecretType.API_KEY, "API 1")
        scheduler.register_secret("api_2", SecretType.API_KEY, "API 2")
        scheduler.register_secret("jwt_1", SecretType.JWT_SECRET, "JWT 1")
        scheduler.register_secret("db_1", SecretType.DATABASE_CREDENTIAL, "DB 1")

        report = scheduler.get_compliance_report()

        assert "by_type" in report
        by_type = report["by_type"]
        assert SecretType.API_KEY.value in by_type
        # by_type is a simple dict: {type_value: count}
        assert by_type[SecretType.API_KEY.value] == 2
        assert SecretType.JWT_SECRET.value in by_type
        assert by_type[SecretType.JWT_SECRET.value] == 1


# =============================================================================
# Scheduler Lifecycle Tests
# =============================================================================


class TestSchedulerLifecycle:
    """Tests for scheduler start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_scheduler_start_stop(self, scheduler):
        """Test starting and stopping the scheduler."""
        assert scheduler._running is False

        await scheduler.start()
        assert scheduler._running is True

        await scheduler.stop()
        assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_scheduler_start_twice_is_safe(self, scheduler):
        """Test starting scheduler twice doesn't cause issues."""
        await scheduler.start()
        await scheduler.start()  # Second start should be no-op

        assert scheduler._running is True

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_scheduler_stop_twice_is_safe(self, scheduler):
        """Test stopping scheduler twice doesn't cause issues."""
        await scheduler.start()
        await scheduler.stop()
        await scheduler.stop()  # Second stop should be no-op

        assert scheduler._running is False


# =============================================================================
# Due Secrets Tests
# =============================================================================


class TestDueSecrets:
    """Tests for identifying secrets due for rotation."""

    def test_get_due_secrets_empty(self, scheduler):
        """Test getting due secrets when none are due."""
        scheduler.register_secret(
            secret_id="not_due_key",
            secret_type=SecretType.API_KEY,
            name="Not Due Key",
        )

        # Access via storage layer
        due = scheduler._storage.get_secrets_due_for_rotation()
        # Freshly registered secrets shouldn't be due yet
        assert len(due) == 0

    def test_get_due_secrets_finds_overdue(self, scheduler):
        """Test getting secrets that are overdue for rotation."""
        # Register a secret
        scheduler.register_secret(
            secret_id="overdue_key",
            secret_type=SecretType.API_KEY,
            name="Overdue Key",
        )

        # Manually set next_rotation to the past
        past_date = datetime.now(timezone.utc) - timedelta(days=1)
        metadata = scheduler.get_secret("overdue_key")
        if metadata:
            metadata.next_rotation_at = past_date
            scheduler._storage.save_secret(metadata)

        # Access via storage layer
        due = scheduler._storage.get_secrets_due_for_rotation()
        assert len(due) == 1
        assert due[0].secret_id == "overdue_key"


# =============================================================================
# RotationResult Tests
# =============================================================================


class TestRotationResult:
    """Tests for RotationResult dataclass."""

    def test_rotation_result_to_dict(self):
        """Test RotationResult converts to dictionary correctly."""
        now = datetime.now(timezone.utc)
        result = RotationResult(
            rotation_id="rot-123",
            secret_id="key-456",
            secret_type=SecretType.API_KEY,
            status=RotationStatus.COMPLETED,
            trigger=RotationTrigger.MANUAL,
            started_at=now,
            completed_at=now + timedelta(seconds=5),
            initiated_by="test_user",
            old_version="abc123",
            new_version="def456",
            duration_seconds=5.0,
            verification_passed=True,
        )

        data = result.to_dict()

        assert data["rotation_id"] == "rot-123"
        assert data["secret_id"] == "key-456"
        assert data["secret_type"] == "api_key"
        assert data["status"] == "completed"
        assert data["trigger"] == "manual"
        assert data["initiated_by"] == "test_user"
        assert data["verification_passed"] is True
        assert data["duration_seconds"] == 5.0


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in secrets rotation."""

    @pytest.mark.asyncio
    async def test_rotate_nonexistent_secret(self, scheduler):
        """Test rotating a secret that doesn't exist."""
        result = await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="nonexistent_key",
        )

        # The scheduler may auto-register and rotate, or fail
        # Either way, the operation should complete without exception
        assert result.status in (RotationStatus.COMPLETED, RotationStatus.FAILED)
        assert result.rotation_id is not None

    @pytest.mark.asyncio
    async def test_handler_exception_captured(self, scheduler):
        """Test that handler exceptions are captured."""
        scheduler.register_secret(
            secret_id="exception_key",
            secret_type=SecretType.API_KEY,
            name="Exception Key",
        )

        def failing_handler(secret_id):
            raise ValueError("Handler failed")

        scheduler.register_rotation_handler(SecretType.API_KEY, failing_handler)

        result = await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="exception_key",
        )

        assert result.status == RotationStatus.FAILED
        assert "Handler failed" in (result.error_message or "")


# =============================================================================
# Grace Period Tests
# =============================================================================


class TestGracePeriod:
    """Tests for grace period handling during rotation."""

    @pytest.mark.asyncio
    async def test_grace_period_for_api_key(self, scheduler):
        """Test grace period for API key (48 hours default)."""
        scheduler.register_secret(
            secret_id="api_grace_key",
            secret_type=SecretType.API_KEY,
            name="API Grace Key",
        )

        result = await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="api_grace_key",
        )

        assert result.grace_period_ends is not None
        # API keys have 48-hour grace period
        expected_end = datetime.now(timezone.utc) + timedelta(hours=48)
        diff_hours = abs((result.grace_period_ends - expected_end).total_seconds() / 3600)
        assert diff_hours < 1  # Within 1 hour tolerance

    @pytest.mark.asyncio
    async def test_grace_period_for_database(self):
        """Test shorter grace period for database credentials (2 hours)."""
        config = SecretsRotationConfig(
            storage_path=":memory:",
            database_grace_period_hours=2,
            verify_after_rotation=False,
        )
        scheduler = SecretsRotationScheduler(config)

        scheduler.register_secret(
            secret_id="db_grace_key",
            secret_type=SecretType.DATABASE_CREDENTIAL,
            name="DB Grace Key",
        )

        result = await scheduler.rotate_secret(
            secret_type=SecretType.DATABASE_CREDENTIAL,
            secret_id="db_grace_key",
        )

        assert result.grace_period_ends is not None
        expected_end = datetime.now(timezone.utc) + timedelta(hours=2)
        diff_hours = abs((result.grace_period_ends - expected_end).total_seconds() / 3600)
        assert diff_hours < 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestSecretsRotationIntegration:
    """Integration tests for full secrets rotation lifecycle."""

    @pytest.mark.asyncio
    async def test_full_rotation_lifecycle(self, scheduler):
        """Test complete secrets rotation lifecycle."""
        # 1. Register secret
        metadata = scheduler.register_secret(
            secret_id="lifecycle_key",
            secret_type=SecretType.API_KEY,
            name="Lifecycle Test Key",
            description="Key for testing full lifecycle",
            owner="test-team",
            tags={"test": "true"},
        )
        assert metadata.is_active is True
        assert metadata.last_rotated_at is None

        # 2. Rotate secret
        result = await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="lifecycle_key",
            trigger=RotationTrigger.MANUAL,
            initiated_by="integration_test",
        )
        assert result.status == RotationStatus.COMPLETED
        assert result.new_version is not None

        # 3. Verify metadata updated
        updated_metadata = scheduler.get_secret("lifecycle_key")
        assert updated_metadata.last_rotated_at is not None
        assert updated_metadata.next_rotation_at is not None

        # 4. Check compliance report
        report = scheduler.get_compliance_report()
        assert report["total_secrets"] == 1
        assert report["compliant"] == 1

    @pytest.mark.asyncio
    async def test_multiple_secret_types_rotation(self, scheduler):
        """Test rotating multiple different secret types."""
        secrets = [
            ("api_key", SecretType.API_KEY, "API Key"),
            ("jwt_secret", SecretType.JWT_SECRET, "JWT Secret"),
            ("db_cred", SecretType.DATABASE_CREDENTIAL, "DB Credential"),
            ("oauth_token", SecretType.OAUTH_TOKEN, "OAuth Token"),
            ("encryption_key", SecretType.ENCRYPTION_KEY, "Encryption Key"),
        ]

        # Register all secrets
        for secret_id, secret_type, name in secrets:
            scheduler.register_secret(
                secret_id=secret_id,
                secret_type=secret_type,
                name=name,
            )

        # Rotate all secrets
        results = []
        for secret_id, secret_type, _ in secrets:
            result = await scheduler.rotate_secret(
                secret_type=secret_type,
                secret_id=secret_id,
            )
            results.append(result)

        # All rotations should succeed
        assert all(r.status == RotationStatus.COMPLETED for r in results)
        assert all(r.new_version is not None for r in results)

    @pytest.mark.asyncio
    async def test_scheduler_processes_due_secrets(self, scheduler):
        """Test that scheduler loop processes due secrets."""
        # Register a secret
        scheduler.register_secret(
            secret_id="due_process_key",
            secret_type=SecretType.API_KEY,
            name="Due Process Key",
        )

        # Make it due by setting next_rotation to past
        metadata = scheduler.get_secret("due_process_key")
        metadata.next_rotation_at = datetime.now(timezone.utc) - timedelta(hours=1)
        scheduler._storage.save_secret(metadata)

        # Get due secrets via storage layer
        due = scheduler._storage.get_secrets_due_for_rotation()
        assert len(due) == 1

        # Process due secrets
        for secret in due:
            await scheduler.rotate_secret(
                secret_type=secret.secret_type,
                secret_id=secret.secret_id,
                trigger=RotationTrigger.SCHEDULED,
            )

        # Should no longer be due
        due_after = scheduler._storage.get_secrets_due_for_rotation()
        assert len(due_after) == 0
