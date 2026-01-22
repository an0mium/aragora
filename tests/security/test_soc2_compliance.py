"""
Tests for SOC 2 compliance modules.

Tests the anomaly detection, access review, DR drill,
and secrets rotation schedulers.
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timedelta, timezone

import pytest


# =============================================================================
# Anomaly Detection Tests
# =============================================================================


class TestAnomalyDetection:
    """Tests for the anomaly detection module."""

    def test_anomaly_detector_initialization(self):
        """AnomalyDetector initializes correctly."""
        from aragora.security.anomaly_detection import AnomalyDetector, AnomalyDetectorConfig

        config = AnomalyDetectorConfig(storage_path=":memory:")
        detector = AnomalyDetector(config)

        assert detector.config.failed_login_threshold == 5
        assert detector.config.brute_force_threshold == 10

    @pytest.mark.asyncio
    async def test_check_auth_event_normal(self):
        """Normal auth events are not flagged as anomalous."""
        from aragora.security.anomaly_detection import AnomalyDetector, AnomalyDetectorConfig

        config = AnomalyDetectorConfig(storage_path=":memory:")
        detector = AnomalyDetector(config)

        result = await detector.check_auth_event(
            user_id="test_user",
            success=True,
            ip_address="192.168.1.1",
        )

        assert result.is_anomalous is False

    @pytest.mark.asyncio
    async def test_detect_brute_force(self):
        """Brute force attacks are detected."""
        from aragora.security.anomaly_detection import (
            AnomalyDetector,
            AnomalyDetectorConfig,
            AnomalyType,
        )

        config = AnomalyDetectorConfig(
            storage_path=":memory:",
            brute_force_threshold=3,
        )
        detector = AnomalyDetector(config)

        # Simulate multiple failed logins
        for _ in range(4):
            result = await detector.check_auth_event(
                user_id="victim_user",
                success=False,
                ip_address="192.168.1.1",
            )

        assert result.is_anomalous is True
        assert result.anomaly_type == AnomalyType.AUTH_BRUTE_FORCE

    @pytest.mark.asyncio
    async def test_detect_credential_stuffing(self):
        """Credential stuffing attacks are detected."""
        from aragora.security.anomaly_detection import (
            AnomalyDetector,
            AnomalyDetectorConfig,
            AnomalyType,
        )

        config = AnomalyDetectorConfig(
            storage_path=":memory:",
            credential_stuffing_threshold=2,
        )
        detector = AnomalyDetector(config)

        # Simulate failed logins for different users from same IP
        attacker_ip = "10.0.0.1"
        for i in range(3):
            result = await detector.check_auth_event(
                user_id=f"user_{i}",
                success=False,
                ip_address=attacker_ip,
            )

        assert result.is_anomalous is True
        assert result.anomaly_type == AnomalyType.AUTH_CREDENTIAL_STUFFING

    @pytest.mark.asyncio
    async def test_detect_request_flood(self):
        """Request floods are detected."""
        from aragora.security.anomaly_detection import (
            AnomalyDetector,
            AnomalyDetectorConfig,
            AnomalyType,
        )

        config = AnomalyDetectorConfig(
            storage_path=":memory:",
            request_flood_per_minute=10,
        )
        detector = AnomalyDetector(config)

        # Simulate flood of requests
        for _ in range(15):
            result = await detector.check_rate_anomaly(
                user_id="flood_user",
                ip_address="192.168.1.1",
                endpoint="/api/test",
            )

        assert result.is_anomalous is True
        assert result.anomaly_type == AnomalyType.RATE_REQUEST_FLOOD

    def test_anomaly_result_to_dict(self):
        """AnomalyResult converts to dictionary correctly."""
        from aragora.security.anomaly_detection import (
            AnomalyResult,
            AnomalyType,
            AnomalySeverity,
        )

        result = AnomalyResult(
            is_anomalous=True,
            anomaly_type=AnomalyType.AUTH_BRUTE_FORCE,
            severity=AnomalySeverity.HIGH,
            confidence=0.95,
            description="Test anomaly",
        )

        data = result.to_dict()
        assert data["is_anomalous"] is True
        assert data["anomaly_type"] == "auth.brute_force"
        assert data["severity"] == "high"
        assert data["confidence"] == 0.95

    def test_get_anomaly_stats(self):
        """Anomaly detector provides statistics."""
        from aragora.security.anomaly_detection import AnomalyDetector, AnomalyDetectorConfig

        config = AnomalyDetectorConfig(storage_path=":memory:")
        detector = AnomalyDetector(config)

        stats = detector.get_anomaly_stats()
        assert "total_24h" in stats
        assert "by_type" in stats
        assert "by_severity" in stats


# =============================================================================
# Access Review Scheduler Tests
# =============================================================================


class TestAccessReviewScheduler:
    """Tests for the access review scheduler."""

    def test_scheduler_initialization(self):
        """AccessReviewScheduler initializes correctly."""
        from aragora.scheduler.access_review_scheduler import (
            AccessReviewScheduler,
            AccessReviewConfig,
        )

        config = AccessReviewConfig(storage_path=":memory:")
        scheduler = AccessReviewScheduler(config)

        assert scheduler.config.monthly_review_day == 1
        assert scheduler.config.stale_credential_days == 90

    @pytest.mark.asyncio
    async def test_create_review(self):
        """Reviews can be created."""
        from aragora.scheduler.access_review_scheduler import (
            AccessReviewScheduler,
            AccessReviewConfig,
            ReviewType,
            ReviewStatus,
        )

        config = AccessReviewConfig(storage_path=":memory:")
        scheduler = AccessReviewScheduler(config)

        review = await scheduler.create_review(
            review_type=ReviewType.MONTHLY,
            created_by="test_user",
        )

        assert review.review_id is not None
        assert review.review_type == ReviewType.MONTHLY
        assert review.status == ReviewStatus.PENDING
        assert review.due_date is not None

    @pytest.mark.asyncio
    async def test_review_items_are_gathered(self):
        """Review items are gathered when creating a review."""
        from aragora.scheduler.access_review_scheduler import (
            AccessReviewScheduler,
            AccessReviewConfig,
            ReviewType,
        )

        config = AccessReviewConfig(storage_path=":memory:")
        scheduler = AccessReviewScheduler(config)

        review = await scheduler.create_review(review_type=ReviewType.MONTHLY)

        # Should have mock data items
        assert review.items is not None
        assert review.summary["total_items"] >= 0

    @pytest.mark.asyncio
    async def test_approve_review_item(self):
        """Review items can be approved."""
        from aragora.scheduler.access_review_scheduler import (
            AccessReviewScheduler,
            AccessReviewConfig,
            ReviewType,
            ReviewItemStatus,
        )

        config = AccessReviewConfig(storage_path=":memory:")
        scheduler = AccessReviewScheduler(config)

        review = await scheduler.create_review(review_type=ReviewType.MONTHLY)

        if review.items:
            item = review.items[0]
            success = await scheduler.approve_item(
                review_id=review.review_id,
                item_id=item.item_id,
                decision_by="reviewer",
                notes="Access verified",
            )

            assert success is True

            # Reload and check
            updated = scheduler.get_review(review.review_id)
            updated_item = next(i for i in updated.items if i.item_id == item.item_id)
            assert updated_item.status == ReviewItemStatus.APPROVED

    @pytest.mark.asyncio
    async def test_revoke_review_item(self):
        """Review items can be revoked."""
        from aragora.scheduler.access_review_scheduler import (
            AccessReviewScheduler,
            AccessReviewConfig,
            ReviewType,
            ReviewItemStatus,
        )

        config = AccessReviewConfig(storage_path=":memory:")
        scheduler = AccessReviewScheduler(config)

        review = await scheduler.create_review(review_type=ReviewType.MONTHLY)

        if review.items:
            item = review.items[0]
            success = await scheduler.revoke_item(
                review_id=review.review_id,
                item_id=item.item_id,
                decision_by="reviewer",
                notes="Access no longer needed",
            )

            assert success is True

    def test_access_review_to_dict(self):
        """AccessReview converts to dictionary correctly."""
        from aragora.scheduler.access_review_scheduler import (
            AccessReview,
            ReviewType,
            ReviewStatus,
        )

        review = AccessReview(
            review_id="test_123",
            review_type=ReviewType.MONTHLY,
            status=ReviewStatus.PENDING,
        )

        data = review.to_dict()
        assert data["review_id"] == "test_123"
        assert data["review_type"] == "monthly"
        assert data["status"] == "pending"


# =============================================================================
# DR Drill Scheduler Tests
# =============================================================================


class TestDRDrillScheduler:
    """Tests for the DR drill scheduler."""

    def test_scheduler_initialization(self):
        """DRDrillScheduler initializes correctly."""
        from aragora.scheduler.dr_drill_scheduler import (
            DRDrillScheduler,
            DRDrillConfig,
        )

        config = DRDrillConfig(storage_path=":memory:")
        scheduler = DRDrillScheduler(config)

        assert scheduler.config.target_rto_seconds == 3600.0
        assert scheduler.config.target_rpo_seconds == 300.0

    @pytest.mark.asyncio
    async def test_execute_drill(self):
        """DR drills can be executed."""
        from aragora.scheduler.dr_drill_scheduler import (
            DRDrillScheduler,
            DRDrillConfig,
            DrillType,
            DrillStatus,
        )

        config = DRDrillConfig(storage_path=":memory:", dry_run=True)
        scheduler = DRDrillScheduler(config)

        drill = await scheduler.execute_drill(
            drill_type=DrillType.BACKUP_RESTORATION,
            initiated_by="test_user",
        )

        assert drill.drill_id is not None
        assert drill.drill_type == DrillType.BACKUP_RESTORATION
        assert drill.status in [DrillStatus.COMPLETED, DrillStatus.PARTIAL]
        assert drill.steps is not None
        assert len(drill.steps) > 0

    @pytest.mark.asyncio
    async def test_drill_metrics_calculated(self):
        """Drill metrics are calculated after execution."""
        from aragora.scheduler.dr_drill_scheduler import (
            DRDrillScheduler,
            DRDrillConfig,
            DrillType,
        )

        config = DRDrillConfig(storage_path=":memory:", dry_run=True)
        scheduler = DRDrillScheduler(config)

        drill = await scheduler.execute_drill(drill_type=DrillType.BACKUP_RESTORATION)

        assert drill.total_duration_seconds >= 0
        assert drill.rto_seconds is not None
        assert drill.rpo_seconds is not None
        assert 0 <= drill.success_rate <= 1.0

    @pytest.mark.asyncio
    async def test_compliance_check(self):
        """Drills check RTO/RPO compliance."""
        from aragora.scheduler.dr_drill_scheduler import (
            DRDrillScheduler,
            DRDrillConfig,
            DrillType,
        )

        config = DRDrillConfig(
            storage_path=":memory:",
            dry_run=True,
            target_rto_seconds=10000.0,  # Very generous RTO
            target_rpo_seconds=10000.0,  # Very generous RPO
        )
        scheduler = DRDrillScheduler(config)

        drill = await scheduler.execute_drill(drill_type=DrillType.DATA_INTEGRITY_CHECK)

        # Should meet targets with generous thresholds
        assert drill.meets_rto is True
        assert drill.meets_rpo is True

    def test_get_compliance_report(self):
        """Compliance report can be generated."""
        from aragora.scheduler.dr_drill_scheduler import (
            DRDrillScheduler,
            DRDrillConfig,
        )

        config = DRDrillConfig(storage_path=":memory:")
        scheduler = DRDrillScheduler(config)

        report = scheduler.get_compliance_report()

        assert "compliance_rate" in report
        assert "total_drills" in report
        assert "average_rto_seconds" in report
        assert "target_rto_seconds" in report

    def test_drill_result_to_dict(self):
        """DRDrillResult converts to dictionary correctly."""
        from aragora.scheduler.dr_drill_scheduler import (
            DRDrillResult,
            DrillType,
            DrillStatus,
        )

        result = DRDrillResult(
            drill_id="drill_123",
            drill_type=DrillType.BACKUP_RESTORATION,
            status=DrillStatus.COMPLETED,
            rto_seconds=120.0,
            rpo_seconds=60.0,
            is_compliant=True,
        )

        data = result.to_dict()
        assert data["drill_id"] == "drill_123"
        assert data["drill_type"] == "backup_restoration"
        assert data["status"] == "completed"
        assert data["rto_seconds"] == 120.0
        assert data["is_compliant"] is True


# =============================================================================
# Secrets Rotation Scheduler Tests
# =============================================================================


class TestSecretsRotationScheduler:
    """Tests for the secrets rotation scheduler."""

    def test_scheduler_initialization(self):
        """SecretsRotationScheduler initializes correctly."""
        from aragora.scheduler.secrets_rotation_scheduler import (
            SecretsRotationScheduler,
            SecretsRotationConfig,
        )

        config = SecretsRotationConfig(storage_path=":memory:")
        scheduler = SecretsRotationScheduler(config)

        assert scheduler.config.api_key_rotation_days == 90
        assert scheduler.config.jwt_rotation_days == 30

    def test_register_secret(self):
        """Secrets can be registered for rotation."""
        from aragora.scheduler.secrets_rotation_scheduler import (
            SecretsRotationScheduler,
            SecretsRotationConfig,
            SecretType,
        )

        config = SecretsRotationConfig(storage_path=":memory:")
        scheduler = SecretsRotationScheduler(config)

        metadata = scheduler.register_secret(
            secret_id="api_key_123",
            secret_type=SecretType.API_KEY,
            name="Production API Key",
            description="Main API key for production",
            owner="security_team",
        )

        assert metadata.secret_id == "api_key_123"
        assert metadata.secret_type == SecretType.API_KEY
        assert metadata.next_rotation_at is not None
        assert metadata.rotation_interval_days == 90  # Default for API keys

    @pytest.mark.asyncio
    async def test_rotate_secret(self):
        """Secrets can be rotated."""
        from aragora.scheduler.secrets_rotation_scheduler import (
            SecretsRotationScheduler,
            SecretsRotationConfig,
            SecretType,
            RotationStatus,
        )

        config = SecretsRotationConfig(
            storage_path=":memory:",
            verify_after_rotation=False,  # Skip verification for test
        )
        scheduler = SecretsRotationScheduler(config)

        # Register secret first
        scheduler.register_secret(
            secret_id="test_key",
            secret_type=SecretType.API_KEY,
            name="Test Key",
        )

        result = await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="test_key",
            initiated_by="test_user",
        )

        assert result.rotation_id is not None
        assert result.status == RotationStatus.COMPLETED
        assert result.old_version is not None
        assert result.new_version is not None
        assert result.old_version != result.new_version

    @pytest.mark.asyncio
    async def test_rotation_updates_next_rotation(self):
        """Rotation updates the next rotation timestamp."""
        from aragora.scheduler.secrets_rotation_scheduler import (
            SecretsRotationScheduler,
            SecretsRotationConfig,
            SecretType,
        )

        config = SecretsRotationConfig(
            storage_path=":memory:",
            verify_after_rotation=False,
        )
        scheduler = SecretsRotationScheduler(config)

        scheduler.register_secret(
            secret_id="rotating_key",
            secret_type=SecretType.API_KEY,
            name="Rotating Key",
            rotation_interval_days=30,
        )

        original = scheduler.get_secret("rotating_key")
        original_next = original.next_rotation_at

        await scheduler.rotate_secret(
            secret_type=SecretType.API_KEY,
            secret_id="rotating_key",
        )

        updated = scheduler.get_secret("rotating_key")
        assert updated.last_rotated_at is not None
        assert updated.next_rotation_at > original_next

    def test_get_compliance_report(self):
        """Compliance report can be generated."""
        from aragora.scheduler.secrets_rotation_scheduler import (
            SecretsRotationScheduler,
            SecretsRotationConfig,
            SecretType,
        )

        config = SecretsRotationConfig(storage_path=":memory:")
        scheduler = SecretsRotationScheduler(config)

        # Register some secrets
        scheduler.register_secret("key_1", SecretType.API_KEY, "Key 1")
        scheduler.register_secret("key_2", SecretType.JWT_SECRET, "JWT Secret")

        report = scheduler.get_compliance_report()

        assert report["total_secrets"] == 2
        assert "compliance_rate" in report
        assert "rotation_success_rate" in report
        assert "by_type" in report

    def test_rotation_result_to_dict(self):
        """RotationResult converts to dictionary correctly."""
        from aragora.scheduler.secrets_rotation_scheduler import (
            RotationResult,
            SecretType,
            RotationStatus,
            RotationTrigger,
        )

        result = RotationResult(
            rotation_id="rot_123",
            secret_id="key_123",
            secret_type=SecretType.API_KEY,
            status=RotationStatus.COMPLETED,
            trigger=RotationTrigger.SCHEDULED,
            verification_passed=True,
        )

        data = result.to_dict()
        assert data["rotation_id"] == "rot_123"
        assert data["secret_type"] == "api_key"
        assert data["status"] == "completed"
        assert data["verification_passed"] is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestSOC2Integration:
    """Integration tests for SOC 2 compliance modules."""

    @pytest.mark.asyncio
    async def test_anomaly_detection_with_siem_events(self):
        """Anomaly detection integrates with security events."""
        from aragora.security.anomaly_detection import (
            AnomalyDetector,
            AnomalyDetectorConfig,
        )

        config = AnomalyDetectorConfig(storage_path=":memory:")
        detector = AnomalyDetector(config)

        # Simulate a series of events
        for i in range(5):
            await detector.check_auth_event(
                user_id="normal_user",
                success=True,
                ip_address=f"192.168.1.{i}",
            )

        # Get recent anomalies
        anomalies = detector.get_recent_anomalies(hours=1)
        assert isinstance(anomalies, list)

    @pytest.mark.asyncio
    async def test_scheduler_lifecycle(self):
        """Schedulers start and stop cleanly."""
        from aragora.scheduler.access_review_scheduler import (
            AccessReviewScheduler,
            AccessReviewConfig,
        )
        from aragora.scheduler.dr_drill_scheduler import (
            DRDrillScheduler,
            DRDrillConfig,
        )
        from aragora.scheduler.secrets_rotation_scheduler import (
            SecretsRotationScheduler,
            SecretsRotationConfig,
        )

        # Test access review scheduler
        ar_scheduler = AccessReviewScheduler(AccessReviewConfig(storage_path=":memory:"))
        await ar_scheduler.start()
        await ar_scheduler.stop()

        # Test DR drill scheduler
        dr_scheduler = DRDrillScheduler(DRDrillConfig(storage_path=":memory:"))
        await dr_scheduler.start()
        await dr_scheduler.stop()

        # Test secrets rotation scheduler
        sr_scheduler = SecretsRotationScheduler(SecretsRotationConfig(storage_path=":memory:"))
        await sr_scheduler.start()
        await sr_scheduler.stop()

    def test_global_instance_getters(self):
        """Global instance getters work correctly."""
        from aragora.security.anomaly_detection import get_anomaly_detector
        from aragora.scheduler.access_review_scheduler import (
            get_access_review_scheduler,
        )
        from aragora.scheduler.dr_drill_scheduler import get_dr_drill_scheduler
        from aragora.scheduler.secrets_rotation_scheduler import (
            get_secrets_rotation_scheduler,
        )

        # These should return instances without errors
        anomaly = get_anomaly_detector()
        access = get_access_review_scheduler()
        dr = get_dr_drill_scheduler()
        secrets = get_secrets_rotation_scheduler()

        assert anomaly is not None
        assert access is not None
        assert dr is not None
        assert secrets is not None
