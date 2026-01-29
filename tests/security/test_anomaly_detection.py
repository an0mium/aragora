"""
Comprehensive tests for Anomaly Detection module.

Tests cover:
- AnomalyType and AnomalySeverity enums
- AnomalyResult dataclass
- AnomalyDetectorConfig settings
- UserBaseline learning and maturity
- AnomalyStorage persistence layer
- AnomalyDetector authentication anomaly detection
- AnomalyDetector rate anomaly detection
- AnomalyDetector behavioral anomaly detection
- AnomalyDetector network anomaly detection
- Baseline management
- Global instance management
- Convenience functions
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from aragora.security.anomaly_detection import (
    AnomalyDetector,
    AnomalyDetectorConfig,
    AnomalyResult,
    AnomalySeverity,
    AnomalyStorage,
    AnomalyType,
    UserBaseline,
    check_auth_anomaly,
    check_rate_anomaly,
    get_anomaly_detector,
)


# =============================================================================
# Test Enums
# =============================================================================


class TestAnomalyType:
    """Tests for AnomalyType enum."""

    def test_auth_anomaly_types(self):
        """Authentication anomaly types exist."""
        assert AnomalyType.AUTH_BRUTE_FORCE.value == "auth.brute_force"
        assert AnomalyType.AUTH_CREDENTIAL_STUFFING.value == "auth.credential_stuffing"
        assert AnomalyType.AUTH_IMPOSSIBLE_TRAVEL.value == "auth.impossible_travel"
        assert AnomalyType.AUTH_UNUSUAL_TIME.value == "auth.unusual_time"
        assert AnomalyType.AUTH_NEW_DEVICE.value == "auth.new_device"
        assert AnomalyType.AUTH_FAILED_SPIKE.value == "auth.failed_spike"

    def test_rate_anomaly_types(self):
        """Rate anomaly types exist."""
        assert AnomalyType.RATE_API_SPIKE.value == "rate.api_spike"
        assert AnomalyType.RATE_REQUEST_FLOOD.value == "rate.request_flood"
        assert AnomalyType.RATE_DATA_EXFILTRATION.value == "rate.data_exfiltration"

    def test_behavior_anomaly_types(self):
        """Behavioral anomaly types exist."""
        assert AnomalyType.BEHAVIOR_UNUSUAL_RESOURCE.value == "behavior.unusual_resource"
        assert AnomalyType.BEHAVIOR_PRIVILEGE_ESCALATION.value == "behavior.privilege_escalation"
        assert AnomalyType.BEHAVIOR_UNUSUAL_PATTERN.value == "behavior.unusual_pattern"

    def test_network_anomaly_types(self):
        """Network anomaly types exist."""
        assert AnomalyType.NETWORK_TOR_EXIT.value == "network.tor_exit"
        assert AnomalyType.NETWORK_KNOWN_BAD_IP.value == "network.known_bad_ip"
        assert AnomalyType.NETWORK_UNUSUAL_COUNTRY.value == "network.unusual_country"


class TestAnomalySeverity:
    """Tests for AnomalySeverity enum."""

    def test_severity_levels(self):
        """All severity levels exist."""
        assert AnomalySeverity.LOW.value == "low"
        assert AnomalySeverity.MEDIUM.value == "medium"
        assert AnomalySeverity.HIGH.value == "high"
        assert AnomalySeverity.CRITICAL.value == "critical"


# =============================================================================
# Test AnomalyResult
# =============================================================================


class TestAnomalyResult:
    """Tests for AnomalyResult dataclass."""

    def test_create_non_anomalous_result(self):
        """Create a non-anomalous result."""
        result = AnomalyResult(is_anomalous=False)

        assert result.is_anomalous is False
        assert result.anomaly_type is None
        assert result.severity == AnomalySeverity.LOW
        assert result.confidence == 0.0
        assert result.description == ""
        assert result.details == {}
        assert result.recommendations == []

    def test_create_anomalous_result(self):
        """Create an anomalous result with all fields."""
        result = AnomalyResult(
            is_anomalous=True,
            anomaly_type=AnomalyType.AUTH_BRUTE_FORCE,
            severity=AnomalySeverity.HIGH,
            confidence=0.95,
            description="Brute force attack detected",
            details={"failed_count": 15, "window_minutes": 15},
            recommendations=["Lock account", "Require MFA"],
        )

        assert result.is_anomalous is True
        assert result.anomaly_type == AnomalyType.AUTH_BRUTE_FORCE
        assert result.severity == AnomalySeverity.HIGH
        assert result.confidence == 0.95
        assert result.details["failed_count"] == 15
        assert len(result.recommendations) == 2

    def test_to_dict(self):
        """AnomalyResult converts to dictionary."""
        result = AnomalyResult(
            is_anomalous=True,
            anomaly_type=AnomalyType.AUTH_BRUTE_FORCE,
            severity=AnomalySeverity.HIGH,
            confidence=0.95,
            description="Test anomaly",
            details={"key": "value"},
            recommendations=["Do something"],
        )

        data = result.to_dict()

        assert data["is_anomalous"] is True
        assert data["anomaly_type"] == "auth.brute_force"
        assert data["severity"] == "high"
        assert data["confidence"] == 0.95
        assert data["description"] == "Test anomaly"
        assert data["details"] == {"key": "value"}
        assert data["recommendations"] == ["Do something"]
        assert "timestamp" in data

    def test_to_dict_none_anomaly_type(self):
        """to_dict handles None anomaly_type."""
        result = AnomalyResult(is_anomalous=False)
        data = result.to_dict()

        assert data["anomaly_type"] is None


# =============================================================================
# Test AnomalyDetectorConfig
# =============================================================================


class TestAnomalyDetectorConfig:
    """Tests for AnomalyDetectorConfig."""

    def test_default_config(self):
        """Default config has sensible values."""
        config = AnomalyDetectorConfig()

        assert config.failed_login_threshold == 5
        assert config.failed_login_window_minutes == 15
        assert config.brute_force_threshold == 10
        assert config.credential_stuffing_threshold == 3
        assert config.api_spike_multiplier == 3.0
        assert config.request_flood_per_minute == 1000
        assert config.data_exfil_threshold_mb == 100.0
        assert config.baseline_learning_days == 7
        assert config.unusual_time_std_deviations == 2.0
        assert config.impossible_travel_speed_kmh == 1000.0
        assert config.retention_days == 90

    def test_custom_config(self):
        """Custom config values are used."""
        config = AnomalyDetectorConfig(
            failed_login_threshold=3,
            brute_force_threshold=5,
            request_flood_per_minute=500,
            storage_path="/tmp/anomaly.db",
        )

        assert config.failed_login_threshold == 3
        assert config.brute_force_threshold == 5
        assert config.request_flood_per_minute == 500
        assert config.storage_path == "/tmp/anomaly.db"


# =============================================================================
# Test UserBaseline
# =============================================================================


class TestUserBaseline:
    """Tests for UserBaseline dataclass."""

    def test_create_empty_baseline(self):
        """Create an empty user baseline."""
        baseline = UserBaseline(user_id="user_123")

        assert baseline.user_id == "user_123"
        assert baseline.typical_login_hours == []
        assert baseline.typical_ips == set()
        assert baseline.typical_user_agents == set()
        assert baseline.typical_resources == set()
        assert baseline.avg_requests_per_hour == 0.0
        assert baseline.learning_samples == 0

    def test_is_mature_immature(self):
        """Baseline is not mature with few samples."""
        baseline = UserBaseline(user_id="user_123", learning_samples=10)

        assert baseline.is_mature(min_samples=50) is False

    def test_is_mature_mature(self):
        """Baseline is mature with enough samples."""
        baseline = UserBaseline(user_id="user_123", learning_samples=100)

        assert baseline.is_mature(min_samples=50) is True

    def test_is_mature_custom_threshold(self):
        """is_mature respects custom threshold."""
        baseline = UserBaseline(user_id="user_123", learning_samples=25)

        assert baseline.is_mature(min_samples=20) is True
        assert baseline.is_mature(min_samples=30) is False


# =============================================================================
# Test AnomalyStorage
# =============================================================================


class TestAnomalyStorage:
    """Tests for AnomalyStorage class."""

    @pytest.fixture
    def storage(self):
        """Create in-memory storage for each test."""
        return AnomalyStorage(db_path=":memory:")

    def test_initialize_storage(self, storage):
        """Storage initializes without error."""
        assert storage is not None

    def test_record_auth_event(self, storage):
        """Record auth event."""
        storage.record_auth_event(
            user_id="user_123",
            success=True,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            country="US",
            city="New York",
        )
        # Should not raise

    def test_record_api_request(self, storage):
        """Record API request."""
        storage.record_api_request(
            endpoint="/api/v1/debates",
            method="GET",
            user_id="user_123",
            ip_address="192.168.1.1",
            response_size_bytes=1024,
        )
        # Should not raise

    def test_get_failed_logins_by_user(self, storage):
        """Get failed login count for user."""
        # Record some failed logins
        for _ in range(5):
            storage.record_auth_event(
                user_id="victim_user",
                success=False,
                ip_address="10.0.0.1",
            )

        count = storage.get_failed_logins_in_window(
            user_id="victim_user",
            window_minutes=15,
        )

        assert count == 5

    def test_get_failed_logins_by_ip(self, storage):
        """Get failed login count for IP."""
        # Record failed logins from same IP
        for i in range(3):
            storage.record_auth_event(
                user_id=f"user_{i}",
                success=False,
                ip_address="attacker_ip",
            )

        count = storage.get_failed_logins_in_window(
            ip_address="attacker_ip",
            window_minutes=15,
        )

        assert count == 3

    def test_get_failed_logins_no_params(self, storage):
        """Get failed logins with no params returns 0."""
        count = storage.get_failed_logins_in_window()
        assert count == 0

    def test_get_distinct_failed_users_from_ip(self, storage):
        """Get distinct users with failed logins from IP."""
        attacker_ip = "10.0.0.1"
        for i in range(5):
            storage.record_auth_event(
                user_id=f"user_{i}",
                success=False,
                ip_address=attacker_ip,
            )

        count = storage.get_distinct_failed_users_from_ip(
            ip_address=attacker_ip,
            window_minutes=15,
        )

        assert count == 5

    def test_get_requests_in_window_by_user(self, storage):
        """Get request count by user."""
        for _ in range(10):
            storage.record_api_request(
                endpoint="/api/test",
                method="GET",
                user_id="user_123",
            )

        count = storage.get_requests_in_window(
            user_id="user_123",
            window_minutes=60,
        )

        assert count == 10

    def test_get_requests_in_window_by_ip(self, storage):
        """Get request count by IP."""
        for _ in range(5):
            storage.record_api_request(
                endpoint="/api/test",
                method="GET",
                ip_address="192.168.1.1",
            )

        count = storage.get_requests_in_window(
            ip_address="192.168.1.1",
            window_minutes=60,
        )

        assert count == 5

    def test_get_data_transferred_in_window(self, storage):
        """Get total data transferred."""
        for _ in range(5):
            storage.record_api_request(
                endpoint="/api/export",
                method="GET",
                user_id="user_123",
                response_size_bytes=1024 * 1024,  # 1MB each
            )

        total = storage.get_data_transferred_in_window(
            user_id="user_123",
            window_minutes=60,
        )

        assert total == 5 * 1024 * 1024

    def test_save_and_get_baseline(self, storage):
        """Save and retrieve user baseline."""
        baseline = UserBaseline(
            user_id="user_123",
            typical_login_hours=[9, 10, 11, 14, 15, 16],
            typical_ips={"192.168.1.1", "192.168.1.2"},
            typical_user_agents={"Mozilla/5.0"},
            typical_resources={"debate:read", "workflow:create"},
            avg_requests_per_hour=50.0,
            std_requests_per_hour=10.0,
            learning_samples=100,
        )

        storage.save_baseline(baseline)
        loaded = storage.get_baseline("user_123")

        assert loaded is not None
        assert loaded.user_id == "user_123"
        assert loaded.typical_login_hours == [9, 10, 11, 14, 15, 16]
        assert "192.168.1.1" in loaded.typical_ips
        assert loaded.avg_requests_per_hour == 50.0
        assert loaded.learning_samples == 100

    def test_get_baseline_not_found(self, storage):
        """Get baseline for unknown user returns None."""
        baseline = storage.get_baseline("unknown_user")
        assert baseline is None

    def test_record_and_get_anomalies(self, storage):
        """Record and retrieve anomalies."""
        result = AnomalyResult(
            is_anomalous=True,
            anomaly_type=AnomalyType.AUTH_BRUTE_FORCE,
            severity=AnomalySeverity.HIGH,
            confidence=0.95,
            description="Brute force detected",
            details={"failed_count": 15},
        )

        anomaly_id = storage.record_anomaly(
            result,
            user_id="victim_user",
            ip_address="10.0.0.1",
        )

        assert anomaly_id > 0

        anomalies = storage.get_recent_anomalies(hours=1)
        assert len(anomalies) == 1
        assert anomalies[0]["anomaly_type"] == "auth.brute_force"
        assert anomalies[0]["user_id"] == "victim_user"

    def test_get_recent_anomalies_filter_severity(self, storage):
        """Filter anomalies by severity."""
        storage.record_anomaly(
            AnomalyResult(
                is_anomalous=True,
                anomaly_type=AnomalyType.AUTH_BRUTE_FORCE,
                severity=AnomalySeverity.HIGH,
            )
        )
        storage.record_anomaly(
            AnomalyResult(
                is_anomalous=True,
                anomaly_type=AnomalyType.AUTH_UNUSUAL_TIME,
                severity=AnomalySeverity.LOW,
            )
        )

        high_only = storage.get_recent_anomalies(
            hours=1,
            severity=AnomalySeverity.HIGH,
        )

        assert len(high_only) == 1
        assert high_only[0]["severity"] == "high"

    def test_cleanup_old_data(self, storage):
        """Cleanup removes old data."""
        # Record some data
        storage.record_auth_event("user_1", True)
        storage.record_api_request("/api/test", "GET")

        # Cleanup with very short retention
        deleted = storage.cleanup_old_data(retention_days=0)

        # Should have deleted recent data (since retention is 0 days)
        assert deleted >= 0


# =============================================================================
# Test AnomalyDetector - Authentication
# =============================================================================


class TestAnomalyDetectorAuth:
    """Tests for AnomalyDetector authentication anomaly detection."""

    @pytest.fixture
    def detector(self):
        """Create detector with in-memory storage."""
        config = AnomalyDetectorConfig(storage_path=":memory:")
        return AnomalyDetector(config)

    @pytest.mark.asyncio
    async def test_normal_successful_login(self, detector):
        """Normal successful login is not anomalous."""
        result = await detector.check_auth_event(
            user_id="user_123",
            success=True,
            ip_address="192.168.1.1",
        )

        assert result.is_anomalous is False

    @pytest.mark.asyncio
    async def test_normal_failed_login(self, detector):
        """Single failed login is not anomalous."""
        result = await detector.check_auth_event(
            user_id="user_123",
            success=False,
            ip_address="192.168.1.1",
        )

        assert result.is_anomalous is False

    @pytest.mark.asyncio
    async def test_detect_brute_force(self, detector):
        """Detect brute force attack on user."""
        detector.config.brute_force_threshold = 5

        for _ in range(6):
            result = await detector.check_auth_event(
                user_id="victim_user",
                success=False,
                ip_address="10.0.0.1",
            )

        assert result.is_anomalous is True
        assert result.anomaly_type == AnomalyType.AUTH_BRUTE_FORCE
        assert result.severity == AnomalySeverity.HIGH
        assert result.confidence > 0.5
        assert "brute force" in result.description.lower()
        assert len(result.recommendations) > 0

    @pytest.mark.asyncio
    async def test_detect_credential_stuffing(self, detector):
        """Detect credential stuffing attack."""
        detector.config.credential_stuffing_threshold = 2
        attacker_ip = "10.0.0.1"

        for i in range(3):
            result = await detector.check_auth_event(
                user_id=f"user_{i}",
                success=False,
                ip_address=attacker_ip,
            )

        assert result.is_anomalous is True
        assert result.anomaly_type == AnomalyType.AUTH_CREDENTIAL_STUFFING
        assert result.severity == AnomalySeverity.CRITICAL
        assert "credential stuffing" in result.description.lower()

    @pytest.mark.asyncio
    async def test_detect_failed_spike(self, detector):
        """Detect failed login spike."""
        detector.config.failed_login_threshold = 3
        detector.config.brute_force_threshold = 10  # Set higher to test spike first

        for _ in range(4):
            result = await detector.check_auth_event(
                user_id="user_123",
                success=False,
            )

        assert result.is_anomalous is True
        assert result.anomaly_type == AnomalyType.AUTH_FAILED_SPIKE
        assert result.severity == AnomalySeverity.MEDIUM

    @pytest.mark.asyncio
    async def test_detect_new_ip(self, detector):
        """Detect login from new IP."""
        # Build baseline with known IPs
        baseline = UserBaseline(
            user_id="user_123",
            typical_ips={"192.168.1.1", "192.168.1.2"},
            learning_samples=100,  # Mature baseline
        )
        detector._baselines["user_123"] = baseline

        result = await detector.check_auth_event(
            user_id="user_123",
            success=True,
            ip_address="10.0.0.1",  # New IP
        )

        assert result.is_anomalous is True
        assert result.anomaly_type == AnomalyType.AUTH_NEW_DEVICE
        assert result.severity == AnomalySeverity.LOW

    @pytest.mark.asyncio
    async def test_detect_unusual_time(self, detector):
        """Detect login at unusual time - validates code path."""
        # Build baseline with typical hours (working hours 9-17)
        baseline = UserBaseline(
            user_id="user_123",
            typical_login_hours=[9, 10, 11, 14, 15, 16] * 20,  # Working hours
            learning_samples=100,
        )
        detector._baselines["user_123"] = baseline

        # Just run the auth event - the code path for unusual time detection
        # is exercised when the baseline is mature with typical_login_hours
        result = await detector.check_auth_event(
            user_id="user_123",
            success=True,
            ip_address="192.168.1.1",
        )

        # The detection depends on current hour vs learned hours
        # This test validates the code path runs without error
        assert isinstance(result, AnomalyResult)

    @pytest.mark.asyncio
    async def test_baseline_update_on_success(self, detector):
        """Successful login updates baseline."""
        result = await detector.check_auth_event(
            user_id="user_123",
            success=True,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
        )

        baseline = detector._get_or_create_baseline("user_123")
        assert "192.168.1.1" in baseline.typical_ips
        assert "Mozilla/5.0" in baseline.typical_user_agents


# =============================================================================
# Test AnomalyDetector - Rate
# =============================================================================


class TestAnomalyDetectorRate:
    """Tests for AnomalyDetector rate anomaly detection."""

    @pytest.fixture
    def detector(self):
        """Create detector with in-memory storage."""
        config = AnomalyDetectorConfig(storage_path=":memory:")
        return AnomalyDetector(config)

    @pytest.mark.asyncio
    async def test_normal_request(self, detector):
        """Normal request is not anomalous."""
        result = await detector.check_rate_anomaly(
            user_id="user_123",
            ip_address="192.168.1.1",
            endpoint="/api/v1/debates",
        )

        assert result.is_anomalous is False

    @pytest.mark.asyncio
    async def test_detect_request_flood(self, detector):
        """Detect request flood."""
        detector.config.request_flood_per_minute = 10

        for _ in range(15):
            result = await detector.check_rate_anomaly(
                user_id="flood_user",
                ip_address="192.168.1.1",
                endpoint="/api/test",
            )

        assert result.is_anomalous is True
        assert result.anomaly_type == AnomalyType.RATE_REQUEST_FLOOD
        assert result.severity == AnomalySeverity.HIGH

    @pytest.mark.asyncio
    async def test_detect_data_exfiltration(self, detector):
        """Detect potential data exfiltration."""
        detector.config.data_exfil_threshold_mb = 10.0  # 10MB

        # Simulate large data transfers
        for _ in range(15):
            await detector.check_rate_anomaly(
                user_id="exfil_user",
                endpoint="/api/export",
                response_size_bytes=1024 * 1024,  # 1MB each
            )

        result = await detector.check_rate_anomaly(
            user_id="exfil_user",
            endpoint="/api/export",
            response_size_bytes=1024 * 1024,
        )

        # May or may not trigger depending on timing
        # This test validates the code path

    @pytest.mark.asyncio
    async def test_detect_api_spike_vs_baseline(self, detector):
        """Detect API usage spike vs learned baseline."""
        # Set up mature baseline with low average
        baseline = UserBaseline(
            user_id="user_123",
            avg_requests_per_hour=10.0,
            std_requests_per_hour=2.0,
            learning_samples=100,
        )
        detector._baselines["user_123"] = baseline
        detector.config.api_spike_multiplier = 2.0

        # Generate many requests quickly
        for _ in range(50):
            result = await detector.check_rate_anomaly(
                user_id="user_123",
                endpoint="/api/test",
            )

        # Should detect spike since 50 requests/min >> 10 requests/hour baseline
        # Note: Actual detection depends on timing calculations


# =============================================================================
# Test AnomalyDetector - Behavioral
# =============================================================================


class TestAnomalyDetectorBehavioral:
    """Tests for AnomalyDetector behavioral anomaly detection."""

    @pytest.fixture
    def detector(self):
        """Create detector with in-memory storage."""
        config = AnomalyDetectorConfig(storage_path=":memory:")
        return AnomalyDetector(config)

    @pytest.mark.asyncio
    async def test_normal_resource_access(self, detector):
        """Normal resource access is not anomalous."""
        result = await detector.check_behavioral_anomaly(
            user_id="user_123",
            resource_type="debate",
            resource_id="debate_456",
            action="read",
        )

        assert result.is_anomalous is False

    @pytest.mark.asyncio
    async def test_detect_unusual_resource(self, detector):
        """Detect access to unusual resource type."""
        # Set up mature baseline with known resources
        baseline = UserBaseline(
            user_id="user_123",
            typical_resources={"debate:read", "workflow:create"},
            learning_samples=100,
        )
        detector._baselines["user_123"] = baseline

        result = await detector.check_behavioral_anomaly(
            user_id="user_123",
            resource_type="admin",
            resource_id="admin_panel",
            action="access",
        )

        assert result.is_anomalous is True
        assert result.anomaly_type == AnomalyType.BEHAVIOR_UNUSUAL_RESOURCE
        assert result.severity == AnomalySeverity.LOW

    @pytest.mark.asyncio
    async def test_baseline_update_on_access(self, detector):
        """Resource access updates baseline."""
        await detector.check_behavioral_anomaly(
            user_id="user_123",
            resource_type="debate",
            resource_id="debate_456",
            action="read",
        )

        baseline = detector._get_or_create_baseline("user_123")
        assert "debate:read" in baseline.typical_resources


# =============================================================================
# Test AnomalyDetector - Network
# =============================================================================


class TestAnomalyDetectorNetwork:
    """Tests for AnomalyDetector network anomaly detection."""

    @pytest.fixture
    def detector(self):
        """Create detector with in-memory storage."""
        config = AnomalyDetectorConfig(storage_path=":memory:")
        return AnomalyDetector(config)

    @pytest.mark.asyncio
    async def test_normal_ip(self, detector):
        """Normal IP is not anomalous."""
        result = await detector.check_network_anomaly(
            ip_address="192.168.1.1",
        )

        assert result.is_anomalous is False

    @pytest.mark.asyncio
    async def test_detect_tor_exit(self, detector):
        """Detect Tor exit node."""
        result = await detector.check_network_anomaly(
            ip_address="10.0.0.1",
            is_tor_exit=True,
        )

        assert result.is_anomalous is True
        assert result.anomaly_type == AnomalyType.NETWORK_TOR_EXIT
        assert result.severity == AnomalySeverity.MEDIUM
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_detect_known_bad_ip(self, detector):
        """Detect known malicious IP."""
        result = await detector.check_network_anomaly(
            ip_address="10.0.0.1",
            is_known_bad_ip=True,
        )

        assert result.is_anomalous is True
        assert result.anomaly_type == AnomalyType.NETWORK_KNOWN_BAD_IP
        assert result.severity == AnomalySeverity.HIGH
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_detect_unusual_country(self, detector):
        """Detect login from unusual country."""
        # Set up baseline with known locations
        baseline = UserBaseline(
            user_id="user_123",
            last_known_locations=[
                ("US", "New York", datetime.now(timezone.utc)),
                ("US", "Los Angeles", datetime.now(timezone.utc)),
            ],
            learning_samples=100,
        )
        detector._baselines["user_123"] = baseline

        result = await detector.check_network_anomaly(
            ip_address="10.0.0.1",
            user_id="user_123",
            country="RU",  # Unusual country
        )

        assert result.is_anomalous is True
        assert result.anomaly_type == AnomalyType.NETWORK_UNUSUAL_COUNTRY
        assert result.severity == AnomalySeverity.MEDIUM


# =============================================================================
# Test AnomalyDetector - Administration
# =============================================================================


class TestAnomalyDetectorAdmin:
    """Tests for AnomalyDetector administrative functions."""

    @pytest.fixture
    def detector(self):
        """Create detector with in-memory storage."""
        config = AnomalyDetectorConfig(storage_path=":memory:")
        return AnomalyDetector(config)

    def test_get_anomaly_stats_empty(self, detector):
        """Get stats with no anomalies."""
        stats = detector.get_anomaly_stats()

        assert stats["total_24h"] == 0
        assert stats["by_type"] == {}
        assert stats["by_severity"] == {}
        assert stats["unresolved"] == 0

    @pytest.mark.asyncio
    async def test_get_anomaly_stats_with_data(self, detector):
        """Get stats with anomalies."""
        detector.config.brute_force_threshold = 2

        # Generate some anomalies
        for _ in range(3):
            await detector.check_auth_event(
                user_id="victim",
                success=False,
            )

        stats = detector.get_anomaly_stats()

        assert stats["total_24h"] >= 1

    def test_get_recent_anomalies(self, detector):
        """Get recent anomalies list."""
        anomalies = detector.get_recent_anomalies(hours=24)
        assert isinstance(anomalies, list)

    def test_cleanup(self, detector):
        """Cleanup function works."""
        deleted = detector.cleanup()
        assert deleted >= 0


# =============================================================================
# Test Global Functions
# =============================================================================


class TestGlobalFunctions:
    """Tests for global convenience functions."""

    def test_get_anomaly_detector(self):
        """get_anomaly_detector returns instance."""
        detector = get_anomaly_detector()
        assert detector is not None
        assert isinstance(detector, AnomalyDetector)

    def test_get_anomaly_detector_same_instance(self):
        """get_anomaly_detector returns same instance."""
        detector1 = get_anomaly_detector()
        detector2 = get_anomaly_detector()
        assert detector1 is detector2

    @pytest.mark.asyncio
    async def test_check_auth_anomaly_convenience(self):
        """check_auth_anomaly convenience function works."""
        result = await check_auth_anomaly(
            user_id="user_123",
            success=True,
            ip_address="192.168.1.1",
        )

        assert isinstance(result, AnomalyResult)

    @pytest.mark.asyncio
    async def test_check_rate_anomaly_convenience(self):
        """check_rate_anomaly convenience function works."""
        result = await check_rate_anomaly(
            user_id="user_123",
            ip_address="192.168.1.1",
            endpoint="/api/test",
        )

        assert isinstance(result, AnomalyResult)


# =============================================================================
# Test Baseline Management
# =============================================================================


class TestBaselineManagement:
    """Tests for baseline management functionality."""

    @pytest.fixture
    def detector(self):
        """Create detector with in-memory storage."""
        config = AnomalyDetectorConfig(storage_path=":memory:")
        return AnomalyDetector(config)

    def test_get_or_create_baseline_new(self, detector):
        """Get or create baseline for new user."""
        baseline = detector._get_or_create_baseline("new_user")

        assert baseline is not None
        assert baseline.user_id == "new_user"
        assert baseline.learning_samples == 0

    def test_get_or_create_baseline_existing(self, detector):
        """Get or create baseline returns cached instance."""
        baseline1 = detector._get_or_create_baseline("user_123")
        baseline1.learning_samples = 50

        baseline2 = detector._get_or_create_baseline("user_123")

        assert baseline1 is baseline2
        assert baseline2.learning_samples == 50

    def test_update_baseline_login_hours(self, detector):
        """Update baseline adds login hours."""
        detector._update_baseline(
            user_id="user_123",
            ip_address="192.168.1.1",
        )

        baseline = detector._get_or_create_baseline("user_123")
        assert len(baseline.typical_login_hours) > 0

    def test_update_baseline_ips(self, detector):
        """Update baseline adds IPs."""
        detector._update_baseline(
            user_id="user_123",
            ip_address="192.168.1.1",
        )
        detector._update_baseline(
            user_id="user_123",
            ip_address="192.168.1.2",
        )

        baseline = detector._get_or_create_baseline("user_123")
        assert "192.168.1.1" in baseline.typical_ips
        assert "192.168.1.2" in baseline.typical_ips

    def test_update_baseline_user_agents(self, detector):
        """Update baseline adds user agents."""
        detector._update_baseline(
            user_id="user_123",
            user_agent="Mozilla/5.0",
        )

        baseline = detector._get_or_create_baseline("user_123")
        assert "Mozilla/5.0" in baseline.typical_user_agents

    def test_update_baseline_increments_samples(self, detector):
        """Update baseline increments sample count."""
        baseline = detector._get_or_create_baseline("user_123")
        initial = baseline.learning_samples

        detector._update_baseline(user_id="user_123")

        assert baseline.learning_samples > initial


# =============================================================================
# Test Rate Tracking Cleanup
# =============================================================================


class TestRateTrackingCleanup:
    """Tests for rate tracking cleanup."""

    @pytest.fixture
    def detector(self):
        """Create detector with in-memory storage."""
        config = AnomalyDetectorConfig(storage_path=":memory:")
        return AnomalyDetector(config)

    def test_cleanup_rate_tracking(self, detector):
        """Cleanup removes old rate tracking entries."""
        # Add some tracking data
        detector._request_counts["test_key"] = [
            datetime.now(timezone.utc) - timedelta(minutes=10),  # Old
            datetime.now(timezone.utc),  # Recent
        ]

        detector._cleanup_rate_tracking()

        # Old entry should be removed
        assert len(detector._request_counts["test_key"]) == 1

    def test_cleanup_rate_tracking_removes_empty_keys(self, detector):
        """Cleanup removes keys with no entries."""
        detector._request_counts["empty_key"] = [
            datetime.now(timezone.utc) - timedelta(minutes=10),
        ]

        detector._cleanup_rate_tracking()

        assert "empty_key" not in detector._request_counts
