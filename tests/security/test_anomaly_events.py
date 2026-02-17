"""Tests for anomaly detection event emissions."""

from __future__ import annotations

from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from aragora.security.anomaly_detection import (
    AnomalyDetector,
    AnomalyDetectorConfig,
    AnomalyResult,
    AnomalySeverity,
    AnomalyType,
)


@pytest.fixture
def detector():
    config = AnomalyDetectorConfig(
        failed_login_threshold=2,
        brute_force_threshold=3,
        credential_stuffing_threshold=2,
    )
    return AnomalyDetector(config=config)


class TestAnomalyEventEmission:
    """Tests for _emit_anomaly_event."""

    def test_emits_risk_warning_for_anomaly(self) -> None:
        detector = AnomalyDetector()
        result = AnomalyResult(
            is_anomalous=True,
            anomaly_type=AnomalyType.AUTH_BRUTE_FORCE,
            severity=AnomalySeverity.HIGH,
            confidence=0.85,
            description="Brute force detected",
        )

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            detector._emit_anomaly_event(result, user_id="user1", ip_address="1.2.3.4")

        mock_dispatch.assert_called_once()
        data = mock_dispatch.call_args[0][1]
        assert data["risk_type"] == "security_anomaly"
        assert data["severity"] == "high"
        assert data["anomaly_type"] == "auth.brute_force"
        assert data["confidence"] == 0.85
        assert data["user_id"] == "user1"
        assert data["ip_address"] == "1.2.3.4"

    def test_skips_non_anomalous_result(self) -> None:
        detector = AnomalyDetector()
        result = AnomalyResult(is_anomalous=False)

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            detector._emit_anomaly_event(result)
            mock_dispatch.assert_not_called()

    def test_handles_import_error(self) -> None:
        detector = AnomalyDetector()
        result = AnomalyResult(
            is_anomalous=True,
            anomaly_type=AnomalyType.RATE_API_SPIKE,
            severity=AnomalySeverity.MEDIUM,
            confidence=0.6,
            description="Spike detected",
        )

        with patch(
            "aragora.events.dispatcher.dispatch_event",
            side_effect=ImportError("no module"),
        ):
            # Should not raise
            detector._emit_anomaly_event(result)

    def test_truncates_long_description(self) -> None:
        detector = AnomalyDetector()
        result = AnomalyResult(
            is_anomalous=True,
            anomaly_type=AnomalyType.AUTH_BRUTE_FORCE,
            severity=AnomalySeverity.HIGH,
            confidence=0.9,
            description="x" * 1000,
        )

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            detector._emit_anomaly_event(result)

        data = mock_dispatch.call_args[0][1]
        assert len(data["description"]) == 500

    def test_handles_none_user_and_ip(self) -> None:
        detector = AnomalyDetector()
        result = AnomalyResult(
            is_anomalous=True,
            anomaly_type=AnomalyType.BEHAVIOR_UNUSUAL_PATTERN,
            severity=AnomalySeverity.LOW,
            confidence=0.5,
            description="Unusual pattern",
        )

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            detector._emit_anomaly_event(result)

        data = mock_dispatch.call_args[0][1]
        assert data["user_id"] == ""
        assert data["ip_address"] == ""

    def test_event_type_is_risk_warning(self) -> None:
        detector = AnomalyDetector()
        result = AnomalyResult(
            is_anomalous=True,
            anomaly_type=AnomalyType.AUTH_CREDENTIAL_STUFFING,
            severity=AnomalySeverity.CRITICAL,
            confidence=0.95,
            description="Credential stuffing",
        )

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            detector._emit_anomaly_event(result)

        assert mock_dispatch.call_args[0][0] == "risk_warning"


class TestBruteForceEmitsEvent:
    """Test that brute force detection triggers event emission."""

    @pytest.mark.asyncio
    async def test_brute_force_emits_event(self, detector) -> None:
        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            # Fill up failures to trigger brute force
            for _ in range(3):
                await detector.check_auth_event(
                    user_id="victim",
                    success=False,
                    ip_address="10.0.0.1",
                )

            # Should have emitted at least one event
            assert mock_dispatch.call_count >= 1
            # Find the brute force event
            calls = [c for c in mock_dispatch.call_args_list if c[0][0] == "risk_warning"]
            assert len(calls) >= 1
            data = calls[-1][0][1]
            assert data["risk_type"] == "security_anomaly"
            assert "brute" in data["anomaly_type"] or "failed" in data["anomaly_type"]


class TestCredentialStuffingEmitsEvent:
    """Test that credential stuffing detection triggers event emission."""

    @pytest.mark.asyncio
    async def test_credential_stuffing_emits_event(self, detector) -> None:
        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            # Multiple users from same IP
            for i in range(3):
                await detector.check_auth_event(
                    user_id=f"user_{i}",
                    success=False,
                    ip_address="malicious_ip",
                )

            calls = [c for c in mock_dispatch.call_args_list if c[0][0] == "risk_warning"]
            assert len(calls) >= 1
