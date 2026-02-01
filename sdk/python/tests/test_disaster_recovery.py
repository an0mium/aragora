"""Tests for Disaster Recovery namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient


class TestDRStatus:
    """Tests for DR status and objectives methods."""

    def test_get_status(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "ready": True,
                "overall_health": "healthy",
                "issues": [],
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.disaster_recovery.get_status()
            mock_request.assert_called_once_with("GET", "/api/v2/dr/status")
            assert result["ready"] is True
            assert result["overall_health"] == "healthy"
            client.close()

    def test_get_objectives(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "rpo_minutes": 15,
                "rto_minutes": 60,
                "rpo_compliant": True,
                "rto_compliant": True,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.disaster_recovery.get_objectives()
            mock_request.assert_called_once_with("GET", "/api/v2/dr/objectives")
            assert result["rpo_minutes"] == 15
            assert result["rto_compliant"] is True
            client.close()


class TestDRDrills:
    """Tests for DR drill operations."""

    def test_run_drill_minimal(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "drill_id": "drill_001",
                "type": "tabletop",
                "success": True,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.disaster_recovery.run_drill(type="tabletop")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v2/dr/drill",
                json={"type": "tabletop"},
            )
            assert result["drill_id"] == "drill_001"
            assert result["success"] is True
            client.close()

    def test_run_drill_with_all_options(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"drill_id": "drill_002", "type": "full"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.disaster_recovery.run_drill(
                type="full",
                components=["database", "cache"],
                notify_team=True,
                dry_run=False,
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v2/dr/drill",
                json={
                    "type": "full",
                    "components": ["database", "cache"],
                    "notify_team": True,
                    "dry_run": False,
                },
            )
            client.close()


class TestDRValidation:
    """Tests for DR validation methods."""

    def test_validate_defaults(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"valid": True, "overall_score": 95}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.disaster_recovery.validate()
            mock_request.assert_called_once_with(
                "POST",
                "/api/v2/dr/validate",
                json={},
            )
            assert result["valid"] is True
            assert result["overall_score"] == 95
            client.close()

    def test_validate_with_specific_checks(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"valid": False, "overall_score": 60}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.disaster_recovery.validate(
                check_backups=True,
                check_replication=True,
                check_failover=False,
                check_dns=True,
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v2/dr/validate",
                json={
                    "check_backups": True,
                    "check_replication": True,
                    "check_failover": False,
                    "check_dns": True,
                },
            )
            client.close()


class TestDRConvenience:
    """Tests for DR convenience methods."""

    def test_is_ready_returns_true(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.side_effect = [
                {"ready": True, "overall_health": "healthy", "issues": []},
                {"valid": True, "overall_score": 100},
            ]
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.disaster_recovery.is_ready()
            assert result is True
            assert mock_request.call_count == 2
            client.close()

    def test_is_ready_returns_false_when_not_ready(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.side_effect = [
                {"ready": False, "overall_health": "critical", "issues": ["backup stale"]},
                {"valid": True, "overall_score": 90},
            ]
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.disaster_recovery.is_ready()
            assert result is False
            client.close()

    def test_get_health_summary(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.side_effect = [
                {
                    "ready": True,
                    "overall_health": "degraded",
                    "issues": ["replication lag"],
                },
                {
                    "rpo_compliant": True,
                    "rto_compliant": False,
                },
            ]
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.disaster_recovery.get_health_summary()
            assert result["ready"] is True
            assert result["health"] == "degraded"
            assert result["rpo_compliant"] is True
            assert result["rto_compliant"] is False
            assert result["issues_count"] == 1
            assert mock_request.call_count == 2
            client.close()


class TestAsyncDisasterRecovery:
    """Tests for async disaster recovery methods."""

    @pytest.mark.asyncio
    async def test_get_status(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "ready": True,
                "overall_health": "healthy",
                "issues": [],
            }
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.disaster_recovery.get_status()
            mock_request.assert_called_once_with("GET", "/api/v2/dr/status")
            assert result["ready"] is True
            await client.close()

    @pytest.mark.asyncio
    async def test_run_drill(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "drill_id": "drill_async_001",
                "type": "simulation",
                "success": True,
            }
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.disaster_recovery.run_drill(
                type="simulation",
                components=["api", "database"],
                notify_team=True,
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v2/dr/drill",
                json={
                    "type": "simulation",
                    "components": ["api", "database"],
                    "notify_team": True,
                },
            )
            assert result["drill_id"] == "drill_async_001"
            await client.close()

    @pytest.mark.asyncio
    async def test_validate(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"valid": True, "overall_score": 88}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.disaster_recovery.validate(check_backups=True)
            mock_request.assert_called_once_with(
                "POST",
                "/api/v2/dr/validate",
                json={"check_backups": True},
            )
            assert result["valid"] is True
            await client.close()

    @pytest.mark.asyncio
    async def test_get_health_summary(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.side_effect = [
                {"ready": True, "overall_health": "healthy", "issues": []},
                {"rpo_compliant": True, "rto_compliant": True},
            ]
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.disaster_recovery.get_health_summary()
            assert result["ready"] is True
            assert result["health"] == "healthy"
            assert result["rpo_compliant"] is True
            assert result["rto_compliant"] is True
            assert result["issues_count"] == 0
            await client.close()
