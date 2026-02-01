"""Tests for SLO (Service Level Objective) namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient


class TestSLOStatus:
    """Tests for SLO status methods."""

    def test_get_status(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "status": "healthy",
                "slos": [],
                "alerts": [],
                "summary": {"total": 5, "meeting": 5},
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.slo.get_status()
            mock_request.assert_called_once_with("GET", "/api/v2/slo/status")
            assert result["status"] == "healthy"
            assert result["summary"]["meeting"] == 5
            client.close()

    def test_get_slo(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "name": "availability",
                "current_percent": 99.95,
                "target_percent": 99.9,
                "is_meeting": True,
                "total_requests": 100000,
                "successful_requests": 99950,
                "failed_requests": 50,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.slo.get_slo("availability")
            mock_request.assert_called_once_with("GET", "/api/v2/slo/availability")
            assert result["is_meeting"] is True
            assert result["current_percent"] == 99.95
            client.close()


class TestSLOErrorBudget:
    """Tests for error budget methods."""

    def test_get_error_budget(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "slo_name": "latency",
                "budget_percent": 0.1,
                "consumed_percent": 0.03,
                "remaining_percent": 0.07,
                "is_exhausted": False,
                "burn_rate": 1.2,
                "window_days": 30,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.slo.get_error_budget("latency")
            mock_request.assert_called_once_with("GET", "/api/v2/slo/latency/error-budget")
            assert result["remaining_percent"] == 0.07
            assert result["is_exhausted"] is False
            client.close()


class TestSLOViolations:
    """Tests for violation listing methods."""

    def test_list_violations_no_params(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"violations": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.slo.list_violations("availability")
            mock_request.assert_called_once_with(
                "GET", "/api/v2/slo/availability/violations", params={}
            )
            assert result["total"] == 0
            client.close()

    def test_list_violations_with_limit_and_since(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "violations": [
                    {
                        "slo_name": "latency",
                        "severity": "critical",
                        "resolved": False,
                    }
                ]
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.slo.list_violations("latency", limit=10, since="2025-06-01T00:00:00Z")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v2/slo/latency/violations",
                params={"limit": 10, "since": "2025-06-01T00:00:00Z"},
            )
            client.close()


class TestSLOCompliance:
    """Tests for compliance check methods."""

    def test_is_compliant(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "compliant": True,
                "slo_name": "debate-success",
                "current_percent": 98.5,
                "target_percent": 95.0,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.slo.is_compliant("debate-success")
            mock_request.assert_called_once_with("GET", "/api/v2/slo/debate-success/compliant")
            assert result["compliant"] is True
            assert result["current_percent"] > result["target_percent"]
            client.close()


class TestSLOAlerts:
    """Tests for alert retrieval methods."""

    def test_get_alerts_no_params(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "alerts": [
                    {
                        "slo_name": "availability",
                        "severity": "warning",
                        "message": "Error budget burn rate elevated",
                        "acknowledged": False,
                    }
                ]
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.slo.get_alerts("availability")
            mock_request.assert_called_once_with(
                "GET", "/api/v2/slo/availability/alerts", params={}
            )
            assert result["alerts"][0]["severity"] == "warning"
            client.close()

    def test_get_alerts_active_only(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"alerts": []}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.slo.get_alerts("latency", active_only=True)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v2/slo/latency/alerts",
                params={"active_only": True},
            )
            client.close()


class TestAsyncSLO:
    """Tests for async SLO methods."""

    @pytest.mark.asyncio
    async def test_get_status(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"status": "degraded", "slos": [], "alerts": []}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.slo.get_status()
            mock_request.assert_called_once_with("GET", "/api/v2/slo/status")
            assert result["status"] == "degraded"
            await client.close()

    @pytest.mark.asyncio
    async def test_get_error_budget(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "slo_name": "availability",
                "remaining_percent": 0.05,
                "is_exhausted": False,
            }
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.slo.get_error_budget("availability")
            mock_request.assert_called_once_with("GET", "/api/v2/slo/availability/error-budget")
            assert result["remaining_percent"] == 0.05
            await client.close()

    @pytest.mark.asyncio
    async def test_list_violations_with_params(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"violations": [{"severity": "critical"}]}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.slo.list_violations("latency", limit=5)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v2/slo/latency/violations",
                params={"limit": 5},
            )
            assert result["violations"][0]["severity"] == "critical"
            await client.close()

    @pytest.mark.asyncio
    async def test_is_compliant(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"compliant": False, "slo_name": "latency"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.slo.is_compliant("latency")
            mock_request.assert_called_once_with("GET", "/api/v2/slo/latency/compliant")
            assert result["compliant"] is False
            await client.close()
