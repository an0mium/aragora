"""Tests for Audit namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient

# =========================================================================
# Event Operations
# =========================================================================

# =========================================================================
# Export Operations
# =========================================================================

# =========================================================================
# Compliance Operations
# =========================================================================

# =========================================================================
# Activity Operations
# =========================================================================

# =========================================================================
# Statistics Operations
# =========================================================================

# =========================================================================
# Entry and Report Operations (OpenAPI-aligned)
# =========================================================================


class TestAuditEntriesAndReport:
    """Tests for entries and report methods."""

    def test_list_entries(self) -> None:
        """List audit entries."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"entries": [], "total": 0}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.audit.list_entries(limit=25, offset=5)

            mock_request.assert_called_once_with(
                "GET", "/api/v1/audit/entries", params={"limit": 25, "offset": 5}
            )
            client.close()

    def test_get_report(self) -> None:
        """Get audit report."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"report": {}}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.audit.get_report()

            mock_request.assert_called_once_with("GET", "/api/v1/audit/report")
            assert "report" in result
            client.close()

    def test_verify_integrity(self) -> None:
        """Verify audit integrity."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"valid": True, "last_verified": "2025-01-31"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.audit.verify()

            mock_request.assert_called_once_with("GET", "/api/v1/audit/verify")
            assert result["valid"] is True
            client.close()


# =========================================================================
# Session Operations
# =========================================================================


class TestAuditSessions:
    """Tests for audit session operations."""

    def test_list_sessions(self) -> None:
        """List audit sessions."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"sessions": [], "total": 0}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.audit.list_sessions(limit=10, offset=0)

            mock_request.assert_called_once_with(
                "GET", "/api/v1/audit/sessions", params={"limit": 10, "offset": 0}
            )
            client.close()

    def test_create_session(self) -> None:
        """Create an audit session."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "sess_123", "name": "Q1 Audit"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.audit.create_session("Q1 Audit")

            mock_request.assert_called_once_with(
                "POST", "/api/v1/audit/sessions", json={"name": "Q1 Audit"}
            )
            assert result["name"] == "Q1 Audit"
            client.close()

    def test_create_session_with_config(self) -> None:
        """Create an audit session with configuration."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "sess_123"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            config = {"scope": "all", "frameworks": ["soc2"]}
            client.audit.create_session("Q1 Audit", config=config)

            call_args = mock_request.call_args
            json_data = call_args[1]["json"]
            assert json_data["config"] == config
            client.close()

    def test_get_session(self) -> None:
        """Get an audit session."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "sess_123", "status": "running"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.audit.get_session("sess_123")

            mock_request.assert_called_once_with("GET", "/api/v1/audit/sessions/sess_123")
            assert result["status"] == "running"
            client.close()

    def test_delete_session(self) -> None:
        """Delete an audit session."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"deleted": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.audit.delete_session("sess_123")

            mock_request.assert_called_once_with("DELETE", "/api/v1/audit/sessions/sess_123")
            assert result["deleted"] is True
            client.close()

    def test_get_session_events(self) -> None:
        """Get events for an audit session."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"events": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.audit.get_session_events("sess_123")

            mock_request.assert_called_once_with("GET", "/api/v1/audit/sessions/sess_123/events")
            client.close()

    def test_get_session_findings(self) -> None:
        """Get findings for an audit session."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"findings": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.audit.get_session_findings("sess_123")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/audit/sessions/sess_123/findings",
                params={"limit": 100, "offset": 0},
            )
            client.close()


# =========================================================================
# Session Lifecycle Operations
# =========================================================================


class TestAuditSessionLifecycle:
    """Tests for audit session lifecycle operations."""

    def test_start_session(self) -> None:
        """Start an audit session."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"status": "running"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.audit.start_session("sess_123")

            mock_request.assert_called_once_with("POST", "/api/v1/audit/sessions/sess_123/start")
            assert result["status"] == "running"
            client.close()

    def test_pause_session(self) -> None:
        """Pause an audit session."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"status": "paused"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.audit.pause_session("sess_123")

            mock_request.assert_called_once_with("POST", "/api/v1/audit/sessions/sess_123/pause")
            assert result["status"] == "paused"
            client.close()

    def test_resume_session(self) -> None:
        """Resume an audit session."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"status": "running"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.audit.resume_session("sess_123")

            mock_request.assert_called_once_with("POST", "/api/v1/audit/sessions/sess_123/resume")
            assert result["status"] == "running"
            client.close()

    def test_cancel_session(self) -> None:
        """Cancel an audit session."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"status": "cancelled"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.audit.cancel_session("sess_123")

            mock_request.assert_called_once_with(
                "POST", "/api/v1/audit/sessions/sess_123/cancel", json={}
            )
            assert result["status"] == "cancelled"
            client.close()

    def test_intervene_session(self) -> None:
        """Intervene in an audit session."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"intervention": "recorded"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.audit.intervene_session("sess_123", "skip_check")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/audit/sessions/sess_123/intervene",
                json={"action": "skip_check"},
            )
            assert result["intervention"] == "recorded"
            client.close()


# =========================================================================
# Async Tests
# =========================================================================


class TestAsyncAudit:
    """Tests for async Audit API."""

    @pytest.mark.asyncio
    async def test_async_create_session(self) -> None:
        """Create session asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"id": "sess_123"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.audit.create_session("Q1 Audit")

                mock_request.assert_called_once_with(
                    "POST", "/api/v1/audit/sessions", json={"name": "Q1 Audit"}
                )
                assert result["id"] == "sess_123"

    @pytest.mark.asyncio
    async def test_async_start_session(self) -> None:
        """Start session asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"status": "running"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.audit.start_session("sess_123")

                mock_request.assert_called_once_with(
                    "POST", "/api/v1/audit/sessions/sess_123/start"
                )
                assert result["status"] == "running"

    @pytest.mark.asyncio
    async def test_async_verify(self) -> None:
        """Verify audit integrity asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"valid": True}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.audit.verify()

                mock_request.assert_called_once_with("GET", "/api/v1/audit/verify")
                assert result["valid"] is True
