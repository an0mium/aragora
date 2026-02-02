"""Tests for Audit namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient

# =========================================================================
# Event Operations
# =========================================================================


class TestAuditEvents:
    """Tests for audit event operations."""

    def test_list_events_default(self) -> None:
        """List events with default parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "events": [{"id": "evt_1", "type": "debate.created"}],
                "total": 1,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.audit.list_events()

            mock_request.assert_called_once_with(
                "GET", "/api/v1/audit/events", params={"limit": 50, "offset": 0}
            )
            assert result["total"] == 1
            client.close()

    def test_list_events_with_filters(self) -> None:
        """List events with all filters applied."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"events": [], "total": 0}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.audit.list_events(
                event_type="debate.created",
                actor_id="user_123",
                resource_type="debate",
                resource_id="deb_456",
                from_date="2025-01-01",
                to_date="2025-01-31",
                limit=25,
                offset=10,
            )

            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["event_type"] == "debate.created"
            assert params["actor_id"] == "user_123"
            assert params["resource_type"] == "debate"
            assert params["resource_id"] == "deb_456"
            assert params["from_date"] == "2025-01-01"
            assert params["to_date"] == "2025-01-31"
            assert params["limit"] == 25
            assert params["offset"] == 10
            client.close()

    def test_get_event(self) -> None:
        """Get a specific audit event."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "id": "evt_123",
                "type": "debate.completed",
                "actor_id": "user_1",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.audit.get_event("evt_123")

            mock_request.assert_called_once_with("GET", "/api/v1/audit/events/evt_123")
            assert result["id"] == "evt_123"
            client.close()


# =========================================================================
# Export Operations
# =========================================================================


class TestAuditExport:
    """Tests for audit export operations."""

    def test_export_json(self) -> None:
        """Export audit events as JSON."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"download_url": "https://cdn.aragora.ai/export/..."}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.audit.export(format="json")

            mock_request.assert_called_once_with(
                "POST", "/api/v1/audit/export", json={"format": "json"}
            )
            assert "download_url" in result
            client.close()

    def test_export_csv_with_filters(self) -> None:
        """Export audit events as CSV with date filters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"download_url": "https://cdn.aragora.ai/export/..."}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.audit.export(
                format="csv",
                from_date="2025-01-01",
                to_date="2025-01-31",
                event_types=["debate.created", "debate.completed"],
            )

            call_args = mock_request.call_args
            json_data = call_args[1]["json"]
            assert json_data["format"] == "csv"
            assert json_data["from_date"] == "2025-01-01"
            assert json_data["to_date"] == "2025-01-31"
            assert json_data["event_types"] == ["debate.created", "debate.completed"]
            client.close()

    def test_export_pdf(self) -> None:
        """Export audit events as PDF."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"download_url": "https://cdn.aragora.ai/export/..."}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.audit.export(format="pdf")

            call_args = mock_request.call_args
            assert call_args[1]["json"]["format"] == "pdf"
            client.close()


# =========================================================================
# Compliance Operations
# =========================================================================


class TestAuditCompliance:
    """Tests for compliance report operations."""

    def test_get_compliance_report_default(self) -> None:
        """Get compliance report with default period."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "period": "monthly",
                "metrics": {},
                "findings": [],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.audit.get_compliance_report()

            mock_request.assert_called_once_with(
                "GET", "/api/v1/audit/compliance/report", params={"period": "monthly"}
            )
            assert result["period"] == "monthly"
            client.close()

    def test_get_compliance_report_with_framework(self) -> None:
        """Get compliance report for a specific framework."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"framework": "soc2", "controls": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.audit.get_compliance_report(period="quarterly", framework="soc2")

            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["period"] == "quarterly"
            assert params["framework"] == "soc2"
            client.close()


# =========================================================================
# Activity Operations
# =========================================================================


class TestAuditActivity:
    """Tests for activity tracking operations."""

    def test_get_actor_activity(self) -> None:
        """Get activity summary for an actor."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "actor_id": "user_123",
                "total_events": 50,
                "event_breakdown": {},
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.audit.get_actor_activity("user_123")

            mock_request.assert_called_once_with(
                "GET", "/api/v1/audit/actors/activity", params={"actor_id": "user_123"}
            )
            assert result["actor_id"] == "user_123"
            client.close()

    def test_get_actor_activity_with_dates(self) -> None:
        """Get activity summary with date range."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"actor_id": "user_123"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.audit.get_actor_activity(
                "user_123", from_date="2025-01-01", to_date="2025-01-31"
            )

            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["from_date"] == "2025-01-01"
            assert params["to_date"] == "2025-01-31"
            client.close()

    def test_get_resource_history(self) -> None:
        """Get audit history for a resource."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"events": [], "total": 0}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.audit.get_resource_history("debate", "deb_123")

            mock_request.assert_called_once_with(
                "GET", "/api/v1/audit/resources/debate/deb_123/history"
            )
            assert "events" in result
            client.close()


# =========================================================================
# Statistics Operations
# =========================================================================


class TestAuditStats:
    """Tests for audit statistics operations."""

    def test_get_stats_default(self) -> None:
        """Get audit statistics with no filters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "total_events": 1000,
                "events_by_type": {},
                "top_actors": [],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.audit.get_stats()

            mock_request.assert_called_once_with("GET", "/api/v1/audit/stats", params={})
            assert result["total_events"] == 1000
            client.close()

    def test_get_stats_with_dates(self) -> None:
        """Get audit statistics with date range."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"total_events": 100}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.audit.get_stats(from_date="2025-01-01", to_date="2025-01-31")

            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["from_date"] == "2025-01-01"
            assert params["to_date"] == "2025-01-31"
            client.close()


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

            mock_request.assert_called_once_with("GET", "/api/v1/audit/sessions/sess_123/findings")
            client.close()

    def test_get_session_report(self) -> None:
        """Get report for an audit session."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"report": {}}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.audit.get_session_report("sess_123")

            mock_request.assert_called_once_with("GET", "/api/v1/audit/sessions/sess_123/report")
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

            mock_request.assert_called_once_with("POST", "/api/v1/audit/sessions/sess_123/cancel")
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
    async def test_async_list_events(self) -> None:
        """List events asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"events": []}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.audit.list_events()

                mock_request.assert_called_once_with(
                    "GET", "/api/v1/audit/events", params={"limit": 50, "offset": 0}
                )
                assert "events" in result

    @pytest.mark.asyncio
    async def test_async_get_event(self) -> None:
        """Get event asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"id": "evt_123"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.audit.get_event("evt_123")

                mock_request.assert_called_once_with("GET", "/api/v1/audit/events/evt_123")
                assert result["id"] == "evt_123"

    @pytest.mark.asyncio
    async def test_async_export(self) -> None:
        """Export audit asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"download_url": "..."}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.audit.export(format="json")

                mock_request.assert_called_once_with(
                    "POST", "/api/v1/audit/export", json={"format": "json"}
                )
                assert "download_url" in result

    @pytest.mark.asyncio
    async def test_async_get_compliance_report(self) -> None:
        """Get compliance report asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"period": "monthly"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.audit.get_compliance_report()

                mock_request.assert_called_once_with(
                    "GET",
                    "/api/v1/audit/compliance/report",
                    params={"period": "monthly"},
                )
                assert result["period"] == "monthly"

    @pytest.mark.asyncio
    async def test_async_get_stats(self) -> None:
        """Get stats asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"total_events": 100}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.audit.get_stats()

                mock_request.assert_called_once_with("GET", "/api/v1/audit/stats", params={})
                assert result["total_events"] == 100

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
