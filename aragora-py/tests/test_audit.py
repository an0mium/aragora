"""Tests for the Audit API."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aragora_client.audit import (
    AuditEvent,
    AuditExportResponse,
    AuditIntegrityResult,
    AuditRetentionPolicy,
    AuditStats,
)


class TestAuditAPI:
    """Tests for AuditAPI methods."""

    @pytest.mark.asyncio
    async def test_list_events(self, mock_client, mock_response):
        """Test listing audit events."""
        response_data = {
            "events": [
                {
                    "id": "event-123",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "actor_id": "user-123",
                    "actor_type": "user",
                    "action": "user.login",
                    "resource_type": "session",
                    "resource_id": "session-456",
                }
            ],
            "total": 1,
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        events, total = await mock_client.audit.list_events()

        assert len(events) == 1
        assert total == 1
        assert events[0].id == "event-123"
        assert events[0].action == "user.login"

    @pytest.mark.asyncio
    async def test_list_events_with_filters(self, mock_client, mock_response):
        """Test listing events with filters."""
        response_data = {"events": [], "total": 0}
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        events, total = await mock_client.audit.list_events(
            start_date="2026-01-01",
            end_date="2026-01-31",
            actor_id="user-123",
            resource_type="debate",
            action="debate.create",
        )

        assert len(events) == 0
        assert total == 0

    @pytest.mark.asyncio
    async def test_get_event(self, mock_client, mock_response):
        """Test getting a specific audit event."""
        response_data = {
            "id": "event-123",
            "timestamp": "2026-01-01T00:00:00Z",
            "actor_id": "user-123",
            "action": "user.login",
            "resource_type": "session",
            "details": {"ip": "127.0.0.1"},
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.audit.get_event("event-123")

        assert isinstance(result, AuditEvent)
        assert result.id == "event-123"
        assert result.details == {"ip": "127.0.0.1"}

    @pytest.mark.asyncio
    async def test_get_stats(self, mock_client, mock_response):
        """Test getting audit statistics."""
        response_data = {
            "total_events": 1000,
            "events_by_action": {"user.login": 500, "debate.create": 200},
            "events_by_resource_type": {"session": 500, "debate": 200},
            "top_actors": [{"actor_id": "user-123", "count": 100}],
            "period_start": "2026-01-01T00:00:00Z",
            "period_end": "2026-01-31T23:59:59Z",
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.audit.get_stats()

        assert isinstance(result, AuditStats)
        assert result.total_events == 1000
        assert result.events_by_action["user.login"] == 500

    @pytest.mark.asyncio
    async def test_export(self, mock_client, mock_response):
        """Test exporting audit logs."""
        response_data = {
            "export_id": "export-123",
            "status": "pending",
            "download_url": None,
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.audit.export(
            start_date="2026-01-01",
            end_date="2026-01-31",
            format="csv",
        )

        assert isinstance(result, AuditExportResponse)
        assert result.export_id == "export-123"
        assert result.status == "pending"

    @pytest.mark.asyncio
    async def test_get_export_status(self, mock_client, mock_response):
        """Test getting export status."""
        response_data = {
            "export_id": "export-123",
            "status": "completed",
            "download_url": "https://example.com/export.csv",
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.audit.get_export_status("export-123")

        assert isinstance(result, AuditExportResponse)
        assert result.status == "completed"
        assert result.download_url == "https://example.com/export.csv"

    @pytest.mark.asyncio
    async def test_verify_integrity(self, mock_client, mock_response):
        """Test verifying audit log integrity."""
        response_data = {
            "verified": True,
            "entries_checked": 1000,
            "tampered_entries": 0,
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.audit.verify_integrity()

        assert isinstance(result, AuditIntegrityResult)
        assert result.verified is True
        assert result.tampered_entries == 0

    @pytest.mark.asyncio
    async def test_search(self, mock_client, mock_response):
        """Test searching audit events."""
        response_data = {
            "events": [
                {
                    "id": "event-123",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "action": "user.login",
                    "resource_type": "session",
                }
            ]
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.audit.search("login")

        assert len(result) == 1
        assert result[0].action == "user.login"

    @pytest.mark.asyncio
    async def test_list_retention_policies(self, mock_client, mock_response):
        """Test listing retention policies."""
        response_data = {
            "policies": [
                {
                    "id": "policy-123",
                    "name": "Standard Policy",
                    "retention_days": 365,
                    "resource_types": ["debate", "user"],
                    "is_active": True,
                }
            ]
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.audit.list_retention_policies()

        assert len(result) == 1
        assert isinstance(result[0], AuditRetentionPolicy)
        assert result[0].name == "Standard Policy"
        assert result[0].retention_days == 365

    @pytest.mark.asyncio
    async def test_create_retention_policy(self, mock_client, mock_response):
        """Test creating a retention policy."""
        response_data = {
            "id": "policy-new",
            "name": "Compliance Policy",
            "retention_days": 730,
            "resource_types": ["audit"],
            "is_active": True,
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.audit.create_retention_policy(
            name="Compliance Policy",
            retention_days=730,
            resource_types=["audit"],
        )

        assert isinstance(result, AuditRetentionPolicy)
        assert result.id == "policy-new"
        assert result.retention_days == 730

    @pytest.mark.asyncio
    async def test_delete_retention_policy(self, mock_client, mock_response):
        """Test deleting a retention policy."""
        mock_client._client.request = AsyncMock(return_value=mock_response(204, {}))

        # Should not raise
        await mock_client.audit.delete_retention_policy("policy-123")

    @pytest.mark.asyncio
    async def test_get_actor_history(self, mock_client, mock_response):
        """Test getting activity history for an actor."""
        response_data = {
            "events": [
                {
                    "id": "event-1",
                    "timestamp": "2026-01-01T12:00:00Z",
                    "actor_id": "user-123",
                    "action": "user.login",
                    "resource_type": "session",
                },
                {
                    "id": "event-2",
                    "timestamp": "2026-01-01T13:00:00Z",
                    "actor_id": "user-123",
                    "action": "debate.create",
                    "resource_type": "debate",
                },
            ]
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.audit.get_actor_history("user-123")

        assert len(result) == 2
        assert all(e.actor_id == "user-123" for e in result)

    @pytest.mark.asyncio
    async def test_get_resource_history(self, mock_client, mock_response):
        """Test getting activity history for a resource."""
        response_data = {
            "events": [
                {
                    "id": "event-1",
                    "timestamp": "2026-01-01T12:00:00Z",
                    "action": "debate.create",
                    "resource_type": "debate",
                    "resource_id": "debate-123",
                },
                {
                    "id": "event-2",
                    "timestamp": "2026-01-01T13:00:00Z",
                    "action": "debate.update",
                    "resource_type": "debate",
                    "resource_id": "debate-123",
                },
            ]
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.audit.get_resource_history("debate", "debate-123")

        assert len(result) == 2
        assert all(e.resource_id == "debate-123" for e in result)
