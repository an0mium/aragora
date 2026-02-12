"""
Tests for SDK Coverage Expansion (Phase 3).

Validates the 5 new/registered SDK namespaces:
- Receipts (pre-existing, verified)
- Approvals (new)
- Audit Trail (new)
- Voice (newly registered)
- OpenClaw (newly registered)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_sync_client():
    """Create a mock sync client."""
    client = MagicMock()
    client.request = MagicMock(return_value={"status": "ok"})
    return client


@pytest.fixture()
def mock_async_client():
    """Create a mock async client for async tests."""
    from unittest.mock import AsyncMock

    client = MagicMock()
    client.request = AsyncMock(return_value={"status": "ok"})
    return client


# =========================================================================
# Receipts (pre-existing, verify registration)
# =========================================================================


class TestReceiptsNamespace:
    def test_receipts_registered(self):
        from aragora_sdk.client import AragoraClient

        client = AragoraClient(base_url="http://localhost")
        assert hasattr(client, "receipts")
        assert type(client.receipts).__name__ == "ReceiptsAPI"

    def test_receipts_list(self, mock_sync_client):
        from aragora_sdk.namespaces.receipts import ReceiptsAPI

        api = ReceiptsAPI(mock_sync_client)
        api.list(verdict="APPROVED", limit=10)
        mock_sync_client.request.assert_called_once()
        args = mock_sync_client.request.call_args
        assert args[0][0] == "GET"
        assert args[0][1] == "/api/v2/receipts"

    def test_receipts_verify(self, mock_sync_client):
        from aragora_sdk.namespaces.receipts import ReceiptsAPI

        api = ReceiptsAPI(mock_sync_client)
        api.verify("receipt-123")
        args = mock_sync_client.request.call_args
        assert args[0][0] == "POST"
        assert "/verify" in args[0][1]

    def test_receipts_export(self, mock_sync_client):
        from aragora_sdk.namespaces.receipts import ReceiptsAPI

        api = ReceiptsAPI(mock_sync_client)
        api.export("receipt-123", format="sarif")
        args = mock_sync_client.request.call_args
        assert "/export" in args[0][1]

    def test_receipts_has_dissent_helper(self):
        from aragora_sdk.namespaces.receipts import ReceiptsAPI

        assert ReceiptsAPI.has_dissent({"dissenting_agents": ["a"]}) is True
        assert ReceiptsAPI.has_dissent({"dissenting_agents": []}) is False
        assert ReceiptsAPI.has_dissent({}) is False


# =========================================================================
# Approvals (new namespace)
# =========================================================================


class TestApprovalsNamespace:
    def test_approvals_registered(self):
        from aragora_sdk.client import AragoraClient

        client = AragoraClient(base_url="http://localhost")
        assert hasattr(client, "approvals")
        assert type(client.approvals).__name__ == "ApprovalsAPI"

    def test_list_pending(self, mock_sync_client):
        from aragora_sdk.namespaces.approvals import ApprovalsAPI

        api = ApprovalsAPI(mock_sync_client)
        api.list_pending()
        args = mock_sync_client.request.call_args
        assert args[0][0] == "GET"
        assert args[0][1] == "/api/v1/approvals/pending"

    def test_list_pending_with_sources(self, mock_sync_client):
        from aragora_sdk.namespaces.approvals import ApprovalsAPI

        api = ApprovalsAPI(mock_sync_client)
        api.list_pending(sources=["workflow", "gateway"])
        args = mock_sync_client.request.call_args
        params = args[1]["params"] if "params" in args[1] else args[0][2]
        assert "workflow,gateway" in str(params)

    def test_list_all(self, mock_sync_client):
        from aragora_sdk.namespaces.approvals import ApprovalsAPI

        api = ApprovalsAPI(mock_sync_client)
        api.list()
        args = mock_sync_client.request.call_args
        assert args[0][0] == "GET"
        assert args[0][1] == "/api/v1/approvals"

    @pytest.mark.asyncio()
    async def test_async_list_pending(self, mock_async_client):
        from aragora_sdk.namespaces.approvals import AsyncApprovalsAPI

        api = AsyncApprovalsAPI(mock_async_client)
        await api.list_pending()
        mock_async_client.request.assert_awaited_once()


# =========================================================================
# Audit Trail (new namespace)
# =========================================================================


class TestAuditTrailNamespace:
    def test_audit_trail_registered(self):
        from aragora_sdk.client import AragoraClient

        client = AragoraClient(base_url="http://localhost")
        assert hasattr(client, "audit_trail")
        assert type(client.audit_trail).__name__ == "AuditTrailAPI"

    def test_list_trails(self, mock_sync_client):
        from aragora_sdk.namespaces.audit_trail import AuditTrailAPI

        api = AuditTrailAPI(mock_sync_client)
        api.list()
        args = mock_sync_client.request.call_args
        assert args[0][0] == "GET"
        assert args[0][1] == "/api/v1/audit-trails"

    def test_get_trail(self, mock_sync_client):
        from aragora_sdk.namespaces.audit_trail import AuditTrailAPI

        api = AuditTrailAPI(mock_sync_client)
        api.get("trail-123")
        args = mock_sync_client.request.call_args
        assert args[0][1] == "/api/v1/audit-trails/trail-123"

    def test_export_trail(self, mock_sync_client):
        from aragora_sdk.namespaces.audit_trail import AuditTrailAPI

        api = AuditTrailAPI(mock_sync_client)
        api.export("trail-123", format="csv")
        args = mock_sync_client.request.call_args
        assert "/export" in args[0][1]

    def test_verify_trail(self, mock_sync_client):
        from aragora_sdk.namespaces.audit_trail import AuditTrailAPI

        api = AuditTrailAPI(mock_sync_client)
        api.verify("trail-123")
        args = mock_sync_client.request.call_args
        assert args[0][0] == "POST"
        assert "/verify" in args[0][1]

    def test_list_receipts(self, mock_sync_client):
        from aragora_sdk.namespaces.audit_trail import AuditTrailAPI

        api = AuditTrailAPI(mock_sync_client)
        api.list_receipts(verdict="APPROVED")
        args = mock_sync_client.request.call_args
        assert args[0][1] == "/api/v1/receipts"

    def test_verify_receipt(self, mock_sync_client):
        from aragora_sdk.namespaces.audit_trail import AuditTrailAPI

        api = AuditTrailAPI(mock_sync_client)
        api.verify_receipt("receipt-456")
        args = mock_sync_client.request.call_args
        assert args[0][0] == "POST"
        assert "/api/v1/receipts/receipt-456/verify" == args[0][1]

    @pytest.mark.asyncio()
    async def test_async_list_trails(self, mock_async_client):
        from aragora_sdk.namespaces.audit_trail import AsyncAuditTrailAPI

        api = AsyncAuditTrailAPI(mock_async_client)
        await api.list()
        mock_async_client.request.assert_awaited_once()


# =========================================================================
# Voice (existing namespace, newly registered)
# =========================================================================


class TestVoiceNamespace:
    def test_voice_registered(self):
        from aragora_sdk.client import AragoraClient

        client = AragoraClient(base_url="http://localhost")
        assert hasattr(client, "voice")
        assert type(client.voice).__name__ == "VoiceAPI"

    def test_synthesize(self, mock_sync_client):
        from aragora_sdk.namespaces.voice import VoiceAPI

        api = VoiceAPI(mock_sync_client)
        api.synthesize("Hello world", voice="default", format="mp3")
        args = mock_sync_client.request.call_args
        assert args[0][0] == "POST"
        assert args[0][1] == "/api/v1/voice/synthesize"

    def test_list_voices(self, mock_sync_client):
        from aragora_sdk.namespaces.voice import VoiceAPI

        api = VoiceAPI(mock_sync_client)
        api.list_voices()
        args = mock_sync_client.request.call_args
        assert args[0][1] == "/api/v1/voice/voices"

    def test_create_session(self, mock_sync_client):
        from aragora_sdk.namespaces.voice import VoiceAPI

        api = VoiceAPI(mock_sync_client)
        api.create_session(debate_id="debate-1")
        args = mock_sync_client.request.call_args
        assert args[0][0] == "POST"
        assert args[0][1] == "/api/v1/voice/sessions"

    def test_end_session(self, mock_sync_client):
        from aragora_sdk.namespaces.voice import VoiceAPI

        api = VoiceAPI(mock_sync_client)
        api.end_session("session-1")
        args = mock_sync_client.request.call_args
        assert args[0][0] == "DELETE"

    @pytest.mark.asyncio()
    async def test_async_synthesize(self, mock_async_client):
        from aragora_sdk.namespaces.voice import AsyncVoiceAPI

        api = AsyncVoiceAPI(mock_async_client)
        await api.synthesize("Hello")
        mock_async_client.request.assert_awaited_once()


# =========================================================================
# OpenClaw (existing namespace, newly registered)
# =========================================================================


class TestOpenclawNamespace:
    def test_openclaw_registered(self):
        from aragora_sdk.client import AragoraClient

        client = AragoraClient(base_url="http://localhost")
        assert hasattr(client, "openclaw")
        assert type(client.openclaw).__name__ == "OpenclawAPI"

    def test_search_cases(self, mock_sync_client):
        from aragora_sdk.namespaces.openclaw import OpenclawAPI

        api = OpenclawAPI(mock_sync_client)
        api.search_cases("contract breach")
        args = mock_sync_client.request.call_args
        assert args[0][0] == "GET"
        assert args[0][1] == "/api/v1/openclaw/cases/search"

    def test_get_case(self, mock_sync_client):
        from aragora_sdk.namespaces.openclaw import OpenclawAPI

        api = OpenclawAPI(mock_sync_client)
        api.get_case("case-1")
        args = mock_sync_client.request.call_args
        assert args[0][1] == "/api/v1/openclaw/cases/case-1"

    def test_analyze_document(self, mock_sync_client):
        from aragora_sdk.namespaces.openclaw import OpenclawAPI

        api = OpenclawAPI(mock_sync_client)
        api.analyze_document("This is a contract...", analysis_type="legal")
        args = mock_sync_client.request.call_args
        assert args[0][0] == "POST"
        assert args[0][1] == "/api/v1/openclaw/analyze"

    def test_verify_citation(self, mock_sync_client):
        from aragora_sdk.namespaces.openclaw import OpenclawAPI

        api = OpenclawAPI(mock_sync_client)
        api.verify_citation("Smith v. Jones, 123 F.3d 456")
        args = mock_sync_client.request.call_args
        assert args[0][0] == "POST"
        assert args[0][1] == "/api/v1/openclaw/citations/verify"

    def test_list_jurisdictions(self, mock_sync_client):
        from aragora_sdk.namespaces.openclaw import OpenclawAPI

        api = OpenclawAPI(mock_sync_client)
        api.list_jurisdictions()
        args = mock_sync_client.request.call_args
        assert args[0][1] == "/api/v1/openclaw/jurisdictions"

    @pytest.mark.asyncio()
    async def test_async_search_cases(self, mock_async_client):
        from aragora_sdk.namespaces.openclaw import AsyncOpenclawAPI

        api = AsyncOpenclawAPI(mock_async_client)
        await api.search_cases("test query")
        mock_async_client.request.assert_awaited_once()


# =========================================================================
# Cross-client registration test
# =========================================================================


class TestAllNamespacesRegistered:
    """Verify all 5 namespaces are accessible from both sync and async clients."""

    EXPECTED_NAMESPACES = [
        ("receipts", "ReceiptsAPI"),
        ("approvals", "ApprovalsAPI"),
        ("audit_trail", "AuditTrailAPI"),
        ("voice", "VoiceAPI"),
        ("openclaw", "OpenclawAPI"),
    ]

    def test_sync_client_has_all(self):
        from aragora_sdk.client import AragoraClient

        client = AragoraClient(base_url="http://localhost")
        for attr_name, class_name in self.EXPECTED_NAMESPACES:
            assert hasattr(client, attr_name), f"Missing sync namespace: {attr_name}"
            assert type(getattr(client, attr_name)).__name__ == class_name

    def test_async_client_has_all(self):
        from aragora_sdk.client import AragoraAsyncClient

        client = AragoraAsyncClient(base_url="http://localhost")
        for attr_name, _ in self.EXPECTED_NAMESPACES:
            assert hasattr(client, attr_name), f"Missing async namespace: {attr_name}"
