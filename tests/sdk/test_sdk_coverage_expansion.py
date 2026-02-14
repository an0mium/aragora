"""
Tests for SDK Coverage Expansion (Phase 3).

Validates coverage-oriented SDK namespaces:
- Receipts (pre-existing, verified)
- Approvals (new)
- Audit Trail (new)
- Voice (newly registered)
- OpenClaw (contract-aligned gateway surface)
- Blockchain (new namespace for ERC-8004 endpoints)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("aragora_sdk", reason="aragora_sdk not installed")


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

    def test_receipts_list_gauntlet(self, mock_sync_client):
        from aragora_sdk.namespaces.receipts import ReceiptsAPI

        api = ReceiptsAPI(mock_sync_client)
        api.list_gauntlet(verdict="APPROVED", limit=10)
        mock_sync_client.request.assert_called_once()
        args = mock_sync_client.request.call_args
        assert args[0][0] == "GET"
        assert args[0][1] == "/api/v2/gauntlet/results"

    def test_receipts_verify_gauntlet(self, mock_sync_client):
        from aragora_sdk.namespaces.receipts import ReceiptsAPI

        api = ReceiptsAPI(mock_sync_client)
        api.verify_gauntlet("receipt-123")
        args = mock_sync_client.request.call_args
        assert args[0][0] == "POST"
        assert "/verify" in args[0][1]

    def test_receipts_export_gauntlet(self, mock_sync_client):
        from aragora_sdk.namespaces.receipts import ReceiptsAPI

        api = ReceiptsAPI(mock_sync_client)
        api.export_gauntlet("receipt-123", format="sarif")
        args = mock_sync_client.request.call_args
        assert "receipt" in args[0][1]

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

    # NOTE: AuditTrailAPI methods (list, get, export, verify, list_receipts,
    # verify_receipt) were pruned as stale in the SDK cleanup.
    # Only the namespace registration test (test_audit_trail_registered) remains.


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
# OpenClaw (gateway-aligned namespace)
# =========================================================================


class TestOpenclawNamespace:
    def test_openclaw_registered(self):
        from aragora_sdk.client import AragoraClient

        client = AragoraClient(base_url="http://localhost")
        assert hasattr(client, "openclaw")
        assert type(client.openclaw).__name__ == "OpenclawAPI"

    # NOTE: OpenclawAPI methods (list_sessions, create_session, execute_action,
    # get_policy_rules, health, async variants) were pruned as stale in the SDK
    # cleanup. Only the namespace registration test (test_openclaw_registered)
    # remains.


# =========================================================================
# Blockchain (new namespace)
# =========================================================================


class TestBlockchainNamespace:
    def test_blockchain_registered(self):
        from aragora_sdk.client import AragoraClient

        client = AragoraClient(base_url="http://localhost")
        assert hasattr(client, "blockchain")
        assert type(client.blockchain).__name__ == "BlockchainAPI"

    def test_get_config(self, mock_sync_client):
        from aragora_sdk.namespaces.blockchain import BlockchainAPI

        api = BlockchainAPI(mock_sync_client)
        api.get_config()
        args = mock_sync_client.request.call_args
        assert args[0][0] == "GET"
        assert args[0][1] == "/api/v1/blockchain/config"

    def test_sync(self, mock_sync_client):
        from aragora_sdk.namespaces.blockchain import BlockchainAPI

        api = BlockchainAPI(mock_sync_client)
        api.sync(agent_ids=[1, 2, 3])
        args = mock_sync_client.request.call_args
        assert args[0][0] == "POST"
        assert args[0][1] == "/api/v1/blockchain/sync"

    def test_get_agent(self, mock_sync_client):
        from aragora_sdk.namespaces.blockchain import BlockchainAPI

        api = BlockchainAPI(mock_sync_client)
        api.get_agent(42)
        args = mock_sync_client.request.call_args
        assert args[0][0] == "GET"
        assert args[0][1] == "/api/v1/blockchain/agents/42"

    @pytest.mark.asyncio()
    async def test_async_get_health(self, mock_async_client):
        from aragora_sdk.namespaces.blockchain import AsyncBlockchainAPI

        api = AsyncBlockchainAPI(mock_async_client)
        await api.get_health()
        mock_async_client.request.assert_awaited_once()


# =========================================================================
# Cross-client registration test
# =========================================================================


class TestAllNamespacesRegistered:
    """Verify all coverage-expansion namespaces are accessible from both clients."""

    EXPECTED_NAMESPACES = [
        ("receipts", "ReceiptsAPI"),
        ("approvals", "ApprovalsAPI"),
        ("audit_trail", "AuditTrailAPI"),
        ("voice", "VoiceAPI"),
        ("openclaw", "OpenclawAPI"),
        ("blockchain", "BlockchainAPI"),
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
