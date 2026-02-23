"""Tests for ERC-8004 blockchain handler (aragora/server/handlers/erc8004.py).

Covers all routes and behavior of the ERC8004Handler class and standalone handlers:
- can_handle() routing
- GET    /api/v1/blockchain/config                         - Get chain configuration
- GET    /api/v1/blockchain/health                         - Get health status
- GET    /api/v1/blockchain/agents                         - List on-chain agents
- POST   /api/v1/blockchain/agents                         - Register new agent
- GET    /api/v1/blockchain/agents/{token_id}              - Get agent identity
- GET    /api/v1/blockchain/agents/{token_id}/reputation   - Get reputation
- GET    /api/v1/blockchain/agents/{token_id}/validations  - Get validations
- POST   /api/v1/blockchain/sync                           - Trigger manual sync
- Circuit breaker integration
- Validation and error paths
- _coerce_metadata_value and _serialize_identity helpers
- _get_query_param and _get_int_query_param helpers
- Module-level lazy initializers
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.base import HandlerResult
from aragora.server.handlers.erc8004 import (
    BLOCKCHAIN_HANDLERS,
    ERC8004Handler,
    _coerce_metadata_value,
    _get_int_query_param,
    _get_query_param,
    _serialize_identity,
    handle_blockchain_config,
    handle_blockchain_health,
    handle_blockchain_sync,
    handle_get_agent,
    handle_get_reputation,
    handle_get_validations,
    handle_list_agents,
    handle_register_agent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: HandlerResult) -> dict:
    """Extract the JSON body from a HandlerResult."""
    if isinstance(result, HandlerResult):
        if isinstance(result.body, bytes):
            return json.loads(result.body.decode("utf-8"))
        return result.body
    if isinstance(result, dict):
        return result.get("body", result)
    return {}


def _status(result: HandlerResult) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, HandlerResult):
        return result.status_code
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return 200


class MockHTTPHandler:
    """Mock HTTP handler for testing (simulates BaseHTTPRequestHandler)."""

    def __init__(self, body: dict[str, Any] | None = None):
        self.rfile = MagicMock()
        self.command = "GET"
        self._body = body
        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers = {"Content-Length": str(len(body_bytes))}
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {"Content-Length": "2"}


def _make_handler(
    body: dict[str, Any] | None = None, method: str = "GET"
) -> MockHTTPHandler:
    """Create a MockHTTPHandler with optional body and method."""
    h = MockHTTPHandler(body=body)
    h.command = method
    return h


# ---------------------------------------------------------------------------
# Mock data helpers
# ---------------------------------------------------------------------------


def _mock_identity(
    token_id: int = 1,
    owner: str = "0xabc123",
    agent_uri: str = "https://agent.example.com",
    wallet_address: str = "0xdef456",
    chain_id: int = 1,
    aragora_agent_id: str = "agent-001",
    registered_at: datetime | None = None,
    tx_hash: str = "0xtxhash",
):
    """Create a mock agent identity object."""
    identity = MagicMock()
    identity.token_id = token_id
    identity.owner = owner
    identity.agent_uri = agent_uri
    identity.wallet_address = wallet_address
    identity.chain_id = chain_id
    identity.aragora_agent_id = aragora_agent_id
    identity.registered_at = registered_at or datetime(2026, 1, 1, tzinfo=timezone.utc)
    identity.tx_hash = tx_hash
    return identity


def _mock_reputation_summary(
    agent_id: int = 1,
    count: int = 10,
    summary_value: int = 850,
    summary_value_decimals: int = 2,
    normalized_value: float = 0.85,
    tag1: str = "",
    tag2: str = "",
):
    """Create a mock reputation summary."""
    summary = MagicMock()
    summary.agent_id = agent_id
    summary.count = count
    summary.summary_value = summary_value
    summary.summary_value_decimals = summary_value_decimals
    summary.normalized_value = normalized_value
    summary.tag1 = tag1
    summary.tag2 = tag2
    return summary


def _mock_validation_summary(
    agent_id: int = 1,
    count: int = 5,
    average_response: float = 0.9,
    tag: str = "",
):
    """Create a mock validation summary."""
    summary = MagicMock()
    summary.agent_id = agent_id
    summary.count = count
    summary.average_response = average_response
    summary.tag = tag
    return summary


def _mock_chain_config(
    chain_id: int = 1,
    rpc_url: str = "http://localhost:8545",
    identity_registry_address: str = "0xidentity",
    reputation_registry_address: str = "0xreputation",
    validation_registry_address: str = "0xvalidation",
    block_confirmations: int = 12,
    has_identity_registry: bool = True,
):
    """Create a mock chain config."""
    config = MagicMock()
    config.chain_id = chain_id
    config.rpc_url = rpc_url
    config.identity_registry_address = identity_registry_address
    config.reputation_registry_address = reputation_registry_address
    config.validation_registry_address = validation_registry_address
    config.block_confirmations = block_confirmations
    config.has_identity_registry = has_identity_registry
    return config


def _mock_sync_result(
    records_synced: int = 5,
    records_skipped: int = 1,
    records_failed: int = 0,
    duration_ms: float = 123.4,
    errors: list | None = None,
):
    """Create a mock sync result."""
    result = MagicMock()
    result.records_synced = records_synced
    result.records_skipped = records_skipped
    result.records_failed = records_failed
    result.duration_ms = duration_ms
    result.errors = errors or []
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Reset module-level lazy singletons between tests."""
    import aragora.server.handlers.erc8004 as mod

    mod._provider = None
    mod._connector = None
    mod._adapter = None
    mod._blockchain_circuit_breaker = None
    yield
    mod._provider = None
    mod._connector = None
    mod._adapter = None
    mod._blockchain_circuit_breaker = None


@pytest.fixture(autouse=True)
def _reset_circuit_breakers():
    """Reset circuit breakers between tests."""
    from aragora.resilience import reset_all_circuit_breakers

    reset_all_circuit_breakers()
    yield
    reset_all_circuit_breakers()


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    """Reset rate limiters between tests."""
    from aragora.server.handlers.utils.rate_limit import clear_all_limiters

    clear_all_limiters()
    yield
    clear_all_limiters()


@pytest.fixture
def handler():
    """Create an ERC8004Handler instance."""
    return ERC8004Handler(server_context={})


@pytest.fixture
def mock_provider():
    """Create a mock Web3 provider."""
    provider = MagicMock()
    provider.get_config.return_value = _mock_chain_config()
    provider.is_connected.return_value = True
    provider.get_health_status.return_value = {"status": "healthy"}
    return provider


@pytest.fixture
def mock_circuit_breaker():
    """Create a mock circuit breaker that always allows execution."""
    cb = MagicMock()
    cb.can_execute.return_value = True
    return cb


# ============================================================================
# _coerce_metadata_value
# ============================================================================


class TestCoerceMetadataValue:
    """Test the _coerce_metadata_value helper."""

    def test_bytes_passthrough(self):
        result = _coerce_metadata_value(b"hello")
        assert result == b"hello"

    def test_string_to_bytes(self):
        result = _coerce_metadata_value("hello")
        assert result == b"hello"

    def test_int_to_bytes(self):
        result = _coerce_metadata_value(42)
        assert result == b"42"

    def test_float_to_bytes(self):
        result = _coerce_metadata_value(3.14)
        assert result == b"3.14"

    def test_bool_true(self):
        result = _coerce_metadata_value(True)
        assert result == b"True"

    def test_bool_false(self):
        result = _coerce_metadata_value(False)
        assert result == b"False"

    def test_empty_string(self):
        result = _coerce_metadata_value("")
        assert result == b""

    def test_empty_bytes(self):
        result = _coerce_metadata_value(b"")
        assert result == b""

    def test_unicode_string(self):
        result = _coerce_metadata_value("caf\u00e9")
        assert result == "caf\u00e9".encode("utf-8")

    def test_invalid_type_list(self):
        with pytest.raises(ValueError, match="must be bytes, str, int, float, or bool"):
            _coerce_metadata_value([1, 2, 3])

    def test_invalid_type_dict(self):
        with pytest.raises(ValueError, match="must be bytes, str, int, float, or bool"):
            _coerce_metadata_value({"key": "value"})

    def test_invalid_type_none(self):
        with pytest.raises(ValueError, match="must be bytes, str, int, float, or bool"):
            _coerce_metadata_value(None)

    def test_zero_int(self):
        result = _coerce_metadata_value(0)
        assert result == b"0"

    def test_negative_int(self):
        result = _coerce_metadata_value(-1)
        assert result == b"-1"


# ============================================================================
# _serialize_identity
# ============================================================================


class TestSerializeIdentity:
    """Test the _serialize_identity helper."""

    def test_basic_serialization(self):
        identity = _mock_identity()
        result = _serialize_identity(identity)
        assert result["token_id"] == 1
        assert result["owner"] == "0xabc123"
        assert result["agent_uri"] == "https://agent.example.com"
        assert result["wallet_address"] == "0xdef456"
        assert result["chain_id"] == 1
        assert result["aragora_agent_id"] == "agent-001"
        assert result["tx_hash"] == "0xtxhash"

    def test_registered_at_serialized(self):
        dt = datetime(2026, 2, 15, 10, 30, 0, tzinfo=timezone.utc)
        identity = _mock_identity(registered_at=dt)
        result = _serialize_identity(identity)
        assert result["registered_at"] == dt.isoformat()

    def test_registered_at_none(self):
        identity = _mock_identity()
        identity.registered_at = None
        result = _serialize_identity(identity)
        assert result["registered_at"] is None

    def test_all_fields_present(self):
        identity = _mock_identity()
        result = _serialize_identity(identity)
        expected_keys = {
            "token_id", "owner", "agent_uri", "wallet_address",
            "chain_id", "aragora_agent_id", "registered_at", "tx_hash",
        }
        assert set(result.keys()) == expected_keys


# ============================================================================
# _get_query_param and _get_int_query_param
# ============================================================================


class TestQueryParamHelpers:
    """Test query parameter extraction helpers."""

    def test_get_query_param_simple(self):
        assert _get_query_param({"key": "value"}, "key") == "value"

    def test_get_query_param_missing(self):
        assert _get_query_param({}, "key") == ""

    def test_get_query_param_with_default(self):
        assert _get_query_param({}, "key", "fallback") == "fallback"

    def test_get_query_param_list_value(self):
        assert _get_query_param({"key": ["first", "second"]}, "key") == "first"

    def test_get_query_param_empty_list(self):
        assert _get_query_param({"key": []}, "key", "default") == "default"

    def test_get_query_param_none_value(self):
        assert _get_query_param({"key": None}, "key", "default") == "default"

    def test_get_int_query_param_basic(self):
        assert _get_int_query_param({"skip": "10"}, "skip", 0) == 10

    def test_get_int_query_param_default(self):
        assert _get_int_query_param({}, "skip", 0) == 0

    def test_get_int_query_param_invalid(self):
        with pytest.raises(ValueError, match="must be an integer"):
            _get_int_query_param({"skip": "abc"}, "skip", 0)

    def test_get_int_query_param_below_min(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            _get_int_query_param({"skip": "-1"}, "skip", 0, min_value=0)

    def test_get_int_query_param_at_min(self):
        assert _get_int_query_param({"val": "0"}, "val", 0, min_value=0) == 0

    def test_get_int_query_param_min_value_custom(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            _get_int_query_param({"limit": "0"}, "limit", 100, min_value=1)


# ============================================================================
# can_handle routing
# ============================================================================


class TestCanHandle:
    """Verify that can_handle correctly accepts or rejects paths."""

    def test_blockchain_config_path(self, handler):
        assert handler.can_handle("/api/v1/blockchain/config")

    def test_blockchain_health_path(self, handler):
        assert handler.can_handle("/api/v1/blockchain/health")

    def test_blockchain_sync_path(self, handler):
        assert handler.can_handle("/api/v1/blockchain/sync")

    def test_blockchain_agents_path(self, handler):
        assert handler.can_handle("/api/v1/blockchain/agents")

    def test_blockchain_agents_with_id(self, handler):
        assert handler.can_handle("/api/v1/blockchain/agents/123")

    def test_blockchain_agents_reputation(self, handler):
        assert handler.can_handle("/api/v1/blockchain/agents/1/reputation")

    def test_blockchain_agents_validations(self, handler):
        assert handler.can_handle("/api/v1/blockchain/agents/1/validations")

    def test_rejects_non_blockchain_path(self, handler):
        assert not handler.can_handle("/api/v1/other/path")

    def test_rejects_empty_path(self, handler):
        assert not handler.can_handle("")

    def test_rejects_root_path(self, handler):
        assert not handler.can_handle("/")

    def test_rejects_partial_prefix(self, handler):
        assert not handler.can_handle("/api/v1/block")


# ============================================================================
# BLOCKCHAIN_HANDLERS registry
# ============================================================================


class TestBlockchainHandlers:
    """Test the handler registry dict."""

    def test_all_handlers_registered(self):
        expected_keys = {
            "blockchain_config",
            "blockchain_list_agents",
            "blockchain_register_agent",
            "blockchain_get_agent",
            "blockchain_get_reputation",
            "blockchain_get_validations",
            "blockchain_sync",
            "blockchain_health",
        }
        assert set(BLOCKCHAIN_HANDLERS.keys()) == expected_keys

    def test_handlers_are_callable(self):
        for name, handler_fn in BLOCKCHAIN_HANDLERS.items():
            assert callable(handler_fn), f"{name} is not callable"


# ============================================================================
# GET /api/v1/blockchain/config
# ============================================================================


class TestBlockchainConfig:
    """Test the blockchain config endpoint."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_config_success(self, mock_get_provider, mock_get_cb):
        config = _mock_chain_config()
        provider = MagicMock()
        provider.get_config.return_value = config
        provider.is_connected.return_value = True
        provider.get_health_status.return_value = {"status": "healthy"}
        mock_get_provider.return_value = provider

        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        result = await handle_blockchain_config()
        assert _status(result) == 200
        body = _body(result)
        assert body["chain_id"] == 1
        assert body["is_connected"] is True
        assert body["block_confirmations"] == 12

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_config_rpc_url_truncated(self, mock_get_provider, mock_get_cb):
        long_url = "http://" + "a" * 100 + ".example.com"
        config = _mock_chain_config(rpc_url=long_url)
        provider = MagicMock()
        provider.get_config.return_value = config
        provider.is_connected.return_value = True
        provider.get_health_status.return_value = {}
        mock_get_provider.return_value = provider

        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        result = await handle_blockchain_config()
        body = _body(result)
        assert body["rpc_url"].endswith("...")
        assert len(body["rpc_url"]) == 53  # 50 chars + "..."

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_config_short_rpc_url_not_truncated(self, mock_get_provider, mock_get_cb):
        config = _mock_chain_config(rpc_url="http://localhost:8545")
        provider = MagicMock()
        provider.get_config.return_value = config
        provider.is_connected.return_value = True
        provider.get_health_status.return_value = {}
        mock_get_provider.return_value = provider

        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        result = await handle_blockchain_config()
        body = _body(result)
        assert not body["rpc_url"].endswith("...")

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    async def test_config_circuit_breaker_open(self, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = False
        mock_get_cb.return_value = cb

        result = await handle_blockchain_config()
        assert _status(result) == 503
        assert "temporarily unavailable" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_config_import_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.side_effect = ImportError("web3 not installed")

        result = await handle_blockchain_config()
        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_config_connection_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.side_effect = ConnectionError("cannot connect")

        result = await handle_blockchain_config()
        assert _status(result) == 500
        assert "error" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_config_runtime_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.side_effect = RuntimeError("bad state")

        result = await handle_blockchain_config()
        assert _status(result) == 500


# ============================================================================
# GET /api/v1/blockchain/agents/{token_id}
# ============================================================================


class TestGetAgent:
    """Test the get agent endpoint."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    @patch("aragora.server.handlers.erc8004.IdentityRegistryContract", create=True)
    async def test_get_agent_success(self, MockContract, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        provider = MagicMock()
        mock_get_provider.return_value = provider

        identity = _mock_identity(token_id=42)
        contract_inst = MagicMock()
        contract_inst.get_agent.return_value = identity

        with patch(
            "aragora.server.handlers.erc8004.IdentityRegistryContract",
            return_value=contract_inst,
            create=True,
        ):
            # Patch the import inside the function
            with patch.dict("sys.modules", {
                "aragora.blockchain.contracts.identity": MagicMock(
                    IdentityRegistryContract=MagicMock(return_value=contract_inst)
                )
            }):
                result = await handle_get_agent(42)

        assert _status(result) == 200
        body = _body(result)
        assert body["token_id"] == 42

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    async def test_get_agent_circuit_open(self, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = False
        mock_get_cb.return_value = cb

        result = await handle_get_agent(1)
        assert _status(result) == 503

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_get_agent_import_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.return_value = MagicMock()

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": None,
        }):
            # Force ImportError on internal import
            result = await handle_get_agent(1)
        assert _status(result) in (404, 503)

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_get_agent_not_found(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.return_value = MagicMock()

        contract_inst = MagicMock()
        contract_inst.get_agent.side_effect = LookupError("not found")

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": MagicMock(
                IdentityRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            result = await handle_get_agent(999)

        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()


# ============================================================================
# GET /api/v1/blockchain/agents/{token_id}/reputation
# ============================================================================


class TestGetReputation:
    """Test the get reputation endpoint."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_get_reputation_success(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.return_value = MagicMock()

        summary = _mock_reputation_summary(agent_id=1, tag1="skill", tag2="domain")
        contract_inst = MagicMock()
        contract_inst.get_summary.return_value = summary

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.reputation": MagicMock(
                ReputationRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            result = await handle_get_reputation(1, tag1="skill", tag2="domain")

        assert _status(result) == 200
        body = _body(result)
        assert body["agent_id"] == 1
        assert body["count"] == 10
        assert body["tag1"] == "skill"
        assert body["tag2"] == "domain"

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    async def test_get_reputation_circuit_open(self, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = False
        mock_get_cb.return_value = cb

        result = await handle_get_reputation(1)
        assert _status(result) == 503

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_get_reputation_not_found(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.return_value = MagicMock()

        contract_inst = MagicMock()
        contract_inst.get_summary.side_effect = LookupError("no rep")

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.reputation": MagicMock(
                ReputationRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            result = await handle_get_reputation(999)

        assert _status(result) == 404

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_get_reputation_import_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.return_value = MagicMock()

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.reputation": None,
        }):
            result = await handle_get_reputation(1)
        assert _status(result) in (404, 503)

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_get_reputation_default_tags(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.return_value = MagicMock()

        summary = _mock_reputation_summary()
        contract_inst = MagicMock()
        contract_inst.get_summary.return_value = summary

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.reputation": MagicMock(
                ReputationRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            result = await handle_get_reputation(1)

        assert _status(result) == 200
        body = _body(result)
        assert body["tag1"] == ""
        assert body["tag2"] == ""


# ============================================================================
# GET /api/v1/blockchain/agents/{token_id}/validations
# ============================================================================


class TestGetValidations:
    """Test the get validations endpoint."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_get_validations_success(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.return_value = MagicMock()

        summary = _mock_validation_summary(agent_id=5, tag="accuracy")
        contract_inst = MagicMock()
        contract_inst.get_summary.return_value = summary

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.validation": MagicMock(
                ValidationRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            result = await handle_get_validations(5, tag="accuracy")

        assert _status(result) == 200
        body = _body(result)
        assert body["agent_id"] == 5
        assert body["count"] == 5
        assert body["tag"] == "accuracy"

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    async def test_get_validations_circuit_open(self, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = False
        mock_get_cb.return_value = cb

        result = await handle_get_validations(1)
        assert _status(result) == 503

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_get_validations_not_found(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.return_value = MagicMock()

        contract_inst = MagicMock()
        contract_inst.get_summary.side_effect = ValueError("bad token")

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.validation": MagicMock(
                ValidationRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            result = await handle_get_validations(999)

        assert _status(result) == 404

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_get_validations_default_tag(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.return_value = MagicMock()

        summary = _mock_validation_summary()
        contract_inst = MagicMock()
        contract_inst.get_summary.return_value = summary

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.validation": MagicMock(
                ValidationRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            result = await handle_get_validations(1)

        assert _status(result) == 200
        body = _body(result)
        assert body["tag"] == ""


# ============================================================================
# POST /api/v1/blockchain/sync
# ============================================================================


class TestBlockchainSync:
    """Test the blockchain sync endpoint."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_adapter")
    async def test_sync_success(self, mock_get_adapter, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        sync_result = _mock_sync_result()
        adapter = AsyncMock()
        adapter.sync_to_km.return_value = sync_result
        mock_get_adapter.return_value = adapter

        result = await handle_blockchain_sync()
        assert _status(result) == 200
        body = _body(result)
        assert body["records_synced"] == 5
        assert body["records_skipped"] == 1
        assert body["records_failed"] == 0
        assert body["errors"] == []

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_adapter")
    async def test_sync_with_params(self, mock_get_adapter, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        adapter = AsyncMock()
        adapter.sync_to_km.return_value = _mock_sync_result()
        mock_get_adapter.return_value = adapter

        result = await handle_blockchain_sync(
            sync_identities=True,
            sync_reputation=False,
            sync_validations=False,
            agent_ids=[1, 2, 3],
        )
        assert _status(result) == 200
        adapter.sync_to_km.assert_called_once_with(
            agent_ids=[1, 2, 3],
            sync_identities=True,
            sync_reputation=False,
            sync_validations=False,
        )

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_adapter")
    async def test_sync_errors_truncated_to_10(self, mock_get_adapter, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        many_errors = [f"error-{i}" for i in range(20)]
        adapter = AsyncMock()
        adapter.sync_to_km.return_value = _mock_sync_result(errors=many_errors)
        mock_get_adapter.return_value = adapter

        result = await handle_blockchain_sync()
        body = _body(result)
        assert len(body["errors"]) == 10

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    async def test_sync_circuit_open(self, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = False
        mock_get_cb.return_value = cb

        result = await handle_blockchain_sync()
        assert _status(result) == 503

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_adapter")
    async def test_sync_import_error(self, mock_get_adapter, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_adapter.side_effect = ImportError("adapter missing")

        result = await handle_blockchain_sync()
        assert _status(result) == 503

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_adapter")
    async def test_sync_connection_error(self, mock_get_adapter, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        adapter = AsyncMock()
        adapter.sync_to_km.side_effect = ConnectionError("network down")
        mock_get_adapter.return_value = adapter

        result = await handle_blockchain_sync()
        assert _status(result) == 400  # default error_response status


# ============================================================================
# GET /api/v1/blockchain/health
# ============================================================================


class TestBlockchainHealth:
    """Test the blockchain health endpoint."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_adapter")
    @patch("aragora.server.handlers.erc8004._get_connector")
    async def test_health_success(self, mock_get_connector, mock_get_adapter):
        connector = AsyncMock()
        health = MagicMock()
        health.to_dict.return_value = {"connected": True, "latency_ms": 10}
        connector.health_check.return_value = health
        mock_get_connector.return_value = connector

        adapter = MagicMock()
        adapter.get_health_status.return_value = {"healthy": True}
        mock_get_adapter.return_value = adapter

        result = await handle_blockchain_health()
        assert _status(result) == 200
        body = _body(result)
        assert "connector" in body
        assert "adapter" in body
        assert body["connector"]["connected"] is True
        assert body["adapter"]["healthy"] is True

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_adapter")
    @patch("aragora.server.handlers.erc8004._get_connector")
    async def test_health_adapter_unavailable(self, mock_get_connector, mock_get_adapter):
        connector = AsyncMock()
        health = MagicMock()
        health.to_dict.return_value = {"connected": True}
        connector.health_check.return_value = health
        mock_get_connector.return_value = connector

        mock_get_adapter.side_effect = ConnectionError("adapter down")

        result = await handle_blockchain_health()
        assert _status(result) == 200
        body = _body(result)
        assert "error" in body["adapter"]
        assert "connector" in body

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_connector")
    async def test_health_connector_import_error(self, mock_get_connector):
        mock_get_connector.side_effect = ImportError("web3 missing")

        result = await handle_blockchain_health()
        assert _status(result) == 200
        body = _body(result)
        assert body["available"] is False
        assert "not available" in body.get("error", "").lower()

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_connector")
    async def test_health_connector_connection_error(self, mock_get_connector):
        mock_get_connector.side_effect = ConnectionError("cannot connect")

        result = await handle_blockchain_health()
        assert _status(result) == 200
        body = _body(result)
        assert body["available"] is False

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_adapter")
    @patch("aragora.server.handlers.erc8004._get_connector")
    async def test_health_adapter_attribute_error(self, mock_get_connector, mock_get_adapter):
        connector = AsyncMock()
        health = MagicMock()
        health.to_dict.return_value = {"connected": True}
        connector.health_check.return_value = health
        mock_get_connector.return_value = connector

        mock_get_adapter.side_effect = AttributeError("bad attr")

        result = await handle_blockchain_health()
        assert _status(result) == 200
        body = _body(result)
        assert "error" in body["adapter"]


# ============================================================================
# GET /api/v1/blockchain/agents (list)
# ============================================================================


class TestListAgents:
    """Test the list agents endpoint."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_list_agents_success(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        config = _mock_chain_config(has_identity_registry=True)
        provider = MagicMock()
        provider.get_config.return_value = config
        mock_get_provider.return_value = provider

        identity = _mock_identity(token_id=1)
        contract_inst = MagicMock()
        contract_inst.get_total_supply.return_value = 1
        contract_inst.get_agent.return_value = identity

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": MagicMock(
                IdentityRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            result = await handle_list_agents(skip=0, limit=100)

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["count"] == 1
        assert len(body["agents"]) == 1

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_list_agents_empty(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        config = _mock_chain_config(has_identity_registry=True)
        provider = MagicMock()
        provider.get_config.return_value = config
        mock_get_provider.return_value = provider

        contract_inst = MagicMock()
        contract_inst.get_total_supply.return_value = 0

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": MagicMock(
                IdentityRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            result = await handle_list_agents()

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 0
        assert body["agents"] == []

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_list_agents_no_identity_registry(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        config = _mock_chain_config(has_identity_registry=False)
        provider = MagicMock()
        provider.get_config.return_value = config
        mock_get_provider.return_value = provider

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": MagicMock()
        }):
            result = await handle_list_agents()

        assert _status(result) == 503
        assert "not configured" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    async def test_list_agents_circuit_open(self, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = False
        mock_get_cb.return_value = cb

        result = await handle_list_agents()
        assert _status(result) == 503

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_list_agents_skip_past_total(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        config = _mock_chain_config(has_identity_registry=True)
        provider = MagicMock()
        provider.get_config.return_value = config
        mock_get_provider.return_value = provider

        contract_inst = MagicMock()
        contract_inst.get_total_supply.return_value = 5

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": MagicMock(
                IdentityRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            result = await handle_list_agents(skip=10, limit=100)

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 5
        assert body["count"] == 0
        assert body["agents"] == []

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_list_agents_limit_clamped(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        config = _mock_chain_config(has_identity_registry=True)
        provider = MagicMock()
        provider.get_config.return_value = config
        mock_get_provider.return_value = provider

        contract_inst = MagicMock()
        contract_inst.get_total_supply.return_value = 2
        contract_inst.get_agent.return_value = _mock_identity()

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": MagicMock(
                IdentityRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            # limit=1000 should be clamped to 500
            result = await handle_list_agents(skip=0, limit=1000)

        assert _status(result) == 200
        body = _body(result)
        # With total=2, even with limit=500, we get all 2
        assert body["count"] == 2

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_list_agents_negative_skip_clamped(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        config = _mock_chain_config(has_identity_registry=True)
        provider = MagicMock()
        provider.get_config.return_value = config
        mock_get_provider.return_value = provider

        contract_inst = MagicMock()
        contract_inst.get_total_supply.return_value = 1
        contract_inst.get_agent.return_value = _mock_identity()

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": MagicMock(
                IdentityRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            result = await handle_list_agents(skip=-5, limit=100)

        assert _status(result) == 200
        body = _body(result)
        assert body["skip"] == 0

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_list_agents_individual_fetch_error(self, mock_get_provider, mock_get_cb):
        """Test that individual agent fetch errors are silently skipped."""
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        config = _mock_chain_config(has_identity_registry=True)
        provider = MagicMock()
        provider.get_config.return_value = config
        mock_get_provider.return_value = provider

        contract_inst = MagicMock()
        contract_inst.get_total_supply.return_value = 3

        # First succeeds, second fails, third succeeds
        identity1 = _mock_identity(token_id=1)
        identity3 = _mock_identity(token_id=3)
        contract_inst.get_agent.side_effect = [
            identity1,
            LookupError("missing"),
            identity3,
        ]

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": MagicMock(
                IdentityRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            result = await handle_list_agents(skip=0, limit=100)

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 3
        assert body["count"] == 2  # one was skipped

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_list_agents_import_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.return_value = MagicMock()

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": None,
        }):
            result = await handle_list_agents()
        assert _status(result) == 503

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_list_agents_runtime_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.side_effect = RuntimeError("bad state")

        result = await handle_list_agents()
        assert _status(result) == 500


# ============================================================================
# POST /api/v1/blockchain/agents (register)
# ============================================================================


class TestRegisterAgent:
    """Test the register agent endpoint."""

    @pytest.mark.asyncio
    async def test_register_missing_agent_uri(self):
        result = await handle_register_agent(agent_uri="")
        assert _status(result) == 400
        assert "agent_uri" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    async def test_register_circuit_open(self, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = False
        mock_get_cb.return_value = cb

        result = await handle_register_agent(agent_uri="https://agent.example.com")
        assert _status(result) == 503

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_register_no_identity_registry(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        config = _mock_chain_config(has_identity_registry=False)
        provider = MagicMock()
        provider.get_config.return_value = config
        mock_get_provider.return_value = provider

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": MagicMock(),
            "aragora.blockchain.models": MagicMock(),
            "aragora.blockchain.wallet": MagicMock(),
        }):
            result = await handle_register_agent(agent_uri="https://agent.example.com")

        assert _status(result) == 503
        assert "not configured" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_register_success(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        config = _mock_chain_config(has_identity_registry=True, chain_id=137)
        provider = MagicMock()
        provider.get_config.return_value = config
        mock_get_provider.return_value = provider

        signer = MagicMock()
        signer.address = "0xsigner_addr"

        contract_inst = MagicMock()
        contract_inst.register_agent.return_value = 42

        mock_wallet_mod = MagicMock()
        mock_wallet_mod.WalletSigner.from_env.return_value = signer

        mock_models_mod = MagicMock()

        mock_identity_mod = MagicMock()
        mock_identity_mod.IdentityRegistryContract.return_value = contract_inst

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": mock_identity_mod,
            "aragora.blockchain.models": mock_models_mod,
            "aragora.blockchain.wallet": mock_wallet_mod,
        }):
            result = await handle_register_agent(
                agent_uri="https://agent.example.com",
                metadata={"role": "verifier"},
            )

        assert _status(result) == 201
        body = _body(result)
        assert body["token_id"] == 42
        assert body["agent_uri"] == "https://agent.example.com"
        assert body["owner"] == "0xsigner_addr"
        assert body["chain_id"] == 137

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_register_wallet_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        config = _mock_chain_config(has_identity_registry=True)
        provider = MagicMock()
        provider.get_config.return_value = config
        mock_get_provider.return_value = provider

        mock_wallet_mod = MagicMock()
        mock_wallet_mod.WalletSigner.from_env.side_effect = ValueError("no key")

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": MagicMock(),
            "aragora.blockchain.models": MagicMock(),
            "aragora.blockchain.wallet": mock_wallet_mod,
        }):
            result = await handle_register_agent(agent_uri="https://agent.example.com")

        assert _status(result) == 400

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_register_invalid_metadata(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        config = _mock_chain_config(has_identity_registry=True)
        provider = MagicMock()
        provider.get_config.return_value = config
        mock_get_provider.return_value = provider

        signer = MagicMock()
        signer.address = "0xsigner"

        mock_wallet_mod = MagicMock()
        mock_wallet_mod.WalletSigner.from_env.return_value = signer

        mock_models_mod = MagicMock()
        # MetadataEntry side-effect: _coerce_metadata_value will raise for list
        mock_models_mod.MetadataEntry = MagicMock()

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": MagicMock(),
            "aragora.blockchain.models": mock_models_mod,
            "aragora.blockchain.wallet": mock_wallet_mod,
        }):
            result = await handle_register_agent(
                agent_uri="https://agent.example.com",
                metadata={"bad": [1, 2, 3]},  # list value is invalid
            )

        assert _status(result) == 400
        assert "metadata" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_register_connection_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        config = _mock_chain_config(has_identity_registry=True)
        provider = MagicMock()
        provider.get_config.return_value = config
        mock_get_provider.return_value = provider

        signer = MagicMock()
        signer.address = "0xsigner"

        mock_wallet_mod = MagicMock()
        mock_wallet_mod.WalletSigner.from_env.return_value = signer

        contract_inst = MagicMock()
        contract_inst.register_agent.side_effect = ConnectionError("tx failed")

        mock_identity_mod = MagicMock()
        mock_identity_mod.IdentityRegistryContract.return_value = contract_inst

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": mock_identity_mod,
            "aragora.blockchain.models": MagicMock(),
            "aragora.blockchain.wallet": mock_wallet_mod,
        }):
            result = await handle_register_agent(agent_uri="https://agent.example.com")

        assert _status(result) == 500

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_register_import_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.return_value = MagicMock()

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": None,
            "aragora.blockchain.models": None,
            "aragora.blockchain.wallet": None,
        }):
            result = await handle_register_agent(agent_uri="https://agent.example.com")
        assert _status(result) == 503

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_register_no_metadata(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        config = _mock_chain_config(has_identity_registry=True)
        provider = MagicMock()
        provider.get_config.return_value = config
        mock_get_provider.return_value = provider

        signer = MagicMock()
        signer.address = "0xsigner"

        mock_wallet_mod = MagicMock()
        mock_wallet_mod.WalletSigner.from_env.return_value = signer

        contract_inst = MagicMock()
        contract_inst.register_agent.return_value = 1

        mock_identity_mod = MagicMock()
        mock_identity_mod.IdentityRegistryContract.return_value = contract_inst

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": mock_identity_mod,
            "aragora.blockchain.models": MagicMock(),
            "aragora.blockchain.wallet": mock_wallet_mod,
        }):
            result = await handle_register_agent(
                agent_uri="https://agent.example.com",
                metadata=None,
            )

        assert _status(result) == 201
        # register_agent should have been called with empty metadata list
        contract_inst.register_agent.assert_called_once()


# ============================================================================
# ERC8004Handler.handle() routing via class
# ============================================================================


class TestHandlerRouting:
    """Test routing through the ERC8004Handler.handle() method."""

    @patch("aragora.server.handlers.erc8004.handle_blockchain_config")
    def test_route_to_config(self, mock_fn, handler):
        mock_fn.return_value = MagicMock()
        h = _make_handler(method="GET")
        handler.handle("/api/v1/blockchain/config", {}, h)
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.erc8004.handle_blockchain_health")
    def test_route_to_health(self, mock_fn, handler):
        mock_fn.return_value = MagicMock()
        h = _make_handler(method="GET")
        handler.handle("/api/v1/blockchain/health", {}, h)
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.erc8004.handle_blockchain_sync")
    def test_route_to_sync(self, mock_fn, handler):
        mock_fn.return_value = MagicMock()
        h = _make_handler(body={"sync_identities": True}, method="POST")
        handler.handle("/api/v1/blockchain/sync", {}, h)
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.erc8004.handle_list_agents")
    def test_route_to_list_agents(self, mock_fn, handler):
        mock_fn.return_value = MagicMock()
        h = _make_handler(method="GET")
        handler.handle("/api/v1/blockchain/agents", {}, h)
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.erc8004.handle_register_agent")
    def test_route_to_register_agent(self, mock_fn, handler):
        mock_fn.return_value = MagicMock()
        h = _make_handler(body={"agent_uri": "https://test.com"}, method="POST")
        handler.handle("/api/v1/blockchain/agents", {}, h)
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.erc8004.handle_get_agent")
    def test_route_to_get_agent(self, mock_fn, handler):
        mock_fn.return_value = MagicMock()
        h = _make_handler(method="GET")
        handler.handle("/api/v1/blockchain/agents/42", {}, h)
        mock_fn.assert_called_once_with(42)

    @patch("aragora.server.handlers.erc8004.handle_get_reputation")
    def test_route_to_get_reputation(self, mock_fn, handler):
        mock_fn.return_value = MagicMock()
        h = _make_handler(method="GET")
        handler.handle(
            "/api/v1/blockchain/agents/10/reputation",
            {"tag1": "skill", "tag2": "domain"},
            h,
        )
        mock_fn.assert_called_once_with(10, tag1="skill", tag2="domain")

    @patch("aragora.server.handlers.erc8004.handle_get_validations")
    def test_route_to_get_validations(self, mock_fn, handler):
        mock_fn.return_value = MagicMock()
        h = _make_handler(method="GET")
        handler.handle(
            "/api/v1/blockchain/agents/7/validations",
            {"tag": "accuracy"},
            h,
        )
        mock_fn.assert_called_once_with(7, tag="accuracy")


class TestHandlerRoutingErrors:
    """Test error paths in ERC8004Handler.handle() routing."""

    def test_agents_method_not_allowed(self, handler):
        h = _make_handler(method="DELETE")
        result = handler.handle("/api/v1/blockchain/agents", {}, h)
        assert _status(result) == 405
        assert "not allowed" in _body(result).get("error", "").lower()

    def test_invalid_token_id(self, handler):
        h = _make_handler(method="GET")
        result = handler.handle("/api/v1/blockchain/agents/notanumber", {}, h)
        assert _status(result) == 400
        assert "invalid token_id" in _body(result).get("error", "").lower()

    def test_invalid_agent_subpath(self, handler):
        h = _make_handler(method="GET")
        result = handler.handle("/api/v1/blockchain/agents/1/unknown", {}, h)
        assert _status(result) == 400

    def test_empty_agent_path_suffix(self, handler):
        h = _make_handler(method="GET")
        result = handler.handle("/api/v1/blockchain/agents/", {}, h)
        assert _status(result) == 400

    def test_invalid_path(self, handler):
        h = _make_handler(method="GET")
        result = handler.handle("/api/v1/blockchain/nonexistent", {}, h)
        assert _status(result) == 400

    def test_invalid_skip_query_param(self, handler):
        h = _make_handler(method="GET")
        result = handler.handle("/api/v1/blockchain/agents", {"skip": "abc"}, h)
        assert _status(result) == 400

    def test_invalid_limit_query_param(self, handler):
        h = _make_handler(method="GET")
        result = handler.handle("/api/v1/blockchain/agents", {"limit": "xyz"}, h)
        assert _status(result) == 400

    def test_metadata_not_object(self, handler):
        h = _make_handler(body={"agent_uri": "https://test.com", "metadata": "string"}, method="POST")
        result = handler.handle("/api/v1/blockchain/agents", {}, h)
        assert _status(result) == 400
        assert "metadata must be an object" in _body(result).get("error", "")

    def test_sync_body_defaults(self, handler):
        """Sync with empty body should use all defaults."""
        with patch("aragora.server.handlers.erc8004.handle_blockchain_sync") as mock_fn:
            mock_fn.return_value = MagicMock()
            h = _make_handler(body={}, method="POST")
            handler.handle("/api/v1/blockchain/sync", {}, h)
            mock_fn.assert_called_once_with(
                sync_identities=True,
                sync_reputation=True,
                sync_validations=True,
                agent_ids=None,
            )

    def test_sync_body_partial(self, handler):
        """Sync with partial body."""
        with patch("aragora.server.handlers.erc8004.handle_blockchain_sync") as mock_fn:
            mock_fn.return_value = MagicMock()
            h = _make_handler(
                body={"sync_identities": False, "agent_ids": [1, 2]},
                method="POST",
            )
            handler.handle("/api/v1/blockchain/sync", {}, h)
            mock_fn.assert_called_once_with(
                sync_identities=False,
                sync_reputation=True,
                sync_validations=True,
                agent_ids=[1, 2],
            )

    def test_reputation_with_list_tag1(self, handler):
        """Query param as list should extract first element."""
        with patch("aragora.server.handlers.erc8004.handle_get_reputation") as mock_fn:
            mock_fn.return_value = MagicMock()
            h = _make_handler(method="GET")
            handler.handle(
                "/api/v1/blockchain/agents/1/reputation",
                {"tag1": ["first", "second"]},
                h,
            )
            mock_fn.assert_called_once_with(1, tag1="first", tag2="")

    def test_validations_with_list_tag(self, handler):
        """Query param as list should extract first element."""
        with patch("aragora.server.handlers.erc8004.handle_get_validations") as mock_fn:
            mock_fn.return_value = MagicMock()
            h = _make_handler(method="GET")
            handler.handle(
                "/api/v1/blockchain/agents/1/validations",
                {"tag": ["first"]},
                h,
            )
            mock_fn.assert_called_once_with(1, tag="first")

    def test_reputation_with_wrong_method(self, handler):
        """POST to reputation endpoint should fail."""
        h = _make_handler(method="POST")
        result = handler.handle("/api/v1/blockchain/agents/1/reputation", {}, h)
        # The handler checks method == "GET" for reputation
        assert _status(result) == 400

    def test_validations_with_wrong_method(self, handler):
        """POST to validations endpoint should fail."""
        h = _make_handler(method="POST")
        result = handler.handle("/api/v1/blockchain/agents/1/validations", {}, h)
        assert _status(result) == 400


# ============================================================================
# Handler ROUTES attribute
# ============================================================================


class TestHandlerRoutes:
    """Test the ROUTES class attribute."""

    def test_routes_defined(self, handler):
        assert len(handler.ROUTES) > 0

    def test_routes_include_config(self, handler):
        assert "/api/v1/blockchain/config" in handler.ROUTES

    def test_routes_include_health(self, handler):
        assert "/api/v1/blockchain/health" in handler.ROUTES

    def test_routes_include_agents(self, handler):
        assert "/api/v1/blockchain/agents" in handler.ROUTES

    def test_routes_include_agents_wildcard(self, handler):
        assert "/api/v1/blockchain/agents/*" in handler.ROUTES

    def test_routes_include_sync(self, handler):
        assert "/api/v1/blockchain/sync" in handler.ROUTES


# ============================================================================
# Module-level lazy initializers
# ============================================================================


class TestLazyInitializers:
    """Test module-level lazy initialization functions."""

    @patch("aragora.server.handlers.erc8004.get_circuit_breaker")
    def test_get_circuit_breaker_caches(self, mock_get_cb):
        import aragora.server.handlers.erc8004 as mod

        mod._blockchain_circuit_breaker = None
        mock_cb = MagicMock()
        mock_get_cb.return_value = mock_cb

        from aragora.server.handlers.erc8004 import _get_circuit_breaker

        cb1 = _get_circuit_breaker()
        cb2 = _get_circuit_breaker()
        assert cb1 is cb2
        # Should only call get_circuit_breaker once (cached)
        mock_get_cb.assert_called_once()

    @patch("aragora.server.handlers.erc8004.get_circuit_breaker")
    def test_get_circuit_breaker_creates_with_params(self, mock_get_cb):
        import aragora.server.handlers.erc8004 as mod

        mod._blockchain_circuit_breaker = None

        mock_cb = MagicMock()
        mock_get_cb.return_value = mock_cb

        result = mod._get_circuit_breaker()
        assert result is mock_cb
        mock_get_cb.assert_called_once_with(
            "erc8004_blockchain",
            failure_threshold=5,
            cooldown_seconds=30,
            half_open_max_calls=2,
        )

    def test_get_provider_import_error(self):
        from aragora.server.handlers.erc8004 import _get_provider
        import aragora.server.handlers.erc8004 as mod

        mod._provider = None

        with patch.dict("sys.modules", {"aragora.blockchain.provider": None}):
            with pytest.raises(ImportError):
                _get_provider()

    def test_get_provider_caches(self):
        from aragora.server.handlers.erc8004 import _get_provider
        import aragora.server.handlers.erc8004 as mod

        mock_provider = MagicMock()
        mod._provider = mock_provider

        result = _get_provider()
        assert result is mock_provider

    def test_get_connector_caches(self):
        from aragora.server.handlers.erc8004 import _get_connector
        import aragora.server.handlers.erc8004 as mod

        mock_connector = MagicMock()
        mod._connector = mock_connector

        result = _get_connector()
        assert result is mock_connector

    def test_get_adapter_caches(self):
        from aragora.server.handlers.erc8004 import _get_adapter
        import aragora.server.handlers.erc8004 as mod

        mock_adapter = MagicMock()
        mod._adapter = mock_adapter

        result = _get_adapter()
        assert result is mock_adapter

    def test_get_provider_connection_error(self):
        from aragora.server.handlers.erc8004 import _get_provider
        import aragora.server.handlers.erc8004 as mod

        mod._provider = None

        mock_mod = MagicMock()
        mock_mod.Web3Provider.from_env.side_effect = ConnectionError("fail")

        with patch.dict("sys.modules", {"aragora.blockchain.provider": mock_mod}):
            with pytest.raises(ConnectionError):
                _get_provider()

    def test_get_provider_timeout_error(self):
        from aragora.server.handlers.erc8004 import _get_provider
        import aragora.server.handlers.erc8004 as mod

        mod._provider = None

        mock_mod = MagicMock()
        mock_mod.Web3Provider.from_env.side_effect = TimeoutError("timed out")

        with patch.dict("sys.modules", {"aragora.blockchain.provider": mock_mod}):
            with pytest.raises(TimeoutError):
                _get_provider()

    def test_get_provider_os_error(self):
        from aragora.server.handlers.erc8004 import _get_provider
        import aragora.server.handlers.erc8004 as mod

        mod._provider = None

        mock_mod = MagicMock()
        mock_mod.Web3Provider.from_env.side_effect = OSError("socket error")

        with patch.dict("sys.modules", {"aragora.blockchain.provider": mock_mod}):
            with pytest.raises(OSError):
                _get_provider()

    def test_get_provider_value_error(self):
        from aragora.server.handlers.erc8004 import _get_provider
        import aragora.server.handlers.erc8004 as mod

        mod._provider = None

        mock_mod = MagicMock()
        mock_mod.Web3Provider.from_env.side_effect = ValueError("bad config")

        with patch.dict("sys.modules", {"aragora.blockchain.provider": mock_mod}):
            with pytest.raises(ValueError):
                _get_provider()

    def test_get_provider_runtime_error(self):
        from aragora.server.handlers.erc8004 import _get_provider
        import aragora.server.handlers.erc8004 as mod

        mod._provider = None

        mock_mod = MagicMock()
        mock_mod.Web3Provider.from_env.side_effect = RuntimeError("bad state")

        with patch.dict("sys.modules", {"aragora.blockchain.provider": mock_mod}):
            with pytest.raises(RuntimeError):
                _get_provider()

    def test_get_connector_creates_via_from_env(self):
        from aragora.server.handlers.erc8004 import _get_connector
        import aragora.server.handlers.erc8004 as mod

        mod._connector = None

        mock_connector = MagicMock()
        mock_conn_mod = MagicMock()
        mock_conn_mod.ERC8004Connector.from_env.return_value = mock_connector

        with patch.dict("sys.modules", {"aragora.connectors.blockchain": mock_conn_mod}):
            result = _get_connector()
        assert result is mock_connector

    def test_get_adapter_creates_with_provider(self):
        from aragora.server.handlers.erc8004 import _get_adapter
        import aragora.server.handlers.erc8004 as mod

        mod._adapter = None
        mock_provider = MagicMock()
        mod._provider = mock_provider

        mock_adapter = MagicMock()
        mock_adapter_mod = MagicMock()
        mock_adapter_mod.ERC8004Adapter.return_value = mock_adapter

        with patch.dict("sys.modules", {
            "aragora.knowledge.mound.adapters.erc8004_adapter": mock_adapter_mod,
        }):
            result = _get_adapter()
        assert result is mock_adapter
        mock_adapter_mod.ERC8004Adapter.assert_called_once_with(provider=mock_provider)


# ============================================================================
# Additional config edge cases
# ============================================================================


class TestBlockchainConfigEdgeCases:
    """Additional edge cases for blockchain config endpoint."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_config_exactly_50_char_rpc_url(self, mock_get_provider, mock_get_cb):
        """RPC URL exactly 50 chars should NOT be truncated."""
        rpc_url = "http://" + "x" * 43  # 7 + 43 = 50
        assert len(rpc_url) == 50
        config = _mock_chain_config(rpc_url=rpc_url)
        provider = MagicMock()
        provider.get_config.return_value = config
        provider.is_connected.return_value = True
        provider.get_health_status.return_value = {}
        mock_get_provider.return_value = provider

        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        result = await handle_blockchain_config()
        body = _body(result)
        assert not body["rpc_url"].endswith("...")
        assert body["rpc_url"] == rpc_url

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_config_51_char_rpc_url_truncated(self, mock_get_provider, mock_get_cb):
        """RPC URL of 51 chars should be truncated."""
        rpc_url = "http://" + "x" * 44  # 7 + 44 = 51
        assert len(rpc_url) == 51
        config = _mock_chain_config(rpc_url=rpc_url)
        provider = MagicMock()
        provider.get_config.return_value = config
        provider.is_connected.return_value = True
        provider.get_health_status.return_value = {}
        mock_get_provider.return_value = provider

        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        result = await handle_blockchain_config()
        body = _body(result)
        assert body["rpc_url"].endswith("...")

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_config_timeout_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.side_effect = TimeoutError("timed out")

        result = await handle_blockchain_config()
        assert _status(result) == 500

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_config_os_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.side_effect = OSError("socket error")

        result = await handle_blockchain_config()
        assert _status(result) == 500

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_config_value_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.side_effect = ValueError("bad value")

        result = await handle_blockchain_config()
        assert _status(result) == 500

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_config_attribute_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        # Provider exists but get_config raises AttributeError
        provider = MagicMock()
        provider.get_config.side_effect = AttributeError("no config")
        mock_get_provider.return_value = provider

        result = await handle_blockchain_config()
        assert _status(result) == 500

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_config_null_registry_addresses(self, mock_get_provider, mock_get_cb):
        """Config with None registry addresses should return None in response."""
        config = _mock_chain_config(
            identity_registry_address=None,
            reputation_registry_address=None,
            validation_registry_address=None,
        )
        # Need to override has_identity_registry since _mock_chain_config uses MagicMock
        config.identity_registry_address = None
        config.reputation_registry_address = None
        config.validation_registry_address = None

        provider = MagicMock()
        provider.get_config.return_value = config
        provider.is_connected.return_value = False
        provider.get_health_status.return_value = {"status": "disconnected"}
        mock_get_provider.return_value = provider

        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        result = await handle_blockchain_config()
        assert _status(result) == 200
        body = _body(result)
        assert body["is_connected"] is False

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_config_all_response_fields(self, mock_get_provider, mock_get_cb):
        """Verify all expected fields are present in config response."""
        config = _mock_chain_config()
        provider = MagicMock()
        provider.get_config.return_value = config
        provider.is_connected.return_value = True
        provider.get_health_status.return_value = {"status": "ok"}
        mock_get_provider.return_value = provider

        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        result = await handle_blockchain_config()
        body = _body(result)
        expected_keys = {
            "chain_id", "rpc_url", "identity_registry", "reputation_registry",
            "validation_registry", "block_confirmations", "is_connected", "health",
        }
        assert set(body.keys()) == expected_keys


# ============================================================================
# Additional get agent edge cases
# ============================================================================


class TestGetAgentEdgeCases:
    """Additional edge cases for get agent endpoint."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_get_agent_timeout_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.return_value = MagicMock()

        contract_inst = MagicMock()
        contract_inst.get_agent.side_effect = TimeoutError("rpc timeout")

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": MagicMock(
                IdentityRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            result = await handle_get_agent(1)
        assert _status(result) == 404

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_get_agent_value_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.return_value = MagicMock()

        contract_inst = MagicMock()
        contract_inst.get_agent.side_effect = ValueError("invalid token id")

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": MagicMock(
                IdentityRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            result = await handle_get_agent(0)
        assert _status(result) == 404

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_get_agent_runtime_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.return_value = MagicMock()

        contract_inst = MagicMock()
        contract_inst.get_agent.side_effect = RuntimeError("contract reverted")

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": MagicMock(
                IdentityRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            result = await handle_get_agent(1)
        assert _status(result) == 404

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_get_agent_connection_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.return_value = MagicMock()

        contract_inst = MagicMock()
        contract_inst.get_agent.side_effect = ConnectionError("connection lost")

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": MagicMock(
                IdentityRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            result = await handle_get_agent(1)
        assert _status(result) == 404


# ============================================================================
# Additional reputation edge cases
# ============================================================================


class TestGetReputationEdgeCases:
    """Additional edge cases for reputation endpoint."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_get_reputation_timeout_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.return_value = MagicMock()

        contract_inst = MagicMock()
        contract_inst.get_summary.side_effect = TimeoutError("rpc timeout")

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.reputation": MagicMock(
                ReputationRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            result = await handle_get_reputation(1)
        assert _status(result) == 404

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_get_reputation_runtime_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.return_value = MagicMock()

        contract_inst = MagicMock()
        contract_inst.get_summary.side_effect = RuntimeError("bad state")

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.reputation": MagicMock(
                ReputationRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            result = await handle_get_reputation(1)
        assert _status(result) == 404

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_get_reputation_all_response_fields(self, mock_get_provider, mock_get_cb):
        """Verify all expected fields in reputation response."""
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.return_value = MagicMock()

        summary = _mock_reputation_summary(
            agent_id=1, count=20, summary_value=900,
            summary_value_decimals=3, normalized_value=0.9,
            tag1="skill", tag2="domain",
        )
        contract_inst = MagicMock()
        contract_inst.get_summary.return_value = summary

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.reputation": MagicMock(
                ReputationRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            result = await handle_get_reputation(1, tag1="skill", tag2="domain")

        body = _body(result)
        expected_keys = {
            "agent_id", "count", "summary_value",
            "summary_value_decimals", "normalized_value", "tag1", "tag2",
        }
        assert set(body.keys()) == expected_keys
        assert body["summary_value"] == 900
        assert body["summary_value_decimals"] == 3
        assert body["normalized_value"] == 0.9


# ============================================================================
# Additional validations edge cases
# ============================================================================


class TestGetValidationsEdgeCases:
    """Additional edge cases for validations endpoint."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_get_validations_timeout_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.return_value = MagicMock()

        contract_inst = MagicMock()
        contract_inst.get_summary.side_effect = TimeoutError("rpc timeout")

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.validation": MagicMock(
                ValidationRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            result = await handle_get_validations(1)
        assert _status(result) == 404

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_get_validations_import_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.return_value = MagicMock()

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.validation": None,
        }):
            result = await handle_get_validations(1)
        assert _status(result) in (404, 503)

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_get_validations_all_response_fields(self, mock_get_provider, mock_get_cb):
        """Verify all expected fields in validation response."""
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.return_value = MagicMock()

        summary = _mock_validation_summary(
            agent_id=7, count=15, average_response=0.95, tag="precision",
        )
        contract_inst = MagicMock()
        contract_inst.get_summary.return_value = summary

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.validation": MagicMock(
                ValidationRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            result = await handle_get_validations(7, tag="precision")

        body = _body(result)
        expected_keys = {"agent_id", "count", "average_response", "tag"}
        assert set(body.keys()) == expected_keys
        assert body["average_response"] == 0.95

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_get_validations_os_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.return_value = MagicMock()

        contract_inst = MagicMock()
        contract_inst.get_summary.side_effect = OSError("socket error")

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.validation": MagicMock(
                ValidationRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            result = await handle_get_validations(1)
        assert _status(result) == 404


# ============================================================================
# Additional sync edge cases
# ============================================================================


class TestBlockchainSyncEdgeCases:
    """Additional edge cases for blockchain sync endpoint."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_adapter")
    async def test_sync_with_none_errors(self, mock_get_adapter, mock_get_cb):
        """Sync result with None errors should return empty list."""
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        sync_result = _mock_sync_result(errors=None)
        adapter = AsyncMock()
        adapter.sync_to_km.return_value = sync_result
        mock_get_adapter.return_value = adapter

        result = await handle_blockchain_sync()
        body = _body(result)
        assert body["errors"] == []

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_adapter")
    async def test_sync_with_exactly_10_errors(self, mock_get_adapter, mock_get_cb):
        """Sync result with exactly 10 errors should not truncate."""
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        errors = [f"error-{i}" for i in range(10)]
        adapter = AsyncMock()
        adapter.sync_to_km.return_value = _mock_sync_result(errors=errors)
        mock_get_adapter.return_value = adapter

        result = await handle_blockchain_sync()
        body = _body(result)
        assert len(body["errors"]) == 10

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_adapter")
    async def test_sync_timeout_error(self, mock_get_adapter, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        adapter = AsyncMock()
        adapter.sync_to_km.side_effect = TimeoutError("sync timeout")
        mock_get_adapter.return_value = adapter

        result = await handle_blockchain_sync()
        # default error_response returns 400
        assert _status(result) == 400

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_adapter")
    async def test_sync_runtime_error(self, mock_get_adapter, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        adapter = AsyncMock()
        adapter.sync_to_km.side_effect = RuntimeError("adapter crash")
        mock_get_adapter.return_value = adapter

        result = await handle_blockchain_sync()
        assert _status(result) == 400

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_adapter")
    async def test_sync_value_error(self, mock_get_adapter, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        adapter = AsyncMock()
        adapter.sync_to_km.side_effect = ValueError("bad input")
        mock_get_adapter.return_value = adapter

        result = await handle_blockchain_sync()
        assert _status(result) == 400

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_adapter")
    async def test_sync_all_flags_false(self, mock_get_adapter, mock_get_cb):
        """Sync with all flags disabled should still call adapter."""
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        adapter = AsyncMock()
        adapter.sync_to_km.return_value = _mock_sync_result(
            records_synced=0, records_skipped=0, records_failed=0,
        )
        mock_get_adapter.return_value = adapter

        result = await handle_blockchain_sync(
            sync_identities=False,
            sync_reputation=False,
            sync_validations=False,
        )
        assert _status(result) == 200
        adapter.sync_to_km.assert_called_once_with(
            agent_ids=None,
            sync_identities=False,
            sync_reputation=False,
            sync_validations=False,
        )

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_adapter")
    async def test_sync_response_fields(self, mock_get_adapter, mock_get_cb):
        """Verify all expected fields in sync response."""
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        adapter = AsyncMock()
        adapter.sync_to_km.return_value = _mock_sync_result()
        mock_get_adapter.return_value = adapter

        result = await handle_blockchain_sync()
        body = _body(result)
        expected_keys = {
            "records_synced", "records_skipped", "records_failed",
            "duration_ms", "errors",
        }
        assert set(body.keys()) == expected_keys


# ============================================================================
# Additional health edge cases
# ============================================================================


class TestBlockchainHealthEdgeCases:
    """Additional edge cases for blockchain health endpoint."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_adapter")
    @patch("aragora.server.handlers.erc8004._get_connector")
    async def test_health_adapter_import_error(self, mock_get_connector, mock_get_adapter):
        """Adapter ImportError should put error in adapter status but still succeed."""
        connector = AsyncMock()
        health = MagicMock()
        health.to_dict.return_value = {"connected": True}
        connector.health_check.return_value = health
        mock_get_connector.return_value = connector

        mock_get_adapter.side_effect = ImportError("adapter not installed")

        result = await handle_blockchain_health()
        assert _status(result) == 200
        body = _body(result)
        assert "error" in body["adapter"]
        assert "connector" in body

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_adapter")
    @patch("aragora.server.handlers.erc8004._get_connector")
    async def test_health_adapter_timeout_error(self, mock_get_connector, mock_get_adapter):
        connector = AsyncMock()
        health = MagicMock()
        health.to_dict.return_value = {"connected": True}
        connector.health_check.return_value = health
        mock_get_connector.return_value = connector

        mock_get_adapter.side_effect = TimeoutError("adapter timed out")

        result = await handle_blockchain_health()
        assert _status(result) == 200
        body = _body(result)
        assert "error" in body["adapter"]

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_connector")
    async def test_health_connector_timeout_error(self, mock_get_connector):
        mock_get_connector.side_effect = TimeoutError("connector timed out")

        result = await handle_blockchain_health()
        assert _status(result) == 200
        body = _body(result)
        assert body["available"] is False

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_connector")
    async def test_health_connector_runtime_error(self, mock_get_connector):
        mock_get_connector.side_effect = RuntimeError("bad connector state")

        result = await handle_blockchain_health()
        assert _status(result) == 200
        body = _body(result)
        assert body["available"] is False

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_connector")
    async def test_health_connector_attribute_error(self, mock_get_connector):
        mock_get_connector.side_effect = AttributeError("missing method")

        result = await handle_blockchain_health()
        assert _status(result) == 200
        body = _body(result)
        assert body["available"] is False

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_connector")
    async def test_health_connector_value_error(self, mock_get_connector):
        mock_get_connector.side_effect = ValueError("bad config")

        result = await handle_blockchain_health()
        assert _status(result) == 200
        body = _body(result)
        assert body["available"] is False

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_adapter")
    @patch("aragora.server.handlers.erc8004._get_connector")
    async def test_health_adapter_runtime_error(self, mock_get_connector, mock_get_adapter):
        connector = AsyncMock()
        health = MagicMock()
        health.to_dict.return_value = {"connected": True}
        connector.health_check.return_value = health
        mock_get_connector.return_value = connector

        mock_get_adapter.side_effect = RuntimeError("adapter broken")

        result = await handle_blockchain_health()
        assert _status(result) == 200
        body = _body(result)
        assert "error" in body["adapter"]


# ============================================================================
# Additional list agents edge cases
# ============================================================================


class TestListAgentsEdgeCases:
    """Additional edge cases for list agents endpoint."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_list_agents_multiple_pages(self, mock_get_provider, mock_get_cb):
        """List agents with skip to simulate pagination."""
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        config = _mock_chain_config(has_identity_registry=True)
        provider = MagicMock()
        provider.get_config.return_value = config
        mock_get_provider.return_value = provider

        contract_inst = MagicMock()
        contract_inst.get_total_supply.return_value = 5

        identities = [_mock_identity(token_id=i) for i in range(3, 6)]
        contract_inst.get_agent.side_effect = identities

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": MagicMock(
                IdentityRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            result = await handle_list_agents(skip=2, limit=3)

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 5
        assert body["skip"] == 2
        assert body["limit"] == 3
        assert body["count"] == 3

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_list_agents_limit_zero_becomes_one(self, mock_get_provider, mock_get_cb):
        """Limit of 0 should be clamped to 1."""
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        config = _mock_chain_config(has_identity_registry=True)
        provider = MagicMock()
        provider.get_config.return_value = config
        mock_get_provider.return_value = provider

        contract_inst = MagicMock()
        contract_inst.get_total_supply.return_value = 5
        contract_inst.get_agent.return_value = _mock_identity()

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": MagicMock(
                IdentityRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            result = await handle_list_agents(skip=0, limit=0)

        assert _status(result) == 200
        body = _body(result)
        # limit clamped to max(0,1) = 1
        assert body["limit"] == 1
        assert body["count"] == 1

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_list_agents_connection_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.side_effect = ConnectionError("connection failed")

        result = await handle_list_agents()
        assert _status(result) == 500

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_list_agents_timeout_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb
        mock_get_provider.side_effect = TimeoutError("timed out")

        result = await handle_list_agents()
        assert _status(result) == 500

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_list_agents_all_agents_fail(self, mock_get_provider, mock_get_cb):
        """When all individual agents fail, count should be 0."""
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        config = _mock_chain_config(has_identity_registry=True)
        provider = MagicMock()
        provider.get_config.return_value = config
        mock_get_provider.return_value = provider

        contract_inst = MagicMock()
        contract_inst.get_total_supply.return_value = 3
        contract_inst.get_agent.side_effect = LookupError("missing")

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": MagicMock(
                IdentityRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            result = await handle_list_agents(skip=0, limit=100)

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 3
        assert body["count"] == 0
        assert body["agents"] == []

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_list_agents_response_fields(self, mock_get_provider, mock_get_cb):
        """Verify all expected fields in list agents response."""
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        config = _mock_chain_config(has_identity_registry=True)
        provider = MagicMock()
        provider.get_config.return_value = config
        mock_get_provider.return_value = provider

        contract_inst = MagicMock()
        contract_inst.get_total_supply.return_value = 1
        contract_inst.get_agent.return_value = _mock_identity()

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": MagicMock(
                IdentityRegistryContract=MagicMock(return_value=contract_inst)
            )
        }):
            result = await handle_list_agents()

        body = _body(result)
        expected_keys = {"total", "skip", "limit", "count", "agents"}
        assert set(body.keys()) == expected_keys


# ============================================================================
# Additional register agent edge cases
# ============================================================================


class TestRegisterAgentEdgeCases:
    """Additional edge cases for register agent endpoint."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_register_runtime_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        config = _mock_chain_config(has_identity_registry=True)
        provider = MagicMock()
        provider.get_config.return_value = config
        mock_get_provider.return_value = provider

        signer = MagicMock()
        signer.address = "0xsigner"

        mock_wallet_mod = MagicMock()
        mock_wallet_mod.WalletSigner.from_env.return_value = signer

        contract_inst = MagicMock()
        contract_inst.register_agent.side_effect = RuntimeError("tx reverted")

        mock_identity_mod = MagicMock()
        mock_identity_mod.IdentityRegistryContract.return_value = contract_inst

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": mock_identity_mod,
            "aragora.blockchain.models": MagicMock(),
            "aragora.blockchain.wallet": mock_wallet_mod,
        }):
            result = await handle_register_agent(agent_uri="https://agent.example.com")

        assert _status(result) == 500

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_register_timeout_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        config = _mock_chain_config(has_identity_registry=True)
        provider = MagicMock()
        provider.get_config.return_value = config
        mock_get_provider.return_value = provider

        signer = MagicMock()
        signer.address = "0xsigner"

        mock_wallet_mod = MagicMock()
        mock_wallet_mod.WalletSigner.from_env.return_value = signer

        contract_inst = MagicMock()
        contract_inst.register_agent.side_effect = TimeoutError("tx timeout")

        mock_identity_mod = MagicMock()
        mock_identity_mod.IdentityRegistryContract.return_value = contract_inst

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": mock_identity_mod,
            "aragora.blockchain.models": MagicMock(),
            "aragora.blockchain.wallet": mock_wallet_mod,
        }):
            result = await handle_register_agent(agent_uri="https://agent.example.com")

        assert _status(result) == 500

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_register_os_error(self, mock_get_provider, mock_get_cb):
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        config = _mock_chain_config(has_identity_registry=True)
        provider = MagicMock()
        provider.get_config.return_value = config
        mock_get_provider.return_value = provider

        signer = MagicMock()
        signer.address = "0xsigner"

        mock_wallet_mod = MagicMock()
        mock_wallet_mod.WalletSigner.from_env.return_value = signer

        contract_inst = MagicMock()
        contract_inst.register_agent.side_effect = OSError("socket error")

        mock_identity_mod = MagicMock()
        mock_identity_mod.IdentityRegistryContract.return_value = contract_inst

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": mock_identity_mod,
            "aragora.blockchain.models": MagicMock(),
            "aragora.blockchain.wallet": mock_wallet_mod,
        }):
            result = await handle_register_agent(agent_uri="https://agent.example.com")

        assert _status(result) == 500

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_register_with_multiple_metadata(self, mock_get_provider, mock_get_cb):
        """Register with multiple valid metadata entries."""
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        config = _mock_chain_config(has_identity_registry=True, chain_id=42)
        provider = MagicMock()
        provider.get_config.return_value = config
        mock_get_provider.return_value = provider

        signer = MagicMock()
        signer.address = "0xsigner"

        mock_wallet_mod = MagicMock()
        mock_wallet_mod.WalletSigner.from_env.return_value = signer

        contract_inst = MagicMock()
        contract_inst.register_agent.return_value = 100

        mock_identity_mod = MagicMock()
        mock_identity_mod.IdentityRegistryContract.return_value = contract_inst

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": mock_identity_mod,
            "aragora.blockchain.models": MagicMock(),
            "aragora.blockchain.wallet": mock_wallet_mod,
        }):
            result = await handle_register_agent(
                agent_uri="https://agent.example.com",
                metadata={"role": "verifier", "level": 42, "active": True},
            )

        assert _status(result) == 201
        body = _body(result)
        assert body["token_id"] == 100

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_register_with_empty_metadata(self, mock_get_provider, mock_get_cb):
        """Register with empty metadata dict should succeed."""
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        config = _mock_chain_config(has_identity_registry=True)
        provider = MagicMock()
        provider.get_config.return_value = config
        mock_get_provider.return_value = provider

        signer = MagicMock()
        signer.address = "0xsigner"

        mock_wallet_mod = MagicMock()
        mock_wallet_mod.WalletSigner.from_env.return_value = signer

        contract_inst = MagicMock()
        contract_inst.register_agent.return_value = 1

        mock_identity_mod = MagicMock()
        mock_identity_mod.IdentityRegistryContract.return_value = contract_inst

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": mock_identity_mod,
            "aragora.blockchain.models": MagicMock(),
            "aragora.blockchain.wallet": mock_wallet_mod,
        }):
            result = await handle_register_agent(
                agent_uri="https://agent.example.com",
                metadata={},
            )

        assert _status(result) == 201

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.erc8004._get_circuit_breaker")
    @patch("aragora.server.handlers.erc8004._get_provider")
    async def test_register_response_fields(self, mock_get_provider, mock_get_cb):
        """Verify all expected fields in register response."""
        cb = MagicMock()
        cb.can_execute.return_value = True
        mock_get_cb.return_value = cb

        config = _mock_chain_config(has_identity_registry=True, chain_id=5)
        provider = MagicMock()
        provider.get_config.return_value = config
        mock_get_provider.return_value = provider

        signer = MagicMock()
        signer.address = "0xaddr"

        mock_wallet_mod = MagicMock()
        mock_wallet_mod.WalletSigner.from_env.return_value = signer

        contract_inst = MagicMock()
        contract_inst.register_agent.return_value = 99

        mock_identity_mod = MagicMock()
        mock_identity_mod.IdentityRegistryContract.return_value = contract_inst

        with patch.dict("sys.modules", {
            "aragora.blockchain.contracts.identity": mock_identity_mod,
            "aragora.blockchain.models": MagicMock(),
            "aragora.blockchain.wallet": mock_wallet_mod,
        }):
            result = await handle_register_agent(agent_uri="https://a.com")

        body = _body(result)
        expected_keys = {"token_id", "agent_uri", "owner", "chain_id"}
        assert set(body.keys()) == expected_keys
        assert body["chain_id"] == 5
        assert body["owner"] == "0xaddr"


# ============================================================================
# Additional handler routing edge cases
# ============================================================================


class TestHandlerRoutingAdditional:
    """Additional routing edge cases for ERC8004Handler.handle()."""

    @patch("aragora.server.handlers.erc8004.handle_get_agent")
    def test_route_to_get_agent_large_token_id(self, mock_fn, handler):
        mock_fn.return_value = MagicMock()
        h = _make_handler(method="GET")
        handler.handle("/api/v1/blockchain/agents/999999", {}, h)
        mock_fn.assert_called_once_with(999999)

    @patch("aragora.server.handlers.erc8004.handle_get_agent")
    def test_route_to_get_agent_zero(self, mock_fn, handler):
        mock_fn.return_value = MagicMock()
        h = _make_handler(method="GET")
        handler.handle("/api/v1/blockchain/agents/0", {}, h)
        mock_fn.assert_called_once_with(0)

    def test_agents_put_method_not_allowed(self, handler):
        h = _make_handler(method="PUT")
        result = handler.handle("/api/v1/blockchain/agents", {}, h)
        assert _status(result) == 405

    def test_agents_patch_method_not_allowed(self, handler):
        h = _make_handler(method="PATCH")
        result = handler.handle("/api/v1/blockchain/agents", {}, h)
        assert _status(result) == 405

    def test_deep_invalid_agent_subpath(self, handler):
        h = _make_handler(method="GET")
        result = handler.handle("/api/v1/blockchain/agents/1/reputation/extra", {}, h)
        assert _status(result) == 400

    def test_triple_suffix_agent_path(self, handler):
        h = _make_handler(method="GET")
        result = handler.handle("/api/v1/blockchain/agents/1/a/b/c", {}, h)
        assert _status(result) == 400

    @patch("aragora.server.handlers.erc8004.handle_list_agents")
    def test_route_list_with_valid_params(self, mock_fn, handler):
        mock_fn.return_value = MagicMock()
        h = _make_handler(method="GET")
        handler.handle("/api/v1/blockchain/agents", {"skip": "5", "limit": "50"}, h)
        mock_fn.assert_called_once_with(skip=5, limit=50)

    @patch("aragora.server.handlers.erc8004.handle_register_agent")
    def test_route_register_with_metadata(self, mock_fn, handler):
        mock_fn.return_value = MagicMock()
        h = _make_handler(
            body={"agent_uri": "https://test.com", "metadata": {"role": "verifier"}},
            method="POST",
        )
        handler.handle("/api/v1/blockchain/agents", {}, h)
        mock_fn.assert_called_once_with(
            agent_uri="https://test.com",
            metadata={"role": "verifier"},
        )

    @patch("aragora.server.handlers.erc8004.handle_register_agent")
    def test_route_register_no_body(self, mock_fn, handler):
        """POST with no agent_uri in body should still route (handler validates)."""
        mock_fn.return_value = MagicMock()
        h = _make_handler(body={}, method="POST")
        handler.handle("/api/v1/blockchain/agents", {}, h)
        mock_fn.assert_called_once_with(agent_uri="", metadata=None)

    def test_metadata_as_list_rejected(self, handler):
        h = _make_handler(body={"agent_uri": "https://test.com", "metadata": [1, 2]}, method="POST")
        result = handler.handle("/api/v1/blockchain/agents", {}, h)
        assert _status(result) == 400
        assert "metadata must be an object" in _body(result).get("error", "")

    def test_metadata_as_int_rejected(self, handler):
        h = _make_handler(body={"agent_uri": "https://test.com", "metadata": 42}, method="POST")
        result = handler.handle("/api/v1/blockchain/agents", {}, h)
        assert _status(result) == 400

    def test_metadata_as_bool_rejected(self, handler):
        h = _make_handler(body={"agent_uri": "https://test.com", "metadata": True}, method="POST")
        result = handler.handle("/api/v1/blockchain/agents", {}, h)
        assert _status(result) == 400

    def test_get_agent_with_post_method(self, handler):
        """GET endpoint for single agent should not work with POST."""
        h = _make_handler(method="POST")
        result = handler.handle("/api/v1/blockchain/agents/1", {}, h)
        assert _status(result) == 400

    @patch("aragora.server.handlers.erc8004.handle_get_reputation")
    def test_reputation_with_empty_tags(self, mock_fn, handler):
        mock_fn.return_value = MagicMock()
        h = _make_handler(method="GET")
        handler.handle("/api/v1/blockchain/agents/1/reputation", {}, h)
        mock_fn.assert_called_once_with(1, tag1="", tag2="")

    @patch("aragora.server.handlers.erc8004.handle_get_validations")
    def test_validations_with_empty_tag(self, mock_fn, handler):
        mock_fn.return_value = MagicMock()
        h = _make_handler(method="GET")
        handler.handle("/api/v1/blockchain/agents/1/validations", {}, h)
        mock_fn.assert_called_once_with(1, tag="")

    @patch("aragora.server.handlers.erc8004.handle_blockchain_sync")
    def test_sync_with_agent_ids(self, mock_fn, handler):
        mock_fn.return_value = MagicMock()
        h = _make_handler(
            body={"agent_ids": [10, 20, 30]},
            method="POST",
        )
        handler.handle("/api/v1/blockchain/sync", {}, h)
        mock_fn.assert_called_once_with(
            sync_identities=True,
            sync_reputation=True,
            sync_validations=True,
            agent_ids=[10, 20, 30],
        )


# ============================================================================
# Additional coerce metadata edge cases
# ============================================================================


class TestCoerceMetadataValueEdgeCases:
    """Additional edge cases for _coerce_metadata_value."""

    def test_large_int(self):
        result = _coerce_metadata_value(10**18)
        assert result == str(10**18).encode("utf-8")

    def test_negative_float(self):
        result = _coerce_metadata_value(-3.14)
        assert result == b"-3.14"

    def test_long_string(self):
        s = "a" * 10000
        result = _coerce_metadata_value(s)
        assert len(result) == 10000

    def test_invalid_type_tuple(self):
        with pytest.raises(ValueError, match="must be bytes, str, int, float, or bool"):
            _coerce_metadata_value((1, 2))

    def test_invalid_type_set(self):
        with pytest.raises(ValueError, match="must be bytes, str, int, float, or bool"):
            _coerce_metadata_value({1, 2})

    def test_bytes_with_null(self):
        result = _coerce_metadata_value(b"\x00\x01\x02")
        assert result == b"\x00\x01\x02"


# ============================================================================
# Additional serialize identity edge cases
# ============================================================================


class TestSerializeIdentityEdgeCases:
    """Additional edge cases for _serialize_identity."""

    def test_serialize_with_special_chars(self):
        identity = _mock_identity(
            agent_uri="https://agent.example.com/path?q=1&r=2",
            owner="0x" + "f" * 40,
        )
        result = _serialize_identity(identity)
        assert "?" in result["agent_uri"]
        assert result["owner"] == "0x" + "f" * 40

    def test_serialize_with_empty_strings(self):
        identity = _mock_identity(
            owner="",
            agent_uri="",
            wallet_address="",
            aragora_agent_id="",
            tx_hash="",
        )
        result = _serialize_identity(identity)
        assert result["owner"] == ""
        assert result["agent_uri"] == ""


# ============================================================================
# Additional query param edge cases
# ============================================================================


class TestQueryParamEdgeCases:
    """Additional edge cases for query param helpers."""

    def test_get_query_param_int_value(self):
        """Non-string value should be returned as-is."""
        assert _get_query_param({"key": 42}, "key") == 42

    def test_get_int_query_param_list_value(self):
        """List value should extract first element."""
        assert _get_int_query_param({"skip": ["5", "10"]}, "skip", 0) == 5

    def test_get_int_query_param_large_number(self):
        assert _get_int_query_param({"val": "999999"}, "val", 0) == 999999

    def test_get_int_query_param_float_string_raises(self):
        with pytest.raises(ValueError, match="must be an integer"):
            _get_int_query_param({"val": "3.14"}, "val", 0)
