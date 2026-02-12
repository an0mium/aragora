"""
Tests for ERC8004Handler - Blockchain HTTP endpoints.

Tests cover:
- Configuration endpoint
- Agent identity retrieval
- Reputation endpoints
- Validation endpoints
- Blockchain sync operations
- Health status
- Circuit breaker behavior
- Rate limiting
- Error handling
- Path routing
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers import erc8004


# =============================================================================
# Mock Classes
# =============================================================================


@dataclass
class MockChainConfig:
    """Mock blockchain configuration."""

    chain_id: int = 1
    rpc_url: str = "https://mainnet.infura.io/v3/xxx"
    identity_registry_address: str | None = "0x1234567890abcdef"
    reputation_registry_address: str | None = "0xabcdef1234567890"
    validation_registry_address: str | None = "0x9876543210fedcba"
    block_confirmations: int = 12


@dataclass
class MockAgentIdentity:
    """Mock on-chain agent identity."""

    token_id: int = 1
    owner: str = "0xOwnerAddress"
    agent_uri: str = "https://example.com/agent.json"
    wallet_address: str | None = "0xWalletAddress"
    chain_id: int = 1
    aragora_agent_id: str | None = "claude"
    registered_at: datetime | None = None
    tx_hash: str = "0xTxHash"

    def __post_init__(self):
        if self.registered_at is None:
            self.registered_at = datetime.now(timezone.utc)


@dataclass
class MockReputationSummary:
    """Mock reputation summary."""

    agent_id: int = 1
    count: int = 10
    summary_value: int = 850
    summary_value_decimals: int = 2
    normalized_value: float = 0.85
    tag1: str = "aragora_elo"
    tag2: str = ""


@dataclass
class MockValidationSummary:
    """Mock validation summary."""

    agent_id: int = 1
    count: int = 5
    average_response: float = 0.92
    tag: str = "validated_claims"


@dataclass
class MockSyncResult:
    """Mock sync result."""

    records_synced: int = 10
    records_skipped: int = 2
    records_failed: int = 1
    duration_ms: int = 1500
    errors: list[str] | None = None


@dataclass
class MockHealthResult:
    """Mock health check result."""

    status: str = "healthy"
    latency_ms: int = 50

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "latency_ms": self.latency_ms}


class MockProvider:
    """Mock Web3 provider."""

    def __init__(self, config: MockChainConfig | None = None, connected: bool = True):
        self._config = config or MockChainConfig()
        self._connected = connected

    def get_config(self) -> MockChainConfig:
        return self._config

    def is_connected(self) -> bool:
        return self._connected

    def get_health_status(self) -> dict[str, Any]:
        return {"rpc_status": "healthy", "latency_ms": 50}


class MockCircuitBreaker:
    """Mock circuit breaker."""

    def __init__(self, can_exec: bool = True):
        self._can_execute = can_exec

    def can_execute(self) -> bool:
        return self._can_execute


class MockHandler:
    """Mock HTTP request handler."""

    def __init__(self, command: str = "GET", body: dict | None = None):
        self.command = command
        self._body = body or {}
        self.headers = {"Content-Length": str(len(json.dumps(self._body)))}

    class rfile:
        pass


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_provider():
    """Create mock provider."""
    return MockProvider()


@pytest.fixture
def mock_circuit_breaker():
    """Create mock circuit breaker."""
    return MockCircuitBreaker(can_exec=True)


@pytest.fixture
def mock_handler():
    """Create mock HTTP handler."""
    return MockHandler()


@pytest.fixture
def handler_instance():
    """Create handler instance."""
    return erc8004.ERC8004Handler({})


# =============================================================================
# Test handle_blockchain_config
# =============================================================================


class TestBlockchainConfig:
    """Tests for blockchain configuration endpoint."""

    @pytest.mark.asyncio
    async def test_config_success(self, mock_provider, mock_circuit_breaker):
        """Should return blockchain configuration."""
        with (
            patch.object(erc8004, "_get_provider", return_value=mock_provider),
            patch.object(erc8004, "_get_circuit_breaker", return_value=mock_circuit_breaker),
        ):
            result = await erc8004.handle_blockchain_config()

            assert result["status"] == 200
            data = json.loads(result["body"])
            assert data["chain_id"] == 1
            assert data["is_connected"] is True
            assert "health" in data

    @pytest.mark.asyncio
    async def test_config_circuit_breaker_open(self):
        """Should return 503 when circuit breaker is open."""
        cb = MockCircuitBreaker(can_exec=False)
        with patch.object(erc8004, "_get_circuit_breaker", return_value=cb):
            result = await erc8004.handle_blockchain_config()

            assert result["status"] == 503
            data = json.loads(result["body"])
            assert "unavailable" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_config_import_error(self, mock_circuit_breaker):
        """Should return 501 when blockchain module not installed."""
        with (
            patch.object(erc8004, "_get_circuit_breaker", return_value=mock_circuit_breaker),
            patch.object(erc8004, "_get_provider", side_effect=ImportError("web3 not installed")),
        ):
            result = await erc8004.handle_blockchain_config()

            assert result["status"] == 501
            data = json.loads(result["body"])
            assert "web3" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_config_provider_error(self, mock_circuit_breaker):
        """Should return 500 on provider error."""
        with (
            patch.object(erc8004, "_get_circuit_breaker", return_value=mock_circuit_breaker),
            patch.object(erc8004, "_get_provider", side_effect=Exception("Connection failed")),
        ):
            result = await erc8004.handle_blockchain_config()

            assert result["status"] == 500

    @pytest.mark.asyncio
    async def test_config_truncates_long_rpc_url(self, mock_circuit_breaker):
        """Should truncate long RPC URLs for security."""
        long_url = "https://mainnet.infura.io/v3/" + "x" * 100
        config = MockChainConfig(rpc_url=long_url)
        provider = MockProvider(config=config)

        with (
            patch.object(erc8004, "_get_provider", return_value=provider),
            patch.object(erc8004, "_get_circuit_breaker", return_value=mock_circuit_breaker),
        ):
            result = await erc8004.handle_blockchain_config()

            assert result["status"] == 200
            data = json.loads(result["body"])
            assert len(data["rpc_url"]) <= 53  # 50 + "..."


# =============================================================================
# Test handle_get_agent
# =============================================================================


class TestGetAgent:
    """Tests for agent identity endpoint."""

    @pytest.mark.asyncio
    async def test_get_agent_success(self, mock_provider, mock_circuit_breaker):
        """Should return agent identity."""
        mock_contract = MagicMock()
        mock_contract.get_agent.return_value = MockAgentIdentity(token_id=42)

        with (
            patch.object(erc8004, "_get_provider", return_value=mock_provider),
            patch.object(erc8004, "_get_circuit_breaker", return_value=mock_circuit_breaker),
            patch(
                "aragora.blockchain.contracts.identity.IdentityRegistryContract",
                return_value=mock_contract,
            ),
        ):
            result = await erc8004.handle_get_agent(42)

            assert result["status"] == 200
            data = json.loads(result["body"])
            assert data["token_id"] == 42

    @pytest.mark.asyncio
    async def test_get_agent_not_found(self, mock_provider, mock_circuit_breaker):
        """Should return 404 for non-existent agent."""
        mock_contract = MagicMock()
        mock_contract.get_agent.side_effect = Exception("Agent not found")

        with (
            patch.object(erc8004, "_get_provider", return_value=mock_provider),
            patch.object(erc8004, "_get_circuit_breaker", return_value=mock_circuit_breaker),
            patch(
                "aragora.blockchain.contracts.identity.IdentityRegistryContract",
                return_value=mock_contract,
            ),
        ):
            result = await erc8004.handle_get_agent(999)

            assert result["status"] == 404

    @pytest.mark.asyncio
    async def test_get_agent_circuit_breaker_open(self):
        """Should return 503 when circuit breaker is open."""
        cb = MockCircuitBreaker(can_exec=False)
        with patch.object(erc8004, "_get_circuit_breaker", return_value=cb):
            result = await erc8004.handle_get_agent(1)

            assert result["status"] == 503


# =============================================================================
# Test handle_get_reputation
# =============================================================================


class TestGetReputation:
    """Tests for agent reputation endpoint."""

    @pytest.mark.asyncio
    async def test_get_reputation_success(self, mock_provider, mock_circuit_breaker):
        """Should return reputation summary."""
        mock_contract = MagicMock()
        mock_contract.get_summary.return_value = MockReputationSummary()

        with (
            patch.object(erc8004, "_get_provider", return_value=mock_provider),
            patch.object(erc8004, "_get_circuit_breaker", return_value=mock_circuit_breaker),
            patch(
                "aragora.blockchain.contracts.reputation.ReputationRegistryContract",
                return_value=mock_contract,
            ),
        ):
            result = await erc8004.handle_get_reputation(1)

            assert result["status"] == 200
            data = json.loads(result["body"])
            assert data["normalized_value"] == 0.85

    @pytest.mark.asyncio
    async def test_get_reputation_with_tags(self, mock_provider, mock_circuit_breaker):
        """Should pass tag filters to contract."""
        mock_contract = MagicMock()
        mock_contract.get_summary.return_value = MockReputationSummary(
            tag1="security", tag2="coding"
        )

        with (
            patch.object(erc8004, "_get_provider", return_value=mock_provider),
            patch.object(erc8004, "_get_circuit_breaker", return_value=mock_circuit_breaker),
            patch(
                "aragora.blockchain.contracts.reputation.ReputationRegistryContract",
                return_value=mock_contract,
            ),
        ):
            result = await erc8004.handle_get_reputation(1, tag1="security", tag2="coding")

            assert result["status"] == 200
            mock_contract.get_summary.assert_called_once_with(1, tag1="security", tag2="coding")

    @pytest.mark.asyncio
    async def test_get_reputation_not_found(self, mock_provider, mock_circuit_breaker):
        """Should return 404 when reputation not found."""
        mock_contract = MagicMock()
        mock_contract.get_summary.side_effect = Exception("No reputation data")

        with (
            patch.object(erc8004, "_get_provider", return_value=mock_provider),
            patch.object(erc8004, "_get_circuit_breaker", return_value=mock_circuit_breaker),
            patch(
                "aragora.blockchain.contracts.reputation.ReputationRegistryContract",
                return_value=mock_contract,
            ),
        ):
            result = await erc8004.handle_get_reputation(999)

            assert result["status"] == 404


# =============================================================================
# Test handle_get_validations
# =============================================================================


class TestGetValidations:
    """Tests for agent validations endpoint."""

    @pytest.mark.asyncio
    async def test_get_validations_success(self, mock_provider, mock_circuit_breaker):
        """Should return validation summary."""
        mock_contract = MagicMock()
        mock_contract.get_summary.return_value = MockValidationSummary()

        with (
            patch.object(erc8004, "_get_provider", return_value=mock_provider),
            patch.object(erc8004, "_get_circuit_breaker", return_value=mock_circuit_breaker),
            patch(
                "aragora.blockchain.contracts.validation.ValidationRegistryContract",
                return_value=mock_contract,
            ),
        ):
            result = await erc8004.handle_get_validations(1)

            assert result["status"] == 200
            data = json.loads(result["body"])
            assert data["average_response"] == 0.92

    @pytest.mark.asyncio
    async def test_get_validations_with_tag(self, mock_provider, mock_circuit_breaker):
        """Should pass tag filter to contract."""
        mock_contract = MagicMock()
        mock_contract.get_summary.return_value = MockValidationSummary(tag="receipts")

        with (
            patch.object(erc8004, "_get_provider", return_value=mock_provider),
            patch.object(erc8004, "_get_circuit_breaker", return_value=mock_circuit_breaker),
            patch(
                "aragora.blockchain.contracts.validation.ValidationRegistryContract",
                return_value=mock_contract,
            ),
        ):
            result = await erc8004.handle_get_validations(1, tag="receipts")

            assert result["status"] == 200
            mock_contract.get_summary.assert_called_once_with(1, tag="receipts")


# =============================================================================
# Test handle_blockchain_sync
# =============================================================================


class TestBlockchainSync:
    """Tests for blockchain sync endpoint."""

    @pytest.mark.asyncio
    async def test_sync_success(self, mock_circuit_breaker):
        """Should trigger sync successfully."""
        mock_adapter = MagicMock()
        mock_adapter.sync_to_km = AsyncMock(return_value=MockSyncResult())

        with (
            patch.object(erc8004, "_get_adapter", return_value=mock_adapter),
            patch.object(erc8004, "_get_circuit_breaker", return_value=mock_circuit_breaker),
        ):
            result = await erc8004.handle_blockchain_sync()

            assert result["status"] == 200
            data = json.loads(result["body"])
            assert data["records_synced"] == 10

    @pytest.mark.asyncio
    async def test_sync_with_specific_agents(self, mock_circuit_breaker):
        """Should sync specific agents."""
        mock_adapter = MagicMock()
        mock_adapter.sync_to_km = AsyncMock(return_value=MockSyncResult())

        with (
            patch.object(erc8004, "_get_adapter", return_value=mock_adapter),
            patch.object(erc8004, "_get_circuit_breaker", return_value=mock_circuit_breaker),
        ):
            result = await erc8004.handle_blockchain_sync(agent_ids=[1, 2, 3])

            assert result["status"] == 200
            mock_adapter.sync_to_km.assert_called_once()
            call_kwargs = mock_adapter.sync_to_km.call_args.kwargs
            assert call_kwargs["agent_ids"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_sync_selective_types(self, mock_circuit_breaker):
        """Should sync selective data types."""
        mock_adapter = MagicMock()
        mock_adapter.sync_to_km = AsyncMock(return_value=MockSyncResult())

        with (
            patch.object(erc8004, "_get_adapter", return_value=mock_adapter),
            patch.object(erc8004, "_get_circuit_breaker", return_value=mock_circuit_breaker),
        ):
            result = await erc8004.handle_blockchain_sync(
                sync_identities=True, sync_reputation=False, sync_validations=False
            )

            assert result["status"] == 200
            call_kwargs = mock_adapter.sync_to_km.call_args.kwargs
            assert call_kwargs["sync_identities"] is True
            assert call_kwargs["sync_reputation"] is False

    @pytest.mark.asyncio
    async def test_sync_error_truncates_errors(self, mock_circuit_breaker):
        """Should truncate error list in response."""
        errors = [f"Error {i}" for i in range(20)]
        mock_adapter = MagicMock()
        mock_adapter.sync_to_km = AsyncMock(
            return_value=MockSyncResult(records_failed=20, errors=errors)
        )

        with (
            patch.object(erc8004, "_get_adapter", return_value=mock_adapter),
            patch.object(erc8004, "_get_circuit_breaker", return_value=mock_circuit_breaker),
        ):
            result = await erc8004.handle_blockchain_sync()

            data = json.loads(result["body"])
            assert len(data["errors"]) <= 10


# =============================================================================
# Test handle_blockchain_health
# =============================================================================


class TestBlockchainHealth:
    """Tests for blockchain health endpoint."""

    @pytest.mark.asyncio
    async def test_health_success(self):
        """Should return health status."""
        mock_connector = MagicMock()
        mock_connector.health_check = AsyncMock(return_value=MockHealthResult())

        mock_adapter = MagicMock()
        mock_adapter.get_health_status.return_value = {"status": "healthy"}

        with (
            patch.object(erc8004, "_get_connector", return_value=mock_connector),
            patch.object(erc8004, "_get_adapter", return_value=mock_adapter),
        ):
            result = await erc8004.handle_blockchain_health()

            assert result["status"] == 200
            data = json.loads(result["body"])
            assert "connector" in data
            assert "adapter" in data

    @pytest.mark.asyncio
    async def test_health_import_error(self):
        """Should return available=False when not installed."""
        with patch.object(erc8004, "_get_connector", side_effect=ImportError("web3 not installed")):
            result = await erc8004.handle_blockchain_health()

            assert result["status"] == 200
            data = json.loads(result["body"])
            assert data["available"] is False

    @pytest.mark.asyncio
    async def test_health_adapter_error(self):
        """Should include adapter error in response."""
        mock_connector = MagicMock()
        mock_connector.health_check = AsyncMock(return_value=MockHealthResult())

        with (
            patch.object(erc8004, "_get_connector", return_value=mock_connector),
            patch.object(erc8004, "_get_adapter", side_effect=Exception("Adapter error")),
        ):
            result = await erc8004.handle_blockchain_health()

            assert result["status"] == 200
            data = json.loads(result["body"])
            assert "error" in data["adapter"]


# =============================================================================
# Test handle_list_agents / handle_register_agent
# =============================================================================


class TestListAndRegisterAgents:
    """Tests for blockchain agent list/register endpoints."""

    @pytest.mark.asyncio
    async def test_list_agents_success(self, mock_provider, mock_circuit_breaker):
        """Should list agents with pagination."""
        mock_contract = MagicMock()
        mock_contract.get_total_supply.return_value = 3
        mock_contract.get_agent.side_effect = [
            MockAgentIdentity(token_id=1),
            MockAgentIdentity(token_id=2),
            MockAgentIdentity(token_id=3),
        ]

        with (
            patch.object(erc8004, "_get_provider", return_value=mock_provider),
            patch.object(erc8004, "_get_circuit_breaker", return_value=mock_circuit_breaker),
            patch(
                "aragora.blockchain.contracts.identity.IdentityRegistryContract",
                return_value=mock_contract,
            ),
        ):
            result = await erc8004.handle_list_agents(skip=0, limit=2)

        assert result["status"] == 200
        data = json.loads(result["body"])
        assert data["total"] == 3
        assert data["count"] == 2
        assert data["agents"][0]["token_id"] == 1
        assert data["agents"][1]["token_id"] == 2

    @pytest.mark.asyncio
    async def test_list_agents_requires_identity_registry(self, mock_circuit_breaker):
        """Should return 501 when identity registry is not configured."""
        provider = MockProvider(config=MockChainConfig(identity_registry_address=None))

        with (
            patch.object(erc8004, "_get_provider", return_value=provider),
            patch.object(erc8004, "_get_circuit_breaker", return_value=mock_circuit_breaker),
        ):
            result = await erc8004.handle_list_agents()

        assert result["status"] == 501

    @pytest.mark.asyncio
    async def test_register_agent_success(self, mock_provider, mock_circuit_breaker):
        """Should register an agent and return 201."""
        mock_contract = MagicMock()
        mock_contract.register_agent.return_value = 42
        mock_signer = MagicMock()
        mock_signer.address = "0xSignerAddress"

        with (
            patch.object(erc8004, "_get_provider", return_value=mock_provider),
            patch.object(erc8004, "_get_circuit_breaker", return_value=mock_circuit_breaker),
            patch(
                "aragora.blockchain.contracts.identity.IdentityRegistryContract",
                return_value=mock_contract,
            ),
            patch("aragora.blockchain.wallet.WalletSigner.from_env", return_value=mock_signer),
        ):
            result = await erc8004.handle_register_agent(
                agent_uri="ipfs://QmAgent",
                metadata={"role": "critic", "score": 99},
            )

        assert result["status"] == 201
        data = json.loads(result["body"])
        assert data["token_id"] == 42
        assert data["owner"] == "0xSignerAddress"
        call_args = mock_contract.register_agent.call_args
        assert call_args is not None
        metadata_entries = call_args.args[2]
        assert metadata_entries[0].key == "role"
        assert metadata_entries[0].value == b"critic"

    @pytest.mark.asyncio
    async def test_register_agent_requires_wallet_credentials(
        self,
        mock_provider,
        mock_circuit_breaker,
    ):
        """Should return 400 when wallet credentials are missing."""
        mock_contract = MagicMock()

        with (
            patch.object(erc8004, "_get_provider", return_value=mock_provider),
            patch.object(erc8004, "_get_circuit_breaker", return_value=mock_circuit_breaker),
            patch(
                "aragora.blockchain.contracts.identity.IdentityRegistryContract",
                return_value=mock_contract,
            ),
            patch(
                "aragora.blockchain.wallet.WalletSigner.from_env",
                side_effect=ValueError("No wallet credentials configured"),
            ),
        ):
            result = await erc8004.handle_register_agent(agent_uri="ipfs://QmAgent")

        assert result["status"] == 400


# =============================================================================
# Test ERC8004Handler class routing
# =============================================================================


class TestERC8004HandlerRouting:
    """Tests for handler class path routing."""

    def test_can_handle_blockchain_paths(self, handler_instance):
        """Should handle blockchain paths."""
        assert handler_instance.can_handle("/api/v1/blockchain/config") is True
        assert handler_instance.can_handle("/api/v1/blockchain/agents") is True
        assert handler_instance.can_handle("/api/v1/blockchain/agents/1") is True
        assert handler_instance.can_handle("/api/v1/blockchain/health") is True

    def test_cannot_handle_non_blockchain_paths(self, handler_instance):
        """Should not handle non-blockchain paths."""
        assert handler_instance.can_handle("/api/v1/debates") is False
        assert handler_instance.can_handle("/api/v1/agents") is False

    def test_handle_config_route(self, handler_instance, mock_handler):
        """Should route config path correctly."""
        with patch.object(erc8004, "handle_blockchain_config") as mock_fn:
            mock_fn.return_value = {"status": 200, "body": "{}"}
            handler_instance.handle("/api/v1/blockchain/config", {}, mock_handler)
            mock_fn.assert_called_once()

    def test_handle_health_route(self, handler_instance, mock_handler):
        """Should route health path correctly."""
        with patch.object(erc8004, "handle_blockchain_health") as mock_fn:
            mock_fn.return_value = {"status": 200, "body": "{}"}
            handler_instance.handle("/api/v1/blockchain/health", {}, mock_handler)
            mock_fn.assert_called_once()

    def test_handle_agent_route(self, handler_instance, mock_handler):
        """Should route agent path correctly."""
        with patch.object(erc8004, "handle_get_agent") as mock_fn:
            mock_fn.return_value = {"status": 200, "body": "{}"}
            handler_instance.handle("/api/v1/blockchain/agents/42", {}, mock_handler)
            mock_fn.assert_called_once_with(42)

    def test_handle_reputation_route(self, handler_instance, mock_handler):
        """Should route reputation path correctly."""
        with patch.object(erc8004, "handle_get_reputation") as mock_fn:
            mock_fn.return_value = {"status": 200, "body": "{}"}
            handler_instance.handle(
                "/api/v1/blockchain/agents/42/reputation",
                {"tag1": ["security"], "tag2": ["coding"]},
                mock_handler,
            )
            mock_fn.assert_called_once_with(42, tag1="security", tag2="coding")

    def test_handle_validations_route(self, handler_instance, mock_handler):
        """Should route validations path correctly."""
        with patch.object(erc8004, "handle_get_validations") as mock_fn:
            mock_fn.return_value = {"status": 200, "body": "{}"}
            handler_instance.handle(
                "/api/v1/blockchain/agents/42/validations",
                {"tag": ["receipts"]},
                mock_handler,
            )
            mock_fn.assert_called_once_with(42, tag="receipts")

    def test_handle_invalid_token_id(self, handler_instance, mock_handler):
        """Should return 400 for invalid token ID."""
        result = handler_instance.handle("/api/v1/blockchain/agents/invalid", {}, mock_handler)
        assert result["status"] == 400

    def test_handle_invalid_path(self, handler_instance, mock_handler):
        """Should return 400 for invalid blockchain path."""
        result = handler_instance.handle("/api/v1/blockchain/unknown", {}, mock_handler)
        assert result["status"] == 400

    def test_handle_agents_listing_route(self, handler_instance, mock_handler):
        """Should route agent listing path correctly."""
        with patch.object(erc8004, "handle_list_agents") as mock_fn:
            mock_fn.return_value = {"status": 200, "body": "{}"}
            handler_instance.handle(
                "/api/v1/blockchain/agents",
                {"skip": ["2"], "limit": ["5"]},
                mock_handler,
            )
            mock_fn.assert_called_once_with(skip=2, limit=5)

    def test_handle_agents_listing_invalid_pagination(self, handler_instance, mock_handler):
        """Should return 400 for invalid pagination query params."""
        result = handler_instance.handle(
            "/api/v1/blockchain/agents",
            {"skip": ["oops"]},
            mock_handler,
        )
        assert result["status"] == 400

    def test_handle_agents_register_route(self, handler_instance):
        """Should route agent registration path correctly."""
        post_handler = MockHandler(command="POST", body={})
        with (
            patch.object(handler_instance, "read_json_body", return_value={
                "agent_uri": "ipfs://QmAgent",
                "metadata": {"role": "critic"},
            }),
            patch.object(erc8004, "handle_register_agent") as mock_fn,
        ):
            mock_fn.return_value = {"status": 201, "body": "{}"}
            handler_instance.handle("/api/v1/blockchain/agents", {}, post_handler)
            mock_fn.assert_called_once_with(
                agent_uri="ipfs://QmAgent",
                metadata={"role": "critic"},
            )


# =============================================================================
# Test Circuit Breaker Integration
# =============================================================================


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker behavior."""

    def test_get_circuit_breaker_creates_singleton(self):
        """Should create circuit breaker singleton."""
        erc8004._blockchain_circuit_breaker = None

        with patch.object(erc8004, "get_circuit_breaker") as mock_get_cb:
            mock_cb = MockCircuitBreaker()
            mock_get_cb.return_value = mock_cb

            cb1 = erc8004._get_circuit_breaker()
            cb2 = erc8004._get_circuit_breaker()

            # Second call should reuse the cached instance
            assert cb1 is cb2

        erc8004._blockchain_circuit_breaker = None


# =============================================================================
# Test Query Parameter Extraction
# =============================================================================


class TestQueryParamExtraction:
    """Tests for query parameter helper."""

    def test_get_query_param_string(self):
        """Should extract string parameter."""
        result = erc8004._get_query_param({"tag": "security"}, "tag", "")
        assert result == "security"

    def test_get_query_param_list(self):
        """Should extract first value from list."""
        result = erc8004._get_query_param({"tag": ["first", "second"]}, "tag", "")
        assert result == "first"

    def test_get_query_param_default(self):
        """Should return default when missing."""
        result = erc8004._get_query_param({}, "tag", "default")
        assert result == "default"

    def test_get_query_param_empty_list(self):
        """Should return default for empty list."""
        result = erc8004._get_query_param({"tag": []}, "tag", "default")
        assert result == "default"

    def test_get_query_param_none(self):
        """Should return default for None value."""
        result = erc8004._get_query_param({"tag": None}, "tag", "default")
        assert result == "default"

    def test_get_int_query_param(self):
        """Should parse integer query params."""
        result = erc8004._get_int_query_param({"skip": ["10"]}, "skip", 0, min_value=0)
        assert result == 10

    def test_get_int_query_param_invalid(self):
        """Should raise for non-integer query params."""
        with pytest.raises(ValueError):
            erc8004._get_int_query_param({"skip": ["abc"]}, "skip", 0, min_value=0)

    def test_get_int_query_param_below_min(self):
        """Should raise for values below minimum."""
        with pytest.raises(ValueError):
            erc8004._get_int_query_param({"skip": ["-1"]}, "skip", 0, min_value=0)
