"""
Tests for ERC-8004 API handler.

Note: These tests are for an ERC8004Handler class that hasn't been implemented yet.
The current erc8004.py uses standalone functions instead of a handler class.
"""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch
import pytest

_HAS_ERC8004_HANDLER = False
try:
    from aragora.blockchain.handler import ERC8004Handler  # noqa: F401

    _HAS_ERC8004_HANDLER = True
except (ImportError, AttributeError):
    pass

pytestmark = pytest.mark.skipif(
    not _HAS_ERC8004_HANDLER,
    reason="ERC8004Handler class not available at aragora.blockchain.handler",
)


class TestERC8004Handler:
    """Tests for ERC8004Handler class."""

    @pytest.fixture
    def mock_context(self, mock_handler_context):
        """Create mock server context."""
        return mock_handler_context

    @pytest.fixture
    def handler(self, mock_context):
        """Create handler instance."""
        return ERC8004Handler(mock_context)

    def test_handler_routes(self, handler):
        """Test handler route definitions."""
        routes = handler.get_routes()

        assert len(routes) > 0
        route_paths = [r["path"] for r in routes]

        assert any("/blockchain" in p for p in route_paths)

    def test_can_handle_blockchain_path(self, handler):
        """Test handler recognizes blockchain paths."""
        assert handler.can_handle("/api/v1/blockchain/agents") is True
        assert handler.can_handle("/api/v1/blockchain/config") is True
        assert handler.can_handle("/api/v1/debates") is False

    @pytest.mark.asyncio
    async def test_handle_get_config(self, handler):
        """Test getting blockchain configuration."""
        with patch("aragora.server.handlers.erc8004.get_chain_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                chain_id=1,
                rpc_url="https://eth.rpc",
                identity_registry_address="0x111...",
            )

            result = await handler.handle_blockchain_config({})

            assert result["chain_id"] == 1
            assert "rpc_url" in result

    @pytest.mark.asyncio
    async def test_handle_get_agent(self, handler, sample_agent_identity):
        """Test getting a specific agent."""
        with patch("aragora.server.handlers.erc8004.ERC8004Connector") as mock_connector_cls:
            mock_connector = MagicMock()
            mock_connector.fetch.return_value = MagicMock(
                id="identity:1:42",
                content=sample_agent_identity["agent_name"],
                metadata={
                    "token_id": 42,
                    "owner": sample_agent_identity["owner"],
                },
            )
            mock_connector_cls.return_value = mock_connector

            result = await handler.handle_get_agent(
                {"token_id": "42"},
                query_params={},
            )

            assert result["token_id"] == 42

    @pytest.mark.asyncio
    async def test_handle_get_agent_not_found(self, handler):
        """Test getting non-existent agent."""
        with patch("aragora.server.handlers.erc8004.ERC8004Connector") as mock_connector_cls:
            mock_connector = MagicMock()
            mock_connector.fetch.return_value = None
            mock_connector_cls.return_value = mock_connector

            with pytest.raises(Exception) as exc_info:
                await handler.handle_get_agent(
                    {"token_id": "99999"},
                    query_params={},
                )

            assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_handle_get_reputation(self, handler, sample_reputation_feedbacks):
        """Test getting agent reputation."""
        with patch("aragora.server.handlers.erc8004.ERC8004Connector") as mock_connector_cls:
            mock_connector = MagicMock()
            mock_connector.fetch.return_value = MagicMock(
                id="reputation:1:42",
                metadata={"feedbacks": sample_reputation_feedbacks},
            )
            mock_connector_cls.return_value = mock_connector

            result = await handler.handle_get_reputation(
                {"token_id": "42"},
                query_params={},
            )

            assert "feedbacks" in result or "reputation" in str(result).lower()

    @pytest.mark.asyncio
    async def test_handle_get_validations(self, handler, sample_validation_records):
        """Test getting agent validations."""
        with patch("aragora.server.handlers.erc8004.ERC8004Connector") as mock_connector_cls:
            mock_connector = MagicMock()
            mock_connector.fetch.return_value = MagicMock(
                id="validation:1:42",
                metadata={"validations": sample_validation_records},
            )
            mock_connector_cls.return_value = mock_connector

            result = await handler.handle_get_validations(
                {"token_id": "42"},
                query_params={},
            )

            assert "validations" in result or len(result) > 0

    @pytest.mark.asyncio
    async def test_handle_sync(self, handler):
        """Test triggering manual sync."""
        with patch("aragora.server.handlers.erc8004.ERC8004Adapter") as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter.sync_to_km = AsyncMock(
                return_value=MagicMock(records_synced=10, errors=[])
            )
            mock_adapter_cls.return_value = mock_adapter

            result = await handler.handle_blockchain_sync({}, query_params={})

            assert result["records_synced"] == 10

    @pytest.mark.asyncio
    async def test_handle_health(self, handler):
        """Test health endpoint."""
        with patch("aragora.server.handlers.erc8004.ERC8004Connector") as mock_connector_cls:
            mock_connector = MagicMock()
            mock_connector.health_check.return_value = {
                "healthy": True,
                "chain_id": 1,
            }
            mock_connector_cls.return_value = mock_connector

            result = await handler.handle_blockchain_health({})

            assert result["healthy"] is True


class TestERC8004HandlerListAgents:
    """Tests for listing agents endpoint."""

    @pytest.fixture
    def handler(self, mock_handler_context):
        """Create handler instance."""
        return ERC8004Handler(mock_handler_context)

    @pytest.mark.asyncio
    async def test_list_agents(self, handler, sample_agent_identities):
        """Test listing all agents."""
        with patch("aragora.server.handlers.erc8004.ERC8004Connector") as mock_connector_cls:
            mock_connector = MagicMock()
            mock_connector.search = AsyncMock(return_value=[
                MagicMock(
                    id=f"identity:1:{a['token_id']}",
                    title=a["agent_name"],
                    metadata={"token_id": a["token_id"]},
                )
                for a in sample_agent_identities
            ])
            mock_connector_cls.return_value = mock_connector

            result = await handler.handle_list_agents({}, query_params={"limit": "10"})

            assert "agents" in result
            assert len(result["agents"]) == 3

    @pytest.mark.asyncio
    async def test_list_agents_with_owner_filter(self, handler, sample_agent_identities):
        """Test listing agents filtered by owner."""
        owner = sample_agent_identities[0]["owner"]

        with patch("aragora.server.handlers.erc8004.ERC8004Connector") as mock_connector_cls:
            mock_connector = MagicMock()
            owned = [a for a in sample_agent_identities if a["owner"] == owner]
            mock_connector.search_by_owner = AsyncMock(return_value=[
                MagicMock(
                    id=f"identity:1:{a['token_id']}",
                    title=a["agent_name"],
                )
                for a in owned
            ])
            mock_connector_cls.return_value = mock_connector

            result = await handler.handle_list_agents(
                {},
                query_params={"owner": owner},
            )

            assert "agents" in result
            assert len(result["agents"]) == 2


class TestERC8004HandlerValidation:
    """Tests for request validation in handler."""

    @pytest.fixture
    def handler(self, mock_handler_context):
        """Create handler instance."""
        return ERC8004Handler(mock_handler_context)

    @pytest.mark.asyncio
    async def test_invalid_token_id(self, handler):
        """Test error with invalid token ID."""
        with pytest.raises(ValueError, match="[Ii]nvalid"):
            await handler.handle_get_agent(
                {"token_id": "not_a_number"},
                query_params={},
            )

    @pytest.mark.asyncio
    async def test_invalid_address(self, handler):
        """Test error with invalid Ethereum address."""
        with pytest.raises(ValueError, match="[Ii]nvalid"):
            await handler.handle_list_agents(
                {},
                query_params={"owner": "not_an_address"},
            )
