"""
Tests for BlockchainIdentityBridge.
"""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from aragora.control_plane.blockchain_identity import BlockchainIdentityBridge


class TestBlockchainIdentityBridge:
    """Tests for BlockchainIdentityBridge class."""

    @pytest.fixture
    def mock_registry(self, mock_agent_registry):
        """Create mock agent registry."""
        return mock_agent_registry

    @pytest.fixture
    def mock_connector(self):
        """Create mock ERC8004Connector."""
        with patch("aragora.control_plane.blockchain_identity.ERC8004Connector") as mock_cls:
            mock_connector = MagicMock()
            mock_cls.return_value = mock_connector
            yield mock_connector

    @pytest.fixture
    def bridge(self, mock_registry, mock_connector):
        """Create bridge instance."""
        return BlockchainIdentityBridge(registry=mock_registry)

    def test_create_bridge(self, bridge):
        """Test creating bridge instance."""
        assert bridge is not None

    @pytest.mark.asyncio
    async def test_link_agent(self, bridge, mock_registry, mock_connector, sample_agent_identity):
        """Test linking an Aragora agent to blockchain identity."""
        # Mock the connector fetch
        mock_connector.fetch.return_value = MagicMock(
            id="identity:1:42",
            metadata={
                "token_id": 42,
                "owner": sample_agent_identity["owner"],
            },
        )

        result = await bridge.link_agent(
            aragora_agent_id="agent_abc123",
            blockchain_token_id=42,
        )

        assert result["linked"] is True
        mock_registry.get_agent.assert_called()

    @pytest.mark.asyncio
    async def test_link_agent_not_found(self, bridge, mock_registry, mock_connector):
        """Test linking when agent not found in registry."""
        mock_registry.get_agent.return_value = None

        with pytest.raises(ValueError, match="not found"):
            await bridge.link_agent(
                aragora_agent_id="nonexistent_agent",
                blockchain_token_id=42,
            )

    @pytest.mark.asyncio
    async def test_link_agent_token_not_found(self, bridge, mock_registry, mock_connector):
        """Test linking when blockchain token not found."""
        mock_registry.get_agent.return_value = MagicMock(id="agent_123")
        mock_connector.fetch.return_value = None

        with pytest.raises(ValueError, match="not found"):
            await bridge.link_agent(
                aragora_agent_id="agent_123",
                blockchain_token_id=99999,
            )

    @pytest.mark.asyncio
    async def test_get_blockchain_identity(
        self, bridge, mock_registry, mock_connector, sample_agent_identity
    ):
        """Test getting blockchain identity for an Aragora agent."""
        mock_registry.get_agent.return_value = MagicMock(
            id="agent_123",
            blockchain_agent_id=42,
        )
        mock_connector.fetch.return_value = MagicMock(
            id="identity:1:42",
            metadata={
                "token_id": 42,
                "owner": sample_agent_identity["owner"],
                "metadata_uri": sample_agent_identity["metadata_uri"],
            },
        )

        identity = await bridge.get_blockchain_identity(aragora_agent_id="agent_123")

        assert identity is not None
        assert identity["token_id"] == 42

    @pytest.mark.asyncio
    async def test_get_blockchain_identity_not_linked(self, bridge, mock_registry):
        """Test getting identity for unlinked agent."""
        mock_registry.get_agent.return_value = MagicMock(
            id="agent_123",
            blockchain_agent_id=None,  # Not linked
        )

        identity = await bridge.get_blockchain_identity(aragora_agent_id="agent_123")

        assert identity is None

    @pytest.mark.asyncio
    async def test_get_agent_by_token(self, bridge, mock_registry, mock_connector):
        """Test getting Aragora agent by blockchain token ID."""
        mock_registry.list_agents.return_value = [
            MagicMock(id="agent_1", blockchain_agent_id=None),
            MagicMock(id="agent_2", blockchain_agent_id=42),
            MagicMock(id="agent_3", blockchain_agent_id=100),
        ]

        agent = await bridge.get_agent_by_token(token_id=42)

        assert agent is not None
        assert agent.id == "agent_2"

    @pytest.mark.asyncio
    async def test_get_agent_by_token_not_found(self, bridge, mock_registry):
        """Test getting agent when token not linked."""
        mock_registry.list_agents.return_value = [
            MagicMock(id="agent_1", blockchain_agent_id=None),
            MagicMock(id="agent_2", blockchain_agent_id=100),
        ]

        agent = await bridge.get_agent_by_token(token_id=42)

        assert agent is None


class TestBlockchainIdentityBridgeSync:
    """Tests for bridge sync operations."""

    @pytest.fixture
    def mock_registry(self, mock_agent_registry):
        """Create mock agent registry."""
        return mock_agent_registry

    @pytest.fixture
    def mock_connector(self):
        """Create mock connector."""
        with patch("aragora.control_plane.blockchain_identity.ERC8004Connector") as mock_cls:
            mock_connector = MagicMock()
            mock_cls.return_value = mock_connector
            yield mock_connector

    @pytest.fixture
    def bridge(self, mock_registry, mock_connector):
        """Create bridge instance."""
        return BlockchainIdentityBridge(registry=mock_registry)

    @pytest.mark.asyncio
    async def test_sync_reputation_to_chain(self, bridge, mock_registry, mock_connector):
        """Test syncing ELO to blockchain reputation."""
        mock_registry.get_agent.return_value = MagicMock(
            id="agent_123",
            blockchain_agent_id=42,
            elo_rating=1650,
        )

        # Mock the reputation contract
        mock_reputation = MagicMock()
        mock_reputation.give_feedback = MagicMock(return_value="0xtxhash")
        bridge._reputation_contract = mock_reputation

        result = await bridge.sync_reputation_to_chain(
            aragora_agent_id="agent_123",
            elo_rating=1650,
        )

        assert result["synced"] is True

    @pytest.mark.asyncio
    async def test_sync_validation_to_chain(self, bridge, mock_registry, mock_connector):
        """Test syncing Gauntlet receipt to blockchain validation."""
        mock_registry.get_agent.return_value = MagicMock(
            id="agent_123",
            blockchain_agent_id=42,
        )

        # Mock the validation contract
        mock_validation = MagicMock()
        mock_validation.submit_response = MagicMock(return_value="0xtxhash")
        bridge._validation_contract = mock_validation

        result = await bridge.sync_validation_to_chain(
            aragora_agent_id="agent_123",
            validation_type="capability",
            passed=True,
            evidence_uri="ipfs://QmEvidence",
        )

        assert result["synced"] is True

    @pytest.mark.asyncio
    async def test_sync_from_chain(
        self, bridge, mock_registry, mock_connector, sample_agent_identity
    ):
        """Test syncing blockchain identity to Aragora registry."""
        mock_connector.fetch.return_value = MagicMock(
            id="identity:1:42",
            metadata={
                "token_id": 42,
                "owner": sample_agent_identity["owner"],
                "agent_name": sample_agent_identity["agent_name"],
            },
        )

        result = await bridge.sync_from_chain(token_id=42)

        assert result is not None
        mock_registry.register_agent.assert_called()


class TestBlockchainIdentityBridgeBatch:
    """Tests for batch bridge operations."""

    @pytest.fixture
    def mock_registry(self, mock_agent_registry):
        """Create mock registry."""
        return mock_agent_registry

    @pytest.fixture
    def mock_connector(self):
        """Create mock connector."""
        with patch("aragora.control_plane.blockchain_identity.ERC8004Connector") as mock_cls:
            mock_connector = MagicMock()
            mock_cls.return_value = mock_connector
            yield mock_connector

    @pytest.fixture
    def bridge(self, mock_registry, mock_connector):
        """Create bridge instance."""
        return BlockchainIdentityBridge(registry=mock_registry)

    @pytest.mark.asyncio
    async def test_batch_sync_from_chain(self, bridge, mock_connector, sample_agent_identities):
        """Test batch syncing from blockchain."""
        mock_connector.search.return_value = [
            MagicMock(
                id=f"identity:1:{a['token_id']}",
                metadata={"token_id": a["token_id"], "agent_name": a["agent_name"]},
            )
            for a in sample_agent_identities
        ]

        results = await bridge.batch_sync_from_chain(token_ids=[1, 2, 3])

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_get_linked_agents(self, bridge, mock_registry):
        """Test getting all linked agents."""
        mock_registry.list_agents.return_value = [
            MagicMock(id="agent_1", blockchain_agent_id=42),
            MagicMock(id="agent_2", blockchain_agent_id=None),
            MagicMock(id="agent_3", blockchain_agent_id=100),
        ]

        linked = await bridge.get_linked_agents()

        assert len(linked) == 2
        assert all(a.blockchain_agent_id is not None for a in linked)
