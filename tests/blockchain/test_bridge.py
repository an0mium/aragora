"""
Tests for BlockchainIdentityBridge.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from aragora.control_plane.blockchain_identity import (
    BlockchainIdentityBridge,
    AgentBlockchainLink,
)


class TestBlockchainIdentityBridge:
    """Tests for BlockchainIdentityBridge class."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock Web3Provider."""
        provider = MagicMock()
        config = MagicMock()
        config.chain_id = 1
        provider.get_config.return_value = config
        return provider

    @pytest.fixture
    def mock_identity_contract(self):
        """Create mock IdentityRegistryContract."""
        contract = MagicMock()
        return contract

    @pytest.fixture
    def bridge(self, mock_provider):
        """Create bridge instance."""
        return BlockchainIdentityBridge(provider=mock_provider)

    def test_create_bridge(self, bridge):
        """Test creating bridge instance."""
        assert bridge is not None

    def test_create_bridge_no_provider(self):
        """Test creating bridge without provider (lazy init)."""
        bridge = BlockchainIdentityBridge()
        assert bridge is not None
        assert bridge._provider is None

    @pytest.mark.asyncio
    async def test_link_agent(self, bridge, mock_provider, mock_identity_contract):
        """Test linking an Aragora agent to blockchain identity."""
        mock_identity = MagicMock()
        mock_identity.owner = "0x1234567890abcdef"
        mock_identity_contract.get_agent.return_value = mock_identity

        with patch.object(bridge, "_get_identity_contract", return_value=mock_identity_contract):
            result = await bridge.link_agent(
                aragora_agent_id="agent_abc123",
                token_id=42,
            )

        assert isinstance(result, AgentBlockchainLink)
        assert result.aragora_agent_id == "agent_abc123"
        assert result.token_id == 42
        assert result.chain_id == 1
        assert result.verified is True
        assert result.owner_address == "0x1234567890abcdef"

    @pytest.mark.asyncio
    async def test_link_agent_no_verify(self, bridge, mock_provider):
        """Test linking without on-chain verification."""
        result = await bridge.link_agent(
            aragora_agent_id="agent_abc123",
            token_id=42,
            verify=False,
        )

        assert isinstance(result, AgentBlockchainLink)
        assert result.verified is False
        assert result.owner_address == ""

    @pytest.mark.asyncio
    async def test_link_agent_verification_fails(self, bridge, mock_identity_contract):
        """Test linking when on-chain verification fails."""
        mock_identity_contract.get_agent.side_effect = RuntimeError("Contract error")

        with patch.object(bridge, "_get_identity_contract", return_value=mock_identity_contract):
            with pytest.raises(ValueError, match="Failed to verify"):
                await bridge.link_agent(
                    aragora_agent_id="agent_abc123",
                    token_id=42,
                )

    @pytest.mark.asyncio
    async def test_get_blockchain_identity(self, bridge, mock_identity_contract):
        """Test getting blockchain identity for a linked agent."""
        # First link the agent
        await bridge.link_agent("agent_123", token_id=42, verify=False)

        mock_identity = MagicMock()
        mock_identity.owner = "0xabcd"
        mock_identity_contract.get_agent.return_value = mock_identity

        with patch.object(bridge, "_get_identity_contract", return_value=mock_identity_contract):
            identity = await bridge.get_blockchain_identity(aragora_agent_id="agent_123")

        assert identity is not None
        mock_identity_contract.get_agent.assert_called_with(42)

    @pytest.mark.asyncio
    async def test_get_blockchain_identity_not_linked(self, bridge):
        """Test getting identity for unlinked agent returns None."""
        identity = await bridge.get_blockchain_identity(aragora_agent_id="nonexistent")
        assert identity is None

    @pytest.mark.asyncio
    async def test_get_blockchain_identity_contract_error(self, bridge, mock_identity_contract):
        """Test getting identity when contract call fails."""
        await bridge.link_agent("agent_123", token_id=42, verify=False)
        mock_identity_contract.get_agent.side_effect = ConnectionError("RPC error")

        with patch.object(bridge, "_get_identity_contract", return_value=mock_identity_contract):
            identity = await bridge.get_blockchain_identity(aragora_agent_id="agent_123")

        assert identity is None

    @pytest.mark.asyncio
    async def test_get_agent_by_token(self, bridge, mock_provider):
        """Test getting Aragora agent by blockchain token ID."""
        await bridge.link_agent("agent_2", token_id=42, verify=False)

        agent = await bridge.get_agent_by_token(token_id=42)
        assert agent == "agent_2"

    @pytest.mark.asyncio
    async def test_get_agent_by_token_not_found(self, bridge, mock_provider):
        """Test getting agent when token not linked."""
        agent = await bridge.get_agent_by_token(token_id=999)
        assert agent is None


class TestBlockchainIdentityBridgeUnlink:
    """Tests for bridge unlink operations."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock Web3Provider."""
        provider = MagicMock()
        config = MagicMock()
        config.chain_id = 1
        provider.get_config.return_value = config
        return provider

    @pytest.fixture
    def bridge(self, mock_provider):
        """Create bridge instance."""
        return BlockchainIdentityBridge(provider=mock_provider)

    @pytest.mark.asyncio
    async def test_unlink_agent(self, bridge):
        """Test unlinking an agent."""
        await bridge.link_agent("agent_1", token_id=42, verify=False)
        assert bridge.get_link("agent_1") is not None

        result = await bridge.unlink_agent("agent_1")
        assert result is True
        assert bridge.get_link("agent_1") is None

    @pytest.mark.asyncio
    async def test_unlink_agent_not_found(self, bridge):
        """Test unlinking non-existent agent."""
        result = await bridge.unlink_agent("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_agents_by_owner(self, bridge, mock_provider):
        """Test getting agents by owner address."""
        mock_identity = MagicMock()
        mock_identity.owner = "0xAbCd"

        mock_contract = MagicMock()
        mock_contract.get_agent.return_value = mock_identity

        with patch.object(bridge, "_get_identity_contract", return_value=mock_contract):
            await bridge.link_agent("agent_1", token_id=1)
            await bridge.link_agent("agent_2", token_id=2)

        await bridge.link_agent("agent_3", token_id=3, verify=False)

        agents = await bridge.get_agents_by_owner("0xabcd")
        assert "agent_1" in agents
        assert "agent_2" in agents
        assert "agent_3" not in agents  # empty owner_address


class TestBlockchainIdentityBridgeLinks:
    """Tests for link query operations."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock Web3Provider."""
        provider = MagicMock()
        config = MagicMock()
        config.chain_id = 1
        provider.get_config.return_value = config
        return provider

    @pytest.fixture
    def bridge(self, mock_provider):
        """Create bridge instance."""
        return BlockchainIdentityBridge(provider=mock_provider)

    @pytest.mark.asyncio
    async def test_get_link(self, bridge):
        """Test getting link record."""
        await bridge.link_agent("agent_1", token_id=42, verify=False)
        link = bridge.get_link("agent_1")
        assert link is not None
        assert link.token_id == 42

    def test_get_link_not_found(self, bridge):
        """Test getting non-existent link."""
        link = bridge.get_link("nonexistent")
        assert link is None

    @pytest.mark.asyncio
    async def test_get_all_links(self, bridge):
        """Test getting all links."""
        await bridge.link_agent("agent_1", token_id=1, verify=False)
        await bridge.link_agent("agent_2", token_id=2, verify=False)

        links = bridge.get_all_links()
        assert len(links) == 2

    @pytest.mark.asyncio
    async def test_verify_link_success(self, bridge):
        """Test verifying a valid link."""
        await bridge.link_agent("agent_1", token_id=42, verify=False)

        mock_identity = MagicMock()
        mock_identity.owner = "0xNewOwner"
        mock_contract = MagicMock()
        mock_contract.get_agent.return_value = mock_identity

        with patch.object(bridge, "_get_identity_contract", return_value=mock_contract):
            result = await bridge.verify_link("agent_1")

        assert result is True
        link = bridge.get_link("agent_1")
        assert link.verified is True
        assert link.owner_address == "0xNewOwner"

    @pytest.mark.asyncio
    async def test_verify_link_failure(self, bridge):
        """Test verifying when contract call fails."""
        await bridge.link_agent("agent_1", token_id=42, verify=False)

        mock_contract = MagicMock()
        mock_contract.get_agent.side_effect = ConnectionError("RPC error")

        with patch.object(bridge, "_get_identity_contract", return_value=mock_contract):
            result = await bridge.verify_link("agent_1")

        assert result is False
        link = bridge.get_link("agent_1")
        assert link.verified is False

    @pytest.mark.asyncio
    async def test_verify_link_not_found(self, bridge):
        """Test verifying non-existent link."""
        result = await bridge.verify_link("nonexistent")
        assert result is False
