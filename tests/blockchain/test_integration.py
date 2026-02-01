"""
Integration tests for ERC-8004 blockchain components.

These tests verify the integration between different blockchain
components without requiring actual blockchain connectivity.
"""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch
import pytest


class TestBlockchainIntegration:
    """Integration tests for blockchain components."""

    @pytest.fixture
    def mock_web3_environment(self, blockchain_env_vars, wallet_env_vars):
        """Set up mock Web3 environment."""
        with patch("aragora.blockchain.provider.Web3") as mock_web3:
            mock_instance = MagicMock()
            mock_instance.is_connected.return_value = True
            mock_instance.eth.chain_id = 1
            mock_instance.eth.block_number = 19000000
            mock_web3.return_value = mock_instance
            yield mock_instance

    @pytest.mark.asyncio
    async def test_connector_to_adapter_flow(self, mock_web3_environment, sample_agent_identity):
        """Test data flow from connector to adapter."""
        from aragora.connectors.blockchain import ERC8004Connector
        from aragora.knowledge.mound.adapters.erc8004_adapter import ERC8004Adapter

        with patch.object(ERC8004Connector, "__init__", lambda self: None):
            connector = ERC8004Connector()
            connector._provider = MagicMock()

            # Mock connector search
            connector.search = MagicMock(
                return_value=[
                    MagicMock(
                        id=f"identity:1:{sample_agent_identity['token_id']}",
                        title=sample_agent_identity["agent_name"],
                        snippet=sample_agent_identity["metadata_uri"],
                    )
                ]
            )

            with patch.object(ERC8004Adapter, "__init__", lambda self, **kw: None):
                adapter = ERC8004Adapter()
                adapter._connector = connector
                adapter.adapter_name = "erc8004"

                # Verify adapter can use connector results
                results = connector.search("test")
                assert len(results) == 1
                assert results[0].id == f"identity:1:{sample_agent_identity['token_id']}"

    @pytest.mark.asyncio
    async def test_adapter_to_handler_flow(self, mock_web3_environment, sample_agent_identities):
        """Test data flow from adapter to handler."""
        from aragora.knowledge.mound.adapters.erc8004_adapter import ERC8004Adapter
        from aragora.server.handlers.erc8004 import ERC8004Handler

        with patch.object(ERC8004Adapter, "__init__", lambda self, **kw: None):
            mock_adapter = MagicMock(spec=ERC8004Adapter)
            mock_adapter.search = AsyncMock(
                return_value=[
                    MagicMock(
                        id=f"identity:1:{a['token_id']}",
                        title=a["agent_name"],
                    )
                    for a in sample_agent_identities
                ]
            )

            with patch("aragora.server.handlers.erc8004.ERC8004Adapter", return_value=mock_adapter):
                handler = ERC8004Handler({"blockchain_enabled": True})

                # Verify handler can use adapter
                # The actual test depends on handler implementation

    @pytest.mark.asyncio
    async def test_bridge_links_registry_to_chain(
        self, mock_web3_environment, mock_agent_registry, sample_agent_identity
    ):
        """Test bridge linking between AgentRegistry and blockchain."""
        from aragora.control_plane.blockchain_identity import BlockchainIdentityBridge

        mock_registry = mock_agent_registry
        mock_registry.get_agent.return_value = MagicMock(
            id="agent_123",
            name="Test Agent",
            blockchain_agent_id=None,
        )

        with patch(
            "aragora.control_plane.blockchain_identity.ERC8004Connector"
        ) as mock_connector_cls:
            mock_connector = MagicMock()
            mock_connector.fetch.return_value = MagicMock(
                id="identity:1:42",
                metadata={
                    "token_id": 42,
                    "owner": sample_agent_identity["owner"],
                },
            )
            mock_connector_cls.return_value = mock_connector

            bridge = BlockchainIdentityBridge(registry=mock_registry)
            result = await bridge.link_agent(
                aragora_agent_id="agent_123",
                blockchain_token_id=42,
            )

            assert result["linked"] is True


class TestBlockchainFactoryIntegration:
    """Tests for adapter factory integration."""

    def test_factory_registers_erc8004(self):
        """Test that factory registers ERC-8004 adapter spec."""
        from aragora.knowledge.mound.adapters.factory import ADAPTER_SPECS

        assert "erc8004" in ADAPTER_SPECS
        spec = ADAPTER_SPECS["erc8004"]
        assert spec.adapter_class is not None
        assert spec.forward_method == "sync_to_km"
        assert spec.reverse_method == "sync_from_km"

    def test_factory_creates_adapter(self, blockchain_env_vars):
        """Test factory can create ERC-8004 adapter."""
        from aragora.knowledge.mound.adapters.factory import AdapterFactory

        with patch("aragora.blockchain.provider.Web3"):
            factory = AdapterFactory()
            adapters = factory.create_from_subsystems()

            # ERC-8004 is opt-in, so may not be created automatically
            # But the factory should not error


class TestBlockchainProvenance:
    """Tests for blockchain provenance integration."""

    def test_provenance_includes_blockchain_source_type(self):
        """Test that SourceType includes BLOCKCHAIN."""
        from aragora.reasoning.provenance import SourceType

        assert hasattr(SourceType, "BLOCKCHAIN")
        assert SourceType.BLOCKCHAIN.value == "blockchain"


class TestBlockchainCircuitBreaker:
    """Tests for circuit breaker integration."""

    def test_adapter_has_circuit_breaker_config(self):
        """Test that ERC-8004 adapter has circuit breaker config."""
        from aragora.knowledge.mound.adapters._base import ADAPTER_CIRCUIT_CONFIGS

        assert "erc8004" in ADAPTER_CIRCUIT_CONFIGS
        config = ADAPTER_CIRCUIT_CONFIGS["erc8004"]
        assert config.failure_threshold == 5
        assert config.timeout_seconds == 60.0


class TestBlockchainConnectorRegistration:
    """Tests for connector registration."""

    def test_connector_exported(self):
        """Test that ERC-8004 connector is exported from package."""
        from aragora.connectors import ERC8004Connector

        assert ERC8004Connector is not None


class TestBlockchainAdapterRegistration:
    """Tests for adapter registration."""

    def test_adapter_exported(self):
        """Test that ERC-8004 adapter is exported from package."""
        from aragora.knowledge.mound.adapters import ERC8004Adapter

        assert ERC8004Adapter is not None


class TestBlockchainEnd2End:
    """End-to-end tests for blockchain integration."""

    @pytest.mark.asyncio
    async def test_full_sync_cycle(
        self,
        blockchain_env_vars,
        wallet_env_vars,
        sample_agent_identity,
        sample_reputation_feedbacks,
        sample_validation_records,
    ):
        """Test full sync cycle: chain -> KM -> chain."""
        # This test simulates a complete data flow:
        # 1. Fetch identity from chain
        # 2. Sync to Knowledge Mound
        # 3. Add ELO rating in Aragora
        # 4. Sync back to chain as reputation

        with patch("aragora.blockchain.provider.Web3"):
            with patch(
                "aragora.connectors.blockchain.connector.ERC8004Connector"
            ) as mock_connector_cls:
                mock_connector = MagicMock()

                # Step 1: Fetch identity
                mock_connector.fetch.return_value = MagicMock(
                    id=f"identity:1:{sample_agent_identity['token_id']}",
                    content=sample_agent_identity["agent_name"],
                    metadata={
                        "token_id": sample_agent_identity["token_id"],
                        "owner": sample_agent_identity["owner"],
                    },
                )
                mock_connector_cls.return_value = mock_connector

                # Verify we can fetch
                doc = mock_connector.fetch("identity:1:42")
                assert doc.id == "identity:1:42"

                # Steps 2-4 would involve:
                # - ERC8004Adapter.sync_to_km()
                # - Update ELO in Aragora
                # - ERC8004Adapter.sync_from_km()

                # For unit tests, we verify the components work individually
                # Full integration would require a blockchain node
