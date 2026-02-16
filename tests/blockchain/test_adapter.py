"""
Tests for ERC-8004 Knowledge Mound adapter.
"""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from aragora.knowledge.mound.adapters.erc8004_adapter import ERC8004Adapter
from aragora.knowledge.mound.adapters._types import SyncResult, ValidationSyncResult


class TestERC8004Adapter:
    """Tests for ERC8004Adapter class."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock Web3 provider with disabled registries."""
        provider = MagicMock()
        config = MagicMock()
        config.has_identity_registry = False
        config.has_reputation_registry = False
        config.has_validation_registry = False
        provider.get_config.return_value = config
        return provider

    @pytest.fixture
    def adapter(self, mock_provider):
        """Create adapter instance."""
        return ERC8004Adapter(provider=mock_provider)

    def test_adapter_name(self, adapter):
        """Test adapter name property."""
        assert adapter.adapter_name == "erc8004"

    @pytest.mark.asyncio
    async def test_sync_to_km(self, adapter):
        """Test syncing blockchain data to Knowledge Mound."""
        result = await adapter.sync_to_km()

        assert isinstance(result, SyncResult)
        assert result.records_synced >= 0

    @pytest.mark.asyncio
    async def test_sync_to_km_with_agent_ids(self, adapter, sample_agent_identity):
        """Test syncing specific agents to KM."""
        adapter._provider.get_config.return_value.has_identity_registry = True
        identity_contract = MagicMock()
        identity_contract.get_agent.return_value = MagicMock(
            token_id=42,
            owner=sample_agent_identity["owner"],
            agent_uri=sample_agent_identity["metadata_uri"],
            wallet_address="0x" + "0" * 40,
            chain_id=1,
            aragora_agent_id="agent_42",
        )

        with patch.object(adapter, "_get_identity_contract", return_value=identity_contract):
            result = await adapter.sync_to_km(agent_ids=[42])

        assert result.records_synced >= 0

    @pytest.mark.asyncio
    async def test_sync_to_km_identities_only(self, adapter):
        """Test syncing only identities."""
        adapter._provider.get_config.return_value.has_identity_registry = True
        identity_contract = MagicMock()
        identity_contract.get_total_supply.return_value = 1
        identity_contract.get_agent.return_value = MagicMock(
            token_id=1,
            owner="0x" + "1" * 40,
            agent_uri="ipfs://test",
            wallet_address="0x" + "0" * 40,
            chain_id=1,
            aragora_agent_id="agent_1",
        )

        with patch.object(adapter, "_get_identity_contract", return_value=identity_contract):
            result = await adapter.sync_to_km(
                sync_identities=True,
                sync_reputation=False,
                sync_validations=False,
            )

        assert isinstance(result, SyncResult)

    @pytest.mark.asyncio
    async def test_sync_from_km_disabled(self, adapter):
        """Test reverse sync disabled by default."""
        result = await adapter.sync_from_km()
        assert isinstance(result, ValidationSyncResult)
        assert "Reverse sync is disabled" in result.errors

    @pytest.mark.asyncio
    async def test_sync_from_km_requires_signer(self, mock_provider):
        """Test reverse sync requires signer when enabled."""
        adapter = ERC8004Adapter(provider=mock_provider, enable_reverse_sync=True)
        result = await adapter.sync_from_km()
        assert any("No signer configured" in err for err in result.errors)

    @pytest.mark.asyncio
    async def test_sync_from_km_no_links(self, mock_provider):
        """Test reverse sync handles missing identity links."""
        adapter = ERC8004Adapter(
            provider=mock_provider,
            enable_reverse_sync=True,
            signer=MagicMock(),
        )
        adapter._provider.get_config.return_value.has_reputation_registry = False
        adapter._provider.get_config.return_value.has_validation_registry = False
        with patch.object(adapter, "_get_identity_bridge") as mock_bridge:
            mock_bridge.return_value.get_all_links.return_value = []
            result = await adapter.sync_from_km()

        assert isinstance(result, ValidationSyncResult)
        assert any("No agents linked" in e for e in result.errors)


class TestERC8004AdapterHealth:
    """Tests for ERC8004Adapter health and status."""

    @pytest.fixture
    def adapter(self):
        """Create adapter instance."""
        return ERC8004Adapter()

    def test_health_check(self, adapter):
        """Test adapter health check."""
        health = adapter.health_check()

        assert health["healthy"] is True
        assert health["adapter"] == "erc8004"

    def test_get_stats(self, adapter):
        """Test getting adapter statistics."""
        stats = adapter.get_reverse_flow_stats()

        assert "validations_applied" in stats
        assert "adjustments_made" in stats
