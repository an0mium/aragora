"""
Tests for ERC-8004 Knowledge Mound adapter.
"""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from aragora.knowledge.mound.adapters.erc8004_adapter import ERC8004Adapter
from aragora.knowledge.mound.adapters._types import SyncResult


class TestERC8004Adapter:
    """Tests for ERC8004Adapter class."""

    @pytest.fixture
    def mock_connector(self):
        """Create mock ERC8004Connector."""
        with patch("aragora.knowledge.mound.adapters.erc8004_adapter.ERC8004Connector") as mock_cls:
            mock_connector = MagicMock()
            mock_cls.return_value = mock_connector
            yield mock_connector

    @pytest.fixture
    def adapter(self, mock_connector):
        """Create adapter instance."""
        return ERC8004Adapter()

    def test_adapter_name(self, adapter):
        """Test adapter name property."""
        assert adapter.adapter_name == "erc8004"

    def test_adapter_priority(self, adapter):
        """Test adapter priority."""
        # Should be between receipt (90) and evidence (70)
        assert 70 < adapter.priority < 90

    @pytest.mark.asyncio
    async def test_sync_to_km(self, adapter, mock_connector, sample_agent_identities):
        """Test syncing blockchain data to Knowledge Mound."""
        mock_connector.search.return_value = [
            MagicMock(
                id=f"identity:1:{a['token_id']}",
                title=a["agent_name"],
                snippet=a["metadata_uri"],
            )
            for a in sample_agent_identities
        ]

        result = await adapter.sync_to_km()

        assert isinstance(result, SyncResult)
        assert result.records_synced >= 0

    @pytest.mark.asyncio
    async def test_sync_to_km_with_agent_ids(self, adapter, mock_connector, sample_agent_identity):
        """Test syncing specific agents to KM."""
        mock_connector.fetch.return_value = MagicMock(
            id="identity:1:42",
            content=sample_agent_identity["agent_name"],
        )

        result = await adapter.sync_to_km(agent_ids=[42])

        assert result.records_synced >= 0

    @pytest.mark.asyncio
    async def test_sync_to_km_identities_only(self, adapter, mock_connector):
        """Test syncing only identities."""
        mock_connector.search.return_value = []

        result = await adapter.sync_to_km(
            sync_identities=True,
            sync_reputation=False,
            sync_validations=False,
        )

        assert isinstance(result, SyncResult)

    @pytest.mark.asyncio
    async def test_sync_from_km(self, adapter, mock_km_mound):
        """Test syncing from Knowledge Mound to blockchain."""
        adapter._mound = mock_km_mound
        mock_km_mound.search.return_value = [
            MagicMock(
                id="km_node_1",
                metadata={"agent_id": "agent_123", "elo_rating": 1500},
            )
        ]

        result = await adapter.sync_from_km()

        assert isinstance(result, SyncResult)

    @pytest.mark.asyncio
    async def test_sync_from_km_elo_to_reputation(self, adapter, mock_km_mound):
        """Test syncing ELO ratings to blockchain reputation."""
        adapter._mound = mock_km_mound

        # Mock KM search returning ELO data
        mock_km_mound.search.return_value = [
            MagicMock(
                id="elo_record_1",
                metadata={
                    "agent_id": "agent_42",
                    "elo_rating": 1650,
                    "calibration": 0.92,
                },
            )
        ]

        result = await adapter.sync_from_km(
            sync_elo_to_reputation=True,
            sync_receipts_to_validation=False,
        )

        assert isinstance(result, SyncResult)

    @pytest.mark.asyncio
    async def test_sync_from_km_receipts_to_validation(self, adapter, mock_km_mound):
        """Test syncing Gauntlet receipts to blockchain validation."""
        adapter._mound = mock_km_mound

        # Mock KM search returning receipt data
        mock_km_mound.search.return_value = [
            MagicMock(
                id="receipt_1",
                metadata={
                    "agent_id": "agent_42",
                    "outcome": "pass",
                    "evidence_hash": "QmEvidence123",
                },
            )
        ]

        result = await adapter.sync_from_km(
            sync_elo_to_reputation=False,
            sync_receipts_to_validation=True,
        )

        assert isinstance(result, SyncResult)


class TestERC8004AdapterSearch:
    """Tests for ERC8004Adapter search methods."""

    @pytest.fixture
    def mock_connector(self):
        """Create mock connector."""
        with patch("aragora.knowledge.mound.adapters.erc8004_adapter.ERC8004Connector") as mock_cls:
            mock_connector = MagicMock()
            mock_cls.return_value = mock_connector
            yield mock_connector

    @pytest.fixture
    def adapter(self, mock_connector):
        """Create adapter instance."""
        return ERC8004Adapter()

    @pytest.mark.asyncio
    async def test_search(self, adapter, mock_connector, sample_agent_identities):
        """Test searching blockchain data via adapter."""
        mock_connector.search.return_value = [
            MagicMock(
                id=f"identity:1:{a['token_id']}",
                title=a["agent_name"],
                snippet=a["metadata_uri"],
                score=0.9,
            )
            for a in sample_agent_identities
        ]

        results = await adapter.search("claude", limit=10)

        assert len(results) > 0
        mock_connector.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_by_token_id(self, adapter, mock_connector, sample_agent_identity):
        """Test searching by token ID."""
        mock_connector.fetch.return_value = MagicMock(
            id="identity:1:42",
            content=sample_agent_identity["agent_name"],
        )

        results = await adapter.search_by_token_id(token_id=42)

        assert len(results) == 1


class TestERC8004AdapterHealth:
    """Tests for ERC8004Adapter health and status."""

    @pytest.fixture
    def mock_connector(self):
        """Create mock connector."""
        with patch("aragora.knowledge.mound.adapters.erc8004_adapter.ERC8004Connector") as mock_cls:
            mock_connector = MagicMock()
            mock_cls.return_value = mock_connector
            yield mock_connector

    @pytest.fixture
    def adapter(self, mock_connector):
        """Create adapter instance."""
        return ERC8004Adapter()

    def test_health_check(self, adapter, mock_connector):
        """Test adapter health check."""
        mock_connector.health_check.return_value = {
            "healthy": True,
            "connector": "erc8004",
            "chain_id": 1,
        }

        health = adapter.health_check()

        assert health["healthy"] is True
        assert health["adapter"] == "erc8004"

    def test_health_check_unhealthy(self, adapter, mock_connector):
        """Test adapter health check when unhealthy."""
        mock_connector.health_check.return_value = {
            "healthy": False,
            "error": "RPC connection failed",
        }

        health = adapter.health_check()

        assert health["healthy"] is False

    def test_get_stats(self, adapter, mock_connector):
        """Test getting adapter statistics."""
        stats = adapter.get_reverse_flow_stats()

        assert "validations_applied" in stats
        assert "adjustments_made" in stats
