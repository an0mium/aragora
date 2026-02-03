"""
Tests for ERC-8004 connector module.
"""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from aragora.connectors.blockchain.connector import ERC8004Connector
from aragora.connectors.blockchain.credentials import BlockchainCredentials


class TestERC8004Connector:
    """Tests for ERC8004Connector class."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock Web3Provider."""
        with patch("aragora.blockchain.provider.Web3Provider") as mock_cls:
            mock_provider = MagicMock()
            mock_cls.from_env.return_value = mock_provider
            mock_cls.from_config.return_value = mock_provider
            yield mock_provider

    @pytest.fixture
    def connector(self, mock_provider):
        """Create connector instance."""
        return ERC8004Connector()

    def test_connector_name(self, connector):
        """Test connector name property."""
        assert connector.name == "erc8004"

    def test_connector_source_type(self, connector):
        """Test connector source type."""
        assert connector.source_type == "blockchain"

    @pytest.mark.asyncio
    async def test_search_agents(self, connector, mock_provider, sample_agent_identities):
        """Test searching for agents."""
        # Mock identity contract
        mock_identity = MagicMock()
        mock_identity.get_agents_by_query.return_value = [
            MagicMock(
                token_id=identity["token_id"],
                agent_name=identity["agent_name"],
                metadata_uri=identity["metadata_uri"],
            )
            for identity in sample_agent_identities
        ]
        connector._identity_contract = mock_identity

        results = await connector.search("claude", limit=10)

        assert len(results) > 0
        assert all(hasattr(r, "id") for r in results)
        assert all(hasattr(r, "title") for r in results)

    @pytest.mark.asyncio
    async def test_search_by_owner(self, connector, mock_provider, sample_agent_identities):
        """Test searching agents by owner address."""
        mock_identity = MagicMock()
        owner = sample_agent_identities[0]["owner"]

        # Filter to agents owned by this address
        owned = [a for a in sample_agent_identities if a["owner"] == owner]
        mock_identity.get_agents_by_owner.return_value = [
            MagicMock(
                token_id=a["token_id"],
                agent_name=a["agent_name"],
                metadata_uri=a["metadata_uri"],
                owner=a["owner"],
            )
            for a in owned
        ]
        connector._identity_contract = mock_identity

        results = await connector.search_by_owner(owner=owner)

        assert len(results) == 2  # Agents 1 and 3 are owned by same address

    @pytest.mark.asyncio
    async def test_fetch_identity(self, connector, mock_provider, sample_agent_identity):
        """Test fetching a specific agent identity."""
        mock_identity = MagicMock()
        mock_identity.get_agent.return_value = MagicMock(
            token_id=42,
            owner=sample_agent_identity["owner"],
            metadata_uri=sample_agent_identity["metadata_uri"],
            agent_name=sample_agent_identity["agent_name"],
        )
        connector._identity_contract = mock_identity

        doc = await connector.fetch("identity:1:42")

        assert doc is not None
        assert doc.id == "identity:1:42"
        assert "42" in str(doc.content) or sample_agent_identity["agent_name"] in doc.content

    @pytest.mark.asyncio
    async def test_fetch_reputation(self, connector, mock_provider, sample_reputation_feedbacks):
        """Test fetching agent reputation."""
        mock_reputation = MagicMock()
        mock_reputation.get_all_feedback.return_value = [
            MagicMock(
                reporter=fb["reporter"],
                value=fb["value"],
                tag=fb["tag"],
                revoked=fb.get("revoked", False),
            )
            for fb in sample_reputation_feedbacks
        ]
        connector._reputation_contract = mock_reputation

        doc = await connector.fetch("reputation:1:42")

        assert doc is not None
        assert "reputation" in doc.id.lower()

    @pytest.mark.asyncio
    async def test_fetch_validations(self, connector, mock_provider, sample_validation_records):
        """Test fetching agent validations."""
        mock_validation = MagicMock()
        mock_validation.get_all_validations.return_value = [
            MagicMock(
                validator=vr["validator"],
                request_type=vr["request_type"],
                response_code=vr.get("response_code", 0),
            )
            for vr in sample_validation_records
        ]
        connector._validation_contract = mock_validation

        doc = await connector.fetch("validation:1:42")

        assert doc is not None
        assert "validation" in doc.id.lower()

    @pytest.mark.asyncio
    async def test_fetch_invalid_id(self, connector, mock_provider):
        """Test fetching with invalid ID format."""
        with pytest.raises(ValueError, match="Invalid identifier"):
            await connector.fetch("invalid_id")

    @pytest.mark.asyncio
    async def test_health_check(self, connector, mock_provider):
        """Test connector health check."""
        mock_provider.health_check.return_value = {"healthy": True, "chain_id": 1}

        health = await connector.health_check()

        assert health["healthy"] is True
        assert health["connector"] == "erc8004"


class TestERC8004ConnectorAsync:
    """Tests for async ERC8004Connector methods."""

    @pytest.fixture
    def mock_async_provider(self):
        """Create mock async Web3Provider."""
        with patch("aragora.blockchain.provider.Web3Provider") as mock_cls:
            mock_provider = MagicMock()
            mock_provider.health_check_async = AsyncMock(
                return_value={"healthy": True, "chain_id": 1}
            )
            mock_cls.from_env.return_value = mock_provider
            mock_cls.from_config.return_value = mock_provider
            yield mock_provider

    @pytest.fixture
    def async_connector(self, mock_async_provider):
        """Create async connector instance."""
        return ERC8004Connector()

    @pytest.mark.asyncio
    async def test_search_async(
        self, async_connector, mock_async_provider, sample_agent_identities
    ):
        """Test async search."""
        mock_identity = MagicMock()
        mock_identity.get_agents_by_query_async = AsyncMock(
            return_value=[
                MagicMock(
                    token_id=a["token_id"],
                    agent_name=a["agent_name"],
                    metadata_uri=a["metadata_uri"],
                )
                for a in sample_agent_identities
            ]
        )
        async_connector._identity_contract = mock_identity

        results = await async_connector.search_async("agent", limit=5)

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_fetch_async(self, async_connector, mock_async_provider, sample_agent_identity):
        """Test async fetch."""
        mock_identity = MagicMock()
        mock_identity.get_agent_async = AsyncMock(
            return_value=MagicMock(
                token_id=42,
                owner=sample_agent_identity["owner"],
                metadata_uri=sample_agent_identity["metadata_uri"],
                agent_name=sample_agent_identity["agent_name"],
            )
        )
        async_connector._identity_contract = mock_identity

        doc = await async_connector.fetch_async("identity:1:42")

        assert doc is not None


class TestBlockchainCredentials:
    """Tests for BlockchainCredentials."""

    def test_create_credentials(self):
        """Test creating credentials."""
        creds = BlockchainCredentials(
            chain_id=1,
            rpc_url="https://eth.rpc",
            private_key="0x" + "a" * 64,
        )
        assert creds.chain_id == 1
        assert creds.rpc_url == "https://eth.rpc"

    def test_credentials_from_env(self, blockchain_env_vars, wallet_env_vars):
        """Test loading credentials from environment."""
        creds = BlockchainCredentials.from_env()

        assert creds is not None
        assert creds.chain_id == 1

    def test_credentials_private_key_hidden(self):
        """Test that private key is not exposed."""
        creds = BlockchainCredentials(
            chain_id=1,
            rpc_url="https://eth.rpc",
            private_key="0xsecret" + "a" * 58,
        )

        repr_str = repr(creds)
        assert "secret" not in repr_str.lower()


class TestERC8004ConnectorExceptionHandling:
    """Tests for ERC8004Connector exception handling with specific error types."""

    @pytest.fixture
    def connector_with_mocks(self):
        """Create connector with mocked internals."""
        with patch("aragora.connectors.blockchain.connector._check_web3", return_value=True):
            connector = ERC8004Connector()
            return connector

    @pytest.mark.asyncio
    async def test_health_check_import_error(self):
        """Test health check returns False on ImportError."""
        with patch("aragora.connectors.blockchain.connector._check_web3", return_value=False):
            connector = ERC8004Connector()

            result = await connector._perform_health_check(timeout=5.0)
            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_connection_error(self, connector_with_mocks):
        """Test health check handles ConnectionError gracefully."""
        connector = connector_with_mocks

        with patch.object(
            connector, "_get_provider", side_effect=ConnectionError("Failed to connect")
        ):
            result = await connector._perform_health_check(timeout=5.0)
            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_timeout_error(self, connector_with_mocks):
        """Test health check handles TimeoutError gracefully."""
        connector = connector_with_mocks

        with patch.object(
            connector, "_get_provider", side_effect=TimeoutError("Connection timed out")
        ):
            result = await connector._perform_health_check(timeout=5.0)
            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_runtime_error(self, connector_with_mocks):
        """Test health check handles RuntimeError gracefully."""
        connector = connector_with_mocks

        with patch.object(
            connector, "_get_provider", side_effect=RuntimeError("Provider not ready")
        ):
            result = await connector._perform_health_check(timeout=5.0)
            assert result is False

    @pytest.mark.asyncio
    async def test_search_value_error_logged(self, connector_with_mocks, caplog):
        """Test search logs ValueError with context."""
        import logging

        caplog.set_level(logging.WARNING)

        connector = connector_with_mocks

        # Search with invalid token_id format
        results = await connector.search("agent:not_a_number")

        assert results == []
        # Check that warning was logged
        assert any(
            "parse" in record.message.lower() or "error" in record.message.lower()
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_search_connection_error_logged(self, connector_with_mocks, caplog):
        """Test search logs ConnectionError with context."""
        import logging

        caplog.set_level(logging.ERROR)

        connector = connector_with_mocks

        # Mock contract to raise ConnectionError
        mock_contract = MagicMock()
        mock_contract.get_agent.side_effect = ConnectionError("Network unreachable")
        connector._identity_contract = mock_contract

        results = await connector.search("agent:42")

        assert results == []
        # Check that error was logged
        assert any("network" in record.message.lower() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_fetch_value_error_for_malformed_id(self, connector_with_mocks, caplog):
        """Test fetch handles malformed evidence_id."""
        import logging

        caplog.set_level(logging.WARNING)

        connector = connector_with_mocks

        # Malformed ID with invalid chain_id
        result = await connector.fetch("identity:not_a_number:42")

        assert result is None
        # Check that warning was logged
        assert any(
            "parse" in record.message.lower() or "error" in record.message.lower()
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_fetch_network_error_logged(self, connector_with_mocks, caplog):
        """Test fetch logs network errors with context."""
        import logging

        caplog.set_level(logging.ERROR)

        connector = connector_with_mocks

        # Mock contract to raise OSError (network error)
        mock_contract = MagicMock()
        mock_contract.get_agent.side_effect = OSError("Socket closed")
        connector._identity_contract = mock_contract

        with patch.object(connector, "_cache_get", return_value=None):
            result = await connector.fetch("identity:1:42")

        assert result is None
        # Check that error was logged with context
        assert any("network" in record.message.lower() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_fetch_runtime_error_logged(self, connector_with_mocks, caplog):
        """Test fetch logs RuntimeError with context."""
        import logging

        caplog.set_level(logging.ERROR)

        connector = connector_with_mocks

        # Mock contract to raise RuntimeError
        mock_contract = MagicMock()
        mock_contract.get_agent.side_effect = RuntimeError("Contract call reverted")
        connector._identity_contract = mock_contract

        with patch.object(connector, "_cache_get", return_value=None):
            result = await connector.fetch("identity:1:42")

        assert result is None
        # Check that error was logged
        assert any("runtime" in record.message.lower() for record in caplog.records)
