"""
Tests for ERC-8004 blockchain connector.

Tests the ERC8004Connector for retrieving on-chain agent identity,
reputation, and validation data with mocked Web3 and contract interfaces.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from datetime import datetime, timezone

from aragora.connectors.blockchain.connector import ERC8004Connector, _check_web3
from aragora.connectors.blockchain.credentials import BlockchainCredentials
from aragora.connectors.blockchain.models import (
    BlockchainEvidence,
    BlockchainSearchResult,
    make_block_explorer_url,
)
from aragora.connectors.base import Evidence


# Mock data for tests
MOCK_CHAIN_ID = 1
MOCK_TOKEN_ID = 42
MOCK_OWNER = "0x1234567890123456789012345678901234567890"
MOCK_AGENT_URI = "ipfs://QmTest123456789/agent.json"
MOCK_WALLET = "0x9876543210987654321098765432109876543210"
MOCK_IDENTITY_REGISTRY = "0xAAAA567890123456789012345678901234567890"
MOCK_REPUTATION_REGISTRY = "0xBBBB567890123456789012345678901234567890"
MOCK_VALIDATION_REGISTRY = "0xCCCC567890123456789012345678901234567890"
MOCK_RPC_URL = "https://rpc.example.com"
MOCK_REQUEST_HASH = "abcd1234" * 8  # 64 hex chars


@pytest.fixture
def mock_credentials():
    """Create mock blockchain credentials."""
    return BlockchainCredentials(
        rpc_url=MOCK_RPC_URL,
        chain_id=MOCK_CHAIN_ID,
        identity_registry=MOCK_IDENTITY_REGISTRY,
        reputation_registry=MOCK_REPUTATION_REGISTRY,
        validation_registry=MOCK_VALIDATION_REGISTRY,
        fallback_rpc_urls=["https://fallback1.example.com"],
    )


@pytest.fixture
def mock_identity():
    """Create a mock OnChainAgentIdentity."""
    from aragora.blockchain.models import OnChainAgentIdentity

    return OnChainAgentIdentity(
        token_id=MOCK_TOKEN_ID,
        owner=MOCK_OWNER,
        agent_uri=MOCK_AGENT_URI,
        wallet_address=MOCK_WALLET,
        chain_id=MOCK_CHAIN_ID,
        tx_hash="0xtxhash123",
    )


@pytest.fixture
def mock_reputation_summary():
    """Create a mock ReputationSummary."""
    from aragora.blockchain.models import ReputationSummary

    return ReputationSummary(
        agent_id=MOCK_TOKEN_ID,
        count=10,
        summary_value=850,
        summary_value_decimals=2,
        tag1="accuracy",
        tag2="",
    )


@pytest.fixture
def mock_validation_summary():
    """Create a mock ValidationSummary."""
    from aragora.blockchain.models import ValidationSummary

    return ValidationSummary(
        agent_id=MOCK_TOKEN_ID,
        count=5,
        average_response=1,
        tag="security",
    )


@pytest.fixture
def mock_validation_record():
    """Create a mock ValidationRecord."""
    from aragora.blockchain.models import ValidationRecord, ValidationResponse

    return ValidationRecord(
        request_hash=MOCK_REQUEST_HASH,
        agent_id=MOCK_TOKEN_ID,
        validator_address="0xVALIDATOR123",
        response=ValidationResponse.PASS,
        tag="security",
        tx_hash="0xvalidation_tx_123",
    )


class TestERC8004ConnectorInit:
    """Tests for connector initialization."""

    def test_connector_init_default(self):
        """Connector should initialize with default credentials."""
        with patch.dict("os.environ", {}, clear=True):
            connector = ERC8004Connector()
            assert connector.name == "erc8004"
            assert connector.source_type == "blockchain"

    def test_connector_init_with_credentials(self, mock_credentials):
        """Connector should accept custom credentials."""
        connector = ERC8004Connector(credentials=mock_credentials)
        assert connector._credentials.rpc_url == MOCK_RPC_URL
        assert connector._credentials.chain_id == MOCK_CHAIN_ID
        assert connector._credentials.identity_registry == MOCK_IDENTITY_REGISTRY

    def test_connector_from_env(self):
        """from_env should create connector from environment variables."""
        env_vars = {
            "ERC8004_RPC_URL": MOCK_RPC_URL,
            "ERC8004_CHAIN_ID": str(MOCK_CHAIN_ID),
            "ERC8004_IDENTITY_REGISTRY": MOCK_IDENTITY_REGISTRY,
        }
        with patch.dict("os.environ", env_vars, clear=True):
            connector = ERC8004Connector.from_env()
            assert connector._credentials.rpc_url == MOCK_RPC_URL
            assert connector._credentials.chain_id == MOCK_CHAIN_ID

    def test_connector_is_configured(self, mock_credentials):
        """is_configured should return True when credentials are set."""
        connector = ERC8004Connector(credentials=mock_credentials)
        assert connector.is_configured is True

    def test_connector_is_configured_false(self):
        """is_configured should return False with empty credentials."""
        empty_creds = BlockchainCredentials()
        connector = ERC8004Connector(credentials=empty_creds)
        assert connector.is_configured is False


class TestWeb3Availability:
    """Tests for web3 dependency checking."""

    def test_is_available_with_web3(self):
        """is_available should return True when web3 is installed."""
        with patch("aragora.connectors.blockchain.connector._check_web3", return_value=True):
            connector = ERC8004Connector()
            assert connector.is_available is True

    def test_is_available_without_web3(self):
        """is_available should return False when web3 is not installed."""
        with patch("aragora.connectors.blockchain.connector._check_web3", return_value=False):
            connector = ERC8004Connector()
            assert connector.is_available is False


class TestProviderInitialization:
    """Tests for lazy provider initialization."""

    def test_get_provider_requires_web3(self, mock_credentials):
        """_get_provider should raise ImportError if web3 unavailable."""
        connector = ERC8004Connector(credentials=mock_credentials)

        with patch("aragora.connectors.blockchain.connector._check_web3", return_value=False):
            with pytest.raises(ImportError) as exc_info:
                connector._get_provider()
            assert "web3 is required" in str(exc_info.value)

    def test_get_provider_lazy_initialization(self, mock_credentials):
        """_get_provider should lazily initialize the Web3Provider."""
        connector = ERC8004Connector(credentials=mock_credentials)
        assert connector._provider is None

        mock_provider = MagicMock()
        mock_config = MagicMock()
        mock_config.chain_id = MOCK_CHAIN_ID

        with patch("aragora.connectors.blockchain.connector._check_web3", return_value=True):
            with patch(
                "aragora.blockchain.provider.Web3Provider.from_config", return_value=mock_provider
            ):
                with patch("aragora.blockchain.config.ChainConfig", return_value=mock_config):
                    provider = connector._get_provider()

        # Provider should now be cached
        assert connector._provider is not None

    def test_provider_caches_instance(self, mock_credentials):
        """Provider should be cached after first initialization."""
        connector = ERC8004Connector(credentials=mock_credentials)

        mock_provider = MagicMock()
        mock_config = MagicMock()

        with patch("aragora.connectors.blockchain.connector._check_web3", return_value=True):
            with patch(
                "aragora.blockchain.provider.Web3Provider.from_config", return_value=mock_provider
            ):
                with patch("aragora.blockchain.config.ChainConfig", return_value=mock_config):
                    provider1 = connector._get_provider()
                    provider2 = connector._get_provider()

        assert provider1 is provider2


class TestSearchFunctionality:
    """Tests for search functionality."""

    @pytest.fixture
    def connector_with_mocks(self, mock_credentials, mock_identity):
        """Create connector with mocked contracts."""
        connector = ERC8004Connector(credentials=mock_credentials)

        mock_identity_contract = MagicMock()
        mock_identity_contract.get_agent.return_value = mock_identity

        connector._identity_contract = mock_identity_contract
        connector._provider = MagicMock()

        return connector

    @pytest.mark.asyncio
    async def test_search_agent_by_token_id(self, connector_with_mocks, mock_identity):
        """Search should return results for agent:{token_id} query."""
        results = await connector_with_mocks.search(f"agent:{MOCK_TOKEN_ID}")

        assert len(results) == 1
        assert isinstance(results[0], BlockchainSearchResult)
        assert f"Agent #{MOCK_TOKEN_ID}" in results[0].title
        assert MOCK_OWNER[:10] in results[0].snippet

    @pytest.mark.asyncio
    async def test_search_reputation_by_token_id(self, mock_credentials, mock_reputation_summary):
        """Search should return results for reputation:{token_id} query."""
        connector = ERC8004Connector(credentials=mock_credentials)

        mock_reputation_contract = MagicMock()
        mock_reputation_contract.get_summary.return_value = mock_reputation_summary
        connector._reputation_contract = mock_reputation_contract
        connector._provider = MagicMock()

        results = await connector.search(f"reputation:{MOCK_TOKEN_ID}")

        assert len(results) == 1
        assert "Reputation" in results[0].title
        assert f"Count: {mock_reputation_summary.count}" in results[0].snippet

    @pytest.mark.asyncio
    async def test_search_validation_by_token_id(self, mock_credentials, mock_validation_summary):
        """Search should return results for validation:{token_id} query."""
        connector = ERC8004Connector(credentials=mock_credentials)

        mock_validation_contract = MagicMock()
        mock_validation_contract.get_summary.return_value = mock_validation_summary
        connector._validation_contract = mock_validation_contract
        connector._provider = MagicMock()

        results = await connector.search(f"validation:{MOCK_TOKEN_ID}")

        assert len(results) == 1
        assert "Validations" in results[0].title
        assert f"Count: {mock_validation_summary.count}" in results[0].snippet

    @pytest.mark.asyncio
    async def test_search_respects_max_results(self, connector_with_mocks):
        """Search should respect the max_results parameter."""
        # Mock multiple results
        connector_with_mocks._identity_contract.get_agent.side_effect = [
            MagicMock(owner=MOCK_OWNER, agent_uri=MOCK_AGENT_URI)
        ]

        results = await connector_with_mocks.search("agent:42", max_results=1)
        assert len(results) <= 1

    @pytest.mark.asyncio
    async def test_search_handles_empty_query(self, connector_with_mocks):
        """Search should handle empty query gracefully."""
        results = await connector_with_mocks.search("")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_handles_invalid_query_format(self, connector_with_mocks):
        """Search should handle invalid query format gracefully."""
        results = await connector_with_mocks.search("invalid_query")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_handles_contract_error(self, connector_with_mocks):
        """Search should handle contract errors gracefully."""
        connector_with_mocks._identity_contract.get_agent.side_effect = Exception("Contract error")

        results = await connector_with_mocks.search("agent:42")
        assert results == []


class TestFetchFunctionality:
    """Tests for fetch functionality."""

    @pytest.fixture
    def connector_with_mocks(self, mock_credentials, mock_identity):
        """Create connector with mocked contracts."""
        connector = ERC8004Connector(credentials=mock_credentials)

        mock_identity_contract = MagicMock()
        mock_identity_contract.get_agent.return_value = mock_identity

        connector._identity_contract = mock_identity_contract
        connector._provider = MagicMock()

        return connector

    @pytest.mark.asyncio
    async def test_fetch_identity_evidence(self, connector_with_mocks, mock_identity):
        """Fetch should return Evidence for identity type."""
        evidence_id = f"identity:{MOCK_CHAIN_ID}:{MOCK_TOKEN_ID}"

        evidence = await connector_with_mocks.fetch(evidence_id)

        assert evidence is not None
        assert isinstance(evidence, Evidence)
        assert f"Agent #{MOCK_TOKEN_ID}" in evidence.title
        assert MOCK_OWNER in evidence.content

    @pytest.mark.asyncio
    async def test_fetch_reputation_evidence(self, mock_credentials, mock_reputation_summary):
        """Fetch should return Evidence for reputation type."""
        connector = ERC8004Connector(credentials=mock_credentials)

        mock_reputation_contract = MagicMock()
        mock_reputation_contract.get_summary.return_value = mock_reputation_summary
        connector._reputation_contract = mock_reputation_contract
        connector._provider = MagicMock()

        evidence_id = f"reputation:{MOCK_CHAIN_ID}:{MOCK_TOKEN_ID}"
        evidence = await connector.fetch(evidence_id)

        assert evidence is not None
        assert "Reputation" in evidence.title
        assert "Feedback Count" in evidence.content

    @pytest.mark.asyncio
    async def test_fetch_validation_evidence(self, mock_credentials, mock_validation_record):
        """Fetch should return Evidence for validation type."""
        connector = ERC8004Connector(credentials=mock_credentials)

        mock_validation_contract = MagicMock()
        mock_validation_contract.get_validation_status.return_value = mock_validation_record
        connector._validation_contract = mock_validation_contract
        connector._provider = MagicMock()

        evidence_id = f"validation:{MOCK_CHAIN_ID}:{MOCK_REQUEST_HASH}"
        evidence = await connector.fetch(evidence_id)

        assert evidence is not None
        assert "Validation" in evidence.title
        assert "Response" in evidence.content

    @pytest.mark.asyncio
    async def test_fetch_invalid_id_format(self, connector_with_mocks):
        """Fetch should return None for invalid ID format."""
        evidence = await connector_with_mocks.fetch("invalid")
        assert evidence is None

    @pytest.mark.asyncio
    async def test_fetch_unknown_type(self, connector_with_mocks):
        """Fetch should return None for unknown evidence type."""
        evidence = await connector_with_mocks.fetch("unknown:1:42")
        assert evidence is None

    @pytest.mark.asyncio
    async def test_fetch_handles_contract_error(self, connector_with_mocks):
        """Fetch should handle contract errors gracefully."""
        connector_with_mocks._identity_contract.get_agent.side_effect = Exception("Contract error")

        evidence = await connector_with_mocks.fetch(f"identity:{MOCK_CHAIN_ID}:{MOCK_TOKEN_ID}")
        assert evidence is None


class TestCacheFunctionality:
    """Tests for evidence caching."""

    @pytest.fixture
    def connector_with_mocks(self, mock_credentials, mock_identity):
        """Create connector with mocked contracts."""
        connector = ERC8004Connector(credentials=mock_credentials)

        mock_identity_contract = MagicMock()
        mock_identity_contract.get_agent.return_value = mock_identity

        connector._identity_contract = mock_identity_contract
        connector._provider = MagicMock()

        return connector

    @pytest.mark.asyncio
    async def test_fetch_caches_result(self, connector_with_mocks):
        """Fetch should cache successful results."""
        evidence_id = f"identity:{MOCK_CHAIN_ID}:{MOCK_TOKEN_ID}"

        # First fetch
        evidence1 = await connector_with_mocks.fetch(evidence_id)
        assert evidence1 is not None

        # Second fetch should use cache
        evidence2 = await connector_with_mocks.fetch(evidence_id)
        assert evidence2 is not None

        # Contract should only be called once
        assert connector_with_mocks._identity_contract.get_agent.call_count == 1

    @pytest.mark.asyncio
    async def test_fetch_returns_cached_evidence(self, connector_with_mocks):
        """Fetch should return cached evidence without calling contract."""
        evidence_id = f"identity:{MOCK_CHAIN_ID}:{MOCK_TOKEN_ID}"

        # First fetch to populate cache
        await connector_with_mocks.fetch(evidence_id)

        # Reset mock to verify no additional calls
        connector_with_mocks._identity_contract.get_agent.reset_mock()

        # Second fetch should use cache
        evidence = await connector_with_mocks.fetch(evidence_id)
        assert evidence is not None
        connector_with_mocks._identity_contract.get_agent.assert_not_called()


class TestHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_credentials):
        """Health check should return True when provider is connected."""
        connector = ERC8004Connector(credentials=mock_credentials)

        mock_provider = MagicMock()
        mock_provider.is_connected.return_value = True
        connector._provider = mock_provider

        result = await connector._perform_health_check(timeout=5.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_credentials):
        """Health check should return False when provider is disconnected."""
        connector = ERC8004Connector(credentials=mock_credentials)

        mock_provider = MagicMock()
        mock_provider.is_connected.return_value = False
        connector._provider = mock_provider

        result = await connector._perform_health_check(timeout=5.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_exception(self, mock_credentials):
        """Health check should return False on exception."""
        connector = ERC8004Connector(credentials=mock_credentials)

        mock_provider = MagicMock()
        mock_provider.is_connected.side_effect = Exception("Connection failed")
        connector._provider = mock_provider

        # _get_provider would raise, mocking it directly
        with patch.object(connector, "_get_provider", return_value=mock_provider):
            result = await connector._perform_health_check(timeout=5.0)
            assert result is False


class TestBlockchainModels:
    """Tests for blockchain-specific models."""

    def test_blockchain_search_result_properties(self):
        """BlockchainSearchResult should correctly parse ID components."""
        result = BlockchainSearchResult(
            id=f"identity:{MOCK_CHAIN_ID}:{MOCK_TOKEN_ID}",
            title="Test Agent",
            snippet="Test snippet",
        )

        assert result.evidence_type == "identity"
        assert result.chain_id == MOCK_CHAIN_ID
        assert result.token_id == MOCK_TOKEN_ID

    def test_blockchain_evidence_default_reliability(self):
        """BlockchainEvidence should have high default reliability scores."""
        evidence = BlockchainEvidence(
            id="test",
            evidence_type="identity",
            chain_id=1,
            token_id=42,
            title="Test",
            content="Test content",
        )

        assert evidence.confidence == 0.9
        assert evidence.freshness == 1.0
        assert evidence.authority == 0.85

    def test_blockchain_evidence_to_dict(self):
        """BlockchainEvidence.to_evidence_dict should serialize correctly."""
        evidence = BlockchainEvidence(
            id="identity:1:42",
            evidence_type="identity",
            chain_id=1,
            token_id=42,
            title="Test Agent",
            content="Agent content",
            contract_address=MOCK_IDENTITY_REGISTRY,
            tx_hash="0x123",
        )

        data = evidence.to_evidence_dict()

        assert data["id"] == "identity:1:42"
        assert data["source_type"] == "blockchain"
        assert data["title"] == "Test Agent"
        assert data["metadata"]["chain_id"] == 1
        assert data["metadata"]["tx_hash"] == "0x123"


class TestBlockExplorerURLs:
    """Tests for block explorer URL generation."""

    def test_etherscan_mainnet_tx(self):
        """Should generate correct Etherscan mainnet transaction URL."""
        url = make_block_explorer_url(1, tx_hash="0x123")
        assert url == "https://etherscan.io/tx/0x123"

    def test_etherscan_mainnet_address(self):
        """Should generate correct Etherscan mainnet address URL."""
        url = make_block_explorer_url(1, address=MOCK_IDENTITY_REGISTRY)
        assert url == f"https://etherscan.io/address/{MOCK_IDENTITY_REGISTRY}"

    def test_sepolia_tx(self):
        """Should generate correct Sepolia testnet URL."""
        url = make_block_explorer_url(11155111, tx_hash="0x456")
        assert url == "https://sepolia.etherscan.io/tx/0x456"

    def test_polygon_address(self):
        """Should generate correct Polygon URL."""
        url = make_block_explorer_url(137, address=MOCK_IDENTITY_REGISTRY)
        assert url == f"https://polygonscan.com/address/{MOCK_IDENTITY_REGISTRY}"

    def test_base_mainnet(self):
        """Should generate correct Base mainnet URL."""
        url = make_block_explorer_url(8453, address=MOCK_IDENTITY_REGISTRY)
        assert url == f"https://basescan.org/address/{MOCK_IDENTITY_REGISTRY}"

    def test_unknown_chain_falls_back_to_etherscan(self):
        """Unknown chain IDs should fall back to Etherscan."""
        url = make_block_explorer_url(999999, address=MOCK_IDENTITY_REGISTRY)
        assert url == f"https://etherscan.io/address/{MOCK_IDENTITY_REGISTRY}"

    def test_no_tx_or_address_returns_base_url(self):
        """Should return base explorer URL when no tx_hash or address provided."""
        url = make_block_explorer_url(1)
        assert url == "https://etherscan.io"


class TestContractInitialization:
    """Tests for lazy contract initialization."""

    def test_identity_contract_lazy_init(self, mock_credentials):
        """Identity contract should be lazily initialized."""
        connector = ERC8004Connector(credentials=mock_credentials)
        assert connector._identity_contract is None

    def test_reputation_contract_lazy_init(self, mock_credentials):
        """Reputation contract should be lazily initialized."""
        connector = ERC8004Connector(credentials=mock_credentials)
        assert connector._reputation_contract is None

    def test_validation_contract_lazy_init(self, mock_credentials):
        """Validation contract should be lazily initialized."""
        connector = ERC8004Connector(credentials=mock_credentials)
        assert connector._validation_contract is None


class TestCredentialsConfiguration:
    """Tests for BlockchainCredentials."""

    def test_credentials_from_env(self):
        """BlockchainCredentials should load from environment."""
        env_vars = {
            "ERC8004_RPC_URL": MOCK_RPC_URL,
            "ERC8004_CHAIN_ID": "137",
            "ERC8004_IDENTITY_REGISTRY": MOCK_IDENTITY_REGISTRY,
            "ERC8004_REPUTATION_REGISTRY": MOCK_REPUTATION_REGISTRY,
            "ERC8004_FALLBACK_RPC_URLS": "https://fb1.com,https://fb2.com",
        }
        with patch.dict("os.environ", env_vars, clear=True):
            creds = BlockchainCredentials.from_env()

        assert creds.rpc_url == MOCK_RPC_URL
        assert creds.chain_id == 137
        assert creds.identity_registry == MOCK_IDENTITY_REGISTRY
        assert creds.reputation_registry == MOCK_REPUTATION_REGISTRY
        assert len(creds.fallback_rpc_urls) == 2

    def test_credentials_is_configured_needs_rpc_and_registry(self):
        """is_configured should require RPC URL and at least one registry."""
        # Only RPC - not configured
        creds1 = BlockchainCredentials(rpc_url=MOCK_RPC_URL)
        assert creds1.is_configured is False

        # Only registry - not configured
        creds2 = BlockchainCredentials(identity_registry=MOCK_IDENTITY_REGISTRY)
        assert creds2.is_configured is False

        # Both RPC and registry - configured
        creds3 = BlockchainCredentials(
            rpc_url=MOCK_RPC_URL,
            identity_registry=MOCK_IDENTITY_REGISTRY,
        )
        assert creds3.is_configured is True

    def test_has_registry_properties(self):
        """Registry presence properties should work correctly."""
        creds = BlockchainCredentials(
            rpc_url=MOCK_RPC_URL,
            identity_registry=MOCK_IDENTITY_REGISTRY,
            reputation_registry="",
            validation_registry=MOCK_VALIDATION_REGISTRY,
        )

        assert creds.has_identity_registry is True
        assert creds.has_reputation_registry is False
        assert creds.has_validation_registry is True
