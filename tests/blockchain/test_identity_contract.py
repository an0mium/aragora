"""
Tests for Identity Registry contract wrapper.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, AsyncMock
import pytest

from aragora.blockchain.contracts.identity import (
    IdentityRegistryContract,
    IDENTITY_REGISTRY_ABI,
)


class TestIdentityRegistryABI:
    """Tests for Identity Registry ABI."""

    def test_abi_exists(self):
        """Test that ABI is defined."""
        assert IDENTITY_REGISTRY_ABI is not None
        assert len(IDENTITY_REGISTRY_ABI) > 0

    def test_abi_has_register(self):
        """Test ABI includes register function."""
        function_names = [
            item.get("name") for item in IDENTITY_REGISTRY_ABI if item.get("type") == "function"
        ]
        assert "register" in function_names

    def test_abi_has_agentURI(self):
        """Test ABI includes agentURI function."""
        function_names = [
            item.get("name") for item in IDENTITY_REGISTRY_ABI if item.get("type") == "function"
        ]
        assert "agentURI" in function_names

    def test_abi_has_events(self):
        """Test ABI includes events."""
        event_names = [
            item.get("name") for item in IDENTITY_REGISTRY_ABI if item.get("type") == "event"
        ]
        assert "Registered" in event_names


class TestIdentityRegistryContract:
    """Tests for IdentityRegistryContract wrapper."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock Web3Provider."""
        provider = MagicMock()
        mock_w3 = MagicMock()
        mock_contract = MagicMock()
        provider.get_web3.return_value = mock_w3
        provider.get_contract.return_value = mock_contract
        return provider, mock_w3, mock_contract

    def test_create_contract(self, mock_provider, mainnet_config):
        """Test creating contract instance."""
        provider, _, _ = mock_provider
        contract = IdentityRegistryContract(
            provider=provider,
            address=mainnet_config["identity_registry_address"],
            chain_id=1,
        )
        assert contract.address == mainnet_config["identity_registry_address"]
        assert contract.chain_id == 1

    def test_get_agent_uri(self, mock_provider):
        """Test getting agent metadata URI."""
        provider, _, mock_contract = mock_provider
        mock_contract.functions.agentURI.return_value.call.return_value = "ipfs://QmTest123"

        contract = IdentityRegistryContract(
            provider=provider,
            address="0x1111111111111111111111111111111111111111",
            chain_id=1,
        )
        uri = contract.get_agent_uri(token_id=42)

        assert uri == "ipfs://QmTest123"
        mock_contract.functions.agentURI.assert_called_once_with(42)

    def test_get_agent_owner(self, mock_provider):
        """Test getting agent owner address."""
        provider, _, mock_contract = mock_provider
        mock_contract.functions.ownerOf.return_value.call.return_value = (
            "0xOWNER1234567890123456789012345678901234"
        )

        contract = IdentityRegistryContract(
            provider=provider,
            address="0x1111111111111111111111111111111111111111",
            chain_id=1,
        )
        owner = contract.get_owner(token_id=42)

        assert owner == "0xOWNER1234567890123456789012345678901234"

    def test_get_agent(self, mock_provider, sample_agent_identity):
        """Test getting full agent identity."""
        provider, _, mock_contract = mock_provider
        mock_contract.functions.ownerOf.return_value.call.return_value = sample_agent_identity[
            "owner"
        ]
        mock_contract.functions.agentURI.return_value.call.return_value = sample_agent_identity[
            "metadata_uri"
        ]

        contract = IdentityRegistryContract(
            provider=provider,
            address="0x1111111111111111111111111111111111111111",
            chain_id=1,
        )
        identity = contract.get_agent(token_id=42)

        assert identity.token_id == 42
        assert identity.owner == sample_agent_identity["owner"]
        assert identity.metadata_uri == sample_agent_identity["metadata_uri"]

    def test_register_agent(self, mock_provider, mock_account):
        """Test registering a new agent."""
        provider, mock_w3, mock_contract = mock_provider

        # Mock the transaction flow
        mock_tx = {"hash": b"\xab" * 32}
        mock_contract.functions.register.return_value.build_transaction.return_value = {
            "to": "0x1111111111111111111111111111111111111111",
            "data": "0x...",
            "gas": 100000,
            "nonce": 0,
        }
        mock_w3.eth.send_raw_transaction.return_value = b"\xab" * 32
        mock_w3.eth.wait_for_transaction_receipt.return_value = {
            "status": 1,
            "transactionHash": b"\xab" * 32,
            "logs": [{"topics": [b"\x00" * 32], "data": b"\x00\x00\x00*"}],  # token_id = 42
        }

        with patch("aragora.blockchain.wallet.WalletSigner") as mock_signer_cls:
            mock_signer = MagicMock()
            mock_signer.address = "0xSIGNER12345678901234567890123456789012"
            mock_signed = MagicMock()
            mock_signed.raw_transaction = b"\x00" * 100
            mock_signer.sign_transaction.return_value = mock_signed
            mock_signer_cls.from_env.return_value = mock_signer

            contract = IdentityRegistryContract(
                provider=provider,
                address="0x1111111111111111111111111111111111111111",
                chain_id=1,
            )
            result = contract.register_agent(
                metadata_uri="ipfs://QmNewAgent",
                signer=mock_signer,
            )

            assert result is not None

    def test_set_agent_uri(self, mock_provider):
        """Test updating agent metadata URI."""
        provider, mock_w3, mock_contract = mock_provider

        mock_contract.functions.setAgentURI.return_value.build_transaction.return_value = {
            "to": "0x1111111111111111111111111111111111111111",
            "data": "0x...",
            "gas": 50000,
            "nonce": 0,
        }
        mock_w3.eth.send_raw_transaction.return_value = b"\xcd" * 32
        mock_w3.eth.wait_for_transaction_receipt.return_value = {"status": 1}

        with patch("aragora.blockchain.wallet.WalletSigner") as mock_signer_cls:
            mock_signer = MagicMock()
            mock_signer.address = "0xSIGNER12345678901234567890123456789012"
            mock_signed = MagicMock()
            mock_signed.raw_transaction = b"\x00" * 100
            mock_signer.sign_transaction.return_value = mock_signed

            contract = IdentityRegistryContract(
                provider=provider,
                address="0x1111111111111111111111111111111111111111",
                chain_id=1,
            )
            tx_hash = contract.set_agent_uri(
                token_id=42,
                uri="ipfs://QmUpdated",
                signer=mock_signer,
            )

            assert tx_hash is not None


class TestIdentityRegistryContractAsync:
    """Tests for async Identity Registry methods."""

    @pytest.fixture
    def mock_async_provider(self):
        """Create mock async Web3Provider."""
        provider = MagicMock()
        mock_w3 = MagicMock()
        mock_contract = MagicMock()
        provider.get_web3_async = AsyncMock(return_value=mock_w3)
        provider.get_contract.return_value = mock_contract
        return provider, mock_w3, mock_contract

    @pytest.mark.asyncio
    async def test_get_agent_async(self, mock_async_provider, sample_agent_identity):
        """Test async get agent."""
        provider, mock_w3, mock_contract = mock_async_provider
        mock_contract.functions.ownerOf.return_value.call = AsyncMock(
            return_value=sample_agent_identity["owner"]
        )
        mock_contract.functions.agentURI.return_value.call = AsyncMock(
            return_value=sample_agent_identity["metadata_uri"]
        )

        contract = IdentityRegistryContract(
            provider=provider,
            address="0x1111111111111111111111111111111111111111",
            chain_id=1,
        )
        identity = await contract.get_agent_async(token_id=42)

        assert identity.token_id == 42
