"""
Tests for Reputation Registry contract wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch, AsyncMock
import pytest

from aragora.blockchain.contracts.reputation import (
    ReputationRegistryContract,
    REPUTATION_REGISTRY_ABI,
)
from aragora.blockchain.models import ReputationFeedback


class TestReputationRegistryABI:
    """Tests for Reputation Registry ABI."""

    def test_abi_exists(self):
        """Test that ABI is defined."""
        assert REPUTATION_REGISTRY_ABI is not None
        assert len(REPUTATION_REGISTRY_ABI) > 0

    def test_abi_has_give_feedback(self):
        """Test ABI includes giveFeedback function."""
        function_names = [
            item.get("name") for item in REPUTATION_REGISTRY_ABI if item.get("type") == "function"
        ]
        assert "giveFeedback" in function_names

    def test_abi_has_revoke_feedback(self):
        """Test ABI includes revokeFeedback function."""
        function_names = [
            item.get("name") for item in REPUTATION_REGISTRY_ABI if item.get("type") == "function"
        ]
        assert "revokeFeedback" in function_names

    def test_abi_has_events(self):
        """Test ABI includes events."""
        event_names = [
            item.get("name") for item in REPUTATION_REGISTRY_ABI if item.get("type") == "event"
        ]
        assert "NewFeedback" in event_names


@dataclass
class MockChainConfig:
    """Mock chain configuration."""

    chain_id: int = 1
    reputation_registry_address: str = "0x2222222222222222222222222222222222222222"
    has_reputation_registry: bool = True
    gas_limit: int = 500_000


class TestReputationRegistryContract:
    """Tests for ReputationRegistryContract wrapper."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock Web3Provider with config."""
        provider = MagicMock()
        mock_w3 = MagicMock()
        mock_contract = MagicMock()
        mock_config = MockChainConfig()

        provider.get_web3.return_value = mock_w3
        provider.get_config.return_value = mock_config
        mock_w3.to_checksum_address.side_effect = lambda x: x
        mock_w3.eth.contract.return_value = mock_contract

        return provider, mock_w3, mock_contract

    def test_create_contract(self, mock_provider):
        """Test creating contract instance."""
        provider, _, _ = mock_provider
        contract = ReputationRegistryContract(
            provider=provider,
            chain_id=1,
        )
        # Contract is lazy-loaded, so we just verify the object is created
        assert contract._provider is provider
        assert contract._chain_id == 1

    def test_get_summary(self, mock_provider):
        """Test getting reputation summary."""
        provider, mock_w3, mock_contract = mock_provider

        # Mock getSummary call
        mock_contract.functions.getSummary.return_value.call.return_value = (
            15,  # count
            850,  # summary_value
            2,  # summary_value_decimals
        )

        contract = ReputationRegistryContract(
            provider=provider,
            chain_id=1,
        )
        summary = contract.get_summary(agent_id=42)

        assert summary.agent_id == 42
        assert summary.count == 15
        assert summary.summary_value == 850

    def test_read_feedback(self, mock_provider, sample_reputation_feedback):
        """Test reading a specific feedback entry."""
        provider, mock_w3, mock_contract = mock_provider

        mock_contract.functions.readFeedback.return_value.call.return_value = (
            sample_reputation_feedback["value"],  # value
            0,  # value_decimals
            sample_reputation_feedback["tag"],  # tag1
            "",  # tag2
            False,  # is_revoked
        )

        contract = ReputationRegistryContract(
            provider=provider,
            chain_id=1,
        )
        feedback = contract.read_feedback(
            agent_id=42,
            client_address=sample_reputation_feedback["reporter"],
            feedback_index=0,
        )

        assert feedback.value == sample_reputation_feedback["value"]
        assert feedback.tag1 == sample_reputation_feedback["tag"]
        assert feedback.is_revoked is False

    def test_give_feedback(self, mock_provider):
        """Test submitting reputation feedback."""
        provider, mock_w3, mock_contract = mock_provider

        mock_contract.functions.giveFeedback.return_value.build_transaction.return_value = {
            "to": "0x2222222222222222222222222222222222222222",
            "data": "0x...",
            "gas": 100000,
            "nonce": 0,
        }
        mock_w3.eth.get_transaction_count.return_value = 0

        mock_signer = MagicMock()
        mock_signer.address = "0xSIGNER12345678901234567890123456789012"
        mock_signer.sign_and_send.return_value = "0x" + "ab" * 32

        contract = ReputationRegistryContract(
            provider=provider,
            chain_id=1,
        )
        tx_hash = contract.give_feedback(
            agent_id=42,
            value=100,
            signer=mock_signer,
            tag1="accuracy",
        )

        assert tx_hash is not None
        mock_contract.functions.giveFeedback.assert_called_once()

    def test_revoke_feedback(self, mock_provider):
        """Test revoking reputation feedback."""
        provider, mock_w3, mock_contract = mock_provider

        mock_contract.functions.revokeFeedback.return_value.build_transaction.return_value = {
            "to": "0x2222222222222222222222222222222222222222",
            "data": "0x...",
            "gas": 50000,
            "nonce": 0,
        }
        mock_w3.eth.get_transaction_count.return_value = 0

        mock_signer = MagicMock()
        mock_signer.address = "0xSIGNER12345678901234567890123456789012"
        mock_signer.sign_and_send.return_value = "0x" + "cd" * 32

        contract = ReputationRegistryContract(
            provider=provider,
            chain_id=1,
        )
        tx_hash = contract.revoke_feedback(
            agent_id=42,
            feedback_index=0,
            signer=mock_signer,
        )

        assert tx_hash is not None

    def test_get_clients(self, mock_provider):
        """Test getting client addresses."""
        provider, mock_w3, mock_contract = mock_provider

        expected_clients = [
            "0x1111111111111111111111111111111111111111",
            "0x2222222222222222222222222222222222222222",
        ]
        mock_contract.functions.getClients.return_value.call.return_value = expected_clients

        contract = ReputationRegistryContract(
            provider=provider,
            chain_id=1,
        )
        clients = contract.get_clients(agent_id=42)

        assert clients == expected_clients

    def test_get_last_index(self, mock_provider):
        """Test getting last feedback index."""
        provider, mock_w3, mock_contract = mock_provider

        mock_contract.functions.getLastIndex.return_value.call.return_value = 5

        contract = ReputationRegistryContract(
            provider=provider,
            chain_id=1,
        )
        index = contract.get_last_index(
            agent_id=42,
            client_address="0x1111111111111111111111111111111111111111",
        )

        assert index == 5
