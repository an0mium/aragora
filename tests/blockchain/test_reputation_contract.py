"""
Tests for Reputation Registry contract wrapper.
"""

from __future__ import annotations

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


class TestReputationRegistryContract:
    """Tests for ReputationRegistryContract wrapper."""

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
        contract = ReputationRegistryContract(
            provider=provider,
            address=mainnet_config["reputation_registry_address"],
            chain_id=1,
        )
        assert contract.address == mainnet_config["reputation_registry_address"]

    def test_get_feedback_count(self, mock_provider):
        """Test getting feedback count for an agent."""
        provider, _, mock_contract = mock_provider
        mock_contract.functions.feedbackCount.return_value.call.return_value = 15

        contract = ReputationRegistryContract(
            provider=provider,
            address="0x2222222222222222222222222222222222222222",
            chain_id=1,
        )
        count = contract.get_feedback_count(token_id=42)

        assert count == 15

    def test_read_feedback(self, mock_provider, sample_reputation_feedback):
        """Test reading a specific feedback entry."""
        provider, _, mock_contract = mock_provider
        mock_contract.functions.readFeedback.return_value.call.return_value = (
            sample_reputation_feedback["reporter"],
            sample_reputation_feedback["value"],
            sample_reputation_feedback["tag"],
            sample_reputation_feedback["comment"],
            False,  # not revoked
        )

        contract = ReputationRegistryContract(
            provider=provider,
            address="0x2222222222222222222222222222222222222222",
            chain_id=1,
        )
        feedback = contract.read_feedback(token_id=42, index=0)

        assert feedback.reporter == sample_reputation_feedback["reporter"]
        assert feedback.value == sample_reputation_feedback["value"]
        assert feedback.tag == sample_reputation_feedback["tag"]

    def test_get_all_feedback(self, mock_provider, sample_reputation_feedbacks):
        """Test getting all feedback for an agent."""
        provider, _, mock_contract = mock_provider
        mock_contract.functions.feedbackCount.return_value.call.return_value = 3

        # Set up individual feedback reads
        def mock_read(token_id, index):
            fb = sample_reputation_feedbacks[index]
            mock_result = MagicMock()
            mock_result.call.return_value = (
                fb["reporter"],
                fb["value"],
                fb["tag"],
                fb.get("comment", ""),
                fb.get("revoked", False),
            )
            return mock_result

        mock_contract.functions.readFeedback.side_effect = mock_read

        contract = ReputationRegistryContract(
            provider=provider,
            address="0x2222222222222222222222222222222222222222",
            chain_id=1,
        )
        feedbacks = contract.get_all_feedback(token_id=42)

        assert len(feedbacks) == 3
        assert feedbacks[0].value == 100
        assert feedbacks[1].value == -50

    def test_give_feedback(self, mock_provider):
        """Test submitting reputation feedback."""
        provider, mock_w3, mock_contract = mock_provider

        mock_contract.functions.giveFeedback.return_value.build_transaction.return_value = {
            "to": "0x2222222222222222222222222222222222222222",
            "data": "0x...",
            "gas": 100000,
            "nonce": 0,
        }
        mock_w3.eth.send_raw_transaction.return_value = b"\xab" * 32
        mock_w3.eth.wait_for_transaction_receipt.return_value = {"status": 1}

        mock_signer = MagicMock()
        mock_signer.address = "0xSIGNER12345678901234567890123456789012"
        mock_signed = MagicMock()
        mock_signed.raw_transaction = b"\x00" * 100
        mock_signer.sign_transaction.return_value = mock_signed

        contract = ReputationRegistryContract(
            provider=provider,
            address="0x2222222222222222222222222222222222222222",
            chain_id=1,
        )
        tx_hash = contract.give_feedback(
            token_id=42,
            value=100,
            tag="accuracy",
            comment="Great performance",
            signer=mock_signer,
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
        mock_w3.eth.send_raw_transaction.return_value = b"\xcd" * 32
        mock_w3.eth.wait_for_transaction_receipt.return_value = {"status": 1}

        mock_signer = MagicMock()
        mock_signer.address = "0xSIGNER12345678901234567890123456789012"
        mock_signed = MagicMock()
        mock_signed.raw_transaction = b"\x00" * 100
        mock_signer.sign_transaction.return_value = mock_signed

        contract = ReputationRegistryContract(
            provider=provider,
            address="0x2222222222222222222222222222222222222222",
            chain_id=1,
        )
        tx_hash = contract.revoke_feedback(
            token_id=42,
            index=0,
            signer=mock_signer,
        )

        assert tx_hash is not None

    def test_get_summary(self, mock_provider, sample_reputation_summary):
        """Test getting reputation summary."""
        provider, _, mock_contract = mock_provider

        # Mock summary call
        mock_contract.functions.feedbackCount.return_value.call.return_value = 15

        contract = ReputationRegistryContract(
            provider=provider,
            address="0x2222222222222222222222222222222222222222",
            chain_id=1,
        )
        summary = contract.get_summary(token_id=42)

        assert summary.token_id == 42
        assert summary.total_feedback_count == 15
