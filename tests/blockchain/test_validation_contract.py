"""
Tests for Validation Registry contract wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, AsyncMock
import pytest

from aragora.blockchain.contracts.validation import (
    ValidationRegistryContract,
    VALIDATION_REGISTRY_ABI,
)
from aragora.blockchain.models import ValidationResponse


class TestValidationRegistryABI:
    """Tests for Validation Registry ABI."""

    def test_abi_exists(self):
        """Test that ABI is defined."""
        assert VALIDATION_REGISTRY_ABI is not None
        assert len(VALIDATION_REGISTRY_ABI) > 0

    def test_abi_has_request_validation(self):
        """Test ABI includes validationRequest function."""
        function_names = [
            item.get("name") for item in VALIDATION_REGISTRY_ABI if item.get("type") == "function"
        ]
        assert "validationRequest" in function_names

    def test_abi_has_submit_response(self):
        """Test ABI includes validationResponse function."""
        function_names = [
            item.get("name") for item in VALIDATION_REGISTRY_ABI if item.get("type") == "function"
        ]
        assert "validationResponse" in function_names

    def test_abi_has_events(self):
        """Test ABI includes events."""
        event_names = [
            item.get("name") for item in VALIDATION_REGISTRY_ABI if item.get("type") == "event"
        ]
        assert "ValidationRequest" in event_names
        assert "ValidationResponse" in event_names


@dataclass
class MockChainConfig:
    """Mock chain configuration."""

    chain_id: int = 1
    validation_registry_address: str = "0x3333333333333333333333333333333333333333"
    has_validation_registry: bool = True
    gas_limit: int = 500_000


class TestValidationRegistryContract:
    """Tests for ValidationRegistryContract wrapper."""

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
        contract = ValidationRegistryContract(
            provider=provider,
            chain_id=1,
        )
        # Contract is lazy-loaded, so we just verify the object is created
        assert contract._provider is provider
        assert contract._chain_id == 1

    def test_get_validation_status(self, mock_provider):
        """Test getting validation status."""
        provider, mock_w3, mock_contract = mock_provider

        request_hash = b"\xab" * 32
        mock_contract.functions.getValidationStatus.return_value.call.return_value = (
            "0xVALIDATOR123456789012345678901234567890",  # validator
            42,  # agent_id
            1,  # response (PASS)
            b"\xcd" * 32,  # response_hash
            "capability",  # tag
            1704067200,  # last_update (timestamp)
        )

        contract = ValidationRegistryContract(
            provider=provider,
            chain_id=1,
        )
        record = contract.get_validation_status(request_hash=request_hash)

        assert record.validator_address == "0xVALIDATOR123456789012345678901234567890"
        assert record.agent_id == 42
        assert record.response == ValidationResponse.PASS
        assert record.tag == "capability"

    def test_get_summary(self, mock_provider):
        """Test getting validation summary."""
        provider, mock_w3, mock_contract = mock_provider

        mock_contract.functions.getSummary.return_value.call.return_value = (
            10,  # count
            1,  # average_response
        )

        contract = ValidationRegistryContract(
            provider=provider,
            chain_id=1,
        )
        summary = contract.get_summary(agent_id=42)

        assert summary.agent_id == 42
        assert summary.count == 10
        assert summary.average_response == 1

    def test_request_validation(self, mock_provider):
        """Test requesting a new validation."""
        provider, mock_w3, mock_contract = mock_provider

        mock_contract.functions.validationRequest.return_value.build_transaction.return_value = {
            "to": "0x3333333333333333333333333333333333333333",
            "data": "0x...",
            "gas": 100000,
            "nonce": 0,
        }
        mock_w3.eth.get_transaction_count.return_value = 0

        mock_signer = MagicMock()
        mock_signer.address = "0xSIGNER12345678901234567890123456789012"
        mock_signer.sign_and_send.return_value = "0x" + "ab" * 32

        contract = ValidationRegistryContract(
            provider=provider,
            chain_id=1,
        )
        tx_hash = contract.request_validation(
            validator_address="0xVALIDATOR123456789012345678901234567890",
            agent_id=42,
            request_uri="ipfs://QmRequest123",
            request_hash=b"\xab" * 32,
            signer=mock_signer,
        )

        assert tx_hash is not None
        mock_contract.functions.validationRequest.assert_called_once()

    def test_submit_response(self, mock_provider):
        """Test submitting validation response."""
        provider, mock_w3, mock_contract = mock_provider

        mock_contract.functions.validationResponse.return_value.build_transaction.return_value = {
            "to": "0x3333333333333333333333333333333333333333",
            "data": "0x...",
            "gas": 80000,
            "nonce": 0,
        }
        mock_w3.eth.get_transaction_count.return_value = 0

        mock_signer = MagicMock()
        mock_signer.address = "0xVALIDATOR123456789012345678901234567890"
        mock_signer.sign_and_send.return_value = "0x" + "cd" * 32

        contract = ValidationRegistryContract(
            provider=provider,
            chain_id=1,
        )
        tx_hash = contract.submit_response(
            request_hash=b"\xab" * 32,
            response=ValidationResponse.PASS,
            response_uri="ipfs://QmResponse123",
            response_hash=b"\xcd" * 32,
            tag="capability",
            signer=mock_signer,
        )

        assert tx_hash is not None

    def test_get_agent_validations(self, mock_provider):
        """Test getting agent validation hashes."""
        provider, mock_w3, mock_contract = mock_provider

        expected_hashes = [b"\xab" * 32, b"\xcd" * 32]
        mock_contract.functions.getAgentValidations.return_value.call.return_value = expected_hashes

        contract = ValidationRegistryContract(
            provider=provider,
            chain_id=1,
        )
        hashes = contract.get_agent_validations(agent_id=42)

        assert hashes == expected_hashes

    def test_get_validator_requests(self, mock_provider):
        """Test getting validator request hashes."""
        provider, mock_w3, mock_contract = mock_provider

        expected_hashes = [b"\xef" * 32]
        mock_contract.functions.getValidatorRequests.return_value.call.return_value = (
            expected_hashes
        )

        contract = ValidationRegistryContract(
            provider=provider,
            chain_id=1,
        )
        hashes = contract.get_validator_requests(
            validator_address="0xVALIDATOR123456789012345678901234567890"
        )

        assert hashes == expected_hashes
