"""
Tests for Validation Registry contract wrapper.
"""

from __future__ import annotations

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
        """Test ABI includes requestValidation function."""
        function_names = [
            item.get("name") for item in VALIDATION_REGISTRY_ABI if item.get("type") == "function"
        ]
        assert "requestValidation" in function_names

    def test_abi_has_submit_response(self):
        """Test ABI includes submitResponse function."""
        function_names = [
            item.get("name") for item in VALIDATION_REGISTRY_ABI if item.get("type") == "function"
        ]
        assert "submitResponse" in function_names

    def test_abi_has_events(self):
        """Test ABI includes events."""
        event_names = [
            item.get("name") for item in VALIDATION_REGISTRY_ABI if item.get("type") == "event"
        ]
        assert "ValidationRequest" in event_names
        assert "ValidationResponse" in event_names


class TestValidationRegistryContract:
    """Tests for ValidationRegistryContract wrapper."""

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
        contract = ValidationRegistryContract(
            provider=provider,
            address=mainnet_config["validation_registry_address"],
            chain_id=1,
        )
        assert contract.address == mainnet_config["validation_registry_address"]

    def test_get_validation_count(self, mock_provider):
        """Test getting validation count for an agent."""
        provider, _, mock_contract = mock_provider
        mock_contract.functions.validationCount.return_value.call.return_value = 10

        contract = ValidationRegistryContract(
            provider=provider,
            address="0x3333333333333333333333333333333333333333",
            chain_id=1,
        )
        count = contract.get_validation_count(token_id=42)

        assert count == 10

    def test_get_validation_status(self, mock_provider, sample_validation_record):
        """Test getting validation status."""
        provider, _, mock_contract = mock_provider
        mock_contract.functions.getValidation.return_value.call.return_value = (
            sample_validation_record["validator"],
            sample_validation_record["request_type"],
            sample_validation_record["response_code"],
            sample_validation_record["evidence_uri"],
        )

        contract = ValidationRegistryContract(
            provider=provider,
            address="0x3333333333333333333333333333333333333333",
            chain_id=1,
        )
        record = contract.get_validation_status(token_id=42, index=0)

        assert record.validator == sample_validation_record["validator"]
        assert record.response_code == 1  # PASS

    def test_get_all_validations(self, mock_provider, sample_validation_records):
        """Test getting all validations for an agent."""
        provider, _, mock_contract = mock_provider
        mock_contract.functions.validationCount.return_value.call.return_value = 3

        def mock_get_validation(token_id, index):
            vr = sample_validation_records[index]
            mock_result = MagicMock()
            mock_result.call.return_value = (
                vr["validator"],
                vr["request_type"],
                vr.get("response_code", 0),
                vr.get("evidence_uri", ""),
            )
            return mock_result

        mock_contract.functions.getValidation.side_effect = mock_get_validation

        contract = ValidationRegistryContract(
            provider=provider,
            address="0x3333333333333333333333333333333333333333",
            chain_id=1,
        )
        validations = contract.get_all_validations(token_id=42)

        assert len(validations) == 3
        assert validations[0].request_type == "capability"
        assert validations[1].request_type == "safety"

    def test_request_validation(self, mock_provider):
        """Test requesting a new validation."""
        provider, mock_w3, mock_contract = mock_provider

        mock_contract.functions.requestValidation.return_value.build_transaction.return_value = {
            "to": "0x3333333333333333333333333333333333333333",
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

        contract = ValidationRegistryContract(
            provider=provider,
            address="0x3333333333333333333333333333333333333333",
            chain_id=1,
        )
        tx_hash = contract.request_validation(
            token_id=42,
            validator="0xVALIDATOR123456789012345678901234567890",
            request_type="capability",
            signer=mock_signer,
        )

        assert tx_hash is not None
        mock_contract.functions.requestValidation.assert_called_once()

    def test_submit_response(self, mock_provider):
        """Test submitting validation response."""
        provider, mock_w3, mock_contract = mock_provider

        mock_contract.functions.submitResponse.return_value.build_transaction.return_value = {
            "to": "0x3333333333333333333333333333333333333333",
            "data": "0x...",
            "gas": 80000,
            "nonce": 0,
        }
        mock_w3.eth.send_raw_transaction.return_value = b"\xcd" * 32
        mock_w3.eth.wait_for_transaction_receipt.return_value = {"status": 1}

        mock_signer = MagicMock()
        mock_signer.address = "0xVALIDATOR123456789012345678901234567890"
        mock_signed = MagicMock()
        mock_signed.raw_transaction = b"\x00" * 100
        mock_signer.sign_transaction.return_value = mock_signed

        contract = ValidationRegistryContract(
            provider=provider,
            address="0x3333333333333333333333333333333333333333",
            chain_id=1,
        )
        tx_hash = contract.submit_response(
            token_id=42,
            index=0,
            response_code=ValidationResponse.PASS,
            evidence_uri="ipfs://QmEvidence",
            signer=mock_signer,
        )

        assert tx_hash is not None

    def test_get_summary(self, mock_provider, sample_validation_summary):
        """Test getting validation summary."""
        provider, _, mock_contract = mock_provider

        mock_contract.functions.validationCount.return_value.call.return_value = 10

        contract = ValidationRegistryContract(
            provider=provider,
            address="0x3333333333333333333333333333333333333333",
            chain_id=1,
        )
        summary = contract.get_summary(token_id=42)

        assert summary.token_id == 42
        assert summary.total_validations == 10

    def test_check_is_validated(self, mock_provider):
        """Test checking if agent is validated."""
        provider, _, mock_contract = mock_provider

        # Mock that agent has passed validations
        mock_contract.functions.validationCount.return_value.call.return_value = 2
        mock_contract.functions.getValidation.side_effect = [
            MagicMock(
                call=MagicMock(
                    return_value=(
                        "0xVAL1",
                        "capability",
                        1,  # PASS
                        "ipfs://ev1",
                    )
                )
            ),
            MagicMock(
                call=MagicMock(
                    return_value=(
                        "0xVAL2",
                        "safety",
                        1,  # PASS
                        "ipfs://ev2",
                    )
                )
            ),
        ]

        contract = ValidationRegistryContract(
            provider=provider,
            address="0x3333333333333333333333333333333333333333",
            chain_id=1,
        )
        is_validated = contract.is_validated(token_id=42, validation_type="capability")

        assert is_validated is True
