"""
Tests for blockchain contract wrappers (ERC-8004 registries).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aragora.blockchain.config import ChainConfig
from aragora.blockchain.contracts.identity import (
    IDENTITY_REGISTRY_ABI,
    IdentityRegistryContract,
)
from aragora.blockchain.contracts.reputation import (
    REPUTATION_REGISTRY_ABI,
    ReputationRegistryContract,
)
from aragora.blockchain.contracts.validation import (
    VALIDATION_REGISTRY_ABI,
    ValidationRegistryContract,
)
from aragora.blockchain.models import (
    MetadataEntry,
    ValidationResponse,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_provider():
    """Create a mock Web3Provider with full registry config."""
    provider = MagicMock()
    config = ChainConfig(
        chain_id=1,
        rpc_url="https://eth.rpc",
        identity_registry_address="0x1111111111111111111111111111111111111111",
        reputation_registry_address="0x2222222222222222222222222222222222222222",
        validation_registry_address="0x3333333333333333333333333333333333333333",
        gas_limit=500_000,
    )
    provider.get_config.return_value = config

    mock_w3 = MagicMock()
    mock_w3.to_checksum_address.side_effect = lambda x: x
    mock_w3.eth.get_transaction_count.return_value = 5
    provider.get_web3.return_value = mock_w3

    return provider


@pytest.fixture
def mock_signer():
    """Create a mock WalletSigner."""
    signer = MagicMock()
    signer.address = "0xSIGNER1234567890SIGNER1234567890SIGNER12"
    signer.sign_and_send.return_value = "0xTXHASH" + "ab" * 31
    return signer


# ============================================================================
# Identity Registry ABI Tests
# ============================================================================


class TestIdentityRegistryABI:
    """Tests for IDENTITY_REGISTRY_ABI structure."""

    def test_abi_is_list(self):
        assert isinstance(IDENTITY_REGISTRY_ABI, list)

    def test_abi_has_events(self):
        events = [x for x in IDENTITY_REGISTRY_ABI if x.get("type") == "event"]
        assert len(events) >= 3
        event_names = {e["name"] for e in events}
        assert "Registered" in event_names
        assert "URIUpdated" in event_names
        assert "MetadataSet" in event_names

    def test_abi_has_functions(self):
        funcs = [x for x in IDENTITY_REGISTRY_ABI if x.get("type") == "function"]
        assert len(funcs) >= 8
        func_names = {f["name"] for f in funcs}
        assert "register" in func_names
        assert "setAgentURI" in func_names
        assert "getMetadata" in func_names
        assert "setMetadata" in func_names
        assert "tokenURI" in func_names
        assert "ownerOf" in func_names

    def test_register_function_signature(self):
        register = next((f for f in IDENTITY_REGISTRY_ABI if f.get("name") == "register"), None)
        assert register is not None
        assert register["stateMutability"] == "nonpayable"
        assert len(register["inputs"]) == 2
        assert register["inputs"][0]["type"] == "string"
        assert register["inputs"][1]["type"] == "tuple[]"


# ============================================================================
# Identity Registry Contract Tests
# ============================================================================


class TestIdentityRegistryContract:
    """Tests for IdentityRegistryContract class."""

    def test_init(self, mock_provider):
        contract = IdentityRegistryContract(mock_provider)
        assert contract._provider is mock_provider
        assert contract._chain_id is None
        assert contract._contract is None

    def test_init_with_chain_id(self, mock_provider):
        contract = IdentityRegistryContract(mock_provider, chain_id=137)
        assert contract._chain_id == 137

    def test_get_contract_creates_instance(self, mock_provider):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract

        contract = IdentityRegistryContract(mock_provider)
        result = contract._get_contract()
        assert result is mock_contract
        mock_w3.eth.contract.assert_called_once()

    def test_get_contract_caches_instance(self, mock_provider):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract

        contract = IdentityRegistryContract(mock_provider)
        contract._get_contract()
        contract._get_contract()
        assert mock_w3.eth.contract.call_count == 1

    def test_get_contract_no_registry_address(self, mock_provider):
        config = ChainConfig(chain_id=1, rpc_url="https://eth.rpc")
        mock_provider.get_config.return_value = config

        contract = IdentityRegistryContract(mock_provider)
        with pytest.raises(ValueError, match="No Identity Registry address"):
            contract._get_contract()

    def test_register_agent(self, mock_provider, mock_signer):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract

        mock_tx = {"from": mock_signer.address}
        mock_contract.functions.register.return_value.build_transaction.return_value = mock_tx
        mock_receipt = MagicMock()
        mock_w3.eth.wait_for_transaction_receipt.return_value = mock_receipt
        mock_log = {"args": {"agentId": 42}}
        mock_contract.events.Registered.return_value.process_receipt.return_value = [mock_log]

        contract = IdentityRegistryContract(mock_provider)
        agent_id = contract.register_agent("https://example.com/agent.json", mock_signer)

        assert agent_id == 42
        mock_provider.record_success.assert_called()

    def test_register_agent_with_metadata(self, mock_provider, mock_signer):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract

        mock_tx = {"from": mock_signer.address}
        mock_contract.functions.register.return_value.build_transaction.return_value = mock_tx
        mock_receipt = MagicMock()
        mock_w3.eth.wait_for_transaction_receipt.return_value = mock_receipt
        mock_log = {"args": {"agentId": 100}}
        mock_contract.events.Registered.return_value.process_receipt.return_value = [mock_log]

        metadata = [
            MetadataEntry(key="model", value=b"claude-3"),
            MetadataEntry(key="version", value=b"1.0"),
        ]

        contract = IdentityRegistryContract(mock_provider)
        agent_id = contract.register_agent(
            "https://example.com/agent.json", mock_signer, metadata=metadata
        )
        assert agent_id == 100

    def test_register_agent_no_event(self, mock_provider, mock_signer):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract

        mock_tx = {"from": mock_signer.address}
        mock_contract.functions.register.return_value.build_transaction.return_value = mock_tx
        mock_receipt = MagicMock()
        mock_w3.eth.wait_for_transaction_receipt.return_value = mock_receipt
        mock_contract.events.Registered.return_value.process_receipt.return_value = []

        contract = IdentityRegistryContract(mock_provider)
        agent_id = contract.register_agent("https://example.com/agent.json", mock_signer)
        assert agent_id == 0

    def test_get_agent(self, mock_provider):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract

        mock_contract.functions.ownerOf.return_value.call.return_value = "0xOWNER"
        mock_contract.functions.tokenURI.return_value.call.return_value = (
            "https://example.com/agent.json"
        )
        mock_contract.functions.getAgentWallet.return_value.call.return_value = "0xWALLET"

        contract = IdentityRegistryContract(mock_provider)
        identity = contract.get_agent(42)

        assert identity.token_id == 42
        assert identity.owner == "0xOWNER"
        assert identity.agent_uri == "https://example.com/agent.json"
        assert identity.wallet_address == "0xWALLET"

    def test_get_agent_no_wallet(self, mock_provider):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract

        mock_contract.functions.ownerOf.return_value.call.return_value = "0xOWNER"
        mock_contract.functions.tokenURI.return_value.call.return_value = "uri"
        mock_contract.functions.getAgentWallet.return_value.call.return_value = "0x" + "0" * 40

        contract = IdentityRegistryContract(mock_provider)
        identity = contract.get_agent(1)
        assert identity.wallet_address is None

    def test_get_agent_failure(self, mock_provider):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        mock_contract.functions.ownerOf.return_value.call.side_effect = ConnectionError("RPC error")

        contract = IdentityRegistryContract(mock_provider)
        with pytest.raises(RuntimeError, match="Failed to get agent"):
            contract.get_agent(999)
        mock_provider.record_failure.assert_called()

    def test_set_agent_uri(self, mock_provider, mock_signer):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        mock_tx = {"from": mock_signer.address}
        mock_contract.functions.setAgentURI.return_value.build_transaction.return_value = mock_tx

        contract = IdentityRegistryContract(mock_provider)
        tx_hash = contract.set_agent_uri(42, "https://new-uri.com", mock_signer)
        assert tx_hash == mock_signer.sign_and_send.return_value

    def test_get_metadata(self, mock_provider):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        mock_contract.functions.getMetadata.return_value.call.return_value = b"claude-3"

        contract = IdentityRegistryContract(mock_provider)
        result = contract.get_metadata(42, "model")
        assert result == b"claude-3"

    def test_set_metadata(self, mock_provider, mock_signer):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        mock_tx = {"from": mock_signer.address}
        mock_contract.functions.setMetadata.return_value.build_transaction.return_value = mock_tx

        contract = IdentityRegistryContract(mock_provider)
        tx_hash = contract.set_metadata(42, "version", b"2.0", mock_signer)
        assert tx_hash == mock_signer.sign_and_send.return_value

    def test_get_total_supply(self, mock_provider):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        mock_contract.functions.totalSupply.return_value.call.return_value = 1000

        contract = IdentityRegistryContract(mock_provider)
        assert contract.get_total_supply() == 1000


# ============================================================================
# Reputation Registry ABI Tests
# ============================================================================


class TestReputationRegistryABI:
    """Tests for REPUTATION_REGISTRY_ABI structure."""

    def test_abi_is_list(self):
        assert isinstance(REPUTATION_REGISTRY_ABI, list)

    def test_abi_has_events(self):
        events = [x for x in REPUTATION_REGISTRY_ABI if x.get("type") == "event"]
        event_names = {e["name"] for e in events}
        assert "NewFeedback" in event_names
        assert "FeedbackRevoked" in event_names
        assert "ResponseAppended" in event_names

    def test_abi_has_functions(self):
        funcs = [x for x in REPUTATION_REGISTRY_ABI if x.get("type") == "function"]
        func_names = {f["name"] for f in funcs}
        assert "giveFeedback" in func_names
        assert "revokeFeedback" in func_names
        assert "getSummary" in func_names
        assert "readFeedback" in func_names
        assert "getClients" in func_names


# ============================================================================
# Reputation Registry Contract Tests
# ============================================================================


class TestReputationRegistryContract:
    """Tests for ReputationRegistryContract class."""

    def test_init(self, mock_provider):
        contract = ReputationRegistryContract(mock_provider)
        assert contract._provider is mock_provider
        assert contract._contract is None

    def test_get_contract_no_registry_address(self, mock_provider):
        config = ChainConfig(chain_id=1, rpc_url="https://eth.rpc")
        mock_provider.get_config.return_value = config

        contract = ReputationRegistryContract(mock_provider)
        with pytest.raises(ValueError, match="No Reputation Registry address"):
            contract._get_contract()

    def test_give_feedback(self, mock_provider, mock_signer):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        mock_tx = {"from": mock_signer.address}
        mock_contract.functions.giveFeedback.return_value.build_transaction.return_value = mock_tx

        contract = ReputationRegistryContract(mock_provider)
        tx_hash = contract.give_feedback(
            agent_id=42, value=100, signer=mock_signer, tag1="accuracy", tag2="reasoning"
        )
        assert tx_hash == mock_signer.sign_and_send.return_value
        mock_provider.record_success.assert_called()

    def test_give_feedback_with_all_params(self, mock_provider, mock_signer):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        mock_tx = {"from": mock_signer.address}
        mock_contract.functions.giveFeedback.return_value.build_transaction.return_value = mock_tx

        contract = ReputationRegistryContract(mock_provider)
        contract.give_feedback(
            agent_id=42,
            value=12345,
            signer=mock_signer,
            value_decimals=2,
            tag1="accuracy",
            tag2="code",
            endpoint="/api/v1/chat",
            feedback_uri="ipfs://QmTest",
            feedback_hash=b"\x01" * 32,
        )
        mock_contract.functions.giveFeedback.assert_called_once()

    def test_revoke_feedback(self, mock_provider, mock_signer):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        mock_tx = {"from": mock_signer.address}
        mock_contract.functions.revokeFeedback.return_value.build_transaction.return_value = mock_tx

        contract = ReputationRegistryContract(mock_provider)
        tx_hash = contract.revoke_feedback(agent_id=42, feedback_index=3, signer=mock_signer)
        assert tx_hash == mock_signer.sign_and_send.return_value

    def test_get_summary(self, mock_provider):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        mock_contract.functions.getSummary.return_value.call.return_value = (15, 850, 0)

        contract = ReputationRegistryContract(mock_provider)
        summary = contract.get_summary(agent_id=42, tag1="accuracy")
        assert summary.agent_id == 42
        assert summary.count == 15
        assert summary.summary_value == 850
        assert summary.tag1 == "accuracy"

    def test_get_summary_with_client_filter(self, mock_provider):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        mock_contract.functions.getSummary.return_value.call.return_value = (5, 100, 2)

        contract = ReputationRegistryContract(mock_provider)
        summary = contract.get_summary(agent_id=42, client_addresses=["0xClient1", "0xClient2"])
        assert summary.count == 5
        assert summary.normalized_value == 1.0  # 100 / 10^2

    def test_read_feedback(self, mock_provider):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        mock_contract.functions.readFeedback.return_value.call.return_value = (
            75,
            0,
            "accuracy",
            "code",
            False,
        )

        contract = ReputationRegistryContract(mock_provider)
        feedback = contract.read_feedback(agent_id=42, client_address="0xClient", feedback_index=0)
        assert feedback.agent_id == 42
        assert feedback.value == 75
        assert feedback.tag1 == "accuracy"
        assert feedback.is_revoked is False

    def test_read_feedback_revoked(self, mock_provider):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        mock_contract.functions.readFeedback.return_value.call.return_value = (
            -50,
            0,
            "reliability",
            "",
            True,
        )

        contract = ReputationRegistryContract(mock_provider)
        feedback = contract.read_feedback(42, "0xClient", 5)
        assert feedback.is_revoked is True
        assert feedback.value == -50

    def test_get_clients(self, mock_provider):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        mock_contract.functions.getClients.return_value.call.return_value = [
            "0xClient1",
            "0xClient2",
            "0xClient3",
        ]

        contract = ReputationRegistryContract(mock_provider)
        clients = contract.get_clients(agent_id=42)
        assert len(clients) == 3
        assert "0xClient1" in clients

    def test_get_last_index(self, mock_provider):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        mock_contract.functions.getLastIndex.return_value.call.return_value = 7

        contract = ReputationRegistryContract(mock_provider)
        assert contract.get_last_index(agent_id=42, client_address="0xClient") == 7


# ============================================================================
# Validation Registry ABI Tests
# ============================================================================


class TestValidationRegistryABI:
    """Tests for VALIDATION_REGISTRY_ABI structure."""

    def test_abi_is_list(self):
        assert isinstance(VALIDATION_REGISTRY_ABI, list)

    def test_abi_has_events(self):
        events = [x for x in VALIDATION_REGISTRY_ABI if x.get("type") == "event"]
        event_names = {e["name"] for e in events}
        assert "ValidationRequest" in event_names
        assert "ValidationResponse" in event_names

    def test_abi_has_functions(self):
        funcs = [x for x in VALIDATION_REGISTRY_ABI if x.get("type") == "function"]
        func_names = {f["name"] for f in funcs}
        assert "validationRequest" in func_names
        assert "validationResponse" in func_names
        assert "getValidationStatus" in func_names
        assert "getSummary" in func_names


# ============================================================================
# Validation Registry Contract Tests
# ============================================================================


class TestValidationRegistryContract:
    """Tests for ValidationRegistryContract class."""

    def test_init(self, mock_provider):
        contract = ValidationRegistryContract(mock_provider)
        assert contract._provider is mock_provider
        assert contract._contract is None

    def test_get_contract_no_registry_address(self, mock_provider):
        config = ChainConfig(chain_id=1, rpc_url="https://eth.rpc")
        mock_provider.get_config.return_value = config

        contract = ValidationRegistryContract(mock_provider)
        with pytest.raises(ValueError, match="No Validation Registry address"):
            contract._get_contract()

    def test_request_validation(self, mock_provider, mock_signer):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        mock_tx = {"from": mock_signer.address}
        mock_contract.functions.validationRequest.return_value.build_transaction.return_value = (
            mock_tx
        )

        contract = ValidationRegistryContract(mock_provider)
        tx_hash = contract.request_validation(
            validator_address="0xValidator123",
            agent_id=42,
            request_uri="ipfs://QmRequest",
            request_hash=b"\x01" * 32,
            signer=mock_signer,
        )
        assert tx_hash == mock_signer.sign_and_send.return_value
        mock_provider.record_success.assert_called()

    def test_submit_response(self, mock_provider, mock_signer):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        mock_tx = {"from": mock_signer.address}
        mock_contract.functions.validationResponse.return_value.build_transaction.return_value = (
            mock_tx
        )

        contract = ValidationRegistryContract(mock_provider)
        tx_hash = contract.submit_response(
            request_hash=b"\x01" * 32,
            response=ValidationResponse.PASS,
            response_uri="ipfs://QmResponse",
            response_hash=b"\x02" * 32,
            tag="capability",
            signer=mock_signer,
        )
        assert tx_hash == mock_signer.sign_and_send.return_value

    def test_submit_response_fail(self, mock_provider, mock_signer):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        mock_tx = {"from": mock_signer.address}
        mock_contract.functions.validationResponse.return_value.build_transaction.return_value = (
            mock_tx
        )

        contract = ValidationRegistryContract(mock_provider)
        contract.submit_response(
            request_hash=b"\x01" * 32,
            response=ValidationResponse.FAIL,
            response_uri="",
            response_hash=b"\x00" * 32,
            tag="safety",
            signer=mock_signer,
        )
        call_args = mock_contract.functions.validationResponse.call_args[0]
        assert call_args[1] == 2  # ValidationResponse.FAIL.value

    def test_get_validation_status(self, mock_provider):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract

        timestamp = 1700000000
        mock_contract.functions.getValidationStatus.return_value.call.return_value = (
            "0xValidator123",
            42,
            1,
            b"\xab" * 32,
            "capability",
            timestamp,
        )

        contract = ValidationRegistryContract(mock_provider)
        record = contract.get_validation_status(b"\x01" * 32)
        assert record.agent_id == 42
        assert record.validator_address == "0xValidator123"
        assert record.response == ValidationResponse.PASS
        assert record.tag == "capability"
        assert record.last_update is not None

    def test_get_validation_status_pending(self, mock_provider):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract

        mock_contract.functions.getValidationStatus.return_value.call.return_value = (
            "0xValidator",
            42,
            0,
            b"\x00" * 32,
            "",
            0,
        )

        contract = ValidationRegistryContract(mock_provider)
        record = contract.get_validation_status(b"\x01" * 32)
        assert record.response == ValidationResponse.PENDING
        assert record.is_pending is True
        assert record.last_update is None

    def test_get_summary(self, mock_provider):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        mock_contract.functions.getSummary.return_value.call.return_value = (10, 1)

        contract = ValidationRegistryContract(mock_provider)
        summary = contract.get_summary(agent_id=42, tag="safety")
        assert summary.agent_id == 42
        assert summary.count == 10
        assert summary.average_response == 1
        assert summary.tag == "safety"

    def test_get_summary_with_validators_filter(self, mock_provider):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        mock_contract.functions.getSummary.return_value.call.return_value = (5, 1)

        contract = ValidationRegistryContract(mock_provider)
        summary = contract.get_summary(
            agent_id=42, validator_addresses=["0xValidator1", "0xValidator2"]
        )
        assert summary.count == 5

    def test_get_agent_validations(self, mock_provider):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        mock_contract.functions.getAgentValidations.return_value.call.return_value = [
            b"\x01" * 32,
            b"\x02" * 32,
            b"\x03" * 32,
        ]

        contract = ValidationRegistryContract(mock_provider)
        hashes = contract.get_agent_validations(agent_id=42)
        assert len(hashes) == 3

    def test_get_validator_requests(self, mock_provider):
        mock_w3 = mock_provider.get_web3.return_value
        mock_contract = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        mock_contract.functions.getValidatorRequests.return_value.call.return_value = [
            b"\x01" * 32,
            b"\x02" * 32,
        ]

        contract = ValidationRegistryContract(mock_provider)
        hashes = contract.get_validator_requests("0xValidator123")
        assert len(hashes) == 2
        mock_provider.record_success.assert_called()
