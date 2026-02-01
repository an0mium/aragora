"""
Validation Registry contract wrapper for ERC-8004.

Wraps the IValidationRegistry Solidity interface with typed Python methods.
Handles validation requests, responses, and status queries.

Key functions:
    validationRequest(validatorAddress, agentId, requestURI, requestHash)
    validationResponse(requestHash, response, responseURI, responseHash, tag)
    getValidationStatus(requestHash) -> (validator, agentId, response, responseHash, tag, lastUpdate)
    getSummary(agentId, validatorAddresses, tag) -> (count, averageResponse)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from aragora.blockchain.models import (
    ValidationRecord,
    ValidationResponse,
    ValidationSummary,
)

logger = logging.getLogger(__name__)

VALIDATION_REGISTRY_ABI: list[dict[str, Any]] = [
    # Events
    {
        "type": "event",
        "name": "ValidationRequest",
        "inputs": [
            {"name": "validatorAddress", "type": "address", "indexed": True},
            {"name": "agentId", "type": "uint256", "indexed": True},
            {"name": "requestURI", "type": "string", "indexed": False},
            {"name": "requestHash", "type": "bytes32", "indexed": True},
        ],
    },
    {
        "type": "event",
        "name": "ValidationResponse",
        "inputs": [
            {"name": "validatorAddress", "type": "address", "indexed": True},
            {"name": "agentId", "type": "uint256", "indexed": True},
            {"name": "requestHash", "type": "bytes32", "indexed": True},
            {"name": "response", "type": "uint8", "indexed": False},
            {"name": "responseURI", "type": "string", "indexed": False},
            {"name": "responseHash", "type": "bytes32", "indexed": False},
            {"name": "tag", "type": "string", "indexed": False},
        ],
    },
    # validationRequest
    {
        "type": "function",
        "name": "validationRequest",
        "inputs": [
            {"name": "validatorAddress", "type": "address"},
            {"name": "agentId", "type": "uint256"},
            {"name": "requestURI", "type": "string"},
            {"name": "requestHash", "type": "bytes32"},
        ],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    # validationResponse
    {
        "type": "function",
        "name": "validationResponse",
        "inputs": [
            {"name": "requestHash", "type": "bytes32"},
            {"name": "response", "type": "uint8"},
            {"name": "responseURI", "type": "string"},
            {"name": "responseHash", "type": "bytes32"},
            {"name": "tag", "type": "string"},
        ],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    # getValidationStatus
    {
        "type": "function",
        "name": "getValidationStatus",
        "inputs": [{"name": "requestHash", "type": "bytes32"}],
        "outputs": [
            {"name": "validatorAddress", "type": "address"},
            {"name": "agentId", "type": "uint256"},
            {"name": "response", "type": "uint8"},
            {"name": "responseHash", "type": "bytes32"},
            {"name": "tag", "type": "string"},
            {"name": "lastUpdate", "type": "uint256"},
        ],
        "stateMutability": "view",
    },
    # getSummary
    {
        "type": "function",
        "name": "getSummary",
        "inputs": [
            {"name": "agentId", "type": "uint256"},
            {"name": "validatorAddresses", "type": "address[]"},
            {"name": "tag", "type": "string"},
        ],
        "outputs": [
            {"name": "count", "type": "uint64"},
            {"name": "averageResponse", "type": "uint8"},
        ],
        "stateMutability": "view",
    },
    # getAgentValidations
    {
        "type": "function",
        "name": "getAgentValidations",
        "inputs": [{"name": "agentId", "type": "uint256"}],
        "outputs": [{"name": "requestHashes", "type": "bytes32[]"}],
        "stateMutability": "view",
    },
    # getValidatorRequests
    {
        "type": "function",
        "name": "getValidatorRequests",
        "inputs": [{"name": "validatorAddress", "type": "address"}],
        "outputs": [{"name": "requestHashes", "type": "bytes32[]"}],
        "stateMutability": "view",
    },
    # getIdentityRegistry
    {
        "type": "function",
        "name": "getIdentityRegistry",
        "inputs": [],
        "outputs": [{"name": "identityRegistry", "type": "address"}],
        "stateMutability": "view",
    },
]


class ValidationRegistryContract:
    """Typed wrapper around the ERC-8004 Validation Registry contract.

    Provides methods for requesting validations, submitting responses,
    and querying validation status.
    """

    def __init__(self, provider: Any, chain_id: int | None = None) -> None:
        self._provider = provider
        self._chain_id = chain_id
        self._contract: Any = None

    def _get_contract(self) -> Any:
        if self._contract is None:
            w3 = self._provider.get_web3(self._chain_id)
            config = self._provider.get_config(self._chain_id)
            if not config.has_validation_registry:
                raise ValueError(
                    f"No Validation Registry address configured for chain {config.chain_id}. "
                    "Set ERC8004_VALIDATION_REGISTRY environment variable."
                )
            self._contract = w3.eth.contract(
                address=w3.to_checksum_address(config.validation_registry_address),
                abi=VALIDATION_REGISTRY_ABI,
            )
        return self._contract

    def request_validation(
        self,
        validator_address: str,
        agent_id: int,
        request_uri: str,
        request_hash: bytes,
        signer: Any,
    ) -> str:
        """Submit a validation request for an agent.

        Args:
            validator_address: Address of the validator to request.
            agent_id: Token ID of the agent to validate.
            request_uri: URI with validation request details.
            request_hash: Hash of the request content.
            signer: WalletSigner for signing.

        Returns:
            Transaction hash.
        """
        contract = self._get_contract()
        w3 = self._provider.get_web3(self._chain_id)
        config = self._provider.get_config(self._chain_id)

        tx = contract.functions.validationRequest(
            validator_address, agent_id, request_uri, request_hash
        ).build_transaction(
            {
                "from": signer.address,
                "gas": config.gas_limit,
                "nonce": w3.eth.get_transaction_count(signer.address),
            }
        )

        tx_hash = signer.sign_and_send(w3, tx)
        self._provider.record_success(self._chain_id)
        return tx_hash

    def submit_response(
        self,
        request_hash: bytes,
        response: ValidationResponse,
        response_uri: str,
        response_hash: bytes,
        tag: str,
        signer: Any,
    ) -> str:
        """Submit a validation response.

        Args:
            request_hash: Hash identifying the validation request.
            response: Response code (PASS, FAIL, REVOKED).
            response_uri: URI to detailed response data.
            response_hash: Hash of the response content.
            tag: Category tag for the validation.
            signer: WalletSigner for signing.

        Returns:
            Transaction hash.
        """
        contract = self._get_contract()
        w3 = self._provider.get_web3(self._chain_id)
        config = self._provider.get_config(self._chain_id)

        tx = contract.functions.validationResponse(
            request_hash, response.value, response_uri, response_hash, tag
        ).build_transaction(
            {
                "from": signer.address,
                "gas": config.gas_limit,
                "nonce": w3.eth.get_transaction_count(signer.address),
            }
        )

        tx_hash = signer.sign_and_send(w3, tx)
        self._provider.record_success(self._chain_id)
        return tx_hash

    def get_validation_status(self, request_hash: bytes) -> ValidationRecord:
        """Get the current status of a validation request.

        Args:
            request_hash: Hash identifying the validation request.

        Returns:
            ValidationRecord with current status.
        """
        contract = self._get_contract()

        validator, agent_id, response, resp_hash, tag, last_update = (
            contract.functions.getValidationStatus(request_hash).call()
        )

        self._provider.record_success(self._chain_id)

        last_update_dt = None
        if last_update > 0:
            last_update_dt = datetime.fromtimestamp(last_update, tz=timezone.utc)

        return ValidationRecord(
            request_hash=request_hash.hex() if isinstance(request_hash, bytes) else request_hash,
            agent_id=agent_id,
            validator_address=validator,
            response=ValidationResponse(response),
            response_hash=resp_hash.hex() if isinstance(resp_hash, bytes) else resp_hash,
            tag=tag,
            last_update=last_update_dt,
        )

    def get_summary(
        self,
        agent_id: int,
        validator_addresses: list[str] | None = None,
        tag: str = "",
    ) -> ValidationSummary:
        """Get aggregated validation summary for an agent.

        Args:
            agent_id: Token ID of the agent.
            validator_addresses: Filter by validators. None = all.
            tag: Filter by tag.

        Returns:
            ValidationSummary with count and average response.
        """
        contract = self._get_contract()
        addresses = validator_addresses or []

        count, avg_response = contract.functions.getSummary(agent_id, addresses, tag).call()

        self._provider.record_success(self._chain_id)
        return ValidationSummary(
            agent_id=agent_id,
            count=count,
            average_response=avg_response,
            tag=tag,
        )

    def get_agent_validations(self, agent_id: int) -> list[bytes]:
        """Get all validation request hashes for an agent.

        Args:
            agent_id: Token ID of the agent.

        Returns:
            List of request hashes.
        """
        contract = self._get_contract()
        result = contract.functions.getAgentValidations(agent_id).call()
        self._provider.record_success(self._chain_id)
        return result

    def get_validator_requests(self, validator_address: str) -> list[bytes]:
        """Get all validation request hashes assigned to a validator.

        Args:
            validator_address: Validator's Ethereum address.

        Returns:
            List of request hashes.
        """
        contract = self._get_contract()
        result = contract.functions.getValidatorRequests(validator_address).call()
        self._provider.record_success(self._chain_id)
        return result


__all__ = [
    "VALIDATION_REGISTRY_ABI",
    "ValidationRegistryContract",
]
