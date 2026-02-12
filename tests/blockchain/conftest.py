"""
Pytest fixtures for ERC-8004 blockchain tests.

Provides mock Web3 providers, contract instances, and blockchain data
for testing without external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Mock Web3 Types
# =============================================================================


class MockWeb3:
    """Mock Web3 instance for testing without blockchain connection."""

    def __init__(self, chain_id: int = 1, is_connected: bool = True):
        self.eth = MockEth(chain_id)
        self._is_connected = is_connected

    def is_connected(self) -> bool:
        return self._is_connected


class MockEth:
    """Mock eth module."""

    def __init__(self, chain_id: int = 1):
        self.chain_id = chain_id
        self._block_number = 19000000
        self._gas_price = 20_000_000_000  # 20 gwei

    @property
    def block_number(self) -> int:
        return self._block_number

    @property
    def gas_price(self) -> int:
        return self._gas_price

    def get_transaction_count(self, address: str) -> int:
        return 42

    def contract(self, address: str, abi: list) -> MockContract:
        return MockContract(address, abi)


class MockContract:
    """Mock contract instance."""

    def __init__(self, address: str, abi: list):
        self.address = address
        self._abi = abi
        self.functions = MockContractFunctions()
        self.events = MockContractEvents()


class MockContractFunctions:
    """Mock contract functions."""

    def __init__(self):
        self._return_values: dict[str, Any] = {}

    def __getattr__(self, name: str):
        def function_wrapper(*args, **kwargs):
            return MockContractCall(name, args, self._return_values.get(name))

        return function_wrapper

    def set_return_value(self, method: str, value: Any) -> None:
        self._return_values[method] = value


class MockContractCall:
    """Mock contract call that can be called or built into a transaction."""

    def __init__(self, method: str, args: tuple, return_value: Any = None):
        self.method = method
        self.args = args
        self._return_value = return_value

    def call(self) -> Any:
        return self._return_value

    def build_transaction(self, tx_params: dict) -> dict:
        return {
            "to": "0x1234567890123456789012345678901234567890",
            "data": f"0x{self.method}",
            "gas": 100000,
            "gasPrice": 20_000_000_000,
            "nonce": tx_params.get("nonce", 0),
            "chainId": tx_params.get("chainId", 1),
        }


class MockContractEvents:
    """Mock contract events."""

    def __init__(self):
        self._events: dict[str, list[dict]] = {}

    def __getattr__(self, name: str):
        return MockEvent(name, self._events.get(name, []))


class MockEvent:
    """Mock event that can create filters."""

    def __init__(self, name: str, logs: list[dict]):
        self.name = name
        self._logs = logs

    def create_filter(self, fromBlock: int = 0, toBlock: int | str = "latest") -> MockFilter:
        return MockFilter(self._logs)

    def get_logs(self, fromBlock: int = 0, toBlock: int | str = "latest") -> list[dict]:
        return self._logs


class MockFilter:
    """Mock event filter."""

    def __init__(self, logs: list[dict]):
        self._logs = logs

    def get_all_entries(self) -> list[dict]:
        return self._logs


class MockAccount:
    """Mock account for signing."""

    def __init__(self, address: str, private_key: str):
        self.address = address
        self._private_key = private_key

    def sign_transaction(self, tx: dict) -> MockSignedTx:
        return MockSignedTx(tx)


class MockSignedTx:
    """Mock signed transaction."""

    def __init__(self, tx: dict):
        self.raw_transaction = b"\x00" * 32
        self.hash = b"\xab\xcd" * 16


# =============================================================================
# Chain Config Fixtures
# =============================================================================


@pytest.fixture
def mainnet_config() -> dict[str, Any]:
    """Ethereum mainnet configuration."""
    return {
        "chain_id": 1,
        "rpc_url": "https://eth.llamarpc.com",
        "identity_registry_address": "0x1111111111111111111111111111111111111111",
        "reputation_registry_address": "0x2222222222222222222222222222222222222222",
        "validation_registry_address": "0x3333333333333333333333333333333333333333",
        "fallback_rpc_urls": ["https://rpc.ankr.com/eth"],
        "block_confirmations": 12,
        "gas_limit": 500_000,
    }


@pytest.fixture
def sepolia_config() -> dict[str, Any]:
    """Ethereum Sepolia testnet configuration."""
    return {
        "chain_id": 11155111,
        "rpc_url": "https://rpc.sepolia.org",
        "identity_registry_address": "0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "reputation_registry_address": "0xBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
        "validation_registry_address": "0xCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
        "fallback_rpc_urls": [],
        "block_confirmations": 3,
        "gas_limit": 300_000,
    }


# =============================================================================
# Web3 Provider Fixtures
# =============================================================================


@pytest.fixture
def mock_web3() -> MockWeb3:
    """Create mock Web3 instance."""
    return MockWeb3(chain_id=1)


@pytest.fixture
def disconnected_web3() -> MockWeb3:
    """Create disconnected mock Web3 instance."""
    return MockWeb3(chain_id=1, is_connected=False)


@pytest.fixture
def mock_account() -> MockAccount:
    """Create mock account for signing."""
    return MockAccount(
        address="0xABCDEF1234567890ABCDEF1234567890ABCDEF12",
        private_key="0x" + "a" * 64,
    )


# =============================================================================
# Identity Registry Fixtures
# =============================================================================


@pytest.fixture
def sample_agent_identity() -> dict[str, Any]:
    """Sample on-chain agent identity data."""
    return {
        "token_id": 42,
        "owner": "0x1234567890123456789012345678901234567890",
        "metadata_uri": "ipfs://QmTest123456789",
        "agent_name": "test-agent-001",
        "aragora_agent_id": "agent_abc123",
        "chain_id": 1,
        "registered_at": datetime.now(timezone.utc),
        "metadata": {
            "capabilities": ["reasoning", "code_generation"],
            "version": "1.0.0",
        },
    }


@pytest.fixture
def sample_agent_identities() -> list[dict[str, Any]]:
    """List of sample agent identities."""
    return [
        {
            "token_id": 1,
            "owner": "0x1111111111111111111111111111111111111111",
            "metadata_uri": "ipfs://QmAgent1",
            "agent_name": "claude-agent",
            "chain_id": 1,
        },
        {
            "token_id": 2,
            "owner": "0x2222222222222222222222222222222222222222",
            "metadata_uri": "ipfs://QmAgent2",
            "agent_name": "gpt-agent",
            "chain_id": 1,
        },
        {
            "token_id": 3,
            "owner": "0x1111111111111111111111111111111111111111",
            "metadata_uri": "ipfs://QmAgent3",
            "agent_name": "gemini-agent",
            "chain_id": 1,
        },
    ]


# =============================================================================
# Reputation Registry Fixtures
# =============================================================================


@pytest.fixture
def sample_reputation_feedback() -> dict[str, Any]:
    """Sample reputation feedback data."""
    return {
        "subject_token_id": 42,
        "reporter": "0xFEDCBA0987654321FEDCBA0987654321FEDCBA09",
        "value": 100,  # Positive feedback
        "tag": "accuracy",
        "comment": "Excellent reasoning capabilities",
        "revoked": False,
        "timestamp": datetime.now(timezone.utc),
        "tx_hash": "0x" + "ab" * 32,
    }


@pytest.fixture
def sample_reputation_feedbacks() -> list[dict[str, Any]]:
    """List of sample reputation feedbacks."""
    return [
        {
            "subject_token_id": 42,
            "reporter": "0x1111111111111111111111111111111111111111",
            "value": 100,
            "tag": "accuracy",
            "comment": "Great work",
            "revoked": False,
        },
        {
            "subject_token_id": 42,
            "reporter": "0x2222222222222222222222222222222222222222",
            "value": -50,
            "tag": "reliability",
            "comment": "Inconsistent responses",
            "revoked": False,
        },
        {
            "subject_token_id": 42,
            "reporter": "0x3333333333333333333333333333333333333333",
            "value": 75,
            "tag": "speed",
            "comment": "Fast responses",
            "revoked": False,
        },
    ]


@pytest.fixture
def sample_reputation_summary() -> dict[str, Any]:
    """Sample reputation summary data."""
    return {
        "token_id": 42,
        "total_feedback_count": 15,
        "positive_count": 12,
        "negative_count": 3,
        "net_score": 850,
        "average_score": 56.67,
        "tags": {
            "accuracy": {"count": 5, "sum": 400},
            "reliability": {"count": 4, "sum": 300},
            "speed": {"count": 6, "sum": 150},
        },
    }


# =============================================================================
# Validation Registry Fixtures
# =============================================================================


@pytest.fixture
def sample_validation_record() -> dict[str, Any]:
    """Sample validation record data."""
    return {
        "subject_token_id": 42,
        "validator": "0xVALIDATOR123456789012345678901234567890",
        "request_type": "capability",
        "response_code": 1,  # PASS
        "evidence_uri": "ipfs://QmEvidence123",
        "timestamp": datetime.now(timezone.utc),
        "tx_hash": "0x" + "cd" * 32,
    }


@pytest.fixture
def sample_validation_records() -> list[dict[str, Any]]:
    """List of sample validation records."""
    return [
        {
            "subject_token_id": 42,
            "validator": "0x1111111111111111111111111111111111111111",
            "request_type": "capability",
            "response_code": 1,  # PASS
            "evidence_uri": "ipfs://QmEv1",
        },
        {
            "subject_token_id": 42,
            "validator": "0x2222222222222222222222222222222222222222",
            "request_type": "safety",
            "response_code": 1,  # PASS
            "evidence_uri": "ipfs://QmEv2",
        },
        {
            "subject_token_id": 42,
            "validator": "0x3333333333333333333333333333333333333333",
            "request_type": "compliance",
            "response_code": 0,  # PENDING
            "evidence_uri": "",
        },
    ]


@pytest.fixture
def sample_validation_summary() -> dict[str, Any]:
    """Sample validation summary data."""
    return {
        "token_id": 42,
        "total_validations": 10,
        "passed": 7,
        "failed": 1,
        "pending": 2,
        "revoked": 0,
        "by_type": {
            "capability": {"passed": 3, "failed": 0, "pending": 1},
            "safety": {"passed": 2, "failed": 1, "pending": 0},
            "compliance": {"passed": 2, "failed": 0, "pending": 1},
        },
    }


# =============================================================================
# Event Fixtures
# =============================================================================


@pytest.fixture
def sample_registered_event() -> dict[str, Any]:
    """Sample Registered event log."""
    return {
        "event": "Registered",
        "args": {
            "tokenId": 42,
            "owner": "0x1234567890123456789012345678901234567890",
        },
        "blockNumber": 19000000,
        "transactionHash": b"\xab" * 32,
    }


@pytest.fixture
def sample_feedback_event() -> dict[str, Any]:
    """Sample NewFeedback event log."""
    return {
        "event": "NewFeedback",
        "args": {
            "subject": 42,
            "reporter": "0xFEDCBA0987654321FEDCBA0987654321FEDCBA09",
            "value": 100,
            "tag": "accuracy",
        },
        "blockNumber": 19000100,
        "transactionHash": b"\xcd" * 32,
    }


# =============================================================================
# Connector Fixtures
# =============================================================================


@pytest.fixture
def mock_connector_context() -> dict[str, Any]:
    """Context for ERC8004Connector."""
    return {
        "chain_id": 1,
        "cache_ttl": 300,
        "timeout": 30.0,
    }


# =============================================================================
# Adapter Fixtures
# =============================================================================


@pytest.fixture
def mock_km_mound() -> MagicMock:
    """Mock Knowledge Mound for adapter testing."""
    mound = MagicMock()
    mound.store = AsyncMock(return_value={"id": "km_node_123"})
    mound.search = AsyncMock(return_value=[])
    mound.get = AsyncMock(return_value=None)
    return mound


# =============================================================================
# Handler Fixtures
# =============================================================================


@pytest.fixture
def mock_handler_context() -> dict[str, Any]:
    """Context for ERC8004Handler."""
    return {
        "blockchain_enabled": True,
        "chain_id": 1,
    }


# =============================================================================
# Bridge Fixtures
# =============================================================================


@pytest.fixture
def mock_agent_registry() -> MagicMock:
    """Mock AgentRegistry for bridge testing."""
    registry = MagicMock()
    registry.get_agent = MagicMock(return_value=None)
    registry.list_agents = MagicMock(return_value=[])
    registry.register_agent = MagicMock(return_value="agent_123")
    return registry


# =============================================================================
# Environment Fixtures
# =============================================================================


@pytest.fixture
def blockchain_env_vars(monkeypatch) -> None:
    """Set up blockchain environment variables for testing."""
    monkeypatch.setenv("ERC8004_RPC_URL", "https://eth.llamarpc.com")
    monkeypatch.setenv("ERC8004_CHAIN_ID", "1")
    monkeypatch.setenv("ERC8004_IDENTITY_REGISTRY", "0x1111111111111111111111111111111111111111")
    monkeypatch.setenv("ERC8004_REPUTATION_REGISTRY", "0x2222222222222222222222222222222222222222")
    monkeypatch.setenv("ERC8004_VALIDATION_REGISTRY", "0x3333333333333333333333333333333333333333")


@pytest.fixture
def wallet_env_vars(monkeypatch) -> None:
    """Set up wallet environment variables for testing."""
    # Use a test private key (never use in production!)
    monkeypatch.setenv("ERC8004_WALLET_KEY", "0x" + "a" * 64)
