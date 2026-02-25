"""
ERC-8004 contract interface wrappers.

Provides typed Python wrappers around the three ERC-8004 registries
plus the economic staking layer:
- IdentityRegistryContract
- ReputationRegistryContract
- ValidationRegistryContract
- StakingRegistry (compute-budget staking and slashing)

Each wrapper encapsulates the contract ABI and provides type-safe methods
for interacting with the deployed smart contracts.
"""

from __future__ import annotations

from aragora.blockchain.contracts.identity import (
    IDENTITY_REGISTRY_ABI,
    IdentityRegistryContract,
)
from aragora.blockchain.contracts.reputation import (
    REPUTATION_REGISTRY_ABI,
    ReputationRegistryContract,
)
from aragora.blockchain.contracts.staking import (
    SlashEvent,
    StakePosition,
    StakingRegistry,
)
from aragora.blockchain.contracts.validation import (
    VALIDATION_REGISTRY_ABI,
    ValidationRegistryContract,
)

__all__ = [
    "IDENTITY_REGISTRY_ABI",
    "REPUTATION_REGISTRY_ABI",
    "VALIDATION_REGISTRY_ABI",
    "IdentityRegistryContract",
    "ReputationRegistryContract",
    "SlashEvent",
    "StakePosition",
    "StakingRegistry",
    "ValidationRegistryContract",
]
