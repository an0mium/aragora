# Blockchain Integration

ERC-8004 Trustless Autonomous Agents integration for Aragora.

## Overview

This module provides multi-chain Ethereum connectivity for:
- Agent identity management (NFT-based)
- On-chain reputation tracking
- Validation record attestations

## Architecture

```
blockchain/
├── __init__.py          # Public API exports
├── config.py            # Chain configurations (mainnet, testnet, local)
├── models.py            # Data models (OnChainAgentIdentity, ReputationFeedback, ValidationRecord)
├── provider.py          # Web3Provider for blockchain connectivity
├── wallet.py            # WalletSigner for transaction signing
├── events.py            # Event listeners for on-chain events
└── contracts/           # Smart contract interfaces
    ├── identity.py      # ERC-8004 Identity Registry
    ├── reputation.py    # ERC-8004 Reputation Registry
    └── validation.py    # ERC-8004 Validation Registry
```

## Usage

```python
from aragora.blockchain import Web3Provider, OnChainAgentIdentity
from aragora.blockchain.config import get_chain_config

# Initialize provider
provider = Web3Provider.from_env()

# Get agent identity
identity = provider.identity_registry.get_agent(token_id=42)

# Submit reputation feedback
await provider.reputation_registry.submit_feedback(
    agent_id=42,
    score=85,
    context="debate-123"
)
```

## Configuration

Set these environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `WEB3_PROVIDER_URI` | Ethereum RPC endpoint | `http://localhost:8545` |
| `CHAIN_ID` | Network chain ID | `1` (mainnet) |
| `IDENTITY_REGISTRY_ADDRESS` | ERC-8004 Identity contract | Required |
| `REPUTATION_REGISTRY_ADDRESS` | ERC-8004 Reputation contract | Required |
| `VALIDATION_REGISTRY_ADDRESS` | ERC-8004 Validation contract | Required |

## Installation

Blockchain support requires optional dependencies:

```bash
pip install aragora[blockchain]
```

## ERC-8004 Standard

This module implements the ERC-8004 standard for Trustless Autonomous Agents:
- **Identity Registry**: NFT-based agent identities with metadata
- **Reputation Registry**: On-chain reputation scores and feedback
- **Validation Registry**: Cryptographic validation attestations

## Integration with Knowledge Mound

The `ERC8004Adapter` in `aragora/knowledge/mound/adapters/` provides bidirectional sync:
- Forward sync: Blockchain data -> Knowledge Mound nodes
- Reverse sync: ELO ratings, gauntlet receipts -> On-chain records

## Security Notes

- Never commit private keys or wallet mnemonics
- Use hardware wallets for production signing
- Validate all contract addresses before transactions
