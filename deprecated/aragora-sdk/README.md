# Aragora SDK

> **DEPRECATED**: This package is deprecated. Please use [`aragora-client`](https://pypi.org/project/aragora-client/) instead, which has more features including WebSocket streaming and Control Plane support.
>
> ```bash
> pip install aragora-client
> ```

Python SDK for the Aragora control plane for multi-agent vetted decisionmaking. Use AI agents to review designs, validate compliance, and stress-test specifications before implementation.

## Installation

```bash
pip install aragora-sdk
```

## Quick Start

```python
import asyncio
from aragora_sdk import AragoraClient

async def main():
    async with AragoraClient(api_key="ara_...") as client:
        result = await client.review(
            spec="""
            # Payment Processing System

            ## Overview
            Process credit card payments with tokenization.

            ## Data Flow
            1. Receive card data from frontend
            2. Tokenize with payment processor
            3. Store token in database
            4. Process recurring payments using token
            """,
            personas=["pci_dss", "security", "sox"],
            rounds=3,
        )

        print(f"Consensus: {result.consensus.status}")
        print(f"Position: {result.final_position}")

        for dissent in result.dissenting_opinions:
            print(f"\nDissent from {dissent.agent}:")
            print(f"  {dissent.position}")

asyncio.run(main())
```

## Available Personas

### Compliance Personas

| Persona | Focus Area |
|---------|------------|
| `sox` | Sarbanes-Oxley financial controls, audit trails |
| `pci_dss` | Payment Card Industry Data Security Standard |
| `hipaa` | Health Insurance Portability and Accountability Act |
| `gdpr` | General Data Protection Regulation |
| `fda_21_cfr` | FDA 21 CFR Part 11 (electronic records) |
| `fisma` | Federal Information Security Management Act |
| `finra` | Financial Industry Regulatory Authority |

### Technical Personas

| Persona | Focus Area |
|---------|------------|
| `security` | General security review |
| `performance` | Performance and scalability |
| `architecture` | System architecture |
| `testing` | Test coverage and quality |

## Features

### Review a Design

```python
result = await client.review(
    spec="Your design document...",
    personas=["sox", "security"],
    rounds=3,
)
```

### Review a File

```python
result = await client.review_file(
    "design.md",
    personas=["hipaa", "security"],
)
```

### Streaming Reviews

```python
async for event in client.review_stream(spec, personas=["security"]):
    if event["event"] == "position":
        print(f"{event['agent']}: {event['content'][:100]}...")
    elif event["event"] == "consensus":
        print(f"Consensus: {event['status']}")
```

### Decision Receipts

Get audit-ready decision receipts for compliance:

```python
result = await client.review(spec, personas=["sox"], include_receipt=True)

receipt = result.decision_receipt
print(f"Receipt ID: {receipt.id}")
print(f"Checksum: {receipt.checksum}")
print(f"Positions: {len(receipt.positions)}")
print(f"Critiques: {len(receipt.critiques)}")
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ARAGORA_API_KEY` | API key for authentication |
| `ARAGORA_BASE_URL` | Custom API base URL (default: https://api.aragora.ai) |

## Error Handling

```python
from aragora_sdk import (
    AragoraError,
    AuthenticationError,
    RateLimitError,
)

try:
    result = await client.review(spec, personas=["sox"])
except AuthenticationError:
    print("Invalid API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except AragoraError as e:
    print(f"Error: {e.message}")
```

## API Reference

### AragoraClient

```python
AragoraClient(
    api_key: str | None = None,  # Or set ARAGORA_API_KEY
    base_url: str | None = None,  # Default: https://api.aragora.ai
    timeout: int = 300,  # Request timeout in seconds
)
```

### Methods

| Method | Description |
|--------|-------------|
| `review(spec, personas, rounds)` | Review a specification |
| `review_file(path, personas, rounds)` | Review a file |
| `review_stream(spec, personas, rounds)` | Stream review events |
| `health()` | Check API health |
| `get_usage()` | Get usage information |
| `list_personas()` | List available personas |

## License

MIT License - see LICENSE file for details.
