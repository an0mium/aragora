# Aragora Python SDK

Official Python client for the Aragora multi-agent debate platform.

## Installation

```bash
pip install aragora
```

## Quick Start

```python
from aragora import AragoraClient

# Create a client
client = AragoraClient(
    base_url="https://api.aragora.ai",
    api_key="your-api-key"
)

# Create a debate
debate = client.debates.create(
    task="Should we adopt microservices architecture?"
)

print(f"Debate created: {debate['debate_id']}")

# Get debate messages
messages = client.debates.get_messages(debate['debate_id'])
for msg in messages['messages']:
    print(f"{msg['agent']}: {msg['content'][:100]}...")
```

## Async Usage

```python
from aragora import AragoraAsyncClient

async def main():
    async with AragoraAsyncClient(
        base_url="https://api.aragora.ai",
        api_key="your-api-key"
    ) as client:
        # Create a debate
        debate = await client.debates.create(
            task="What's the best approach for state management?"
        )

        # List agents
        agents = await client.agents.list()
        print(f"Available agents: {len(agents['agents'])}")

import asyncio
asyncio.run(main())
```

## Available Namespaces

- `client.debates` - Create and manage debates
- `client.agents` - List agents and view performance
- `client.workflows` - Create and execute automated workflows

## Error Handling

```python
from aragora import AragoraClient, AragoraError, RateLimitError

client = AragoraClient(base_url="https://api.aragora.ai")

try:
    debate = client.debates.get("invalid-id")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except AragoraError as e:
    print(f"API error: {e.message}")
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy aragora

# Linting
ruff check aragora
```

## License

MIT License - see LICENSE file for details.
