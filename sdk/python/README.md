# Aragora Python SDK

Official Python SDK for the [Aragora](https://aragora.ai) multi-agent debate platform.

## Installation

```bash
pip install aragora
```

For WebSocket support:

```bash
pip install aragora[websocket]
```

## Quick Start

### Async Usage (Recommended)

```python
import asyncio
from aragora import AragoraClient

async def main():
    async with AragoraClient(
        base_url="https://api.aragora.ai",
        api_key="your-api-key"
    ) as client:
        # Create a debate
        debate = await client.create_debate(
            question="What is the best database for a high-traffic web app?",
            agents=["claude", "gpt-4", "gemini"],
            rounds=3
        )
        print(f"Debate created: {debate['debate_id']}")

        # Get debate results
        result = await client.get_debate(debate["debate_id"])
        print(f"Status: {result['status']}")

        # List all debates
        debates = await client.list_debates(limit=10)
        print(f"Total debates: {len(debates['debates'])}")

asyncio.run(main())
```

### Sync Usage

```python
from aragora import AragoraClientSync

client = AragoraClientSync(
    base_url="https://api.aragora.ai",
    api_key="your-api-key"
)

# Create a debate
debate = client.create_debate(
    question="Should we use microservices or monolith?",
    agents=["claude", "gpt-4"],
)
print(f"Debate created: {debate['debate_id']}")

# Don't forget to close
client.close()
```

## API Reference

### Debates

```python
# Create a debate
await client.create_debate(
    question="Your question here",
    agents=["claude", "gpt-4"],
    rounds=3
)

# List debates
await client.list_debates(limit=50, offset=0)

# Get a specific debate
await client.get_debate(debate_id)
```

### Explainability

```python
# Get full explanation
await client.get_explanation(debate_id)

# Get contributing factors
await client.get_factors(debate_id, min_contribution=0.1)

# Get counterfactual scenarios
await client.get_counterfactuals(debate_id, max_scenarios=5)

# Get natural language narrative
await client.get_narrative(debate_id, format="detailed")
```

### Batch Operations

```python
# Create batch explanation job
batch = await client.create_batch_explanation(
    debate_ids=["id1", "id2", "id3"],
    include_evidence=True
)

# Check status
status = await client.get_batch_status(batch["batch_id"])

# Get results
results = await client.get_batch_results(batch["batch_id"])

# Compare multiple debates
comparison = await client.compare_explanations(["id1", "id2"])
```

### Workflows

```python
# List workflow templates
templates = await client.list_workflow_templates(category="security")

# Run a template
execution = await client.run_workflow_template(
    template_id="security-review",
    inputs={"code": "..."}
)
```

### Marketplace

```python
# Browse templates
templates = await client.browse_marketplace(
    category="compliance",
    sort_by="rating"
)

# Import a template
await client.import_template(template_id)

# Rate a template
await client.rate_template(template_id, rating=5)
```

### Gauntlet

```python
# List receipts
receipts = await client.list_gauntlet_receipts(verdict="pass")

# Get a receipt
receipt = await client.get_gauntlet_receipt(receipt_id)

# Verify integrity
verification = await client.verify_gauntlet_receipt(receipt_id)
```

## Configuration

### Environment Variables

```bash
export ARAGORA_API_KEY="your-api-key"
export ARAGORA_API_URL="https://api.aragora.ai"
```

### Client Options

```python
client = AragoraClient(
    base_url="https://api.aragora.ai",
    api_key="your-api-key",
    timeout=60.0,  # Request timeout in seconds
    headers={"X-Custom-Header": "value"}  # Additional headers
)
```

## Error Handling

```python
import httpx
from aragora import AragoraClient

async with AragoraClient(api_key="...") as client:
    try:
        debate = await client.get_debate("invalid-id")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            print("Debate not found")
        elif e.response.status_code == 401:
            print("Invalid API key")
        else:
            print(f"API error: {e.response.status_code}")
```

## Requirements

- Python 3.9+
- httpx

## License

MIT
