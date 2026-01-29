# Aragora Python SDK Examples

Example scripts demonstrating the Aragora Python SDK.

## Setup

```bash
# Install the SDK
pip install aragora

# Set environment variables
export ARAGORA_API_KEY="your-api-key"
export ARAGORA_API_URL="https://api.aragora.ai"
```

## Examples

| File | Description |
|------|-------------|
| `basic_debate.py` | Create a debate, poll for results, print consensus |
| `streaming_debate.py` | Real-time WebSocket event streaming |
| `workflow_automation.py` | Create and execute workflow templates |
| `agent_selection.py` | List agents, compare, view leaderboard |
| `receipts_demo.py` | Decision receipt retrieval and verification |

## Running

```bash
# Run any example
python examples/basic_debate.py

# Run with a custom API URL
ARAGORA_API_URL=http://localhost:8080 python examples/basic_debate.py
```
