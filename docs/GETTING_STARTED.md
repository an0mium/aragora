# Getting Started with Aragora

Get from zero to a working debate in under 2 minutes.

## 1. Install

```bash
pip install aragora-sdk
```

## 2. Try it instantly (no server, no API keys)

```python
from aragora_sdk import AragoraClient

client = AragoraClient(demo=True)

# Run a debate
result = client.request("POST", "/api/v1/debates", json={
    "task": "Should we use microservices or a monolith?"
})
print(f"Consensus: {result['consensus']['conclusion']}")

# Check agent rankings
rankings = client.request("GET", "/api/v1/rankings")
for r in rankings["rankings"]:
    print(f"  #{r['rank']} {r['agent']} (ELO {r['elo']})")

# Discover available APIs
print(f"Available namespaces: {client.namespaces[:10]}...")
```

Demo mode returns realistic mock data — debates with consensus, agent rankings, gauntlet findings with cryptographic receipts — so you can explore the full API surface immediately.

## 3. Connect to a real server

When you're ready to run real debates with LLM agents:

```bash
# Set at least one LLM provider key
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."

# Optional: Aragora API key for authenticated endpoints
export ARAGORA_API_KEY="ara_your_key_here"
```

Start the server:

```bash
# Quick start (SQLite, no external dependencies)
python -m aragora.server --http-port 8080 --ws-port 8765 --offline

# Or use the CLI quickstart wizard
aragora quickstart
```

Then use the SDK with auto-configuration:

```python
from aragora_sdk import AragoraClient

# Reads ARAGORA_API_URL, ARAGORA_API_KEY from environment
client = AragoraClient.from_env()

debate = client.debates.create(task="Should we use microservices or a monolith?")
print(f"Debate ID: {debate['debate_id']}")
```

Or use the CLI directly:

```bash
aragora ask "What are the trade-offs of event-driven architecture?"
aragora decide "Should we migrate to Kubernetes?" --rounds 3
```

## 4. Explore further

### CLI demo (no API keys needed)

```bash
aragora review --demo
```

### Interactive REPL

```bash
aragora repl
```

### Run an example

```bash
# From the examples directory
python examples/python-debate/main.py health
python examples/python-debate/main.py debate --task "Evaluate our tech stack"
```

## What's next

| Goal | Guide |
|------|-------|
| Understand authentication | [Auth Guide](guides/AUTH_GUIDE.md) |
| Use the Python SDK | [SDK Quickstart](guides/SDK_QUICKSTART_PYTHON.md) |
| Use the TypeScript SDK | [SDK Guide](guides/SDK_GUIDE.md) |
| Deploy to production | [Deployment Guide](DEPLOYMENT.md) |
| Explore all features | [Feature Discovery](FEATURE_DISCOVERY.md) |
| Self-host with Docker | [Docker Compose](../deploy/docker-compose.yml) |

## Troubleshooting

### "Connection refused" on localhost:8080

The Aragora server isn't running. Start it with:

```bash
python -m aragora.server --http-port 8080 --ws-port 8765 --offline
```

### "No ANTHROPIC_API_KEY configured"

Set at least one LLM provider API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### "Authentication failed" (401)

Your `ARAGORA_API_KEY` is invalid or expired. Generate a new one:

```bash
aragora auth login
aragora auth create-key --name "dev-key"
```

### Import errors

Make sure you're using the correct package:

```python
# Correct: use aragora-sdk for the API client
from aragora_sdk import AragoraClient

# Correct: use aragora for the platform internals
from aragora import Arena, Environment, DebateProtocol
```
