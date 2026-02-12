# Getting Started with Aragora

Get from zero to a working debate in under 5 minutes.

## Prerequisites

- Python 3.10+
- At least one LLM API key (Anthropic, OpenAI, Gemini, or xAI)

## 1. Install

```bash
pip install aragora-sdk
```

Or install the full platform for self-hosting:

```bash
pip install aragora
```

## 2. Set up your environment

Create a `.env` file or export environment variables:

```bash
# Required: at least one LLM provider
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."

# Optional: Aragora server URL (defaults to http://localhost:8080)
export ARAGORA_API_URL="http://localhost:8080"

# Optional: API key for authenticated endpoints
export ARAGORA_API_KEY="ara_your_key_here"
```

## 3. Start the server

```bash
# Quick start with demo mode (no database required)
python -m aragora.server.unified_server --port 8080 --offline

# Or use the CLI quickstart wizard
aragora quickstart
```

The `--offline` flag runs with SQLite and mock data, perfect for evaluation.

## 4. Run your first debate

```python
from aragora_sdk import AragoraClient

# Auto-configure from environment variables
client = AragoraClient.from_env()

# Create a debate
debate = client.debates.create(task="Should we use microservices or a monolith?")
print(f"Debate ID: {debate['debate_id']}")
print(f"Consensus: {debate.get('consensus', {}).get('conclusion', 'Pending...')}")
```

Or use the CLI directly:

```bash
# Interactive question
aragora ask "What are the trade-offs of event-driven architecture?"

# Full adversarial debate
aragora decide "Should we migrate to Kubernetes?" --rounds 3
```

## 5. Explore further

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
python -m aragora.server.unified_server --port 8080 --offline
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
