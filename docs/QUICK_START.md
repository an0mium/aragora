# 5-Minute Quick Start

Get Aragora running in 5 minutes or less.

## Prerequisites

- Python 3.10+
- One API key (Anthropic or OpenAI)

## 1. Install

```bash
# Clone the repository
git clone https://github.com/aragora-ai/aragora.git
cd aragora

# Install dependencies
pip install -e .
```

## 2. Configure

```bash
# Copy minimal config
cp minimal.env.example .env

# Edit .env and add your API key
# ANTHROPIC_API_KEY=sk-ant-...
# OR
# OPENAI_API_KEY=sk-...
```

## 3. Run

### Option A: Command Line

```bash
# Run a simple debate
aragora ask "What's the best programming language for web development?"

# Run with streaming output
aragora ask "Should we use microservices?" --stream
```

### Option B: Start the Server

```bash
# Start the unified server (HTTP + WebSocket)
python -m aragora.server.unified_server

# Server runs at:
# - HTTP API: http://localhost:8080
# - WebSocket: ws://localhost:8080/ws
# - API Docs: http://localhost:8080/api/docs
# - Health: http://localhost:8080/api/health
```

## 4. Use the API

### Create a Debate

```bash
curl -X POST http://localhost:8080/api/debates \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Should remote work be the default?",
    "rounds": 3
  }'
```

### List Debates

```bash
curl http://localhost:8080/api/debates
```

### Get a Specific Debate

```bash
curl http://localhost:8080/api/debates/{debate_id}
```

## 5. Python SDK

```python
from aragora import Arena, Environment, DebateProtocol

# Create environment
env = Environment(task="Design a rate limiter")

# Configure protocol
protocol = DebateProtocol(rounds=3, consensus="majority")

# Run debate
arena = Arena(env, protocol=protocol)
result = await arena.run()

print(f"Consensus: {result.consensus}")
print(f"Winner: {result.winner}")
```

## What's Next?

- **Full Configuration**: See [ENVIRONMENT.md](ENVIRONMENT.md) for all options
- **API Reference**: Visit http://localhost:8080/api/docs
- **WebSocket Streaming**: See [API_EXAMPLES.md](API_EXAMPLES.md)
- **Architecture**: See [CLAUDE.md](../CLAUDE.md) for codebase overview

## Troubleshooting

### "No API key configured"

Add at least one API key to your `.env` file:
```bash
ANTHROPIC_API_KEY=sk-ant-...
```

### "Port 8080 already in use"

Change the port:
```bash
ARAGORA_PORT=8081 python -m aragora.server.unified_server
```

### "Module not found"

Install in editable mode:
```bash
pip install -e .
```

## Need Help?

- [Full Documentation](../README.md)
- [Environment Variables](ENVIRONMENT.md)
- [GitHub Issues](https://github.com/aragora-ai/aragora/issues)
