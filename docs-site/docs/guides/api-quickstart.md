---
title: Aragora API Quick Start
description: Aragora API Quick Start
---

# Aragora API Quick Start

Get started with the Aragora API in 5 minutes.

## Prerequisites

- Python 3.11+
- At least one API key: `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`

## Installation

```bash
pip install aragora
```

## Quick Start

### 1. Start the Server

```bash
python -m aragora.server.unified_server --port 8080
```

### 2. Create a Debate

```bash
curl -X POST http://localhost:8080/api/debates \
  -H "Content-Type: application/json" \
  -d '{
    "task": "What is the best approach to implement rate limiting?",
    "agents": ["claude", "gpt4"],
    "rounds": 3
  }'
```

**Response:**
```json
{
  "debate_id": "d-abc123",
  "status": "created",
  "stream_url": "/api/debates/d-abc123/stream"
}
```

### 3. Stream the Debate

```bash
curl -N http://localhost:8080/api/debates/d-abc123/stream
```

**Event Stream:**
```
event: debate_start
data: {"debate_id": "d-abc123", "task": "What is the best approach..."}

event: agent_message
data: {"agent": "claude", "role": "proposer", "content": "I propose..."}

event: critique
data: {"agent": "gpt4", "target": "claude", "severity": 0.6, "content": "While valid..."}

event: agent_error
data: {"agent": "claude", "error_type": "timeout", "message": "Agent response was empty", "recoverable": true, "phase": "proposal"}

event: vote
data: {"agent": "claude", "choice": "gpt4", "confidence": 0.85}

event: consensus
data: {"reached": true, "confidence": 0.82, "answer": "...", "status": "consensus_reached", "agent_failures": {}}

event: debate_end
data: {"debate_id": "d-abc123", "duration_ms": 12500}
```

### 4. Get the Result

```bash
curl http://localhost:8080/api/debates/d-abc123
```

**Response:**
```json
{
  "id": "d-abc123",
  "task": "What is the best approach to implement rate limiting?",
  "final_answer": "Token bucket algorithm with Redis backend...",
  "winner": "gpt4",
  "confidence": 0.82,
  "consensus_reached": true,
  "status": "consensus_reached",
  "agent_failures": {},
  "rounds_used": 3,
  "participants": ["claude", "gpt4"],
  "messages": [...],
  "votes": [...],
  "duration_ms": 12500
}
```

## Python SDK

```python
from aragora import Arena, Environment, DebateProtocol
from aragora.agents.api_agents import AnthropicAgent, OpenAIAgent

# Create agents
agents = [
    AnthropicAgent(name="claude"),
    OpenAIAgent(name="gpt4")
]

# Configure the debate
env = Environment(task="Design a rate limiter for an API")
protocol = DebateProtocol(rounds=3, consensus="majority")

# Run the debate
async def main():
    arena = Arena(env, agents, protocol)
    result = await arena.run()

    print(f"Winner: {result.winner}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Answer: {result.final_answer}")

import asyncio
asyncio.run(main())
```

## Common Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/debates` | POST | Create a new debate |
| `/api/debates/\{id\}` | GET | Get debate result |
| `/api/debates/\{id\}/stream` | GET | Stream debate events (SSE) |
| `/api/debates` | GET | List recent debates |
| `/api/agents` | GET | List available agents |
| `/api/agents/\{name\}/stats` | GET | Get agent statistics |
| `/api/health` | GET | Server health check |

## Configuration Options

### Debate Protocol

```json
{
  "task": "Your question or task",
  "agents": ["claude", "gpt4", "gemini"],
  "rounds": 3,
  "consensus": "majority",           // "majority" | "unanimous" | "judge"
  "early_stopping": true,            // Stop if agents converge
  "enable_calibration": false,       // Track prediction accuracy
  "enable_checkpointing": true       // Enable resume on failure
}
```

### Available Agents

| Agent | Provider | Model |
|-------|----------|-------|
| `claude` | Anthropic | claude-3-5-sonnet |
| `gpt4` | OpenAI | gpt-4-turbo |
| `gemini` | Google | gemini-pro |
| `mistral` | Mistral | mistral-large |
| `grok` | xAI | grok-beta |

## Error Handling

All errors return JSON with this structure:

```json
{
  "error": {
    "code": "AGENT_TIMEOUT",
    "message": "Agent claude timed out after 30s",
    "details": {
      "agent": "claude",
      "timeout_ms": 30000
    }
  }
}
```

### Common Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `INVALID_REQUEST` | Malformed request body | Check JSON syntax |
| `AGENT_UNAVAILABLE` | Agent API key missing | Set environment variable |
| `AGENT_TIMEOUT` | Agent response timeout | Increase timeout or retry |
| `RATE_LIMITED` | Provider rate limit hit | Wait and retry |
| `DEBATE_NOT_FOUND` | Invalid debate ID | Check debate ID |

## WebSocket Events

Connect to `/ws` for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  switch (data.type) {
    case 'debate_start':
      console.log('Debate started:', data.debate_id);
      break;
    case 'agent_message':
      console.log(`${data.agent}: ${data.content}`);
      break;
    case 'consensus':
      console.log('Consensus:', data.final_answer);
      break;
  }
};
```

## Next Steps

- [Full API Reference](../api/reference)
- [Architecture Overview](../core-concepts/architecture)
- [Environment Variables](../getting-started/environment)
- [Feature Documentation](./features)

## Support

- GitHub Issues: https://github.com/an0mium/aragora/issues
- Documentation: https://aragora.dev/docs
