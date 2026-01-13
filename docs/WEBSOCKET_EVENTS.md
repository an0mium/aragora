# WebSocket Events Reference

Complete reference for Aragora WebSocket events.

## Connection

### Endpoint

```
ws://localhost:8765/ws
wss://api.aragora.ai/ws  # Production
```

The WebSocket server accepts `/` or `/ws`. The live UI uses `/ws`.

### Authentication

If `ARAGORA_API_TOKEN` is set, include an `Authorization: Bearer` header during the handshake:

```bash
wscat -c ws://localhost:8765/ws -H "Authorization: Bearer $ARAGORA_API_TOKEN"
```

Browser clients cannot set custom headers; use a server-side proxy if auth is enforced.

### Subscribing to a Debate

After connecting, subscribe to a debate:

```json
{"type": "subscribe", "debate_id": "dbt_abc123"}
```

### Connection Lifecycle

```
Client                              Server
   |                                   |
   |------- WebSocket Connect -------->|
   |                                   |
   |<------ connection_established ----|
   |                                   |
   |<--------- debate_start -----------|
   |                                   |
   |<-------- round_start -------------|
   |<-------- agent_message -----------|
   |<-------- agent_message -----------|
   |<-------- critique ----------------|
   |<-------- round_end ---------------|
   |                                   |
   |---------- vote ------------------>|  (user input)
   |---------- suggestion ------------>|  (user input)
   |                                   |
   |<-------- consensus ---------------|
   |<-------- debate_end --------------|
   |                                   |
   X------- Connection closed ---------X
```

## Event Types

### Server → Client Events

#### `connection_established`

Sent immediately after WebSocket connection is established.

```json
{
  "type": "connection_established",
  "debate_id": "dbt_abc123",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "data": {
    "connection_id": "conn_xyz789",
    "server_version": "1.0.0",
    "capabilities": ["vote", "suggest", "breakpoint"]
  }
}
```

#### `debate_start`

Signals the beginning of a debate.

```json
{
  "type": "debate_start",
  "debate_id": "dbt_abc123",
  "timestamp": "2024-01-15T10:30:01.000Z",
  "data": {
    "task": "Should we use microservices architecture?",
    "agents": ["anthropic-api", "openai-api", "gemini"],
    "max_rounds": 5,
    "consensus_threshold": 0.8,
    "protocol": {
      "type": "standard",
      "voting_enabled": true,
      "user_participation": true
    }
  }
}
```

#### `round_start`

Signals the start of a new debate round.

```json
{
  "type": "round_start",
  "debate_id": "dbt_abc123",
  "timestamp": "2024-01-15T10:30:05.000Z",
  "data": {
    "round": 1,
    "phase": "proposal",
    "agents_participating": ["anthropic-api", "openai-api", "gemini"],
    "time_limit_seconds": 120
  }
}
```

#### `agent_message`

An agent has submitted a message.

```json
{
  "type": "agent_message",
  "debate_id": "dbt_abc123",
  "timestamp": "2024-01-15T10:30:15.000Z",
  "data": {
    "agent_id": "anthropic-api",
    "round": 1,
    "message_type": "proposal",
    "content": "I propose we adopt a hybrid approach...",
    "position": "for_microservices",
    "confidence": 0.85,
    "arguments": [
      "Improved scalability for individual services",
      "Independent deployment cycles",
      "Technology diversity per service"
    ],
    "metadata": {
      "tokens_used": 245,
      "latency_ms": 1230
    }
  }
}
```

#### `critique`

An agent has critiqued another agent's message.

```json
{
  "type": "critique",
  "debate_id": "dbt_abc123",
  "timestamp": "2024-01-15T10:31:00.000Z",
  "data": {
    "critic_id": "openai-api",
    "target_agent_id": "anthropic-api",
    "target_round": 1,
    "severity": "medium",
    "content": "While the scalability argument is valid, it overlooks...",
    "critique_type": "counterargument"
  }
}
```

#### `vote`

Voting results from agents or users.

```json
{
  "type": "vote",
  "debate_id": "dbt_abc123",
  "timestamp": "2024-01-15T10:32:00.000Z",
  "data": {
    "round": 1,
    "voter_type": "agent",
    "voter_id": "gemini",
    "votes": {
      "anthropic-api": 0.7,
      "openai-api": 0.3
    },
    "reasoning": "anthropic-api's proposal better addresses scalability concerns"
  }
}
```

#### `consensus`

Consensus has been reached (or failed).

```json
{
  "type": "consensus",
  "debate_id": "dbt_abc123",
  "timestamp": "2024-01-15T10:35:00.000Z",
  "data": {
    "reached": true,
    "round": 3,
    "conclusion": "Adopt microservices with careful service boundary design",
    "confidence": 0.87,
    "supporting_agents": ["anthropic-api", "openai-api"],
    "dissenting_agents": ["gemini"],
    "convergence_type": "semantic"
  }
}
```

#### `round_end`

Signals the end of a debate round.

```json
{
  "type": "round_end",
  "debate_id": "dbt_abc123",
  "timestamp": "2024-01-15T10:33:00.000Z",
  "data": {
    "round": 1,
    "summary": "First round established initial positions",
    "convergence_score": 0.45,
    "next_phase": "critique"
  }
}
```

#### `debate_end`

Signals the debate has concluded.

```json
{
  "type": "debate_end",
  "debate_id": "dbt_abc123",
  "timestamp": "2024-01-15T10:36:00.000Z",
  "data": {
    "status": "completed",
    "total_rounds": 3,
    "consensus_reached": true,
    "final_conclusion": "Adopt microservices with careful service boundary design",
    "duration_seconds": 360,
    "replay_id": "rpl_def456"
  }
}
```

#### `breakpoint`

Human-in-the-loop breakpoint triggered.

```json
{
  "type": "breakpoint",
  "debate_id": "dbt_abc123",
  "timestamp": "2024-01-15T10:32:30.000Z",
  "data": {
    "breakpoint_id": "brk_xyz789",
    "round": 2,
    "reason": "High disagreement detected",
    "options": ["continue", "modify", "stop"],
    "timeout_seconds": 300
  }
}
```

#### `error`

An error occurred during the debate.

```json
{
  "type": "error",
  "debate_id": "dbt_abc123",
  "timestamp": "2024-01-15T10:34:00.000Z",
  "data": {
    "code": "AGENT_TIMEOUT",
    "message": "Agent openai-api timed out after 60 seconds",
    "severity": "warning",
    "recoverable": true
  }
}
```

### Client → Server Events

#### `vote` (User Vote)

Submit a user vote on proposals.

```json
{
  "type": "vote",
  "data": {
    "round": 1,
    "votes": { "anthropic-api": 0.8, "openai-api": 0.2 },
    "comment": "Claude's argument is more compelling"
  }
}
```

#### `suggestion`

Submit a user suggestion to influence the debate.

```json
{
  "type": "suggestion",
  "data": {
    "content": "Consider the impact on team structure",
    "target_agents": ["all"],
    "priority": "high"
  }
}
```

#### `breakpoint_resolve`

Resolve a human-in-the-loop breakpoint.

```json
{
  "type": "breakpoint_resolve",
  "data": {
    "breakpoint_id": "brk_xyz789",
    "action": "modify",
    "modification": "Focus on practical implementation"
  }
}
```

#### `ping`

Keep-alive ping (server responds with `pong`).

```json
{ "type": "ping" }
```

## Rate Limits

| Event Type | Limit |
|------------|-------|
| `vote` | 1 per round per user |
| `suggestion` | 5 per debate |
| `breakpoint_resolve` | 1 per breakpoint |
| `ping` | 1 per 30 seconds |

## Error Codes

| Code | Description |
|------|-------------|
| `DEBATE_NOT_FOUND` | Debate ID does not exist |
| `UNAUTHORIZED` | Invalid or missing authentication |
| `DEBATE_ENDED` | Cannot interact with ended debate |
| `RATE_LIMITED` | Too many requests |
| `INVALID_EVENT` | Malformed event data |

## Related Documentation

- [API Reference](API_REFERENCE.md)
- [Error Codes](ERROR_CODES.md)
- [SDK Documentation](../aragora-js/README.md)
