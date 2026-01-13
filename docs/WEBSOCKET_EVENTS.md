# WebSocket Events Reference

This document covers control messages and stream events sent by the Aragora
WebSocket server. Stream events use the `StreamEvent` envelope from
`aragora/server/stream/events.py`.

## Connection

### Endpoint

```
ws://localhost:8765/ws
wss://api.aragora.ai/ws  # Production
```

The WebSocket server accepts `/` or `/ws`. The live UI uses `/ws`.

### Authentication

If `ARAGORA_API_TOKEN` is set, include an `Authorization: Bearer` header during
the handshake:

```bash
wscat -c ws://localhost:8765/ws -H "Authorization: Bearer $ARAGORA_API_TOKEN"
```

Browser clients cannot set custom headers; use a server-side proxy if auth is
enforced.

### Initial Messages

On connection, the server sends:

- `connection_info` (auth and client metadata)
- `loop_list` (active loops/debates)
- `sync` for each cached loop state

Example:

```json
{
  "type": "connection_info",
  "data": {
    "authenticated": false,
    "client_id": "abc123...",
    "write_access": true
  }
}
```

## Client → Server Messages

### Request loop list

```json
{ "type": "get_loops" }
```

### Submit a vote

```json
{
  "type": "user_vote",
  "loop_id": "loop_abc123",
  "payload": {
    "choice": "Option A",
    "intensity": 7
  }
}
```

### Submit a suggestion

```json
{
  "type": "user_suggestion",
  "loop_id": "loop_abc123",
  "payload": {
    "text": "Consider a phased rollout for safety."
  }
}
```

### Submit audience wisdom

```json
{
  "type": "wisdom_submission",
  "loop_id": "loop_abc123",
  "text": "Start with a rollback plan and chaos testing."
}
```

The server replies with `ack`, `error`, or `auth_revoked` depending on the
request and auth state.

## Stream Event Envelope

Stream events (emitted by `SyncEventEmitter`) share a common envelope:

```json
{
  "type": "consensus",
  "data": { "reached": true, "confidence": 0.88, "answer": "Use token bucket" },
  "timestamp": 1732735053.123,
  "round": 3,
  "agent": "anthropic-api",
  "seq": 42,
  "agent_seq": 7,
  "loop_id": "loop_abc123"
}
```

Notes:
- `timestamp` is epoch seconds (float).
- `loop_id` is present when the emitter is scoped to a debate/loop.
- `round`, `agent`, and `agent_seq` are populated when relevant.

## Common Event Payloads

These payloads are emitted by `create_arena_hooks()`:

```json
{ "type": "debate_start", "data": { "task": "...", "agents": ["a", "b"] } }
{ "type": "round_start", "data": { "round": 1 } }
{ "type": "agent_message", "data": { "content": "...", "role": "proposal" } }
{ "type": "critique", "data": { "target": "...", "issues": ["..."], "severity": 0.4, "content": "..." } }
{ "type": "vote", "data": { "vote": "...", "confidence": 0.72 } }
{ "type": "consensus", "data": { "reached": true, "confidence": 0.85, "answer": "..." } }
{ "type": "debate_end", "data": { "duration": 42.2, "rounds": 3 } }
```

Audience metrics emitted after `user_vote` look like:

```json
{
  "type": "audience_metrics",
  "data": {
    "votes": { "Option A": 3 },
    "weighted_votes": { "Option A": 4.5 },
    "suggestions": 1,
    "total": 4,
    "histograms": { "Option A": { "1": 0, "2": 1, "3": 0 } },
    "conviction_distribution": { "1": 0, "2": 1, "3": 0 }
  }
}
```

## Stream Event Types

These are the canonical event names from `StreamEventType`.

### Debate lifecycle
- `debate_start`
- `round_start`
- `agent_message`
- `critique`
- `vote`
- `consensus`
- `debate_end`

### Token streaming
- `token_start`
- `token_delta`
- `token_end`

### Nomic loop
- `cycle_start`
- `cycle_end`
- `phase_start`
- `phase_end`
- `task_start`
- `task_complete`
- `task_retry`
- `verification_start`
- `verification_result`
- `commit`
- `backup_created`
- `backup_restored`
- `error`
- `log_message`

### Multi-loop management
- `loop_register`
- `loop_unregister`
- `loop_list`

### Audience participation
- `user_vote`
- `user_suggestion`
- `audience_summary`
- `audience_metrics`
- `audience_drain`

### Memory & learning
- `memory_recall`
- `insight_extracted`

### Rankings & leaderboard
- `match_recorded`
- `leaderboard_update`
- `grounded_verdict`
- `moment_detected`
- `agent_elo_updated`

### Claim verification
- `claim_verification_result`
- `formal_verification_result`

### Memory tiers
- `memory_tier_promotion`
- `memory_tier_demotion`

### Graph debates
- `graph_node_added`
- `graph_branch_created`
- `graph_branch_merged`

### Position tracking
- `flip_detected`

### Feature integration
- `trait_emerged`
- `risk_warning`
- `evidence_found`
- `calibration_update`
- `genesis_evolution`
- `training_data_exported`

### Rhetorical analysis
- `rhetorical_observation`

### Trickster events
- `hollow_consensus`
- `trickster_intervention`

### Breakpoints
- `breakpoint`
- `breakpoint_resolved`

### Mood/sentiment
- `mood_detected`
- `mood_shift`
- `debate_energy`

### Capability probes
- `probe_start`
- `probe_result`
- `probe_complete`

### Deep audit
- `audit_start`
- `audit_round`
- `audit_finding`
- `audit_cross_exam`
- `audit_verdict`

### Telemetry
- `telemetry_thought`
- `telemetry_capability`
- `telemetry_redaction`
- `telemetry_diagnostic`

### Gauntlet
- `gauntlet_start`
- `gauntlet_phase`
- `gauntlet_agent_active`
- `gauntlet_attack`
- `gauntlet_finding`
- `gauntlet_probe`
- `gauntlet_verification`
- `gauntlet_risk`
- `gauntlet_progress`
- `gauntlet_verdict`
- `gauntlet_complete`

Some events are emitted only when the corresponding feature is enabled. Use
`loop_list` and `sync` control messages to seed UI state before processing
stream events.
