# WebSocket Events Reference

This document covers control messages and stream events sent by the Aragora
WebSocket server. Stream events use the `StreamEvent` envelope from
`aragora/events/types.py` (re-exported by `aragora/server/stream/events.py`).

## Connection

### Endpoint

```
ws://localhost:8765/ws
wss://api.aragora.ai/ws  # Production
```

The WebSocket server accepts `/` or `/ws`. The live UI uses `/ws`.
The server broadcasts all events to all connected clients, so clients should
filter by `loop_id` (or `data.debate_id`) when they only care about one debate.

### Authentication

If `ARAGORA_API_TOKEN` is set, include an `Authorization: Bearer` header during
the handshake:

```bash
wscat -c ws://localhost:8765/ws -H "Authorization: Bearer $ARAGORA_API_TOKEN"
```

Browser clients cannot set custom headers; the server accepts a `token` query
parameter for WebSocket authentication (e.g. `wss://.../ws?token=...`).

**SDK note:** The Python streaming client (`AragoraWebSocket`) appends the API
key as a `token` query parameter. Header-based auth is still supported for
proxies or non-browser clients.

### Initial Messages

On connection, the server sends:

- `connection_info` (auth and client metadata)
- `loop_list` (active loops/debates)
- `sync` for each cached loop state (DebateStreamServer on port 8765 only)

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
{ "type": "agent_error", "data": { "error_type": "timeout", "message": "...", "recoverable": true, "phase": "proposal" } }
{ "type": "vote", "data": { "vote": "...", "confidence": 0.72 } }
{ "type": "consensus", "data": { "reached": true, "confidence": 0.85, "answer": "...", "status": "consensus_reached", "agent_failures": {} } }
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

### Agent Failure Tracking

Consensus events may include `status` and `agent_failures`:

- `status`: `consensus_reached`, `completed`, or `insufficient_participation`
- `agent_failures`: map of agent → list of failure records

Each failure record includes:

```json
{
  "phase": "proposal",
  "error_type": "timeout",
  "message": "Agent response was empty",
  "provider": "openai",
  "timestamp": 1732735053.123
}
```

## Stream Event Types

Canonical event names are defined in `aragora/events/types.py` (the server re-exports them via `aragora/server/stream/events.py`).
The list below is the full `StreamEventType` catalog, grouped by the category comments in code (197 event types as of 2026-02-03).

### Debate events
- `debate_start`
- `round_start`
- `agent_message`
- `critique`
- `vote`
- `consensus`
- `synthesis`
- `debate_end`

### Quick preview events (shown in first 5 seconds of debate initialization)
- `quick_classification`
- `agent_preview`
- `context_preview`

### Token streaming events (for real-time response display)
- `token_start`
- `token_delta`
- `token_end`

### Nomic loop events
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
- `phase_timeout`
- `log_message`

### Multi-loop management events
- `loop_register`
- `loop_unregister`
- `loop_list`

### Audience participation events
- `user_vote`
- `user_suggestion`
- `audience_summary`
- `audience_metrics`
- `audience_drain`

### Memory/learning events
- `memory_recall`
- `insight_extracted`
- `memory_stored`
- `memory_retrieved`

### Ranking/leaderboard events (debate consensus feature)
- `match_recorded`
- `leaderboard_update`
- `grounded_verdict`
- `moment_detected`
- `agent_elo_updated`
- `agent_calibration_changed`
- `agent_fallback_triggered`

### Knowledge Mound events (cross-pollination)
- `knowledge_indexed`
- `knowledge_queried`
- `mound_updated`
- `knowledge_stale`
- `km_batch`

### Belief Network events (bidirectional KM integration)
- `belief_converged`
- `crux_detected`

### KM Adapter sync events (bidirectional tracking)
- `km_adapter_forward_sync`
- `km_adapter_reverse_query`
- `km_adapter_validation`

### RLM events (bidirectional KM integration)
- `rlm_compression_complete`

### Claim verification events
- `claim_verification_result`

### Memory tier events
- `memory_tier_promotion`
- `memory_tier_demotion`

### Graph debate events (branching/merging visualization)
- `graph_node_added`
- `graph_branch_created`
- `graph_branch_merged`

### Position tracking events
- `flip_detected`

### Feature integration events (data flow from backends to panels)
- `trait_emerged`
- `risk_warning`
- `evidence_found`
- `calibration_update`
- `genesis_evolution`
- `training_data_exported`
- `selection_feedback`
- `memory_coordination`

### Rhetorical analysis events
- `rhetorical_observation`

### Trickster/hollow consensus events
- `hollow_consensus`
- `trickster_intervention`

### Human intervention breakpoint events
- `breakpoint`
- `breakpoint_resolved`

### Progress/heartbeat events (for detecting stalled debates)
- `heartbeat`
- `agent_error`
- `phase_progress`

### Mood/sentiment events (Real-Time Debate Drama)
- `mood_detected`
- `mood_shift`
- `debate_energy`

### Capability probe events (Adversarial Testing)
- `probe_start`
- `probe_result`
- `probe_complete`

### Deep Audit events (Intensive Multi-Round Analysis)
- `audit_start`
- `audit_round`
- `audit_finding`
- `audit_cross_exam`
- `audit_verdict`

### Telemetry events (Cognitive Firewall)
- `telemetry_thought`
- `telemetry_capability`
- `telemetry_redaction`
- `telemetry_diagnostic`

### Gauntlet events (Adversarial Validation)
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

### Phase 2: Workflow Builder Events
- `workflow_created`
- `workflow_updated`
- `workflow_deleted`
- `workflow_start`
- `workflow_step_start`
- `workflow_step_progress`
- `workflow_step_complete`
- `workflow_step_failed`
- `workflow_step_skipped`
- `workflow_transition`
- `workflow_checkpoint`
- `workflow_resumed`
- `workflow_human_approval_required`
- `workflow_human_approval_received`
- `workflow_human_approval_timeout`
- `workflow_debate_start`
- `workflow_debate_round`
- `workflow_debate_complete`
- `workflow_memory_read`
- `workflow_memory_write`
- `workflow_complete`
- `workflow_failed`
- `workflow_terminated`
- `workflow_metrics`

### Voice/Transcription events (Speech-to-Text and Text-to-Speech)
- `voice_start`
- `voice_chunk`
- `voice_transcript`
- `voice_end`
- `voice_response`
- `voice_response_start`
- `voice_response_end`
- `transcription_queued`
- `transcription_started`
- `transcription_progress`
- `transcription_complete`
- `transcription_failed`

### 5.1 Approval Flow events
- `approval_requested`
- `approval_approved`
- `approval_rejected`
- `approval_timeout`
- `approval_auto_approved`

### 5.1 Rollback events
- `rollback_point_created`
- `rollback_executed`

### 5.1 Verification events (improvement cycle)
- `improvement_cycle_start`
- `improvement_cycle_verified`
- `improvement_cycle_failed`
- `improvement_cycle_complete`

### 5.2 Continuous Learning events
- `learning_event`
- `elo_updated`
- `pattern_discovered`
- `calibration_updated`
- `knowledge_decayed`

### 5.3 Alert events
- `alert_created`
- `alert_acknowledged`
- `alert_resolved`
- `alert_escalated`

### 5.3 Trigger events
- `trigger_added`
- `trigger_removed`
- `trigger_executed`
- `trigger_scheduler_start`
- `trigger_scheduler_stop`

### 5.3 Monitoring events
- `trend_detected`
- `anomaly_detected`
- `metric_recorded`

### Explainability Events (real-time explanation generation)
- `explainability_started`
- `explainability_factors`
- `explainability_counterfactual`
- `explainability_provenance`
- `explainability_narrative`
- `explainability_complete`

### Workflow Template Events (template execution updates)
- `template_execution_started`
- `template_execution_progress`
- `template_execution_step`
- `template_execution_complete`
- `template_execution_failed`
- `template_instantiated`

### Gauntlet Receipt Events (receipt lifecycle updates)
- `receipt_generated`
- `receipt_verified`
- `receipt_exported`
- `receipt_shared`
- `receipt_integrity_failed`

### KM Resilience Events (real-time resilience status)
- `km_circuit_breaker_state`
- `km_retry_exhausted`
- `km_cache_invalidated`
- `km_integrity_error`

### Connector Webhook Events (external service notifications)
- `connector_webhook_received`
- `connector_pagerduty_incident`
- `connector_plaid_transaction_sync`
- `connector_qbo_webhook`

## Selected Payload Examples

### Explainability (2026-01-20)
Real-time events during explanation generation:
- `explainability_started` - Explanation generation started for debate
- `explainability_factors` - Contributing factors have been computed
- `explainability_counterfactual` - Counterfactual scenario generated
- `explainability_provenance` - Decision provenance chain built
- `explainability_narrative` - Natural language narrative ready
- `explainability_complete` - Full explanation ready

Example:
```json
{
  "type": "explainability_factors",
  "data": {
    "debate_id": "debate-123",
    "factors_count": 8,
    "top_factor": {
      "name": "evidence_quality",
      "contribution": 0.35
    }
  },
  "timestamp": 1737356400.0
}
```

### Workflow Templates (2026-01-20)
Template execution lifecycle events:
- `template_execution_started` - Template execution began
- `template_execution_progress` - Progress update (percentage, current step)
- `template_execution_step` - Individual step completed
- `template_execution_complete` - Execution finished successfully
- `template_execution_failed` - Execution failed with error
- `template_instantiated` - New template created from pattern

Example:
```json
{
  "type": "template_execution_progress",
  "data": {
    "execution_id": "exec-456",
    "template_id": "security/code-review",
    "progress_pct": 45,
    "current_step": "vulnerability_scan",
    "steps_completed": 3,
    "steps_total": 7
  },
  "timestamp": 1737356400.0
}
```

### Gauntlet Receipts (2026-01-20)
Receipt lifecycle events:
- `receipt_generated` - New receipt generated for completed gauntlet
- `receipt_verified` - Receipt integrity successfully verified
- `receipt_exported` - Receipt exported to format (json/html/md/sarif)
- `receipt_shared` - Shareable link created for receipt
- `receipt_integrity_failed` - Receipt integrity verification failed

Example:
```json
{
  "type": "receipt_generated",
  "data": {
    "receipt_id": "receipt-789",
    "debate_id": "debate-123",
    "verdict": "pass",
    "confidence": 0.92,
    "hash": "sha256:a1b2c3..."
  },
  "timestamp": 1737356400.0
}
```

### KM Resilience (2026-01-20)
Real-time resilience status events:
- `km_circuit_breaker_state` - Circuit breaker state changed
- `km_retry_exhausted` - All retries exhausted for operation
- `km_cache_invalidated` - Cache was invalidated
- `km_integrity_error` - Data integrity error detected

Example:
```json
{
  "type": "km_circuit_breaker_state",
  "data": {
    "service": "postgres",
    "state": "open",
    "previous_state": "closed",
    "failures": 5,
    "recovery_at": "2026-01-20T11:30:00Z"
  },
  "timestamp": 1737356400.0
}
```

Some events are emitted only when the corresponding feature is enabled. Use
`loop_list` and `sync` control messages to seed UI state before processing
stream events.

## Control Plane Stream

The control plane provides a separate WebSocket stream for agent orchestration
and task lifecycle events. This is useful for monitoring dashboards and
coordination UIs.

### Endpoint

```
ws://localhost:8766/api/control-plane/stream
ws://localhost:8766/ws/control-plane
```

### Connection Flow

On connection, the server sends a `connected` event:

```json
{
  "type": "connected",
  "timestamp": 1732735053.123,
  "data": {
    "message": "Connected to control plane stream"
  }
}
```

### Agent Events

Events related to agent lifecycle and status:

#### `agent_registered`
Emitted when an agent registers with the control plane.

```json
{
  "type": "agent_registered",
  "timestamp": 1732735053.123,
  "data": {
    "agent_id": "agent-abc123",
    "capabilities": ["debate", "critique", "summarize"],
    "model": "claude-3-opus",
    "provider": "anthropic"
  }
}
```

#### `agent_unregistered`
Emitted when an agent unregisters or is removed.

```json
{
  "type": "agent_unregistered",
  "timestamp": 1732735053.123,
  "data": {
    "agent_id": "agent-abc123",
    "reason": "graceful_shutdown"
  }
}
```

#### `agent_status_changed`
Emitted when an agent's status changes (idle, busy, etc.).

```json
{
  "type": "agent_status_changed",
  "timestamp": 1732735053.123,
  "data": {
    "agent_id": "agent-abc123",
    "old_status": "idle",
    "new_status": "busy"
  }
}
```

#### `agent_heartbeat`
Periodic heartbeat from an agent.

```json
{
  "type": "agent_heartbeat",
  "timestamp": 1732735053.123,
  "data": {
    "agent_id": "agent-abc123",
    "status": "idle",
    "tasks_completed": 42
  }
}
```

#### `agent_timeout`
Emitted when an agent fails to send heartbeats within the expected interval.

```json
{
  "type": "agent_timeout",
  "timestamp": 1732735053.123,
  "data": {
    "agent_id": "agent-abc123",
    "last_heartbeat": 1732734053.123
  }
}
```

### Task Events

Events related to task lifecycle:

#### `task_submitted`
Emitted when a new task is submitted to the scheduler.

```json
{
  "type": "task_submitted",
  "timestamp": 1732735053.123,
  "data": {
    "task_id": "task-xyz789",
    "task_type": "debate",
    "priority": "high",
    "required_capabilities": ["debate", "critique"]
  }
}
```

#### `task_claimed`
Emitted when an agent claims a task.

```json
{
  "type": "task_claimed",
  "timestamp": 1732735053.123,
  "data": {
    "task_id": "task-xyz789",
    "agent_id": "agent-abc123"
  }
}
```

#### `task_started`
Emitted when an agent begins executing a task.

```json
{
  "type": "task_started",
  "timestamp": 1732735053.123,
  "data": {
    "task_id": "task-xyz789",
    "agent_id": "agent-abc123"
  }
}
```

#### `task_completed`
Emitted when a task completes successfully.

```json
{
  "type": "task_completed",
  "timestamp": 1732735053.123,
  "data": {
    "task_id": "task-xyz789",
    "agent_id": "agent-abc123",
    "result_summary": "Debate concluded with consensus..."
  }
}
```

#### `task_failed`
Emitted when a task fails.

```json
{
  "type": "task_failed",
  "timestamp": 1732735053.123,
  "data": {
    "task_id": "task-xyz789",
    "agent_id": "agent-abc123",
    "error": "Agent timeout during execution",
    "retries_left": 2
  }
}
```

#### `task_cancelled`
Emitted when a task is cancelled.

```json
{
  "type": "task_cancelled",
  "timestamp": 1732735053.123,
  "data": {
    "task_id": "task-xyz789",
    "reason": "user_requested"
  }
}
```

#### `task_retrying`
Emitted when a failed task is being retried.

```json
{
  "type": "task_retrying",
  "timestamp": 1732735053.123,
  "data": {
    "task_id": "task-xyz789",
    "attempt": 2,
    "max_attempts": 3
  }
}
```

#### `task_dead_lettered`
Emitted when a task exhausts retries and is moved to the dead letter queue.

```json
{
  "type": "task_dead_lettered",
  "timestamp": 1732735053.123,
  "data": {
    "task_id": "task-xyz789",
    "reason": "Max retries exceeded"
  }
}
```

### System Events

Events related to system health and metrics:

#### `health_update`
Periodic system health status.

```json
{
  "type": "health_update",
  "timestamp": 1732735053.123,
  "data": {
    "status": "healthy",
    "agents": {
      "total": 5,
      "idle": 3,
      "busy": 2
    }
  }
}
```

#### `metrics_update`
System metrics update.

```json
{
  "type": "metrics_update",
  "timestamp": 1732735053.123,
  "data": {
    "tasks_per_minute": 12.5,
    "avg_task_duration": 4.2,
    "queue_depth": 8
  }
}
```

#### `scheduler_stats`
Detailed scheduler statistics.

```json
{
  "type": "scheduler_stats",
  "timestamp": 1732735053.123,
  "data": {
    "pending_tasks": 15,
    "running_tasks": 4,
    "completed_today": 342,
    "failed_today": 3,
    "avg_wait_time": 1.2
  }
}
```

### Error Events

#### `error`
System error notification.

```json
{
  "type": "error",
  "timestamp": 1732735053.123,
  "data": {
    "error": "Redis connection lost",
    "context": {
      "component": "scheduler",
      "retry_in": 5
    }
  }
}
```

### Client Messages

Clients can send the following messages to the control plane stream:

#### Ping
Keep-alive ping:

```json
{ "type": "ping" }
```

Server responds with:

```json
{ "type": "pong", "timestamp": 1732735053.123 }
```

#### Subscribe (Future)
Subscribe to specific event types (not yet implemented):

```json
{
  "type": "subscribe",
  "events": ["task_completed", "task_failed"]
}
```

### Example: JavaScript Client

```javascript
const ws = new WebSocket('ws://localhost:8766/api/control-plane/stream');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch (data.type) {
    case 'agent_registered':
      console.log(`Agent ${data.data.agent_id} joined`);
      break;
    case 'task_completed':
      console.log(`Task ${data.data.task_id} completed`);
      break;
    case 'task_failed':
      console.error(`Task ${data.data.task_id} failed: ${data.data.error}`);
      break;
    case 'health_update':
      updateDashboard(data.data);
      break;
  }
};

// Keep connection alive
setInterval(() => {
  ws.send(JSON.stringify({ type: 'ping' }));
}, 30000);
```
