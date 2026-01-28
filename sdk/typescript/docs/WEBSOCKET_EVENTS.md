# WebSocket Events Reference

This document describes all WebSocket events available in the Aragora TypeScript SDK for real-time debate streaming and monitoring.

## Connection

### Connecting to WebSocket

```typescript
import { createClient } from '@aragora/sdk';

const client = createClient({
  baseUrl: 'https://api.aragora.ai',
  apiKey: 'your-api-key',
});

// Stream a debate
const stream = client.streamDebate({
  debateId: 'debate-123',
  onEvent: (event) => console.log(event),
  onError: (error) => console.error(error),
});

// Stop streaming
stream.close();
```

---

## Core Debate Events

### `connected`
Emitted when WebSocket connection is established.

```typescript
interface ConnectedEvent {
  type: 'connected';
  debate_id: string;
}
```

### `debate_start`
Emitted when a debate begins.

```typescript
interface DebateStartEvent {
  type: 'debate_start';
  debate_id: string;
  task: string;
  agents: string[];
  protocol: DebateProtocol;
}
```

### `round_start` / `round_end`
Emitted at the beginning and end of each debate round.

```typescript
interface RoundEvent {
  type: 'round_start' | 'round_end';
  debate_id: string;
  round_number: number;
}
```

### `agent_message`
Emitted when an agent produces a message.

```typescript
interface AgentMessageEvent {
  type: 'agent_message';
  debate_id: string;
  agent: string;
  content: string;
  round_number: number;
  role: 'propose' | 'critique' | 'revision' | 'synthesis';
}
```

### `consensus` / `consensus_reached`
Emitted when agents reach consensus.

```typescript
interface ConsensusEvent {
  type: 'consensus' | 'consensus_reached';
  debate_id: string;
  conclusion: string;
  confidence: number;
  supporting_agents: string[];
}
```

### `debate_end`
Emitted when debate concludes.

```typescript
interface DebateEndEvent {
  type: 'debate_end';
  debate_id: string;
  status: 'completed' | 'failed' | 'cancelled';
  conclusion?: string;
  duration_seconds: number;
}
```

---

## Token Streaming Events

Real-time token-by-token streaming for agent responses.

### `token_start`
Emitted when an agent begins generating a response.

```typescript
interface TokenStartEvent {
  debate_id: string;
  agent: string;
  round_number: number;
}
```

### `token_delta`
Emitted for each token generated (streaming).

```typescript
interface TokenDeltaEvent {
  debate_id: string;
  agent: string;
  delta: string;  // The new token(s)
  seq: number;    // Sequence number for ordering
}
```

### `token_end`
Emitted when agent response is complete.

```typescript
interface TokenEndEvent {
  debate_id: string;
  agent: string;
  full_content: string;
}
```

---

## Preview Events

Fast early signals during debate initialization.

### `quick_classification`
Initial classification of the debate topic.

```typescript
interface QuickClassificationEvent {
  debate_id: string;
  classification: string;
  confidence: number;
}
```

### `agent_preview`
Preview of agents that will participate.

```typescript
interface AgentPreviewEvent {
  debate_id: string;
  agents: Array<{ name: string; role: string; stance?: string }>;
}
```

### `context_preview`
Preview of relevant context being used.

```typescript
interface ContextPreviewEvent {
  debate_id: string;
  context_summary: string;
  relevant_knowledge_count: number;
}
```

---

## Audience Analytics Events

Real-time audience participation metrics.

### `audience_summary`
Summary of audience participation.

```typescript
interface AudienceSummaryEvent {
  debate_id: string;
  total_votes: number;
  suggestions: number;
}
```

### `audience_metrics`
Detailed audience engagement metrics.

```typescript
interface AudienceMetricsEvent {
  debate_id: string;
  vote_distribution: Record<string, number>;
  engagement_score: number;
}
```

### `user_vote` / `audience_suggestion`
Individual user participation events.

```typescript
interface UserVoteEvent {
  debate_id: string;
  user_id: string;
  vote: string;
  round_number: number;
}
```

---

## Leaderboard & ELO Events

Agent ranking and performance updates.

### `leaderboard_update`
Updated agent rankings.

```typescript
interface LeaderboardUpdateEvent {
  rankings: Array<{ agent: string; elo: number; rank: number }>;
}
```

### `agent_elo_updated`
Individual agent ELO change.

```typescript
interface AgentEloUpdatedEvent {
  agent: string;
  old_elo: number;
  new_elo: number;
  change: number;
  debate_id: string;
}
```

### `agent_fallback_triggered`
When an agent fails and fallback is used.

```typescript
interface AgentFallbackTriggeredEvent {
  original_agent: string;
  fallback_agent: string;
  reason: string;
  debate_id: string;
}
```

### `moment_detected`
Significant debate moments (breakthroughs, comebacks).

```typescript
interface MomentDetectedEvent {
  debate_id: string;
  agent: string;
  moment_type: 'breakthrough' | 'comeback' | 'decisive_argument' | 'consensus_catalyst' | 'upset';
  description: string;
  impact_score: number;
}
```

---

## Nomic Loop Events

Self-improvement cycle monitoring.

### `cycle_start` / `cycle_end`
Nomic improvement cycle lifecycle.

```typescript
interface CycleStartEvent {
  loop_id: string;
  cycle_number: number;
  timestamp: string;
}

interface CycleEndEvent {
  loop_id: string;
  cycle_number: number;
  duration_seconds: number;
  status: 'completed' | 'failed' | 'cancelled';
}
```

### `phase_start` / `phase_end`
Individual phase within a cycle.

```typescript
interface PhaseStartEvent {
  loop_id: string;
  phase: string;  // 'context' | 'debate' | 'design' | 'implement' | 'verify'
  cycle_number: number;
}
```

### `task_start` / `task_complete` / `task_retry`
Task execution events.

```typescript
interface TaskCompleteEvent {
  loop_id: string;
  task_id: string;
  status: 'completed' | 'failed';
  result?: Record<string, unknown>;
}
```

---

## Memory Events

Memory system activity.

### `memory_recall`
When memories are retrieved.

```typescript
interface MemoryRecallEvent {
  debate_id?: string;
  query: string;
  results_count: number;
  tiers: string[];  // 'fast' | 'medium' | 'slow' | 'glacial'
}
```

### `insight_extracted`
New insight extracted from debate.

```typescript
interface InsightExtractedEvent {
  debate_id: string;
  insight: string;
  confidence: number;
  source_round: number;
}
```

---

## Belief Network Events

Reasoning and belief propagation.

### `belief_converged`
When a belief reaches stable confidence.

```typescript
interface BeliefConvergedEvent {
  debate_id: string;
  claim: string;
  final_confidence: number;
  supporting_agents: string[];
}
```

### `crux_detected`
Key disagreement point identified.

```typescript
interface CruxDetectedEvent {
  debate_id: string;
  crux: string;
  agents_involved: string[];
  round_detected: number;
}
```

---

## Progress & Error Events

### `phase_progress`
Progress updates during long operations.

```typescript
interface PhaseProgressEvent {
  debate_id?: string;
  loop_id?: string;
  phase: string;
  progress_pct: number;
  eta_seconds?: number;
}
```

### `error` / `agent_error`
Error notifications.

```typescript
interface AgentErrorEvent {
  debate_id: string;
  agent: string;
  error_code: string;
  error_message: string;
  recoverable: boolean;
}
```

### `warning`
Non-fatal warnings.

```typescript
interface WarningEvent {
  debate_id?: string;
  code: string;
  message: string;
}
```

### `heartbeat`
Connection keepalive (every 30 seconds).

---

## Event Handling Example

```typescript
import { createClient, WebSocketEventType } from '@aragora/sdk';

const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'key' });

const handlers: Partial<Record<WebSocketEventType, (e: any) => void>> = {
  debate_start: (e) => console.log(`Debate started: ${e.task}`),
  token_delta: (e) => process.stdout.write(e.delta),
  token_end: (e) => console.log('\n--- Response complete ---'),
  consensus_reached: (e) => console.log(`Consensus: ${e.conclusion}`),
  agent_error: (e) => console.error(`Agent ${e.agent} error: ${e.error_message}`),
  error: (e) => console.error(`Error: ${e.message}`),
};

const stream = client.streamDebate({
  debateId: 'debate-123',
  onEvent: (event) => handlers[event.type]?.(event),
});
```

---

## Performance & Reliability

| Metric | Target |
|--------|--------|
| Connection latency | < 100ms |
| Event delivery latency | < 50ms |
| Availability | 99.9% |
| Automatic reconnection | Yes (exponential backoff) |

The SDK automatically handles reconnection with exponential backoff (1s, 2s, 4s, 8s, max 30s).
