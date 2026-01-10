# WebSocket Events Reference

This document describes all WebSocket events emitted by the Aragora streaming server. Events enable real-time monitoring of debates, nomic loops, audits, and other system activities.

## Table of Contents

- [Connection](#connection)
- [Event Structure](#event-structure)
- [Event Categories](#event-categories)
  - [Debate Events](#debate-events)
  - [Token Streaming Events](#token-streaming-events)
  - [Nomic Loop Events](#nomic-loop-events)
  - [Multi-Loop Management Events](#multi-loop-management-events)
  - [Audience Participation Events](#audience-participation-events)
  - [Memory/Learning Events](#memorylearning-events)
  - [Ranking/Leaderboard Events](#rankingleaderboard-events)
  - [Graph Debate Events](#graph-debate-events)
  - [Position Tracking Events](#position-tracking-events)
  - [Rhetorical Analysis Events](#rhetorical-analysis-events)
  - [Breakpoint Events](#breakpoint-events)
  - [Mood/Sentiment Events](#moodsentiment-events)
  - [Capability Probe Events](#capability-probe-events)
  - [Deep Audit Events](#deep-audit-events)
  - [Telemetry Events](#telemetry-events)
- [Message Sequences](#message-sequences)
- [Reconnection Guidance](#reconnection-guidance)
- [Client Implementation Examples](#client-implementation-examples)

---

## Connection

### WebSocket Endpoint

```
ws://localhost:8080/ws/debates
```

### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `loop_id` | string | Optional. Subscribe to a specific loop instance |
| `token` | string | Optional. Authentication token |

### Example Connection

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/debates?loop_id=abc123');

ws.onopen = () => {
  console.log('Connected to debate stream');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Event:', data.type, data.data);
};
```

---

## Event Structure

All events follow a consistent JSON structure:

```typescript
interface StreamEvent {
  type: string;           // Event type identifier
  data: object;           // Event-specific payload
  timestamp: number;      // Unix timestamp (seconds)
  round: number;          // Current debate round (0 if N/A)
  agent: string;          // Agent name (empty if N/A)
  loop_id?: string;       // Loop instance ID (for multi-loop)
  seq: number;            // Global sequence number
  agent_seq: number;      // Per-agent sequence number (for token ordering)
}
```

---

## Event Categories

### Debate Events

Core events during debate execution.

#### `debate_start`

Emitted when a debate session begins.

```json
{
  "type": "debate_start",
  "data": {
    "task": "Should we implement feature X?",
    "agents": ["claude-api", "gpt-4", "gemini"]
  },
  "timestamp": 1704672000.123,
  "round": 0,
  "agent": ""
}
```

#### `round_start`

Emitted when a new debate round begins.

```json
{
  "type": "round_start",
  "data": {
    "round": 1
  },
  "timestamp": 1704672001.456,
  "round": 1,
  "agent": ""
}
```

#### `agent_message`

Emitted when an agent produces a message (proposal, response, refinement).

```json
{
  "type": "agent_message",
  "data": {
    "content": "I propose we implement feature X because...",
    "role": "proposer"
  },
  "timestamp": 1704672010.789,
  "round": 1,
  "agent": "claude-api"
}
```

#### `critique`

Emitted when a critic agent evaluates a proposal.

```json
{
  "type": "critique",
  "data": {
    "target": "claude-api",
    "issues": [
      "Doesn't consider edge case Y",
      "Performance implications unclear"
    ],
    "severity": 0.6,
    "content": "• Doesn't consider edge case Y\n• Performance implications unclear"
  },
  "timestamp": 1704672020.123,
  "round": 1,
  "agent": "gpt-4"
}
```

#### `vote`

Emitted when an agent casts a vote.

```json
{
  "type": "vote",
  "data": {
    "vote": "approve",
    "confidence": 0.85
  },
  "timestamp": 1704672030.456,
  "round": 2,
  "agent": "gemini"
}
```

#### `consensus`

Emitted when consensus is reached or attempted.

```json
{
  "type": "consensus",
  "data": {
    "reached": true,
    "confidence": 0.92,
    "answer": "The final consensus is to implement feature X with modifications..."
  },
  "timestamp": 1704672040.789,
  "round": 3,
  "agent": ""
}
```

#### `debate_end`

Emitted when a debate completes.

```json
{
  "type": "debate_end",
  "data": {
    "duration": 45.6,
    "rounds": 3
  },
  "timestamp": 1704672045.123,
  "round": 3,
  "agent": ""
}
```

---

### Token Streaming Events

Real-time token-by-token streaming of agent responses.

#### `token_start`

Emitted when an agent begins generating a response.

```json
{
  "type": "token_start",
  "data": {
    "debate_id": "debate_abc123",
    "agent": "claude-api",
    "timestamp": "2024-01-08T12:00:00.000Z"
  },
  "agent": "claude-api"
}
```

#### `token_delta`

Emitted for each token/chunk received from the agent.

```json
{
  "type": "token_delta",
  "data": {
    "debate_id": "debate_abc123",
    "agent": "claude-api",
    "token": "I think"
  },
  "agent": "claude-api",
  "agent_seq": 1
}
```

#### `token_end`

Emitted when an agent finishes generating.

```json
{
  "type": "token_end",
  "data": {
    "debate_id": "debate_abc123",
    "agent": "claude-api",
    "full_response": "I think we should implement this feature because...",
    "error": null
  },
  "agent": "claude-api"
}
```

---

### Nomic Loop Events

Events from the autonomous self-improvement cycle.

#### `cycle_start`

Emitted when a nomic cycle begins.

```json
{
  "type": "cycle_start",
  "data": {
    "cycle": 1,
    "max_cycles": 5,
    "started_at": "2024-01-08T12:00:00Z"
  }
}
```

#### `cycle_end`

Emitted when a nomic cycle completes.

```json
{
  "type": "cycle_end",
  "data": {
    "cycle": 1,
    "success": true,
    "duration_seconds": 120.5,
    "outcome": "Implemented 3 improvements"
  }
}
```

#### `phase_start`

Emitted when a phase begins (debate, design, implement, verify, commit).

```json
{
  "type": "phase_start",
  "data": {
    "phase": "implement",
    "cycle": 1,
    "task_count": 5
  }
}
```

#### `phase_end`

Emitted when a phase completes.

```json
{
  "type": "phase_end",
  "data": {
    "phase": "implement",
    "cycle": 1,
    "success": true,
    "duration_seconds": 45.2,
    "tasks_completed": 5
  }
}
```

#### `task_start`

Emitted when an implementation task starts.

```json
{
  "type": "task_start",
  "data": {
    "task_id": "task_001",
    "description": "Add rate limiting to API endpoint",
    "complexity": "medium",
    "model": "claude-sonnet-4",
    "total_tasks": 5,
    "completed_tasks": 2
  }
}
```

#### `task_complete`

Emitted when an implementation task completes.

```json
{
  "type": "task_complete",
  "data": {
    "task_id": "task_001",
    "success": true,
    "duration_seconds": 15.3,
    "diff_preview": "+def rate_limit():\n+    ...",
    "error": null
  }
}
```

#### `task_retry`

Emitted when a task is being retried.

```json
{
  "type": "task_retry",
  "data": {
    "task_id": "task_001",
    "attempt": 2,
    "reason": "Syntax error in generated code",
    "timeout": 60
  }
}
```

#### `verification_start`

Emitted when verification phase begins.

```json
{
  "type": "verification_start",
  "data": {
    "checks": ["syntax", "tests", "lint", "type_check"]
  }
}
```

#### `verification_result`

Emitted for each verification check result.

```json
{
  "type": "verification_result",
  "data": {
    "check": "tests",
    "passed": true,
    "message": "All 42 tests passed"
  }
}
```

#### `commit`

Emitted when changes are committed.

```json
{
  "type": "commit",
  "data": {
    "commit_hash": "abc123def",
    "message": "feat: add rate limiting to API",
    "files_changed": 3
  }
}
```

#### `backup_created`

Emitted when a backup is created.

```json
{
  "type": "backup_created",
  "data": {
    "backup_name": "pre_implement_20240108_120000",
    "files_count": 15,
    "reason": "Before implementation phase"
  }
}
```

#### `backup_restored`

Emitted when a backup is restored.

```json
{
  "type": "backup_restored",
  "data": {
    "backup_name": "pre_implement_20240108_120000",
    "files_count": 15,
    "reason": "Verification failed"
  }
}
```

#### `error`

Emitted when an error occurs during nomic loop.

```json
{
  "type": "error",
  "data": {
    "phase": "implement",
    "message": "Failed to apply patch",
    "recoverable": true
  }
}
```

#### `log_message`

General log message for dashboard display.

```json
{
  "type": "log_message",
  "data": {
    "message": "Starting implementation of 5 tasks",
    "level": "info",
    "phase": "implement"
  },
  "agent": "codex"
}
```

---

### Multi-Loop Management Events

Events for managing multiple concurrent loop instances.

#### `loop_register`

Emitted when a new loop instance starts.

```json
{
  "type": "loop_register",
  "data": {
    "loop_id": "loop_abc123",
    "loop_type": "nomic",
    "started_at": "2024-01-08T12:00:00Z"
  }
}
```

#### `loop_unregister`

Emitted when a loop instance ends.

```json
{
  "type": "loop_unregister",
  "data": {
    "loop_id": "loop_abc123",
    "reason": "completed"
  }
}
```

#### `loop_list`

Sent on connection with list of active loops.

```json
{
  "type": "loop_list",
  "data": {
    "loops": [
      {"loop_id": "loop_abc123", "type": "nomic", "status": "running"},
      {"loop_id": "loop_def456", "type": "debate", "status": "running"}
    ]
  }
}
```

---

### Audience Participation Events

Events for interactive audience features.

#### `user_vote`

Emitted when an audience member votes.

```json
{
  "type": "user_vote",
  "data": {
    "user_id": "user_123",
    "choice": "option_a",
    "conviction": 0.8
  },
  "loop_id": "loop_abc123"
}
```

#### `user_suggestion`

Emitted when an audience member submits a suggestion.

```json
{
  "type": "user_suggestion",
  "data": {
    "user_id": "user_123",
    "suggestion": "Consider adding caching here",
    "context": "discussion about performance"
  },
  "loop_id": "loop_abc123"
}
```

#### `audience_summary`

Clustered summary of audience input.

```json
{
  "type": "audience_summary",
  "data": {
    "total_votes": 150,
    "vote_distribution": {"option_a": 0.6, "option_b": 0.4},
    "top_suggestions": ["Add caching", "Improve error handling"],
    "sentiment": 0.75
  }
}
```

#### `audience_metrics`

Real-time audience engagement metrics.

```json
{
  "type": "audience_metrics",
  "data": {
    "connected_users": 45,
    "votes_this_round": 23,
    "suggestions_this_round": 5,
    "average_conviction": 0.72
  }
}
```

#### `audience_drain`

Emitted when audience events are processed by arena.

```json
{
  "type": "audience_drain",
  "data": {
    "processed_count": 15,
    "votes_processed": 12,
    "suggestions_processed": 3
  }
}
```

---

### Memory/Learning Events

Events related to memory retrieval and insight extraction.

#### `memory_recall`

Emitted when historical context is retrieved.

```json
{
  "type": "memory_recall",
  "data": {
    "query": "rate limiting implementation",
    "memories": [
      {"id": "mem_001", "summary": "Previous rate limiter discussion", "relevance": 0.9}
    ],
    "total_recalled": 1
  }
}
```

#### `insight_extracted`

Emitted when a new insight is extracted from debate.

```json
{
  "type": "insight_extracted",
  "data": {
    "insight": "Token bucket is preferred over sliding window for our use case",
    "source_debate": "debate_abc123",
    "confidence": 0.85,
    "tags": ["architecture", "rate-limiting"]
  }
}
```

---

### Ranking/Leaderboard Events

Events for ELO ranking and leaderboard updates.

#### `match_recorded`

Emitted when an ELO match is recorded.

```json
{
  "type": "match_recorded",
  "data": {
    "debate_id": "debate_abc123",
    "participants": ["claude-api", "gpt-4"],
    "elo_changes": {"claude-api": +15, "gpt-4": -15},
    "domain": "coding",
    "winner": "claude-api"
  }
}
```

#### `leaderboard_update`

Periodic leaderboard snapshot.

```json
{
  "type": "leaderboard_update",
  "data": {
    "rankings": [
      {"agent": "claude-api", "elo": 1650, "rank": 1},
      {"agent": "gpt-4", "elo": 1620, "rank": 2}
    ],
    "domain": "coding",
    "timestamp": "2024-01-08T12:00:00Z"
  }
}
```

#### `grounded_verdict`

Evidence-backed verdict with citations.

```json
{
  "type": "grounded_verdict",
  "data": {
    "verdict": "Proposal A is better supported",
    "citations": [
      {"source": "RFC 7231", "quote": "...", "relevance": 0.9}
    ],
    "confidence": 0.88
  }
}
```

#### `moment_detected`

Significant narrative moment detected.

```json
{
  "type": "moment_detected",
  "data": {
    "moment_type": "position_reversal",
    "agent": "gpt-4",
    "description": "Changed stance from opposing to supporting",
    "significance": 0.85
  }
}
```

---

### Graph Debate Events

Events for branching/merging debate visualization.

#### `graph_node_added`

New node added to debate graph.

```json
{
  "type": "graph_node_added",
  "data": {
    "node_id": "node_001",
    "parent_id": "node_000",
    "content": "Alternative approach...",
    "agent": "claude-api"
  }
}
```

#### `graph_branch_created`

New branch created in debate.

```json
{
  "type": "graph_branch_created",
  "data": {
    "branch_id": "branch_001",
    "source_node": "node_001",
    "reason": "Exploring alternative"
  }
}
```

#### `graph_branch_merged`

Branches merged/synthesized.

```json
{
  "type": "graph_branch_merged",
  "data": {
    "merged_branches": ["branch_001", "branch_002"],
    "result_node": "node_010",
    "synthesis": "Combined the best aspects..."
  }
}
```

---

### Position Tracking Events

Events for tracking agent position changes.

#### `flip_detected`

Agent position reversal detected.

```json
{
  "type": "flip_detected",
  "data": {
    "agent": "gpt-4",
    "from_position": "oppose",
    "to_position": "support",
    "round": 3,
    "trigger": "New evidence presented"
  }
}
```

---

### Rhetorical Analysis Events

Events for rhetorical pattern detection during debates.

#### `rhetorical_observation`

Rhetorical pattern detected in agent response.

```json
{
  "type": "rhetorical_observation",
  "data": {
    "agent": "claude-api",
    "round_num": 2,
    "observations": [
      {
        "pattern": "concession",
        "confidence": 0.75,
        "excerpt": "I agree with the point about security, but...",
        "audience_commentary": "Claude shows intellectual humility, acknowledging a valid point"
      },
      {
        "pattern": "synthesis",
        "confidence": 0.82,
        "excerpt": "Combining both perspectives, we can...",
        "audience_commentary": "Claude attempts synthesis - weaving ideas together!"
      }
    ]
  }
}
```

**Pattern Types:**
- `concession` - Acknowledging opponent's valid points
- `rebuttal` - Challenging the prevailing view
- `synthesis` - Combining ideas from multiple sources
- `appeal_to_authority` - Citing authoritative sources
- `appeal_to_evidence` - Backing claims with concrete evidence
- `technical_depth` - Diving into technical details
- `rhetorical_question` - Using questions to make points
- `analogy` - Drawing comparisons to clarify
- `qualification` - Adding nuance and context

**Enable via Protocol:**
```python
protocol = DebateProtocol(enable_rhetorical_observer=True)
```

---

### Breakpoint Events

Human intervention breakpoints allow operators to pause debates at critical moments for guidance or oversight.

#### `breakpoint`

Emitted when a human intervention breakpoint is triggered.

```json
{
  "type": "breakpoint",
  "data": {
    "breakpoint_id": "bp_001",
    "debate_id": "debate_abc123",
    "reason": "High-stakes decision requires human review",
    "trigger_type": "confidence_threshold",
    "context": {
      "round": 3,
      "topic": "Security architecture decision",
      "agents_involved": ["claude-api", "gpt-4"],
      "confidence_scores": {"claude-api": 0.45, "gpt-4": 0.52}
    },
    "suggested_actions": [
      "Review agent proposals",
      "Provide additional context",
      "Override consensus"
    ],
    "timeout_seconds": 300
  },
  "round": 3
}
```

**Trigger Types:**
- `confidence_threshold` - Confidence below minimum threshold
- `topic_sensitive` - Sensitive topic detected
- `agent_disagreement` - Significant agent disagreement
- `manual` - Operator-requested pause
- `safety_check` - Safety concern flagged

#### `breakpoint_resolved`

Emitted when a breakpoint is resolved with human guidance.

```json
{
  "type": "breakpoint_resolved",
  "data": {
    "breakpoint_id": "bp_001",
    "debate_id": "debate_abc123",
    "resolution": "continue_with_guidance",
    "guidance": "Prioritize security over performance in this decision",
    "resolved_by": "operator_123",
    "duration_seconds": 45.2,
    "actions_taken": [
      "Added context about compliance requirements",
      "Requested focus on security implications"
    ]
  }
}
```

**Resolution Types:**
- `continue` - Resume debate without changes
- `continue_with_guidance` - Resume with additional guidance injected
- `override` - Override with human decision
- `abort` - Terminate debate
- `restart_round` - Restart current round

**Enable via Protocol:**
```python
protocol = DebateProtocol(
    enable_breakpoints=True,
    breakpoint_config={
        "confidence_threshold": 0.5,
        "sensitive_topics": ["security", "legal", "financial"],
        "timeout_seconds": 300,
    }
)
```

---

### Mood/Sentiment Events

Real-time debate drama and sentiment analysis.

#### `mood_detected`

Agent emotional state analyzed.

```json
{
  "type": "mood_detected",
  "data": {
    "agent": "claude-api",
    "mood": "confident",
    "intensity": 0.8,
    "indicators": ["assertive language", "strong claims"]
  }
}
```

#### `mood_shift`

Significant mood change detected.

```json
{
  "type": "mood_shift",
  "data": {
    "agent": "gpt-4",
    "from_mood": "defensive",
    "to_mood": "collaborative",
    "trigger": "Critique acknowledged"
  }
}
```

#### `debate_energy`

Overall debate intensity level.

```json
{
  "type": "debate_energy",
  "data": {
    "energy_level": 0.75,
    "trend": "increasing",
    "dominant_mood": "competitive"
  }
}
```

---

### Capability Probe Events

Adversarial testing and vulnerability probing.

#### `probe_start`

Probe session started for agent.

```json
{
  "type": "probe_start",
  "data": {
    "probe_id": "probe_001",
    "target_agent": "claude-api",
    "probe_types": ["jailbreak", "hallucination", "consistency"],
    "probes_per_type": 3,
    "total_probes": 9
  }
}
```

#### `probe_result`

Individual probe result.

```json
{
  "type": "probe_result",
  "data": {
    "probe_id": "probe_001",
    "probe_type": "jailbreak",
    "passed": true,
    "severity": null,
    "description": "Agent correctly refused harmful request",
    "response_time_ms": 234.5
  }
}
```

#### `probe_complete`

All probes complete, report ready.

```json
{
  "type": "probe_complete",
  "data": {
    "report_id": "report_001",
    "target_agent": "claude-api",
    "probes_run": 9,
    "vulnerabilities_found": 1,
    "vulnerability_rate": 0.11,
    "elo_penalty": -5,
    "by_severity": {"low": 1, "medium": 0, "high": 0}
  }
}
```

---

### Deep Audit Events

Intensive multi-round analysis events.

#### `audit_start`

Deep audit session started.

```json
{
  "type": "audit_start",
  "data": {
    "audit_id": "audit_001",
    "task": "Review authentication implementation",
    "agents": ["claude-api", "gpt-4", "gemini"],
    "config": {"rounds": 6, "depth": "thorough"},
    "rounds": 6
  }
}
```

#### `audit_round`

Audit round completed.

```json
{
  "type": "audit_round",
  "data": {
    "audit_id": "audit_001",
    "round": 1,
    "name": "Initial Analysis",
    "cognitive_role": "analyst",
    "messages": [
      {"agent": "claude-api", "content": "..."}
    ],
    "duration_ms": 5000
  },
  "round": 1
}
```

#### `audit_finding`

Individual finding discovered.

```json
{
  "type": "audit_finding",
  "data": {
    "audit_id": "audit_001",
    "category": "security",
    "summary": "Token expiration not enforced",
    "details": "The JWT tokens don't have proper expiration...",
    "agents_agree": ["claude-api", "gpt-4"],
    "agents_disagree": [],
    "confidence": 0.95,
    "severity": 0.8
  }
}
```

#### `audit_cross_exam`

Cross-examination phase completed.

```json
{
  "type": "audit_cross_exam",
  "data": {
    "audit_id": "audit_001",
    "synthesizer": "gemini",
    "questions": [
      "How does this compare to industry standards?",
      "What's the migration path?"
    ],
    "notes": "All agents agree on the core issues"
  }
}
```

#### `audit_verdict`

Final audit verdict ready.

```json
{
  "type": "audit_verdict",
  "data": {
    "audit_id": "audit_001",
    "task": "Review authentication implementation",
    "recommendation": "Implement token refresh and expiration",
    "confidence": 0.92,
    "unanimous_issues": ["Token expiration"],
    "split_opinions": ["Refresh token approach"],
    "risk_areas": ["Session hijacking"],
    "rounds_completed": 6,
    "total_duration_ms": 45000,
    "agents": ["claude-api", "gpt-4", "gemini"],
    "elo_adjustments": {"claude-api": +10, "gpt-4": +5, "gemini": +5}
  }
}
```

---

### Telemetry Events

Cognitive firewall and diagnostic events.

#### `telemetry_thought`

Agent thought process (may be redacted).

```json
{
  "type": "telemetry_thought",
  "data": {
    "agent": "claude-api",
    "thought": "Considering the trade-offs between...",
    "redacted": false
  }
}
```

#### `telemetry_capability`

Agent capability verification result.

```json
{
  "type": "telemetry_capability",
  "data": {
    "agent": "claude-api",
    "capability": "code_generation",
    "verified": true,
    "score": 0.95
  }
}
```

#### `telemetry_redaction`

Content was redacted notification.

```json
{
  "type": "telemetry_redaction",
  "data": {
    "reason": "Potentially sensitive information",
    "original_length": 500,
    "redacted_length": 50
  }
}
```

#### `telemetry_diagnostic`

Internal diagnostic info (dev only).

```json
{
  "type": "telemetry_diagnostic",
  "data": {
    "memory_usage_mb": 256,
    "active_connections": 12,
    "queue_depth": 5
  }
}
```

---

## Message Sequences

### Typical Debate Flow

```
debate_start
├── round_start (round=1)
│   ├── token_start
│   ├── token_delta (multiple)
│   ├── token_end
│   ├── agent_message (proposer)
│   ├── token_start
│   ├── token_delta (multiple)
│   ├── token_end
│   └── critique (critic)
├── round_start (round=2)
│   ├── agent_message (refined proposal)
│   ├── critique
│   └── vote
├── round_start (round=3)
│   ├── agent_message
│   └── consensus
└── debate_end
```

### Nomic Loop Cycle Flow

```
cycle_start
├── phase_start (phase="debate")
│   ├── debate_start
│   └── debate_end
├── phase_end
├── phase_start (phase="design")
├── phase_end
├── phase_start (phase="implement")
│   ├── backup_created
│   ├── task_start
│   ├── task_complete (or task_retry)
│   └── ... (more tasks)
├── phase_end
├── phase_start (phase="verify")
│   ├── verification_start
│   └── verification_result (multiple)
├── phase_end
├── commit (if verification passed)
└── cycle_end
```

### Deep Audit Flow

```
audit_start
├── audit_round (round=1, "Initial Analysis")
│   └── audit_finding (0-n findings)
├── audit_round (round=2, "Deep Dive")
│   └── audit_finding
├── audit_round (round=3, "Devil's Advocate")
│   └── audit_finding
├── audit_round (round=4, "Cross-Examination")
│   └── audit_cross_exam
├── audit_round (round=5, "Synthesis")
├── audit_round (round=6, "Final Verdict")
└── audit_verdict
```

---

## Reconnection Guidance

### Handling Disconnects

1. **Exponential Backoff**: Start with 1 second delay, double on each retry, max 30 seconds
2. **Sequence Tracking**: Store the last `seq` number received
3. **State Recovery**: Request missed events using `since_seq` parameter

### Example Reconnection Logic

```javascript
let lastSeq = 0;
let reconnectDelay = 1000;
const maxDelay = 30000;

function connect() {
  const ws = new WebSocket(`ws://localhost:8080/ws/debates?since_seq=${lastSeq}`);

  ws.onopen = () => {
    console.log('Connected');
    reconnectDelay = 1000; // Reset on successful connect
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    lastSeq = data.seq; // Track sequence
    handleEvent(data);
  };

  ws.onclose = () => {
    console.log(`Disconnected. Reconnecting in ${reconnectDelay}ms`);
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, maxDelay);
  };
}
```

### State Recovery

On reconnection, the server will send:
1. `loop_list` - Current active loops
2. Recent events since `since_seq` (if provided)
3. Current debate state summary

---

## Client Implementation Examples

### Python (websockets)

```python
import asyncio
import websockets
import json

async def listen_to_debates():
    uri = "ws://localhost:8080/ws/debates"
    async with websockets.connect(uri) as ws:
        async for message in ws:
            event = json.loads(message)
            print(f"[{event['type']}] {event.get('agent', 'system')}: {event['data']}")

asyncio.run(listen_to_debates())
```

### JavaScript/TypeScript

```typescript
interface StreamEvent {
  type: string;
  data: Record<string, unknown>;
  timestamp: number;
  round: number;
  agent: string;
  seq: number;
}

const ws = new WebSocket('ws://localhost:8080/ws/debates');

ws.onmessage = (event: MessageEvent) => {
  const data: StreamEvent = JSON.parse(event.data);

  switch (data.type) {
    case 'debate_start':
      console.log('Debate started:', data.data.task);
      break;
    case 'agent_message':
      console.log(`${data.agent}: ${data.data.content}`);
      break;
    case 'consensus':
      console.log('Consensus reached:', data.data.answer);
      break;
    // Handle other event types...
  }
};
```

### Sending Audience Messages

```javascript
// Vote
ws.send(JSON.stringify({
  type: 'user_vote',
  loop_id: 'loop_abc123',
  payload: {
    choice: 'option_a',
    conviction: 0.8
  }
}));

// Suggestion
ws.send(JSON.stringify({
  type: 'user_suggestion',
  loop_id: 'loop_abc123',
  payload: {
    suggestion: 'Consider caching here',
    context: 'performance discussion'
  }
}));
```

---

## See Also

- [API Reference](./API_REFERENCE.md) - REST API documentation
- [Debate Phases](./DEBATE_PHASES.md) - Debate execution phases
- [Architecture](./ARCHITECTURE.md) - System architecture overview
