# Aragora API Reference

This document describes the HTTP and WebSocket APIs for the Aragora debate platform.

## Server Configuration

The unified server runs on port 8080 by default and serves both HTTP API and static files.

```bash
python -m aragora.server --port 8080 --nomic-dir .nomic
```

## Authentication

API requests may include an `Authorization` header with a bearer token:
```
Authorization: Bearer <token>
```

Rate limiting: 60 requests per minute per token (sliding window).

## HTTP API Endpoints

### Health & Status

#### GET /api/health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-04T12:00:00Z"
}
```

#### GET /api/nomic/state
Get current nomic loop state.

**Response:**
```json
{
  "phase": "debate",
  "stage": "executing",
  "cycle": 5,
  "total_tasks": 9,
  "completed_tasks": 3
}
```

#### GET /api/nomic/log
Get recent nomic loop log lines.

**Parameters:**
- `lines` (int, default=100, max=1000): Number of log lines to return

---

### Debates

#### GET /api/debates
List recent debates.

**Parameters:**
- `limit` (int, default=20, max=100): Maximum debates to return

**Response:**
```json
{
  "debates": [
    {
      "id": "debate-123",
      "topic": "Rate limiter implementation",
      "consensus_reached": true,
      "confidence": 0.85,
      "timestamp": "2026-01-04T12:00:00Z"
    }
  ]
}
```

#### GET /api/debates/:slug
Get a specific debate by slug/ID.

#### POST /api/debate
Start an ad-hoc debate. **Rate limited**.

**Request Body:**
```json
{
  "question": "Should we use token bucket or sliding window for rate limiting?",
  "agents": "claude,gemini",
  "rounds": 3,
  "consensus": "majority"
}
```

**Parameters:**
- `question` (string, required): Topic/question to debate
- `agents` (string, default="claude,openai"): Comma-separated agent list (max 10)
- `rounds` (int, default=3): Number of debate rounds
- `consensus` (string, default="majority"): Consensus method

**Response:**
```json
{
  "debate_id": "debate-abc123",
  "status": "started",
  "message": "Debate started with 2 agents"
}
```

---

### History (Supabase)

#### GET /api/history/cycles
Get cycle history.

**Parameters:**
- `loop_id` (string, optional): Filter by loop ID
- `limit` (int, default=50, max=200): Maximum cycles to return

#### GET /api/history/events
Get event history.

**Parameters:**
- `loop_id` (string, optional): Filter by loop ID
- `limit` (int, default=100, max=500): Maximum events to return

#### GET /api/history/debates
Get debate history.

**Parameters:**
- `loop_id` (string, optional): Filter by loop ID
- `limit` (int, default=50, max=200): Maximum debates to return

#### GET /api/history/summary
Get summary statistics.

**Parameters:**
- `loop_id` (string, optional): Filter by loop ID

---

### Leaderboard & ELO

#### GET /api/leaderboard
Get agent rankings by ELO.

**Parameters:**
- `limit` (int, default=20, max=50): Maximum agents to return
- `domain` (string, optional): Filter by domain

**Response:**
```json
{
  "rankings": [
    {
      "agent": "claude",
      "elo": 1523,
      "wins": 45,
      "losses": 12,
      "domain": "general"
    }
  ]
}
```

#### GET /api/matches/recent
Get recent ELO matches.

**Parameters:**
- `limit` (int, default=10, max=50): Maximum matches to return
- `loop_id` (string, optional): Filter by loop ID

#### GET /api/agent/:name/history
Get an agent's match history.

**Parameters:**
- `limit` (int, default=30, max=100): Maximum matches to return

---

### Insights

#### GET /api/insights/recent
Get recent debate insights.

**Parameters:**
- `limit` (int, default=20, max=100): Maximum insights to return

**Response:**
```json
{
  "insights": [
    {
      "type": "pattern",
      "content": "Agents prefer incremental implementations",
      "confidence": 0.78,
      "source_debate": "debate-123"
    }
  ]
}
```

---

### Flip Detection

Position reversal detection API for tracking agent consistency.

#### GET /api/flips/recent
Get recent position flips across all agents.

**Parameters:**
- `limit` (int, default=20, max=100): Maximum flips to return

**Response:**
```json
{
  "flips": [
    {
      "id": "flip-abc123",
      "agent_name": "gemini",
      "original_claim": "X is optimal",
      "new_claim": "Y is optimal",
      "flip_type": "contradiction",
      "similarity_score": 0.82,
      "detected_at": "2026-01-04T12:00:00Z"
    }
  ],
  "count": 15
}
```

#### GET /api/flips/summary
Get aggregate flip statistics.

**Response:**
```json
{
  "total_flips": 42,
  "by_type": {
    "contradiction": 10,
    "refinement": 25,
    "retraction": 5,
    "qualification": 2
  },
  "by_agent": {
    "gemini": 15,
    "claude": 12,
    "codex": 10,
    "grok": 5
  },
  "recent_24h": 8
}
```

#### GET /api/agent/:name/consistency
Get consistency score for an agent.

**Response:**
```json
{
  "agent_name": "claude",
  "total_positions": 150,
  "total_flips": 12,
  "consistency_score": 0.92,
  "contradictions": 2,
  "refinements": 8,
  "retractions": 1,
  "qualifications": 1
}
```

#### GET /api/agent/:name/flips
Get flips for a specific agent.

**Parameters:**
- `limit` (int, default=20, max=100): Maximum flips to return

---

### Consensus Memory

Historical consensus data and similarity search.

#### GET /api/consensus/similar
Find debates similar to a topic.

**Parameters:**
- `topic` (string, required): Topic to search for
- `limit` (int, default=5, max=20): Maximum results to return

**Response:**
```json
{
  "query": "rate limiting",
  "results": [
    {
      "topic": "Rate limiter design",
      "conclusion": "Token bucket preferred",
      "strength": "strong",
      "confidence": 0.85,
      "similarity": 0.92,
      "agents": ["claude", "gemini"],
      "dissent_count": 1,
      "timestamp": "2026-01-03T10:00:00Z"
    }
  ],
  "count": 3
}
```

#### GET /api/consensus/settled
Get high-confidence settled topics.

**Parameters:**
- `min_confidence` (float, default=0.8): Minimum confidence threshold
- `limit` (int, default=20, max=100): Maximum topics to return

**Response:**
```json
{
  "min_confidence": 0.8,
  "topics": [
    {
      "topic": "Consensus algorithm choice",
      "conclusion": "Weighted voting preferred",
      "confidence": 0.95,
      "strength": "strong",
      "timestamp": "2026-01-04T08:00:00Z"
    }
  ],
  "count": 15
}
```

#### GET /api/consensus/stats
Get consensus memory statistics.

**Response:**
```json
{
  "total_consensus": 150,
  "total_dissents": 42,
  "by_strength": {
    "strong": 80,
    "moderate": 50,
    "weak": 20
  },
  "by_domain": {
    "general": 100,
    "architecture": 30,
    "security": 20
  },
  "avg_confidence": 0.78
}
```

#### GET /api/consensus/dissents
Get dissenting views relevant to a topic.

**Parameters:**
- `topic` (string, required): Topic to search for (max 500 chars)
- `domain` (string, optional): Filter by domain

**Response:**
```json
{
  "topic": "rate limiting",
  "domain": null,
  "similar_debates": [
    {
      "topic": "Rate limiter design",
      "conclusion": "Token bucket preferred",
      "confidence": 0.85
    }
  ],
  "dissents_by_type": {
    "alternative": 3,
    "concern": 2,
    "objection": 1
  },
  "unacknowledged_dissents": 2
}
```

#### GET /api/consensus/domain/:domain
Get consensus history for a domain.

**Parameters:**
- `limit` (int, default=50, max=200): Maximum records to return

---

### Pulse (Trending Topics)

Real-time trending topic ingestion for dynamic debate generation.

#### GET /api/pulse/trending
Get trending topics from social media platforms.

**Parameters:**
- `limit` (int, default=10, max=50): Maximum topics to return per platform

**Response:**
```json
{
  "topics": [
    {
      "topic": "AI regulation debate",
      "platform": "twitter",
      "volume": 15000,
      "category": "tech"
    }
  ],
  "count": 5
}
```

---

### Agent Profile (Combined)

#### GET /api/agent/:name/profile
Get a combined profile with ELO, persona, consistency, and calibration data.

**Response:**
```json
{
  "agent": "claude",
  "ranking": {
    "rating": 1523,
    "recent_matches": 10
  },
  "persona": {
    "type": "analytical",
    "primary_stance": "pragmatic",
    "specializations": ["architecture", "security"],
    "debate_count": 45
  },
  "consistency": {
    "score": 0.92,
    "recent_flips": 2
  },
  "calibration": {
    "brier_score": 0.15,
    "prediction_count": 30
  }
}
```

---

### Agent Relationship Network

Analyze agent relationships, alliances, and rivalries.

#### GET /api/agent/:name/network
Get complete influence/relationship network for an agent.

**Response:**
```json
{
  "agent": "claude",
  "influences": [["gemini", 0.75], ["openai", 0.62]],
  "influenced_by": [["codex", 0.58]],
  "rivals": [["grok", 0.81]],
  "allies": [["gemini", 0.72]]
}
```

#### GET /api/agent/:name/rivals
Get top rivals for an agent.

**Parameters:**
- `limit` (int, default=5, max=20): Maximum rivals to return

**Response:**
```json
{
  "agent": "claude",
  "rivals": [["grok", 0.81], ["openai", 0.65]],
  "count": 2
}
```

#### GET /api/agent/:name/allies
Get top allies for an agent.

**Parameters:**
- `limit` (int, default=5, max=20): Maximum allies to return

**Response:**
```json
{
  "agent": "claude",
  "allies": [["gemini", 0.72], ["codex", 0.55]],
  "count": 2
}
```

---

### Critique Patterns

Retrieve high-impact critique patterns for learning.

#### GET /api/critiques/patterns
Get critique patterns ranked by success rate.

**Parameters:**
- `limit` (int, default=10, max=50): Maximum patterns to return
- `min_success` (float, default=0.5): Minimum success rate threshold

**Response:**
```json
{
  "patterns": [
    {
      "issue_type": "security",
      "pattern": "Consider input validation",
      "success_rate": 0.85,
      "usage_count": 12
    }
  ],
  "count": 5,
  "stats": {
    "total_critiques": 150,
    "total_patterns": 42
  }
}
```

---

### Replays

#### GET /api/replays
List available debate replays.

**Response:**
```json
{
  "replays": [
    {
      "id": "nomic-cycle-1",
      "name": "Nomic Cycle 1",
      "event_count": 245,
      "created_at": "2026-01-03T10:00:00Z"
    }
  ]
}
```

#### GET /api/replays/:id
Get a specific replay by ID.

---

### Broadcast (Podcast Generation)

Generate audio podcasts from debate traces.

#### POST /api/debates/:id/broadcast
Generate an MP3 podcast from a debate.

**Rate limited**. Requires the broadcast module (`pip install aragora[broadcast]`).

**Response:**
```json
{
  "success": true,
  "debate_id": "rate-limiter-2026-01-01",
  "audio_path": "/tmp/aragora_debate_rate-limiter.mp3",
  "format": "mp3"
}
```

**Error Response (503):**
```json
{
  "error": "Broadcast module not available"
}
```

---

### Documents

#### GET /api/documents
List uploaded documents.

#### POST /api/documents/upload
Upload a document for processing.

**Headers:**
- `Content-Type: multipart/form-data`
- `X-Filename: document.pdf`

**Supported formats:** PDF, Markdown, Python, JavaScript, TypeScript, Jupyter notebooks

**Max size:** 10MB

---

### Debate Analytics

Real-time debate analytics and pattern detection.

#### GET /api/analytics/disagreements
Get debates with significant disagreements or failed consensus.

**Parameters:**
- `limit` (int, default=20, max=100): Maximum records to return

**Response:**
```json
{
  "disagreements": [
    {
      "debate_id": "debate-123",
      "topic": "Rate limiter design",
      "agents": ["claude", "gemini"],
      "dissent_count": 2,
      "consensus_reached": false,
      "confidence": 0.45,
      "timestamp": "2026-01-04T12:00:00Z"
    }
  ],
  "count": 5
}
```

#### GET /api/analytics/role-rotation
Get agent role assignments across debates.

**Parameters:**
- `limit` (int, default=50, max=200): Maximum rotations to return

**Response:**
```json
{
  "rotations": [
    {
      "debate_id": "debate-123",
      "agent": "claude",
      "role": "proposer",
      "timestamp": "2026-01-04T12:00:00Z"
    }
  ],
  "summary": {
    "claude": {"proposer": 10, "critic": 8, "judge": 5},
    "gemini": {"proposer": 8, "critic": 12, "judge": 3}
  },
  "count": 50
}
```

#### GET /api/analytics/early-stops
Get debates that terminated before completing all planned rounds.

**Parameters:**
- `limit` (int, default=20, max=100): Maximum records to return

**Response:**
```json
{
  "early_stops": [
    {
      "debate_id": "debate-123",
      "topic": "Consensus algorithm",
      "rounds_completed": 2,
      "rounds_planned": 5,
      "reason": "early_consensus",
      "consensus_early": true,
      "timestamp": "2026-01-04T12:00:00Z"
    }
  ],
  "count": 3
}
```

---

### Learning Evolution

#### GET /api/learning/evolution
Get learning evolution patterns.

---

### Modes

List available debate and operational modes.

#### GET /api/modes
Get all available modes.

**Response:**
```json
{
  "modes": [
    {
      "name": "architect",
      "description": "High-level design and planning mode",
      "category": "operational",
      "tool_groups": ["read", "browser", "mcp"]
    },
    {
      "name": "redteam",
      "description": "Adversarial red-teaming for security analysis",
      "category": "debate"
    }
  ],
  "count": 8
}
```

---

### Agent Position Tracking

Track agent positions and consistency via truth-grounding system.

#### GET /api/agent/:name/positions
Get position history for an agent.

**Parameters:**
- `limit` (int, default=50, max=200): Maximum positions to return

**Response:**
```json
{
  "agent": "claude",
  "total_positions": 45,
  "avg_confidence": 0.82,
  "reversal_count": 3,
  "consistency_score": 0.93,
  "positions_by_debate": {
    "rate-limiter-2026-01-01": 5,
    "consensus-algo-2026-01-02": 3
  }
}
```

---

### System Statistics

System-wide metrics and health monitoring.

#### GET /api/ranking/stats
Get ELO ranking system statistics.

**Response:**
```json
{
  "total_agents": 8,
  "total_matches": 245,
  "avg_rating": 1523,
  "rating_spread": 312,
  "domains": ["general", "architecture", "security"],
  "active_last_24h": 5
}
```

#### GET /api/memory/stats
Get memory tier statistics from continuum memory system.

**Response:**
```json
{
  "tiers": {
    "fast": {"count": 50, "avg_importance": 0.85},
    "slow": {"count": 200, "avg_importance": 0.65},
    "glacial": {"count": 1000, "avg_importance": 0.35}
  },
  "total_entries": 1250,
  "last_consolidation": "2026-01-04T12:00:00Z"
}
```

#### GET /api/critiques/patterns
Get successful critique patterns for learning.

**Parameters:**
- `limit` (int, default=10, max=50): Maximum patterns to return
- `min_success` (float, default=0.5): Minimum success rate filter

**Response:**
```json
{
  "patterns": [
    {
      "issue_type": "edge_case",
      "pattern": "Consider boundary conditions for...",
      "success_rate": 0.85,
      "usage_count": 23
    }
  ],
  "count": 5,
  "stats": {"total_patterns": 150, "avg_success_rate": 0.72}
}
```

#### GET /api/critiques/archive
Get archive statistics for resolved patterns.

**Response:**
```json
{
  "archived": 42,
  "by_type": {
    "security": 15,
    "performance": 12,
    "edge_case": 15
  }
}
```

---

### Agent Reputation

Track agent reliability and voting weights.

#### GET /api/reputation/all
Get all agent reputations ranked by score.

**Response:**
```json
{
  "reputations": [
    {
      "agent": "claude",
      "score": 0.85,
      "vote_weight": 1.35,
      "proposal_acceptance_rate": 0.78,
      "critique_value": 0.82,
      "debates_participated": 45
    }
  ],
  "count": 4
}
```

#### GET /api/agent/:name/reputation
Get reputation for a specific agent.

**Response:**
```json
{
  "agent": "claude",
  "score": 0.85,
  "vote_weight": 1.35,
  "proposal_acceptance_rate": 0.78,
  "critique_value": 0.82,
  "debates_participated": 45
}
```

---

### Agent Comparison

Compare agents head-to-head.

#### GET /api/agent/compare
Get head-to-head comparison between two agents.

**Parameters:**
- `agent_a` (string, required): First agent name
- `agent_b` (string, required): Second agent name

**Response:**
```json
{
  "agent_a": "claude",
  "agent_b": "gemini",
  "matches": 15,
  "agent_a_wins": 9,
  "agent_b_wins": 6,
  "win_rate_a": 0.6,
  "domains": {
    "architecture": {"a_wins": 5, "b_wins": 2},
    "security": {"a_wins": 4, "b_wins": 4}
  }
}
```

---

### Agent Expertise & Grounded Personas

Evidence-based agent identity and expertise tracking.

#### GET /api/agent/:name/domains
Get agent's best expertise domains by calibration score.

**Parameters:**
- `limit` (int, default=5, max=20): Maximum domains to return

**Response:**
```json
{
  "agent": "claude",
  "domains": [
    {"domain": "security", "calibration_score": 0.89},
    {"domain": "api_design", "calibration_score": 0.85}
  ],
  "count": 2
}
```

#### GET /api/agent/:name/grounded-persona
Get truth-grounded persona synthesized from performance data.

**Response:**
```json
{
  "agent": "claude",
  "elo": 1523,
  "domain_elos": {"security": 1580, "architecture": 1490},
  "games_played": 45,
  "win_rate": 0.62,
  "calibration_score": 0.78,
  "position_accuracy": 0.72,
  "positions_taken": 128,
  "reversals": 8
}
```

#### GET /api/agent/:name/identity-prompt
Get evidence-grounded identity prompt for agent initialization.

**Parameters:**
- `sections` (string, optional): Comma-separated sections to include (performance, calibration, relationships, positions)

**Response:**
```json
{
  "agent": "claude",
  "identity_prompt": "## Your Identity: claude\nYour approach: analytical, thorough...",
  "sections": ["performance", "calibration"]
}
```

---

### Contrarian Views & Risk Warnings

Historical dissenting views and edge case concerns from past debates.

#### GET /api/consensus/contrarian-views
Get historical contrarian views on a topic.

**Parameters:**
- `topic` (string, required): Topic to search for contrarian views
- `domain` (string, optional): Filter by domain
- `limit` (int, default=5, max=20): Maximum views to return

**Response:**
```json
{
  "topic": "rate limiting implementation",
  "domain": null,
  "contrarian_views": [
    {
      "agent": "gemini",
      "position": "Token bucket has edge cases",
      "reasoning": "Under burst traffic conditions...",
      "confidence": 0.75,
      "timestamp": "2026-01-03T10:00:00Z"
    }
  ],
  "count": 1
}
```

#### GET /api/consensus/risk-warnings
Get risk warnings and edge case concerns from past debates.

**Parameters:**
- `topic` (string, required): Topic to search for risk warnings
- `domain` (string, optional): Filter by domain
- `limit` (int, default=5, max=20): Maximum warnings to return

**Response:**
```json
{
  "topic": "database migration",
  "domain": "infrastructure",
  "risk_warnings": [
    {
      "agent": "claude",
      "warning": "Consider rollback strategy for schema changes",
      "severity": "high",
      "timestamp": "2026-01-02T15:00:00Z"
    }
  ],
  "count": 1
}
```

---

## WebSocket API

Connect to the WebSocket server for real-time streaming:

```javascript
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.type, data.payload);
};
```

### Event Types

| Event Type | Description |
|------------|-------------|
| `phase_start` | A nomic phase has started |
| `phase_end` | A nomic phase has completed |
| `agent_message` | An agent has sent a message |
| `debate_round` | A debate round has completed |
| `consensus` | Consensus has been reached |
| `elo_update` | ELO ratings have been updated |
| `flip_detected` | A position flip was detected |

### Event Format

```json
{
  "type": "agent_message",
  "timestamp": "2026-01-04T12:00:00Z",
  "loop_id": "loop-abc123",
  "payload": {
    "agent": "claude",
    "role": "critic",
    "content": "I disagree with this approach because..."
  }
}
```

---

## Error Responses

All endpoints return errors in this format:

```json
{
  "error": "Description of the error"
}
```

Common HTTP status codes:
- `400` - Bad request (invalid parameters)
- `403` - Forbidden (access denied)
- `404` - Not found
- `500` - Internal server error

---

## CORS Policy

The API allows cross-origin requests from:
- `http://localhost:3000`
- `http://localhost:8080`
- `https://aragora.ai`
- `https://www.aragora.ai`

Other origins are blocked by the browser.

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| General API | 60 req/min per token |
| Document upload | 10 req/min |
| WebSocket | Unlimited messages |

---

## Security Notes

1. **Path traversal protection**: All file paths are validated to prevent directory traversal attacks
2. **Input validation**: All integer/float parameters have bounds checking
3. **Error sanitization**: API keys and tokens are redacted from error messages
4. **Origin validation**: CORS uses allowlist instead of wildcard
5. **SQL injection prevention**: LIKE patterns are escaped to prevent wildcard injection
6. **Rate limiting**: Expensive endpoints (debate creation, uploads) are rate limited
7. **Query bounds**: Maximum 10 agents per debate, 10 parts per multipart upload
8. **Database timeouts**: SQLite connections have 30-second timeout to prevent deadlocks
9. **Content-Length validation**: Headers validated to prevent integer parsing attacks
