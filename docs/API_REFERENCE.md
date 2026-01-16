# Aragora API Reference

This document describes the HTTP and WebSocket APIs for the Aragora AI red team / decision stress-test platform.

## Endpoint Usage Status

**Last audited:** 2026-01-15

| Category | Count | Notes |
|----------|-------|-------|
| **Total Endpoints** | 297 | All API routes from 78 handler modules |
| **Actively Used** | ~40 | Called from frontend components |
| **Ready to Wire** | ~10 | High-value, not yet connected to frontend |
| **Advanced/Analytics** | ~247 | Specialized features, plugins, analytics |

## OpenAPI Specification

The generated OpenAPI spec lives in `docs/api/openapi.json` and `docs/api/openapi.yaml`.
Regenerate them with:

```bash
python scripts/export_openapi.py --output-dir docs/api
```

The contract tests use `aragora/server/openapi.yaml` as the canonical spec. If you
add or change endpoints, update both to keep tests and docs in sync.

### New Endpoints (2026-01-09)

| Endpoint | Description | Status |
|----------|-------------|--------|
| `POST /api/debates/graph` | Graph-structured debates with branching | NEW |
| `GET /api/debates/graph/:id` | Get graph debate by ID | NEW |
| `GET /api/debates/graph/:id/branches` | Get branches for graph debate | NEW |
| `GET /api/debates/graph/:id/nodes` | Get nodes for graph debate | NEW |
| `POST /api/debates/matrix` | Parallel scenario debates | NEW |
| `GET /api/debates/matrix/:id` | Get matrix debate by ID | NEW |
| `GET /api/debates/matrix/:id/scenarios` | Get scenario results | NEW |
| `GET /api/debates/matrix/:id/conclusions` | Get matrix conclusions | NEW |
| `GET /api/breakpoints/pending` | List pending human-in-the-loop breakpoints | NEW |
| `GET /api/breakpoints/:id/status` | Get breakpoint status | NEW |
| `POST /api/breakpoints/:id/resolve` | Resolve breakpoint with human input | NEW |
| `GET /api/introspection/all` | Get all agent introspection data | NEW |
| `GET /api/introspection/leaderboard` | Agents ranked by reputation | NEW |
| `GET /api/introspection/agents` | List available agents | NEW |
| `GET /api/introspection/agents/:name` | Get agent introspection | NEW |
| `GET /api/gallery` | List public debates | NEW |
| `GET /api/gallery/:id` | Get public debate details | NEW |
| `GET /api/gallery/:id/embed` | Get embeddable debate summary | NEW |
| `GET /api/billing/plans` | List subscription plans | NEW |
| `GET /api/billing/usage` | Get current usage | NEW |
| `GET /api/billing/subscription` | Get subscription status | NEW |
| `POST /api/billing/checkout` | Create Stripe checkout | NEW |
| `POST /api/billing/portal` | Create billing portal session | NEW |
| `POST /api/billing/cancel` | Cancel subscription | NEW |
| `POST /api/billing/resume` | Resume subscription | NEW |
| `POST /api/webhooks/stripe` | Handle Stripe webhooks | NEW |
| `GET /api/memory/analytics` | Get comprehensive memory tier analytics | NEW |
| `GET /api/memory/analytics/tier/:tier` | Get stats for specific tier | NEW |
| `POST /api/memory/analytics/snapshot` | Take manual analytics snapshot | NEW |
| `GET /api/evolution/ab-tests` | List all A/B tests | NEW |
| `GET /api/evolution/ab-tests/:id` | Get specific A/B test | NEW |
| `GET /api/evolution/ab-tests/:agent/active` | Get active test for agent | NEW |
| `POST /api/evolution/ab-tests` | Create new A/B test | NEW |
| `POST /api/evolution/ab-tests/:id/record` | Record debate result | NEW |
| `POST /api/evolution/ab-tests/:id/conclude` | Conclude test | NEW |
| `DELETE /api/evolution/ab-tests/:id` | Cancel test | NEW |

### Recently Connected Endpoints

The following endpoints were identified as unused but are now connected:

| Endpoint | Component | Status |
|----------|-----------|--------|
| `GET /api/debates` | DebateListPanel | ✅ Connected |
| `GET /api/agent/{agent}/profile` | AgentProfileWrapper | ✅ Connected |
| `GET /api/agent/compare` | AgentComparePanel | ✅ Connected |
| `GET /api/agent/{agent}/head-to-head/{opponent}` | AgentProfileWrapper | ✅ Connected |
| `GET /api/agent/{agent}/network` | AgentProfileWrapper | ✅ Connected (includes rivals/allies) |
| `GET /api/history/debates` | HistoryPanel | ✅ Connected (local API fallback) |
| `GET /api/history/summary` | HistoryPanel | ✅ Connected (local API fallback) |
| `GET /api/history/cycles` | HistoryPanel | ✅ Connected (local API fallback) |
| `GET /api/history/events` | HistoryPanel | ✅ Connected (local API fallback) |
| `GET /api/pulse/trending` | TrendingTopicsPanel | ✅ Connected |
| `GET /api/analytics/disagreements` | AnalyticsPanel | ✅ Connected |

### Remaining High-Value Endpoints (Ready to Wire)

| Endpoint | Feature | Priority |
|----------|---------|----------|
| `GET /api/agent/{agent}/history` | Agent debate history | MEDIUM |
| `GET /api/agent/{agent}/rivals` | Direct rivals endpoint | LOW |
| `GET /api/agent/{agent}/allies` | Direct allies endpoint | LOW |
| `GET /api/debates/{id}` | Individual debate detail view | LOW |

## Server Configuration

The unified server exposes HTTP on port 8080 and WebSocket on port 8765 by default.

```bash
aragora serve --api-port 8080 --ws-port 8765
# Or directly:
python -m aragora.server.unified_server --api-port 8080 --ws-port 8765
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
  "agents": "anthropic-api,openai-api",
  "rounds": 3,
  "consensus": "majority"
}
```

**Parameters:**
- `question` (string, required): Topic/question to debate
- `agents` (string, default="grok,anthropic-api,openai-api,deepseek-r1"): Comma-separated agent list (max 10)
- `rounds` (int, default=3): Number of debate rounds
- `consensus` (string, default="majority"): Consensus method

**Available Agent Types:**

API (direct):

| Type | Default Model | Notes |
|------|---------------|-------|
| `anthropic-api` | claude-opus-4-5-20251101 | Anthropic API, streaming |
| `openai-api` | gpt-5.2 | OpenAI API, streaming |
| `gemini` | gemini-3-pro-preview | Google API, streaming |
| `grok` | grok-3 | xAI API, streaming |
| `mistral-api` | mistral-large-2512 | Mistral API |
| `codestral` | codestral-latest | Mistral code model |
| `ollama` | llama3.2 | Local Ollama |
| `lm-studio` | local-model | Local LM Studio |
| `kimi` | moonshot-v1-8k | Moonshot API |

OpenRouter:

| Type | Default Model | Notes |
|------|---------------|-------|
| `openrouter` | deepseek/deepseek-chat-v3-0324 | Model via `model` parameter |
| `deepseek` | deepseek/deepseek-chat-v3-0324 | DeepSeek V3 (chat) |
| `deepseek-r1` | deepseek/deepseek-r1 | DeepSeek reasoning |
| `llama` | meta-llama/llama-3.3-70b-instruct | Llama 3.3 70B |
| `mistral` | mistralai/mistral-large-2411 | Mistral Large |
| `qwen` | qwen/qwen-2.5-coder-32b-instruct | Qwen 2.5 Coder |
| `qwen-max` | qwen/qwen-max | Qwen Max |
| `yi` | 01-ai/yi-large | Yi Large |

CLI:

| Type | Default Model | Notes |
|------|---------------|-------|
| `claude` | claude-sonnet-4 | Claude CLI |
| `codex` | gpt-5.2-codex | Codex CLI |
| `openai` | gpt-4o | OpenAI CLI |
| `gemini-cli` | gemini-3-pro-preview | Gemini CLI |
| `grok-cli` | grok-4 | Grok CLI |
| `qwen-cli` | qwen3-coder | Qwen CLI |
| `deepseek-cli` | deepseek-v3 | DeepSeek CLI |
| `kilocode` | gemini-explorer | Codebase explorer |

**Response:**
```json
{
  "debate_id": "debate-abc123",
  "status": "started",
  "message": "Debate started with 2 agents"
}
```

#### GET /api/debates/:id/export/:format
Export a debate in various formats.

**Path Parameters:**
- `id` (string, required): Debate slug/ID
- `format` (string, required): Export format - `json`, `csv`, `dot`, or `html`

**Query Parameters:**
- `table` (string, optional): For CSV format only - `summary` (default), `messages`, `critiques`, `votes`, or `verifications`

**Response (JSON format):**
```json
{
  "artifact_id": "abc123",
  "debate_id": "debate-slug",
  "task": "Rate limiter implementation",
  "consensus_proof": {
    "reached": true,
    "confidence": 0.85,
    "vote_breakdown": {"anthropic-api": true, "gemini": true}
  },
  "agents": ["anthropic-api", "gemini"],
  "rounds": 3,
  "content_hash": "sha256:abcd1234"
}
```

**Response (CSV format):** Returns text/csv with debate data table
**Response (DOT format):** Returns text/vnd.graphviz for visualization with GraphViz
**Response (HTML format):** Returns self-contained HTML viewer with interactive graph

#### POST /api/debates/graph
Run a graph-structured debate with automatic branching.

**Request Body:**
```json
{
  "task": "Design a distributed caching system",
  "agents": ["anthropic-api", "openai-api"],
  "max_rounds": 5,
  "branch_policy": {
    "min_disagreement": 0.7,
    "max_branches": 3,
    "auto_merge": true,
    "merge_strategy": "synthesis"
  }
}
```

**Response:**
```json
{
  "debate_id": "uuid",
  "task": "Design a distributed caching system",
  "graph": {
    "nodes": [...],
    "edges": [...],
    "root_id": "node-123"
  },
  "branches": [...],
  "merge_results": [...],
  "node_count": 15,
  "branch_count": 2
}
```

**Branch Policy Options:**
- `min_disagreement` (float): Threshold to trigger branching (default: 0.7)
- `max_branches` (int): Maximum concurrent branches (default: 3)
- `auto_merge` (bool): Automatically merge converging branches (default: true)
- `merge_strategy` (string): "best_path", "synthesis", "vote", "weighted", "preserve_all"

#### GET /api/debates/graph/:id
Get a graph debate by ID.

#### GET /api/debates/graph/:id/branches
Get all branches for a graph debate.

#### GET /api/debates/graph/:id/nodes
Get all nodes in a graph debate.

#### POST /api/debates/matrix
Run parallel scenario debates with comparative analysis.

**Request Body:**
```json
{
  "task": "Design a rate limiter",
  "agents": ["anthropic-api", "openai-api"],
  "scenarios": [
    {
      "name": "High throughput",
      "parameters": {"rps": 10000},
      "constraints": ["Must handle burst traffic"]
    },
    {
      "name": "Low latency",
      "parameters": {"latency_ms": 10},
      "constraints": ["P99 under 10ms"]
    },
    {
      "name": "Baseline",
      "is_baseline": true
    }
  ],
  "max_rounds": 3
}
```

**Response:**
```json
{
  "matrix_id": "uuid",
  "task": "Design a rate limiter",
  "scenario_count": 3,
  "results": [
    {
      "scenario_name": "High throughput",
      "parameters": {"rps": 10000},
      "winner": "anthropic-api",
      "final_answer": "...",
      "confidence": 0.85,
      "consensus_reached": true
    }
  ],
  "universal_conclusions": ["All scenarios reached consensus"],
  "conditional_conclusions": [
    {
      "condition": "When High throughput",
      "conclusion": "Use token bucket algorithm",
      "confidence": 0.85
    }
  ],
  "comparison_matrix": {
    "scenarios": ["High throughput", "Low latency", "Baseline"],
    "consensus_rate": 1.0,
    "avg_confidence": 0.82,
    "avg_rounds": 2.5
  }
}
```

#### GET /api/debates/matrix/:id
Get matrix debate results by ID.

#### GET /api/debates/matrix/:id/scenarios
Get all scenario results for a matrix debate.

#### GET /api/debates/matrix/:id/conclusions
Get conclusions (universal and conditional) for a matrix debate.

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
      "agent": "anthropic-api",
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
    "anthropic-api": 12,
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
  "agent_name": "anthropic-api",
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
      "agents": ["anthropic-api", "gemini"],
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

#### POST /api/pulse/debate-topic
Start a debate on a trending topic.

**Request Body:**
```json
{
  "topic": "Should AI be regulated?",
  "rounds": 3,
  "consensus_threshold": 0.7
}
```

**Response:**
```json
{
  "debate_id": "pulse-1234567890-abc123",
  "topic": "Should AI be regulated?",
  "status": "started"
}
```

### Pulse Scheduler

Automated debate creation from trending topics.

#### GET /api/pulse/scheduler/status
Get current scheduler state and metrics.

**Response:**
```json
{
  "state": "running",
  "run_id": "run-1234567890-abc123",
  "config": {
    "poll_interval_seconds": 300,
    "max_debates_per_hour": 6,
    "min_volume_threshold": 100,
    "min_controversy_score": 0.3,
    "dedup_window_hours": 24
  },
  "metrics": {
    "polls_completed": 15,
    "topics_evaluated": 120,
    "debates_created": 8,
    "duplicates_skipped": 5,
    "uptime_seconds": 3600
  }
}
```

#### POST /api/pulse/scheduler/start
Start the pulse debate scheduler.

**Response:**
```json
{
  "status": "started",
  "run_id": "run-1234567890-abc123"
}
```

#### POST /api/pulse/scheduler/stop
Stop the pulse debate scheduler.

**Response:**
```json
{
  "status": "stopped"
}
```

#### POST /api/pulse/scheduler/pause
Pause the scheduler (maintains state but stops polling).

**Response:**
```json
{
  "status": "paused"
}
```

#### POST /api/pulse/scheduler/resume
Resume a paused scheduler.

**Response:**
```json
{
  "status": "running"
}
```

#### PATCH /api/pulse/scheduler/config
Update scheduler configuration.

**Request Body:**
```json
{
  "max_debates_per_hour": 10,
  "min_controversy_score": 0.4
}
```

**Response:**
```json
{
  "status": "updated",
  "config": { ... }
}
```

#### GET /api/pulse/scheduler/history
Get history of scheduled debates.

**Parameters:**
- `limit` (int, default=50): Maximum records to return
- `offset` (int, default=0): Pagination offset
- `platform` (string, optional): Filter by platform
- `category` (string, optional): Filter by category

**Response:**
```json
{
  "history": [
    {
      "id": "sched-123",
      "topic_text": "AI ethics debate",
      "platform": "hackernews",
      "category": "tech",
      "debate_id": "pulse-456",
      "created_at": 1704067200,
      "consensus_reached": true,
      "confidence": 0.85
    }
  ],
  "total": 100
}
```

---

### Slack Integration

Slack bot integration for debate notifications and commands.

#### GET /api/integrations/slack/status
Get Slack integration status.

**Response:**
```json
{
  "enabled": true,
  "signing_secret_configured": true,
  "bot_token_configured": false,
  "webhook_configured": true
}
```

#### POST /api/integrations/slack/commands
Handle Slack slash commands (called by Slack).

**Supported Commands:**
- `/aragora help` - Show available commands
- `/aragora status` - Get system status
- `/aragora debate "topic"` - Start a debate
- `/aragora agents` - List top agents by ELO

#### POST /api/integrations/slack/interactive
Handle Slack interactive components (buttons, menus).

#### POST /api/integrations/slack/events
Handle Slack Events API callbacks.

---

### Agent Profile (Combined)

#### GET /api/agent/:name/profile
Get a combined profile with ELO, persona, consistency, and calibration data.

**Response:**
```json
{
  "agent": "anthropic-api",
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
  "agent": "anthropic-api",
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
  "agent": "anthropic-api",
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
  "agent": "anthropic-api",
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

#### GET /api/documents/formats
Get supported document formats and metadata.

**Response:**
```json
{
  "formats": {
    ".pdf": {"name": "PDF", "mime": "application/pdf"},
    ".md": {"name": "Markdown", "mime": "text/markdown"},
    ".py": {"name": "Python", "mime": "text/x-python"},
    ".js": {"name": "JavaScript", "mime": "text/javascript"},
    ".ts": {"name": "TypeScript", "mime": "text/typescript"},
    ".ipynb": {"name": "Jupyter Notebook", "mime": "application/x-ipynb+json"}
  }
}
```

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
      "agents": ["anthropic-api", "gemini"],
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
      "agent": "anthropic-api",
      "role": "proposer",
      "timestamp": "2026-01-04T12:00:00Z"
    }
  ],
  "summary": {
    "anthropic-api": {"proposer": 10, "critic": 8, "judge": 5},
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
  "agent": "anthropic-api",
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

#### GET /api/memory/analytics
Get comprehensive memory tier analytics.

**Parameters:**
- `days` (int, default=30, min=1, max=365): Number of days to analyze

**Response:**
```json
{
  "total_memories": 1000,
  "tier_stats": {
    "fast": {"count": 100, "avg_importance": 0.8},
    "medium": {"count": 300, "avg_importance": 0.6},
    "slow": {"count": 400, "avg_importance": 0.4},
    "glacial": {"count": 200, "avg_importance": 0.2}
  },
  "learning_velocity": 0.75,
  "promotion_effectiveness": 0.82
}
```

#### GET /api/memory/analytics/tier/:tier
Get stats for a specific memory tier.

**Path Parameters:**
- `tier` (string, required): Tier name - `fast`, `medium`, `slow`, or `glacial`

**Parameters:**
- `days` (int, default=30, min=1, max=365): Number of days to analyze

**Response:**
```json
{
  "tier": "fast",
  "count": 100,
  "avg_importance": 0.8,
  "hit_rate": 0.95,
  "promotion_rate": 0.3
}
```

#### POST /api/memory/analytics/snapshot
Take a manual analytics snapshot for all memory tiers.

**Response:**
```json
{
  "status": "success",
  "message": "Snapshot recorded for all tiers"
}
```

---

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
      "agent": "anthropic-api",
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
  "agent": "anthropic-api",
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
  "agent_a": "anthropic-api",
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
  "agent": "anthropic-api",
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
  "agent": "anthropic-api",
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
  "agent": "anthropic-api",
  "identity_prompt": "## Your Identity: anthropic-api\nYour approach: analytical, thorough...",
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
      "agent": "anthropic-api",
      "warning": "Consider rollback strategy for schema changes",
      "severity": "high",
      "timestamp": "2026-01-02T15:00:00Z"
    }
  ],
  "count": 1
}
```

---

### Head-to-Head & Opponent Analysis

Detailed comparison and strategic briefings between agents.

#### GET /api/agent/:agent/head-to-head/:opponent
Get detailed head-to-head statistics between two agents.

**Response:**
```json
{
  "agent": "anthropic-api",
  "opponent": "gemini",
  "matches": 12,
  "agent_wins": 5,
  "opponent_wins": 4,
  "draws": 3,
  "win_rate": 0.42,
  "recent_form": "WDLWW"
}
```

#### GET /api/agent/:agent/opponent-briefing/:opponent
Get strategic briefing about an opponent for an agent.

**Response:**
```json
{
  "agent": "anthropic-api",
  "opponent": "gemini",
  "briefing": {
    "relationship": "rival",
    "strength": 0.7,
    "head_to_head": {"wins": 5, "losses": 4, "draws": 3},
    "opponent_strengths": ["visual reasoning", "synthesis"],
    "opponent_weaknesses": ["consistency", "edge cases"],
    "recommended_strategy": "Focus on logical rigor and consistency"
  }
}
```

---

### Calibration Analysis

Detailed calibration curves and prediction accuracy.

#### GET /api/agent/:name/calibration-curve
Get calibration curve showing expected vs actual accuracy per confidence bucket.

**Parameters:**
- `buckets` (int, default=10, max=20): Number of confidence buckets
- `domain` (string, optional): Filter by domain

**Response:**
```json
{
  "agent": "anthropic-api",
  "domain": null,
  "buckets": [
    {
      "range_start": 0.0,
      "range_end": 0.1,
      "total_predictions": 15,
      "correct_predictions": 2,
      "accuracy": 0.13,
      "expected_accuracy": 0.05,
      "brier_score": 0.08
    },
    {
      "range_start": 0.9,
      "range_end": 1.0,
      "total_predictions": 42,
      "correct_predictions": 38,
      "accuracy": 0.90,
      "expected_accuracy": 0.95,
      "brier_score": 0.02
    }
  ],
  "count": 10
}
```

---

### Meta-Critique Analysis

Analyze debate quality and identify process issues.

#### GET /api/debate/:id/meta-critique
Get meta-level analysis of a debate including repetition, circular arguments, and ignored critiques.

**Response:**
```json
{
  "debate_id": "debate-123",
  "overall_quality": 0.72,
  "productive_rounds": [1, 2, 4],
  "unproductive_rounds": [3],
  "observations": [
    {
      "type": "repetition",
      "severity": "medium",
      "agent": "gemini",
      "round": 3,
      "description": "Agent repeated similar points from round 1"
    }
  ],
  "recommendations": [
    "Encourage agents to address critiques directly",
    "Consider reducing round count for simple topics"
  ]
}
```

---

### Agent Personas

Agent persona definitions and customizations.

#### GET /api/personas
Get all agent personas.

**Response:**
```json
{
  "personas": [
    {
      "agent_name": "anthropic-api",
      "description": "Analytical reasoner focused on logical consistency",
      "traits": ["analytical", "precise", "evidence-focused"],
      "expertise": ["security", "architecture"],
      "created_at": "2026-01-01T00:00:00Z"
    },
    {
      "agent_name": "grok",
      "description": "Creative problem-solver with lateral thinking",
      "traits": ["creative", "unconventional", "synthesis-focused"],
      "expertise": ["innovation", "edge-cases"],
      "created_at": "2026-01-01T00:00:00Z"
    }
  ],
  "count": 2
}
```

---

### Persona Laboratory

Experimental framework for evolving agent personas and detecting emergent traits.

#### GET /api/laboratory/emergent-traits
Get emergent traits detected from agent performance patterns.

**Parameters:**
- `min_confidence` (float, default=0.5): Minimum confidence threshold
- `limit` (int, default=10, max=50): Maximum traits to return

**Response:**
```json
{
  "emergent_traits": [
    {
      "agent": "anthropic-api",
      "trait": "adversarial_robustness",
      "domain": "security",
      "confidence": 0.85,
      "evidence": "Consistently identifies edge cases in security debates",
      "detected_at": "2026-01-04T10:00:00Z"
    }
  ],
  "count": 1,
  "min_confidence": 0.5
}
```

#### POST /api/laboratory/cross-pollinations/suggest
Suggest beneficial trait transfers for a target agent.

**Request Body:**
```json
{
  "target_agent": "gemini"
}
```

**Response:**
```json
{
  "target_agent": "gemini",
  "suggestions": [
    {
      "source_agent": "anthropic-api",
      "trait_or_domain": "logical_rigor",
      "reason": "Target agent underperforms in formal reasoning domains"
    }
  ],
  "count": 1
}
```

---

### Belief Network Analysis

Bayesian belief network for probabilistic debate reasoning.

#### GET /api/belief-network/:debate_id/cruxes
Identify key claims that would most impact the debate outcome.

**Parameters:**
- `top_k` (int, default=3, max=10): Number of cruxes to return

**Response:**
```json
{
  "debate_id": "debate-123",
  "cruxes": [
    {
      "claim_id": "claim-456",
      "statement": "The proposed architecture scales linearly",
      "score": 0.87,
      "centrality": 0.9,
      "uncertainty": 0.75
    }
  ],
  "count": 1
}
```

---

### Provenance & Evidence Chain

Verify evidence provenance and claim support.

#### GET /api/provenance/:debate_id/claims/:claim_id/support
Get verification status of all evidence supporting a claim.

**Response:**
```json
{
  "debate_id": "debate-123",
  "claim_id": "claim-456",
  "support": {
    "verified": true,
    "evidence_count": 3,
    "supporting": [
      {
        "evidence_id": "ev-001",
        "type": "citation",
        "integrity_verified": true,
        "relevance": 0.92
      }
    ],
    "contradicting": []
  }
}
```

---

### Agent Routing & Selection

Optimal agent selection for tasks based on ELO, expertise, and team dynamics.

#### POST /api/routing/recommendations
Get agent recommendations for a task.

**Request Body:**
```json
{
  "task_id": "design-review",
  "primary_domain": "architecture",
  "secondary_domains": ["security", "performance"],
  "required_traits": ["analytical"],
  "limit": 5
}
```

**Response:**
```json
{
  "task_id": "design-review",
  "primary_domain": "architecture",
  "recommendations": [
    {
      "name": "anthropic-api",
      "type": "anthropic-api",
      "match_score": 0.92,
      "domain_expertise": 0.85
    }
  ],
  "count": 1
}
```

---

### Tournament System

Tournament management and standings.

#### GET /api/tournaments/:tournament_id/standings
Get current tournament standings.

**Response:**
```json
{
  "tournament_id": "round-robin-2026",
  "standings": [
    {
      "agent": "anthropic-api",
      "wins": 8,
      "losses": 2,
      "draws": 1,
      "points": 25,
      "total_score": 142.5,
      "win_rate": 0.73
    }
  ],
  "count": 4
}
```

---

### Team Analytics

Analyze team performance and find optimal combinations.

#### GET /api/routing/best-teams
Get best-performing team combinations from history.

**Parameters:**
- `min_debates` (int, default=3, max=20): Minimum debates for a team to qualify
- `limit` (int, default=10, max=50): Maximum combinations to return

**Response:**
```json
{
  "min_debates": 3,
  "combinations": [
    {
      "agents": ["anthropic-api", "gemini"],
      "success_rate": 0.85,
      "total_debates": 12,
      "wins": 10
    }
  ],
  "count": 5
}
```

---

### Prompt Evolution

Track agent prompt evolution and learning.

#### GET /api/evolution/:agent/history
Get prompt evolution history for an agent.

**Parameters:**
- `limit` (int, default=10, max=50): Maximum history entries to return

**Response:**
```json
{
  "agent": "anthropic-api",
  "history": [
    {
      "from_version": 1,
      "to_version": 2,
      "strategy": "pattern_mining",
      "patterns_applied": ["logical_rigor", "edge_case_handling"],
      "created_at": "2026-01-04T08:00:00Z"
    }
  ],
  "count": 3
}
```

---

### Evolution A/B Testing

Run controlled experiments comparing prompt versions to determine which performs better.

#### GET /api/evolution/ab-tests
List all A/B tests with optional filters.

**Parameters:**
- `agent` (string, optional): Filter by agent name
- `status` (string, optional): Filter by status (active, concluded, cancelled)
- `limit` (int, default=50, max=200): Maximum tests to return

**Response:**
```json
{
  "tests": [
    {
      "id": "test-123",
      "agent": "anthropic-api",
      "baseline_prompt_version": 1,
      "evolved_prompt_version": 2,
      "status": "active",
      "baseline_wins": 5,
      "evolved_wins": 7,
      "started_at": "2026-01-10T12:00:00Z"
    }
  ],
  "count": 1
}
```

#### GET /api/evolution/ab-tests/:id
Get a specific A/B test by ID.

**Response:**
```json
{
  "id": "test-123",
  "agent": "anthropic-api",
  "baseline_prompt_version": 1,
  "evolved_prompt_version": 2,
  "status": "active",
  "baseline_wins": 5,
  "evolved_wins": 7,
  "baseline_debates": 10,
  "evolved_debates": 10,
  "evolved_win_rate": 0.58,
  "is_significant": false,
  "started_at": "2026-01-10T12:00:00Z",
  "concluded_at": null
}
```

#### GET /api/evolution/ab-tests/:agent/active
Get the active A/B test for a specific agent.

**Response:**
```json
{
  "agent": "anthropic-api",
  "has_active_test": true,
  "test": {
    "id": "test-123",
    "baseline_prompt_version": 1,
    "evolved_prompt_version": 2,
    "status": "active"
  }
}
```

#### POST /api/evolution/ab-tests
Create a new A/B test.

**Request Body:**
```json
{
  "agent": "anthropic-api",
  "baseline_version": 1,
  "evolved_version": 2,
  "metadata": {"description": "Test new reasoning patterns"}
}
```

**Response (201 Created):**
```json
{
  "message": "A/B test created",
  "test": {
    "id": "test-456",
    "agent": "anthropic-api",
    "status": "active"
  }
}
```

**Error (409 Conflict):** Agent already has an active test.

#### POST /api/evolution/ab-tests/:id/record
Record a debate result for an A/B test.

**Request Body:**
```json
{
  "debate_id": "debate-789",
  "variant": "evolved",
  "won": true
}
```

**Response:**
```json
{
  "message": "Result recorded",
  "test": {
    "id": "test-123",
    "baseline_wins": 5,
    "evolved_wins": 8
  }
}
```

#### POST /api/evolution/ab-tests/:id/conclude
Conclude an A/B test and determine the winner.

**Request Body:**
```json
{
  "force": false
}
```

**Response:**
```json
{
  "message": "A/B test concluded",
  "result": {
    "test_id": "test-123",
    "winner": "evolved",
    "confidence": 0.85,
    "recommendation": "Adopt evolved prompt",
    "stats": {
      "evolved_win_rate": 0.65,
      "baseline_win_rate": 0.35,
      "total_debates": 20
    }
  }
}
```

#### DELETE /api/evolution/ab-tests/:id
Cancel an active A/B test.

**Response:**
```json
{
  "message": "A/B test cancelled",
  "test_id": "test-123"
}
```

---

### Load-Bearing Claims

Identify claims with highest structural importance in debates.

#### GET /api/belief-network/:debate_id/load-bearing-claims
Get claims with highest centrality (most load-bearing).

**Parameters:**
- `limit` (int, default=5, max=20): Maximum claims to return

**Response:**
```json
{
  "debate_id": "debate-123",
  "load_bearing_claims": [
    {
      "claim_id": "claim-456",
      "statement": "The architecture must support horizontal scaling",
      "author": "anthropic-api",
      "centrality": 0.92
    }
  ],
  "count": 3
}
```

---

### Calibration Summary

Comprehensive agent calibration analysis.

#### GET /api/agent/:name/calibration-summary
Get comprehensive calibration summary for an agent.

**Parameters:**
- `domain` (string, optional): Filter by domain

**Response:**
```json
{
  "agent": "anthropic-api",
  "domain": null,
  "total_predictions": 250,
  "total_correct": 215,
  "accuracy": 0.86,
  "brier_score": 0.12,
  "ece": 0.05,
  "is_overconfident": false,
  "is_underconfident": true
}
```

#### GET /api/calibration/leaderboard
Get agents ranked by calibration quality (accuracy vs confidence).

**Parameters:**
- `limit` (int, default=10, max=50): Maximum agents to return
- `domain` (string, optional): Filter by domain

**Response:**
```json
{
  "agents": [
    {
      "name": "anthropic-api",
      "elo": 1542,
      "calibration_score": 0.92,
      "brier_score": 0.08,
      "accuracy": 0.89,
      "games": 45
    },
    {
      "name": "grok",
      "elo": 1518,
      "calibration_score": 0.88,
      "brier_score": 0.11,
      "accuracy": 0.85,
      "games": 38
    }
  ],
  "count": 2
}
```

---

### Continuum Memory

Multi-timescale memory system with surprise-weighted importance scoring.

#### GET /api/memory/continuum/retrieve
Retrieve memories from the continuum memory system.

**Parameters:**
- `query` (string, optional): Search query for memory content
- `tiers` (string, default="fast,medium"): Comma-separated tier names (fast, medium, slow, glacial)
- `limit` (int, default=10, max=50): Maximum memories to return
- `min_importance` (float, default=0.0): Minimum importance threshold (0.0-1.0)

**Response:**
```json
{
  "query": "error patterns",
  "tiers": ["FAST", "MEDIUM"],
  "memories": [
    {
      "id": "mem-123",
      "tier": "FAST",
      "content": "TypeError pattern in agent responses",
      "importance": 0.85,
      "surprise_score": 0.3,
      "consolidation_score": 0.7,
      "success_rate": 0.92,
      "update_count": 15,
      "created_at": "2026-01-03T10:00:00Z",
      "updated_at": "2026-01-04T08:00:00Z"
    }
  ],
  "count": 1
}
```

#### GET /api/memory/continuum/consolidate
Run memory consolidation and get tier transition statistics.

**Response:**
```json
{
  "consolidation": {
    "promoted": 5,
    "demoted": 2,
    "pruned": 1
  },
  "message": "Memory consolidation complete"
}
```

---

### Formal Verification

#### POST /api/verification/formal-verify
Attempt formal verification of a claim using Z3 SMT solver.

**Request Body:**
```json
{
  "claim": "If X > Y and Y > Z, then X > Z",
  "claim_type": "logical",
  "context": "Transitivity property of greater-than",
  "timeout": 30
}
```

**Parameters:**
- `claim` (required): The claim to verify
- `claim_type` (optional): Hint for the type (assertion, logical, arithmetic, constraint)
- `context` (optional): Additional context for translation
- `timeout` (optional): Timeout in seconds (default: 30, max: 120)

**Response:**
```json
{
  "claim": "If X > Y and Y > Z, then X > Z",
  "status": "proof_found",
  "is_verified": true,
  "language": "z3_smt",
  "formal_statement": "(declare-const X Int)...",
  "proof_hash": "a1b2c3d4e5f6",
  "proof_search_time_ms": 15.2,
  "prover_version": "z3-4.12.0"
}
```

**Status Values:**
- `proof_found`: Claim is formally verified
- `proof_failed`: Counterexample found (claim is false)
- `translation_failed`: Could not translate to formal language
- `timeout`: Solver timed out
- `not_supported`: Claim type not suitable for formal proof
- `backend_unavailable`: Z3 not installed

---

### Insight Extraction

#### POST /api/insights/extract-detailed
Extract detailed insights from debate content.

**Request Body:**
```json
{
  "content": "The debate transcript content...",
  "debate_id": "debate-123",
  "extract_claims": true,
  "extract_evidence": true,
  "extract_patterns": true
}
```

**Response:**
```json
{
  "debate_id": "debate-123",
  "content_length": 5420,
  "claims": [
    {
      "text": "Therefore, we should adopt this approach",
      "position": 12,
      "type": "argument"
    }
  ],
  "evidence_chains": [
    {
      "text": "According to the 2024 study",
      "type": "citation",
      "source": "the 2024 study"
    }
  ],
  "patterns": [
    {
      "type": "causal_reasoning",
      "strength": "strong",
      "instances": 5
    }
  ]
}
```

**Pattern Types:**
- `balanced_comparison`: Uses "on one hand... on the other hand"
- `concession_rebuttal`: Uses "while... however"
- `enumerated_argument`: Uses "first... second..."
- `conditional_reasoning`: Uses "if... then"
- `causal_reasoning`: Uses "because" (with instance count)

---

## Capability Probing

#### POST /api/probes/run
Run capability probes against an agent to detect vulnerabilities.

**Request Body:**
```json
{
  "agent": "anthropic-api",
  "strategies": ["contradiction", "hallucination"],
  "probe_count": 3
}
```

**Response:**
```json
{
  "agent": "anthropic-api",
  "probe_count": 3,
  "available_strategies": ["contradiction", "hallucination", "sycophancy", "persistence"],
  "status": "ready"
}
```

---

## Red Team Analysis

#### POST /api/debates/:id/red-team
Run adversarial analysis on a debate's conclusions.

**Request Body:**
```json
{
  "attack_types": ["steelman", "strawman"],
  "intensity": 5
}
```

**Response:**
```json
{
  "debate_id": "debate-123",
  "task": "Security implementation",
  "consensus_reached": true,
  "intensity": 5,
  "available_attacks": ["steelman", "strawman", "edge_case", "assumption_probe", "counterexample"],
  "status": "ready"
}
```

---

## Usage Examples

Common API operations with curl. Replace `localhost:8080` with your server address.

### Health Check

```bash
# Check server health
curl http://localhost:8080/api/health
```

### Starting a Debate

```bash
# Start a new debate
curl -X POST http://localhost:8080/api/debates/start \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Is microservices architecture better than monolith?",
    "agents": ["anthropic-api", "openai-api", "gemini"],
    "rounds": 3
  }'
```

### Listing and Viewing Debates

```bash
# List recent debates (default limit: 20)
curl http://localhost:8080/api/debates

# List with custom limit
curl "http://localhost:8080/api/debates?limit=50"

# Get specific debate details
curl http://localhost:8080/api/debates/debate-abc123
```

### Exporting Debates

```bash
# Export as JSON
curl http://localhost:8080/api/debates/debate-abc123/export?format=json

# Export messages as CSV
curl http://localhost:8080/api/debates/debate-abc123/export?format=csv&table=messages \
  -o debate-messages.csv

# Export as standalone HTML
curl http://localhost:8080/api/debates/debate-abc123/export?format=html \
  -o debate-report.html
```

### Agent Information

```bash
# Get leaderboard
curl http://localhost:8080/api/agent/leaderboard

# Get agent profile
curl http://localhost:8080/api/agent/anthropic-api/profile

# Compare two agents
curl "http://localhost:8080/api/agent/compare?a=anthropic-api&b=openai-api"

# Get head-to-head record
curl http://localhost:8080/api/agent/anthropic-api/head-to-head/openai-api
```

### Nomic Loop Status

```bash
# Get current nomic loop state
curl http://localhost:8080/api/nomic/state

# Get nomic loop logs (last 100 lines)
curl http://localhost:8080/api/nomic/log

# Get more log lines
curl "http://localhost:8080/api/nomic/log?lines=500"
```

### Tournaments

```bash
# List tournaments
curl http://localhost:8080/api/tournaments

# Get tournament details
curl http://localhost:8080/api/tournaments/tourney-abc123

# Get tournament bracket
curl http://localhost:8080/api/tournaments/tourney-abc123/bracket
```

### Authenticated Requests

```bash
# With bearer token
curl http://localhost:8080/api/debates \
  -H "Authorization: Bearer your-token-here"

# Tokens in query parameters are not accepted; use Authorization header instead
```

### WebSocket Connection (wscat)

```bash
# Connect to WebSocket for real-time events
wscat -c ws://localhost:8765/ws

# With authentication token
wscat -c ws://localhost:8765/ws -H "Authorization: Bearer your-token-here"
```

---

## WebSocket API

Connect to the WebSocket server for real-time streaming:

```javascript
const ws = new WebSocket('ws://localhost:8765/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.type, data.data);
};
```

### Event Types

WebSocket events use a shared envelope and are documented in
`docs/WEBSOCKET_EVENTS.md`. Filter by `loop_id` to scope to a single debate.

Common debate lifecycle events:
- `debate_start`, `round_start`, `agent_message`, `critique`, `vote`, `consensus`, `debate_end`

Token streaming events:
- `token_start`, `token_delta`, `token_end`

Control messages (on connect / acknowledgements):
- `connection_info`, `loop_list`, `sync`, `ack`, `error`, `auth_revoked`

### Event Format

```json
{
  "type": "agent_message",
  "data": { "content": "I disagree...", "role": "critic" },
  "timestamp": 1732735053.123,
  "round": 2,
  "agent": "anthropic-api",
  "loop_id": "loop-abc123",
  "seq": 42
}
```

---

## Formal Verification

Formal claim verification using theorem provers.

#### POST /api/verification/formal-verify
Attempt formal verification of a logical claim.

**Request Body:**
```json
{
  "claim": "If A implies B and B implies C, then A implies C",
  "debate_id": "debate-123",
  "context": "Optional context for the claim"
}
```

**Response:**
```json
{
  "verified": true,
  "method": "lean",
  "proof_sketch": "...",
  "confidence": 0.95
}
```

---

### Detailed Insights

#### POST /api/insights/extract-detailed
Extract detailed insights from debate content.

**Request Body:**
```json
{
  "debate_id": "debate-123",
  "content": "The debate transcript or summary...",
  "focus": "security"
}
```

**Response:**
```json
{
  "insights": [
    {
      "type": "pattern",
      "description": "Security vulnerability pattern detected",
      "confidence": 0.85,
      "actionable": true
    }
  ],
  "count": 3
}
```

---

### Formal Verification

Attempt formal verification of claims using SMT solvers.

#### GET /api/verification/status
Get status of formal verification backends.

**Response:**
```json
{
  "available": true,
  "backends": [
    {"language": "z3_smt", "available": true},
    {"language": "lean4", "available": false}
  ]
}
```

#### POST /api/verification/formal-verify
Attempt formal verification of a claim.

**Request Body:**
```json
{
  "claim": "If X > Y and Y > Z, then X > Z",
  "claim_type": "logical",
  "context": "Transitivity check",
  "timeout": 30
}
```

**Response:**
```json
{
  "status": "proof_found",
  "is_verified": true,
  "language": "z3_smt",
  "formal_statement": "(assert (not (=> (and (> x y) (> y z)) (> x z))))",
  "proof_hash": "a1b2c3d4e5f6",
  "proof_search_time_ms": 15.4
}
```

---

### Analytics & Patterns

Analyze debate patterns and agent behavior.

#### GET /api/analytics/disagreements
Get analysis of agent disagreement patterns.

**Parameters:**
- `limit` (int, default=20, max=100): Maximum entries

**Response:**
```json
{
  "patterns": [
    {
      "topic": "Error handling strategies",
      "agents": ["anthropic-api", "gemini"],
      "disagreement_rate": 0.73,
      "debates_count": 8
    }
  ]
}
```

#### GET /api/analytics/role-rotation
Get role rotation analysis across debates.

**Parameters:**
- `limit` (int, default=50, max=200): Maximum entries

#### GET /api/analytics/early-stops
Get early termination signals and patterns.

**Parameters:**
- `limit` (int, default=20, max=100): Maximum entries

---

### Agent Network & Relationships

Analyze agent relationships based on debate history.

#### GET /api/agent/:name/network
Get agent's relationship network with other agents.

**Response:**
```json
{
  "agent": "anthropic-api",
  "connections": [
    {"agent": "gemini", "relationship": "rival", "strength": 0.8},
    {"agent": "deepseek", "relationship": "ally", "strength": 0.65}
  ]
}
```

#### GET /api/agent/:name/rivals
Get agent's top rivals (agents they often disagree with).

**Parameters:**
- `limit` (int, default=5, max=20): Maximum rivals

#### GET /api/agent/:name/allies
Get agent's top allies (agents they often agree with).

**Parameters:**
- `limit` (int, default=5, max=20): Maximum allies

#### GET /api/agent/:name/positions
Get agent's historical positions on topics.

**Parameters:**
- `limit` (int, default=50, max=200): Maximum positions

---

### Laboratory & Emergent Traits

Track emergent agent behaviors and traits.

#### GET /api/laboratory/emergent-traits
Get emergent traits discovered across agents.

**Parameters:**
- `min_confidence` (float, default=0.5): Minimum confidence threshold
- `limit` (int, default=10, max=50): Maximum traits

**Response:**
```json
{
  "traits": [
    {
      "trait": "contrarian_stance",
      "agents": ["grok"],
      "confidence": 0.82,
      "evidence_count": 15
    }
  ]
}
```

#### POST /api/laboratory/cross-pollinations/suggest
Suggest trait cross-pollination between agents.

---

### Belief Network

Analyze claim dependencies and belief propagation.

#### GET /api/belief-network/:debate_id/cruxes
Get debate cruxes (key disagreement points).

**Parameters:**
- `top_k` (int, default=3, max=10): Number of cruxes to return

**Response:**
```json
{
  "debate_id": "debate-123",
  "cruxes": [
    {
      "claim": "Performance matters more than readability",
      "centrality": 0.85,
      "agents_for": ["gemini"],
      "agents_against": ["anthropic-api"]
    }
  ]
}
```

#### GET /api/belief-network/:debate_id/load-bearing-claims
Get claims that most influence the debate outcome.

**Parameters:**
- `limit` (int, default=5, max=20): Maximum claims

---

### Tournaments

Agent tournament management.

#### GET /api/tournaments
List tournaments.

**Response:**
```json
{
  "tournaments": [
    {
      "id": "t-001",
      "name": "Weekly Championship",
      "status": "in_progress",
      "participants": 8
    }
  ]
}
```

#### GET /api/tournaments/:id/standings
Get tournament standings and bracket.

---

### Modes

Available stress-test and interaction modes.

#### GET /api/modes
List available modes.

**Response:**
```json
{
  "modes": [
    {"name": "standard", "description": "Standard decision stress-test (multi-agent debate engine)"},
    {"name": "adversarial", "description": "Red-team adversarial mode"},
    {"name": "consensus", "description": "Consensus-building mode"}
  ]
}
```

---

### Routing & Team Selection

Agent selection and team composition.

#### GET /api/routing/best-teams
Get best performing agent team combinations.

**Parameters:**
- `min_debates` (int, default=3, max=20): Minimum debates together
- `limit` (int, default=10, max=50): Maximum teams

**Response:**
```json
{
  "teams": [
    {
      "agents": ["anthropic-api", "gemini", "deepseek"],
      "win_rate": 0.85,
      "debates": 12,
      "avg_confidence": 0.82
    }
  ]
}
```

#### POST /api/routing/recommendations
Get agent routing recommendations for a topic.

**Request Body:**
```json
{
  "topic": "API security design",
  "required_count": 3
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

---

## Python Module APIs

The following sections document the Python APIs for extending Aragora.

### Plugin System (`aragora.plugins`)

The plugin system provides a sandboxed execution environment for code analysis, linting, and security scanning extensions.

#### PluginManifest

Defines plugin metadata and capabilities.

```python
from aragora.plugins.manifest import PluginManifest, PluginCapability, PluginRequirement

manifest = PluginManifest(
    name="my-linter",
    version="1.0.0",
    description="Custom code linter",
    entry_point="my_linter:run",  # module:function format
    capabilities=[PluginCapability.LINT, PluginCapability.CODE_ANALYSIS],
    requirements=[PluginRequirement.READ_FILES],
    timeout_seconds=30.0,
    tags=["python", "lint"],
)

# Validate manifest
valid, errors = manifest.validate()
```

**PluginCapability Enum:**
- `CODE_ANALYSIS` - Analyze code structure
- `LINT` - Check code style/issues
- `SECURITY_SCAN` - Security vulnerability detection
- `FORMAT` - Code formatting
- `TEST` - Test execution

**PluginRequirement Enum:**
- `READ_FILES` - Read file access
- `WRITE_FILES` - Write file access
- `NETWORK` - Network access
- `SUBPROCESS` - Subprocess execution

#### PluginRunner

Executes plugins in a sandboxed environment with restricted builtins.

```python
from aragora.plugins.runner import PluginRunner, PluginContext

runner = PluginRunner(manifest)

# Create execution context
ctx = PluginContext(
    working_dir="/path/to/project",
    input_data={"files": ["main.py", "utils.py"]},
)
ctx.allowed_operations = {"read_files"}

# Run plugin
result = await runner.run(ctx, timeout_override=60.0)

if result.success:
    print(result.output)
else:
    print(result.errors)
```

**Security Features:**
- Restricted builtins (no `exec`, `eval`, `compile`, `__import__`, `open`)
- Path traversal prevention (cannot access parent directories)
- Configurable timeouts
- Capability-based permissions

#### PluginRegistry

Manages plugin discovery and instantiation.

```python
from aragora.plugins.runner import get_registry, run_plugin

# Get global registry
registry = get_registry()

# List all plugins
plugins = registry.list_plugins()

# Get plugins by capability
lint_plugins = registry.list_by_capability(PluginCapability.LINT)

# Run a plugin by name
result = await run_plugin("lint", {"files": ["main.py"]})
```

---

### Genesis System (`aragora.genesis`)

The Genesis system implements evolutionary agent algorithms with genetic operators and provenance tracking.

#### AgentGenome

Represents an agent's genetic makeup with traits, expertise, and fitness tracking.

```python
from aragora.genesis import AgentGenome, generate_genome_id

# Create a genome
genome = AgentGenome(
    genome_id=generate_genome_id(
        traits={"analytical": 0.8, "creative": 0.6},
        expertise={"security": 0.9},
        parents=[]
    ),
    name="security-specialist-v1",
    traits={"analytical": 0.8, "creative": 0.6},
    expertise={"security": 0.9, "backend": 0.7},
    model_preference="anthropic-api",
    generation=0,
    fitness_score=0.5,
)

# Create from existing Persona
genome = AgentGenome.from_persona(persona, model="gemini")

# Convert back to Persona for debate use
persona = genome.to_persona()

# Update fitness based on debate outcome
genome.update_fitness(
    consensus_win=True,
    critique_accepted=True,
    prediction_correct=False
)

# Calculate genetic similarity (0-1)
similarity = genome1.similarity_to(genome2)
```

#### GenomeStore

SQLite-based persistence for genomes.

```python
from aragora.genesis import GenomeStore

store = GenomeStore(db_path=".nomic/genesis.db")

# Save genome
store.save(genome)

# Retrieve by ID
genome = store.get("abc123def456")

# Get by name (returns latest generation)
genome = store.get_by_name("security-specialist")

# Get top performers
top_10 = store.get_top_by_fitness(n=10)

# Get lineage (ancestors)
lineage = store.get_lineage(genome_id)  # [child, parent, grandparent, ...]
```

#### GenomeBreeder

Genetic operators for evolving agent populations.

```python
from aragora.genesis import GenomeBreeder

breeder = GenomeBreeder(
    mutation_rate=0.1,      # Probability of mutating each trait
    crossover_ratio=0.5,    # Blend ratio (0.5 = equal parents)
    elite_ratio=0.2,        # Fraction preserved unchanged
)

# Crossover: blend two parents
child = breeder.crossover(
    parent_a=genome1,
    parent_b=genome2,
    name="hybrid-agent",
    debate_id="debate-123"
)

# Mutation: random modifications
mutated = breeder.mutate(genome, rate=0.2)

# Spawn specialist: create domain-focused agent
specialist = breeder.spawn_specialist(
    domain="security",
    parent_pool=[genome1, genome2, genome3],
    debate_id="debate-123"
)
```

#### Population

Collection of genomes with aggregate statistics.

```python
from aragora.genesis import Population

pop = Population(
    population_id="gen-5",
    genomes=[genome1, genome2, genome3],
    generation=5,
)

print(f"Size: {pop.size}")
print(f"Average fitness: {pop.average_fitness:.2f}")
print(f"Best: {pop.best_genome.name}")

# Find by ID
genome = pop.get_by_id("abc123")
```

#### GenesisLedger

Immutable event log with cryptographic hashing.

```python
from aragora.genesis import GenesisLedger, GenesisEvent, GenesisEventType

ledger = GenesisLedger(db_path=".nomic/genesis.db")

# Events are automatically hashed and linked
# Common event types:
# - DEBATE_START, DEBATE_END, DEBATE_SPAWN, DEBATE_MERGE
# - CONSENSUS_REACHED, TENSION_DETECTED, TENSION_RESOLVED
# - AGENT_BIRTH, AGENT_DEATH, AGENT_MUTATION, AGENT_CROSSOVER
# - FITNESS_UPDATE, POPULATION_EVOLVED, GENERATION_ADVANCE
```

#### FractalTree

Tree structure for nested sub-debates.

```python
from aragora.genesis import FractalTree

tree = FractalTree(root_id="main-debate")
tree.add_node("main-debate", parent_id=None, depth=0, success=True)
tree.add_node("sub-1", parent_id="main-debate", tension="scope", depth=1)
tree.add_node("sub-2", parent_id="main-debate", tension="definition", depth=1)

children = tree.get_children("main-debate")  # ["sub-1", "sub-2"]
nested_dict = tree.to_dict()  # Recursive structure
```

---

### Evolution System (`aragora.evolution`)

Prompt evolution enables agents to improve their system prompts based on successful debate patterns.

#### PromptEvolver

Main class for evolving agent prompts.

```python
from aragora.evolution import PromptEvolver, EvolutionStrategy

evolver = PromptEvolver(
    db_path="aragora_evolution.db",
    strategy=EvolutionStrategy.HYBRID,
)

# Extract patterns from successful debates
patterns = evolver.extract_winning_patterns(
    debates=recent_debates,
    min_confidence=0.6,
)

# Store patterns for future use
evolver.store_patterns(patterns)

# Get most effective patterns
top_patterns = evolver.get_top_patterns(
    pattern_type="issue_identification",
    limit=10,
)

# Evolve an agent's prompt
new_prompt = evolver.evolve_prompt(
    agent=my_agent,
    patterns=top_patterns,
    strategy=EvolutionStrategy.APPEND,
)

# Apply evolution and save version
evolver.apply_evolution(agent, patterns)

# Track performance
evolver.update_performance(
    agent_name="anthropic-api",
    version=3,
    debate_result=result,
)
```

#### EvolutionStrategy

Available strategies for prompt evolution:

```python
from aragora.evolution import EvolutionStrategy

# APPEND: Add new learnings section to existing prompt
# REPLACE: Replace old learnings section with new patterns
# REFINE: Use LLM to synthesize patterns into coherent prompt
# HYBRID: Append first, refine if prompt exceeds 2000 chars
```

#### Prompt Versioning

Track prompt versions and their performance:

```python
# Save a new version
version = evolver.save_prompt_version(
    agent_name="anthropic-api",
    prompt="You are a helpful assistant...",
    metadata={"source": "manual", "note": "Added security focus"},
)

# Get specific version
v1 = evolver.get_prompt_version("anthropic-api", version=1)

# Get latest version
latest = evolver.get_prompt_version("anthropic-api")

# Get evolution history
history = evolver.get_evolution_history("anthropic-api", limit=10)
# Returns: [{"from_version": 2, "to_version": 3, "strategy": "append", ...}, ...]
```

---

## Breakpoints API

Human-in-the-loop breakpoint management for debate supervision.

### List Pending Breakpoints

```http
GET /api/breakpoints/pending
```

Returns all breakpoints awaiting human resolution.

**Response:**
```json
{
  "breakpoints": [
    {
      "id": "bp_abc123",
      "debate_id": "debate_xyz",
      "type": "consensus_uncertain",
      "created_at": "2026-01-09T10:00:00Z",
      "context": {
        "agents_involved": ["anthropic-api", "openai-api"],
        "disagreement_score": 0.85
      }
    }
  ]
}
```

### Get Breakpoint Status

```http
GET /api/breakpoints/:id/status
```

Returns the current status of a specific breakpoint.

### Resolve Breakpoint

```http
POST /api/breakpoints/:id/resolve
```

Resolve a pending breakpoint with human guidance.

**Request Body:**
```json
{
  "resolution": "continue",
  "guidance": "The agents should focus on practical implementation",
  "override_winner": null
}
```

---

## Introspection API

Agent self-awareness and reputation metrics.

### Get All Agent Introspection

```http
GET /api/introspection/all
```

Returns introspection data for all agents.

### Get Introspection Leaderboard

```http
GET /api/introspection/leaderboard
```

Returns agents ranked by reputation score.

**Response:**
```json
{
  "rankings": [
    {
      "agent": "anthropic-api",
      "reputation_score": 0.92,
      "consistency": 0.88,
      "win_rate": 0.67
    }
  ]
}
```

### List Available Agents

```http
GET /api/introspection/agents
```

Returns list of agents available for introspection.

### Get Agent Introspection

```http
GET /api/introspection/agents/:name
```

Returns detailed introspection for a specific agent.

---

## Gallery API

Public debate gallery for sharing and embedding.

### List Public Debates

```http
GET /api/gallery
```

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 20 | Max debates to return |
| `offset` | int | 0 | Pagination offset |
| `agent` | string | - | Filter by agent name |

**Response:**
```json
{
  "debates": [
    {
      "id": "gallery_abc123",
      "title": "Should AI systems be open-sourced?",
      "agents": ["anthropic-api", "openai-api"],
      "winner": "anthropic-api",
      "created_at": "2026-01-09T10:00:00Z",
      "view_count": 1234
    }
  ],
  "total": 156,
  "has_more": true
}
```

### Get Public Debate

```http
GET /api/gallery/:id
```

Returns full debate history for a public debate.

### Get Embeddable Summary

```http
GET /api/gallery/:id/embed
```

Returns an embeddable summary suitable for sharing.

**Response:**
```json
{
  "embed_html": "<div class='aragora-embed'>...</div>",
  "og_tags": {
    "title": "AI Debate: Open Source AI",
    "description": "Claude vs GPT-4 debate on open-source AI",
    "image": "https://api.aragora.ai/og/gallery_abc123.png"
  }
}
```

---

## Billing API

Subscription and usage management (requires authentication).

### List Plans

```http
GET /api/billing/plans
```

Returns available subscription plans with features and pricing.

**Response:**
```json
{
  "plans": [
    {
      "id": "free",
      "name": "Free",
      "price": 0,
      "debates_per_month": 10,
      "features": ["basic_agents", "public_gallery"]
    },
    {
      "id": "pro",
      "name": "Pro",
      "price": 29,
      "debates_per_month": 500,
      "features": ["all_agents", "private_debates", "api_access"]
    }
  ]
}
```

### Get Current Usage

```http
GET /api/billing/usage
```

Returns current usage for the authenticated user.

**Response:**
```json
{
  "debates_used": 45,
  "debates_limit": 500,
  "api_calls_used": 1234,
  "api_calls_limit": 10000,
  "period_ends": "2026-02-01T00:00:00Z"
}
```

### Get Subscription Status

```http
GET /api/billing/subscription
```

Returns current subscription details.

### Create Checkout Session

```http
POST /api/billing/checkout
```

Create a Stripe checkout session for plan upgrade.

**Request Body:**
```json
{
  "plan_id": "pro",
  "success_url": "https://aragora.ai/billing/success",
  "cancel_url": "https://aragora.ai/billing/cancel"
}
```

**Response:**
```json
{
  "checkout_url": "https://checkout.stripe.com/..."
}
```

### Create Billing Portal Session

```http
POST /api/billing/portal
```

Create a Stripe billing portal session for subscription management.

### Cancel Subscription

```http
POST /api/billing/cancel
```

Cancel subscription at end of current billing period.

### Resume Subscription

```http
POST /api/billing/resume
```

Resume a previously canceled subscription.

### Stripe Webhook

```http
POST /api/webhooks/stripe
```

Handle Stripe webhook events (subscription updates, payment events).

---

## Deprecated Endpoints

The following endpoints are deprecated and will be removed in future versions.

| Endpoint | Status | Removal | Migration |
|----------|--------|---------|-----------|
| `GET /api/debates/list` | Deprecated | v2.0 | Use `GET /api/debates` |
| `POST /api/debate/new` | Deprecated | v2.0 | Use `POST /api/debates/start` |
| `GET /api/elo/rankings` | Deprecated | v2.0 | Use `GET /api/agent/leaderboard` |
| `GET /api/agent/elo` | Deprecated | v2.0 | Use `GET /api/agent/{name}/profile` |
| `POST /api/stream/start` | Deprecated | v2.0 | Use WebSocket `/ws` connection |

### Migration Guide

#### Debate Listing

```bash
# Old (deprecated)
curl http://localhost:8080/api/debates/list

# New (recommended)
curl http://localhost:8080/api/debates
```

#### Starting Debates

```bash
# Old (deprecated)
curl -X POST http://localhost:8080/api/debate/new \
  -d '{"question": "..."}'

# New (recommended)
curl -X POST http://localhost:8080/api/debates/start \
  -H "Content-Type: application/json" \
  -d '{"question": "...", "agents": ["anthropic-api", "openai-api"]}'
```

#### Agent Rankings

```bash
# Old (deprecated)
curl http://localhost:8080/api/elo/rankings
curl http://localhost:8080/api/agent/elo?name=anthropic-api

# New (recommended)
curl http://localhost:8080/api/agent/leaderboard
curl http://localhost:8080/api/agent/anthropic-api/profile
```

### Deprecation Timeline

- **v1.5** (Current): Deprecated endpoints still functional, emit deprecation warnings in response headers
- **v2.0** (Planned): Deprecated endpoints return 410 Gone status
- **v2.1** (Planned): Deprecated endpoints removed entirely

Response headers for deprecated endpoints:

```
Deprecation: true
Sunset: 2026-06-01
Link: </api/debates>; rel="successor-version"
```

---

## Changelog

### 2026-01-09
- Added Graph Debates API (4 endpoints for branching debates)
- Added Matrix Debates API (4 endpoints for parallel scenarios)
- Added Breakpoints API (3 endpoints for human-in-the-loop)
- Added Introspection API (4 endpoints for agent self-awareness)
- Added Gallery API (3 endpoints for public debate sharing)
- Added Billing API (8 endpoints for subscription management)
- Updated endpoint count from 106 to 124

### 2026-01-05
- Added CSV and HTML export formats for debates
- Added token streaming events (`token_start`, `token_delta`, `token_end`)
- Added phase timeout events (`phase_start` now includes timeout, added `phase_timeout`)
- Added usage examples section with curl commands
- Added deprecated endpoints documentation with migration guide
