# HTTP Handler Reference

This document indexes all HTTP handlers in `aragora/server/handlers/`.

## Overview

The server uses a modular handler architecture with 89 handler modules organized by domain. Each handler extends `BaseHandler` and registers routes it can handle.

## Handler Structure

```
handlers/
├── admin/           # Administration & system management
├── agents/          # Agent management & leaderboards
├── auth/            # Authentication & authorization
├── debates/         # Core debate operations
├── evolution/       # Prompt evolution & A/B testing
├── features/        # Feature modules (audio, evidence, etc.)
├── memory/          # Memory & learning systems
├── social/          # Social features & integrations
├── utils/           # Shared utilities
└── verification/    # Formal verification
```

---

## Admin Handlers

| Module | Routes | Description |
|--------|--------|-------------|
| `admin/admin.py` | `/api/admin/*` | System-wide administration, user/org listing |
| `admin/billing.py` | `/api/billing/*` | Subscription & payment management |
| `admin/dashboard.py` | `/api/dashboard/*` | Consolidated debate metrics |
| `admin/health.py` | `/health`, `/ready` | Health & readiness probes |
| `admin/system.py` | `/api/system/*` | System config & maintenance |

### Key Endpoints

```
GET  /health              - Health check
GET  /ready               - Readiness check
GET  /api/admin/users     - List users (admin only)
GET  /api/admin/stats     - System statistics
GET  /api/dashboard       - Aggregated metrics
```

---

## Agent Handlers

| Module | Routes | Description |
|--------|--------|-------------|
| `agents/agents.py` | `/api/agents/*` | Agent registration, status, configuration |
| `agents/calibration.py` | `/api/calibration/*` | Calibration curves & prediction accuracy |
| `agents/leaderboard.py` | `/api/leaderboard/*` | ELO rankings & tournament results |
| `agents/probes.py` | `/api/probes/*` | Capability probing & adversarial testing |

### Key Endpoints

```
GET  /api/agents                    - List agents
GET  /api/agents/:name              - Get agent details
GET  /api/leaderboard               - ELO rankings
GET  /api/calibration/:agent        - Agent calibration data
POST /api/probes/run                - Run capability probes
```

---

## Auth Handlers

| Module | Routes | Description |
|--------|--------|-------------|
| `auth/handler.py` | `/api/auth/*` | Login, logout, session management |
| `oauth.py` | `/api/oauth/*` | OAuth2 provider integration |
| `sso.py` | `/api/sso/*` | SAML/SSO enterprise auth |

### Key Endpoints

```
POST /api/auth/login      - User login
POST /api/auth/logout     - User logout
GET  /api/auth/me         - Current user info
GET  /api/oauth/:provider - OAuth initiation
POST /api/oauth/callback  - OAuth callback
```

---

## Debate Handlers

| Module | Routes | Description |
|--------|--------|-------------|
| `debates/handler.py` | `/api/debates/*` | Core debate CRUD operations |
| `debates/batch.py` | `/api/debates/batch/*` | Batch debate submission |
| `debates/fork.py` | `/api/debates/:id/fork` | Counterfactual forking |
| `debates/graph_debates.py` | `/api/graph-debates/*` | Branching graph debates |
| `debates/matrix_debates.py` | `/api/matrix-debates/*` | Multi-scenario matrix debates |

### Key Endpoints

```
POST /api/debates               - Create debate
GET  /api/debates/:id           - Get debate status
GET  /api/debates/:id/result    - Get debate result
POST /api/debates/:id/vote      - Submit vote
POST /api/debates/:id/fork      - Fork debate
POST /api/graph-debates         - Create graph debate
POST /api/matrix-debates        - Create matrix debate
```

---

## Feature Handlers

| Module | Routes | Description |
|--------|--------|-------------|
| `features/audio.py` | `/api/audio/*` | Audio transcription & narration |
| `features/broadcast.py` | `/api/broadcast/*` | Live debate broadcasting |
| `features/documents.py` | `/api/documents/*` | Document processing |
| `features/evidence.py` | `/api/evidence/*` | Evidence collection & search |
| `features/features.py` | `/api/features/*` | Feature flags & toggles |
| `features/plugins.py` | `/api/plugins/*` | Plugin management |
| `features/pulse.py` | `/api/pulse/*` | Trending topics |

### Key Endpoints

```
GET  /api/features              - List feature flags
POST /api/evidence/collect      - Collect evidence
GET  /api/pulse/trending        - Get trending topics
POST /api/audio/transcribe      - Transcribe audio
POST /api/broadcast/start       - Start live broadcast
```

---

## Memory Handlers

| Module | Routes | Description |
|--------|--------|-------------|
| `memory/memory.py` | `/api/memory/*` | Memory tier management |
| `memory/insights.py` | `/api/insights/*` | Extracted insights |
| `memory/learning.py` | `/api/learning/*` | Continuous learning |
| `memory/memory_analytics.py` | `/api/memory-analytics/*` | Memory usage analytics |

### Key Endpoints

```
GET  /api/memory/:agent         - Agent memory
POST /api/memory/recall         - Recall memories
GET  /api/insights              - List insights
GET  /api/memory-analytics      - Memory statistics
```

---

## Social Handlers

| Module | Routes | Description |
|--------|--------|-------------|
| `social/collaboration.py` | `/api/collaboration/*` | Team debate collaboration |
| `social/notifications.py` | `/api/notifications/*` | User notifications |
| `social/relationship.py` | `/api/relationship/*` | Agent relationships |
| `social/sharing.py` | `/api/share/*` | Debate sharing |
| `social/slack.py` | `/api/slack/*` | Slack integration |
| `social/social_media.py` | `/api/social/*` | Social media posting |

### Key Endpoints

```
POST /api/share/:id             - Share debate
GET  /api/notifications         - User notifications
GET  /api/relationship/:a/:b    - Agent relationship
POST /api/slack/post            - Post to Slack
```

---

## Verification Handlers

| Module | Routes | Description |
|--------|--------|-------------|
| `verification/verification.py` | `/api/verify/*` | Claim verification |
| `verification/formal_verification.py` | `/api/formal/*` | Formal proofs (Z3/Lean) |

### Key Endpoints

```
POST /api/verify/claim          - Verify a claim
GET  /api/verify/status/:id     - Verification status
POST /api/formal/prove          - Generate formal proof
```

---

## Other Handlers

| Module | Routes | Description |
|--------|--------|-------------|
| `analytics.py` | `/api/analytics/*` | Usage analytics |
| `audit_export.py` | `/api/audit/*` | Audit log export |
| `auditing.py` | `/api/auditing/*` | Security analysis |
| `belief.py` | `/api/belief/*` | Belief network analysis |
| `breakpoints.py` | `/api/breakpoints/*` | Human-in-the-loop |
| `checkpoints.py` | `/api/checkpoints/*` | Debate checkpoints |
| `consensus.py` | `/api/consensus/*` | Consensus memory |
| `critique.py` | `/api/critique/*` | Critique patterns |
| `docs.py` | `/api/docs/*` | API documentation |
| `gallery.py` | `/api/gallery/*` | Debate gallery |
| `gauntlet.py` | `/api/gauntlet/*` | Adversarial testing |
| `genesis.py` | `/api/genesis/*` | Agent breeding |
| `introspection.py` | `/api/introspection/*` | Agent introspection |
| `laboratory.py` | `/api/laboratory/*` | Persona experiments |
| `metrics.py` | `/api/metrics/*` | Prometheus metrics |
| `moments.py` | `/api/moments/*` | Significant moments |
| `nomic.py` | `/api/nomic/*` | Nomic loop control |
| `organizations.py` | `/api/organizations/*` | Organization management |
| `persona.py` | `/api/persona/*` | Persona management |
| `privacy.py` | `/api/privacy/*` | Data privacy |
| `replays.py` | `/api/replays/*` | Debate replays |
| `reviews.py` | `/api/reviews/*` | Debate reviews |
| `tournaments.py` | `/api/tournaments/*` | Tournament management |
| `training.py` | `/api/training/*` | Training data export |
| `webhooks.py` | `/api/webhooks/*` | Webhook management |

---

## Utility Modules

| Module | Description |
|--------|-------------|
| `base.py` | Base handler class with common utilities |
| `exceptions.py` | Handler exception types |
| `types.py` | Type definitions |
| `utils/database.py` | Database access helpers |
| `utils/decorators.py` | Handler decorators |
| `utils/params.py` | Query parameter parsing |
| `utils/rate_limit.py` | Rate limiting utilities |
| `utils/responses.py` | Response formatting |
| `utils/routing.py` | Route matching utilities |
| `utils/safe_data.py` | Data sanitization |

---

## Creating New Handlers

```python
from aragora.server.handlers.base import (
    BaseHandler,
    json_response,
    error_response,
)

class MyHandler(BaseHandler):
    """My custom handler."""

    routes = [
        "GET /api/my-feature",
        "POST /api/my-feature",
    ]

    def can_handle(self, path: str) -> bool:
        return path.startswith("/api/my-feature")

    def handle(self, path, query_params, handler):
        return json_response({"status": "ok"})

    def handle_post(self, path, query_params, handler):
        body, err = self.read_json_body_validated(handler)
        if err:
            return err
        return json_response({"received": body})
```

Register in `handlers/__init__.py`:

```python
from .my_handler import MyHandler

ALL_HANDLERS = [
    # ... existing handlers ...
    MyHandler,
]
```

---

## See Also

- [API Endpoints](API_ENDPOINTS.md) - Full endpoint reference
- [API Rate Limits](API_RATE_LIMITS.md) - Rate limiting configuration
- [WebSocket Events](WEBSOCKET_EVENTS.md) - Real-time event streaming
