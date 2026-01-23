---
title: HTTP Handler Reference
description: HTTP Handler Reference
---

# HTTP Handler Reference

This document indexes all HTTP handlers in `aragora/server/handlers/`.

## Overview

The server uses a modular handler architecture organized by domain. Each handler extends `BaseHandler` and registers routes it can handle. Counts vary by deployment; run `python scripts/generate_api_docs.py --format json` to enumerate. See the [Experimental Handlers](#experimental-handlers) section for features in development.

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
| `auth/signup_handlers.py` | `/api/v1/auth/*` | Self-service signup, org setup, invitations |
| `auth/sso_handlers.py` | `/api/v1/auth/sso/*` | OIDC-based SSO endpoints |
| `sso.py` | `/auth/sso/*` | SAML/legacy SSO endpoints |

### Key Endpoints

```
POST /api/auth/login      - User login
POST /api/auth/logout     - User logout
GET  /api/auth/me         - Current user info
GET  /api/oauth/:provider - OAuth initiation
POST /api/oauth/callback  - OAuth callback
POST /api/v1/auth/signup  - Self-service signup
GET  /api/v1/auth/sso/login - SSO login (OIDC)
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
| `features/advertising.py` | `/api/v1/advertising/*` | Advertising platform APIs |
| `features/analytics_platforms.py` | `/api/v1/analytics/*` | Analytics/BI platform APIs |
| `features/crm.py` | `/api/v1/crm/*` | CRM platform APIs |
| `features/ecommerce.py` | `/api/v1/ecommerce/*` | Ecommerce platform APIs |
| `features/support.py` | `/api/v1/support/*` | Support/helpdesk platform APIs |
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
| `accounting.py` | `/api/accounting/*` | QuickBooks + Gusto accounting |
| `ap_automation.py` | `/api/v1/accounting/ap/*` | Accounts payable automation |
| `ar_automation.py` | `/api/v1/accounting/ar/*` | Accounts receivable automation |
| `audit_export.py` | `/api/audit/*` | Audit log export |
| `auditing.py` | `/api/auditing/*` | Security analysis |
| `belief.py` | `/api/belief/*` | Belief network analysis |
| `breakpoints.py` | `/api/breakpoints/*` | Human-in-the-loop |
| `checkpoints.py` | `/api/checkpoints/*` | Debate checkpoints |
| `consensus.py` | `/api/consensus/*` | Consensus memory |
| `critique.py` | `/api/critique/*` | Critique patterns |
| `code_review.py` | `/api/v1/code-review/*` | Multi-agent code review |
| `dashboard.py` | `/api/v1/dashboard/*` | Main dashboard API |
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
| `connectors.py` | `/api/connectors/*` | Data connector management |
| `control_plane.py` | `/api/control-plane/*` | Enterprise control plane |
| `routing.py` | `/api/routing/*` | Agent routing & team selection |
| `selection.py` | `/api/selection/*` | Selection plugin API |
| `evaluation.py` | `/api/evaluation/*` | LLM-as-Judge evaluation |
| `knowledge.py` | `/api/knowledge/*` | Knowledge base API |
| `ml.py` | `/api/ml/*` | ML capabilities API |
| `policy.py` | `/api/policy/*` | Policy compliance |
| `queue.py` | `/api/queue/*` | Job queue management |
| `repository.py` | `/api/repository/*` | Repository indexing |
| `uncertainty.py` | `/api/uncertainty/*` | Uncertainty estimation |
| `verticals.py` | `/api/verticals/*` | Vertical specialists |
| `workspace.py` | `/api/workspace/*` | Enterprise workspace |
| `workflows.py` | `/api/workflows/*` | Workflow engine API |

---

## Experimental Handlers

These handlers are functional but APIs may change. Check `aragora/server/handlers/__init__.py` for current stability levels.

| Handler | Phase | Description |
|---------|-------|-------------|
| `AnalyticsDashboardHandler` | EXPERIMENTAL | Enterprise analytics dashboard |
| `RoutingHandler` | Phase A | Agent routing and team selection |
| `SelectionHandler` | Phase A | Selection plugin API |
| `ControlPlaneHandler` | Phase 0 | Enterprise control plane |
| `DocumentQueryHandler` | Phase A | Natural language document querying |
| `EvaluationHandler` | EXPERIMENTAL | LLM-as-Judge evaluation |
| `EvidenceEnrichmentHandler` | Phase A | Evidence enrichment for findings |
| `FindingWorkflowHandler` | Phase A | Finding workflow management |
| `FolderUploadHandler` | Phase A | Folder upload support |
| `KnowledgeHandler` | EXPERIMENTAL | Knowledge base API |
| `KnowledgeMoundHandler` | Phase A1 | Knowledge Mound system |
| `MLHandler` | EXPERIMENTAL | ML capabilities API |
| `PolicyHandler` | Phase 2 | Policy and compliance management |
| `QueueHandler` | Phase A1 | Job queue management |
| `RepositoryHandler` | Phase A3 | Repository indexing |
| `SchedulerHandler` | Phase A | Audit scheduling |
| `SlackHandler` | EXPERIMENTAL | Slack integration |
| `UncertaintyHandler` | Phase A1 | Uncertainty estimation |
| `VerticalsHandler` | Phase A1 | Vertical specialist API |
| `WorkflowHandler` | Phase 2 | Workflow engine API |
| `WorkspaceHandler` | Phase 2 | Enterprise workspace/privacy |

### Stability Levels

- **STABLE**: Production-ready, backwards compatible
- **PREVIEW**: Feature complete, API may change slightly
- **EXPERIMENTAL**: In development, expect breaking changes
- **Phase 0/A/A1/A3/2**: Enterprise feature rollout phases

---

## Utility Modules

| Module | Description |
|--------|-------------|
| `base.py` | Base handler class with common utilities |
| `exceptions.py` | Handler exception types |
| `interface.py` | Handler Protocol definitions and contracts |
| `types.py` | Type definitions |
| `utilities.py` | Shared handler utility functions |
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

- [API Endpoints](../api/endpoints) - Full endpoint reference
- [API Rate Limits](../api/rate-limits) - Rate limiting configuration
- [WebSocket Events](../guides/websocket-events) - Real-time event streaming
