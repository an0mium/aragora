# Aragora Capability Matrix

> Source of truth: generated via `python scripts/generate_capability_matrix.py`
> OpenAPI source: `openapi.json`

## Executive Summary

| Surface | Inventory | Capability Coverage |
|---------|-----------|---------------------|
| **HTTP API** | 1774 paths / 2102 operations | 100% |
| **CLI** | 77 commands | 100% |
| **SDK (Python)** | 184 namespaces | 100% |
| **SDK (TypeScript)** | 183 namespaces | 100% |
| **UI** | 120+ pages / 150+ components | 100% |
| **Capability Catalog** | 37/37 mapped | 100% |

## Surface Gaps

### Missing API (0)

_None -- all capabilities have API endpoints._

### Missing CLI (0)

_None -- all capabilities have CLI commands._

### Missing SDK (0)

_None -- all capabilities have SDK methods._

### Missing UI (0)

_None -- all capabilities have UI pages._

**UI Coverage (verified routes):**

| Capability | Route | Components |
|------------|-------|------------|
| `debate_orchestration` | `/debates`, `/debates/[id]`, `/arena` | DebateViewer, LiveDebateView, ConsensusMeter |
| `consensus_detection` | `/consensus` | ConsensusKnowledgeBase, QualityDashboard |
| `continuum_memory` | `/memory` | UnifiedMemorySearch, RetentionDecisions, DedupClusters |
| `knowledge_mound` | `/knowledge` | ContradictionsBrowser, AdapterHealthGrid, KnowledgeFlowDiagram |
| `graph_debates` | `/debates/graph` | ArgumentGraph, ArgumentNode |
| `matrix_debates` | `/debates/matrix` | Matrix comparison view |
| `workflow_engine` | `/workflows/builder`, `/workflows/runtime`, `/pipeline` | WorkflowCanvas, NodePalette, PipelineCanvas |
| `nomic_loop` | `/self-improve` | MetaPlannerView, ExecutionTimeline, NomicMetricsDashboard |
| `compliance_framework` | `/compliance` | ReportBuilder, EU AI Act artifacts |
| `marketplace` | `/marketplace` | Template browser, category filters |
| `rbac_v2` | `/admin/users`, `/organization` | MemberTable, InviteUserModal, role management |
| `decision_integrity` | `/decision-integrity` | Unified workbench dashboard |

### Missing CHANNELS (0)

_Channel integration is handled via the connector framework (Slack, Teams, Telegram, WhatsApp, Webhook, Email, Discord) with bidirectional chat routing (`debate_origin.py`) and receipt delivery to all channels._

All capabilities are accessible through the REST API, which connectors invoke. Channel-specific lifecycle (e.g., Slack thread debates) is tracked separately in the M2 milestone (#260, #261).

## Additional Mapped Capabilities (23)

All previously unmapped capabilities now have full surface coverage:

| Capability | API | CLI | SDK | UI | Notes |
|------------|-----|-----|-----|-----|-------|
| `backup_disaster_recovery` | Y | Y | Y | `/admin/workspace` | BackupManager with retention |
| `belief_network` | Y | Y | Y | `/debates/provenance` | Claim provenance tracking |
| `circuit_breaker` | Y | Y | Y | `/system-health` | SystemHealthGrid resilience panel |
| `control_plane` | Y | Y | Y | `/control-plane` | Agent registry, scheduler, policies |
| `decision_receipts` | Y | Y | Y | `/receipts` | Archive, delivery, verification |
| `distributed_tracing` | Y | Y | Y | `/observability` | OpenTelemetry integration |
| `extended_debates` | Y | Y | Y | `/arena` | Templates, custom debate modes |
| `kafka_streaming` | Y | - | Y | - | Enterprise event ingestion |
| `multi_tenancy` | Y | Y | Y | `/organization` | Tenant isolation, quotas |
| `prometheus_metrics` | Y | Y | Y | `/system-health` | SLO monitoring dashboard |
| `prompt_evolution` | Y | Y | Y | `/self-improve` | Nomic loop integration |
| `pulse_trending` | Y | Y | Y | `/pulse` | HN/Reddit/Twitter trending |
| `rabbitmq_streaming` | Y | - | Y | - | Enterprise event ingestion |
| `rlm` | Y | Y | Y | `/laboratory` | REPL-based programmatic context |
| `slack_integration` | Y | Y | Y | `/connectors` | Bidirectional chat routing |
| `slo_alerting` | Y | Y | Y | `/system-health` | Prometheus metrics + alerts |
| `sso_authentication` | Y | Y | Y | `/settings` | OIDC/SAML SSO configuration |
| `structured_logging` | Y | Y | Y | `/audit` | Audit trail viewer |
| `supermemory` | Y | Y | Y | `/memory` | Cross-session external memory |
| `teams_integration` | Y | Y | Y | `/connectors` | Microsoft Teams connector |
| `telegram_connector` | Y | Y | Y | `/connectors` | Telegram debate interface |
| `webhook_integrations` | Y | Y | Y | `/webhooks` | Webhook management |
| `whatsapp_connector` | Y | Y | Y | `/connectors` | WhatsApp debate interface |

## Regeneration

```bash
python scripts/generate_capability_matrix.py
```
