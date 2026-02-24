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
| **UI** | tracked in capability surfaces | 14.3% |
| **Capability Catalog** | 14/37 mapped | 37.8% |

## Surface Gaps

### Missing API (0)

_None -- all capabilities have API endpoints._

### Missing CLI (0)

_None -- all capabilities have CLI commands._

### Missing SDK (0)

_None -- all capabilities have SDK methods._

### Missing UI (12)

- `compliance_framework`
- `consensus_detection`
- `continuum_memory`
- `debate_orchestration`
- `decision_integrity`
- `graph_debates`
- `knowledge_mound`
- `marketplace`
- `matrix_debates`
- `nomic_loop`
- `rbac_v2`
- `workflow_engine`

### Missing CHANNELS (11)

- `agent_team_selection`
- `compliance_framework`
- `continuum_memory`
- `graph_debates`
- `knowledge_mound`
- `marketplace`
- `matrix_debates`
- `nomic_loop`
- `rbac_v2`
- `vertical_specialists`
- `workflow_engine`

## Unmapped Capabilities (23)

- `backup_disaster_recovery`
- `belief_network`
- `circuit_breaker`
- `control_plane`
- `decision_receipts`
- `distributed_tracing`
- `extended_debates`
- `kafka_streaming`
- `multi_tenancy`
- `prometheus_metrics`
- `prompt_evolution`
- `pulse_trending`
- `rabbitmq_streaming`
- `rlm`
- `slack_integration`
- `slo_alerting`
- `sso_authentication`
- `structured_logging`
- `supermemory`
- `teams_integration`
- `telegram_connector`
- `webhook_integrations`
- `whatsapp_connector`

## Regeneration

```bash
python scripts/generate_capability_matrix.py
```
