# Aragora Capability Matrix

> Source of truth: generated via `python scripts/generate_capability_matrix.py`
> OpenAPI source: `openapi.json`

## Executive Summary

| Surface | Inventory | Capability Coverage |
|---------|-----------|---------------------|
| **HTTP API** | 1635 paths / 1896 operations | 92.9% |
| **CLI** | 64 commands | 78.6% |
| **SDK (Python)** | 155 namespaces | 92.9% |
| **SDK (TypeScript)** | 154 namespaces | 92.9% |
| **UI** | tracked in capability surfaces | 14.3% |
| **Capability Catalog** | 14/37 mapped | 37.8% |

## Surface Gaps

### Missing API (1)

- `consensus_detection`

### Missing CLI (3)

- `compliance_framework`
- `consensus_detection`
- `rbac_v2`

### Missing SDK (1)

- `consensus_detection`

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

### Missing CHANNELS (12)

- `agent_team_selection`
- `compliance_framework`
- `consensus_detection`
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
