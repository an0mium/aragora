# Aragora Capability Matrix

> Source of truth: generated via `python scripts/generate_capability_matrix.py`
> OpenAPI source: `openapi.json`

## Executive Summary

| Surface | Inventory | Capability Coverage |
|---------|-----------|---------------------|
| **HTTP API** | 1772 paths / 2100 operations | 100.0% |
| **CLI** | 75 commands | 85.7% |
| **SDK (Python)** | 184 namespaces | 100.0% |
| **SDK (TypeScript)** | 183 namespaces | 100.0% |
| **UI** | 14/14 capabilities | 100.0% |
| **Capability Catalog** | 14/37 mapped | 37.8% |

## Capability Surface Matrix

| Capability | API | CLI | SDK | UI | Channels |
|------------|-----|-----|-----|----|----------|
| `agent_team_selection` | ✅ | ✅ | ✅ | ✅ | ❌ |
| `compliance_framework` | ✅ | ❌ | ✅ | ✅ | ❌ |
| `consensus_detection` | ✅ | ✅ | ✅ | ✅ | ❌ |
| `continuum_memory` | ✅ | ✅ | ✅ | ✅ | ❌ |
| `debate_orchestration` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `decision_integrity` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `graph_debates` | ✅ | ✅ | ✅ | ✅ | ❌ |
| `knowledge_mound` | ✅ | ✅ | ✅ | ✅ | ❌ |
| `marketplace` | ✅ | ✅ | ✅ | ✅ | ❌ |
| `matrix_debates` | ✅ | ✅ | ✅ | ✅ | ❌ |
| `nomic_loop` | ✅ | ✅ | ✅ | ✅ | ❌ |
| `rbac_v2` | ✅ | ❌ | ✅ | ✅ | ❌ |
| `vertical_specialists` | ✅ | ✅ | ✅ | ✅ | ❌ |
| `workflow_engine` | ✅ | ✅ | ✅ | ✅ | ❌ |

**Coverage:** API 14/14 (100%) | CLI 12/14 (85.7%) | SDK 14/14 (100%) | UI 14/14 (100%) | Channels 2/14 (14.3%)

## Surface Gaps

### Missing API (0)

- None

### Missing CLI (2)

- `compliance_framework`
- `rbac_v2`

### Missing SDK (0)

- None

### Missing UI (0)

- None

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
