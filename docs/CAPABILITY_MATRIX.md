# Capability Matrix

This document maps user-facing capabilities to code and tests in the repo.
It is used to keep product claims aligned with implementation. It is not
exhaustive; update it whenever claims or implementations change.

Status legend:
- stable: intended for production use
- experimental: implemented but evolving or guarded by feature flags
- planned: documented elsewhere but not implemented (do not claim as current)

Version baseline: v2.5.0 (see `pyproject.toml` and `aragora/__version__.py`).

## Core Debate & Consensus

| Capability | Status | Code refs | Tests / validation | Notes |
| --- | --- | --- | --- | --- |
| Multi-agent debate orchestration (propose/critique/revise) | stable | `aragora/debate/orchestrator.py`, `aragora/debate/phases/` | `tests/debate/`, `tests/integration/` | Core engine |
| Debate protocol configuration (rounds, roles, consensus) | stable | `aragora/debate/protocol.py`, `aragora/debate/consensus.py` | `tests/debate/` | Majority, unanimous, judge |
| Semantic convergence detection (early stop) | stable | `aragora/debate/convergence.py` | `tests/debate/`, `tests/ml/` | Embedding + fallback |
| Debate graph / forking | experimental | `aragora/debate/graph.py`, `aragora/debate/forking.py` | `tests/debate/` | Parallel branch exploration |
| Vote weighting and calibration | experimental | `aragora/debate/phases/vote_weighter.py`, `aragora/ranking/` | `tests/ranking/`, `tests/debate/` | Reliability-weighted votes |

## Agents & Providers

| Capability | Status | Code refs | Tests / validation | Notes |
| --- | --- | --- | --- | --- |
| Agent base types and roles | stable | `aragora/core/__init__.py`, `aragora/core_types.py` | `tests/core/`, `tests/agents/` | Agent, Message, Critique, Vote |
| Agent factory and catalog | stable | `aragora/agents/base.py`, `aragora/agents/__init__.py` | `tests/agents/` | `create_agent`, registry |
| API agents (Anthropic/OpenAI/Gemini/etc.) | stable | `aragora/agents/api_agents/` | `tests/agents/`, `tests/integration/` | Requires provider API keys |
| CLI agents (claude/codex/gemini/etc.) | stable | `aragora/agents/cli_agents.py` | `tests/agents/` | Requires external CLI tools |
| Personas and grounded identities | experimental | `aragora/agents/personas.py`, `aragora/agents/grounded.py`, `aragora/agents/truth_grounding.py` | `tests/agents/`, `tests/insights/` | Traits, position tracking |
| Capability probing and red team modes | experimental | `aragora/modes/prober.py`, `aragora/modes/gauntlet.py` | `tests/modes/`, `tests/gauntlet/` | Adversarial probing |

## Knowledge, Memory, Evidence

| Capability | Status | Code refs | Tests / validation | Notes |
| --- | --- | --- | --- | --- |
| Knowledge Mound ingestion and retrieval | stable | `aragora/knowledge/`, `aragora/knowledge/mound/` | `tests/knowledge/` | Unified knowledge store |
| CritiqueStore (pattern learning) | stable | `aragora/memory/store.py` | `tests/memory/` | SQLite-backed patterns |
| Continuum memory tiers | stable | `aragora/memory/continuum/core.py` | `tests/memory/` | Fast/medium/slow/glacial |
| Semantic retrieval / embeddings | stable | `aragora/memory/embeddings.py`, `aragora/knowledge/embeddings.py` | `tests/memory/`, `tests/ml/` | Vector retrieval |
| Evidence collection and attribution | experimental | `aragora/evidence/`, `aragora/reasoning/citations.py` | `tests/evidence/`, `tests/reasoning/` | Provenance + citations |
| Claims and belief networks | experimental | `aragora/reasoning/claims.py`, `aragora/reasoning/belief.py` | `tests/reasoning/` | Structured claims |
| Formal verification hooks | experimental | `aragora/verification/` | `tests/verification/` | Z3-based checks |

## Outputs & Audit

| Capability | Status | Code refs | Tests / validation | Notes |
| --- | --- | --- | --- | --- |
| Decision receipts and exports | stable | `aragora/export/` | `tests/export/`, `tests/gauntlet/` | PDF/HTML/JSON outputs |
| Gauntlet stress testing | stable | `aragora/gauntlet/` | `tests/gauntlet/` | Red team, risk reports |
| Risk heatmaps | stable | `aragora/gauntlet/heatmap.py` | `tests/gauntlet/` | Severity visualization |
| Replay and audit trails | stable | `aragora/replay/`, `aragora/persistence/` | `tests/replay/`, `tests/persistence/` | Deterministic replay |

## Interfaces & Channels

| Capability | Status | Code refs | Tests / validation | Notes |
| --- | --- | --- | --- | --- |
| CLI entry points | stable | `aragora/cli/` | `tests/cli/` | `aragora ask`, `gauntlet`, `review` |
| HTTP API + handlers | stable | `aragora/server/unified_server.py`, `aragora/server/handlers/` | `tests/server/`, `tests/handlers/` | REST endpoints |
| WebSocket streaming | stable | `aragora/server/stream/`, `aragora/spectate/` | `tests/stream/`, `tests/server/` | Real-time events |
| Live dashboard (Next.js) | stable | `aragora/live/` | `aragora/live/__tests__/`, `aragora/live/e2e/` | UI + realtime |
| Python SDK (aragora-client) | stable | `aragora-py/aragora_client/`, `aragora/client/` | `aragora-py/tests/`, `tests/client/` | HTTP client |
| TypeScript SDK | stable | `sdk/typescript/` | `sdk/typescript/` | `@aragora/sdk` |
| TypeScript SDK (legacy) | deprecated | `aragora-js/` | `aragora-js/tests/` | `@aragora/client` |
| Bots and channel integrations | experimental | `aragora/bots/`, `aragora/channels/`, `aragora/integrations/` | `tests/bots/`, `tests/channels/`, `tests/integrations/` | Slack/Teams/etc. |
| External connectors | experimental | `aragora/connectors/` | `tests/connectors/` | Data source adapters |
| MCP server integration | experimental | `aragora/mcp/` | `tests/mcp/` | Claude Desktop / MCP |

## Operations & Deployment

| Capability | Status | Code refs | Tests / validation | Notes |
| --- | --- | --- | --- | --- |
| Auth, RBAC, tenancy | stable | `aragora/auth/`, `aragora/rbac/`, `aragora/tenancy/` | `tests/auth/`, `tests/rbac/`, `tests/tenancy/` | Access control |
| Rate limiting and security middleware | stable | `aragora/server/middleware/`, `aragora/security/` | `tests/middleware/`, `tests/security/` | API protections |
| Observability and telemetry | stable | `aragora/observability/`, `aragora/telemetry/` | `tests/observability/` | Metrics + tracing |
| Docker and Compose deployment | stable | `deploy/`, `docker-compose*.yml`, `Dockerfile` | Manual validation | Local + prod deploys |
| Kubernetes operator and manifests | experimental | `aragora-operator/`, `k8s/` | Manual validation | Scale-out ops |
| Nomic loop self-improvement | experimental | `scripts/nomic_loop.py`, `aragora/nomic/` | `tests/nomic/` | Guarded automation |
