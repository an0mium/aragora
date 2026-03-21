---
title: Architecture Decision Records (ADRs)
description: Architecture Decision Records (ADRs)
---

# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) documenting significant architectural decisions made in the Aragora project.

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [001](./001-phase-based-debate-execution) | Phase-Based Debate Execution | Accepted | Jan 2026 |
| [002](./002-agent-fallback-openrouter) | Agent Fallback via OpenRouter | Accepted | Jan 2026 |
| [003](./003-multi-tier-memory-system) | Multi-Tier Memory System | Accepted | Jan 2026 |
| [004](./004-incremental-type-safety) | Incremental Type Safety Migration | Accepted | Jan 2026 |
| [005](./005-composition-over-inheritance) | Composition Over Inheritance for APIs | Accepted | Jan 2026 |
| [006](./006-api-versioning-strategy) | API Versioning Strategy | Accepted | Jan 2026 |
| [007](./007-selection-plugin-architecture) | Selection Plugin Architecture | Accepted | Jan 2026 |
| [008](./008-rlm-semantic-compression) | RLM Semantic Compression | Accepted | Jan 2026 |
| [009](./009-control-plane-architecture) | Control Plane Architecture | Accepted | Jan 2026 |
| [010](./010-debate-orchestration-pattern) | Debate Orchestration Pattern | Accepted | Jan 2026 |
| [011](./011-multi-tier-memory-comparison) | Multi-Tier Memory Comparison | Accepted | Jan 2026 |
| [012](./012-agent-fallback-strategy) | Agent Fallback Strategy | Accepted | Jan 2026 |
| [013](./013-workflow-dag-design) | Workflow DAG Design | Accepted | Jan 2026 |
| [014](./014-knowledge-mound-architecture) | Knowledge Mound Architecture | Accepted | Jan 2026 |
| [015](./015-lazy-import-patterns) | Lazy Import Patterns | Accepted | Jan 2026 |
| [016](./016-marketplace-architecture) | Agent Template Marketplace Architecture | Accepted | Feb 2026 |
| [017](017-backend-runtime-entrypoint-and-compatibility.md) | Backend Runtime Entrypoint and Compatibility | Accepted | Mar 2026 |
| [018](018-self-hosted-worker-canonicalization.md) | Self-Hosted Worker Canonicalization | Accepted | Mar 2026 |
| [019](019-standardized-health-check-endpoints.md) | Standardized Health Check Endpoints | Accepted | Mar 2026 |
| [020](020-event-dispatch-consolidation.md) | Event Dispatch Consolidation | Accepted | Mar 2026 |
| [021](021-storage-layer-consolidation.md) | Storage Layer Consolidation | Accepted | Mar 2026 |

## ADR Format

Each ADR follows this template:

```markdown
# ADR-NNN: Title

## Status
[Proposed | Accepted | Deprecated | Superseded]

## Context
What is the issue that we're seeing that is motivating this decision?

## Decision
What is the change that we're proposing and/or doing?

## Consequences
What becomes easier or more difficult to do because of this change?
```

## Contributing

When making significant architectural changes:
1. Create a new ADR with the next available number
2. Follow the template format
3. Link related ADRs if applicable
4. Update this README index
