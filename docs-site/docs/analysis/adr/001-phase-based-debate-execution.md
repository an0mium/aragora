---
slug: 001-phase-based-debate-execution
title: "ADR-001: Phase-Based Debate Execution"
description: "ADR-001: Phase-Based Debate Execution"
---

# ADR-001: Phase-Based Debate Execution

## Status
Accepted

## Context

The original `Arena._run_inner()` method grew to over 1,300 lines, handling all debate logic in a single monolithic function:
- Context initialization
- Proposal generation
- Debate rounds (critique/revision)
- Consensus detection
- Analytics and metrics
- Feedback and ELO updates

This made the code difficult to:
- Test individual phases in isolation
- Modify one phase without risk to others
- Understand the overall flow
- Add new phases or modify existing ones

## Decision

Extract each debate phase into a dedicated class with a standardized interface:

```
aragora/debate/phases/
├── context_init.py      # Phase 0: Context initialization
├── proposal_phase.py    # Phase 1: Initial proposals
├── debate_rounds.py     # Phase 2: Critique/revision loop
├── consensus_phase.py   # Phase 3: Voting and consensus
├── analytics_phase.py   # Phases 4-6: Metrics and insights
└── feedback_phase.py    # Phase 7: ELO and memory updates
```

Each phase:
1. Receives a `DebateContext` with shared state
2. Implements an `async execute(ctx)` method
3. Updates context with results
4. Can be tested independently

A `PhaseExecutor` orchestrates phase execution with:
- OpenTelemetry tracing per phase
- Error handling and recovery
- Timeout management
- Phase skip/retry logic

## Consequences

### Positive
- **Testability**: Each phase can be unit tested in isolation
- **Maintainability**: Changes to one phase don't affect others
- **Observability**: Clear tracing boundaries per phase
- **Extensibility**: New phases can be added without modifying core orchestrator
- **Reduced orchestrator size**: From 1,300+ lines to ~180 lines

### Negative
- **Increased file count**: 18 new files in phases/ directory
- **Context passing**: Shared state must be carefully managed
- **Learning curve**: Developers must understand phase boundaries

### Neutral
- Some phases (consensus, feedback) remain large and may need further decomposition
- Vote collection/aggregation logic partially extracted to helper modules

## Related
- `aragora/debate/orchestrator.py` - Reduced coordinator
- `aragora/debate/phase_executor.py` - Phase orchestration
- `aragora/debate/context.py` - Shared debate context
