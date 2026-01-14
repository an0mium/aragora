# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) documenting significant architectural decisions made in the Aragora project.

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [001](001-phase-based-debate-execution.md) | Phase-Based Debate Execution | Accepted | Jan 2026 |
| [002](002-agent-fallback-openrouter.md) | Agent Fallback via OpenRouter | Accepted | Jan 2026 |
| [003](003-multi-tier-memory-system.md) | Multi-Tier Memory System | Accepted | Jan 2026 |
| [004](004-incremental-type-safety.md) | Incremental Type Safety Migration | Accepted | Jan 2026 |
| [005](005-composition-over-inheritance.md) | Composition Over Inheritance for APIs | Accepted | Jan 2026 |

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
