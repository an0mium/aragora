---
slug: 007-selection-plugin-architecture
title: "ADR-007: Selection Plugin Architecture"
description: "ADR-007: Selection Plugin Architecture"
---

# ADR-007: Selection Plugin Architecture

## Status
Accepted

## Context

The agent selection system (`aragora/routing/selection.py`) determines which agents participate in debates and what roles they play. The original implementation hardcoded:
- Scoring algorithm (ELO + domain expertise + calibration)
- Team selection strategy (quality-diversity balance)
- Role assignment logic (domain-based)

This made it difficult to:
- Experiment with alternative selection strategies
- A/B test different algorithms
- Allow users to customize selection behavior
- Add ML-based selection without modifying core code

## Decision

Implement a Protocol-based plugin architecture in `aragora/plugins/selection/`:

### Core Protocols
```python
class ScorerProtocol(Protocol):
    def score_agent(self, agent, requirements, context) -> float: ...

class TeamSelectorProtocol(Protocol):
    def select_team(self, scored_agents, requirements, context) -> list: ...

class RoleAssignerProtocol(Protocol):
    def assign_roles(self, team, requirements, context, phase) -> dict: ...
```

### Plugin Registry
```python
@register_scorer("my-scorer")
class MyScorer:
    def score_agent(self, agent, requirements, context):
        return agent.elo_rating / 2000  # Custom logic
```

### Built-in Strategies
- `ELOWeightedScorer`: Default ELO + expertise scoring
- `DiverseTeamSelector`: Quality-diversity balanced selection
- `GreedyTeamSelector`: Simple top-N selection
- `RandomTeamSelector`: Weighted random for exploration
- `DomainBasedRoleAssigner`: Expertise-based role assignment
- `SimpleRoleAssigner`: Positional role assignment

### SelectionContext
Plugins receive a `SelectionContext` with:
- Agent pool and profiles
- System integrations (ELO, calibration, probes)
- Performance insights
- Selection history for meta-learning

## Consequences

### Positive
- **Extensibility**: New strategies without core changes
- **Testability**: Strategies can be tested in isolation
- **A/B testing**: Easy to swap strategies for experiments
- **Type safety**: Protocol-based design catches errors early
- **Discovery**: Registry provides introspection of available plugins

### Negative
- **Complexity**: Additional abstraction layer
- **Learning curve**: Contributors must understand Protocol pattern
- **State management**: Context must be properly populated

### Neutral
- Default behavior unchanged (existing strategies registered as defaults)
- Decorator-based registration is optional (manual registration also works)
- Protocols use `runtime_checkable` for flexibility

## Example: Custom ML-Based Scorer

```python
from aragora.plugins.selection import register_scorer, SelectionContext

@register_scorer("ml-scorer")
class MLScorer:
    def __init__(self):
        self.model = load_model("selection_model.pt")

    @property
    def name(self) -> str:
        return "ml-scorer"

    @property
    def description(self) -> str:
        return "ML-based agent scoring using historical debate outcomes"

    def score_agent(self, agent, requirements, context):
        features = self._extract_features(agent, requirements)
        return self.model.predict(features)
```

## Related
- `aragora/plugins/selection/` - Plugin implementation
- `aragora/routing/selection.py` - Original selection system
- `tests/plugins/test_selection_plugins.py` - Plugin tests
