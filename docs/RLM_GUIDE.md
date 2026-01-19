# RLM (Recursive Language Model) Patterns in Aragora

This guide explains how Aragora implements RLM-inspired patterns for efficient multi-agent debates, based on research from [Prime Intellect's RLM](https://www.primeintellect.ai/blog/rlm).

## Overview

RLM patterns focus on **iterative refinement with early termination** rather than context compression. The key insight is that agents can signal when their position has stabilized, allowing the system to terminate early and save compute.

## Core Patterns

### 1. Ready Signal Pattern

Agents emit "ready signals" when their position has stabilized and further debate won't change their stance.

#### Signal Format

Agents can emit ready signals in several formats:

```html
<!-- READY_SIGNAL: {"confidence": 0.9, "ready": true, "reasoning": "Position finalized"} -->
```

```markdown
```ready_signal
{"confidence": 0.85, "ready": true}
```
```

```
[READY: confidence=0.9, ready=true]
```

Or natural language:
- "I have reached my final position on this matter"
- "No further refinement needed"
- "My stance is now final"

#### Implementation

Located in `aragora/debate/phases/debate_rounds.py`:

```python
from aragora.debate.phases.debate_rounds import (
    AgentReadinessSignal,
    CollectiveReadiness,
    parse_ready_signal,
    RLM_READY_CONFIDENCE_THRESHOLD,  # 0.8
    RLM_READY_QUORUM_THRESHOLD,      # 0.75
)

# Parse an agent's response for ready signals
signal = parse_ready_signal("claude", response_content, round_num=3)

if signal.should_terminate():
    print(f"{signal.agent} is ready with {signal.confidence:.0%} confidence")
```

#### Collective Readiness

The system tracks readiness across all agents:

```python
from aragora.debate.phases.debate_rounds import CollectiveReadiness

collective = CollectiveReadiness()
collective.add_signal(signal1)
collective.add_signal(signal2)

if collective.has_quorum(total_agents=4):
    print(f"Quorum reached! {collective.ready_count} of {total_agents} agents ready")
    print(f"Average confidence: {collective.average_confidence:.0%}")
```

### 2. Early Termination in Vote Collection

Vote collection uses "batch parallelism" with early termination when a clear majority emerges.

#### Configuration

Located in `aragora/debate/phases/vote_collector.py`:

```python
from aragora.debate.phases.vote_collector import (
    VoteCollector,
    VoteCollectorConfig,
    create_vote_collector,
    RLM_EARLY_TERMINATION_THRESHOLD,  # 0.75 (75% of votes needed)
    RLM_MAJORITY_LEAD_THRESHOLD,      # 0.25 (25% lead required)
)

# Create collector with RLM optimization enabled
collector = create_vote_collector(
    enable_rlm_early_termination=True,
    rlm_early_termination_threshold=0.75,
    rlm_majority_lead_threshold=0.25,
)
```

#### How It Works

1. **Parallel Vote Collection**: Votes are collected in parallel batches
2. **Majority Detection**: After each vote, the system checks for clear majority
3. **Early Termination**: If 75%+ votes collected AND clear winner exists, remaining votes are cancelled

```python
# Early termination criteria:
# 1. At least 75% of votes have been collected
# 2. Leader has > 50% of total possible votes
# 3. Lead over second place >= 25% of total agents

def _check_clear_majority(votes, total_agents):
    vote_counts = Counter(v.choice for v in votes)
    leader, leader_count = vote_counts.most_common(1)[0]
    second_count = vote_counts.most_common(2)[1][1] if len(vote_counts) > 1 else 0

    # Check majority
    has_majority = leader_count > total_agents / 2

    # Check lead
    lead = leader_count - second_count
    min_lead = total_agents * RLM_MAJORITY_LEAD_THRESHOLD

    return has_majority and lead >= min_lead, leader
```

### 3. Protocol Configuration

Enable RLM patterns in debate protocols:

```python
from aragora import DebateProtocol

protocol = DebateProtocol(
    rounds=5,
    consensus="majority",

    # RLM Ready Signal settings
    enable_rlm_ready_signals=True,
    rlm_ready_confidence_threshold=0.8,
    rlm_ready_quorum_threshold=0.75,

    # RLM Vote Collection settings
    enable_rlm_early_termination=True,
    rlm_early_termination_threshold=0.75,
    rlm_majority_lead_threshold=0.25,
)
```

## Performance Benefits

### Compute Savings

| Scenario | Without RLM | With RLM | Savings |
|----------|-------------|----------|---------|
| 5 agents, clear consensus at round 2 | 5 rounds, 25 turns | 2 rounds, 10 turns | 60% |
| 10 agents voting, 8-2 split | 10 votes | 8 votes | 20% |
| Unanimous agreement | Full debate | Early termination | 40-70% |

### When RLM Helps Most

- **High-agreement debates**: Clear consensus emerges early
- **Large agent pools**: More opportunity for early termination
- **Expensive models**: Savings multiply with model cost

### When to Disable

- **Exploratory debates**: Want full discussion even with consensus
- **Audit requirements**: Need complete vote records
- **Close decisions**: Don't want to miss late-changing votes

## Hooks and Events

### Ready Signal Hooks

```python
from aragora import Arena, ArenaConfig

def on_ready_signal(agent: str, signal: dict, round_num: int):
    print(f"Agent {agent} signaled ready at round {round_num}")

config = ArenaConfig(
    hooks={
        "on_rlm_ready_signal": on_ready_signal,
        "on_rlm_quorum_reached": lambda agents, confidence: print(f"Quorum!"),
    }
)
```

### Vote Collection Hooks

```python
collector = create_vote_collector(
    hooks={
        "on_rlm_early_termination": lambda leader, votes, total:
            print(f"Early termination: {leader} wins with {votes}/{total}")
    }
)
```

### WebSocket Events

The following events are emitted for real-time monitoring:

| Event | Data | Description |
|-------|------|-------------|
| `rlm_ready_signal` | `{agent, confidence, round}` | Agent signaled ready |
| `rlm_quorum_reached` | `{agents, avg_confidence}` | Quorum threshold met |
| `rlm_early_termination` | `{leader, votes_collected, total}` | Vote collection terminated |

## Best Practices

### 1. Prompt Engineering for Ready Signals

Include instructions in agent prompts:

```
When you have reached your final position and further debate would not change
your stance, include a ready signal:

<!-- READY_SIGNAL: {"confidence": 0.9, "ready": true, "reasoning": "..."} -->

Only signal ready when you are confident (>80%) that additional arguments
won't change your conclusion.
```

### 2. Threshold Tuning

| Use Case | Ready Threshold | Quorum | Vote Threshold |
|----------|-----------------|--------|----------------|
| Quick decisions | 0.7 | 0.6 | 0.6 |
| Balanced (default) | 0.8 | 0.75 | 0.75 |
| Thorough analysis | 0.9 | 0.9 | 0.9 |
| No early termination | 1.0 | 1.0 | 1.0 |

### 3. Monitoring

Use the observability dashboard (`/observability`) to track:
- Average rounds before quorum
- Early termination frequency
- Compute savings metrics

## API Reference

### AgentReadinessSignal

```python
@dataclass
class AgentReadinessSignal:
    agent: str              # Agent name
    confidence: float       # 0.0-1.0 confidence score
    ready: bool            # Whether agent is ready
    reasoning: str         # Optional reasoning
    round_num: int         # Round when signal was emitted

    def should_terminate(self) -> bool:
        """Check if this signal indicates termination readiness."""
        return self.ready and self.confidence >= RLM_READY_CONFIDENCE_THRESHOLD
```

### CollectiveReadiness

```python
@dataclass
class CollectiveReadiness:
    signals: Dict[str, AgentReadinessSignal]

    def add_signal(self, signal: AgentReadinessSignal) -> None
    def has_quorum(self, total_agents: int) -> bool
    @property
    def ready_count(self) -> int
    @property
    def ready_agents(self) -> List[str]
    @property
    def average_confidence(self) -> float
```

### VoteCollectorConfig

```python
@dataclass
class VoteCollectorConfig:
    enable_rlm_early_termination: bool = True
    rlm_early_termination_threshold: float = 0.75
    rlm_majority_lead_threshold: float = 0.25
    vote_with_agent: Optional[Callable] = None
    hooks: Dict[str, Callable] = field(default_factory=dict)
    notify_spectator: Optional[Callable] = None
```

## Further Reading

- [Prime Intellect RLM Blog Post](https://www.primeintellect.ai/blog/rlm)
- [RLM Paper (arXiv:2512.24601)](https://arxiv.org/abs/2512.24601)
- [Aragora Memory Strategy](./MEMORY_STRATEGY.md)
- [Debate Phases](./DEBATE_PHASES.md)
