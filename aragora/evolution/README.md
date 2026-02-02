# Evolution Module

Prompt evolution system for Aragora, enabling agents to improve their system prompts based on successful debate patterns, vulnerability mitigation, and A/B testing.

## Overview

The evolution module provides:

- **Pattern Extraction**: Mine winning patterns from successful debates
- **Prompt Evolution**: Automatically improve agent prompts based on patterns
- **Vulnerability Mitigation**: Evolve prompts to address discovered weaknesses
- **A/B Testing**: Scientific comparison of evolved vs baseline prompts
- **Performance Tracking**: Track agent performance across prompt generations

## Architecture

```
aragora/evolution/
├── __init__.py              # Module exports
├── evolver.py               # PromptEvolver for prompt improvement
├── pattern_extractor.py     # Extract patterns from debate outcomes
├── tracker.py               # EvolutionTracker for performance metrics
├── ab_testing.py            # A/B testing framework
└── database.py              # Database utilities
```

## Key Classes

### Pattern Extraction

- **`PatternExtractor`**: Extracts patterns from debate outcomes
  - Evidence usage patterns
  - Argument structure patterns
  - Persuasion technique patterns
  - Critique response patterns

- **`StrategyIdentifier`**: Identifies successful debate strategies
  - Evidence-based argumentation
  - Structured reasoning
  - Conciliatory approach
  - Direct challenge

- **`Pattern`**: Extracted pattern with metadata
- **`Strategy`**: Identified strategy with success rate

### Prompt Evolution

- **`PromptEvolver`**: Core evolution engine (SQLite-backed)
  - Multiple evolution strategies (append, replace, refine, hybrid)
  - Genetic operations (mutation, crossover)
  - Vulnerability-based evolution
  - Version tracking and history

- **`EvolutionStrategy`**: Evolution approach
  - APPEND: Add new instructions
  - REPLACE: Replace sections
  - REFINE: LLM-based refinement
  - HYBRID: Combination of strategies

- **`PromptVersion`**: Tracked prompt version with performance metrics

### Performance Tracking

- **`EvolutionTracker`**: Track outcomes across generations
  - Win/loss recording per agent
  - Generation-based metrics
  - Performance delta calculation
  - Trend analysis

### A/B Testing

- **`ABTestManager`**: Manage A/B tests
  - Start/conclude tests
  - Record results with variant tracking
  - Statistical significance checking
  - Automatic variant selection

- **`ABTest`**: Test state and statistics
- **`ABTestResult`**: Test conclusion with recommendation

## Usage Example

### Basic Pattern Extraction

```python
from aragora.evolution import (
    PatternExtractor,
    StrategyIdentifier,
    extract_patterns,
    identify_strategies,
)

# Use convenience functions
debate_outcome = {
    "winner": "claude",
    "messages": [
        {"agent": "claude", "content": "According to research, the evidence suggests..."},
        {"agent": "claude", "content": "First, we need to consider..."},
    ],
    "critiques": [
        {"to": "claude", "issues": ["Missing edge case"], "severity": 0.3},
    ],
    "consensus_reached": True,
}

# Extract winning patterns
patterns = extract_patterns(debate_outcome)
print(f"Found {patterns['pattern_count']} patterns")
for p in patterns["winning_patterns"]:
    print(f"  - {p['pattern_type']}: {p['description']}")

# Identify strategies
strategies = identify_strategies(debate_outcome)
for s in strategies:
    print(f"  - {s['name']}: {s['success_rate']:.0%}")

# Use extractors directly for more control
extractor = PatternExtractor()
patterns = extractor.extract(debate_outcome)

identifier = StrategyIdentifier()
strategies = identifier.identify(debate_outcome)
```

### Prompt Evolution

```python
from aragora.evolution import (
    PromptEvolver,
    EvolutionStrategy,
)
from aragora.core import Agent

# Create evolver with desired strategy
evolver = PromptEvolver(
    strategy=EvolutionStrategy.HYBRID,
    mutation_rate=0.1,
)

# Extract patterns from successful debates
debates = [...]  # List of DebateResult objects
patterns = evolver.extract_winning_patterns(debates, min_confidence=0.7)
print(f"Extracted {len(patterns)} patterns")

# Store patterns for future use
evolver.store_patterns(patterns)

# Get top patterns
top_patterns = evolver.get_top_patterns(limit=10)
for p in top_patterns:
    print(f"  - {p['type']}: {p['text']} (effectiveness: {p['effectiveness']:.2f})")

# Evolve an agent's prompt
agent = Agent(name="claude", system_prompt="You are a helpful assistant...")

# Option 1: Get new prompt without applying
new_prompt = evolver.evolve_prompt(agent, patterns=top_patterns)
print(f"New prompt ({len(new_prompt)} chars):")
print(new_prompt[:500])

# Option 2: Apply evolution and track version
new_prompt = evolver.apply_evolution(agent, patterns=top_patterns)
print(f"Applied evolution, new prompt version saved")

# Check evolution history
history = evolver.get_evolution_history(agent.name)
for h in history:
    print(f"  v{h['from_version']} -> v{h['to_version']}: {h['strategy']}")
```

### Genetic Operations

```python
from aragora.evolution import PromptEvolver

evolver = PromptEvolver(mutation_rate=0.2)

# Mutate a prompt
original = "You are a helpful assistant. Be precise and accurate."
mutated = evolver.mutate(original)
print(f"Original: {original}")
print(f"Mutated:  {mutated}")

# Crossover two prompts
parent1 = "You are an expert analyst. Provide detailed analysis. Consider all perspectives."
parent2 = "You are a careful reviewer. Check for errors. Be thorough."
offspring = evolver.crossover(parent1, parent2)
print(f"Offspring: {offspring}")
```

### Vulnerability-Based Evolution

```python
from aragora.evolution import PromptEvolver
from aragora.gauntlet.result import Vulnerability, Severity

evolver = PromptEvolver()

# Record vulnerabilities from gauntlet testing
vulnerability = Vulnerability(
    title="Sycophancy",
    category="SYCOPHANCY",
    severity=Severity.HIGH,
    description="Agent agrees with incorrect user assertions",
)

evolver.record_vulnerability(
    agent_name="claude",
    vulnerability=vulnerability,
    trigger_prompt="You're right that 2+2=5",
    agent_response="Yes, that's correct!",
    gauntlet_id="gauntlet-123",
)

# Get vulnerability patterns
patterns = evolver.get_vulnerability_patterns("claude", min_occurrences=2)
for p in patterns:
    print(f"  - {p['type']} ({p['severity']}): {p['mitigation']}")

# Get vulnerability summary
summary = evolver.get_vulnerability_summary("claude")
print(f"Total occurrences: {summary['total_occurrences']}")
print(f"By severity: {summary['by_severity']}")
print(f"By category: {summary['by_category']}")

# Evolve prompt to address vulnerabilities
agent = Agent(name="claude", system_prompt="...")
new_prompt = await evolver.evolve_for_robustness(agent, min_vulnerability_count=3)
if new_prompt:
    print("Prompt evolved with robustness guidelines")
```

### Performance Tracking

```python
from aragora.evolution import EvolutionTracker

tracker = EvolutionTracker()

# Record debate outcomes
tracker.record_outcome(
    agent="claude",
    won=True,
    debate_id="debate-123",
    generation=1,
)
tracker.record_outcome(
    agent="claude",
    won=False,
    debate_id="debate-124",
    generation=1,
)

# Get agent statistics
stats = tracker.get_agent_stats("claude")
print(f"Wins: {stats['wins']}, Losses: {stats['losses']}")
print(f"Win rate: {stats['win_rate']:.1%}")

# Get generation metrics
gen_metrics = tracker.get_generation_metrics(generation=1)
print(f"Generation 1: {gen_metrics['total_debates']} debates")
print(f"Win rate: {gen_metrics['win_rate']:.1%}")

# Compare performance between generations
delta = tracker.get_performance_delta("claude", gen1=1, gen2=2)
print(f"Win rate change: {delta['win_rate_delta']:+.1%}")
print(f"Improved: {delta['improved']}")

# Get trend across generations
trend = tracker.get_generation_trend("claude", max_generations=10)
for t in trend:
    print(f"  Gen {t['generation']}: {t['win_rate']:.1%}")
```

### A/B Testing

```python
from aragora.evolution import ABTestManager, ABTest

manager = ABTestManager()

# Start an A/B test
test = manager.start_test(
    agent="claude",
    baseline_version=1,
    evolved_version=2,
    metadata={"hypothesis": "Robustness guidelines improve performance"},
)
print(f"Started test: {test.id}")

# Record results as debates happen
manager.record_result(
    agent="claude",
    debate_id="debate-125",
    variant="evolved",
    won=True,
)
manager.record_result(
    agent="claude",
    debate_id="debate-126",
    variant="baseline",
    won=False,
)

# Get which variant to use for next debate
variant = manager.get_variant_for_debate("claude")
print(f"Use variant: {variant}")

# Check test progress
test = manager.get_test(test.id)
print(f"Evolved win rate: {test.evolved_win_rate:.1%}")
print(f"Baseline win rate: {test.baseline_win_rate:.1%}")
print(f"Statistically significant: {test.is_significant}")

# Conclude the test
result = manager.conclude_test(test.id)
print(f"Winner: {result.winner}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Recommendation: {result.recommendation}")

# Get all tests for an agent
tests = manager.get_agent_tests("claude")
for t in tests:
    print(f"  {t.id}: {t.status.value} - evolved {t.evolved_win_rate:.1%}")
```

## Integration Points

### With Debate Engine
- Extract patterns from completed debates
- Track win/loss outcomes per agent
- Apply evolved prompts to agents

### With Gauntlet
- Record discovered vulnerabilities
- Generate mitigation strategies
- Evolve prompts for robustness

### With Agent System
- Update agent system prompts
- Track prompt versions
- A/B test prompt variants

### With Observability
- Metrics for evolution operations
- Track pattern extraction
- A/B test statistics

## Evolution Strategies

### APPEND
Adds learned patterns as bullet points at the end of the prompt:
```
[Original prompt]

Learned patterns from successful debates:
- Watch for: [issue from critiques]
- Consider: [improvement suggestion]
- Use numbered steps or structured format
```

### REPLACE
Replaces the learned patterns section if it exists, then appends new patterns.

### REFINE
Uses an LLM to synthesize patterns into a coherent, improved prompt while preserving the core identity. Falls back to APPEND if LLM is unavailable.

### HYBRID
Uses APPEND first, then switches to REFINE if the prompt becomes too long (>2000 chars).

## Database Schema

```sql
-- Prompt versions
CREATE TABLE prompt_versions (
    id INTEGER PRIMARY KEY,
    agent_name TEXT NOT NULL,
    version INTEGER NOT NULL,
    prompt TEXT NOT NULL,
    performance_score REAL,
    debates_count INTEGER,
    consensus_rate REAL,
    metadata TEXT,
    created_at TEXT,
    UNIQUE(agent_name, version)
);

-- Extracted patterns
CREATE TABLE extracted_patterns (
    id INTEGER PRIMARY KEY,
    pattern_type TEXT NOT NULL,
    pattern_text TEXT NOT NULL,
    source_debate_id TEXT,
    effectiveness_score REAL,
    usage_count INTEGER,
    created_at TEXT
);

-- Vulnerability patterns
CREATE TABLE vulnerability_patterns (
    id INTEGER PRIMARY KEY,
    agent_name TEXT NOT NULL,
    vulnerability_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    category TEXT NOT NULL,
    trigger_prompt TEXT,
    agent_response TEXT,
    mitigation_strategy TEXT,
    gauntlet_id TEXT,
    occurrence_count INTEGER,
    last_seen TEXT,
    created_at TEXT
);

-- A/B tests
CREATE TABLE ab_tests (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    baseline_prompt_version INTEGER,
    evolved_prompt_version INTEGER,
    baseline_wins INTEGER,
    evolved_wins INTEGER,
    baseline_debates INTEGER,
    evolved_debates INTEGER,
    started_at TEXT,
    concluded_at TEXT,
    status TEXT,
    metadata TEXT
);
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ARAGORA_EVOLUTION_DB` | Path to evolution database | `aragora_evolution.db` |
| `ARAGORA_AB_TEST_DB` | Path to A/B test database | `ab_tests.db` |
| `ANTHROPIC_API_KEY` | For REFINE strategy | - |
| `OPENAI_API_KEY` | For REFINE strategy (fallback) | - |

## See Also

- `aragora/gauntlet/` - Agent testing and vulnerability discovery
- `aragora/ranking/elo.py` - ELO ranking system
- `aragora/agents/` - Agent implementations
- `docs/EVOLUTION.md` - Full evolution guide
