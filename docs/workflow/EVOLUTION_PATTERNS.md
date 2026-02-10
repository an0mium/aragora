# Evolution Patterns

Aragora's pattern extraction system analyzes winning debates to identify effective argument patterns, evidence usage, and rhetorical techniques that can improve agent prompts over time.

## Overview

The pattern extraction module:
- Analyzes debate transcripts for winning strategies
- Identifies argument structures, evidence patterns, and persuasion techniques
- Aggregates patterns across debates for statistical significance
- Provides recommendations for prompt evolution

## Pattern Types

| Type | Description | Example |
|------|-------------|---------|
| `argument` | Logical argument structures | "If A, then B. A is true. Therefore B" |
| `structure` | Organizational patterns | "First... Second... Finally..." |
| `evidence` | Evidence usage patterns | "According to research from MIT..." |
| `persuasion` | Rhetorical techniques | "Consider the alternative..." |
| `rebuttal` | Counter-argument patterns | "While that's true, we must also consider..." |

## API Endpoints

### Get Top Patterns

```http
GET /api/evolution/patterns?limit=10&type=argument
```

**Response:**
```json
{
  "patterns": [
    {
      "pattern_type": "argument",
      "description": "Acknowledges opponent's point before presenting counter-evidence",
      "frequency": 45,
      "effectiveness": 0.82,
      "examples": [
        "While that perspective has merit, the data suggests...",
        "You raise a valid point, however research indicates..."
      ],
      "agent": "anthropic-api"
    },
    {
      "pattern_type": "argument",
      "description": "Uses concrete examples to illustrate abstract concepts",
      "frequency": 38,
      "effectiveness": 0.78,
      "examples": [
        "To illustrate, consider the case of...",
        "A concrete example would be..."
      ]
    }
  ],
  "total": 156,
  "filter": "argument"
}
```

### Get Evolution Summary

```http
GET /api/evolution/summary
```

**Response:**
```json
{
  "total_prompt_versions": 234,
  "total_agents": 12,
  "total_patterns": 1567,
  "pattern_distribution": {
    "argument": 456,
    "structure": 312,
    "evidence": 389,
    "persuasion": 278,
    "rebuttal": 132
  },
  "top_agents": [
    {
      "agent": "anthropic-api",
      "best_score": 0.89,
      "latest_version": 15
    },
    {
      "agent": "openai-api",
      "best_score": 0.85,
      "latest_version": 12
    }
  ],
  "recent_activity": [
    {
      "agent": "gemini",
      "strategy": "gradient_descent",
      "created_at": "2026-01-10T10:00:00Z"
    }
  ]
}
```

### Get Agent Evolution History

```http
GET /api/evolution/{agent}/history?limit=10
```

**Response:**
```json
{
  "agent": "anthropic-api",
  "history": [
    {
      "version": 15,
      "performance_score": 0.89,
      "debates_count": 45,
      "consensus_rate": 0.72,
      "strategy": "pattern_incorporation",
      "patterns_added": ["evidence_stacking", "concession_rebuttal"],
      "created_at": "2026-01-10T08:00:00Z"
    },
    {
      "version": 14,
      "performance_score": 0.84,
      "debates_count": 38,
      "consensus_rate": 0.68,
      "strategy": "a_b_test_graduation",
      "created_at": "2026-01-08T12:00:00Z"
    }
  ],
  "count": 15
}
```

## Python API

### Extract Patterns from Debate

```python
from aragora.evolution.pattern_extractor import PatternExtractor, extract_patterns

# Initialize extractor
extractor = PatternExtractor()

# Extract patterns from a debate outcome
debate_outcome = {
    "winner": "anthropic-api",
    "transcript": [
        {"agent": "anthropic-api", "content": "First, let's examine..."},
        {"agent": "openai-api", "content": "I disagree because..."},
        # ... more messages
    ],
    "votes": [
        {"judge": "judge1", "winner": "anthropic-api", "reasoning": "..."}
    ]
}

patterns = extractor.extract(debate_outcome)

for pattern in patterns:
    print(f"Type: {pattern.pattern_type}")
    print(f"Description: {pattern.description}")
    print(f"Effectiveness: {pattern.effectiveness:.2f}")
```

### Quick Pattern Extraction

```python
from aragora.evolution.pattern_extractor import extract_patterns

# Simplified extraction from debate dict
result = extract_patterns(debate_outcome)

print(f"Found {len(result['patterns'])} patterns")
print(f"Found {len(result['strategies'])} strategies")
```

### Pattern Aggregation

```python
from aragora.evolution.evolver import PromptEvolver

evolver = PromptEvolver(db_path="evolution.db")

# Get aggregated patterns across all debates
top_patterns = evolver.get_top_patterns(
    pattern_type="argument",
    limit=20,
    min_frequency=5,
)

for pattern in top_patterns:
    print(f"{pattern.description}: {pattern.frequency} occurrences")
```

## Pattern Markers

The extractor uses regex patterns to identify different pattern types:

### Evidence Markers

```python
EVIDENCE_MARKERS = [
    r"according to",
    r"research shows",
    r"studies indicate",
    r"evidence suggests",
    r"data from",
    r"statistics show",
    r"for example",
    r"as demonstrated by",
]
```

### Structure Markers

```python
STRUCTURE_MARKERS = [
    r"first(?:ly)?[\s,]",
    r"second(?:ly)?[\s,]",
    r"third(?:ly)?[\s,]",
    r"finally[\s,]",
    r"in conclusion",
    r"to summarize",
    r"on one hand.*on the other",
    r"while.*however",
]
```

### Persuasion Markers

```python
PERSUASION_MARKERS = [
    r"it's important to note",
    r"we must consider",
    r"the key point is",
    r"consider the alternative",
    r"what if we",
    r"imagine if",
]
```

## Evolution Strategies

Patterns are used in different evolution strategies:

| Strategy | Description | When Used |
|----------|-------------|-----------|
| `pattern_incorporation` | Add successful patterns to prompt | After pattern reaches threshold frequency |
| `gradient_descent` | Optimize prompt parameters | Continuous improvement |
| `a_b_test_graduation` | Promote A/B test winner | After statistical significance |
| `crossover` | Combine patterns from multiple agents | When agents have complementary strengths |
| `mutation` | Random variation of patterns | Exploration of pattern space |

### Example: Pattern Incorporation

```python
# Get patterns that should be incorporated
patterns = evolver.get_incorporation_candidates(
    agent_name="anthropic-api",
    min_effectiveness=0.7,
    min_frequency=10,
)

for pattern in patterns:
    # Generate prompt modification
    modification = evolver.generate_pattern_instruction(pattern)
    print(f"Add to prompt: {modification}")
```

## Effectiveness Scoring

Pattern effectiveness is calculated based on:

1. **Win Rate**: Debates won when pattern was used
2. **Judge Preference**: How often judges cite the pattern positively
3. **Critique Resistance**: How well arguments using the pattern withstand critique
4. **Consensus Contribution**: Whether pattern usage led to consensus

```python
effectiveness = (
    win_rate * 0.4 +
    judge_preference * 0.3 +
    critique_resistance * 0.2 +
    consensus_contribution * 0.1
)
```

## Database Schema

```sql
-- Extracted patterns
CREATE TABLE extracted_patterns (
    id TEXT PRIMARY KEY,
    pattern_type TEXT NOT NULL,
    description TEXT NOT NULL,
    agent TEXT,
    debate_id TEXT,
    effectiveness REAL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Pattern aggregates
CREATE TABLE pattern_aggregates (
    pattern_hash TEXT PRIMARY KEY,
    pattern_type TEXT NOT NULL,
    description TEXT NOT NULL,
    frequency INTEGER DEFAULT 1,
    avg_effectiveness REAL DEFAULT 0,
    examples TEXT,  -- JSON array
    last_seen TIMESTAMP
);

-- Prompt versions
CREATE TABLE prompt_versions (
    id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    version INTEGER NOT NULL,
    prompt TEXT NOT NULL,
    performance_score REAL DEFAULT 0,
    debates_count INTEGER DEFAULT 0,
    consensus_rate REAL DEFAULT 0,
    patterns_incorporated TEXT,  -- JSON array
    metadata TEXT,  -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (agent_name, version)
);

-- Evolution history
CREATE TABLE evolution_history (
    id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    strategy TEXT NOT NULL,
    from_version INTEGER,
    to_version INTEGER,
    patterns_used TEXT,  -- JSON array
    performance_delta REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Best Practices

1. **Minimum Sample Size**: Wait for at least 5 occurrences before using a pattern

2. **Effectiveness Threshold**: Only incorporate patterns with effectiveness > 0.6

3. **Diversity**: Balance pattern types to avoid over-optimization

4. **Regular Review**: Periodically review and prune ineffective patterns

5. **Agent Specificity**: Some patterns work better for certain agents

## Troubleshooting

### Pattern Not Detected

```python
# Check if content matches markers
extractor = PatternExtractor()
content = "Your debate message here..."

# Get matching markers
matches = extractor._find_marker_matches(content, PatternExtractor.EVIDENCE_MARKERS)
print(f"Evidence markers found: {matches}")
```

### Low Effectiveness Scores

```python
# Analyze pattern performance
stats = evolver.get_pattern_stats("pattern_hash_123")

print(f"Win rate when used: {stats['win_rate']:.2%}")
print(f"Usage count: {stats['usage_count']}")
print(f"Recent effectiveness: {stats['recent_effectiveness']:.2f}")
```

## See Also

- [A/B Testing](A_B_TESTING.md) - Testing prompt variations
- [API Reference](API_REFERENCE.md) - Full endpoint documentation
- [Features](FEATURES.md) - Feature status overview
