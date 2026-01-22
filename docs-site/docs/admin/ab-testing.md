---
title: A/B Testing for Prompt Evolution
description: A/B Testing for Prompt Evolution
---

# A/B Testing for Prompt Evolution

Aragora's A/B testing system allows controlled experimentation with prompt variations to determine which prompts perform better in debates.

## Overview

The A/B testing module enables:
- Testing two prompt variants simultaneously
- Recording debate outcomes per variant
- Statistical analysis to determine winners
- Automatic graduation of winning prompts

## Quick Start

### Prerequisites

```bash
# A/B testing is included in the core package
# No additional dependencies required
```

### Create an A/B Test

```python
from aragora.evolution.ab_testing import ABTestManager, ABTest

# Initialize manager
manager = ABTestManager(db_path="ab_tests.db")

# Create a test comparing two prompts
test = manager.create_test(
    agent_name="anthropic-api",
    control_prompt="You are a thoughtful debate participant...",
    variant_prompt="You are an analytical debater who prioritizes evidence...",
    description="Testing evidence-focused prompt",
    target_sample_size=50,  # Debates before conclusion
)

print(f"Test ID: {test.id}")
print(f"Status: {test.status}")  # RUNNING
```

## API Endpoints

### List All Tests

```http
GET /api/evolution/ab-tests?limit=10&status=running
```

**Response:**
```json
{
  "tests": [
    {
      "id": "test_abc123",
      "agent_name": "anthropic-api",
      "status": "running",
      "control_wins": 12,
      "variant_wins": 18,
      "sample_size": 30,
      "target_sample_size": 50,
      "created_at": "2026-01-10T10:00:00Z"
    }
  ],
  "total": 5
}
```

### Get Active Test for Agent

```http
GET /api/evolution/ab-tests/\{agent_name\}/active
```

**Response:**
```json
{
  "test": {
    "id": "test_abc123",
    "agent_name": "anthropic-api",
    "control_prompt": "You are a thoughtful...",
    "variant_prompt": "You are an analytical...",
    "control_wins": 12,
    "variant_wins": 18,
    "statistical_significance": 0.82
  }
}
```

### Start New A/B Test

```http
POST /api/evolution/ab-tests
Content-Type: application/json

{
  "agent_name": "anthropic-api",
  "control_prompt": "You are a thoughtful debate participant...",
  "variant_prompt": "You are an analytical debater...",
  "description": "Testing evidence-focused prompt",
  "target_sample_size": 50
}
```

**Response:**
```json
{
  "test": {
    "id": "test_xyz789",
    "status": "running",
    "created_at": "2026-01-10T12:00:00Z"
  }
}
```

### Record Debate Result

```http
POST /api/evolution/ab-tests/\{test_id\}/record
Content-Type: application/json

{
  "variant": "control",  // or "variant"
  "won": true,
  "debate_id": "debate_123",
  "metrics": {
    "confidence": 0.85,
    "critique_acceptance": 0.7
  }
}
```

### Conclude Test

```http
POST /api/evolution/ab-tests/\{test_id\}/conclude
Content-Type: application/json

{
  "force": false  // Set true to conclude before target sample size
}
```

**Response:**
```json
{
  "test": {
    "id": "test_abc123",
    "status": "concluded",
    "winner": "variant",
    "control_wins": 15,
    "variant_wins": 35,
    "p_value": 0.003,
    "concluded_at": "2026-01-15T10:00:00Z"
  }
}
```

### Cancel Test

```http
DELETE /api/evolution/ab-tests/\{test_id\}
```

## Workflow Example

### 1. Hypothesis Formation

Identify a potential improvement to test:
- New argumentation style
- Different evidence presentation
- Modified confidence calibration

### 2. Create Test

```python
test = manager.create_test(
    agent_name="anthropic-api",
    control_prompt=current_prompt,  # Existing prompt
    variant_prompt=new_prompt,       # Proposed improvement
    description="Testing structured argumentation",
    target_sample_size=100,
)
```

### 3. Run Debates

During normal debate operation, the system randomly assigns prompts:

```python
# Get prompt for next debate
active_test = manager.get_active_test(agent_name)
if active_test:
    # 50/50 random assignment
    variant = random.choice(["control", "variant"])
    prompt = active_test.get_prompt(variant)
else:
    prompt = default_prompt
```

### 4. Record Results

After each debate:

```python
# Record the outcome
manager.record_result(
    test_id=test.id,
    variant=variant,  # "control" or "variant"
    won=debate_result.won,
    metrics={
        "confidence": debate_result.confidence,
        "critique_acceptance": debate_result.critique_rate,
    }
)
```

### 5. Analyze Results

```python
# Get current statistics
stats = manager.get_test_statistics(test.id)

print(f"Control win rate: {stats['control_win_rate']:.1%}")
print(f"Variant win rate: {stats['variant_win_rate']:.1%}")
print(f"P-value: {stats['p_value']:.4f}")
print(f"Significant: {stats['is_significant']}")
```

### 6. Conclude and Graduate

```python
# Conclude when target reached
result = manager.conclude_test(test.id)

if result.winner == "variant":
    # Graduate winning prompt
    evolver.update_prompt(
        agent_name=test.agent_name,
        new_prompt=test.variant_prompt,
        reason=f"Won A/B test {test.id} with p-value {result.p_value}",
    )
```

## Test Status Values

| Status | Description |
|--------|-------------|
| `RUNNING` | Test is actively collecting data |
| `CONCLUDED` | Test completed with a winner |
| `CANCELLED` | Test was manually cancelled |
| `INCONCLUSIVE` | No significant difference found |

## Statistical Methods

### Significance Testing

A/B tests use the binomial test to determine significance:
- Default significance threshold: p < 0.05
- Minimum sample size before conclusion: 20 debates per variant

### Early Stopping

Tests can be concluded early if:
- One variant has overwhelming advantage (p < 0.001)
- Both variants perform identically (p > 0.5 with 50+ samples)

## Best Practices

1. **One Test at a Time**: Only run one A/B test per agent to avoid confounding

2. **Sufficient Sample Size**: Aim for at least 50 debates per variant for reliable results

3. **Clear Hypotheses**: Define what you expect the variant to improve before starting

4. **Minimal Changes**: Test one change at a time for clear attribution

5. **Document Results**: Record learnings even from unsuccessful tests

## Database Schema

```sql
CREATE TABLE ab_tests (
    id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    control_prompt TEXT NOT NULL,
    variant_prompt TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'running',
    control_wins INTEGER DEFAULT 0,
    variant_wins INTEGER DEFAULT 0,
    target_sample_size INTEGER DEFAULT 50,
    p_value REAL,
    winner TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    concluded_at TIMESTAMP
);

CREATE TABLE ab_test_results (
    id TEXT PRIMARY KEY,
    test_id TEXT NOT NULL,
    variant TEXT NOT NULL,
    won BOOLEAN NOT NULL,
    debate_id TEXT,
    metrics TEXT,  -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (test_id) REFERENCES ab_tests(id)
);
```

## Frontend Components

### ABTestResultsPanel

The `ABTestResultsPanel` component provides a comprehensive UI for viewing A/B test results:

**Features:**
- Summary statistics (active tests, concluded, significance rate)
- Test list with quick status indicators
- Detailed comparison view with win rate visualization
- Statistical analysis display
- Recommendations based on results

**Usage:**
```tsx
import { ABTestResultsPanel } from '@/components/ABTestResultsPanel';

<ABTestResultsPanel
  apiBase="/api"           // API base URL
  showListView=\{true\}      // Show test list sidebar
  onTestSelect={(test) => console.log(test)}
/>
```

**Integration with EvolutionPanel:**

The A/B test results are also accessible via the Evolution Panel:
1. Navigate to `/evolution` in the dashboard
2. Select the "A/B TESTS" tab
3. Click any test to view detailed results

### Visual Elements

- **Win Rate Bar**: Shows baseline vs evolved performance visually
- **Significance Indicator**: Highlights statistically significant results
- **Recommendation Badge**: Shows adoption/rejection recommendation for concluded tests

## See Also

- [Evolution Patterns](../advanced/evolution-patterns) - Pattern extraction from winning prompts
- [Genesis](../advanced/genesis) - Full evolutionary system documentation
- [API Reference](../api/reference) - Full endpoint documentation
- [Features](../guides/features) - Feature status overview
