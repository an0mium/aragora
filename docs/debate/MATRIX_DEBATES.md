# Matrix Debates

Matrix debates run the same question across multiple scenarios in parallel, then compare results to identify universal conclusions (true in all scenarios) vs conditional conclusions (true only under specific conditions).

## Overview

Traditional debates answer: "What's the best approach to X?"

Matrix debates answer: "How does the best approach change based on context Y, Z, W?"

For example, instead of debating "Should we use microservices?", a matrix debate explores:
- Scenario A: 5-person startup, MVP phase
- Scenario B: 50-person company, scaling phase
- Scenario C: 500-person enterprise, maintenance phase

## API Endpoints

### POST /api/debates/matrix

Run parallel scenario debates.

**Request Body:**

```json
{
  "task": "What database architecture should we use?",
  "agents": ["anthropic-api", "openai-api"],
  "max_rounds": 3,
  "scenarios": [
    {
      "name": "High-write workload",
      "parameters": {
        "writes_per_second": 100000,
        "reads_per_second": 10000,
        "data_size_tb": 50
      },
      "constraints": ["Must support ACID transactions"],
      "is_baseline": true
    },
    {
      "name": "Read-heavy analytics",
      "parameters": {
        "writes_per_second": 1000,
        "reads_per_second": 500000,
        "data_size_tb": 500
      },
      "constraints": ["Sub-second query latency required"]
    },
    {
      "name": "Global distribution",
      "parameters": {
        "regions": 12,
        "consistency_requirement": "eventual",
        "latency_budget_ms": 50
      },
      "constraints": ["Must comply with GDPR data residency"]
    }
  ]
}
```

**Parameters:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `task` | string | required | Base debate topic |
| `agents` | string[] | `["anthropic-api", "openai-api"]` | Agent names |
| `max_rounds` | int | 3 | Rounds per scenario |
| `scenarios` | array | required | Scenario configurations |

**Scenario Configuration:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Human-readable scenario name |
| `parameters` | object | no | Key-value pairs for this scenario |
| `constraints` | string[] | no | Additional constraints to enforce |
| `is_baseline` | bool | no | Mark as baseline for comparison |

**Response:**

```json
{
  "matrix_id": "550e8400-e29b-41d4-a716-446655440000",
  "task": "What database architecture should we use?",
  "scenario_count": 3,
  "results": [
    {
      "scenario_name": "High-write workload",
      "parameters": {"writes_per_second": 100000, ...},
      "is_baseline": true,
      "winner": "anthropic-api",
      "final_answer": "For write-heavy workloads, use Apache Cassandra with...",
      "confidence": 0.85,
      "consensus_reached": true,
      "rounds_used": 3
    },
    {
      "scenario_name": "Read-heavy analytics",
      "parameters": {"reads_per_second": 500000, ...},
      "is_baseline": false,
      "winner": "openai-api",
      "final_answer": "For analytical queries, use ClickHouse with...",
      "confidence": 0.92,
      "consensus_reached": true,
      "rounds_used": 2
    },
    {
      "scenario_name": "Global distribution",
      "parameters": {"regions": 12, ...},
      "is_baseline": false,
      "winner": null,
      "final_answer": "For global distribution, CockroachDB or Spanner...",
      "confidence": 0.78,
      "consensus_reached": false,
      "rounds_used": 3
    }
  ],
  "universal_conclusions": [
    "All scenarios require data replication for durability",
    "Schema design matters more than database choice for most use cases"
  ],
  "conditional_conclusions": [
    {
      "condition": "When High-write workload",
      "parameters": {"writes_per_second": 100000},
      "conclusion": "Use Apache Cassandra with...",
      "confidence": 0.85
    },
    {
      "condition": "When Read-heavy analytics",
      "parameters": {"reads_per_second": 500000},
      "conclusion": "Use ClickHouse with...",
      "confidence": 0.92
    }
  ],
  "comparison_matrix": {
    "scenarios": ["High-write workload", "Read-heavy analytics", "Global distribution"],
    "consensus_rate": 0.67,
    "avg_confidence": 0.85,
    "avg_rounds": 2.67
  }
}
```

### GET /api/debates/matrix/{id}

Retrieve a matrix debate by ID.

### GET /api/debates/matrix/{id}/scenarios

Get detailed results for all scenarios.

### GET /api/debates/matrix/{id}/conclusions

Get universal and conditional conclusions.

## Scenario Design

### Parameter Types

Parameters can represent any contextual factor:

**Quantitative:**
```json
{
  "users": 1000000,
  "requests_per_second": 50000,
  "budget_usd": 100000
}
```

**Categorical:**
```json
{
  "environment": "production",
  "region": "eu-west",
  "compliance": "hipaa"
}
```

**Boolean:**
```json
{
  "real_time_required": true,
  "multi_tenant": false
}
```

### Constraint Syntax

Constraints are natural language strings that agents must respect:

```json
{
  "constraints": [
    "Must support horizontal scaling",
    "Cannot use AWS services (vendor lock-in)",
    "Team has no Kubernetes experience",
    "Budget must stay under $10k/month"
  ]
}
```

### Baseline Scenarios

Mark one scenario as `is_baseline: true` to enable comparative analysis. Other scenarios are compared against the baseline to identify what changes across contexts.

## Conclusion Extraction

### Universal Conclusions

Universal conclusions hold true across ALL scenarios. The system identifies these by:

1. Looking for claims that appear in all scenario results
2. Checking semantic similarity of conclusions across scenarios
3. Extracting shared reasoning patterns

### Conditional Conclusions

Conditional conclusions are specific to certain scenarios. Format:

```
"When [scenario conditions], then [conclusion]"
```

Examples:
- "When write throughput exceeds 50k/s, prefer Cassandra over PostgreSQL"
- "When multi-region is required, eventual consistency is acceptable"
- "When team size < 10, prefer managed services over self-hosted"

## Use Cases

### Technology Selection

Compare solutions across different scale/complexity scenarios:

```json
{
  "task": "Which message queue should we use?",
  "scenarios": [
    {"name": "Startup MVP", "parameters": {"messages_per_day": 10000}},
    {"name": "Growth stage", "parameters": {"messages_per_day": 10000000}},
    {"name": "Enterprise", "parameters": {"messages_per_day": 1000000000}}
  ]
}
```

### Policy Analysis

Explore how policies affect different stakeholder groups:

```json
{
  "task": "Should we implement a 4-day work week?",
  "scenarios": [
    {"name": "Engineering team", "parameters": {"role": "software_engineer"}},
    {"name": "Sales team", "parameters": {"role": "sales"}},
    {"name": "Customer support", "parameters": {"role": "support"}}
  ]
}
```

### Risk Assessment

Evaluate decisions under different failure conditions:

```json
{
  "task": "How should we handle database failover?",
  "scenarios": [
    {"name": "Single region outage", "parameters": {"failure_scope": "region"}},
    {"name": "Full provider outage", "parameters": {"failure_scope": "provider"}},
    {"name": "Data corruption", "parameters": {"failure_scope": "data"}}
  ]
}
```

## Example: Full Matrix Debate

```python
import httpx

async def run_matrix_debate():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.aragora.ai/api/debates/matrix",
            json={
                "task": "How should we handle authentication?",
                "agents": ["anthropic-api", "openai-api", "gemini"],
                "max_rounds": 3,
                "scenarios": [
                    {
                        "name": "Consumer B2C app",
                        "parameters": {"users": 1000000, "session_duration": "long"},
                        "constraints": ["Must support social login"],
                        "is_baseline": True
                    },
                    {
                        "name": "Enterprise B2B",
                        "parameters": {"users": 5000, "session_duration": "short"},
                        "constraints": ["Must support SAML/SSO", "Audit logging required"]
                    },
                    {
                        "name": "API-first service",
                        "parameters": {"api_calls_per_day": 10000000},
                        "constraints": ["No browser involved", "Rate limiting required"]
                    }
                ]
            },
            headers={"Authorization": "Bearer YOUR_TOKEN"}
        )
        result = response.json()

        print("=== Universal Conclusions ===")
        for conclusion in result.get("universal_conclusions", []):
            print(f"  - {conclusion}")

        print("\n=== Conditional Conclusions ===")
        for cond in result.get("conditional_conclusions", []):
            print(f"  - {cond['condition']}: {cond['conclusion']}")

        print(f"\n=== Comparison Matrix ===")
        matrix = result["comparison_matrix"]
        print(f"  Consensus rate: {matrix['consensus_rate']:.0%}")
        print(f"  Average confidence: {matrix['avg_confidence']:.0%}")
```

## Performance Considerations

Matrix debates run scenarios in parallel using `asyncio.gather()`. Consider:

- **API rate limits**: Each scenario makes multiple LLM calls
- **Cost**: Total cost = scenarios * rounds * agents * (input + output tokens)
- **Timeout**: Set appropriate timeouts for parallel execution

```python
# Example: Running 5 scenarios with 3 agents and 3 rounds
# = 5 * 3 * 3 = 45 LLM calls (plus critiques and votes)
```

## Integration with Other Features

- **Trickster**: Each scenario is monitored independently for hollow consensus
- **ELO Rankings**: Scenario results contribute to agent rankings
- **Memory**: Scenario patterns are stored for future reference
- **A/B Testing**: Matrix debates can be used to compare prompt versions across scenarios
