# Graph Debates

Graph debates extend the standard linear debate format by allowing branching when agents fundamentally disagree. Instead of forcing convergence, the system explores divergent perspectives in parallel branches, then synthesizes findings.

## Overview

In a traditional debate, agents take turns in a linear sequence of rounds. Graph debates introduce a tree structure where:

- When high disagreement is detected, the debate **branches** into parallel tracks
- Each branch explores a different perspective independently
- Branches can be **merged** back using synthesis strategies
- The result is a directed acyclic graph (DAG) of debate nodes

## API Endpoints

### POST /api/debates/graph

Run a graph-structured debate with automatic branching.

**Request Body:**

```json
{
  "task": "Should we prioritize renewable energy or nuclear power?",
  "agents": ["anthropic-api", "openai-api", "gemini"],
  "max_rounds": 5,
  "branch_policy": {
    "min_disagreement": 0.7,
    "max_branches": 3,
    "auto_merge": true,
    "merge_strategy": "synthesis"
  }
}
```

**Parameters:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `task` | string | required | The debate topic or question |
| `agents` | string[] | `["anthropic-api", "openai-api"]` | Agent names to participate |
| `max_rounds` | int | 5 | Maximum rounds per branch |
| `branch_policy` | object | see below | Branching configuration |

**Branch Policy Options:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `min_disagreement` | float | 0.7 | Minimum semantic dissimilarity to trigger branch (0-1) |
| `max_branches` | int | 3 | Maximum parallel branches allowed |
| `auto_merge` | bool | true | Automatically merge branches at end |
| `merge_strategy` | string | "synthesis" | How to merge: "synthesis", "vote", "best_evidence" |

**Response:**

```json
{
  "debate_id": "550e8400-e29b-41d4-a716-446655440000",
  "task": "Should we prioritize renewable energy or nuclear power?",
  "graph": {
    "root_node_id": "node-001",
    "nodes": [
      {
        "id": "node-001",
        "agent": "anthropic-api",
        "round": 1,
        "branch_id": "main",
        "content": "...",
        "children": ["node-002", "node-003"]
      }
    ]
  },
  "branches": [
    {
      "id": "main",
      "parent_id": null,
      "reason": "root",
      "started_at_round": 1
    },
    {
      "id": "renewable-focus",
      "parent_id": "main",
      "reason": "disagreement_on_scalability",
      "started_at_round": 2
    }
  ],
  "merge_results": [
    {
      "merged_branches": ["main", "renewable-focus"],
      "strategy": "synthesis",
      "synthesis": "Both perspectives highlight valid concerns..."
    }
  ],
  "node_count": 15,
  "branch_count": 2
}
```

### GET /api/debates/graph/{id}

Retrieve a previously run graph debate by ID.

### GET /api/debates/graph/{id}/branches

Get all branches for a graph debate.

**Response:**

```json
{
  "debate_id": "550e8400-e29b-41d4-a716-446655440000",
  "branches": [
    {
      "id": "main",
      "parent_id": null,
      "node_count": 8,
      "final_position": "Nuclear offers baseload reliability..."
    },
    {
      "id": "renewable-focus",
      "parent_id": "main",
      "node_count": 7,
      "final_position": "Renewables with storage can match baseload..."
    }
  ]
}
```

### GET /api/debates/graph/{id}/nodes

Get all nodes in the debate graph.

## Branching Strategies

### Automatic Branching

The orchestrator monitors semantic similarity between agent responses. When disagreement exceeds `min_disagreement`, it creates a new branch:

```
Round 1: All agents respond (main branch)
Round 2: High disagreement detected (0.85)
         → Branch "nuclear-advocate" created
         → Branch "renewable-advocate" created
Round 3-5: Each branch continues independently
Round 6: Branches merged with synthesis
```

### Branch Reasons

Branches are created with a recorded reason:

- `disagreement_on_scalability` - Agents disagree on scaling approach
- `opposing_evidence` - Agents cite contradicting evidence
- `different_frameworks` - Agents use incompatible analytical frameworks
- `value_conflict` - Agents prioritize different values

## Merge Strategies

### Synthesis (default)

A synthesis agent creates a unified position that acknowledges both branches:

```
"Both branches make valid points. The nuclear branch emphasizes
baseload reliability, while the renewable branch highlights
declining costs and distributed resilience. A hybrid approach
could leverage nuclear for baseload while scaling renewables..."
```

### Vote

Agents vote on which branch has the stronger argument. Winner becomes the final position.

### Best Evidence

The branch with the highest evidence quality score (from the trickster system) is selected.

## Use Cases

### Exploring Trade-offs

Graph debates excel when there are genuine trade-offs without a clear "right" answer:

- "Microservices vs monolith for a startup?"
- "Static typing vs dynamic typing for rapid prototyping?"
- "Privacy vs convenience in user data collection?"

### Multi-stakeholder Analysis

When different stakeholders have legitimately different priorities:

- "How should we allocate the engineering budget?"
- "What features should we prioritize for the next release?"

### Adversarial Red-teaming

Intentionally fork branches to explore attack vectors:

- "How could this API be abused?"
- "What are the failure modes of this architecture?"

## Example: Full Graph Debate

```python
import httpx

async def run_graph_debate():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.aragora.ai/api/debates/graph",
            json={
                "task": "Should AI models be open-source or closed?",
                "agents": ["anthropic-api", "openai-api", "gemini"],
                "max_rounds": 4,
                "branch_policy": {
                    "min_disagreement": 0.6,
                    "max_branches": 2,
                    "merge_strategy": "synthesis"
                }
            },
            headers={"Authorization": "Bearer YOUR_TOKEN"}
        )
        result = response.json()

        print(f"Debate ID: {result['debate_id']}")
        print(f"Branches created: {result['branch_count']}")
        print(f"Total nodes: {result['node_count']}")

        for merge in result.get("merge_results", []):
            print(f"\nMerge synthesis:\n{merge['synthesis']}")
```

## Integration with Other Features

- **Trickster**: Each branch is monitored for hollow consensus independently
- **Genesis**: Evolved agents can be spawned for specific branches
- **ELO Rankings**: Branch winners contribute to agent rankings
- **Memory**: Branch explorations are stored for future reference
