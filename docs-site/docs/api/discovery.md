---
title: Aragora API Discovery Guide
description: Aragora API Discovery Guide
---

# Aragora API Discovery Guide

> **Note:** For the complete API documentation, see **[API_REFERENCE.md](./API_REFERENCE.md)**. This guide focuses on powerful but underutilized endpoints that aren't yet integrated into the frontend.

This guide documents powerful but underutilized API endpoints. These endpoints provide access to sophisticated capabilities that are fully implemented but not yet integrated into the frontend.

## Quick Reference

| Endpoint | Category | Description |
|----------|----------|-------------|
| `GET /api/features/discover` | Discovery | List all available API endpoints |
| `POST /api/verification/formal-verify` | Verification | Verify claims using Z3 SMT solver |
| `GET /api/verify/history` | Verification | List verification history with pagination |
| `GET /api/verify/history/\{id\}` | Verification | Get specific verification entry |
| `GET /api/verify/history/\{id\}/tree` | Verification | Get proof tree visualization |
| `GET /api/belief-network/\{id\}/cruxes` | Analysis | Get key claims affecting debate outcome |
| `GET /api/belief-network/\{id\}/load-bearing-claims` | Analysis | Get high-centrality claims |
| `GET /api/belief-network/\{id\}/graph` | Analysis | Get network as force-directed graph |
| `GET /api/belief-network/\{id\}/export` | Analysis | Export network (JSON/GraphML/CSV) |
| `GET /api/evolution/patterns` | Evolution | Get prompt evolution patterns |
| `GET /api/evolution/\{agent\}/history` | Evolution | Get agent prompt history |
| `POST /api/gauntlet/run` | Testing | Start adversarial stress test |
| `GET /api/gauntlet/\{id\}/heatmap` | Testing | Get risk heatmap visualization |
| `GET /api/gauntlet/\{id\}/receipt` | Testing | Get decision audit receipt |
| `GET /api/provenance/\{id\}/claims/\{claim_id\}/support` | Provenance | Get claim verification chain |

---

## 1. Feature Discovery API

### GET /api/features/discover

Discover all available API endpoints with their categories, stability, and integration status.

**Response:**
```json
{
  "total_endpoints": 284,
  "by_category": {
    "core": { "count": 45, "endpoints": [...] },
    "analysis": { "count": 16, "endpoints": [...] },
    "verification": { "count": 2, "endpoints": [...] }
  },
  "by_stability": {
    "STABLE": 267,
    "EXPERIMENTAL": 12,
    "PREVIEW": 5
  },
  "frontend_integrated": 7,
  "hidden_features": 277
}
```

**Use Case:** Build custom integrations, audit API surface, discover hidden capabilities.

---

## 2. Formal Verification API

### POST /api/verification/formal-verify

Verify logical claims using the Z3 SMT solver. Supports propositional logic, arithmetic constraints, and quantified formulas.

**Request:**
```json
{
  "claim": "If agent A and agent B agree, then consensus is reached",
  "claim_type": "logical",
  "context": "Debate consensus verification"
}
```

**Response:**
```json
{
  "verified": true,
  "proof_type": "z3_sat",
  "confidence": 0.95,
  "reasoning": "Formula is satisfiable under given constraints",
  "proof_steps": [
    "Parsed: (A ∧ B) → C",
    "Negation: ¬((A ∧ B) → C) ≡ A ∧ B ∧ ¬C",
    "Result: UNSAT (original claim is valid)"
  ]
}
```

**Claim Types:**
- `logical` - Propositional/predicate logic
- `arithmetic` - Numeric constraints
- `assertion` - Simple true/false claims
- `temporal` - Sequence/ordering claims

**Rate Limit:** 10 requests/minute (burst: 3)

### GET /api/verification/status

Check availability of verification backends.

**Response:**
```json
{
  "available": true,
  "backends": [
    { "name": "z3", "version": "4.12.4", "status": "ready" },
    { "name": "lean", "version": "4.3.0", "status": "available" }
  ]
}
```

### GET /api/verify/history

List verification history with pagination and filtering.

**Parameters:**
- `limit` (query, default: 20) - Number of entries per page (1-100)
- `offset` (query, default: 0) - Skip first N entries
- `status` (query) - Filter by status: `proof_found`, `translation_failed`, `no_proof`

**Response:**
```json
{
  "entries": [
    {
      "id": "a1b2c3d4e5f67890",
      "claim": "1 + 1 = 2",
      "claim_type": "MATHEMATICAL",
      "result": {
        "status": "proof_found",
        "is_verified": true
      },
      "timestamp": 1705161600.0,
      "timestamp_iso": "2026-01-13T12:00:00Z",
      "has_proof_tree": true
    }
  ],
  "total": 42,
  "limit": 20,
  "offset": 0
}
```

### GET /api/verify/history/\{id\}

Get a specific verification history entry by ID.

**Response:**
```json
{
  "id": "a1b2c3d4e5f67890",
  "claim": "1 + 1 = 2",
  "claim_type": "MATHEMATICAL",
  "context": "arithmetic verification",
  "result": {
    "status": "proof_found",
    "is_verified": true,
    "formal_statement": "theorem t : 1 + 1 = 2 := rfl",
    "language": "lean4",
    "proof_hash": "sha256:abc123..."
  },
  "timestamp": 1705161600.0,
  "timestamp_iso": "2026-01-13T12:00:00Z",
  "has_proof_tree": true
}
```

### GET /api/verify/history/\{id\}/tree

Get the proof tree visualization for a verification entry.

**Response:**
```json
{
  "entry_id": "a1b2c3d4e5f67890",
  "nodes": [
    {
      "id": "root",
      "type": "claim",
      "content": "1 + 1 = 2",
      "children": ["translation"]
    },
    {
      "id": "translation",
      "type": "translation",
      "content": "theorem t : 1 + 1 = 2 := rfl",
      "language": "lean4",
      "children": ["verification"]
    },
    {
      "id": "verification",
      "type": "verification",
      "content": "Proof verified by lean4",
      "is_verified": true,
      "proof_hash": "sha256:abc123...",
      "children": []
    }
  ]
}
```

**Node Types:**
- `claim` - The original natural language claim
- `translation` - Formal language translation (Lean4/Z3)
- `verification` - Verification result
- `proof_step` - Individual proof steps (for complex proofs)

---

## 3. Belief Network Analysis

### GET /api/belief-network/\{debate_id\}/cruxes

Identify the key "crux" claims that most influence the debate outcome. These are claims where agent agreement/disagreement has the highest impact.

**Parameters:**
- `debate_id` (path) - The debate identifier
- `top_k` (query, default: 3) - Number of cruxes to return (1-10)

**Response:**
```json
{
  "debate_id": "debate_abc123",
  "cruxes": [
    {
      "claim_id": "claim_001",
      "text": "The proposed algorithm has O(n log n) complexity",
      "impact_score": 0.87,
      "supporting_agents": ["claude", "gpt4"],
      "opposing_agents": ["gemini"],
      "resolution_status": "contested"
    },
    {
      "claim_id": "claim_002",
      "text": "Memory usage is acceptable for production",
      "impact_score": 0.72,
      "supporting_agents": ["claude", "gemini", "gpt4"],
      "opposing_agents": [],
      "resolution_status": "consensus"
    }
  ],
  "analysis_method": "belief_propagation"
}
```

### GET /api/belief-network/\{debate_id\}/load-bearing-claims

Get claims with highest centrality in the argument graph. These are foundational claims that support many downstream conclusions.

**Parameters:**
- `debate_id` (path) - The debate identifier
- `limit` (query, default: 5) - Number of claims (1-20)

**Response:**
```json
{
  "debate_id": "debate_abc123",
  "load_bearing_claims": [
    {
      "claim_id": "claim_003",
      "text": "The input is always well-formed JSON",
      "centrality": 0.92,
      "dependent_claims": 12,
      "if_invalidated": "15 downstream claims would need re-evaluation"
    }
  ]
}
```

### GET /api/belief-network/\{debate_id\}/graph

Get the belief network as a graph structure for force-directed visualization.

**Parameters:**
- `debate_id` (path) - The debate identifier
- `include_cruxes` (query, default: true) - Include crux analysis

**Response:**
```json
{
  "nodes": [
    {
      "id": "claim_001",
      "claim_id": "claim_001",
      "statement": "The algorithm has O(n log n) complexity",
      "author": "claude",
      "centrality": 0.85,
      "is_crux": true,
      "crux_score": 0.92,
      "entropy": 0.65,
      "belief": {
        "true_prob": 0.6,
        "false_prob": 0.2,
        "uncertain_prob": 0.2
      }
    }
  ],
  "links": [
    {
      "source": "claim_001",
      "target": "claim_002",
      "weight": 0.7,
      "type": "supports"
    }
  ],
  "metadata": {
    "debate_id": "debate_abc123",
    "total_claims": 15,
    "crux_count": 3
  }
}
```

**Link Types:**
- `supports` - Source claim supports target claim
- `opposes` - Source claim opposes target claim
- `influences` - General influence relationship

### GET /api/belief-network/\{debate_id\}/export

Export the belief network in various formats for external analysis tools.

**Parameters:**
- `debate_id` (path) - The debate identifier
- `format` (query, default: "json") - Export format: `json`, `graphml`, `csv`

**JSON Response:**
```json
{
  "format": "json",
  "debate_id": "debate_abc123",
  "nodes": [...],
  "edges": [...],
  "summary": {
    "total_nodes": 15,
    "total_edges": 23,
    "crux_count": 3
  }
}
```

**GraphML Response:**
```json
{
  "format": "graphml",
  "debate_id": "debate_abc123",
  "content": "<?xml version=\"1.0\"?>...",
  "content_type": "application/xml"
}
```

**CSV Response:**
```json
{
  "format": "csv",
  "debate_id": "debate_abc123",
  "nodes_csv": [...],
  "edges_csv": [...],
  "headers": {
    "nodes": ["id", "statement", "author", "centrality", "is_crux"],
    "edges": ["source", "target", "weight", "type"]
  }
}
```

---

## 4. Prompt Evolution API

### GET /api/evolution/patterns

Discover patterns in how prompts evolve across agents. Identifies successful mutation strategies.

**Parameters:**
- `type` (query) - Filter by pattern type: `improvement`, `regression`, `neutral`
- `limit` (query, default: 10) - Number of patterns (1-50)

**Response:**
```json
{
  "patterns": [
    {
      "pattern_id": "pat_001",
      "name": "Chain-of-thought injection",
      "type": "improvement",
      "frequency": 23,
      "avg_improvement": 0.12,
      "example": "Added 'Let's think step by step' prefix",
      "affected_agents": ["claude", "gpt4"]
    },
    {
      "pattern_id": "pat_002",
      "name": "Explicit role assignment",
      "type": "improvement",
      "frequency": 18,
      "avg_improvement": 0.08,
      "example": "Added 'You are a critical reviewer' preamble"
    }
  ],
  "total_mutations_analyzed": 1247
}
```

### GET /api/evolution/\{agent\}/history

Get the complete prompt evolution history for a specific agent.

**Parameters:**
- `agent` (path) - Agent identifier (e.g., `claude`, `gpt4`)
- `limit` (query, default: 10) - Number of versions (1-50)

**Response:**
```json
{
  "agent": "claude",
  "current_version": 15,
  "history": [
    {
      "version": 15,
      "timestamp": "2026-01-13T10:30:00Z",
      "prompt_hash": "abc123...",
      "performance_delta": +0.05,
      "mutation_type": "self_improvement",
      "parent_version": 14
    },
    {
      "version": 14,
      "timestamp": "2026-01-12T14:20:00Z",
      "prompt_hash": "def456...",
      "performance_delta": +0.02,
      "mutation_type": "tournament_winner"
    }
  ],
  "lineage_depth": 15
}
```

---

## 5. Gauntlet Stress Testing

### POST /api/gauntlet/run

Start an adversarial stress test against a debate configuration. The gauntlet throws various attack patterns to find weaknesses.

**Request:**
```json
{
  "debate_config": {
    "topic": "Evaluate the security of this authentication flow",
    "agents": ["claude", "gpt4", "gemini"]
  },
  "attack_types": ["semantic_manipulation", "prompt_injection", "edge_cases"],
  "intensity": "medium",
  "max_rounds": 10
}
```

**Response:**
```json
{
  "gauntlet_id": "gauntlet_xyz789",
  "status": "running",
  "estimated_duration_seconds": 120,
  "websocket_channel": "gauntlet:xyz789"
}
```

**Attack Types:**
- `semantic_manipulation` - Subtle meaning shifts
- `prompt_injection` - Instruction override attempts
- `edge_cases` - Boundary condition testing
- `contradiction` - Logical inconsistency probing
- `hallucination_trigger` - Fabrication induction

### GET /api/gauntlet/\{id\}/heatmap

Get a risk heatmap showing vulnerability distribution across attack types and agents.

**Response:**
```json
{
  "gauntlet_id": "gauntlet_xyz789",
  "heatmap": {
    "rows": ["claude", "gpt4", "gemini"],
    "columns": ["semantic", "injection", "edge_cases"],
    "values": [
      [0.1, 0.3, 0.2],
      [0.2, 0.1, 0.4],
      [0.3, 0.2, 0.1]
    ],
    "scale": "vulnerability_score"
  },
  "hotspots": [
    {
      "agent": "gpt4",
      "attack": "edge_cases",
      "score": 0.4,
      "recommendation": "Add input validation for numeric bounds"
    }
  ]
}
```

### GET /api/gauntlet/\{id\}/receipt

Get an auditable decision receipt documenting the stress test results.

**Response:**
```json
{
  "gauntlet_id": "gauntlet_xyz789",
  "receipt": {
    "created_at": "2026-01-13T10:45:00Z",
    "hash": "sha256:abc123...",
    "summary": {
      "total_attacks": 47,
      "successful_defenses": 42,
      "vulnerabilities_found": 5,
      "overall_resilience": 0.89
    },
    "signature": "signed_by_aragora_v1.3.0",
    "reproducibility_seed": 42
  }
}
```

---

## 6. Claim Provenance

### GET /api/provenance/\{debate_id\}/claims/\{claim_id\}/support

Trace the verification chain for a specific claim, showing what evidence supports it.

**Response:**
```json
{
  "claim_id": "claim_001",
  "text": "The algorithm terminates in finite time",
  "support_chain": [
    {
      "level": 0,
      "type": "direct_assertion",
      "source": "claude",
      "confidence": 0.85
    },
    {
      "level": 1,
      "type": "corroboration",
      "source": "gpt4",
      "confidence": 0.90,
      "reasoning": "Confirmed via loop invariant analysis"
    },
    {
      "level": 2,
      "type": "formal_proof",
      "source": "z3_verification",
      "confidence": 0.99,
      "proof_reference": "proof_abc123"
    }
  ],
  "aggregate_confidence": 0.95,
  "verification_status": "verified"
}
```

---

## Integration Examples

### Python Client

```python
import httpx

async def discover_features():
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://localhost:8080/api/features/discover")
        data = resp.json()
        print(f"Found {data['total_endpoints']} endpoints")
        print(f"Hidden features: {data['hidden_features']}")

async def verify_claim(claim: str):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:8080/api/verification/formal-verify",
            json={"claim": claim, "claim_type": "logical"}
        )
        return resp.json()

async def get_debate_cruxes(debate_id: str):
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"http://localhost:8080/api/belief-network/\{debate_id\}/cruxes",
            params={"top_k": 5}
        )
        return resp.json()
```

### TypeScript/JavaScript Client

```typescript
// Feature discovery
const discover = async () => {
  const resp = await fetch('/api/features/discover');
  const data = await resp.json();
  console.log(`${data.hidden_features} hidden features available`);
};

// Start gauntlet test
const runGauntlet = async (config: DebateConfig) => {
  const resp = await fetch('/api/gauntlet/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      debate_config: config,
      attack_types: ['semantic_manipulation', 'edge_cases'],
      intensity: 'medium'
    })
  });
  return resp.json();
};
```

---

## Rate Limits

| Endpoint Category | Limit | Burst |
|-------------------|-------|-------|
| Feature Discovery | 60/min | 10 |
| Formal Verification | 10/min | 3 |
| Belief Network | 60/min | 10 |
| Evolution | 10/min | 5 |
| Gauntlet | 5/min | 2 |
| Provenance | 30/min | 5 |

---

## Version History

- **v1.4.0** (2026-01-13): Added verification history endpoints (`/api/verify/history/*`), belief network graph and export endpoints (`/api/belief-network/\{id\}/graph`, `/api/belief-network/\{id\}/export`), frontend crux highlighting
- **v1.3.0** (2026-01-13): Added `/api/features/discover` and `/api/features/endpoints`
- **v1.2.0**: Added gauntlet comparison endpoint
- **v1.1.0**: Added belief network crux detection
- **v1.0.0**: Initial hidden endpoint documentation
