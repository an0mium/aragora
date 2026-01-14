# Evidence API Guide

The Evidence API provides endpoints for collecting, storing, and retrieving evidence to support AI agent debates with factual information.

## Overview

Evidence in Aragora consists of:
- **Citations**: Links to external sources
- **Quotes**: Verbatim text from sources
- **Data Points**: Structured facts and statistics
- **Quality Scores**: Credibility ratings for each piece of evidence

## API Endpoints

| Endpoint | Method | Description | Rate Limit |
|----------|--------|-------------|------------|
| `/api/evidence` | GET | List all evidence | 60/min |
| `/api/evidence/:id` | GET | Get evidence by ID | 60/min |
| `/api/evidence/search` | POST | Full-text search | 60/min |
| `/api/evidence/collect` | POST | Collect new evidence | 10/min |
| `/api/evidence/debate/:id` | GET | Get debate evidence | 60/min |
| `/api/evidence/debate/:id` | POST | Associate evidence | 10/min |
| `/api/evidence/statistics` | GET | Store statistics | 60/min |
| `/api/evidence/:id` | DELETE | Delete evidence | 10/min |

## Usage Examples

### List Evidence

```bash
# Basic listing with pagination
curl "http://localhost:8080/api/evidence?limit=20&offset=0"

# Filter by source type
curl "http://localhost:8080/api/evidence?source_type=academic"

# Filter by minimum quality score
curl "http://localhost:8080/api/evidence?min_quality=0.7"
```

Response:
```json
{
  "evidence": [
    {
      "id": "ev-12345",
      "title": "Climate Change Impact Study",
      "source": "https://example.com/study.pdf",
      "source_type": "academic",
      "content": "Global temperatures have risen...",
      "quality_score": 0.92,
      "created_at": "2024-01-15T10:30:00Z",
      "metadata": {
        "author": "Dr. Smith",
        "publication": "Nature",
        "year": 2024
      }
    }
  ],
  "total": 150,
  "limit": 20,
  "offset": 0
}
```

### Search Evidence

```bash
curl -X POST "http://localhost:8080/api/evidence/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "renewable energy efficiency",
    "source_types": ["academic", "government"],
    "min_quality": 0.6,
    "limit": 10
  }'
```

Response:
```json
{
  "results": [
    {
      "id": "ev-67890",
      "title": "Solar Panel Efficiency Report",
      "relevance_score": 0.95,
      "quality_score": 0.88,
      "snippet": "...renewable energy sources have shown 40% efficiency improvements..."
    }
  ],
  "query": "renewable energy efficiency",
  "total_matches": 42
}
```

### Collect Evidence

Automatically gather evidence for a topic:

```bash
curl -X POST "http://localhost:8080/api/evidence/collect" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "topic": "Impact of AI on job market",
    "sources": ["academic", "news", "government"],
    "max_items": 10,
    "quality_threshold": 0.7
  }'
```

Response:
```json
{
  "collected": 8,
  "failed": 2,
  "evidence_ids": ["ev-001", "ev-002", "ev-003", "..."],
  "quality_summary": {
    "average_score": 0.82,
    "high_quality_count": 6,
    "sources_used": ["arxiv", "bls.gov", "reuters"]
  }
}
```

### Associate Evidence with Debate

```bash
curl -X POST "http://localhost:8080/api/evidence/debate/debate-123" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "evidence_ids": ["ev-001", "ev-002"],
    "relevance_notes": "Supporting arguments for pro side"
  }'
```

### Get Debate Evidence

```bash
curl "http://localhost:8080/api/evidence/debate/debate-123"
```

Response:
```json
{
  "debate_id": "debate-123",
  "evidence": [
    {
      "id": "ev-001",
      "title": "Economic Impact Study",
      "used_by_agents": ["claude-opus", "gpt-4"],
      "citation_count": 3,
      "quality_score": 0.89
    }
  ],
  "total": 5
}
```

### Get Statistics

```bash
curl "http://localhost:8080/api/evidence/statistics"
```

Response:
```json
{
  "total_evidence": 1250,
  "by_source_type": {
    "academic": 450,
    "news": 380,
    "government": 220,
    "other": 200
  },
  "average_quality": 0.76,
  "storage_size_mb": 45.2,
  "oldest_evidence": "2023-06-15T00:00:00Z",
  "newest_evidence": "2024-01-15T12:00:00Z"
}
```

## TypeScript SDK Usage

```typescript
import { AragoraClient } from 'aragora-js';

const client = new AragoraClient({
  baseUrl: 'http://localhost:8080',
  accessToken: 'your-token',
});

// List evidence
const evidence = await client.evidence.list({
  limit: 20,
  sourceType: 'academic',
  minQuality: 0.7,
});

// Search evidence
const results = await client.evidence.search({
  query: 'machine learning healthcare',
  sourceTypes: ['academic', 'government'],
  limit: 10,
});

// Collect new evidence
const collected = await client.evidence.collect({
  topic: 'Quantum computing applications',
  sources: ['academic'],
  maxItems: 5,
});

// Get debate evidence
const debateEvidence = await client.evidence.forDebate('debate-123');
```

## Quality Scoring

Evidence quality is scored based on:

| Factor | Weight | Description |
|--------|--------|-------------|
| Source Credibility | 30% | Academic, government sources score higher |
| Recency | 20% | More recent evidence preferred |
| Citation Count | 20% | Well-cited sources score higher |
| Content Relevance | 20% | Match to query/topic |
| Verification Status | 10% | Fact-checked content bonus |

Quality score ranges:
- **0.8-1.0**: High quality, authoritative sources
- **0.6-0.8**: Good quality, reliable sources
- **0.4-0.6**: Moderate quality, verify claims
- **0.0-0.4**: Low quality, use with caution

## Source Types

| Type | Description | Typical Quality |
|------|-------------|-----------------|
| `academic` | Peer-reviewed papers, journals | High |
| `government` | Official reports, statistics | High |
| `news` | Major news outlets | Medium-High |
| `industry` | Company reports, whitepapers | Medium |
| `social` | Social media, forums | Low-Medium |
| `other` | Miscellaneous sources | Varies |

## Error Handling

Common error responses:

```json
// Rate limited
{
  "error": "Rate limit exceeded",
  "retry_after": 60
}

// Not found
{
  "error": "Evidence not found",
  "evidence_id": "ev-invalid"
}

// Collection failed
{
  "error": "Evidence collection failed",
  "reason": "External API unavailable",
  "partial_results": ["ev-001"]
}
```

## Related Documentation

- [API Reference](./API_REFERENCE.md)
- [Debate Phases](./DEBATE_PHASES.md)
- [Agent Development](./AGENT_DEVELOPMENT.md)
