# Evidence Module

Collects, scores, and persists factual evidence used to ground debates.
Evidence flows through a pipeline: collection from external sources, metadata
enrichment, quality scoring, and storage with full attribution tracking.

## Modules

| Module | Description |
|--------|-------------|
| `collector.py` | Fetches citations/snippets from connectors with SSRF protection |
| `metadata.py` | Enriches snippets with source type, provenance, and confidence |
| `quality.py` | Scores relevance, freshness, authority, and semantic confidence |
| `attribution.py` | Cross-debate attribution chains and source reputation tracking |
| `store.py` | SQLite-backed persistence with FTS search and deduplication |

## Key Classes

- **`EvidenceCollector`** -- Gathers `EvidenceSnippet`s into an `EvidencePack` from registered connectors.
- **`MetadataEnricher`** -- Classifies `SourceType` (academic, news, code, etc.) and attaches `Provenance`.
- **`QualityScorer`** / **`QualityFilter`** -- Assigns `QualityScores` and a `QualityTier` (excellent to unreliable).
- **`SourceReputationManager`** -- Maintains `SourceReputation` across debates; updates on `VerificationOutcome`.
- **`EvidenceStore`** / **`InMemoryEvidenceStore`** -- Save, search, and retrieve evidence per debate.

## Usage

```python
from aragora.evidence import (
    EvidenceCollector,
    EvidenceStore,
    QualityScorer,
    MetadataEnricher,
)

# Collect evidence for a debate topic
collector = EvidenceCollector()
pack = await collector.collect("rate limiter design patterns")

# Enrich and score
enricher = MetadataEnricher()
scorer = QualityScorer()
for snippet in pack.snippets:
    enriched = enricher.enrich(snippet)
    scores = scorer.score(snippet)

# Persist to SQLite store
store = EvidenceStore(db_path="evidence.db")
store.save_pack(pack, debate_id="debate-001")

# Search later
results = store.search("rate limiter", debate_id="debate-001")
```

## Quality Tiers

| Tier | Score Range |
|------|-------------|
| Excellent | >= 0.85 |
| Good | >= 0.70 |
| Fair | >= 0.50 |
| Poor | >= 0.30 |
| Unreliable | < 0.30 |
