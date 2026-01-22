---
title: Aragora Database Schema Reference
description: Aragora Database Schema Reference
---

# Aragora Database Schema Reference

This document provides comprehensive documentation of all SQLite databases used in Aragora.

## Overview

Aragora uses **6 major SQLite databases** for persistence:

| Database | File | Purpose | Typical Size |
|----------|------|---------|--------------|
| Continuum Memory | `continuum.db` | Multi-tier learning memory | Large (50K+ entries) |
| Consensus Memory | `consensus_memory.db` | Debate outcomes & dissents | Medium (1K-10K debates) |
| ELO Rankings | `aragora_elo.db` | Agent rankings & calibration | Small-Medium (10K matches) |
| Critique Store | `aragora_traces.db` | Pattern learning | Large (100K+ critiques) |
| Evidence Store | `evidence.db` | Evidence persistence | Medium (10K+ snippets) |
| Circuit Breaker | `.data/circuit_breaker.db` | Resilience tracking | Small |

All databases use **SQLite with WAL mode** for concurrent access.

---

## 1. Continuum Memory Database

**Path:** `continuum.db`
**Purpose:** Multi-tier memory system for nested learning (Google Research paradigm)
**Schema Version:** 3

### Memory Tiers

| Tier | Half-life | Purpose |
|------|-----------|---------|
| Fast | 1 hour | Immediate pattern learning |
| Medium | 24 hours | Session memory |
| Slow | 7 days | Cross-session learning |
| Glacial | 30 days | Foundational knowledge |

### Tables

#### `continuum_memory`

Main memory storage table.

```sql
CREATE TABLE continuum_memory (
    id TEXT PRIMARY KEY,
    tier TEXT NOT NULL,                    -- fast/medium/slow/glacial
    content TEXT NOT NULL,
    importance REAL DEFAULT 0.5,           -- 0-1
    surprise_score REAL DEFAULT 0.0,       -- Novelty score
    consolidation_score REAL DEFAULT 0.0,  -- 0-1
    update_count INTEGER DEFAULT 1,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    semantic_centroid BLOB,                -- Vector embedding
    last_promotion_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT,                         -- JSON
    expires_at TEXT,                       -- TTL expiration
    red_line INTEGER DEFAULT 0,            -- Protected from deletion
    red_line_reason TEXT
);

-- Key indexes
CREATE INDEX idx_continuum_tier ON continuum_memory(tier);
CREATE INDEX idx_continuum_surprise ON continuum_memory(surprise_score DESC);
CREATE INDEX idx_continuum_importance ON continuum_memory(importance DESC);
CREATE INDEX idx_continuum_tier_updated ON continuum_memory(tier, updated_at DESC);
CREATE INDEX idx_continuum_expires ON continuum_memory(expires_at);
CREATE INDEX idx_continuum_red_line ON continuum_memory(red_line);
```

#### `meta_learning_state`

Tracks meta-learning hyperparameters.

```sql
CREATE TABLE meta_learning_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hyperparams TEXT,              -- JSON
    learning_efficiency REAL,
    pattern_retention_rate REAL,
    forgetting_rate REAL,
    cycles_evaluated INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

#### `tier_transitions`

Records memory tier promotions/demotions.

```sql
CREATE TABLE tier_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    from_tier TEXT NOT NULL,
    to_tier TEXT NOT NULL,
    reason TEXT,
    surprise_score REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

#### `continuum_memory_archive`

Archive for deleted/expired memories.

```sql
CREATE TABLE continuum_memory_archive (
    id TEXT PRIMARY KEY,
    tier TEXT,
    content TEXT,
    importance REAL,
    surprise_score REAL,
    consolidation_score REAL,
    update_count INTEGER,
    success_count INTEGER,
    failure_count INTEGER,
    semantic_centroid BLOB,
    created_at TEXT,
    updated_at TEXT,
    archived_at TEXT DEFAULT CURRENT_TIMESTAMP,
    archive_reason TEXT,
    metadata TEXT
);
```

---

## 2. Consensus Memory Database

**Path:** `consensus_memory.db`
**Purpose:** Persistent storage of debate outcomes and dissenting views
**Schema Version:** 2

### Consensus Strength Values

- `unanimous` - All agents agreed
- `strong` - >80% agreement
- `moderate` - 60-80% agreement
- `weak` - 50-60% agreement
- `split` - No clear majority
- `contested` - Active disagreement

### Dissent Types

- `minor_quibble` - Small disagreement
- `alternative_approach` - Different valid approach
- `fundamental_disagreement` - Core disagreement
- `edge_case_concern` - Edge case issues
- `risk_warning` - Risk identification
- `abstention` - No position taken

### Tables

#### `consensus`

```sql
CREATE TABLE consensus (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    topic_hash TEXT NOT NULL,       -- Normalized for similarity
    conclusion TEXT,
    strength TEXT,                  -- Consensus strength enum
    confidence REAL,                -- 0-1
    domain TEXT,
    tags TEXT,                      -- JSON array
    timestamp TEXT,
    data TEXT                       -- Complete JSON record
);

CREATE INDEX idx_consensus_topic_hash ON consensus(topic_hash);
CREATE INDEX idx_consensus_domain ON consensus(domain);
CREATE INDEX idx_consensus_confidence_ts ON consensus(confidence DESC, timestamp DESC);
```

#### `dissent`

```sql
CREATE TABLE dissent (
    id TEXT PRIMARY KEY,
    debate_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    dissent_type TEXT NOT NULL,
    content TEXT,
    confidence REAL,
    timestamp TEXT,
    data TEXT                       -- Complete JSON record
);

CREATE INDEX idx_dissent_debate ON dissent(debate_id);
CREATE INDEX idx_dissent_type ON dissent(dissent_type);
CREATE INDEX idx_dissent_timestamp ON dissent(timestamp DESC);
```

#### `verified_proofs`

Formal verification results (v2+).

```sql
CREATE TABLE verified_proofs (
    id TEXT PRIMARY KEY,
    debate_id TEXT NOT NULL,
    proof_status TEXT,              -- proof_found, failed, timeout
    language TEXT,                  -- z3_smt, lean4
    formal_statement TEXT,
    is_verified INTEGER DEFAULT 0,
    proof_hash TEXT,
    translation_time_ms REAL,
    proof_search_time_ms REAL,
    prover_version TEXT,
    error_message TEXT,
    timestamp TEXT,
    data TEXT
);

CREATE INDEX idx_verified_proofs_debate ON verified_proofs(debate_id);
CREATE INDEX idx_verified_proofs_status ON verified_proofs(proof_status);
```

---

## 3. ELO Rankings Database

**Path:** `aragora_elo.db`
**Purpose:** Agent skill tracking, match history, calibration scoring
**Schema Version:** 2

### Tables

#### `ratings`

```sql
CREATE TABLE ratings (
    agent_name TEXT PRIMARY KEY,
    elo REAL DEFAULT 1500.0,
    domain_elos TEXT,               -- JSON dict of domain-specific ELOs
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    draws INTEGER DEFAULT 0,
    debates_count INTEGER DEFAULT 0,
    critiques_accepted INTEGER DEFAULT 0,
    critiques_total INTEGER DEFAULT 0,
    calibration_correct INTEGER DEFAULT 0,    -- v2
    calibration_total INTEGER DEFAULT 0,      -- v2
    calibration_brier_sum REAL DEFAULT 0.0,   -- v2
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

#### `matches`

```sql
CREATE TABLE matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    debate_id TEXT UNIQUE,
    winner TEXT,
    participants TEXT,              -- JSON array
    domain TEXT,
    scores TEXT,                    -- JSON dict
    elo_changes TEXT,               -- JSON dict
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_matches_winner ON matches(winner);
CREATE INDEX idx_matches_created ON matches(created_at DESC);
CREATE INDEX idx_matches_domain ON matches(domain);
```

#### `elo_history`

```sql
CREATE TABLE elo_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    elo REAL NOT NULL,
    debate_id TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_elo_history_agent ON elo_history(agent_name);
CREATE INDEX idx_elo_history_created ON elo_history(created_at DESC);
```

#### `agent_relationships`

Tracks pairwise agent interactions.

```sql
CREATE TABLE agent_relationships (
    agent_a TEXT NOT NULL,
    agent_b TEXT NOT NULL,
    debate_count INTEGER DEFAULT 0,
    agreement_count INTEGER DEFAULT 0,
    critique_count_a_to_b INTEGER DEFAULT 0,
    critique_count_b_to_a INTEGER DEFAULT 0,
    critique_accepted_a_to_b INTEGER DEFAULT 0,
    critique_accepted_b_to_a INTEGER DEFAULT 0,
    position_changes_a_after_b INTEGER DEFAULT 0,
    position_changes_b_after_a INTEGER DEFAULT 0,
    a_wins_over_b INTEGER DEFAULT 0,
    b_wins_over_a INTEGER DEFAULT 0,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (agent_a, agent_b),
    CHECK (agent_a < agent_b)       -- Canonical ordering
);
```

#### `calibration_predictions`

```sql
CREATE TABLE calibration_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tournament_id TEXT NOT NULL,
    predictor_agent TEXT NOT NULL,
    predicted_winner TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(tournament_id, predictor_agent)
);
```

#### `domain_calibration`

```sql
CREATE TABLE domain_calibration (
    agent_name TEXT NOT NULL,
    domain TEXT NOT NULL,
    total_predictions INTEGER DEFAULT 0,
    total_correct INTEGER DEFAULT 0,
    brier_sum REAL DEFAULT 0.0,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (agent_name, domain)
);
```

---

## 4. Critique Store Database

**Path:** `aragora_traces.db`
**Purpose:** Store debate patterns, critiques, and agent reputation

### Tables

#### `debates`

```sql
CREATE TABLE debates (
    id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    final_answer TEXT,
    consensus_reached INTEGER DEFAULT 0,
    confidence REAL,
    rounds_used INTEGER,
    duration_seconds REAL,
    grounded_verdict TEXT,          -- JSON
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

#### `critiques`

```sql
CREATE TABLE critiques (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    debate_id TEXT NOT NULL,
    agent TEXT NOT NULL,
    target_agent TEXT,
    issues TEXT,                    -- JSON array
    suggestions TEXT,               -- JSON array
    severity REAL,                  -- 0-1
    reasoning TEXT,
    led_to_improvement INTEGER DEFAULT 0,
    expected_usefulness REAL,
    actual_usefulness REAL,
    prediction_error REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_critiques_debate ON critiques(debate_id);
```

#### `patterns`

```sql
CREATE TABLE patterns (
    id TEXT PRIMARY KEY,
    issue_type TEXT NOT NULL,
    issue_text TEXT NOT NULL,
    suggestion_text TEXT,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    avg_severity REAL DEFAULT 0.5,
    surprise_score REAL DEFAULT 1.0,
    base_rate REAL DEFAULT 0.5,
    avg_prediction_error REAL DEFAULT 0.0,
    prediction_count INTEGER DEFAULT 0,
    example_task TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_patterns_type ON patterns(issue_type);
CREATE INDEX idx_patterns_success ON patterns(success_count DESC);
```

#### `agent_reputation`

```sql
CREATE TABLE agent_reputation (
    agent_name TEXT PRIMARY KEY,
    proposals_made INTEGER DEFAULT 0,
    proposals_accepted INTEGER DEFAULT 0,
    critiques_given INTEGER DEFAULT 0,
    critiques_valuable INTEGER DEFAULT 0,
    total_predictions INTEGER DEFAULT 0,
    total_prediction_error REAL DEFAULT 0.0,
    calibration_score REAL DEFAULT 0.5,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_reputation_score ON agent_reputation(calibration_score DESC);
```

---

## 5. Evidence Store Database

**Path:** `evidence.db`
**Purpose:** Persistence for evidence snippets used in debates

### Tables

#### `evidence`

```sql
CREATE TABLE evidence (
    id TEXT PRIMARY KEY,
    content_hash TEXT UNIQUE,       -- SHA256 for deduplication
    source TEXT NOT NULL,           -- github, web, arxiv, etc.
    title TEXT,
    snippet TEXT NOT NULL,
    url TEXT,
    reliability_score REAL DEFAULT 0.5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata_json TEXT,
    enriched_metadata_json TEXT,
    quality_scores_json TEXT
);

CREATE INDEX idx_evidence_source ON evidence(source);
CREATE INDEX idx_evidence_created ON evidence(created_at DESC);
```

#### `debate_evidence`

```sql
CREATE TABLE debate_evidence (
    debate_id TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    round_number INTEGER,
    relevance_score REAL,
    used_in_consensus BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (debate_id, evidence_id)
);

CREATE INDEX idx_debate_evidence_debate ON debate_evidence(debate_id);
```

#### `evidence_fts` (Full-Text Search)

```sql
CREATE VIRTUAL TABLE evidence_fts USING fts5(
    evidence_id,
    title,
    snippet,
    topics
);
```

---

## Migration System

### Version Tracking

```sql
CREATE TABLE _schema_versions (
    module TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Migration History

| Database | Version | Changes |
|----------|---------|---------|
| Continuum | v2→v3 | Added `red_line`, `red_line_reason` columns |
| ELO | v1→v2 | Added calibration columns to ratings |
| Consensus | v1→v2 | Added `verified_proofs` table |

---

## Database Configuration

All databases are configured with:

```python
PRAGMA journal_mode = WAL;          # Write-Ahead Logging
PRAGMA synchronous = NORMAL;        # Safety + performance balance
PRAGMA busy_timeout = 30000;        # 30 second timeout
PRAGMA cache_size = -64000;         # 64MB cache
```

---

## Data Retention

| Database | Default Retention | Cleanup Strategy |
|----------|-------------------|------------------|
| Continuum | TTL per tier | Automatic expiration |
| Consensus | Indefinite | Manual archive |
| ELO | Indefinite | History pruning (optional) |
| Critiques | 90 days | Archive old patterns |
| Evidence | 90 days | TTL cleanup |

---

## Backup Procedures

See [DISASTER_RECOVERY.md](./disaster-recovery) for backup and restoration procedures.
