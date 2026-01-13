# Database Consolidation Analysis

> **Note**: This is a historical document. The migration has been completed. For current database architecture, see [DATABASE.md](./DATABASE.md).

**Status:** Implementation Complete
**Date:** 2026-01-05 (Updated: 2026-01-07)
**Purpose:** Analyze database overlap and recommend consolidation strategy

## Implementation Status (2026-01-07)

The database consolidation plan has been implemented with the following artifacts:

### Created Files

| File | Purpose |
|------|---------|
| `aragora/persistence/schemas/core.sql` | Consolidated schema for debates, traces, tournaments, embeddings |
| `aragora/persistence/schemas/analytics.sql` | Consolidated schema for ELO, calibration, insights |
| `aragora/persistence/schemas/memory.sql` | Consolidated schema for continuum memory, consensus |
| `aragora/persistence/schemas/agents.sql` | Consolidated schema for personas, relationships, genomes |
| `aragora/persistence/db_config.py` | Centralized database path configuration |
| `scripts/migrate_databases.py` | Migration script with rollback support |

### Migration Features

- **Dual-mode support**: `ARAGORA_DB_MODE=legacy` (default) or `ARAGORA_DB_MODE=consolidated`
- **Atomic migration**: Backup, migrate, verify, rollback if needed
- **Dry-run mode**: `--dry-run` to preview changes without modifying databases
- **Data verification**: Row count validation after migration

### Usage

```bash
# Preview migration
python scripts/migrate_databases.py --dry-run

# Run migration
python scripts/migrate_databases.py

# Rollback (if needed)
python scripts/migrate_databases.py --rollback

# Switch to consolidated mode
export ARAGORA_DB_MODE=consolidated
```

---

---

## Executive Summary

The Aragora codebase contains **12+ distinct SQLite databases** and **multiple in-memory caches**. Several databases have overlapping responsibilities, leading to:
- Data duplication across tables
- Inconsistent state between stores
- Complex cross-store queries
- Maintenance burden from multiple schemas

This document inventories all storage systems, identifies consolidation opportunities, and proposes a phased migration strategy.

---

## 1. Database Inventory

### SQLite Databases

| # | Database File | Owner Class | Tables | Purpose |
|---|---------------|-------------|--------|---------|
| 1 | `agora_memory.db` | CritiqueStore | 6 | Critique patterns, agent reputation |
| 2 | `aragora_memory.db` | ContinuumMemory | 3 | Multi-tier memory system |
| 3 | `consensus_memory.db` | ConsensusMemory | 2 | Debate consensus, dissent |
| 4 | `aragora_elo.db` | EloSystem | 7 | ELO ratings, calibration |
| 5 | `aragora_insights.db` | InsightStore | 4 | Insights, pattern clusters |
| 6 | `aragora_debates.db` | DebateStorage | 1 | Debate artifacts, permalinks |
| 7 | `.nomic/genesis.db` | GenesisLedger | 1 | Provenance ledger |
| 8 | `.nomic/genesis.db` | GenomeStore | 1 | Agent genomes (shared DB) |
| 9 | `persona_lab.db`* | PersonaLaboratory | 2 | A/B experiments |
| 10 | `position_ledger.db`* | PositionLedger | 1 | Position tracking |

*\* Created on-demand in working directory*

### JSON/File-Based Storage

| File | Owner | Purpose |
|------|-------|---------|
| `elo_snapshot.json` | EloSystem | Fast-read leaderboard cache |
| `.nomic/nomic_state.json` | NomicLoop | Loop phase/cycle state |
| `.nomic/replays/{id}/` | ReplayStorage | Event logs per session |

### In-Memory Caches

| Class | Purpose | Lifetime |
|-------|---------|----------|
| IntrospectionCache | Agent reputation/traits | Per-debate |
| MemoryStream | Agent memory buffer | Per-agent session |
| BeliefNetwork | Claim DAG | Per-debate |

---

## 2. Overlap Analysis

### 2.1 Position Tracking (3 overlapping systems)

**Problem:** Agent positions tracked in 3 places with different schemas.

| System | Location | Storage | Tracks |
|--------|----------|---------|--------|
| PositionTracker | `agents/truth_grounding.py` | In-memory dict | Claim, stance, confidence |
| PositionLedger | `agents/grounded.py` | SQLite | Position, citations, outcomes |
| ConsensusMemory | `memory/consensus.py` | SQLite | Consensus positions, dissent |

**Overlap:**
- All three track "what position did agent X take on topic Y"
- PositionLedger and ConsensusMemory both link to debate outcomes
- PositionTracker lacks persistence; state lost on restart

**Recommendation:** Merge into **ConsensusMemory** with expanded schema:
```sql
ALTER TABLE consensus ADD COLUMN positions JSON;
-- Store per-agent position snapshots with the consensus
```

---

### 2.2 Persona/Identity Systems (4 overlapping systems)

**Problem:** Agent identity scattered across multiple stores.

| System | Location | Storage | Tracks |
|--------|----------|---------|--------|
| PersonaManager | `agents/personas.py` | JSON file | Traits, expertise, specialization |
| PersonaLaboratory | `agents/laboratory.py` | SQLite | A/B experiments, performance |
| PersonaSynthesizer | `agents/grounded.py` | Computed | Generated identity prompts |
| GenomeStore | `genesis/genome.py` | SQLite | Genetic traits, lineage |

**Overlap:**
- PersonaManager and GenomeStore both track agent traits
- PersonaLaboratory experiments reference PersonaManager configs
- PersonaSynthesizer pulls from multiple sources without caching

**Recommendation:** Consolidate into **GenomeStore** as source of truth:
- Add `experiments` table to genesis.db
- PersonaManager becomes thin wrapper over GenomeStore
- PersonaSynthesizer caches generated prompts in GenomeStore

---

### 2.3 Memory Systems (4 overlapping systems)

**Problem:** Multiple memory tiers with unclear boundaries.

| System | Location | Storage | Tracks |
|--------|----------|---------|--------|
| ContinuumMemory | `memory/continuum.py` | SQLite | 4-tier memories (fast/slow) |
| MemoryStream | `memory/streams.py` | In-memory | Per-agent conversation buffer |
| CritiqueStore | `memory/store.py` | SQLite | Critique patterns, reputation |
| InsightStore | `insights/store.py` | SQLite | Extracted debate insights |

**Overlap:**
- ContinuumMemory and CritiqueStore both store debate outcomes
- InsightStore duplicates some CritiqueStore pattern data
- MemoryStream is ephemeral; valuable context lost

**Recommendation:** Hierarchical consolidation:
```
ContinuumMemory (primary)
  ├── Absorb CritiqueStore tables (reputation → slow tier)
  ├── Absorb InsightStore (insights → medium tier)
  └── Periodic flush from MemoryStream → fast tier
```

---

### 2.4 Debate Storage (2 overlapping systems)

**Problem:** Debate data split between stores.

| System | Location | Storage | Tracks |
|--------|----------|---------|--------|
| DebateStorage | `server/storage.py` | SQLite | Full debate JSON, slugs |
| ConsensusMemory | `memory/consensus.py` | SQLite | Consensus outcomes, dissent |

**Overlap:**
- Both store debate outcomes
- DebateStorage has full artifact; ConsensusMemory has summary
- Cross-referencing requires debate_id lookups

**Recommendation:** Add foreign key relationship:
```sql
-- In consensus_memory.db
ALTER TABLE consensus ADD COLUMN debate_artifact_id TEXT;
-- Reference into aragora_debates.db
```

---

## 3. Proposed Consolidated Architecture

### Target: 4 Core Databases

| Database | Purpose | Migrated From |
|----------|---------|---------------|
| `aragora_core.db` | Debates, consensus, positions | DebateStorage, ConsensusMemory, PositionLedger, PositionTracker |
| `aragora_memory.db` | Multi-tier learning | ContinuumMemory, CritiqueStore, InsightStore, MemoryStream |
| `aragora_agents.db` | Agent identity, rankings | EloSystem, GenomeStore, PersonaManager, PersonaLaboratory |
| `.nomic/genesis.db` | Immutable provenance | GenesisLedger (unchanged) |

### Schema Overview

```
aragora_core.db
├── debates (from DebateStorage)
├── consensus (from ConsensusMemory)
├── dissent (from ConsensusMemory)
└── positions (NEW: merged from PositionLedger/Tracker)

aragora_memory.db (expanded)
├── continuum_memory (existing)
├── tier_transitions (existing)
├── meta_learning_state (existing)
├── critiques (from CritiqueStore)
├── patterns (from CritiqueStore)
├── agent_reputation (from CritiqueStore)
├── insights (from InsightStore)
├── pattern_clusters (from InsightStore)
└── memory_snapshots (NEW: from MemoryStream)

aragora_agents.db
├── ratings (from EloSystem)
├── matches (from EloSystem)
├── elo_history (from EloSystem)
├── calibration_* (from EloSystem)
├── genomes (from GenomeStore)
├── genome_versions (from GenomeStore)
├── personas (NEW: from PersonaManager)
└── experiments (from PersonaLaboratory)
```

---

## 4. Migration Strategy

### Phase 1: Schema Alignment (Low Risk)

1. **Add missing foreign keys** to existing tables
2. **Create views** that join across databases for compatibility
3. **Implement SchemaManager** migrations for all databases

```python
# Example migration
schema_manager.register_migration(
    module="consensus",
    version=2,
    up="""
        ALTER TABLE consensus ADD COLUMN debate_artifact_id TEXT;
        CREATE INDEX idx_consensus_artifact ON consensus(debate_artifact_id);
    """,
    down="ALTER TABLE consensus DROP COLUMN debate_artifact_id;"
)
```

**Risk:** Low - additive changes only
**Duration:** 1-2 days

---

### Phase 2: Create Unified Interfaces (Medium Risk)

1. **Create adapter classes** that read from old locations, write to new
2. **Add deprecation warnings** to old direct database access
3. **Update high-level APIs** to use adapters

```python
class UnifiedAgentStore:
    """Single interface for agent identity data."""

    def __init__(self, db_path: str = "aragora_agents.db"):
        self._elo = EloSystem(db_path)
        self._genomes = GenomeStore(db_path)  # Point to same file
        self._personas = PersonaManager(db_path)

    def get_agent_profile(self, agent: str) -> dict:
        """Unified profile from all sources."""
        return {
            "ratings": self._elo.get_agent_stats(agent),
            "genome": self._genomes.get_genome(agent),
            "persona": self._personas.get_persona(agent),
        }
```

**Risk:** Medium - API changes may affect consumers
**Duration:** 3-5 days

---

### Phase 3: Data Migration (High Risk)

1. **Export all data** from old databases to JSON backups
2. **Run migration scripts** to populate new schema
3. **Validate data integrity** with checksums
4. **Switch production** to new databases
5. **Keep old databases read-only** for 30 days

```bash
# Migration script outline
python scripts/migrate_databases.py \
  --source-dir ./old_dbs \
  --target-dir ./new_dbs \
  --validate \
  --dry-run
```

**Risk:** High - data loss if migration fails
**Duration:** 1-2 weeks (including testing)

---

### Phase 4: Cleanup (Low Risk)

1. **Remove deprecated code paths**
2. **Delete old database files** after validation period
3. **Update documentation**
4. **Archive migration scripts**

**Risk:** Low - only after successful validation
**Duration:** 1-2 days

---

## 5. Risk Assessment

### Data Loss Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Migration script bug | Medium | High | JSON backup before migration |
| Schema mismatch | Low | Medium | Dry-run validation |
| Foreign key violations | Medium | Low | Deferred constraint checks |
| Transaction rollback | Low | Medium | Batch commits with checkpoints |

### API Breakage Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Handler expects old schema | High | Medium | Adapter layer |
| Direct SQL queries break | Medium | High | Deprecation warnings |
| Cache invalidation | Low | Low | Clear all caches on migration |

### Performance Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Larger DB = slower queries | Medium | Medium | Add appropriate indexes |
| Lock contention | Low | High | WAL mode, connection pooling |
| Memory usage increase | Low | Low | Lazy loading |

---

## 6. Open Questions

1. **Should genesis.db remain separate?**
   - Pro: Immutability guarantees clearer with isolated DB
   - Con: Extra file to manage

2. **SQLite vs PostgreSQL for consolidation?**
   - SQLite sufficient for current scale (< 100K debates)
   - PostgreSQL if horizontal scaling needed later

3. **Backward compatibility period?**
   - 30 days of dual-write recommended
   - Longer if external consumers exist

4. **Migration timing?**
   - During low-activity period
   - After next major release tagged

---

## 7. Recommended Next Steps

1. **Immediate:** Add SchemaManager to all databases without it
2. **Week 1:** Create unified adapter interfaces (Phase 2)
3. **Week 2:** Write and test migration scripts
4. **Week 3:** Staging environment migration + validation
5. **Week 4:** Production migration with rollback plan

---

## Appendix: Current Database Locations

```
project_root/
├── agora_memory.db          # CritiqueStore
├── aragora_memory.db        # ContinuumMemory
├── aragora_elo.db           # EloSystem
├── aragora_insights.db      # InsightStore
├── aragora_debates.db       # DebateStorage
├── consensus_memory.db      # ConsensusMemory
├── persona_lab.db           # PersonaLaboratory (on-demand)
├── position_ledger.db       # PositionLedger (on-demand)
├── elo_snapshot.json        # EloSystem cache
└── .nomic/
    ├── genesis.db           # GenesisLedger + GenomeStore
    ├── nomic_state.json     # Loop state
    └── replays/             # ReplayStorage
```

---

## Appendix: Table Count Summary

| Current Databases | Tables |
|-------------------|--------|
| agora_memory.db | 6 |
| aragora_memory.db | 3 |
| consensus_memory.db | 2 |
| aragora_elo.db | 7 |
| aragora_insights.db | 4 |
| aragora_debates.db | 1 |
| genesis.db | 2 |
| persona_lab.db | 2 |
| position_ledger.db | 1 |
| **Total** | **28** |

| Proposed Databases | Tables |
|--------------------|--------|
| aragora_core.db | 4 |
| aragora_memory.db | 9 |
| aragora_agents.db | 9 |
| genesis.db | 1 |
| **Total** | **23** |

**Reduction:** 28 → 23 tables, 10 → 4 database files
