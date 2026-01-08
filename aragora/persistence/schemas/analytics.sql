-- Aragora Analytics Database Schema
-- Consolidated from: agent_elo.db, agent_calibration.db, aragora_insights.db,
--                    prompt_evolution.db, meta_learning.db
-- Version: 1.0.0
-- Last Updated: 2026-01-07

-- Schema version tracking
CREATE TABLE IF NOT EXISTS _schema_versions (
    module TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- ELO RATINGS & MATCHES
-- Source: agent_elo.db
-- =============================================================================

CREATE TABLE IF NOT EXISTS ratings (
    agent_name TEXT PRIMARY KEY,
    elo REAL DEFAULT 1500,
    domain_elos TEXT,  -- JSON map of domain -> elo
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    draws INTEGER DEFAULT 0,
    debates_count INTEGER DEFAULT 0,
    critiques_accepted INTEGER DEFAULT 0,
    critiques_total INTEGER DEFAULT 0,
    calibration_correct INTEGER DEFAULT 0,
    calibration_total INTEGER DEFAULT 0,
    calibration_brier_sum REAL DEFAULT 0.0,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    debate_id TEXT UNIQUE,
    winner TEXT,
    participants TEXT,  -- JSON array of agent names
    domain TEXT,
    scores TEXT,  -- JSON map of agent -> score
    elo_changes TEXT,  -- JSON map of agent -> elo change
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS elo_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    elo REAL NOT NULL,
    debate_id TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_elo_history_agent ON elo_history(agent_name);
CREATE INDEX IF NOT EXISTS idx_elo_history_created ON elo_history(created_at);
CREATE INDEX IF NOT EXISTS idx_elo_history_debate ON elo_history(debate_id);
CREATE INDEX IF NOT EXISTS idx_matches_winner ON matches(winner);
CREATE INDEX IF NOT EXISTS idx_matches_created ON matches(created_at);

-- =============================================================================
-- CALIBRATION DATA
-- Source: agent_elo.db, agent_calibration.db
-- =============================================================================

CREATE TABLE IF NOT EXISTS calibration_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tournament_id TEXT NOT NULL,
    predictor_agent TEXT NOT NULL,
    predicted_winner TEXT NOT NULL,
    confidence REAL NOT NULL,
    actual_winner TEXT,
    resolved_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(tournament_id, predictor_agent)
);

CREATE TABLE IF NOT EXISTS domain_calibration (
    agent_name TEXT NOT NULL,
    domain TEXT NOT NULL,
    total_predictions INTEGER DEFAULT 0,
    total_correct INTEGER DEFAULT 0,
    brier_sum REAL DEFAULT 0.0,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (agent_name, domain)
);

CREATE TABLE IF NOT EXISTS calibration_buckets (
    agent_name TEXT NOT NULL,
    domain TEXT NOT NULL,
    bucket_key TEXT NOT NULL,  -- e.g., "0.7-0.8" for confidence range
    predictions INTEGER DEFAULT 0,
    correct INTEGER DEFAULT 0,
    brier_sum REAL DEFAULT 0.0,
    PRIMARY KEY (agent_name, domain, bucket_key)
);

CREATE INDEX IF NOT EXISTS idx_domain_cal_agent ON domain_calibration(agent_name);

-- =============================================================================
-- INSIGHTS & PATTERNS
-- Source: aragora_insights.db
-- =============================================================================

CREATE TABLE IF NOT EXISTS insights (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,  -- 'pattern', 'anomaly', 'trend', etc.
    title TEXT NOT NULL,
    description TEXT,
    confidence REAL DEFAULT 0.5,
    debate_id TEXT NOT NULL,
    agents_involved TEXT,  -- JSON array
    evidence TEXT,  -- JSON array
    metadata TEXT DEFAULT '{}',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS debate_summaries (
    debate_id TEXT PRIMARY KEY,
    topic TEXT,
    summary TEXT,
    key_arguments TEXT,  -- JSON array
    consensus_reached INTEGER DEFAULT 0,
    final_positions TEXT,  -- JSON map of agent -> position
    duration_seconds REAL,
    rounds_completed INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pattern_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_name TEXT NOT NULL,
    pattern_type TEXT NOT NULL,
    centroid BLOB,  -- Embedding centroid
    member_count INTEGER DEFAULT 0,
    sample_patterns TEXT,  -- JSON array of sample pattern IDs
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agent_performance_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    metric_name TEXT NOT NULL,  -- 'win_rate', 'calibration', 'influence', etc.
    metric_value REAL NOT NULL,
    period TEXT,  -- 'daily', 'weekly', 'monthly'
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    sample_size INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_insights_debate ON insights(debate_id);
CREATE INDEX IF NOT EXISTS idx_insights_type ON insights(type);
CREATE INDEX IF NOT EXISTS idx_perf_history_agent ON agent_performance_history(agent_name);
CREATE INDEX IF NOT EXISTS idx_perf_history_period ON agent_performance_history(period_start);

-- =============================================================================
-- PROMPT EVOLUTION
-- Source: prompt_evolution.db
-- =============================================================================

CREATE TABLE IF NOT EXISTS prompt_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_name TEXT NOT NULL,
    version INTEGER NOT NULL,
    content TEXT NOT NULL,
    parent_version INTEGER,
    fitness_score REAL,
    usage_count INTEGER DEFAULT 0,
    success_rate REAL,
    metadata TEXT DEFAULT '{}',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(prompt_name, version)
);

CREATE TABLE IF NOT EXISTS evolution_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_name TEXT NOT NULL,
    from_version INTEGER NOT NULL,
    to_version INTEGER NOT NULL,
    mutation_type TEXT,  -- 'crossover', 'mutation', 'selection'
    fitness_delta REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS extracted_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,  -- 'debate', 'prompt', 'critique'
    source_id TEXT,
    pattern_text TEXT NOT NULL,
    pattern_type TEXT,
    frequency INTEGER DEFAULT 1,
    effectiveness_score REAL,
    embedding BLOB,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_prompt_versions_name ON prompt_versions(prompt_name);
CREATE INDEX IF NOT EXISTS idx_evolution_prompt ON evolution_history(prompt_name);
CREATE INDEX IF NOT EXISTS idx_patterns_type ON extracted_patterns(pattern_type);

-- =============================================================================
-- META-LEARNING
-- Source: meta_learning.db
-- =============================================================================

CREATE TABLE IF NOT EXISTS meta_hyperparams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    param_name TEXT NOT NULL,
    param_value REAL NOT NULL,
    context TEXT,  -- JSON context for when this param applies
    effectiveness REAL,
    sample_size INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS meta_efficiency_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT NOT NULL,
    learning_rate REAL,
    pattern_retention_rate REAL,
    forgetting_rate REAL,
    convergence_speed REAL,
    hyperparams_snapshot TEXT,  -- JSON snapshot of hyperparams used
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_meta_efficiency_cycle ON meta_efficiency_log(cycle_id);

-- Initialize schema version
INSERT OR REPLACE INTO _schema_versions (module, version)
VALUES ('analytics', 1);
