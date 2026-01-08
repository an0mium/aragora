-- Aragora Memory Database Schema
-- Consolidated from: agent_memories.db, continuum.db, consensus_memory.db,
--                    agora_memory.db, semantic_patterns.db, suggestion_feedback.db
-- Version: 1.0.0
-- Last Updated: 2026-01-07

-- Schema version tracking
CREATE TABLE IF NOT EXISTS _schema_versions (
    module TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- CONTINUUM MEMORY (Multi-tier memory system)
-- Source: continuum.db
-- =============================================================================

CREATE TABLE IF NOT EXISTS continuum_memory (
    id TEXT PRIMARY KEY,
    tier TEXT NOT NULL DEFAULT 'slow',  -- 'fast', 'medium', 'slow', 'glacial'
    content TEXT NOT NULL,
    importance REAL DEFAULT 0.5,
    surprise_score REAL DEFAULT 0.0,
    consolidation_score REAL DEFAULT 0.0,
    update_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    semantic_centroid BLOB,  -- Embedding vector
    last_promotion_at TEXT,
    expires_at TEXT,
    metadata TEXT DEFAULT '{}',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tier_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    from_tier TEXT NOT NULL,
    to_tier TEXT NOT NULL,
    reason TEXT,
    surprise_score REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (memory_id) REFERENCES continuum_memory(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS continuum_memory_archive (
    id TEXT PRIMARY KEY,
    tier TEXT NOT NULL,
    content TEXT NOT NULL,
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

CREATE TABLE IF NOT EXISTS meta_learning_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hyperparams TEXT NOT NULL,  -- JSON
    learning_efficiency REAL,
    pattern_retention_rate REAL,
    forgetting_rate REAL,
    cycles_evaluated INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_continuum_tier ON continuum_memory(tier);
CREATE INDEX IF NOT EXISTS idx_continuum_surprise ON continuum_memory(surprise_score DESC);
CREATE INDEX IF NOT EXISTS idx_continuum_importance ON continuum_memory(importance DESC);
CREATE INDEX IF NOT EXISTS idx_continuum_expires ON continuum_memory(expires_at);
CREATE INDEX IF NOT EXISTS idx_archive_tier ON continuum_memory_archive(tier);
CREATE INDEX IF NOT EXISTS idx_archive_archived_at ON continuum_memory_archive(archived_at);
CREATE INDEX IF NOT EXISTS idx_tier_transitions_memory ON tier_transitions(memory_id);

-- =============================================================================
-- AGENT MEMORIES (Short-term and working memory)
-- Source: agent_memories.db
-- =============================================================================

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    memory_type TEXT NOT NULL,  -- 'debate', 'reflection', 'learning', 'interaction'
    content TEXT NOT NULL,
    context TEXT,  -- JSON context (legacy: metadata)
    importance REAL DEFAULT 0.5,
    decay_rate REAL DEFAULT 0.1,
    embedding BLOB,
    debate_id TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    expires_at TEXT
);

CREATE TABLE IF NOT EXISTS reflection_schedule (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    reflection_type TEXT NOT NULL,
    scheduled_for TEXT NOT NULL,
    completed_at TEXT,
    memory_ids TEXT,  -- JSON array of memory IDs to reflect on
    result TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_name);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_debate ON memories(debate_id);
CREATE INDEX IF NOT EXISTS idx_memories_expires ON memories(expires_at);
CREATE INDEX IF NOT EXISTS idx_reflection_agent ON reflection_schedule(agent_name);
CREATE INDEX IF NOT EXISTS idx_reflection_scheduled ON reflection_schedule(scheduled_for);

-- =============================================================================
-- CONSENSUS MEMORY
-- Source: consensus_memory.db
-- =============================================================================

CREATE TABLE IF NOT EXISTS consensus (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    topic_hash TEXT,
    conclusion TEXT,  -- Legacy: position
    strength TEXT,
    confidence REAL DEFAULT 0.5,
    domain TEXT,
    tags TEXT,
    timestamp TEXT,
    data TEXT,  -- JSON full consensus data
    supporting_agents TEXT,  -- JSON array
    opposing_agents TEXT,  -- JSON array
    evidence TEXT,  -- JSON array of evidence items
    debate_ids TEXT,  -- JSON array of debate IDs that formed this consensus
    stability_score REAL DEFAULT 0.5,  -- How stable this consensus has been
    last_challenged_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dissent (
    id TEXT PRIMARY KEY,
    debate_id TEXT,  -- Legacy: consensus_id
    agent_id TEXT NOT NULL,  -- Legacy: agent_name
    dissent_type TEXT,
    content TEXT,  -- Legacy: dissent_position
    confidence REAL DEFAULT 0.5,
    timestamp TEXT,
    data TEXT,  -- JSON full dissent data
    reasoning TEXT,
    strength REAL DEFAULT 0.5,
    resolved INTEGER DEFAULT 0,
    resolved_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (debate_id) REFERENCES consensus(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_consensus_topic ON consensus(topic);
CREATE INDEX IF NOT EXISTS idx_consensus_confidence ON consensus(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_dissent_debate ON dissent(debate_id);
CREATE INDEX IF NOT EXISTS idx_dissent_agent ON dissent(agent_id);

-- =============================================================================
-- CRITIQUE PATTERNS (Learning from critiques)
-- Source: agora_memory.db
-- =============================================================================

-- Debates table from agora_memory.db
CREATE TABLE IF NOT EXISTS debates (
    id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    final_answer TEXT,
    consensus_reached INTEGER,
    confidence REAL,
    rounds_used INTEGER,
    duration_seconds REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS critiques (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    debate_id TEXT,
    agent TEXT NOT NULL,  -- Critic agent
    target_agent TEXT,
    issues TEXT,  -- JSON array
    suggestions TEXT,  -- JSON array
    severity REAL,
    reasoning TEXT,
    led_to_improvement INTEGER DEFAULT 0,
    expected_usefulness REAL DEFAULT 0.5,
    actual_usefulness REAL,
    prediction_error REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (debate_id) REFERENCES debates(id)
);

CREATE TABLE IF NOT EXISTS patterns (
    id TEXT PRIMARY KEY,
    issue_type TEXT NOT NULL,  -- Pattern category
    issue_text TEXT NOT NULL,
    suggestion_text TEXT,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    avg_severity REAL DEFAULT 0.5,
    surprise_score REAL DEFAULT 0.0,
    base_rate REAL DEFAULT 0.5,
    avg_prediction_error REAL DEFAULT 0.0,
    prediction_count INTEGER DEFAULT 0,
    example_task TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS patterns_archive (
    id TEXT,
    issue_type TEXT,
    issue_text TEXT,
    suggestion_text TEXT,
    success_count INTEGER,
    failure_count INTEGER,
    avg_severity REAL,
    surprise_score REAL,
    example_task TEXT,
    created_at TEXT,
    updated_at TEXT,
    archived_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pattern_embeddings (
    pattern_id TEXT PRIMARY KEY,
    embedding BLOB,
    FOREIGN KEY (pattern_id) REFERENCES patterns(id)
);

CREATE TABLE IF NOT EXISTS agent_reputation (
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

CREATE INDEX IF NOT EXISTS idx_debates_task ON debates(task);
CREATE INDEX IF NOT EXISTS idx_critiques_debate ON critiques(debate_id);
CREATE INDEX IF NOT EXISTS idx_critiques_agent ON critiques(agent);
CREATE INDEX IF NOT EXISTS idx_critiques_target ON critiques(target_agent);
CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(issue_type);
CREATE INDEX IF NOT EXISTS idx_patterns_success ON patterns(success_count DESC);
CREATE INDEX IF NOT EXISTS idx_patterns_type_success ON patterns(issue_type, success_count DESC);
CREATE INDEX IF NOT EXISTS idx_patterns_success_updated ON patterns(success_count DESC, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_reputation_agent ON agent_reputation(agent_name);

-- =============================================================================
-- SEMANTIC PATTERNS (Embeddings for semantic search)
-- Source: semantic_patterns.db
-- =============================================================================

-- Source table is 'embeddings' in semantic_patterns.db
CREATE TABLE IF NOT EXISTS semantic_embeddings (
    id TEXT PRIMARY KEY,
    text_hash TEXT UNIQUE,
    text TEXT,
    embedding BLOB,
    provider TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_semantic_hash ON semantic_embeddings(text_hash);

-- =============================================================================
-- USER SUGGESTIONS & FEEDBACK
-- Source: suggestion_feedback.db
-- =============================================================================

CREATE TABLE IF NOT EXISTS suggestion_injections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    debate_id TEXT NOT NULL,
    user_id TEXT,
    suggestion_type TEXT NOT NULL,  -- 'argument', 'question', 'evidence'
    content TEXT NOT NULL,
    target_agent TEXT,
    accepted INTEGER DEFAULT 0,
    impact_score REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS contributor_stats (
    user_id TEXT PRIMARY KEY,
    suggestions_total INTEGER DEFAULT 0,
    suggestions_accepted INTEGER DEFAULT 0,
    acceptance_rate REAL DEFAULT 0.0,
    impact_score_sum REAL DEFAULT 0.0,
    last_contribution_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_suggestions_debate ON suggestion_injections(debate_id);
CREATE INDEX IF NOT EXISTS idx_suggestions_user ON suggestion_injections(user_id);

-- Initialize schema version
INSERT OR REPLACE INTO _schema_versions (module, version)
VALUES ('memory', 1);
