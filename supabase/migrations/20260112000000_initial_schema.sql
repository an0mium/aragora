-- Aragora Core Database Schema
-- Consolidated from: debates storage, traces, tournaments, embeddings, positions
-- Version: 1.0.0
-- Last Updated: 2026-01-07

-- Schema version tracking
CREATE TABLE IF NOT EXISTS _schema_versions (
    module TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- DEBATES (Core debate records)
-- Source: server/storage.py
-- =============================================================================

CREATE TABLE IF NOT EXISTS debates (
    id TEXT PRIMARY KEY,
    slug TEXT UNIQUE NOT NULL,
    task TEXT NOT NULL,
    agents TEXT NOT NULL,  -- JSON array
    artifact_json TEXT NOT NULL,
    consensus_reached BOOLEAN,
    confidence REAL,
    view_count INTEGER DEFAULT 0,
    audio_path TEXT,
    audio_generated_at TIMESTAMP,
    audio_duration_seconds INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_debates_slug ON debates(slug);
CREATE INDEX IF NOT EXISTS idx_debates_created ON debates(created_at);
CREATE INDEX IF NOT EXISTS idx_debates_consensus ON debates(consensus_reached);

-- =============================================================================
-- DEBATE METADATA (Configuration tracking)
-- Source: runtime/metadata.py
-- =============================================================================

CREATE TABLE IF NOT EXISTS debate_metadata (
    debate_id TEXT PRIMARY KEY,
    config_hash TEXT,
    task_hash TEXT,
    metadata_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (debate_id) REFERENCES debates(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_metadata_config_hash ON debate_metadata(config_hash);
CREATE INDEX IF NOT EXISTS idx_metadata_task_hash ON debate_metadata(task_hash);

-- =============================================================================
-- TRACES (Debate execution traces for replay)
-- Source: debate/traces.py
-- =============================================================================

CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    debate_id TEXT,
    task TEXT,
    agents TEXT,  -- JSON array
    random_seed INTEGER,
    checksum TEXT,
    trace_json TEXT,  -- Full trace data
    started_at TEXT,
    completed_at TEXT,
    FOREIGN KEY (debate_id) REFERENCES debates(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS trace_events (
    event_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    round_num INTEGER,
    agent TEXT,
    content TEXT,  -- JSON event content
    timestamp TEXT,
    FOREIGN KEY (trace_id) REFERENCES traces(trace_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_traces_debate ON traces(debate_id);
CREATE INDEX IF NOT EXISTS idx_trace_events_trace ON trace_events(trace_id);
CREATE INDEX IF NOT EXISTS idx_trace_events_type ON trace_events(event_type);

-- =============================================================================
-- TOURNAMENTS (Tournament system)
-- Source: tournaments/tournament.py
-- =============================================================================

CREATE TABLE IF NOT EXISTS tournaments (
    tournament_id TEXT PRIMARY KEY,
    name TEXT,
    format TEXT,  -- 'round_robin', 'swiss', 'single_elimination', 'free_for_all'
    agents TEXT,  -- JSON array of participating agents
    tasks TEXT,  -- JSON array of task IDs
    standings TEXT,  -- JSON map of agent -> score
    champion TEXT,
    started_at TEXT,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS tournament_matches (
    match_id TEXT PRIMARY KEY,
    tournament_id TEXT NOT NULL,
    round_num INTEGER,
    participants TEXT,  -- JSON array
    task_id TEXT,
    scores TEXT,  -- JSON map of agent -> score
    winner TEXT,
    started_at TEXT,
    completed_at TEXT,
    FOREIGN KEY (tournament_id) REFERENCES tournaments(tournament_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tournament_matches_tournament ON tournament_matches(tournament_id);
CREATE INDEX IF NOT EXISTS idx_tournament_matches_round ON tournament_matches(round_num);

-- =============================================================================
-- EMBEDDINGS (Vector embedding cache)
-- Source: memory/embeddings.py
-- =============================================================================

CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY,
    text_hash TEXT UNIQUE,
    text TEXT,  -- Truncated to 1000 chars for storage
    embedding BYTEA NOT NULL,
    provider TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_embeddings_hash ON embeddings(text_hash);

-- =============================================================================
-- POSITIONS (Agent positions and flips)
-- Source: insights/flip_detector.py, agents/grounded.py
-- =============================================================================

CREATE TABLE IF NOT EXISTS positions (
    id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    claim TEXT NOT NULL,
    confidence REAL NOT NULL,
    debate_id TEXT NOT NULL,
    round_num INTEGER NOT NULL,
    outcome TEXT DEFAULT 'pending',  -- 'pending', 'correct', 'incorrect', 'unknown'
    reversed INTEGER DEFAULT 0,
    reversal_debate_id TEXT,
    domain TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS detected_flips (
    id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    original_claim TEXT NOT NULL,
    new_claim TEXT NOT NULL,
    original_confidence REAL,
    new_confidence REAL,
    original_debate_id TEXT,
    new_debate_id TEXT,
    original_position_id TEXT,
    new_position_id TEXT,
    similarity_score REAL,
    flip_type TEXT,  -- 'contradiction', 'retraction', 'refinement', 'strengthening'
    domain TEXT,
    detected_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_positions_agent ON positions(agent_name);
CREATE INDEX IF NOT EXISTS idx_positions_debate ON positions(debate_id);
CREATE INDEX IF NOT EXISTS idx_positions_outcome ON positions(outcome);
CREATE INDEX IF NOT EXISTS idx_flips_agent ON detected_flips(agent_name);
CREATE INDEX IF NOT EXISTS idx_flips_type ON detected_flips(flip_type);

-- Initialize schema version
INSERT INTO _schema_versions (module, version)
VALUES ('core', 1) ON CONFLICT (module) DO UPDATE SET version = EXCLUDED.version, updated_at = CURRENT_TIMESTAMP;
-- Aragora Agents Database Schema
-- Consolidated from: agent_personas.db, persona_lab.db, agent_relationships.db,
--                    grounded_positions.db, genesis.db, agent_calibration.db
-- Version: 1.0.0
-- Last Updated: 2026-01-07

-- Schema version tracking
CREATE TABLE IF NOT EXISTS _schema_versions (
    module TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- PERSONAS (Agent personality definitions)
-- Source: agents/personas.py
-- =============================================================================

CREATE TABLE IF NOT EXISTS personas (
    agent_name TEXT PRIMARY KEY,
    description TEXT,
    traits TEXT,  -- JSON array of trait strings
    expertise TEXT,  -- JSON map of domain -> proficiency
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS performance_history (
    id SERIAL PRIMARY KEY,
    agent_name TEXT NOT NULL,
    debate_id TEXT,
    domain TEXT,
    action TEXT,  -- 'argue', 'critique', 'vote', etc.
    success INTEGER,  -- 1 = success, 0 = failure
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_performance_agent ON performance_history(agent_name);
CREATE INDEX IF NOT EXISTS idx_performance_domain ON performance_history(domain);
CREATE INDEX IF NOT EXISTS idx_performance_action ON performance_history(action);

-- =============================================================================
-- AGENT RELATIONSHIPS (Inter-agent dynamics)
-- Source: agents/grounded.py, ranking/elo.py
-- =============================================================================

CREATE TABLE IF NOT EXISTS agent_relationships (
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
    CHECK (agent_a < agent_b)  -- Canonical ordering
);

CREATE INDEX IF NOT EXISTS idx_relationships_a ON agent_relationships(agent_a);
CREATE INDEX IF NOT EXISTS idx_relationships_b ON agent_relationships(agent_b);

-- =============================================================================
-- POSITION HISTORY & TRUTH GROUNDING
-- Source: agents/truth_grounding.py
-- =============================================================================

CREATE TABLE IF NOT EXISTS position_history (
    id SERIAL PRIMARY KEY,
    debate_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    position_type TEXT NOT NULL,  -- 'initial', 'revised', 'final'
    position_text TEXT NOT NULL,
    round_num INTEGER DEFAULT 0,
    confidence REAL DEFAULT 0.5,
    was_winning_position INTEGER DEFAULT NULL,
    verified_correct INTEGER DEFAULT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(debate_id, agent_name, position_type, round_num)
);

CREATE TABLE IF NOT EXISTS debate_outcomes (
    debate_id TEXT PRIMARY KEY,
    winning_agent TEXT,
    winning_position TEXT,
    consensus_confidence REAL,
    verified_at TEXT DEFAULT NULL,
    verification_result INTEGER DEFAULT NULL,  -- 1 = correct, 0 = incorrect
    verification_source TEXT DEFAULT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_position_history_agent ON position_history(agent_name);
CREATE INDEX IF NOT EXISTS idx_position_history_debate ON position_history(debate_id);

-- =============================================================================
-- PERSONA LABORATORY (A/B testing & trait experiments)
-- Source: agents/laboratory.py
-- =============================================================================

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    control_persona TEXT NOT NULL,  -- JSON persona state
    variant_persona TEXT NOT NULL,  -- JSON persona state
    hypothesis TEXT,
    status TEXT DEFAULT 'running',  -- 'running', 'completed', 'cancelled'
    control_successes INTEGER DEFAULT 0,
    control_trials INTEGER DEFAULT 0,
    variant_successes INTEGER DEFAULT 0,
    variant_trials INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS emergent_traits (
    id SERIAL PRIMARY KEY,
    trait_name TEXT NOT NULL,
    source_agents TEXT NOT NULL,  -- JSON array of contributing agents
    supporting_evidence TEXT,  -- JSON array
    confidence REAL DEFAULT 0.5,
    first_detected TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trait_transfers (
    id SERIAL PRIMARY KEY,
    from_agent TEXT NOT NULL,
    to_agent TEXT NOT NULL,
    trait TEXT NOT NULL,
    expertise_domain TEXT,
    success_rate_before REAL,
    success_rate_after REAL,
    transferred_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agent_evolution_history (
    id SERIAL PRIMARY KEY,
    agent_name TEXT NOT NULL,
    mutation_type TEXT NOT NULL,  -- 'trait_add', 'trait_remove', 'expertise_change'
    before_state TEXT NOT NULL,  -- JSON state
    after_state TEXT NOT NULL,  -- JSON state
    reason TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_experiments_agent ON experiments(agent_name);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_trait_transfers_from ON trait_transfers(from_agent);
CREATE INDEX IF NOT EXISTS idx_trait_transfers_to ON trait_transfers(to_agent);
CREATE INDEX IF NOT EXISTS idx_evolution_agent ON agent_evolution_history(agent_name);

-- =============================================================================
-- CALIBRATION PREDICTIONS
-- Source: agents/calibration.py
-- =============================================================================

CREATE TABLE IF NOT EXISTS calibration_predictions (
    id SERIAL PRIMARY KEY,
    agent TEXT NOT NULL,
    confidence REAL NOT NULL,
    correct INTEGER NOT NULL,  -- 1 = correct, 0 = incorrect
    domain TEXT DEFAULT 'general',
    debate_id TEXT,
    position_id TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cal_pred_agent ON calibration_predictions(agent);
CREATE INDEX IF NOT EXISTS idx_cal_pred_domain ON calibration_predictions(domain);
CREATE INDEX IF NOT EXISTS idx_cal_pred_confidence ON calibration_predictions(confidence);

-- =============================================================================
-- GENESIS SYSTEM (Agent breeding and evolution)
-- Source: genesis/genome.py, genesis/breeding.py, genesis/ledger.py
-- =============================================================================

CREATE TABLE IF NOT EXISTS genomes (
    genome_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    traits TEXT,  -- JSON array
    expertise TEXT,  -- JSON map of domain -> proficiency
    model_preference TEXT,
    parent_genomes TEXT,  -- JSON array of parent genome IDs
    generation INTEGER DEFAULT 0,
    fitness_score REAL DEFAULT 0.5,
    birth_debate_id TEXT,
    consensus_contributions INTEGER DEFAULT 0,
    critiques_accepted INTEGER DEFAULT 0,
    predictions_correct INTEGER DEFAULT 0,
    debates_participated INTEGER DEFAULT 0,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS populations (
    population_id TEXT PRIMARY KEY,
    genome_ids TEXT,  -- JSON array of genome IDs
    generation INTEGER DEFAULT 0,
    debate_history TEXT,  -- JSON array of debate IDs
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS active_population (
    id INTEGER PRIMARY KEY CHECK (id = 1),  -- Singleton row
    population_id TEXT,
    FOREIGN KEY (population_id) REFERENCES populations(population_id)
);

CREATE TABLE IF NOT EXISTS genesis_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,  -- 'birth', 'mutation', 'crossover', 'selection', 'extinction'
    timestamp TEXT NOT NULL,
    parent_event_id TEXT,
    content_hash TEXT NOT NULL,
    data TEXT,  -- JSON event data
    FOREIGN KEY (parent_event_id) REFERENCES genesis_events(event_id)
);

CREATE INDEX IF NOT EXISTS idx_genomes_fitness ON genomes(fitness_score DESC);
CREATE INDEX IF NOT EXISTS idx_genomes_generation ON genomes(generation);
CREATE INDEX IF NOT EXISTS idx_populations_generation ON populations(generation);
CREATE INDEX IF NOT EXISTS idx_genesis_events_type ON genesis_events(event_type);
CREATE INDEX IF NOT EXISTS idx_genesis_events_timestamp ON genesis_events(timestamp);

-- Initialize schema version
INSERT INTO _schema_versions (module, version)
VALUES ('agents', 1) ON CONFLICT (module) DO UPDATE SET version = EXCLUDED.version, updated_at = CURRENT_TIMESTAMP;
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
    id SERIAL PRIMARY KEY,
    debate_id TEXT UNIQUE,
    winner TEXT,
    participants TEXT,  -- JSON array of agent names
    domain TEXT,
    scores TEXT,  -- JSON map of agent -> score
    elo_changes TEXT,  -- JSON map of agent -> elo change
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS elo_history (
    id SERIAL PRIMARY KEY,
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
    id SERIAL PRIMARY KEY,
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
    id SERIAL PRIMARY KEY,
    cluster_name TEXT NOT NULL,
    pattern_type TEXT NOT NULL,
    centroid BYTEA,  -- Embedding centroid
    member_count INTEGER DEFAULT 0,
    sample_patterns TEXT,  -- JSON array of sample pattern IDs
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agent_performance_history (
    id SERIAL PRIMARY KEY,
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
    id SERIAL PRIMARY KEY,
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
    id SERIAL PRIMARY KEY,
    prompt_name TEXT NOT NULL,
    from_version INTEGER NOT NULL,
    to_version INTEGER NOT NULL,
    mutation_type TEXT,  -- 'crossover', 'mutation', 'selection'
    fitness_delta REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS extracted_patterns (
    id SERIAL PRIMARY KEY,
    source_type TEXT NOT NULL,  -- 'debate', 'prompt', 'critique'
    source_id TEXT,
    pattern_text TEXT NOT NULL,
    pattern_type TEXT,
    frequency INTEGER DEFAULT 1,
    effectiveness_score REAL,
    embedding BYTEA,
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
    id SERIAL PRIMARY KEY,
    param_name TEXT NOT NULL,
    param_value REAL NOT NULL,
    context TEXT,  -- JSON context for when this param applies
    effectiveness REAL,
    sample_size INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS meta_efficiency_log (
    id SERIAL PRIMARY KEY,
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
INSERT INTO _schema_versions (module, version)
VALUES ('analytics', 1) ON CONFLICT (module) DO UPDATE SET version = EXCLUDED.version, updated_at = CURRENT_TIMESTAMP;
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
    semantic_centroid BYTEA,  -- Embedding vector
    last_promotion_at TEXT,
    expires_at TEXT,
    metadata TEXT DEFAULT '{}',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tier_transitions (
    id SERIAL PRIMARY KEY,
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
    semantic_centroid BYTEA,
    created_at TEXT,
    updated_at TEXT,
    archived_at TEXT DEFAULT CURRENT_TIMESTAMP,
    archive_reason TEXT,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS meta_learning_state (
    id SERIAL PRIMARY KEY,
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
    embedding BYTEA,
    debate_id TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    expires_at TEXT
);

CREATE TABLE IF NOT EXISTS reflection_schedule (
    id SERIAL PRIMARY KEY,
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
    id SERIAL PRIMARY KEY,
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
    embedding BYTEA,
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
    embedding BYTEA,
    provider TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_semantic_hash ON semantic_embeddings(text_hash);

-- =============================================================================
-- USER SUGGESTIONS & FEEDBACK
-- Source: suggestion_feedback.db
-- =============================================================================

CREATE TABLE IF NOT EXISTS suggestion_injections (
    id SERIAL PRIMARY KEY,
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
INSERT INTO _schema_versions (module, version)
VALUES ('memory', 1) ON CONFLICT (module) DO UPDATE SET version = EXCLUDED.version, updated_at = CURRENT_TIMESTAMP;
