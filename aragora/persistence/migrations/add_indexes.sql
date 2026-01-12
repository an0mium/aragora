-- Performance indexes for Aragora database tables
-- Run with: sqlite3 aragora.db < add_indexes.sql

-- Debates table indexes
CREATE INDEX IF NOT EXISTS idx_debates_created_at ON debates(created_at);
CREATE INDEX IF NOT EXISTS idx_debates_user_id ON debates(user_id);
CREATE INDEX IF NOT EXISTS idx_debates_org_id ON debates(org_id);
CREATE INDEX IF NOT EXISTS idx_debates_status ON debates(status);
CREATE INDEX IF NOT EXISTS idx_debates_domain ON debates(domain);

-- Composite index for common query patterns
CREATE INDEX IF NOT EXISTS idx_debates_user_created ON debates(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_debates_org_created ON debates(org_id, created_at DESC);

-- Consensus records indexes
CREATE INDEX IF NOT EXISTS idx_consensus_debate_id ON consensus(debate_id);
CREATE INDEX IF NOT EXISTS idx_consensus_created_at ON consensus(created_at);
CREATE INDEX IF NOT EXISTS idx_consensus_domain ON consensus(domain);

-- Composite index for consensus queries
CREATE INDEX IF NOT EXISTS idx_consensus_confidence_ts ON consensus(confidence DESC, timestamp DESC);

-- Dissent records indexes
CREATE INDEX IF NOT EXISTS idx_dissent_debate_id ON dissent(debate_id);
CREATE INDEX IF NOT EXISTS idx_dissent_timestamp ON dissent(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_dissent_agent_id ON dissent(agent_id);

-- Agent metrics indexes
CREATE INDEX IF NOT EXISTS idx_agent_metrics_agent_name ON agent_metrics(agent_name);
CREATE INDEX IF NOT EXISTS idx_agent_metrics_timestamp ON agent_metrics(timestamp DESC);

-- Stream events indexes
CREATE INDEX IF NOT EXISTS idx_stream_events_loop_id ON stream_events(loop_id);
CREATE INDEX IF NOT EXISTS idx_stream_events_timestamp ON stream_events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_stream_events_event_type ON stream_events(event_type);

-- Composite index for event queries
CREATE INDEX IF NOT EXISTS idx_stream_events_loop_ts ON stream_events(loop_id, timestamp DESC);

-- Nomic cycles indexes
CREATE INDEX IF NOT EXISTS idx_nomic_cycles_loop_id ON nomic_cycles(loop_id);
CREATE INDEX IF NOT EXISTS idx_nomic_cycles_started_at ON nomic_cycles(started_at DESC);

-- Usage tracking indexes (if table exists)
CREATE INDEX IF NOT EXISTS idx_usage_records_org_id ON usage_records(org_id);
CREATE INDEX IF NOT EXISTS idx_usage_records_created_at ON usage_records(created_at);
CREATE INDEX IF NOT EXISTS idx_usage_records_org_period ON usage_records(org_id, period_start);

-- Verified proofs indexes
CREATE INDEX IF NOT EXISTS idx_verified_proofs_debate_id ON verified_proofs(debate_id);
CREATE INDEX IF NOT EXISTS idx_verified_proofs_status ON verified_proofs(proof_status);
CREATE INDEX IF NOT EXISTS idx_verified_proofs_verified ON verified_proofs(is_verified);

-- Note: Run ANALYZE after adding indexes for optimal query planning
-- ANALYZE;
