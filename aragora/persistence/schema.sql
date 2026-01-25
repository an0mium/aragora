-- Aragora Supabase Schema
-- Run this in the Supabase SQL Editor (Database → SQL Editor)

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- NOMIC CYCLES
-- Tracks each cycle of the nomic loop
-- ============================================================================

CREATE TABLE IF NOT EXISTS nomic_cycles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    loop_id TEXT NOT NULL,
    cycle_number INTEGER NOT NULL,
    phase TEXT NOT NULL,  -- debate, design, implement, verify, commit
    stage TEXT,           -- proposing, critiquing, voting, executing, etc.
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    success BOOLEAN,
    git_commit TEXT,
    task_description TEXT,
    total_tasks INTEGER DEFAULT 0,
    completed_tasks INTEGER DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(loop_id, cycle_number, phase)
);

-- Index for querying by loop
CREATE INDEX idx_nomic_cycles_loop ON nomic_cycles(loop_id, started_at DESC);
CREATE INDEX idx_nomic_cycles_phase ON nomic_cycles(phase, started_at DESC);

-- ============================================================================
-- DEBATE ARTIFACTS
-- Full debate transcripts and results
-- ============================================================================

CREATE TABLE IF NOT EXISTS debate_artifacts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    loop_id TEXT NOT NULL,
    cycle_number INTEGER NOT NULL,
    phase TEXT NOT NULL,
    task TEXT NOT NULL,
    agents JSONB NOT NULL,           -- Array of agent names
    transcript JSONB NOT NULL,        -- Full message history
    consensus_reached BOOLEAN NOT NULL DEFAULT FALSE,
    confidence FLOAT NOT NULL DEFAULT 0,
    winning_proposal TEXT,
    vote_tally JSONB,                 -- {agent: votes}
    created_at TIMESTAMPTZ DEFAULT NOW(),
    view_count INTEGER DEFAULT 0
);

-- Indexes
CREATE INDEX idx_debate_artifacts_loop ON debate_artifacts(loop_id, created_at DESC);
CREATE INDEX idx_debate_artifacts_phase ON debate_artifacts(phase, created_at DESC);
CREATE INDEX idx_debate_artifacts_consensus ON debate_artifacts(consensus_reached, created_at DESC);

-- ============================================================================
-- STREAM EVENTS
-- Real-time events from the nomic loop (time-series)
-- ============================================================================

CREATE TABLE IF NOT EXISTS stream_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    loop_id TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    event_type TEXT NOT NULL,  -- cycle_start, phase_start, task_complete, error, etc.
    event_data JSONB NOT NULL DEFAULT '{}',
    agent TEXT,                -- Which agent generated the event (if applicable)
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for time-series queries
CREATE INDEX idx_stream_events_loop_time ON stream_events(loop_id, timestamp DESC);
CREATE INDEX idx_stream_events_type ON stream_events(event_type, timestamp DESC);
CREATE INDEX idx_stream_events_agent ON stream_events(agent, timestamp DESC) WHERE agent IS NOT NULL;

-- ============================================================================
-- AGENT METRICS
-- Performance tracking for each agent
-- ============================================================================

CREATE TABLE IF NOT EXISTS agent_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    loop_id TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    agent_name TEXT NOT NULL,
    model TEXT NOT NULL,
    phase TEXT NOT NULL,
    messages_sent INTEGER DEFAULT 0,
    proposals_made INTEGER DEFAULT 0,
    critiques_given INTEGER DEFAULT 0,
    votes_won INTEGER DEFAULT 0,
    votes_received INTEGER DEFAULT 0,
    consensus_contributions INTEGER DEFAULT 0,
    avg_response_time_ms FLOAT,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_agent_metrics_agent ON agent_metrics(agent_name, timestamp DESC);
CREATE INDEX idx_agent_metrics_loop ON agent_metrics(loop_id, cycle);

-- ============================================================================
-- ENABLE REAL-TIME
-- Required for aragora.ai subscriptions
-- ============================================================================

-- Enable realtime for stream_events (most important for live dashboard)
ALTER PUBLICATION supabase_realtime ADD TABLE stream_events;

-- Optionally enable for other tables
ALTER PUBLICATION supabase_realtime ADD TABLE nomic_cycles;
ALTER PUBLICATION supabase_realtime ADD TABLE debate_artifacts;

-- ============================================================================
-- ROW LEVEL SECURITY (RLS)
-- For now, allow all access (can restrict later)
-- ============================================================================

-- Enable RLS
ALTER TABLE nomic_cycles ENABLE ROW LEVEL SECURITY;
ALTER TABLE debate_artifacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE stream_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_metrics ENABLE ROW LEVEL SECURITY;

-- Allow all access with service role key
CREATE POLICY "Allow all for service role" ON nomic_cycles FOR ALL USING (true);
CREATE POLICY "Allow all for service role" ON debate_artifacts FOR ALL USING (true);
CREATE POLICY "Allow all for service role" ON stream_events FOR ALL USING (true);
CREATE POLICY "Allow all for service role" ON agent_metrics FOR ALL USING (true);

-- Allow public read access (for dashboard)
CREATE POLICY "Allow public read" ON nomic_cycles FOR SELECT USING (true);
CREATE POLICY "Allow public read" ON debate_artifacts FOR SELECT USING (true);
CREATE POLICY "Allow public read" ON stream_events FOR SELECT USING (true);
CREATE POLICY "Allow public read" ON agent_metrics FOR SELECT USING (true);

-- ============================================================================
-- SLACK WORKSPACES
-- OAuth credentials for multi-workspace Slack integration
-- ============================================================================

CREATE TABLE IF NOT EXISTS slack_workspaces (
    workspace_id TEXT PRIMARY KEY,
    workspace_name TEXT NOT NULL,
    access_token TEXT NOT NULL,  -- Encrypted bot token (xoxb-*)
    bot_user_id TEXT NOT NULL,
    installed_at TIMESTAMPTZ NOT NULL,
    installed_by TEXT,           -- User ID who installed
    scopes TEXT[],               -- OAuth scopes granted
    tenant_id TEXT,              -- Link to Aragora tenant
    is_active BOOLEAN DEFAULT TRUE,
    signing_secret TEXT,         -- Workspace-specific signing secret
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_slack_workspaces_tenant ON slack_workspaces(tenant_id);
CREATE INDEX idx_slack_workspaces_active ON slack_workspaces(is_active);

-- Enable RLS
ALTER TABLE slack_workspaces ENABLE ROW LEVEL SECURITY;

-- Service role only (tokens are sensitive)
CREATE POLICY "Allow service role only" ON slack_workspaces
    FOR ALL USING (auth.role() = 'service_role');

-- ============================================================================
-- USAGE METRICS
-- Token usage and cost tracking for billing
-- ============================================================================

CREATE TABLE IF NOT EXISTS usage_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id TEXT,           -- Slack workspace or tenant
    tenant_id TEXT,              -- Aragora tenant
    debate_id UUID,              -- Associated debate
    provider TEXT NOT NULL,      -- anthropic, openai, etc.
    model TEXT NOT NULL,         -- claude-3-opus, gpt-4, etc.
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd DECIMAL(10,6) NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for billing queries
CREATE INDEX idx_usage_metrics_workspace ON usage_metrics(workspace_id, created_at);
CREATE INDEX idx_usage_metrics_tenant ON usage_metrics(tenant_id, created_at);
CREATE INDEX idx_usage_metrics_provider ON usage_metrics(provider, created_at);

-- Enable RLS
ALTER TABLE usage_metrics ENABLE ROW LEVEL SECURITY;

-- Allow all for service role
CREATE POLICY "Allow all for service role" ON usage_metrics FOR ALL USING (true);

-- ============================================================================
-- VIEWS FOR ANALYTICS
-- ============================================================================

-- Recent activity summary
CREATE OR REPLACE VIEW recent_activity AS
SELECT
    loop_id,
    COUNT(DISTINCT cycle_number) as total_cycles,
    COUNT(*) FILTER (WHERE success = true) as successful_phases,
    COUNT(*) FILTER (WHERE success = false) as failed_phases,
    MAX(started_at) as last_activity,
    MIN(started_at) as first_activity
FROM nomic_cycles
WHERE started_at > NOW() - INTERVAL '7 days'
GROUP BY loop_id
ORDER BY last_activity DESC;

-- Agent leaderboard
CREATE OR REPLACE VIEW agent_leaderboard AS
SELECT
    agent_name,
    model,
    COUNT(*) as total_participations,
    SUM(proposals_made) as total_proposals,
    SUM(votes_won) as total_votes_won,
    SUM(consensus_contributions) as total_consensus,
    AVG(avg_response_time_ms) as avg_response_time
FROM agent_metrics
GROUP BY agent_name, model
ORDER BY total_votes_won DESC;

-- ============================================================================
-- TEAMS TENANTS
-- OAuth credentials for multi-tenant Microsoft Teams integration
-- ============================================================================

CREATE TABLE IF NOT EXISTS teams_tenants (
    tenant_id TEXT PRIMARY KEY,        -- Azure AD tenant ID
    tenant_name TEXT NOT NULL,
    access_token TEXT NOT NULL,        -- Encrypted bot access token
    refresh_token TEXT,                -- OAuth refresh token
    bot_id TEXT NOT NULL,              -- Bot ID in Teams
    installed_at TIMESTAMPTZ NOT NULL,
    installed_by TEXT,                 -- User ID who installed
    scopes TEXT[],                     -- OAuth scopes granted
    aragora_org_id TEXT,               -- Link to Aragora organization
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMPTZ,            -- Token expiration
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_teams_tenants_org ON teams_tenants(aragora_org_id);
CREATE INDEX idx_teams_tenants_active ON teams_tenants(is_active);
CREATE INDEX idx_teams_tenants_expires ON teams_tenants(expires_at);

-- Enable RLS
ALTER TABLE teams_tenants ENABLE ROW LEVEL SECURITY;

-- Service role only (tokens are sensitive)
CREATE POLICY "Allow service role only" ON teams_tenants
    FOR ALL USING (auth.role() = 'service_role');
