-- =============================================================================
-- Aragora PostgreSQL Schema
-- =============================================================================
-- Unified PostgreSQL schema for multi-user production deployment.
-- All tables are created in the 'aragora' schema for isolation.
--
-- Usage:
--   psql -d aragora -f postgres_schema.sql
--
-- Or via migration:
--   python scripts/migrate_sqlite_to_postgres.py --target-url postgresql://...
-- =============================================================================

-- Create schema
CREATE SCHEMA IF NOT EXISTS aragora;
SET search_path TO aragora, public;

-- =============================================================================
-- Schema Versioning
-- =============================================================================

CREATE TABLE IF NOT EXISTS _schema_versions (
    module TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- Webhook Configurations
-- =============================================================================

CREATE TABLE IF NOT EXISTS webhook_configs (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    events_json JSONB NOT NULL,
    secret TEXT NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    name TEXT,
    description TEXT,
    last_delivery_at TIMESTAMPTZ,
    last_delivery_status INTEGER,
    delivery_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    user_id TEXT,
    workspace_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_webhook_configs_user ON webhook_configs(user_id);
CREATE INDEX IF NOT EXISTS idx_webhook_configs_workspace ON webhook_configs(workspace_id);
CREATE INDEX IF NOT EXISTS idx_webhook_configs_active ON webhook_configs(active);

-- =============================================================================
-- Integration Configurations
-- =============================================================================

CREATE TABLE IF NOT EXISTS integrations (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notify_on_consensus BOOLEAN DEFAULT TRUE,
    notify_on_debate_end BOOLEAN DEFAULT TRUE,
    notify_on_error BOOLEAN DEFAULT FALSE,
    notify_on_leaderboard BOOLEAN DEFAULT FALSE,
    default_channel TEXT,
    config_json JSONB NOT NULL DEFAULT '{}',
    user_id TEXT NOT NULL,
    workspace_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_integrations_user ON integrations(user_id);
CREATE INDEX IF NOT EXISTS idx_integrations_type ON integrations(type);
CREATE INDEX IF NOT EXISTS idx_integrations_workspace ON integrations(workspace_id);

-- User ID Mappings (cross-platform identity)
CREATE TABLE IF NOT EXISTS user_id_mappings (
    email TEXT NOT NULL,
    platform TEXT NOT NULL,
    platform_user_id TEXT NOT NULL,
    display_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id TEXT NOT NULL DEFAULT 'default',
    PRIMARY KEY (email, platform, user_id)
);

CREATE INDEX IF NOT EXISTS idx_user_mappings_platform ON user_id_mappings(platform);
CREATE INDEX IF NOT EXISTS idx_user_mappings_user ON user_id_mappings(user_id);

-- =============================================================================
-- Gmail Token Store
-- =============================================================================

CREATE TABLE IF NOT EXISTS gmail_tokens (
    user_id TEXT NOT NULL,
    email TEXT NOT NULL,
    access_token TEXT NOT NULL,
    refresh_token TEXT,
    token_uri TEXT,
    client_id TEXT,
    client_secret TEXT,
    scopes_json JSONB,
    expiry TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, email)
);

CREATE INDEX IF NOT EXISTS idx_gmail_tokens_user ON gmail_tokens(user_id);

-- =============================================================================
-- Finding Workflow Store
-- =============================================================================

CREATE TABLE IF NOT EXISTS finding_workflows (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    config_json JSONB NOT NULL DEFAULT '{}',
    results_json JSONB,
    error_message TEXT,
    user_id TEXT,
    workspace_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_finding_workflows_status ON finding_workflows(status);
CREATE INDEX IF NOT EXISTS idx_finding_workflows_user ON finding_workflows(user_id);

-- =============================================================================
-- Gauntlet Run Store
-- =============================================================================

CREATE TABLE IF NOT EXISTS gauntlet_runs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    config_json JSONB NOT NULL DEFAULT '{}',
    results_json JSONB,
    metrics_json JSONB,
    error_message TEXT,
    challenge_id TEXT,
    user_id TEXT,
    workspace_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_gauntlet_runs_status ON gauntlet_runs(status);
CREATE INDEX IF NOT EXISTS idx_gauntlet_runs_user ON gauntlet_runs(user_id);
CREATE INDEX IF NOT EXISTS idx_gauntlet_runs_challenge ON gauntlet_runs(challenge_id);

-- =============================================================================
-- Job Queue Store
-- =============================================================================

CREATE TABLE IF NOT EXISTS job_queue (
    id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    priority INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    scheduled_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    payload_json JSONB NOT NULL DEFAULT '{}',
    result_json JSONB,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    user_id TEXT,
    workspace_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_job_queue_status ON job_queue(status);
CREATE INDEX IF NOT EXISTS idx_job_queue_type ON job_queue(job_type);
CREATE INDEX IF NOT EXISTS idx_job_queue_scheduled ON job_queue(scheduled_at);
CREATE INDEX IF NOT EXISTS idx_job_queue_priority ON job_queue(priority DESC);

-- =============================================================================
-- Governance Store
-- =============================================================================

CREATE TABLE IF NOT EXISTS governance_artifacts (
    id TEXT PRIMARY KEY,
    artifact_type TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    content_json JSONB NOT NULL DEFAULT '{}',
    metadata_json JSONB,
    version INTEGER DEFAULT 1,
    parent_id TEXT REFERENCES governance_artifacts(id),
    user_id TEXT,
    workspace_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_governance_type ON governance_artifacts(artifact_type);
CREATE INDEX IF NOT EXISTS idx_governance_status ON governance_artifacts(status);
CREATE INDEX IF NOT EXISTS idx_governance_user ON governance_artifacts(user_id);

-- =============================================================================
-- Marketplace Store
-- =============================================================================

CREATE TABLE IF NOT EXISTS marketplace_items (
    id TEXT PRIMARY KEY,
    item_type TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    published_at TIMESTAMPTZ,
    content_json JSONB NOT NULL DEFAULT '{}',
    metadata_json JSONB,
    price_cents INTEGER DEFAULT 0,
    download_count INTEGER DEFAULT 0,
    rating REAL,
    author_id TEXT,
    workspace_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_marketplace_type ON marketplace_items(item_type);
CREATE INDEX IF NOT EXISTS idx_marketplace_status ON marketplace_items(status);
CREATE INDEX IF NOT EXISTS idx_marketplace_author ON marketplace_items(author_id);

-- =============================================================================
-- Federation Registry Store
-- =============================================================================

CREATE TABLE IF NOT EXISTS federation_nodes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    url TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ,
    public_key TEXT,
    capabilities_json JSONB,
    trust_level INTEGER DEFAULT 0,
    owner_id TEXT,
    workspace_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_federation_status ON federation_nodes(status);
CREATE INDEX IF NOT EXISTS idx_federation_owner ON federation_nodes(owner_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_federation_url ON federation_nodes(url);

-- =============================================================================
-- Approval Request Store
-- =============================================================================

CREATE TABLE IF NOT EXISTS approval_requests (
    id TEXT PRIMARY KEY,
    request_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    requester_id TEXT NOT NULL,
    approver_id TEXT,
    resource_id TEXT,
    resource_type TEXT,
    payload_json JSONB NOT NULL DEFAULT '{}',
    response_json JSONB,
    comment TEXT,
    workspace_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_approval_status ON approval_requests(status);
CREATE INDEX IF NOT EXISTS idx_approval_requester ON approval_requests(requester_id);
CREATE INDEX IF NOT EXISTS idx_approval_type ON approval_requests(request_type);

-- =============================================================================
-- Token Blacklist Store
-- =============================================================================

CREATE TABLE IF NOT EXISTS token_blacklist (
    jti TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_blacklist_user ON token_blacklist(user_id);
CREATE INDEX IF NOT EXISTS idx_blacklist_expires ON token_blacklist(expires_at);

-- =============================================================================
-- User Store
-- =============================================================================

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    username TEXT UNIQUE,
    password_hash TEXT,
    role TEXT NOT NULL DEFAULT 'user',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login_at TIMESTAMPTZ,
    profile_json JSONB,
    preferences_json JSONB,
    workspace_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);

-- =============================================================================
-- Webhooks (legacy delivery tracking)
-- =============================================================================

CREATE TABLE IF NOT EXISTS webhooks (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    events_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    secret TEXT,
    active BOOLEAN DEFAULT TRUE,
    user_id TEXT,
    workspace_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_webhooks_user ON webhooks(user_id);
CREATE INDEX IF NOT EXISTS idx_webhooks_active ON webhooks(active);

-- =============================================================================
-- Cleanup Functions
-- =============================================================================

-- Function to clean expired blacklist entries
CREATE OR REPLACE FUNCTION cleanup_expired_blacklist() RETURNS void AS $$
BEGIN
    DELETE FROM token_blacklist WHERE expires_at < NOW();
END;
$$ LANGUAGE plpgsql;

-- Function to clean old completed jobs
CREATE OR REPLACE FUNCTION cleanup_old_jobs(days_old INTEGER DEFAULT 30) RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM job_queue
    WHERE status IN ('completed', 'failed')
    AND completed_at < NOW() - (days_old || ' days')::INTERVAL;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Grant permissions (adjust role names as needed)
-- =============================================================================

-- GRANT ALL ON SCHEMA aragora TO aragora_app;
-- GRANT ALL ON ALL TABLES IN SCHEMA aragora TO aragora_app;
-- GRANT ALL ON ALL SEQUENCES IN SCHEMA aragora TO aragora_app;

-- =============================================================================
-- Schema Version Record
-- =============================================================================

INSERT INTO _schema_versions (module, version, updated_at)
VALUES ('aragora_core', 1, NOW())
ON CONFLICT (module) DO UPDATE SET version = 1, updated_at = NOW();
