-- Aragora PostgreSQL Schema
-- Consolidated schema for all storage modules
-- Generated for production multi-user deployment
--
-- Usage:
--   psql -U postgres -d aragora -f postgres_schema.sql
--
-- Or via Python:
--   python scripts/init_postgres_db.py
--

-- Create schema namespace
CREATE SCHEMA IF NOT EXISTS aragora;
SET search_path TO aragora, public;

-- Schema version tracking (used by all stores)
CREATE TABLE IF NOT EXISTS _schema_versions (
    module TEXT PRIMARY KEY,
    version INTEGER NOT NULL DEFAULT 1,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- USERS AND ORGANIZATIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    password_salt TEXT NOT NULL,
    name TEXT DEFAULT '',
    org_id TEXT,
    role TEXT DEFAULT 'member',
    is_active BOOLEAN DEFAULT TRUE,
    email_verified BOOLEAN DEFAULT FALSE,
    api_key TEXT,
    api_key_hash TEXT,
    api_key_prefix TEXT,
    api_key_created_at TIMESTAMPTZ,
    api_key_expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_login_at TIMESTAMPTZ,
    mfa_secret TEXT,
    mfa_enabled BOOLEAN DEFAULT FALSE,
    mfa_backup_codes TEXT,
    token_version INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS organizations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    slug TEXT UNIQUE,
    tier TEXT DEFAULT 'free',
    owner_id TEXT REFERENCES users(id),
    stripe_customer_id TEXT,
    stripe_subscription_id TEXT,
    debates_used_this_month INTEGER DEFAULT 0,
    billing_cycle_start TIMESTAMPTZ,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS organization_invitations (
    id TEXT PRIMARY KEY,
    org_id TEXT REFERENCES organizations(id),
    email TEXT NOT NULL,
    role TEXT DEFAULT 'member',
    token TEXT UNIQUE NOT NULL,
    invited_by TEXT REFERENCES users(id),
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    accepted_by TEXT REFERENCES users(id),
    accepted_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS oauth_providers (
    user_id TEXT REFERENCES users(id),
    provider TEXT NOT NULL,
    provider_user_id TEXT NOT NULL,
    access_token TEXT,
    refresh_token TEXT,
    token_expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, provider)
);

CREATE TABLE IF NOT EXISTS login_attempts (
    id SERIAL PRIMARY KEY,
    user_id TEXT REFERENCES users(id),
    email TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    ip_address TEXT,
    user_agent TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_org_id ON users(org_id);
CREATE INDEX IF NOT EXISTS idx_users_api_key_hash ON users(api_key_hash);
CREATE INDEX IF NOT EXISTS idx_org_slug ON organizations(slug);
CREATE INDEX IF NOT EXISTS idx_invitations_token ON organization_invitations(token);

-- ============================================================================
-- AUDIT AND SECURITY
-- ============================================================================

CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    user_id TEXT,
    org_id TEXT,
    action TEXT NOT NULL,
    resource_type TEXT,
    resource_id TEXT,
    old_value JSONB,
    new_value JSONB,
    metadata JSONB,
    ip_address TEXT,
    user_agent TEXT
);

CREATE TABLE IF NOT EXISTS token_blacklist (
    jti TEXT PRIMARY KEY,
    user_id TEXT,
    revoked_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    reason TEXT
);

CREATE TABLE IF NOT EXISTS impersonation_sessions (
    id TEXT PRIMARY KEY,
    admin_user_id TEXT NOT NULL,
    target_user_id TEXT NOT NULL,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    reason TEXT,
    actions JSONB DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_token_blacklist_expires ON token_blacklist(expires_at);

-- ============================================================================
-- GOVERNANCE AND APPROVALS
-- ============================================================================

CREATE TABLE IF NOT EXISTS approval_requests (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    risk_level TEXT DEFAULT 'medium',
    status TEXT DEFAULT 'pending',
    requested_by TEXT,
    requested_at TIMESTAMPTZ DEFAULT NOW(),
    changes JSONB DEFAULT '[]',
    timeout_seconds INTEGER DEFAULT 3600,
    approved_by TEXT,
    approved_at TIMESTAMPTZ,
    rejection_reason TEXT,
    org_id TEXT,
    workspace_id TEXT,
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS verification_records (
    id TEXT PRIMARY KEY,
    claim TEXT NOT NULL,
    claim_type TEXT,
    context TEXT,
    result JSONB,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    verified_by TEXT,
    confidence REAL DEFAULT 0.0,
    proof_tree JSONB,
    org_id TEXT,
    workspace_id TEXT
);

CREATE TABLE IF NOT EXISTS decision_records (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    decision TEXT,
    confidence REAL DEFAULT 0.0,
    participants JSONB DEFAULT '[]',
    votes JSONB DEFAULT '[]',
    reasoning JSONB,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    org_id TEXT,
    workspace_id TEXT
);

CREATE TABLE IF NOT EXISTS rollback_points (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    state JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by TEXT,
    org_id TEXT,
    workspace_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_approval_status ON approval_requests(status);
CREATE INDEX IF NOT EXISTS idx_approval_org ON approval_requests(org_id);
CREATE INDEX IF NOT EXISTS idx_verification_timestamp ON verification_records(timestamp);

-- ============================================================================
-- INTEGRATIONS AND WEBHOOKS
-- ============================================================================

CREATE TABLE IF NOT EXISTS integrations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    config JSONB DEFAULT '{}',
    credentials JSONB DEFAULT '{}',
    status TEXT DEFAULT 'active',
    org_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_sync_at TIMESTAMPTZ,
    sync_status TEXT
);

CREATE TABLE IF NOT EXISTS webhook_configs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    url TEXT NOT NULL,
    events JSONB DEFAULT '[]',
    secret TEXT,
    headers JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    org_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    failure_count INTEGER DEFAULT 0,
    last_failure_at TIMESTAMPTZ,
    last_success_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS webhook_deliveries (
    id TEXT PRIMARY KEY,
    webhook_id TEXT REFERENCES webhook_configs(id),
    event_type TEXT NOT NULL,
    payload JSONB,
    status TEXT DEFAULT 'pending',
    response_code INTEGER,
    response_body TEXT,
    error TEXT,
    attempt_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    delivered_at TIMESTAMPTZ,
    next_retry_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS gmail_tokens (
    user_id TEXT PRIMARY KEY,
    access_token TEXT NOT NULL,
    refresh_token TEXT NOT NULL,
    token_expiry TIMESTAMPTZ,
    scopes JSONB DEFAULT '[]',
    email TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_integrations_org ON integrations(org_id);
CREATE INDEX IF NOT EXISTS idx_integrations_type ON integrations(type);
CREATE INDEX IF NOT EXISTS idx_webhook_configs_org ON webhook_configs(org_id);
CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_status ON webhook_deliveries(status);

-- ============================================================================
-- MARKETPLACE
-- ============================================================================

CREATE TABLE IF NOT EXISTS marketplace_templates (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT NOT NULL,
    author_id TEXT NOT NULL,
    author_name TEXT NOT NULL,
    category TEXT NOT NULL,
    pattern TEXT NOT NULL,
    tags JSONB DEFAULT '[]',
    workflow_definition JSONB DEFAULT '{}',
    download_count INTEGER DEFAULT 0,
    rating_sum REAL DEFAULT 0.0,
    rating_count INTEGER DEFAULT 0,
    is_featured BOOLEAN DEFAULT FALSE,
    is_trending BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS marketplace_reviews (
    id TEXT PRIMARY KEY,
    template_id TEXT REFERENCES marketplace_templates(id),
    user_id TEXT NOT NULL,
    user_name TEXT NOT NULL,
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    title TEXT,
    content TEXT,
    helpful_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_templates_category ON marketplace_templates(category);
CREATE INDEX IF NOT EXISTS idx_templates_author ON marketplace_templates(author_id);
CREATE INDEX IF NOT EXISTS idx_reviews_template ON marketplace_reviews(template_id);

-- ============================================================================
-- JOB QUEUE
-- ============================================================================

CREATE TABLE IF NOT EXISTS job_queue (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    payload JSONB NOT NULL,
    status TEXT DEFAULT 'pending',
    priority INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    retry_count INTEGER DEFAULT 0,
    scheduled_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error TEXT,
    result JSONB,
    org_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_job_status ON job_queue(status);
CREATE INDEX IF NOT EXISTS idx_job_scheduled ON job_queue(scheduled_at);
CREATE INDEX IF NOT EXISTS idx_job_type ON job_queue(type);

-- ============================================================================
-- FEDERATION
-- ============================================================================

CREATE TABLE IF NOT EXISTS federation_peers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    public_key TEXT,
    status TEXT DEFAULT 'pending',
    trust_level REAL DEFAULT 0.5,
    capabilities JSONB DEFAULT '[]',
    last_seen_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS federation_messages (
    id TEXT PRIMARY KEY,
    peer_id TEXT REFERENCES federation_peers(id),
    direction TEXT NOT NULL,
    message_type TEXT NOT NULL,
    payload JSONB,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_federation_peers_status ON federation_peers(status);
CREATE INDEX IF NOT EXISTS idx_federation_messages_peer ON federation_messages(peer_id);

-- ============================================================================
-- GAUNTLET (TESTING FRAMEWORK)
-- ============================================================================

CREATE TABLE IF NOT EXISTS gauntlet_runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    config JSONB DEFAULT '{}',
    status TEXT DEFAULT 'pending',
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    results JSONB DEFAULT '{}',
    metrics JSONB DEFAULT '{}',
    org_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS gauntlet_cases (
    id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES gauntlet_runs(id),
    name TEXT NOT NULL,
    input JSONB,
    expected_output JSONB,
    actual_output JSONB,
    status TEXT DEFAULT 'pending',
    score REAL,
    duration_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_gauntlet_runs_status ON gauntlet_runs(status);
CREATE INDEX IF NOT EXISTS idx_gauntlet_cases_run ON gauntlet_cases(run_id);

-- ============================================================================
-- FINDING WORKFLOWS
-- ============================================================================

CREATE TABLE IF NOT EXISTS finding_workflows (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    trigger_type TEXT NOT NULL,
    trigger_config JSONB DEFAULT '{}',
    actions JSONB DEFAULT '[]',
    is_active BOOLEAN DEFAULT TRUE,
    org_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS finding_workflow_runs (
    id TEXT PRIMARY KEY,
    workflow_id TEXT REFERENCES finding_workflows(id),
    trigger_data JSONB,
    status TEXT DEFAULT 'pending',
    results JSONB DEFAULT '{}',
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_finding_workflows_org ON finding_workflows(org_id);
CREATE INDEX IF NOT EXISTS idx_finding_workflow_runs_workflow ON finding_workflow_runs(workflow_id);

-- ============================================================================
-- NOTIFICATIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS notification_configs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    event_types JSONB DEFAULT '[]',
    is_enabled BOOLEAN DEFAULT TRUE,
    config JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS notifications (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    type TEXT NOT NULL,
    title TEXT NOT NULL,
    message TEXT,
    data JSONB DEFAULT '{}',
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    read_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_notifications_unread ON notifications(user_id, is_read) WHERE NOT is_read;

-- ============================================================================
-- SHARING
-- ============================================================================

CREATE TABLE IF NOT EXISTS shares (
    id TEXT PRIMARY KEY,
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    shared_by TEXT NOT NULL,
    shared_with TEXT,
    share_type TEXT DEFAULT 'link',
    permissions JSONB DEFAULT '["read"]',
    token TEXT UNIQUE,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    accessed_at TIMESTAMPTZ,
    access_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_shares_resource ON shares(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_shares_token ON shares(token);

-- ============================================================================
-- DECISION RESULTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS decision_results (
    id TEXT PRIMARY KEY,
    debate_id TEXT,
    question TEXT NOT NULL,
    decision TEXT,
    confidence REAL DEFAULT 0.0,
    reasoning TEXT,
    participants JSONB DEFAULT '[]',
    votes JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    org_id TEXT,
    workspace_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_decision_results_debate ON decision_results(debate_id);
CREATE INDEX IF NOT EXISTS idx_decision_results_org ON decision_results(org_id);

-- ============================================================================
-- FINALIZE
-- ============================================================================

-- Insert initial schema versions
INSERT INTO _schema_versions (module, version, updated_at)
VALUES
    ('users', 1, NOW()),
    ('organizations', 1, NOW()),
    ('audit', 1, NOW()),
    ('governance', 1, NOW()),
    ('integrations', 1, NOW()),
    ('webhooks', 1, NOW()),
    ('marketplace', 1, NOW()),
    ('job_queue', 1, NOW()),
    ('federation', 1, NOW()),
    ('gauntlet', 1, NOW()),
    ('finding_workflows', 1, NOW()),
    ('notifications', 1, NOW()),
    ('shares', 1, NOW()),
    ('decisions', 1, NOW())
ON CONFLICT (module) DO NOTHING;

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL ON SCHEMA aragora TO aragora_app;
-- GRANT ALL ON ALL TABLES IN SCHEMA aragora TO aragora_app;
-- GRANT ALL ON ALL SEQUENCES IN SCHEMA aragora TO aragora_app;
