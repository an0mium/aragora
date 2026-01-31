-- Initialize Aragora database for OpenClaw gateway

-- Audit trail table
CREATE TABLE IF NOT EXISTS audit_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    record_id VARCHAR(255) NOT NULL UNIQUE,
    event_type VARCHAR(100) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source VARCHAR(100) NOT NULL,
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    tenant_id VARCHAR(255),
    action_type VARCHAR(100),
    action_target TEXT,
    success BOOLEAN DEFAULT true,
    error TEXT,
    signature VARCHAR(64),
    previous_hash VARCHAR(64),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_audit_user_id ON audit_records(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_session_id ON audit_records(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_tenant_id ON audit_records(tenant_id);
CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_records(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_records(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_action_type ON audit_records(action_type);

-- Sessions table
CREATE TABLE IF NOT EXISTS proxy_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL UNIQUE,
    user_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    workspace_id VARCHAR(255) NOT NULL,
    roles TEXT[] DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'active',
    action_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_activity TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON proxy_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_tenant_id ON proxy_sessions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON proxy_sessions(status);

-- Pending approvals table
CREATE TABLE IF NOT EXISTS pending_approvals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    approval_id VARCHAR(255) NOT NULL UNIQUE,
    session_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    action_type VARCHAR(100) NOT NULL,
    action_params JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    approver_id VARCHAR(255),
    approved_at TIMESTAMPTZ,
    reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_approvals_tenant_id ON pending_approvals(tenant_id);
CREATE INDEX IF NOT EXISTS idx_approvals_status ON pending_approvals(status);
CREATE INDEX IF NOT EXISTS idx_approvals_expires_at ON pending_approvals(expires_at);

-- Policy rules table (for dynamic policy updates)
CREATE TABLE IF NOT EXISTS policy_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255),
    action_types TEXT[] NOT NULL,
    decision VARCHAR(50) NOT NULL,
    priority INTEGER DEFAULT 0,
    config JSONB NOT NULL,
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(255)
);

CREATE INDEX IF NOT EXISTS idx_policy_tenant_id ON policy_rules(tenant_id);
CREATE INDEX IF NOT EXISTS idx_policy_enabled ON policy_rules(enabled);

-- Rate limit tracking (in-memory alternative for persistence)
CREATE TABLE IF NOT EXISTS rate_limit_counters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key VARCHAR(500) NOT NULL,
    count INTEGER DEFAULT 0,
    window_start TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    window_seconds INTEGER NOT NULL DEFAULT 60,
    UNIQUE(key)
);

CREATE INDEX IF NOT EXISTS idx_rate_limit_key ON rate_limit_counters(key);

-- Function to clean up expired data
CREATE OR REPLACE FUNCTION cleanup_expired_data() RETURNS void AS $$
BEGIN
    -- Delete expired approvals
    DELETE FROM pending_approvals
    WHERE status = 'pending' AND expires_at < NOW();

    -- Delete old rate limit counters
    DELETE FROM rate_limit_counters
    WHERE window_start + (window_seconds || ' seconds')::interval < NOW();

    -- Archive old audit records (optional - uncomment if needed)
    -- DELETE FROM audit_records WHERE timestamp < NOW() - INTERVAL '90 days';
END;
$$ LANGUAGE plpgsql;

-- Trigger for automatic cleanup (runs hourly via pg_cron if installed)
-- SELECT cron.schedule('cleanup-expired', '0 * * * *', 'SELECT cleanup_expired_data()');
