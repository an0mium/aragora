# Supabase Setup Checklist for aragora.ai

Complete checklist for configuring Supabase as the production database for aragora.ai.

## Prerequisites

- [ ] Supabase account created at [supabase.com](https://supabase.com)
- [ ] Supabase project created (free tier works for getting started)
- [ ] Access to aragora.ai server (EC2 or local)

## Step 1: Get Supabase Credentials

### 1.1 API Credentials (Settings > API)

```bash
# Project URL
SUPABASE_URL=https://[your-project-ref].supabase.co

# Service Role Key (server-side, never expose to client)
SUPABASE_KEY=[your-service-role-key]

# Anon Key (client-side, for frontend if needed)
NEXT_PUBLIC_SUPABASE_ANON_KEY=[your-anon-key]
```

### 1.2 Database Connection String (Settings > Database)

1. Go to **Settings > Database > Connection string**
2. Select **Transaction** pooler mode (recommended for serverless)
3. Copy the connection string:

```bash
# Format
ARAGORA_POSTGRES_DSN=postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres

# Example
ARAGORA_POSTGRES_DSN=postgresql://postgres.abcdefghijklmnop:MySecurePassword123@aws-0-us-west-1.pooler.supabase.com:6543/postgres
```

## Step 2: Configure EC2/Server Environment

### 2.1 AWS Secrets Manager (Recommended)

Store secrets in AWS Secrets Manager:

```bash
aws secretsmanager create-secret \
  --name aragora/production \
  --secret-string '{
    "SUPABASE_URL": "https://xxx.supabase.co",
    "SUPABASE_KEY": "your-service-role-key",
    "ARAGORA_POSTGRES_DSN": "postgresql://...",
    "ANTHROPIC_API_KEY": "sk-ant-...",
    "OPENAI_API_KEY": "sk-...",
    "ARAGORA_JWT_SECRET": "your-jwt-secret"
  }'
```

### 2.2 Systemd Environment Override

Or set via systemd override (already configured in deploy workflow):

```bash
sudo mkdir -p /etc/systemd/system/aragora.service.d/

cat << 'EOF' | sudo tee /etc/systemd/system/aragora.service.d/supabase.conf
[Service]
Environment="SUPABASE_URL=https://xxx.supabase.co"
Environment="SUPABASE_KEY=your-service-role-key"
Environment="ARAGORA_POSTGRES_DSN=postgresql://..."
EOF

sudo systemctl daemon-reload
sudo systemctl restart aragora
```

## Step 3: Initialize Database

Run the initialization script to create all tables:

```bash
# SSH into server
ssh ec2-user@your-server

# Activate virtual environment
cd ~/aragora
source venv/bin/activate

# Initialize PostgreSQL stores
python scripts/init_postgres_db.py

# Verify tables exist
python scripts/init_postgres_db.py --verify
```

Expected output:
```
2026-01-20 22:30:00 - INFO - Connected to PostgreSQL
2026-01-20 22:30:00 - INFO - Initialized webhook_configs
2026-01-20 22:30:00 - INFO - Initialized integrations
...
2026-01-20 22:30:01 - INFO - Successfully initialized 13 stores
```

## Step 4: Validate Configuration

Run the production validation script:

```bash
python scripts/validate_production.py --fix
```

Expected output:
```
============================================================
ARAGORA.AI PRODUCTION VALIDATION
============================================================

✓ [OK  ] ARAGORA_ENVIRONMENT
          production
✓ [OK  ] ARAGORA_JWT_SECRET
          Set (43 chars)
✓ [OK  ] Supabase
          All credentials configured
✓ [OK  ] PostgreSQL Connection
          Connected (PostgreSQL 15.1)
✓ [OK  ] Database Tables
          13 tables verified
✓ [OK  ] AI Providers
          Configured: Anthropic, OpenAI
⚠ [WARN] CORS
          Localhost in allowed origins (OK for staging)
○ [SKIP] SSL/TLS
          Not enabled (OK if behind reverse proxy)

============================================================
SUMMARY: 6 OK, 1 warnings, 0 failures
============================================================
```

## Step 5: GitHub Secrets (for CI/CD)

Set these secrets in your GitHub repository (Settings > Secrets > Actions):

| Secret Name | Value |
|-------------|-------|
| `SUPABASE_URL` | `https://xxx.supabase.co` |
| `SUPABASE_KEY` | `your-service-role-key` |
| `NEXT_PUBLIC_SUPABASE_URL` | Same as SUPABASE_URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Your anon key |
| `ARAGORA_POSTGRES_DSN` | PostgreSQL connection string |

## Step 6: Verify Deployment

After deployment, verify the setup:

```bash
# Check health endpoint
curl https://api.aragora.ai/api/health | jq

# Expected response includes
{
  "status": "healthy",
  "checks": {
    "database": "connected",
    ...
  }
}
```

## Supabase Dashboard Features

Once configured, you can use these Supabase features:

### Real-time Subscriptions
The SupabaseClient in `aragora/persistence/supabase_client.py` already supports:
- `nomic_cycles` table
- `debate_artifacts` table
- `stream_events` table
- `agent_metrics` table

### Database Management
- **Table Editor**: View/edit data directly
- **SQL Editor**: Run ad-hoc queries
- **Backups**: Automatic daily backups (Pro plan: point-in-time recovery)

### Monitoring
- **Database Health**: Connection count, latency
- **API Logs**: Request/response logs
- **Realtime Inspector**: Debug subscriptions

## Troubleshooting

### Connection Refused
```
Error: could not connect to server
```
**Fix**: Check if your IP is allowed in Supabase (Settings > Database > Connection Pooling)

### Authentication Failed
```
Error: password authentication failed
```
**Fix**: Verify the password in your connection string matches Supabase dashboard

### Tables Missing
```
Missing tables: webhook_configs, integrations, ...
```
**Fix**: Run `python scripts/init_postgres_db.py`

### Connection Pool Exhausted
```
Error: too many connections
```
**Fix**: Use Transaction pooler mode, reduce `ARAGORA_DB_POOL_SIZE`

## Connection Limits by Supabase Tier

| Tier | Direct Connections | Pooler Connections |
|------|-------------------|-------------------|
| Free | 20 | 200 |
| Pro | 60 | 400 |
| Team | 200 | 1500 |

Aragora default pool size is 10, well within limits.

## Next Steps

After Supabase is configured:

1. [ ] Test debate creation and storage
2. [ ] Verify webhook configurations save
3. [ ] Check user authentication flow
4. [ ] Monitor connection pool usage in Supabase dashboard
5. [ ] Set up Grafana dashboards for database metrics
