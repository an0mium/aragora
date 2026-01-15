# Database Migration Guide: SQLite to PostgreSQL

This guide covers migrating Aragora from SQLite to PostgreSQL for multi-instance deployment.

## Prerequisites

**EC2 packages installed:**
- asyncpg 0.31.0
- psycopg2-binary 2.9.11
- SQLAlchemy 2.0.45
- alembic 1.18.1

## Step 1: Provision PostgreSQL

### Option A: Supabase (Recommended)
1. Go to https://supabase.com
2. Create new project
3. Copy connection string from Settings → Database → Connection string
4. Format: `postgresql://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres`

### Option B: AWS RDS
1. Go to AWS Console → RDS → Create database
2. Choose PostgreSQL 15+
3. Instance: db.t3.micro (free tier eligible)
4. Note endpoint and credentials

## Step 2: Test Connection

```bash
# SSH into EC2
ssh ec2-user@3.141.158.91

# Activate venv
source ~/aragora/venv/bin/activate

# Test connection
python3 << 'EOF'
import asyncio
import asyncpg

async def test():
    conn = await asyncpg.connect('postgresql://USER:PASS@HOST:5432/DB')
    version = await conn.fetchval('SELECT version()')
    print(f'Connected: {version[:50]}...')
    await conn.close()

asyncio.run(test())
EOF
```

## Step 3: Run Migrations

```bash
cd ~/aragora

# Set database URL
export DATABASE_URL="postgresql://USER:PASS@HOST:5432/DB"

# Run alembic migrations (if available)
alembic upgrade head

# Or use built-in migration
python3 -c "
from aragora.persistence.database import init_database
import asyncio
asyncio.run(init_database('$DATABASE_URL'))
print('Database initialized')
"
```

## Step 4: Update EC2 Environment

```bash
# Add to .env file
echo 'DATABASE_URL=postgresql://USER:PASS@HOST:5432/DB' >> ~/aragora/.env

# Or add to systemd service
sudo mkdir -p /etc/systemd/system/aragora.service.d/
sudo tee /etc/systemd/system/aragora.service.d/database.conf << EOF
[Service]
Environment="DATABASE_URL=postgresql://USER:PASS@HOST:5432/DB"
EOF

sudo systemctl daemon-reload
sudo systemctl restart aragora
```

## Step 5: Verify Migration

```bash
# Check health endpoint shows database connected
curl http://localhost:8080/api/health | jq '.checks.database'

# Verify debates persist
curl -X POST http://localhost:8080/api/debates \
  -H "Content-Type: application/json" \
  -d '{"question": "Migration test", "rounds": 1}'

# Restart service and check debate still exists
sudo systemctl restart aragora
curl http://localhost:8080/api/debates
```

## Step 6: Multi-Instance Setup

Once PostgreSQL is working on EC2 #1:

1. Launch EC2 #2 from AMI snapshot
2. Configure same `DATABASE_URL` on EC2 #2
3. Update Cloudflare tunnel to include both origins
4. Verify both instances share data:
   - Create debate on EC2 #1
   - Retrieve on EC2 #2

## Rollback Procedure

If migration fails:

```bash
# Remove DATABASE_URL from environment
sudo rm /etc/systemd/system/aragora.service.d/database.conf
sudo systemctl daemon-reload

# Restart with SQLite (default)
sudo systemctl restart aragora

# Verify SQLite working
curl http://localhost:8080/api/health
```

## Security Notes

- Never commit DATABASE_URL to git
- Use GitHub secrets for CI/CD
- Enable SSL for PostgreSQL connections in production
- Rotate credentials periodically
