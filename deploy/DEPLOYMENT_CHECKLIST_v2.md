# Aragora EC2 Deployment Checklist v2

Updated: February 2026 | Includes Research Integration Phase 5+

---

## Pre-Deployment: Infrastructure Requirements

### Compute Resources
- [ ] **Primary Instance**: t3.xlarge or better (4 vCPU, 16GB RAM)
- [ ] **Redis**: r6g.medium (or ElastiCache equivalent)
- [ ] **Database**: db.t3.medium (or RDS equivalent)
- [ ] SSD storage: 100GB minimum
- [ ] EBS backup enabled

### Network Configuration
- [ ] VPC with private subnets configured
- [ ] Security groups for: web (443), app (8080), WS (8765), redis (6379), postgres (5432)
- [ ] NAT Gateway for outbound API calls
- [ ] Application Load Balancer (HTTPS termination)

### AWS Services
- [ ] **Secrets Manager**: Multi-region replication enabled (us-east-1, us-east-2)
- [ ] **Parameter Store**: Configuration parameters stored
- [ ] **CloudWatch**: Log groups created
- [ ] **S3**: Backup bucket configured

---

## Pre-Deployment: Secrets Configuration

### AWS Secrets Manager (Required)
```bash
# Verify secrets exist in both regions
aws secretsmanager describe-secret --secret-id aragora/production/core --region us-east-1
aws secretsmanager describe-secret --secret-id aragora/production/core --region us-east-2
```

- [ ] `aragora/production/core` contains:
  - [ ] `POSTGRES_PASSWORD`
  - [ ] `REDIS_PASSWORD`
  - [ ] `ARAGORA_JWT_SECRET`
  - [ ] `ARAGORA_SECRET_KEY`

- [ ] `aragora/production/ai-providers` contains:
  - [ ] `ANTHROPIC_API_KEY`
  - [ ] `OPENAI_API_KEY` (optional)
  - [ ] `OPENROUTER_API_KEY` (fallback)

- [ ] `aragora/production/connectors` contains:
  - [ ] `COURTLISTENER_API_KEY` (recommended for legal)
  - [ ] `GOVINFO_API_KEY` (free, register at api.data.gov)
  - [ ] `NICE_API_KEY` (clinical guidelines)

- [ ] `aragora/production/premium-legal` (if applicable):
  - [ ] `WESTLAW_API_BASE`, `WESTLAW_API_KEY`
  - [ ] `LEXIS_API_BASE`, `LEXIS_API_KEY`

- [ ] `aragora/production/accounting` (if applicable):
  - [ ] `FASB_API_BASE`, `FASB_API_KEY`
  - [ ] `IRS_API_BASE`, `IRS_API_KEY`

### Claude-Mem Worker Configuration
- [ ] `aragora/production/supermemory` contains:
  - [ ] `SUPERMEMORY_API_KEY`
  - [ ] `SUPERMEMORY_BASE_URL` (default: https://api.supermemory.ai)
  - [ ] `CLAUDE_MEM_SYNC_ENABLED=true`

---

## Pre-Deployment: Code Validation

### Run Tests
```bash
# Full test suite
python -m pytest tests/ -x -q --tb=short

# Critical path tests
python -m pytest tests/server/handlers/memory/ tests/knowledge/ tests/verticals/ -v
```

- [ ] Test suite passes (125,000+ tests; full suite may be slow)
- [ ] No import errors
- [ ] RBAC compliance tests pass
- [ ] Memory handler tests pass

### Connector Smoke Test
```bash
python -c "
from aragora.connectors import CourtListenerConnector, GovInfoConnector
from aragora.connectors.accounting import FASBConnector, IRSConnector
from aragora.connectors.legal import WestlawConnector, LexisConnector

print('Connectors import successfully')
for c in [CourtListenerConnector(), GovInfoConnector()]:
    print(f'  {c.name}: available={c.is_available}')
"
```

- [ ] All connectors instantiate
- [ ] No missing dependencies

### OpenAPI Validation
```bash
python scripts/validate_openapi_spec.py
```

- [ ] OpenAPI spec valid
- [ ] All endpoints documented

---

## Deployment Steps

### Step 1: Prepare EC2 Instance

```bash
# SSH to instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Install prerequisites
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose git

# Add ubuntu to docker group
sudo usermod -aG docker ubuntu
newgrp docker
```

- [ ] Docker installed
- [ ] Docker Compose installed
- [ ] User in docker group

### Step 2: Clone and Configure

```bash
cd /opt
sudo git clone https://github.com/an0mium/aragora.git
sudo chown -R ubuntu:ubuntu aragora
cd aragora

# Fetch secrets from AWS Secrets Manager
./scripts/fetch-aws-secrets.sh production
```

- [ ] Repository cloned
- [ ] Secrets fetched to `.env`

### Step 3: Install Systemd Services

```bash
# Copy service files
sudo cp deploy/systemd/aragora.service /etc/systemd/system/
sudo cp deploy/systemd/claude-mem.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable services
sudo systemctl enable aragora
sudo systemctl enable claude-mem
```

- [ ] aragora.service installed
- [ ] claude-mem.service installed

### Step 4: Build and Start

```bash
# Build images
docker compose build --no-cache

# Run migrations
docker compose run --rm aragora python -m alembic upgrade head

# Start services
sudo systemctl start aragora
sudo systemctl start claude-mem

# Verify
docker compose ps
sudo systemctl status aragora
sudo systemctl status claude-mem
```

- [ ] Build successful
- [ ] Migrations applied
- [ ] Services running

### Step 5: Configure Nginx/ALB

```bash
# If using nginx
sudo cp deploy/nginx-aragora.conf /etc/nginx/sites-available/aragora
sudo ln -s /etc/nginx/sites-available/aragora /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

- [ ] Reverse proxy configured
- [ ] SSL termination working
- [ ] WebSocket upgrade headers set

---

## Post-Deployment Verification

### Health Checks
```bash
# Core health
curl -f http://localhost:8080/healthz
curl -f http://localhost:8080/readyz

# Claude-mem worker
curl -f http://localhost:37777/health

# Database connectivity
curl -f http://localhost:8080/api/v1/status/db

# Redis connectivity
curl -f http://localhost:8080/api/v1/status/redis
```

- [ ] `/healthz` returns 200
- [ ] `/readyz` returns 200
- [ ] Claude-mem worker healthy
- [ ] Database connected
- [ ] Redis connected

### Connector Verification
```bash
# Test CourtListener (public API)
curl -f "http://localhost:8080/api/v1/connectors/courtlistener/status"

# Test GovInfo (public API)
curl -f "http://localhost:8080/api/v1/connectors/govinfo/status"
```

- [ ] CourtListener reachable
- [ ] GovInfo reachable
- [ ] NICE Guidance reachable (if API key configured)

### Memory System Verification
```bash
# Test memory progressive search
curl -X POST http://localhost:8080/api/v1/memory/progressive/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "limit": 5}'

# Test Knowledge Mound health
curl -f http://localhost:8080/api/v1/knowledge/mound/health
```

- [ ] Progressive memory search works
- [ ] Knowledge Mound accessible

### Vertical Specialists Verification
```bash
# Test legal specialist initialization
curl http://localhost:8080/api/v1/verticals/legal/status

# Test accounting specialist initialization
curl http://localhost:8080/api/v1/verticals/accounting/status
```

- [ ] Legal specialist available
- [ ] Accounting specialist available
- [ ] Connectors properly wired

---

## Monitoring Setup

### CloudWatch Alarms
- [ ] CPU utilization > 80%
- [ ] Memory utilization > 85%
- [ ] Disk utilization > 80%
- [ ] HTTP 5xx errors > 10/min
- [ ] Response time p99 > 5s

### Log Groups
- [ ] `/aragora/application` - Application logs
- [ ] `/aragora/access` - Access logs
- [ ] `/aragora/claude-mem` - Claude-mem worker logs
- [ ] `/aragora/connectors` - Connector activity

### Metrics to Monitor
- [ ] Request rate and latency
- [ ] Database connection pool usage
- [ ] Redis memory usage
- [ ] Connector success/failure rates
- [ ] Memory tier distribution

---

## Rollback Procedure

```bash
# 1. Stop services
sudo systemctl stop aragora claude-mem

# 2. Checkout previous version
cd /opt/aragora
git checkout <previous-tag>

# 3. Rebuild
docker compose build

# 4. Rollback migrations (if needed)
docker compose run --rm aragora python -m alembic downgrade -1

# 5. Restart
sudo systemctl start aragora claude-mem

# 6. Verify
curl -f http://localhost:8080/healthz
```

---

## Connector Configuration Reference

### Free/Public APIs (No Key Required)
| Connector | Environment Variable | Notes |
|-----------|---------------------|-------|
| CourtListener | `COURTLISTENER_API_KEY` | Optional, increases rate limits |
| GovInfo | `GOVINFO_API_KEY` | Free key from api.data.gov |
| PubMed | None | Public API |
| ArXiv | None | Public API |

### Licensed APIs (Configuration Required)
| Connector | Environment Variables | Notes |
|-----------|----------------------|-------|
| FASB GAAP | `FASB_API_BASE`, `FASB_API_KEY` | Enterprise license |
| IRS | `IRS_API_BASE`, `IRS_API_KEY` | Internal proxy or license |
| Westlaw | `WESTLAW_API_BASE`, `WESTLAW_API_KEY` | Enterprise license |
| LexisNexis | `LEXIS_API_BASE`, `LEXIS_API_KEY` | Enterprise license |
| NICE | `NICE_API_KEY` | Developer portal registration |

---

## Sign-off

| Check | Verified By | Date | Notes |
|-------|-------------|------|-------|
| Infrastructure ready | | | |
| Secrets configured | | | |
| Tests passing | | | |
| Deployment complete | | | |
| Health checks passing | | | |
| Connectors verified | | | |
| Monitoring configured | | | |

**Deployment Approved:** [ ] Yes / [ ] No

**Deployer:** _________________ **Date:** _________________

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-01-27 | Initial checklist |
| 2.0 | 2026-02-04 | Added connector config, claude-mem, vertical specialists |
