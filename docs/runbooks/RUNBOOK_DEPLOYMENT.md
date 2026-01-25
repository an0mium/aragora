# Deployment Operations Runbook

Operational procedures for Aragora deployments.

## Quick Reference

| Target | Health Endpoint | Access |
|--------|-----------------|--------|
| EC2 Staging | `http://{EC2_HOST}:8080/api/health` | Via SSM Run Command |
| EC2 Production | `http://{EC2_HOST}:8080/api/health` | Via SSM Run Command |
| Load Balancer | `https://api.aragora.ai/api/health` | Cloudflare Dashboard |
| Frontend | `https://aragora.ai` | Cloudflare Pages |

## Standard Deployment

### Trigger via GitHub Actions

```bash
# Deploy to all targets (requires AWS OIDC, production requires approval)
gh workflow run deploy-secure.yml

# Deploy to specific target
gh workflow run deploy-secure.yml -f environment=cloudflare
gh workflow run deploy-secure.yml -f environment=ec2-staging
gh workflow run deploy-secure.yml -f environment=ec2-production
```

### Monitor Deployment

```bash
# Watch workflow status
gh run watch

# View deployment logs
gh run view --log
```

## Manual Deployment Procedures

### EC2 Staging

```bash
# 1. SSH to instance
ssh -i ~/.ssh/ec2 ec2-user@$EC2_HOST

# 2. Update code
cd ~/aragora
git stash --include-untracked || true
git pull origin main

# 3. Install dependencies
source venv/bin/activate
pip install -e . --quiet

# 4. Verify import works
python -c "from aragora.server.unified_server import UnifiedServer; print('OK')"

# 5. Restart service (release ports first)
sudo fuser -k 8765/tcp 2>/dev/null || true
sudo fuser -k 8080/tcp 2>/dev/null || true
sudo systemctl restart aragora

# 6. Verify health
curl -sf http://localhost:8080/api/health
```

### Lightsail Production

```bash
# 1. SSH to instance
ssh -i ~/.ssh/lightsail ubuntu@$LIGHTSAIL_HOST

# 2. Update code
cd /home/ubuntu/aragora
git stash --include-untracked || true
git pull origin main

# 3. Install dependencies
source venv/bin/activate
pip install -e . --quiet --no-cache-dir

# 4. Verify import works
python -c "from aragora.server.unified_server import UnifiedServer; print('OK')"

# 5. Restart service
sudo systemctl restart aragora

# 6. Verify health
curl -sf http://localhost:8080/api/health
```

### Cloudflare Pages

```bash
# Build and deploy from local machine
cd aragora/live
npm ci
npm run build:export
npx wrangler pages deploy out --project-name=aragora
```

## Rollback Procedures

### Rollback to Previous Commit

```bash
# 1. Find the commit to rollback to
git log --oneline -10

# 2. SSH to target server
ssh -i ~/.ssh/{target} user@host

# 3. Checkout specific commit
cd ~/aragora
git checkout <commit-hash>

# 4. Reinstall and restart
source venv/bin/activate
pip install -e . --quiet
sudo systemctl restart aragora
```

### Rollback Cloudflare Deployment

```bash
# List recent deployments
npx wrangler pages deployments list --project-name=aragora

# Rollback to specific deployment
npx wrangler pages deployments rollback <deployment-id> --project-name=aragora
```

## Health Check Verification

### Full Health Check

```bash
# EC2 Staging
curl -s http://$EC2_HOST:8080/api/health | jq .

# Lightsail Production
curl -s http://$LIGHTSAIL_HOST:8080/api/health | jq .

# API via Cloudflare LB
curl -s https://api.aragora.ai/api/health | jq .
```

### Expected Healthy Response

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "providers": {
    "anthropic": "available",
    "openai": "available",
    "openrouter": "available"
  }
}
```

## Port Conflicts

### Symptoms

- Service fails to start
- "Address already in use" errors
- Health check times out

### Resolution

```bash
# Find process using port
sudo lsof -i :8765
sudo lsof -i :8080

# Kill process by port
sudo fuser -k 8765/tcp
sudo fuser -k 8080/tcp

# Wait for port release
sleep 2

# Restart service
sudo systemctl start aragora
```

## Service Management

### View Logs

```bash
# Recent logs
sudo journalctl -u aragora -n 100 --no-pager

# Follow logs in real-time
sudo journalctl -u aragora -f

# Logs since last restart
sudo journalctl -u aragora --since "$(systemctl show -p ActiveEnterTimestamp aragora | cut -d= -f2)"
```

### Service Status

```bash
# Check service status
sudo systemctl status aragora

# Check if active
sudo systemctl is-active aragora

# Check if failed
sudo systemctl is-failed aragora
```

### Restart Service

```bash
# Graceful restart
sudo systemctl restart aragora

# Force restart (if stuck)
sudo systemctl kill aragora
sleep 2
sudo systemctl start aragora
```

## SSH Connectivity Issues

### GitHub Actions Can't Reach Host

1. **Verify instance is running**:
   - AWS Console → EC2/Lightsail → Check instance state

2. **Verify security group allows SSH**:
   - Security Groups → Inbound Rules → Port 22 → 0.0.0.0/0

3. **Verify correct IP in secrets**:
   ```bash
   # Update GitHub secret with correct IP
   echo "NEW_IP_ADDRESS" | gh secret set LIGHTSAIL_HOST
   echo "NEW_IP_ADDRESS" | gh secret set EC2_HOST
   ```

4. **Test connectivity locally**:
   ```bash
   ssh-keyscan -H $HOST 2>/dev/null
   ```

## Deployment Checklist

### Pre-deployment

- [ ] Tests pass locally: `pytest tests/ -x --timeout=60`
- [ ] Build succeeds: `npm run build` (for frontend)
- [ ] No uncommitted changes: `git status`
- [ ] On main branch: `git branch --show-current`

### Post-deployment

- [ ] Health check passes on all targets
- [ ] No errors in recent logs
- [ ] Key functionality verified manually
- [ ] Monitoring shows normal metrics

## Incident Response

### Service Down

1. Check health: `curl -sf http://HOST:PORT/api/health`
2. Check logs: `sudo journalctl -u aragora -n 50 --no-pager`
3. Check service: `sudo systemctl status aragora`
4. Restart: `sudo systemctl restart aragora`
5. If still failing, rollback to previous known-good commit

### High Error Rate

1. Check logs for error patterns
2. Check provider status (Anthropic, OpenAI status pages)
3. Enable fallback mode if provider is down
4. Scale up if load-related

### Database Issues

1. Check disk space: `df -h`
2. Check SQLite file: `ls -la /app/data/aragora.db`
3. Backup before any repair: `cp aragora.db aragora.db.backup`
4. Run integrity check: `sqlite3 aragora.db "PRAGMA integrity_check;"`

## Emergency Contacts

| Role | Contact |
|------|---------|
| Primary Oncall | Check PagerDuty/Opsgenie |
| Infrastructure | AWS Console / Cloudflare Dashboard |
| Code Issues | GitHub Issues |

## Related Runbooks

- [Provider Failure](RUNBOOK_PROVIDER_FAILURE.md)
- [Database Issues](RUNBOOK_DATABASE_ISSUES.md)
- [Incident Response](RUNBOOK_INCIDENT.md)
