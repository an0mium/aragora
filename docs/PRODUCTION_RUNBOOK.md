# Aragora Production Runbook

This runbook provides operational procedures for managing the Aragora production environment.

## Infrastructure Overview

| Component | Service | Region | Notes |
|-----------|---------|--------|-------|
| API Server #1 | EC2 `i-0dbd51f74a9a11fcc` | us-east-2 | aragora-api-server |
| API Server #2 | EC2 `i-016b3e32625bf967e` | us-east-2 | aragora-api-2 |
| Database | Supabase PostgreSQL | - | Transaction pooler mode |
| Cache | Upstash Redis | us-east-2 | TLS enabled |
| CDN/WAF | Cloudflare | - | SSL termination, load balancing |
| Secrets | AWS Secrets Manager | us-east-2 | `aragora/production` |
| Monitoring | CloudWatch | us-east-2 | CPU and status alarms |

## Common Operations

### Check Service Status

```bash
# Via AWS SSM (recommended)
aws ssm send-command \
  --instance-ids "i-0dbd51f74a9a11fcc" \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["systemctl status aragora --no-pager"]' \
  --output text --query "Command.CommandId"

# Get command result
aws ssm get-command-invocation --command-id "<COMMAND_ID>" --instance-id "i-0dbd51f74a9a11fcc"
```

### Check Health Endpoint

```bash
# External (via Cloudflare)
curl -s https://api.aragora.ai/api/health | jq

# Internal (via SSM)
aws ssm send-command \
  --instance-ids "i-0dbd51f74a9a11fcc" \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["curl -s http://localhost:8080/api/health | jq"]' \
  --output text --query "Command.CommandId"
```

### View Service Logs

```bash
# Recent logs (last 100 lines)
aws ssm send-command \
  --instance-ids "i-0dbd51f74a9a11fcc" \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["sudo journalctl -u aragora -n 100 --no-pager"]' \
  --output text --query "Command.CommandId"

# Follow logs (for debugging)
aws ssm send-command \
  --instance-ids "i-0dbd51f74a9a11fcc" \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["sudo journalctl -u aragora -f --no-pager | head -200"]' \
  --output text --query "Command.CommandId"
```

### Restart Service

```bash
aws ssm send-command \
  --instance-ids "i-0dbd51f74a9a11fcc" \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["sudo systemctl restart aragora && sleep 5 && systemctl status aragora --no-pager | head -15"]' \
  --output text --query "Command.CommandId"
```

### Deploy Updates

```bash
# Manual deploy to single instance
aws ssm send-command \
  --instance-ids "i-0dbd51f74a9a11fcc" \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=[
    "export HOME=/home/ec2-user",
    "cd /home/ec2-user/aragora",
    "git config --global --add safe.directory /home/ec2-user/aragora",
    "git pull origin main",
    "source venv/bin/activate",
    "pip install -e . --quiet",
    "sudo systemctl restart aragora",
    "sleep 5",
    "curl -s http://localhost:8080/api/health | jq .status"
  ]' \
  --output text --query "Command.CommandId"

# Deploy to both instances
aws ssm send-command \
  --instance-ids "i-0dbd51f74a9a11fcc" "i-016b3e32625bf967e" \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=[...]'
```

## Authentication

### Generate API Token

```bash
# On the server via SSM
aws ssm send-command \
  --instance-ids "i-0dbd51f74a9a11fcc" \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=[
    "export HOME=/home/ec2-user",
    "cd /home/ec2-user/aragora",
    "source venv/bin/activate",
    "export $(grep Environment /etc/systemd/system/aragora.service.d/env.conf | sed -e \"s/Environment=//g\" -e \"s/\\\"//g\" | tr \" \" \"\\n\" | grep ARAGORA_API_TOKEN) 2>/dev/null",
    "python3 -c \"from aragora.server.auth import auth_config; auth_config.configure_from_env(); print(auth_config.generate_token('admin', 3600))\""
  ]' \
  --output text --query "Command.CommandId"
```

Token format: `{loop_id}:{expires_timestamp}:{hmac_signature}`

### Test Authenticated Request

```bash
TOKEN="<your_token>"
curl -s -X POST "https://api.aragora.ai/api/debates" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{"question":"Test question","config":{"rounds":1}}'
```

## Database Operations

### Check PostgreSQL Connection

```bash
aws ssm send-command \
  --instance-ids "i-0dbd51f74a9a11fcc" \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["curl -s http://localhost:8080/api/health | jq .checks.database"]' \
  --output text --query "Command.CommandId"
```

### Initialize PostgreSQL Stores

```bash
aws ssm send-command \
  --instance-ids "i-0dbd51f74a9a11fcc" \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=[
    "export HOME=/home/ec2-user",
    "cd /home/ec2-user/aragora",
    "source venv/bin/activate",
    "export ARAGORA_USE_SECRETS_MANAGER=true",
    "python scripts/init_postgres_db.py"
  ]' \
  --output text --query "Command.CommandId"
```

### Supabase Dashboard

- URL: https://supabase.com/dashboard
- Backups: Settings > Database > Backups
- Connection pooling: Settings > Database > Connection pooling

## Redis Operations

### Check Redis Connection

```bash
aws ssm send-command \
  --instance-ids "i-0dbd51f74a9a11fcc" \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["curl -s http://localhost:8080/api/health | jq .checks.redis"]' \
  --output text --query "Command.CommandId"
```

### Redis Configuration

- Service: Upstash Redis
- URL stored in: AWS Secrets Manager (`aragora/production`)
- Environment variable: `REDIS_URL` (with `rediss://` for TLS)

## Monitoring & Alerts

### CloudWatch Alarms

| Alarm Name | Metric | Threshold | Action |
|------------|--------|-----------|--------|
| `aragora-api-server-cpu-high` | CPUUtilization > 80% | 5 min | Email alert |
| `aragora-api-server-status-check` | StatusCheckFailed | 1 | Email alert |
| `aragora-api-2-cpu-high` | CPUUtilization > 80% | 5 min | Email alert |
| `aragora-api-2-status-check` | StatusCheckFailed | 1 | Email alert |

### Check Alarm Status

```bash
aws cloudwatch describe-alarms --alarm-names \
  "aragora-api-server-cpu-high" \
  "aragora-api-server-status-check" \
  "aragora-api-2-cpu-high" \
  "aragora-api-2-status-check" \
  --query 'MetricAlarms[*].[AlarmName,StateValue]' \
  --output table
```

## Incident Response

### Service Down

1. Check health endpoint: `curl https://api.aragora.ai/api/health`
2. Check CloudWatch alarms
3. Check service status via SSM
4. View service logs for errors
5. Restart service if needed
6. Verify health after restart

### High CPU Usage

1. Check active debates: Review recent debate creation
2. Check for runaway processes via SSM
3. Consider scaling horizontally (add instances)
4. Review rate limiting settings

### Database Connection Issues

1. Check Supabase status page
2. Verify connection pooler settings
3. Check if IP is whitelisted (IPv6 required)
4. Review connection pool size

### Authentication Failures

1. Verify token format: `{loop_id}:{expires}:{signature}`
2. Check token expiration
3. Regenerate token if needed
4. Verify `ARAGORA_API_TOKEN` in secrets

## Maintenance Windows

### Recommended Maintenance Time

- **Best time**: UTC 06:00-08:00 (low traffic)
- **Avoid**: UTC 14:00-22:00 (peak usage)

### Pre-Maintenance Checklist

- [ ] Notify stakeholders
- [ ] Create backup/checkpoint
- [ ] Verify rollback procedure
- [ ] Have monitoring dashboards open

### Post-Maintenance Checklist

- [ ] Verify health endpoints
- [ ] Test debate creation
- [ ] Check CloudWatch metrics
- [ ] Confirm no error spikes in logs

## Secrets Management

### View Secret Keys (not values)

```bash
aws secretsmanager get-secret-value \
  --secret-id aragora/production \
  --query SecretString --output text | jq 'keys'
```

### Update Secret

```bash
# Get current secret
aws secretsmanager get-secret-value \
  --secret-id aragora/production \
  --query SecretString --output text > /tmp/secret.json

# Edit /tmp/secret.json

# Update secret
aws secretsmanager put-secret-value \
  --secret-id aragora/production \
  --secret-string file:///tmp/secret.json

# Restart services to pick up new secrets
# (Services read secrets at startup)
```

## SSL Certificates

- **Managed by**: Cloudflare
- **Auto-renewal**: Yes (30 days before expiration)
- **Current expiry**: April 2026

### Check Certificate

```bash
echo | openssl s_client -servername api.aragora.ai -connect api.aragora.ai:443 2>/dev/null | openssl x509 -noout -dates
```

## Contact Information

- **Infrastructure**: [Your team/contact]
- **On-call**: [Rotation/schedule]
- **Escalation**: [Escalation path]

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-21 | 1.0 | Initial production runbook |
