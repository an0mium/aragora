# Cloudflare Load Balancer Setup for Aragora

This document describes how to configure Cloudflare Load Balancer for `api.aragora.ai`.

## Architecture Overview

```
                         ┌─────────────────┐
                         │  Cloudflare LB  │
                         │ api.aragora.ai  │
                         │  (HTTP mode)    │
                         └────────┬────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    us-east-2    │     │    us-east-2    │     │    eu-west-1    │
│     Primary     │     │    Secondary    │     │       DR        │
│   Weight: 1.0   │     │   Weight: 1.0   │     │   Weight: 0.5   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
       nginx                  nginx                   nginx
        :80                    :80                     :80
```

### Multi-Region Deployment

| Instance | Region | Role | Weight | Purpose |
|----------|--------|------|--------|---------|
| ec2-primary | us-east-2 | Primary | 1.0 | Main traffic handler |
| ec2-secondary | us-east-2 | Secondary | 1.0 | Load distribution |
| ec2-dr-eu | eu-west-1 | DR | 0.5 | Disaster recovery |

The DR instance provides resilience against single-region AWS outages.

**Key Points:**
- Cloudflare handles TLS termination (proxied mode)
- Origin servers run nginx on port 80 (HTTP-only)
- Health checks route to `/api/health`
- WebSocket connections use session affinity

## Prerequisites

### 1. AWS Security Group Configuration

Each EC2 instance must allow inbound traffic from Cloudflare IPs:

**Inbound Rules:**
| Type | Port | Source |
|------|------|--------|
| HTTP | 80 | Cloudflare IP ranges |
| HTTPS | 443 | Cloudflare IP ranges (optional) |
| SSH | 22 | Your IP / GitHub Actions |

**Cloudflare IP Ranges:**
- IPv4: https://www.cloudflare.com/ips-v4
- IPv6: https://www.cloudflare.com/ips-v6

Or use AWS-managed prefix list `com.amazonaws.global.cloudfront.origin-facing` if available.

### 2. EC2 Instance Setup

Each EC2 must have:
- nginx installed and configured (use `deploy/nginx-aragora-http.conf`)
- aragora service running on port 8080
- WebSocket server running on port 8765

Verify locally:
```bash
curl http://localhost/api/health
curl http://localhost:8080/api/health
```

## Cloudflare Dashboard Configuration

### Step 1: Create Origin Pool

1. Go to **Traffic > Load Balancing > Pools**
2. Click **Create Pool**
3. Configure:

| Setting | Value |
|---------|-------|
| Pool Name | `aragora-api-pool` |
| Description | EC2 origins for api.aragora.ai |
| Origin Steering | Failover (recommended) |

### Step 2: Add Origins

For each EC2 instance:

| Setting | Value |
|---------|-------|
| Origin Name | `ec2-primary` (or `ec2-secondary`) |
| Origin Address | EC2 public IP (e.g., `3.15.x.x`) |
| Port | 80 |
| Weight | 1 |
| Host Header | `api.aragora.ai` |

### Step 3: Configure Health Check

| Setting | Value |
|---------|-------|
| Monitor | Create new |
| Type | HTTP |
| Path | `/api/health` |
| Port | 80 |
| Expected Codes | 200 |
| Interval | 60 seconds |
| Retries | 2 |
| Timeout | 5 seconds |
| Follow Redirects | Yes |

### Step 4: Create Load Balancer

1. Go to **Traffic > Load Balancing > Load Balancers**
2. Click **Create Load Balancer**
3. Configure:

| Setting | Value |
|---------|-------|
| Hostname | `api.aragora.ai` |
| Session Affinity | IP Cookie (for WebSocket) |
| Session TTL | 1800 (30 minutes) |
| Fallback Pool | (optional secondary pool) |
| Proxied | Yes (orange cloud) |

### Step 5: Configure Steering Policy

| Policy | Use Case |
|--------|----------|
| Failover | Primary with automatic failover to secondary |
| Round Robin | Equal distribution across all healthy origins |
| Random | Random selection weighted by origin weight |

**Recommended:** Failover for critical API with clear primary.

## DNS Configuration

Ensure the Load Balancer manages the DNS record:

| Type | Name | Content | Proxy Status |
|------|------|---------|--------------|
| CNAME | api | (Load Balancer) | Proxied |

The Load Balancer will automatically create/manage this record.

## WebSocket Configuration

WebSocket connections require special handling:

1. **Session Affinity:** Enable IP Cookie affinity in Load Balancer settings
2. **Timeout:** Cloudflare has a 100-second WebSocket idle timeout by default
   - Enterprise plans can increase this
   - Application should send periodic pings (every 30s)

3. **nginx Configuration:** Ensure WebSocket upgrade headers are set:
```nginx
location /ws {
    proxy_pass http://127.0.0.1:8765;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 86400;
}
```

## Verification

### Check Load Balancer Health

In Cloudflare Dashboard:
1. Go to **Traffic > Load Balancing > Pools**
2. View pool health status (green = healthy)

### Test API Endpoint

```bash
# Should return 200 with JSON health response
curl -v https://api.aragora.ai/api/health

# Check response includes origin info
curl -s https://api.aragora.ai/api/health | jq .
```

### Test WebSocket

```bash
# Should return 426 Upgrade Required (expected for non-WS client)
curl -I https://api.aragora.ai/ws

# Or use wscat for full WebSocket test
npx wscat -c wss://api.aragora.ai/ws
```

### Check Origin Routing

The `/api/health` response includes the server's nomic_dir path:
- `/home/ec2-user/...` = EC2 origin
- `/home/ubuntu/...` = Lightsail origin (deprecated)

## Troubleshooting

### 521 Error (Web Server Down)
- Check EC2 security group allows Cloudflare IPs on port 80
- Verify nginx is running: `sudo systemctl status nginx`
- Verify aragora service is running: `sudo systemctl status aragora`

### 522 Error (Connection Timed Out)
- Check EC2 instance is running
- Check security group allows inbound on port 80
- Verify nginx is listening: `sudo netstat -tlnp | grep :80`

### 523 Error (Origin Unreachable)
- DNS resolution issue - verify EC2 public IP is correct
- Check if EC2 elastic IP changed

### WebSocket Disconnects
- Increase nginx `proxy_read_timeout` and `proxy_send_timeout`
- Implement application-level ping/pong (30s interval)
- Check Cloudflare WebSocket timeout settings (Enterprise)

## Adding a New EC2 Origin

### Using Terraform (Recommended)

```bash
cd deploy/terraform/ec2-multiregion
terraform workspace new <region>-<role>
terraform apply -var="region=<region>" -var="role=<role>"
```

See `deploy/terraform/ec2-multiregion/README.md` for full instructions.

### Manual Setup

1. Launch new EC2 with Amazon Linux 2023
2. Run bootstrap script:
   ```bash
   curl -O https://raw.githubusercontent.com/aragora/aragora/main/deploy/scripts/al2023-bootstrap.sh
   chmod +x al2023-bootstrap.sh
   sudo ./al2023-bootstrap.sh production <role> <region>
   ```
3. Configure secrets: `sudo nano /etc/aragora/env`
4. Start services: `sudo systemctl start aragora aragora-ws`
5. Add EC2 IP to security group for Cloudflare IPs
6. Add origin to Cloudflare pool (Dashboard > Traffic > Load Balancing > Pools)
7. Wait for health check to pass

## Disaster Recovery (DR) Setup

### Adding DR Instance to Cloudflare Pool

After deploying the DR instance in `eu-west-1`:

**Via Cloudflare Dashboard:**

1. Go to **Traffic > Load Balancing > Pools**
2. Select `aragora-api-pool`
3. Click **Add Origin**
4. Configure:

| Setting | Value |
|---------|-------|
| Origin Name | `ec2-dr-eu` |
| Origin Address | `<DR Elastic IP>` |
| Port | 80 |
| Weight | 0.5 |
| Host Header | `api.aragora.ai` |

**Via Cloudflare API:**

```bash
curl -X PATCH "https://api.cloudflare.com/client/v4/accounts/{account_id}/load_balancers/pools/{pool_id}" \
  -H "Authorization: Bearer {api_token}" \
  -H "Content-Type: application/json" \
  --data '{
    "origins": [
      {"name": "ec2-primary", "address": "<PRIMARY_IP>", "enabled": true, "weight": 1.0},
      {"name": "ec2-secondary", "address": "<SECONDARY_IP>", "enabled": true, "weight": 1.0},
      {"name": "ec2-dr-eu", "address": "<DR_IP>", "enabled": true, "weight": 0.5}
    ]
  }'
```

### Failover Testing

Test DR failover regularly to ensure readiness:

1. **Disable primary origins** in Cloudflare Dashboard:
   - Traffic > Load Balancing > Pools > aragora-api-pool
   - Toggle off `ec2-primary` and `ec2-secondary`

2. **Verify traffic routes to DR**:
   ```bash
   # Multiple requests should all go to DR
   for i in {1..5}; do
     curl -s https://api.aragora.ai/api/health | jq -r '.server'
   done
   ```

3. **Test application functionality**:
   - API endpoints respond correctly
   - WebSocket connections establish
   - Database connectivity works

4. **Re-enable primary origins**:
   - Toggle on `ec2-primary` and `ec2-secondary`
   - Verify traffic distribution returns to normal

### Recovery Procedures

**If Primary Region (us-east-2) Fails:**

1. Cloudflare automatically routes 100% traffic to DR
2. Monitor DR instance health and capacity
3. Scale DR instance if needed:
   ```bash
   # Via AWS CLI
   aws ec2 modify-instance-attribute --instance-id <DR_INSTANCE_ID> \
     --instance-type '{"Value": "t3.xlarge"}' --region eu-west-1
   ```
4. Deploy additional DR instances if extended outage expected
5. Notify stakeholders of degraded redundancy

**When Primary Region Recovers:**

1. Verify primary instances are healthy:
   ```bash
   curl http://<PRIMARY_IP>/api/health
   curl http://<SECONDARY_IP>/api/health
   ```
2. Re-enable origins in Cloudflare pool
3. Monitor traffic distribution
4. Scale down DR instance if it was scaled up

### DR Health Verification

Check all origins are healthy:

```bash
# Test each origin directly
for origin in "PRIMARY_IP" "SECONDARY_IP" "DR_IP"; do
  echo "Testing $origin..."
  response=$(curl -s -o /dev/null -w "%{http_code}" -H "Host: api.aragora.ai" "http://$origin/api/health")
  echo "Status: $response"
done

# Test through load balancer
curl -s https://api.aragora.ai/api/health | jq
```

## Removing Lightsail (Migration)

The Lightsail instance has been deprecated. If you need to fully remove it:

1. Remove from Cloudflare pool (if still present)
2. Stop the Lightsail instance
3. (Optional) Delete the instance
4. Remove `LIGHTSAIL_HOST` and `LIGHTSAIL_SSH_KEY` from GitHub secrets

The deploy workflow already has Lightsail disabled (`if: false`).
