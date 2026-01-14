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
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │   EC2 #1    │  │   EC2 #2    │  │  (Future)   │
    │  Primary    │  │  Secondary  │  │   EC2 #3    │
    └─────────────┘  └─────────────┘  └─────────────┘
         nginx           nginx           nginx
         :80             :80             :80
```

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

1. Launch new EC2 with same AMI/setup as existing instances
2. Configure nginx using `deploy/configure-nginx.sh`
3. Start aragora service
4. Add EC2 IP to security group for Cloudflare IPs
5. Add origin to Cloudflare pool (Dashboard > Traffic > Load Balancing > Pools)
6. Wait for health check to pass
7. Update `deploy.yml` to include new instance (optional for automated deploys)

## Removing Lightsail (Migration)

The Lightsail instance has been deprecated. If you need to fully remove it:

1. Remove from Cloudflare pool (if still present)
2. Stop the Lightsail instance
3. (Optional) Delete the instance
4. Remove `LIGHTSAIL_HOST` and `LIGHTSAIL_SSH_KEY` from GitHub secrets

The deploy workflow already has Lightsail disabled (`if: false`).
