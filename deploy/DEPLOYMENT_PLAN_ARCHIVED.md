# [ARCHIVED] Aragora Production Deployment Plan

> **DEPRECATED:** This document is outdated and kept for historical reference only.
>
> **Current documentation:**
> - `README.md` - Quick start guide for EC2 deployment
> - `cloudflare-lb-setup.md` - Cloudflare tunnel and load balancer configuration
>
> **Current architecture:** EC2-only via Cloudflare tunnel (`ringrift-cluster`)

---

## Overview (Historical)

This document outlines the deployment architecture for aragora's production infrastructure.

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLOUDFLARE                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐              ┌─────────────────┐           │
│  │ aragora.ai      │              │  api.aragora.ai │           │
│  │ www.aragora.ai  │              │                 │           │
│  └────────┬────────┘              └────────┬────────┘           │
│           │                                │                     │
│           ▼                                ▼                     │
│  ┌─────────────────┐              ┌─────────────────┐           │
│  │ Cloudflare      │              │ Cloudflare      │           │
│  │ Pages (Static)  │              │ Tunnel          │           │
│  └─────────────────┘              └────────┬────────┘           │
│                                            │                     │
└────────────────────────────────────────────┼─────────────────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │ Mac localhost   │  ← BOTTLENECK
                                    │ :8765           │
                                    └─────────────────┘
```

## Target Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLOUDFLARE                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐              ┌─────────────────┐           │
│  │ aragora.ai      │              │  api.aragora.ai │           │
│  │ www.aragora.ai  │              │                 │           │
│  └────────┬────────┘              └────────┬────────┘           │
│           │                                │                     │
│           ▼                                ▼                     │
│  ┌─────────────────┐              ┌─────────────────┐           │
│  │ Cloudflare      │              │ Cloudflare      │           │
│  │ Pages (Static)  │              │ Tunnel          │           │
│  └─────────────────┘              └────────┬────────┘           │
│                                            │                     │
└────────────────────────────────────────────┼─────────────────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │ AWS Lightsail   │  ← ALWAYS ON
                                    │ Ubuntu 22.04   │
                                    │ :8765           │
                                    └─────────────────┘
```

## Components

### 1. Static Frontend (Cloudflare Pages)
- **URL:** https://aragora.ai, https://www.aragora.ai
- **Source:** `aragora/live/out/` (Next.js static export)
- **Deployment:** `npx wrangler pages deploy out --project-name=aragora-live`
- **Cost:** Free tier (unlimited requests)

### 2. WebSocket API (AWS Lightsail)
- **URL:** https://api.aragora.ai (via Cloudflare Tunnel)
- **Source:** `aragora/server/unified_server.py`
- **Instance:** Ubuntu 22.04, nano_3_0 bundle
- **Port:** 8765
- **Cost:** $5/month

### 3. Cloudflare Tunnel
- **Purpose:** Securely expose Lightsail to internet without opening ports
- **Config:** Routes api.aragora.ai to Lightsail:8765

---

## Deployment Steps

### Step 1: Create Lightsail Instance

```bash
# Create instance
aws lightsail create-instances \
  --instance-names aragora-api \
  --availability-zone us-east-1a \
  --blueprint-id ubuntu_22_04 \
  --bundle-id nano_3_0

# Wait for instance to be running
aws lightsail get-instance --instance-name aragora-api \
  --query 'instance.state.name'

# Get public IP
aws lightsail get-instance --instance-name aragora-api \
  --query 'instance.publicIpAddress' --output text
```

### Step 2: Open Firewall Ports

```bash
# Open port 8765 for WebSocket (optional - only if not using tunnel)
aws lightsail open-instance-public-ports \
  --instance-name aragora-api \
  --port-info fromPort=8765,toPort=8765,protocol=TCP

# Open port 22 for SSH
aws lightsail open-instance-public-ports \
  --instance-name aragora-api \
  --port-info fromPort=22,toPort=22,protocol=TCP
```

### Step 3: Download SSH Key

```bash
# Download the default key pair
aws lightsail download-default-key-pair \
  --output text --query 'privateKeyBase64' | base64 -d > ~/.ssh/lightsail-aragora.pem
chmod 600 ~/.ssh/lightsail-aragora.pem
```

### Step 4: SSH and Run Setup Script

```bash
# Get the IP
LIGHTSAIL_IP=$(aws lightsail get-instance --instance-name aragora-api \
  --query 'instance.publicIpAddress' --output text)

# SSH into instance
ssh -i ~/.ssh/lightsail-aragora.pem ubuntu@$LIGHTSAIL_IP

# On the instance, run:
curl -sL https://raw.githubusercontent.com/an0mium/aragora/main/deploy/lightsail-setup.sh | bash
```

### Step 5: Install Cloudflare Tunnel on Lightsail

```bash
# On the Lightsail instance:

# Install cloudflared
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb -o cloudflared.deb
sudo dpkg -i cloudflared.deb

# Authenticate (opens browser for Cloudflare login)
cloudflared tunnel login

# Create tunnel config
mkdir -p ~/.cloudflared
cat > ~/.cloudflared/config.yml << 'EOF'
tunnel: aragora-live
credentials-file: /home/ubuntu/.cloudflared/<TUNNEL_ID>.json

ingress:
  - hostname: api.aragora.ai
    service: http://localhost:8765
  - service: http_status:404
EOF

# Run tunnel as service
sudo cloudflared service install
sudo systemctl start cloudflared
sudo systemctl enable cloudflared
```

### Step 6: Update DNS (if needed)

The tunnel should automatically update DNS. Verify at:
https://dash.cloudflare.com → aragora.ai → DNS

---

## Verification

### Check Service Status
```bash
# On Lightsail instance
sudo systemctl status aragora-api
sudo journalctl -u aragora-api -f
```

### Test WebSocket Connection
```bash
# From local machine
wscat -c wss://api.aragora.ai/ws
```

### Test API Endpoint
```bash
curl https://api.aragora.ai/api/health
```

---

## Maintenance

### Update Code
```bash
ssh -i ~/.ssh/lightsail-aragora.pem ubuntu@$LIGHTSAIL_IP
cd ~/aragora
git pull
sudo systemctl restart aragora-api
```

### View Logs
```bash
sudo journalctl -u aragora-api -f
```

### Restart Service
```bash
sudo systemctl restart aragora-api
```

---

## Costs

| Component | Monthly Cost |
|-----------|--------------|
| Cloudflare Pages | $0 (free tier) |
| Cloudflare Tunnel | $0 (free) |
| AWS Lightsail nano_3_0 | $5 |
| **Total** | **$5/month** |

---

## Future Scaling

When traffic exceeds Lightsail capacity:

1. **Vertical scaling:** Upgrade to larger Lightsail bundle ($10-40/mo)
2. **Horizontal scaling:** Migrate to AWS Fargate with ALB
3. **Global scaling:** Migrate to Cloudflare Workers + Durable Objects

---

## Rollback

To revert to Mac-based hosting:

1. Stop Lightsail cloudflared: `sudo systemctl stop cloudflared`
2. Start local tunnel on Mac: `launchctl start com.aragora.cloudflared`
3. Ensure Mac server is running on :8765

---

## Files

- `deploy/lightsail-setup.sh` - Instance setup script
- `deploy/DEPLOYMENT_PLAN.md` - This document
