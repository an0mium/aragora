# Aragora Deployment Guide

Deploy the Aragora server to keep `api.aragora.ai` running 24/7.

## Current Architecture

```
api.aragora.ai → Cloudflare Tunnel → EC2 (3.141.158.91)
                                      ├── nginx :80
                                      ├── aragora :8080 (HTTP API)
                                      └── aragora :8765 (WebSocket)
```

**Primary:** AWS EC2 via Cloudflare tunnel (`ringrift-cluster`)
**Docs:** See `cloudflare-lb-setup.md` for Cloudflare configuration

## Quick Start (AWS EC2)

1. **Launch Instance**
   - AWS EC2, Amazon Linux 2 or Ubuntu 22.04
   - Instance type: t3.micro or larger
   - Security group: Allow SSH (22), HTTP (80) from Cloudflare IPs

2. **Connect and Install**
   ```bash
   ssh ec2-user@your-instance-ip
   git clone https://github.com/yourusername/aragora.git
   cd aragora
   ./deploy/setup-server.sh
   ```

3. **Configure API Keys**
   ```bash
   nano ~/aragora/.env
   ```
   Add your keys:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   OPENAI_API_KEY=sk-proj-...
   OPENROUTER_API_KEY=sk-or-...
   ```

4. **Configure Nginx**
   ```bash
   ./deploy/configure-nginx.sh
   ```

5. **Start Services**
   ```bash
   sudo systemctl enable aragora
   sudo systemctl start aragora
   ```

## GitHub Actions Deployment

Pushes to `main` branch automatically deploy via `.github/workflows/deploy.yml`:
- Deploys to EC2 via SSH
- Runs health checks
- Rolls back on failure

Required secrets:
- `EC2_HOST` - EC2 public IP
- `EC2_SSH_KEY` - SSH private key

## Management Commands

```bash
# Check status
sudo systemctl status aragora

# View logs
sudo journalctl -u aragora -f

# Restart server
sudo systemctl restart aragora

# Health check
curl http://localhost:8080/api/health
curl http://localhost/api/health  # via nginx
```

## Ports

| Port | Service | Description |
|------|---------|-------------|
| 80   | nginx   | HTTP reverse proxy |
| 8080 | aragora | HTTP API (internal) |
| 8765 | aragora | WebSocket (internal) |

## Monitoring

Health endpoints:
- `GET /api/health` - Basic health check
- `GET /api/health/detailed` - Detailed health with all checks

## Cloudflare Configuration

See `cloudflare-lb-setup.md` for:
- Tunnel configuration
- Security group IP ranges
- Health check settings
- WebSocket session affinity

## Troubleshooting

**Server won't start**
```bash
sudo journalctl -u aragora -n 100
python -c "from aragora.server.unified_server import UnifiedServer; print('OK')"
```

**Health check fails via nginx**
```bash
# Check nginx config
sudo nginx -t
# Check nginx routes to correct port
curl -v http://localhost/api/health
```

**WebSocket fails**
- Check nginx has WebSocket upgrade headers
- Check port 8765 is running: `ss -tlnp | grep 8765`

**Cloudflare 521 error**
- Check security group allows Cloudflare IPs
- Verify nginx is running on port 80
- Check tunnel connector is running: `sudo systemctl status cloudflared`
