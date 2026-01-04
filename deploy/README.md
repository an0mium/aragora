# Aragora Production Deployment

## Quick Deploy

```bash
# 1. Clone and setup
git clone https://github.com/an0mium/aragora.git /opt/aragora
cd /opt/aragora
python -m venv venv
source venv/bin/activate
pip install -e .

# 2. Install systemd service
sudo cp deploy/aragora.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable aragora
sudo systemctl start aragora

# 3. Install nginx config
sudo cp deploy/nginx.conf.example /etc/nginx/sites-available/aragora
sudo ln -s /etc/nginx/sites-available/aragora /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# 4. Setup SSL (Let's Encrypt)
sudo certbot --nginx -d api.aragora.ai
```

## Architecture

```
                    ┌──────────────────┐
                    │   Nginx (443)    │
                    │ api.aragora.ai   │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌─────────────────┐          ┌─────────────────┐
    │  HTTP API       │          │  WebSocket      │
    │  Port 8080      │          │  Port 8765      │
    │  /api/*         │          │  wss://         │
    └─────────────────┘          └─────────────────┘
```

## Ports

| Service | Port | Protocol |
|---------|------|----------|
| HTTP API | 8080 | HTTP |
| WebSocket | 8765 | WS |
| Nginx | 443 | HTTPS/WSS |

## Troubleshooting

### WebSocket not connecting

```bash
# Check if server is running
sudo systemctl status aragora

# Test WebSocket directly
wscat -c ws://localhost:8765

# Check nginx config
sudo nginx -t

# View logs
sudo journalctl -u aragora -f
```

### CORS issues

Ensure `https://aragora.ai` is in the allowed origins:
- Backend: `ARAGORA_ALLOWED_ORIGINS` env var
- Nginx: CORS headers in nginx.conf

### SSL certificate renewal

```bash
sudo certbot renew --dry-run
```
