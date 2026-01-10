#!/bin/bash
# EC2 user data script for aragora API server

set -e

# Update system
yum update -y

# Install Python 3.10+, nginx, and git
amazon-linux-extras install nginx1 -y
amazon-linux-extras install python3.11 -y || amazon-linux-extras install python3.10 -y
yum install git python3-pip -y

# Ensure Python version meets minimum requirement
python3 - << 'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Aragora requires Python 3.10+")
PY

# Install certbot for SSL
amazon-linux-extras install epel -y
yum install certbot python3-certbot-nginx -y

# Create aragora user
useradd -m -s /bin/bash aragora || true

# Clone aragora repository
cd /home/aragora
sudo -u aragora git clone https://github.com/an0mium/aragora.git || (cd aragora && sudo -u aragora git pull)

# Install Python dependencies
cd /home/aragora/aragora
sudo -u aragora python3 -m pip install --user . || true

# Create systemd service for aragora API
cat > /etc/systemd/system/aragora-api.service << 'EOF'
[Unit]
Description=Aragora API Server
After=network.target

[Service]
Type=simple
User=aragora
WorkingDirectory=/home/aragora/aragora
Environment=PATH=/home/aragora/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/usr/bin/python3 -c "
import asyncio
import sys
sys.path.insert(0, '/home/aragora/aragora')
from aragora.server.unified_server import UnifiedServer

async def main():
    server = UnifiedServer(http_port=8080, ws_port=8765, enable_persistence=True)
    await server.start()
    # Keep running
    while True:
        await asyncio.sleep(3600)

asyncio.run(main())
"
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Configure nginx as reverse proxy with WebSocket support
cat > /etc/nginx/conf.d/aragora-api.conf << 'EOF'
# HTTP to HTTPS redirect
server {
    listen 80;
    server_name api.aragora.ai;
    return 301 https://$server_name$request_uri;
}

# HTTPS server
server {
    listen 443 ssl http2;
    server_name api.aragora.ai;

    # SSL will be configured by certbot
    # ssl_certificate /etc/letsencrypt/live/api.aragora.ai/fullchain.pem;
    # ssl_certificate_key /etc/letsencrypt/live/api.aragora.ai/privkey.pem;

    # HTTP API proxy
    location /api/ {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket proxy
    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 86400;
    }
}
EOF

# Start services
systemctl daemon-reload
systemctl enable aragora-api
systemctl start aragora-api
systemctl enable nginx
systemctl start nginx

echo "Aragora API server setup complete!"
echo "Don't forget to:"
echo "1. Update DNS for api.aragora.ai to point to this instance"
echo "2. Run: sudo certbot --nginx -d api.aragora.ai"
