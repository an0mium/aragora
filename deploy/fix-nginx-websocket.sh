#!/bin/bash
# Script to fix nginx WebSocket routing on EC2/Lightsail
# Run this on each server: bash fix-nginx-websocket.sh

set -e

echo "=== Fixing nginx WebSocket routing ==="

# Backup existing config
if [ -f /etc/nginx/sites-available/aragora ]; then
    sudo cp /etc/nginx/sites-available/aragora /etc/nginx/sites-available/aragora.backup.$(date +%Y%m%d%H%M%S)
    echo "Backed up existing config"
fi

# Create new nginx config with proper WebSocket routing
sudo tee /etc/nginx/sites-available/aragora > /dev/null << 'EOF'
upstream aragora_api {
    server 127.0.0.1:8080;
}

upstream aragora_ws {
    server 127.0.0.1:8765;
}

server {
    listen 80;
    server_name api.aragora.ai _;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.aragora.ai _;

    # SSL - adjust paths based on your cert location
    ssl_certificate /etc/ssl/certs/cloudflare-origin.pem;
    ssl_certificate_key /etc/ssl/private/cloudflare-origin-key.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;

    # WebSocket endpoint - CRITICAL: must come before location /
    location /ws {
        proxy_pass http://aragora_ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }

    # API endpoints
    location / {
        proxy_pass http://aragora_api;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300;
    }
}
EOF

echo "Created new nginx config"

# Enable site if not already enabled
sudo ln -sf /etc/nginx/sites-available/aragora /etc/nginx/sites-enabled/aragora

# Remove default site if it exists and conflicts
if [ -f /etc/nginx/sites-enabled/default ]; then
    sudo rm /etc/nginx/sites-enabled/default
    echo "Removed default site"
fi

# Test nginx config
echo "Testing nginx configuration..."
sudo nginx -t

# Reload nginx
echo "Reloading nginx..."
sudo systemctl reload nginx

echo "=== Done! Testing endpoints ==="
echo "API health:"
curl -s http://127.0.0.1:8080/api/health | head -c 100
echo ""
echo "WebSocket port:"
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8765/
echo " (426 = WebSocket server running correctly)"
