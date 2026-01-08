#!/bin/bash
# Setup Cloudflare Tunnel for Aragora staging
#
# Usage: ./setup-cloudflare-tunnel.sh <TUNNEL_TOKEN>
#
# To get a tunnel token:
# 1. Go to https://one.dash.cloudflare.com/
# 2. Click "Networks" -> "Tunnels"
# 3. Click "Create a tunnel"
# 4. Name it "aragora-staging"
# 5. Copy the token shown

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <TUNNEL_TOKEN>"
    echo ""
    echo "To get a tunnel token:"
    echo "1. Go to https://one.dash.cloudflare.com/"
    echo "2. Click 'Networks' -> 'Tunnels'"
    echo "3. Click 'Create a tunnel'"
    echo "4. Name it 'aragora-staging'"
    echo "5. Copy the token shown"
    exit 1
fi

TUNNEL_TOKEN="$1"

# Create cloudflared config directory
mkdir -p ~/.cloudflared

# Create config file
cat > ~/.cloudflared/config.yml << EOF
tunnel: aragora-staging
token: ${TUNNEL_TOKEN}

ingress:
  # WebSocket endpoint
  - hostname: staging.aragora.ai
    path: /ws
    service: http://localhost:8765
  # HTTP API endpoints
  - hostname: staging.aragora.ai
    service: http://localhost:8080
  # Fallback
  - service: http_status:404
EOF

# Create systemd service
sudo tee /etc/systemd/system/cloudflared.service > /dev/null << 'EOF'
[Unit]
Description=Cloudflare Tunnel for Aragora Staging
After=network.target

[Service]
Type=simple
User=ec2-user
ExecStart=/usr/local/bin/cloudflared tunnel --config /home/ec2-user/.cloudflared/config.yml run
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable cloudflared
sudo systemctl start cloudflared

echo ""
echo "Cloudflare tunnel configured!"
echo "Staging API will be available at: https://staging.aragora.ai"
echo ""
echo "To verify: sudo systemctl status cloudflared"
