#!/bin/bash
# Aragora Lightsail Setup Script
# Run this on a fresh Ubuntu 22.04 Lightsail instance
#
# Usage:
#   curl -sL https://raw.githubusercontent.com/an0mium/aragora/main/deploy/lightsail-setup.sh | bash
#
# Or manually:
#   chmod +x lightsail-setup.sh && ./lightsail-setup.sh

set -e

echo "=== Aragora API Server Setup ==="
echo ""

# Update system
echo "[1/8] Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install dependencies
echo "[2/8] Installing Python and dependencies..."
sudo apt install -y python3-pip python3-venv python3-dev git build-essential curl

# Clone repository
echo "[3/8] Cloning aragora repository..."
cd /home/ubuntu
if [ -d "aragora" ]; then
    cd aragora && git pull
else
    git clone https://github.com/an0mium/aragora.git
    cd aragora
fi

# Create virtual environment
echo "[4/8] Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "[5/8] Installing Python packages..."
pip install --upgrade pip
pip install -e .

# Create .nomic directory if it doesn't exist
echo "[6/8] Setting up .nomic directory..."
mkdir -p /home/ubuntu/aragora/.nomic

# Create the server runner script
echo "[7/8] Creating server runner script..."
cat > /home/ubuntu/aragora/run_server.py << 'PYEOF'
#!/usr/bin/env python3
"""Simple server runner for production deployment."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aragora.server.stream import AiohttpUnifiedServer

async def main():
    nomic_dir = Path(__file__).parent / ".nomic"
    nomic_dir.mkdir(exist_ok=True)

    server = AiohttpUnifiedServer(
        port=8765,
        nomic_dir=nomic_dir
    )
    print(f"Starting Aragora API server on port 8765...")
    print(f"  Nomic dir: {nomic_dir}")
    await server.start()

if __name__ == "__main__":
    asyncio.run(main())
PYEOF

chmod +x /home/ubuntu/aragora/run_server.py

# Create systemd service
echo "[8/8] Creating systemd service..."
sudo tee /etc/systemd/system/aragora-api.service > /dev/null << 'EOF'
[Unit]
Description=Aragora WebSocket API Server
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/aragora
Environment="PATH=/home/ubuntu/aragora/venv/bin:/usr/bin"
ExecStart=/home/ubuntu/aragora/venv/bin/python /home/ubuntu/aragora/run_server.py
Restart=always
RestartSec=5

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=aragora-api

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable aragora-api
sudo systemctl start aragora-api

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Service status:"
sudo systemctl status aragora-api --no-pager || true
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "NEXT STEPS: Install Cloudflare Tunnel"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Install cloudflared:"
echo "   curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb"
echo "   sudo dpkg -i cloudflared.deb"
echo ""
echo "2. Authenticate (opens browser):"
echo "   cloudflared tunnel login"
echo ""
echo "3. Create tunnel:"
echo "   cloudflared tunnel create aragora-production"
echo ""
echo "4. Create config at ~/.cloudflared/config.yml:"
echo "   tunnel: aragora-production"
echo "   credentials-file: /home/ubuntu/.cloudflared/<TUNNEL_ID>.json"
echo "   ingress:"
echo "     - hostname: api.aragora.ai"
echo "       service: http://localhost:8765"
echo "     - service: http_status:404"
echo ""
echo "5. Route DNS:"
echo "   cloudflared tunnel route dns aragora-production api.aragora.ai"
echo ""
echo "6. Run as service:"
echo "   sudo cloudflared service install"
echo "   sudo systemctl start cloudflared"
echo "   sudo systemctl enable cloudflared"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Useful commands:"
echo "  View API logs:     sudo journalctl -u aragora-api -f"
echo "  Restart API:       sudo systemctl restart aragora-api"
echo "  View tunnel logs:  sudo journalctl -u cloudflared -f"
echo ""
