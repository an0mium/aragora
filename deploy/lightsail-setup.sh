#!/bin/bash
# Aragora Lightsail Setup Script
# Run this on a fresh Ubuntu 22.04 Lightsail instance

set -e

echo "=== Aragora API Server Setup ==="

# Update system
echo "[1/6] Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install dependencies
echo "[2/6] Installing Python and dependencies..."
sudo apt install -y python3-pip python3-venv git build-essential

# Clone repository
echo "[3/6] Cloning aragora repository..."
cd /home/ubuntu
if [ -d "aragora" ]; then
    cd aragora && git pull
else
    git clone https://github.com/an0mium/aragora.git
    cd aragora
fi

# Create virtual environment
echo "[4/6] Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "[5/6] Installing Python packages..."
pip install --upgrade pip
pip install -e .

# Create systemd service
echo "[6/6] Creating systemd service..."
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
ExecStart=/home/ubuntu/aragora/venv/bin/python -m aragora.server.unified_server --host 0.0.0.0 --port 8765
Restart=always
RestartSec=5

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=aragora-api

# Security
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/home/ubuntu/aragora

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
sudo systemctl status aragora-api --no-pager
echo ""
echo "View logs with: sudo journalctl -u aragora-api -f"
echo ""
echo "The API server is now running on port 8765"
echo "Update your Cloudflare tunnel to point to this server's IP"
