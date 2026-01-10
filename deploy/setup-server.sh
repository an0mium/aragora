#!/bin/bash
# Aragora Server Setup Script for AWS EC2 / Lightsail
#
# Usage:
#   1. Launch Ubuntu 22.04 instance (t3.small or larger recommended)
#   2. SSH into the instance
#   3. Run: curl -sSL https://raw.githubusercontent.com/yourusername/aragora/main/deploy/setup-server.sh | bash
#
# Or manually:
#   git clone https://github.com/yourusername/aragora.git
#   cd aragora
#   ./deploy/setup-server.sh

set -e

echo "=== Aragora Server Setup ==="

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install dependencies
echo "Installing dependencies..."
sudo apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    nginx \
    certbot \
    python3-certbot-nginx

# Create aragora user if not exists
if ! id "aragora" &>/dev/null; then
    echo "Creating aragora user..."
    sudo useradd -m -s /bin/bash aragora
fi

# Clone or update repository
ARAGORA_DIR="/opt/aragora"
if [ -d "$ARAGORA_DIR" ]; then
    echo "Updating existing installation..."
    cd "$ARAGORA_DIR"
    sudo -u aragora git pull
else
    echo "Cloning repository..."
    sudo git clone https://github.com/yourusername/aragora.git "$ARAGORA_DIR"
    sudo chown -R aragora:aragora "$ARAGORA_DIR"
fi

cd "$ARAGORA_DIR"

# Create virtual environment
echo "Setting up Python environment..."
sudo -u aragora python3.11 -m venv venv
sudo -u aragora ./venv/bin/pip install --upgrade pip
sudo -u aragora ./venv/bin/pip install .

# Create .env file if not exists
if [ ! -f "$ARAGORA_DIR/.env" ]; then
    echo "Creating .env file..."
    sudo -u aragora cp .env.example .env
    echo ""
    echo "!!! IMPORTANT !!!"
    echo "Edit /opt/aragora/.env and add your API keys:"
    echo "  sudo nano /opt/aragora/.env"
    echo ""
fi

# Create systemd service
echo "Creating systemd service..."
sudo tee /etc/systemd/system/aragora.service > /dev/null << 'EOF'
[Unit]
Description=Aragora Server
After=network.target

[Service]
Type=simple
User=aragora
Group=aragora
WorkingDirectory=/opt/aragora
Environment=PATH=/opt/aragora/venv/bin:/usr/bin
EnvironmentFile=/opt/aragora/.env
ExecStart=/opt/aragora/venv/bin/python -m aragora.server --host 0.0.0.0 --http-port 8080 --port 8765
Restart=always
RestartSec=5

# Security hardening
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/aragora/.nomic

[Install]
WantedBy=multi-user.target
EOF

# Create .nomic directory for state
sudo -u aragora mkdir -p "$ARAGORA_DIR/.nomic"

# Enable and start service
echo "Starting Aragora service..."
sudo systemctl daemon-reload
sudo systemctl enable aragora
sudo systemctl start aragora

# Configure nginx reverse proxy
echo "Configuring nginx..."
sudo tee /etc/nginx/sites-available/aragora > /dev/null << 'EOF'
# Aragora Server Nginx Configuration
# Replace YOUR_DOMAIN with your actual domain (e.g., api.aragora.ai)

server {
    listen 80;
    server_name YOUR_DOMAIN;

    # HTTP API
    location /api {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Health checks
    location /healthz {
        proxy_pass http://127.0.0.1:8080;
    }

    location /readyz {
        proxy_pass http://127.0.0.1:8080;
    }

    # WebSocket
    location /ws {
        proxy_pass http://127.0.0.1:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;
    }

    # Static files (optional - for live dashboard)
    location / {
        proxy_pass http://127.0.0.1:8080;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/aragora /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Edit /opt/aragora/.env with your API keys"
echo "2. Replace YOUR_DOMAIN in /etc/nginx/sites-available/aragora"
echo "3. Run: sudo certbot --nginx -d YOUR_DOMAIN"
echo "4. Restart: sudo systemctl restart aragora"
echo ""
echo "Check status: sudo systemctl status aragora"
echo "View logs: sudo journalctl -u aragora -f"
echo ""
