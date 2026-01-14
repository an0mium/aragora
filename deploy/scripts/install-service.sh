#!/bin/bash
# Install Aragora service with environment-specific configuration
#
# Usage: sudo ./install-service.sh [--user USER] [--home DIR]
#
# Detects environment (EC2, Lightsail, or generic) and configures accordingly.

set -e

# Parse arguments
ARAGORA_USER=""
ARAGORA_HOME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --user) ARAGORA_USER="$2"; shift 2 ;;
        --home) ARAGORA_HOME="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Auto-detect environment if not specified
if [ -z "$ARAGORA_USER" ]; then
    if id ec2-user &>/dev/null; then
        ARAGORA_USER="ec2-user"
        ARAGORA_HOME="${ARAGORA_HOME:-/home/ec2-user/aragora}"
        echo "[detect] EC2 environment detected"
    elif id ubuntu &>/dev/null; then
        ARAGORA_USER="ubuntu"
        ARAGORA_HOME="${ARAGORA_HOME:-/home/ubuntu/aragora}"
        echo "[detect] Ubuntu/Lightsail environment detected"
    else
        ARAGORA_USER="aragora"
        ARAGORA_HOME="${ARAGORA_HOME:-/opt/aragora}"
        echo "[detect] Generic environment, using aragora user"
    fi
fi

ARAGORA_HOME="${ARAGORA_HOME:-/opt/aragora}"

echo "[config] User: $ARAGORA_USER"
echo "[config] Home: $ARAGORA_HOME"

# Stop existing services
echo "[cleanup] Stopping existing services..."
systemctl stop aragora-staging aragora-unified aragora 2>/dev/null || true
systemctl disable aragora-staging aragora-unified 2>/dev/null || true

# Remove old service files
rm -f /etc/systemd/system/aragora-staging.service
rm -f /etc/systemd/system/aragora-unified.service

# Create service file
echo "[install] Creating systemd service..."
cat > /etc/systemd/system/aragora.service << EOF
[Unit]
Description=Aragora API Server (HTTP + WebSocket)
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=$ARAGORA_USER
Group=$ARAGORA_USER
WorkingDirectory=$ARAGORA_HOME

# Environment setup
Environment="PATH=$ARAGORA_HOME/venv/bin:/usr/local/bin:/usr/bin"
EnvironmentFile=-$ARAGORA_HOME/.env

# Unified server: HTTP (8080) + WebSocket (8765)
ExecStart=$ARAGORA_HOME/venv/bin/python -m aragora.server \\
    --host 0.0.0.0 \\
    --http-port 8080 \\
    --port 8765 \\
    --nomic-dir $ARAGORA_HOME/.nomic

# Restart policy with crash loop prevention
Restart=always
RestartSec=5
StartLimitIntervalSec=300
StartLimitBurst=5

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=aragora

# Security hardening
NoNewPrivileges=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
EOF

# Reload and enable
echo "[install] Enabling service..."
systemctl daemon-reload
systemctl enable aragora.service

# Start service
echo "[install] Starting service..."
systemctl start aragora.service

# Wait and verify
sleep 5
if systemctl is-active --quiet aragora.service; then
    echo "[success] Aragora service is running"
    systemctl status aragora.service --no-pager | head -15
else
    echo "[error] Service failed to start"
    journalctl -u aragora.service -n 30 --no-pager
    exit 1
fi
