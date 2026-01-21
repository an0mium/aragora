#!/bin/bash
# Production Diagnostic and Recovery Script for Aragora
# Run this on the EC2 production instance to diagnose and fix issues

set -e

echo "=============================================="
echo "ARAGORA PRODUCTION DIAGNOSTICS"
echo "=============================================="
echo "Date: $(date)"
echo ""

# 1. Check service status
echo "=== Service Status ==="
systemctl is-active aragora && echo "Service: RUNNING" || echo "Service: NOT RUNNING"
systemctl is-enabled aragora && echo "Enabled: YES" || echo "Enabled: NO"
echo ""

# 2. Check recent logs
echo "=== Recent Logs (last 50 lines) ==="
journalctl -u aragora -n 50 --no-pager 2>/dev/null || echo "Could not fetch logs"
echo ""

# 3. Check if Python can import the server
echo "=== Import Check ==="
cd /home/ec2-user/aragora
source venv/bin/activate
python -c "from aragora.server.unified_server import UnifiedServer; print('Import OK')" 2>&1 || echo "Import FAILED"
echo ""

# 4. Check secrets manager access
echo "=== Secrets Manager Check ==="
export ARAGORA_USE_SECRETS_MANAGER=true
python -c "
from aragora.config.secrets import get_secret
dsn = get_secret('ARAGORA_POSTGRES_DSN')
if dsn:
    print(f'PostgreSQL DSN: Found ({len(dsn)} chars)')
else:
    print('PostgreSQL DSN: NOT FOUND')
" 2>&1 || echo "Secrets check failed"
echo ""

# 5. Check PostgreSQL connectivity
echo "=== PostgreSQL Connectivity ==="
python -c "
import asyncio
from aragora.config.secrets import get_secret
dsn = get_secret('ARAGORA_POSTGRES_DSN')
if not dsn:
    print('No DSN available')
    exit(1)

async def test():
    import asyncpg
    try:
        conn = await asyncpg.connect(dsn)
        version = await conn.fetchval('SELECT version()')
        print(f'Connected: {version[:50]}...')
        await conn.close()
    except Exception as e:
        print(f'Connection failed: {e}')

asyncio.run(test())
" 2>&1 || echo "PostgreSQL check failed"
echo ""

# 6. Check disk space
echo "=== Disk Space ==="
df -h / | tail -1
echo ""

# 7. Check memory
echo "=== Memory ==="
free -h | head -2
echo ""

# 8. Check port 8080
echo "=== Port 8080 ==="
ss -tlnp | grep 8080 || echo "Port 8080 not listening"
echo ""

echo "=============================================="
echo "DIAGNOSTICS COMPLETE"
echo "=============================================="
echo ""

# Ask if user wants to attempt recovery
read -p "Attempt automatic recovery? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "=== Attempting Recovery ==="

    # Ensure secrets manager is enabled in systemd
    echo "Updating systemd environment..."
    sudo mkdir -p /etc/systemd/system/aragora.service.d/
    cat << 'EOF' | sudo tee /etc/systemd/system/aragora.service.d/secrets.conf
[Service]
Environment="ARAGORA_USE_SECRETS_MANAGER=true"
EOF

    sudo systemctl daemon-reload

    echo "Restarting aragora service..."
    sudo systemctl restart aragora

    sleep 5

    if systemctl is-active --quiet aragora; then
        echo "SUCCESS: Service is running!"

        # Test health endpoint
        sleep 3
        if curl -sf http://localhost:8080/api/health > /dev/null 2>&1; then
            echo "SUCCESS: Health endpoint responding!"
        else
            echo "WARNING: Health endpoint not responding yet (may need more time)"
        fi
    else
        echo "FAILED: Service did not start"
        echo ""
        echo "Last 20 log lines:"
        journalctl -u aragora -n 20 --no-pager
    fi
fi
