#!/bin/bash
# Diagnose CORS configuration on Aragora server
# Run this on the EC2 instance to check why CORS isn't working

echo "=== Aragora CORS Diagnostic Script ==="
echo ""

# Check environment variables
echo "1. Checking environment variables..."
echo "   ARAGORA_ENV = ${ARAGORA_ENV:-<not set>}"
echo "   ARAGORA_ALLOWED_ORIGINS = ${ARAGORA_ALLOWED_ORIGINS:-<not set>}"
echo ""

# Check systemd service environment
echo "2. Checking systemd service environment..."
if [ -f /opt/aragora/.env ]; then
    echo "   /opt/aragora/.env exists"
    grep -E "ARAGORA_ENV|ARAGORA_ALLOWED_ORIGINS" /opt/aragora/.env 2>/dev/null || echo "   No CORS vars found in .env"
else
    echo "   /opt/aragora/.env NOT FOUND"
fi
echo ""

# Check systemd environment file
echo "3. Checking systemd service file..."
if [ -f /etc/systemd/system/aragora.service ]; then
    grep -E "Environment|EnvironmentFile" /etc/systemd/system/aragora.service 2>/dev/null || echo "   No environment directives found"
else
    echo "   /etc/systemd/system/aragora.service NOT FOUND"
fi
echo ""

# Test CORS directly from Python
echo "4. Testing CORS config from Python..."
python3 -c "
import os
# Simulate the module being imported fresh
os.environ.pop('ARAGORA_ALLOWED_ORIGINS', None)  # Clear to test default

# Check what environment is detected
from aragora.server.cors_config import cors_config, _IS_PRODUCTION, DEFAULT_ORIGINS
print(f'   Production mode: {_IS_PRODUCTION}')
print(f'   Default origins: {DEFAULT_ORIGINS}')
print(f'   Configured origins: {cors_config.allowed_origins}')
print(f'   Using env config: {cors_config._using_env_config}')
print(f'   live.aragora.ai allowed: {cors_config.is_origin_allowed(\"https://live.aragora.ai\")}')
" 2>/dev/null || echo "   Python test failed - check if aragora is installed"
echo ""

# Check if service is running
echo "5. Checking service status..."
systemctl is-active aragora 2>/dev/null || echo "   Service not running via systemctl"
pgrep -f "aragora.server" > /dev/null && echo "   Aragora process IS running" || echo "   Aragora process NOT running"
echo ""

# Test local CORS response
echo "6. Testing local API with Origin header..."
curl -s -I -H "Origin: https://live.aragora.ai" http://localhost:8080/api/health 2>/dev/null | grep -i "access-control-allow-origin" || echo "   Access-Control-Allow-Origin header NOT returned"
echo ""

echo "=== Diagnostic Complete ==="
echo ""
echo "RECOMMENDED FIX:"
echo "----------------"
echo "Add these lines to /opt/aragora/.env (or wherever your env is):"
echo ""
echo "ARAGORA_ENV=production"
echo "ARAGORA_ALLOWED_ORIGINS=https://aragora.ai,https://www.aragora.ai,https://live.aragora.ai,https://api.aragora.ai"
echo ""
echo "Then restart the service:"
echo "sudo systemctl restart aragora"
