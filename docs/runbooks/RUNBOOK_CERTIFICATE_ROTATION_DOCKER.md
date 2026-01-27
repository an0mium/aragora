# Certificate Rotation Runbook (Docker)

**Purpose:** Rotate TLS certificates in Docker deployments with zero downtime.
**Audience:** DevOps, SRE, Platform Engineers
**Last Updated:** January 2026

---

## Overview

This runbook covers:
- Certificate rotation for Docker/Docker Compose deployments
- Zero-downtime certificate updates
- Automation with certbot and ACME
- Monitoring certificate expiration
- Emergency certificate replacement

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Certificate Flow                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐     │
│  │   Certbot   │─────▶│   Volume    │─────▶│   Nginx     │     │
│  │  Container  │      │  /certs     │      │   Proxy     │     │
│  └─────────────┘      └─────────────┘      └─────────────┘     │
│         │                    │                    │              │
│         │                    │                    ▼              │
│         │                    │           ┌─────────────┐        │
│         │                    └──────────▶│   Aragora   │        │
│         │                                │   Server    │        │
│         │                                └─────────────┘        │
│         ▼                                                        │
│  ┌─────────────┐                                                │
│  │ Let's       │                                                │
│  │ Encrypt     │                                                │
│  └─────────────┘                                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Requirement | Specification |
|-------------|---------------|
| Docker | 20.10+ |
| Docker Compose | v2.0+ |
| Domain | DNS pointing to server |
| Ports | 80, 443 open |

---

## Phase 1: Initial Certificate Setup

### 1.1 Docker Compose Configuration

```yaml
# docker-compose.yml

version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - certbot-webroot:/var/www/certbot:ro
      - certbot-certs:/etc/letsencrypt:ro
    depends_on:
      - aragora
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "nginx", "-t"]
      interval: 30s
      timeout: 10s
      retries: 3

  certbot:
    image: certbot/certbot:latest
    volumes:
      - certbot-webroot:/var/www/certbot
      - certbot-certs:/etc/letsencrypt
    entrypoint: /bin/sh -c 'trap exit TERM; while :; do certbot renew; sleep 12h; done'
    restart: unless-stopped

  aragora:
    image: aragora/server:latest
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/aragora
    restart: unless-stopped

volumes:
  certbot-webroot:
  certbot-certs:
```

### 1.2 Nginx Configuration

```nginx
# nginx/conf.d/aragora.conf

# HTTP - redirect to HTTPS and serve ACME challenges
server {
    listen 80;
    server_name aragora.example.com;

    # ACME challenge for Let's Encrypt
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    # Redirect all other traffic to HTTPS
    location / {
        return 301 https://$host$request_uri;
    }
}

# HTTPS - main server
server {
    listen 443 ssl http2;
    server_name aragora.example.com;

    # SSL certificates
    ssl_certificate /etc/letsencrypt/live/aragora.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/aragora.example.com/privkey.pem;

    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;
    ssl_session_tickets off;

    # HSTS
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Proxy to Aragora
    location / {
        proxy_pass http://aragora:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket support
    location /ws {
        proxy_pass http://aragora:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
```

### 1.3 Obtain Initial Certificate

```bash
#!/bin/bash
# scripts/init_certs.sh

set -euo pipefail

DOMAIN="aragora.example.com"
EMAIL="admin@example.com"

# Start nginx without SSL first
docker compose up -d nginx

# Wait for nginx to be ready
sleep 5

# Obtain certificate
docker compose run --rm certbot certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    --email "${EMAIL}" \
    --agree-tos \
    --no-eff-email \
    -d "${DOMAIN}"

# Reload nginx with new certificates
docker compose exec nginx nginx -s reload

echo "Certificate obtained successfully for ${DOMAIN}"
```

---

## Phase 2: Automatic Rotation

### 2.1 Renewal Script

```bash
#!/bin/bash
# scripts/renew_certs.sh

set -euo pipefail

LOG_FILE="/var/log/aragora/cert-renewal.log"
SLACK_WEBHOOK="${SLACK_WEBHOOK_URL:-}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

alert() {
    log "ALERT: $1"
    if [ -n "${SLACK_WEBHOOK}" ]; then
        curl -s -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"Certificate Renewal: $1\"}" \
            "${SLACK_WEBHOOK}"
    fi
}

log "Starting certificate renewal check..."

# Attempt renewal
if docker compose run --rm certbot renew --quiet; then
    log "Certificate renewal completed successfully"

    # Check if certificates were actually renewed
    if docker compose run --rm certbot certificates 2>&1 | grep -q "VALID"; then
        log "Certificates are valid"

        # Reload nginx to pick up new certificates
        if docker compose exec nginx nginx -s reload; then
            log "Nginx reloaded successfully"
            alert "Certificates renewed and nginx reloaded successfully"
        else
            alert "WARNING: Nginx reload failed after certificate renewal"
        fi
    fi
else
    alert "ERROR: Certificate renewal failed"
    exit 1
fi
```

### 2.2 Cron Schedule

```bash
# /etc/cron.d/aragora-cert-renewal

# Check for renewal twice daily (Let's Encrypt recommends this)
0 0,12 * * * root /opt/aragora/scripts/renew_certs.sh >> /var/log/aragora/cert-renewal.log 2>&1

# Weekly certificate expiration check
0 9 * * 1 root /opt/aragora/scripts/check_cert_expiry.sh
```

### 2.3 Expiration Check Script

```bash
#!/bin/bash
# scripts/check_cert_expiry.sh

set -euo pipefail

DOMAIN="aragora.example.com"
WARN_DAYS=14
CRIT_DAYS=7
SLACK_WEBHOOK="${SLACK_WEBHOOK_URL:-}"

alert() {
    local severity="$1"
    local message="$2"
    if [ -n "${SLACK_WEBHOOK}" ]; then
        curl -s -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"[$severity] Certificate Alert: $message\"}" \
            "${SLACK_WEBHOOK}"
    fi
    echo "[$severity] $message"
}

# Get certificate expiration date
EXPIRY=$(echo | openssl s_client -servername "${DOMAIN}" -connect "${DOMAIN}:443" 2>/dev/null | \
    openssl x509 -noout -enddate 2>/dev/null | \
    cut -d= -f2)

if [ -z "${EXPIRY}" ]; then
    alert "CRITICAL" "Could not retrieve certificate for ${DOMAIN}"
    exit 2
fi

# Calculate days until expiration
EXPIRY_EPOCH=$(date -d "${EXPIRY}" +%s)
NOW_EPOCH=$(date +%s)
DAYS_LEFT=$(( (EXPIRY_EPOCH - NOW_EPOCH) / 86400 ))

echo "Certificate for ${DOMAIN} expires in ${DAYS_LEFT} days (${EXPIRY})"

if [ "${DAYS_LEFT}" -le "${CRIT_DAYS}" ]; then
    alert "CRITICAL" "${DOMAIN} certificate expires in ${DAYS_LEFT} days!"
    exit 2
elif [ "${DAYS_LEFT}" -le "${WARN_DAYS}" ]; then
    alert "WARNING" "${DOMAIN} certificate expires in ${DAYS_LEFT} days"
    exit 1
else
    echo "OK: Certificate valid for ${DAYS_LEFT} days"
    exit 0
fi
```

---

## Phase 3: Manual Rotation

### 3.1 Force Renewal

```bash
#!/bin/bash
# scripts/force_renew.sh

set -euo pipefail

DOMAIN="aragora.example.com"

echo "Forcing certificate renewal for ${DOMAIN}..."

# Force renewal even if not due
docker compose run --rm certbot certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    --force-renewal \
    -d "${DOMAIN}"

# Reload nginx
docker compose exec nginx nginx -s reload

echo "Certificate renewed and nginx reloaded"
```

### 3.2 Replace with Custom Certificate

```bash
#!/bin/bash
# scripts/install_custom_cert.sh

set -euo pipefail

DOMAIN="aragora.example.com"
CERT_FILE="$1"
KEY_FILE="$2"
CHAIN_FILE="${3:-}"

# Validate inputs
if [ ! -f "${CERT_FILE}" ] || [ ! -f "${KEY_FILE}" ]; then
    echo "Usage: $0 <cert.pem> <key.pem> [chain.pem]"
    exit 1
fi

# Validate certificate and key match
CERT_MOD=$(openssl x509 -noout -modulus -in "${CERT_FILE}" | openssl md5)
KEY_MOD=$(openssl rsa -noout -modulus -in "${KEY_FILE}" | openssl md5)

if [ "${CERT_MOD}" != "${KEY_MOD}" ]; then
    echo "ERROR: Certificate and key do not match"
    exit 1
fi

# Backup existing certificates
BACKUP_DIR="/opt/aragora/cert-backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${BACKUP_DIR}"
cp -r /opt/aragora/certs/* "${BACKUP_DIR}/" 2>/dev/null || true

# Install new certificates
cp "${CERT_FILE}" "/opt/aragora/certs/${DOMAIN}.crt"
cp "${KEY_FILE}" "/opt/aragora/certs/${DOMAIN}.key"

if [ -n "${CHAIN_FILE}" ] && [ -f "${CHAIN_FILE}" ]; then
    cat "${CERT_FILE}" "${CHAIN_FILE}" > "/opt/aragora/certs/${DOMAIN}.fullchain.crt"
fi

# Set permissions
chmod 644 "/opt/aragora/certs/${DOMAIN}.crt"
chmod 600 "/opt/aragora/certs/${DOMAIN}.key"

# Test nginx configuration
if docker compose exec nginx nginx -t; then
    echo "Nginx configuration valid"
    docker compose exec nginx nginx -s reload
    echo "Nginx reloaded with new certificate"
else
    echo "ERROR: Nginx configuration invalid, rolling back"
    cp "${BACKUP_DIR}"/* /opt/aragora/certs/
    exit 1
fi
```

---

## Phase 4: Zero-Downtime Rotation

### 4.1 Blue-Green Certificate Update

```bash
#!/bin/bash
# scripts/zero_downtime_rotate.sh

set -euo pipefail

DOMAIN="aragora.example.com"

echo "Starting zero-downtime certificate rotation..."

# 1. Obtain new certificate (certbot keeps both old and new)
docker compose run --rm certbot renew

# 2. Test new certificate with a temporary nginx instance
docker run --rm \
    -v certbot-certs:/etc/letsencrypt:ro \
    -v ./nginx/conf.d:/etc/nginx/conf.d:ro \
    nginx:alpine nginx -t

if [ $? -ne 0 ]; then
    echo "ERROR: New certificate configuration invalid"
    exit 1
fi

# 3. Graceful reload (maintains existing connections)
docker compose exec nginx nginx -s reload

# 4. Verify new certificate is being served
sleep 5
NEW_EXPIRY=$(echo | openssl s_client -servername "${DOMAIN}" -connect "${DOMAIN}:443" 2>/dev/null | \
    openssl x509 -noout -enddate | cut -d= -f2)

echo "New certificate expiry: ${NEW_EXPIRY}"
echo "Zero-downtime rotation complete"
```

### 4.2 Rolling Update for Multiple Instances

```bash
#!/bin/bash
# scripts/rolling_cert_update.sh

set -euo pipefail

INSTANCES=("nginx-1" "nginx-2" "nginx-3")

for instance in "${INSTANCES[@]}"; do
    echo "Updating ${instance}..."

    # Copy new certificates
    docker cp /opt/aragora/certs/. "${instance}:/etc/nginx/ssl/"

    # Reload configuration
    docker exec "${instance}" nginx -s reload

    # Wait for reload to complete
    sleep 5

    # Health check
    if docker exec "${instance}" nginx -t; then
        echo "${instance} updated successfully"
    else
        echo "ERROR: ${instance} failed health check"
        exit 1
    fi
done

echo "All instances updated"
```

---

## Phase 5: Wildcard Certificates

### 5.1 DNS Challenge for Wildcard

```bash
#!/bin/bash
# scripts/wildcard_cert.sh

set -euo pipefail

DOMAIN="example.com"
EMAIL="admin@example.com"

# Wildcard requires DNS challenge
docker compose run --rm certbot certonly \
    --manual \
    --preferred-challenges dns \
    --email "${EMAIL}" \
    --agree-tos \
    -d "*.${DOMAIN}" \
    -d "${DOMAIN}"

# Follow prompts to add DNS TXT record
# Then verify and reload nginx
```

### 5.2 Automated DNS Challenge (Cloudflare)

```yaml
# docker-compose.yml addition

  certbot:
    image: certbot/dns-cloudflare:latest
    volumes:
      - certbot-certs:/etc/letsencrypt
      - ./cloudflare.ini:/etc/cloudflare.ini:ro
    command: certonly --dns-cloudflare --dns-cloudflare-credentials /etc/cloudflare.ini -d "*.example.com" -d "example.com"
```

```ini
# cloudflare.ini
dns_cloudflare_api_token = YOUR_CLOUDFLARE_API_TOKEN
```

---

## Monitoring

### Prometheus Metrics

```yaml
# prometheus/rules/certificates.yml

groups:
  - name: certificates
    rules:
      - alert: CertificateExpiringSoon
        expr: probe_ssl_earliest_cert_expiry - time() < 86400 * 14
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "Certificate expires in less than 14 days"
          description: "{{ $labels.instance }} certificate expires in {{ $value | humanizeDuration }}"

      - alert: CertificateExpiryCritical
        expr: probe_ssl_earliest_cert_expiry - time() < 86400 * 7
        for: 1h
        labels:
          severity: critical
        annotations:
          summary: "Certificate expires in less than 7 days"

      - alert: CertificateExpired
        expr: probe_ssl_earliest_cert_expiry - time() < 0
        labels:
          severity: critical
        annotations:
          summary: "Certificate has expired!"
```

### Blackbox Exporter Config

```yaml
# prometheus/blackbox.yml

modules:
  https_2xx:
    prober: http
    timeout: 5s
    http:
      valid_http_versions: ["HTTP/1.1", "HTTP/2"]
      valid_status_codes: [200]
      method: GET
      tls_config:
        insecure_skip_verify: false
```

---

## Troubleshooting

### Certificate Not Updating

```bash
# Check certbot logs
docker compose logs certbot

# Verify ACME challenge is accessible
curl -v http://aragora.example.com/.well-known/acme-challenge/test

# Check certificate files
docker compose exec nginx ls -la /etc/letsencrypt/live/

# Verify nginx is reading correct certificate
docker compose exec nginx nginx -T | grep ssl_certificate
```

### SSL Handshake Failures

```bash
# Test SSL connection
openssl s_client -connect aragora.example.com:443 -servername aragora.example.com

# Check certificate chain
openssl s_client -connect aragora.example.com:443 -showcerts

# Verify certificate validity
echo | openssl s_client -servername aragora.example.com -connect aragora.example.com:443 2>/dev/null | openssl x509 -noout -dates
```

### Permission Issues

```bash
# Fix certificate permissions
docker compose exec nginx chmod 644 /etc/letsencrypt/live/*/fullchain.pem
docker compose exec nginx chmod 600 /etc/letsencrypt/live/*/privkey.pem

# Check nginx user can read certificates
docker compose exec nginx cat /etc/letsencrypt/live/aragora.example.com/fullchain.pem > /dev/null
```

### Rollback Procedure

```bash
#!/bin/bash
# scripts/rollback_cert.sh

BACKUP_DIR="/opt/aragora/cert-backups"
LATEST_BACKUP=$(ls -t "${BACKUP_DIR}" | head -1)

if [ -z "${LATEST_BACKUP}" ]; then
    echo "No backup found"
    exit 1
fi

echo "Rolling back to ${LATEST_BACKUP}..."
cp "${BACKUP_DIR}/${LATEST_BACKUP}"/* /opt/aragora/certs/
docker compose exec nginx nginx -s reload
echo "Rollback complete"
```

---

## Reference

### Key Files

| File | Purpose |
|------|---------|
| `/etc/letsencrypt/live/<domain>/` | Certificate files |
| `/etc/letsencrypt/renewal/<domain>.conf` | Renewal configuration |
| `/var/log/letsencrypt/` | Certbot logs |

### Certificate Files

| File | Contents |
|------|----------|
| `fullchain.pem` | Certificate + intermediate chain |
| `privkey.pem` | Private key |
| `cert.pem` | Certificate only |
| `chain.pem` | Intermediate certificates |

### Useful Commands

```bash
# View certificate info
openssl x509 -in cert.pem -noout -text

# Check certificate expiration
openssl x509 -in cert.pem -noout -enddate

# Verify certificate chain
openssl verify -CAfile chain.pem cert.pem

# Test HTTPS connection
curl -vI https://aragora.example.com
```

---

**Document Owner:** Platform Team
**Review Cycle:** Quarterly
