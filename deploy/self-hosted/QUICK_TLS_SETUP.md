# TLS Quick Setup Guide

This guide helps you enable HTTPS for your Aragora deployment quickly.

## Option 1: Traefik with Let's Encrypt (Recommended)

For production deployments with automatic certificate renewal.

### 1. Create `docker-compose.override.yml`

```yaml
services:
  traefik:
    image: traefik:v3.0
    command:
      - "--api.dashboard=true"
      - "--providers.docker=true"
      - "--providers.docker.exposedbydefault=false"
      - "--entrypoints.web.address=:80"
      - "--entrypoints.websecure.address=:443"
      - "--certificatesresolvers.letsencrypt.acme.httpchallenge=true"
      - "--certificatesresolvers.letsencrypt.acme.httpchallenge.entrypoint=web"
      - "--certificatesresolvers.letsencrypt.acme.email=${ACME_EMAIL:-admin@example.com}"
      - "--certificatesresolvers.letsencrypt.acme.storage=/letsencrypt/acme.json"
      # Redirect HTTP to HTTPS
      - "--entrypoints.web.http.redirections.entrypoint.to=websecure"
      - "--entrypoints.web.http.redirections.entrypoint.scheme=https"
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - "/var/run/docker.sock:/var/run/docker.sock:ro"
      - "letsencrypt:/letsencrypt"
    networks:
      - aragora-network

  aragora:
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.aragora.rule=Host(`${ARAGORA_DOMAIN:-aragora.example.com}`)"
      - "traefik.http.routers.aragora.entrypoints=websecure"
      - "traefik.http.routers.aragora.tls.certresolver=letsencrypt"
      - "traefik.http.services.aragora.loadbalancer.server.port=8080"
    # Remove direct port exposure when using Traefik
    ports: []

volumes:
  letsencrypt:

networks:
  aragora-network:
    external: true
```

### 2. Add to `.env`

```bash
ARAGORA_DOMAIN=aragora.yourdomain.com
ACME_EMAIL=admin@yourdomain.com
```

### 3. Deploy

```bash
# Create network if not exists
docker network create aragora-network 2>/dev/null || true

# Start with TLS
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

---

## Option 2: Self-Signed Certificate (Development)

For local testing or internal networks.

### 1. Generate Certificate

```bash
# Create certs directory
mkdir -p certs

# Generate self-signed certificate (valid for 1 year)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout certs/aragora.key \
  -out certs/aragora.crt \
  -subj "/CN=localhost/O=Aragora/C=US"

# Set permissions
chmod 600 certs/aragora.key
chmod 644 certs/aragora.crt
```

### 2. Update `.env`

```bash
ARAGORA_TLS_CERT=/app/certs/aragora.crt
ARAGORA_TLS_KEY=/app/certs/aragora.key
```

### 3. Mount Certificates

Add to `docker-compose.override.yml`:

```yaml
services:
  aragora:
    volumes:
      - ./certs:/app/certs:ro
    ports:
      - "8443:8443"  # HTTPS port
    environment:
      - ARAGORA_TLS_CERT=/app/certs/aragora.crt
      - ARAGORA_TLS_KEY=/app/certs/aragora.key
```

---

## Option 3: Nginx Reverse Proxy

For existing nginx infrastructure.

### 1. Nginx Configuration

Create `/etc/nginx/sites-available/aragora`:

```nginx
upstream aragora {
    server 127.0.0.1:8080;
    keepalive 32;
}

server {
    listen 80;
    server_name aragora.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name aragora.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/aragora.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/aragora.yourdomain.com/privkey.pem;

    # SSL settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;
    ssl_stapling on;
    ssl_stapling_verify on;

    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;

    location / {
        proxy_pass http://aragora;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }

    location /ws {
        proxy_pass http://aragora;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
```

### 2. Get Let's Encrypt Certificate

```bash
# Install certbot
apt install certbot python3-certbot-nginx

# Get certificate
certbot --nginx -d aragora.yourdomain.com
```

### 3. Enable Site

```bash
ln -s /etc/nginx/sites-available/aragora /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
```

---

## Option 4: Cloudflare Proxy

For Cloudflare users (simplest for public sites).

### 1. DNS Setup

In Cloudflare dashboard:
- Add A record pointing to your server IP
- Enable proxy (orange cloud)

### 2. SSL/TLS Settings

In Cloudflare SSL/TLS:
- Set encryption mode to "Full (strict)"
- Enable "Always Use HTTPS"
- Enable "Automatic HTTPS Rewrites"

### 3. Origin Certificate (Optional)

For "Full (strict)" mode:
1. Go to SSL/TLS > Origin Server
2. Create Certificate
3. Download certificate and key
4. Configure as Option 2 (Self-Signed)

---

## Verification

After setup, verify TLS:

```bash
# Test HTTPS endpoint
curl -v https://aragora.yourdomain.com/healthz

# Check certificate
openssl s_client -connect aragora.yourdomain.com:443 -servername aragora.yourdomain.com

# Test WebSocket over TLS
wscat -c wss://aragora.yourdomain.com/ws/events
```

---

## Security Best Practices

1. **Use TLS 1.2+** - Disable older protocols
2. **Enable HSTS** - Prevents downgrade attacks
3. **Use strong ciphers** - Modern cipher suites only
4. **Certificate renewal** - Automate with certbot or Traefik
5. **Test regularly** - Use SSL Labs (ssllabs.com/ssltest)

---

## Troubleshooting

### Certificate not working
```bash
# Check certificate validity
openssl x509 -in /path/to/cert.crt -text -noout

# Verify key matches certificate
openssl x509 -noout -modulus -in cert.crt | openssl md5
openssl rsa -noout -modulus -in cert.key | openssl md5
# Both should output the same hash
```

### Mixed content warnings
Ensure all resources use HTTPS:
```bash
# In .env
ARAGORA_BASE_URL=https://aragora.yourdomain.com
```

### WebSocket connection failed
Check nginx/traefik is forwarding upgrade headers:
```nginx
proxy_set_header Upgrade $http_upgrade;
proxy_set_header Connection "upgrade";
```
