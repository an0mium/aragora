---
title: TLS Configuration Guide
description: TLS Configuration Guide
---

# TLS Configuration Guide

This document covers TLS/SSL configuration for Aragora deployments.

## Table of Contents

- [Overview](#overview)
- [Kubernetes (cert-manager)](#kubernetes-cert-manager)
- [Manual Certificate Setup](#manual-certificate-setup)
- [Self-Signed Certificates](#self-signed-certificates)
- [Certificate Monitoring](#certificate-monitoring)
- [Troubleshooting](#troubleshooting)

---

## Overview

### TLS Requirements

| Environment | TLS Required | Certificate Type |
|-------------|--------------|------------------|
| Production | **Yes** | Let's Encrypt / Commercial |
| Staging | Yes | Let's Encrypt staging |
| Development | Optional | Self-signed |

### Supported TLS Versions

- **TLS 1.3**: Recommended (strongest security)
- **TLS 1.2**: Supported (required for some clients)
- **TLS 1.1/1.0**: Not supported (deprecated)

### Cipher Suites

Recommended cipher configuration:

```nginx
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
ssl_prefer_server_ciphers off;
```

---

## Kubernetes (cert-manager)

### Prerequisites

1. Kubernetes cluster 1.26+
2. cert-manager installed
3. ClusterIssuer configured

### Install cert-manager

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.14.0/cert-manager.yaml

# Verify installation
kubectl get pods -n cert-manager
```

### Create ClusterIssuer (Let's Encrypt)

```yaml
# cluster-issuer-prod.yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: ops@aragora.ai
    privateKeySecretRef:
      name: letsencrypt-prod-key
    solvers:
      - http01:
          ingress:
            class: nginx
```

```yaml
# cluster-issuer-staging.yaml (for testing)
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-staging
spec:
  acme:
    server: https://acme-staging-v02.api.letsencrypt.org/directory
    email: ops@aragora.ai
    privateKeySecretRef:
      name: letsencrypt-staging-key
    solvers:
      - http01:
          ingress:
            class: nginx
```

```bash
# Apply issuers
kubectl apply -f cluster-issuer-prod.yaml
kubectl apply -f cluster-issuer-staging.yaml
```

### Deploy with TLS

The Aragora ingress already includes cert-manager annotations:

```yaml
# deploy/k8s/ingress.yaml (excerpt)
metadata:
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
    - hosts:
        - aragora.example.com
      secretName: aragora-tls
```

**Deploy:**

```bash
# Update hostname
sed -i 's/aragora.example.com/your-domain.com/g' deploy/k8s/ingress.yaml

# Apply
kubectl apply -f deploy/k8s/ingress.yaml -n aragora

# Check certificate status
kubectl get certificate -n aragora
kubectl describe certificate aragora-tls -n aragora
```

### DNS Challenge (Wildcard Certificates)

For wildcard certificates (`*.aragora.ai`), use DNS-01 challenge:

```yaml
# cluster-issuer-dns.yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-dns
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: ops@aragora.ai
    privateKeySecretRef:
      name: letsencrypt-dns-key
    solvers:
      - dns01:
          cloudflare:
            email: ops@aragora.ai
            apiTokenSecretRef:
              name: cloudflare-api-token
              key: api-token
```

**Create Cloudflare API token secret:**

```bash
kubectl create secret generic cloudflare-api-token \
  --from-literal=api-token=YOUR_CLOUDFLARE_API_TOKEN \
  -n cert-manager
```

---

## Manual Certificate Setup

### Let's Encrypt with Certbot

**Install Certbot:**

```bash
# Ubuntu/Debian
apt-get install certbot

# Or via pip
pip install certbot
```

**Obtain Certificate:**

```bash
# HTTP challenge (requires port 80 accessible)
certbot certonly --standalone -d aragora.yourdomain.com

# Or with webroot (if web server running)
certbot certonly --webroot -w /var/www/html -d aragora.yourdomain.com
```

**Configure Nginx:**

```nginx
server {
    listen 443 ssl http2;
    server_name aragora.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/aragora.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/aragora.yourdomain.com/privkey.pem;

    # Modern TLS configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:10m;
    ssl_session_tickets off;

    # HSTS (uncomment after testing)
    # add_header Strict-Transport-Security "max-age=63072000" always;

    # OCSP Stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    resolver 1.1.1.1 8.8.8.8 valid=300s;
    resolver_timeout 5s;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name aragora.yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

**Auto-renewal:**

```bash
# Test renewal
certbot renew --dry-run

# Add to crontab (runs twice daily)
echo "0 0,12 * * * root certbot renew --quiet --post-hook 'systemctl reload nginx'" >> /etc/crontab
```

---

## Self-Signed Certificates

For development environments only.

### Generate Self-Signed Certificate

```bash
# Create certificate directory
mkdir -p /etc/aragora/certs
cd /etc/aragora/certs

# Generate private key
openssl genrsa -out aragora.key 4096

# Generate certificate signing request
openssl req -new -key aragora.key -out aragora.csr \
  -subj "/CN=localhost/O=Aragora Development/C=US"

# Generate self-signed certificate (valid 365 days)
openssl x509 -req -days 365 -in aragora.csr \
  -signkey aragora.key -out aragora.crt

# Set permissions
chmod 600 aragora.key
chmod 644 aragora.crt
```

### With Subject Alternative Names (SAN)

```bash
# Create OpenSSL config
cat > san.cnf << EOF
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
CN = localhost

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = aragora.local
IP.1 = 127.0.0.1
IP.2 = ::1
EOF

# Generate certificate with SANs
openssl req -x509 -nodes -days 365 \
  -newkey rsa:4096 \
  -keyout aragora.key \
  -out aragora.crt \
  -config san.cnf
```

### Trust Self-Signed Certificate (Development)

**macOS:**
```bash
sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain aragora.crt
```

**Linux:**
```bash
sudo cp aragora.crt /usr/local/share/ca-certificates/
sudo update-ca-certificates
```

**Windows (PowerShell as Admin):**
```powershell
Import-Certificate -FilePath aragora.crt -CertStoreLocation Cert:\LocalMachine\Root
```

---

## Certificate Monitoring

### Prometheus Alerts

The alerting rules already include TLS certificate monitoring:

```yaml
# deploy/observability/alerts.rules (excerpt)
- alert: TLSCertificateExpiringSoon
  expr: (probe_ssl_earliest_cert_expiry - time()) / 86400 < 7
  labels:
    severity: critical
  annotations:
    summary: "TLS certificate expiring within 7 days"

- alert: TLSCertificateExpiringWarning
  expr: (probe_ssl_earliest_cert_expiry - time()) / 86400 < 30
  labels:
    severity: warning
  annotations:
    summary: "TLS certificate expiring within 30 days"
```

### Enable Blackbox Exporter

To enable certificate monitoring, deploy the Blackbox Exporter:

```yaml
# blackbox-exporter.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: blackbox-exporter-config
data:
  blackbox.yml: |
    modules:
      http_2xx:
        prober: http
        timeout: 5s
        http:
          valid_http_versions: ["HTTP/1.1", "HTTP/2.0"]
          valid_status_codes: []
          method: GET
          follow_redirects: true
          fail_if_ssl: false
          fail_if_not_ssl: true
          tls_config:
            insecure_skip_verify: false
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: blackbox-exporter
spec:
  replicas: 1
  selector:
    matchLabels:
      app: blackbox-exporter
  template:
    metadata:
      labels:
        app: blackbox-exporter
    spec:
      containers:
      - name: blackbox-exporter
        image: prom/blackbox-exporter:v0.24.0
        ports:
        - containerPort: 9115
        volumeMounts:
        - name: config
          mountPath: /etc/blackbox_exporter
      volumes:
      - name: config
        configMap:
          name: blackbox-exporter-config
```

**Add Prometheus scrape config:**

```yaml
# Add to prometheus.yml
scrape_configs:
  - job_name: 'blackbox-tls'
    metrics_path: /probe
    params:
      module: [http_2xx]
    static_configs:
      - targets:
        - https://aragora.yourdomain.com
    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_target
      - source_labels: [__param_target]
        target_label: instance
      - target_label: __address__
        replacement: blackbox-exporter:9115
```

### Manual Certificate Check

```bash
# Check certificate expiration
echo | openssl s_client -connect aragora.yourdomain.com:443 2>/dev/null | \
  openssl x509 -noout -enddate

# Check full certificate chain
echo | openssl s_client -connect aragora.yourdomain.com:443 -showcerts 2>/dev/null | \
  openssl x509 -noout -text

# Check certificate via curl
curl -vI https://aragora.yourdomain.com 2>&1 | grep -A 6 "Server certificate"
```

---

## Troubleshooting

### Certificate Not Ready (cert-manager)

```bash
# Check certificate status
kubectl describe certificate aragora-tls -n aragora

# Check certificate request
kubectl get certificaterequest -n aragora
kubectl describe certificaterequest aragora-tls-xxxxx -n aragora

# Check challenges (HTTP-01)
kubectl get challenges -n aragora
kubectl describe challenge aragora-tls-xxxxx -n aragora

# Check cert-manager logs
kubectl logs -n cert-manager deployment/cert-manager -f
```

**Common issues:**

1. **HTTP-01 challenge failing**: Ensure port 80 is accessible and `/.well-known/acme-challenge/` path is not blocked
2. **DNS-01 challenge failing**: Verify DNS API credentials
3. **Rate limited**: Wait 1 hour (Let's Encrypt rate limits)

### Certificate Mismatch

```bash
# Verify certificate matches domain
echo | openssl s_client -connect aragora.yourdomain.com:443 2>/dev/null | \
  openssl x509 -noout -subject -issuer

# Check SANs
echo | openssl s_client -connect aragora.yourdomain.com:443 2>/dev/null | \
  openssl x509 -noout -ext subjectAltName
```

### Mixed Content / Insecure Requests

Ensure all resources use HTTPS. Check:

1. API calls use `https://` URLs
2. WebSocket connections use `wss://`
3. Static assets served over HTTPS

**Headers to add:**

```nginx
add_header Content-Security-Policy "upgrade-insecure-requests" always;
```

### HSTS Issues

If you enabled HSTS and need to disable it:

1. Set `max-age=0` header
2. Wait for browser cache to expire
3. Remove HSTS header entirely

```nginx
# Disable HSTS
add_header Strict-Transport-Security "max-age=0" always;
```

### Certificate Chain Issues

```bash
# Check certificate chain
openssl s_client -connect aragora.yourdomain.com:443 -servername aragora.yourdomain.com </dev/null 2>/dev/null | \
  openssl x509 -noout -text | grep -A 1 "Issuer:"

# Verify chain integrity
openssl verify -CAfile /etc/ssl/certs/ca-certificates.crt fullchain.pem
```

---

## Security Best Practices

### HTTP Strict Transport Security (HSTS)

After confirming HTTPS works correctly:

```nginx
# Enable HSTS (1 year)
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
```

### OCSP Stapling

Improves TLS handshake performance:

```nginx
ssl_stapling on;
ssl_stapling_verify on;
ssl_trusted_certificate /etc/letsencrypt/live/domain/chain.pem;
resolver 1.1.1.1 8.8.8.8 valid=300s;
```

### Key Rotation

Rotate TLS private keys annually:

```bash
# Force certificate renewal with new key
certbot renew --force-renewal --reuse-key=false

# Or for cert-manager, delete the secret
kubectl delete secret aragora-tls -n aragora
```

### Certificate Transparency

Monitor CT logs for unauthorized certificates:

- [crt.sh](https://crt.sh/?q=yourdomain.com)
- [Google CT Dashboard](https://transparencyreport.google.com/https/certificates)

---

*Last updated: 2026-01-13*
