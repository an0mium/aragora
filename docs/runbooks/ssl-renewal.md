# SSL/TLS Certificate Renewal Runbook

Procedures for TLS certificate management and renewal.

## Certificate Types

| Type | Purpose | Renewal |
|------|---------|---------|
| Let's Encrypt | Public endpoints | Auto (90 days) |
| Internal CA | Service mesh | Manual/Auto |
| Purchased | Enterprise requirements | Manual |

---

## Let's Encrypt with cert-manager

### Installation

```bash
# Install cert-manager
helm repo add jetstack https://charts.jetstack.io
helm repo update

helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --set installCRDs=true
```

### Cluster Issuer

```yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@company.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
```

### Certificate

```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: aragora-tls
  namespace: aragora
spec:
  secretName: aragora-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  commonName: aragora.company.com
  dnsNames:
  - aragora.company.com
  - api.aragora.company.com
```

### Verify Auto-Renewal

```bash
# Check certificate status
kubectl get certificates -n aragora

# Check certificate details
kubectl describe certificate aragora-tls -n aragora

# Check expiry
kubectl get secret aragora-tls -n aragora -o jsonpath='{.data.tls\.crt}' | \
  base64 -d | openssl x509 -noout -dates
```

---

## Manual Certificate Renewal

### Generate CSR

```bash
# Generate private key
openssl genrsa -out aragora.key 4096

# Generate CSR
openssl req -new -key aragora.key -out aragora.csr \
  -subj "/C=US/ST=CA/L=SF/O=Company/CN=aragora.company.com" \
  -addext "subjectAltName=DNS:aragora.company.com,DNS:api.aragora.company.com"
```

### Update Certificate

```bash
# Create/update secret
kubectl create secret tls aragora-tls \
  --cert=aragora.crt \
  --key=aragora.key \
  --namespace=aragora \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart ingress controller to pick up new cert
kubectl rollout restart deployment/ingress-nginx-controller -n ingress-nginx
```

### Docker Compose

```bash
# Update certificates
cp new-cert.crt /etc/nginx/ssl/aragora.crt
cp new-key.key /etc/nginx/ssl/aragora.key

# Reload nginx
docker compose exec nginx nginx -s reload
```

---

## Certificate Monitoring

### Alert for Expiring Certificates

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: cert-alerts
  namespace: monitoring
spec:
  groups:
  - name: certificates
    rules:
    - alert: CertificateExpiringSoon
      expr: |
        (certmanager_certificate_expiration_timestamp_seconds - time()) < 86400 * 14
      for: 1h
      labels:
        severity: warning
      annotations:
        summary: "Certificate expiring soon"
        description: "Certificate {{ $labels.name }} expires in {{ $value | humanizeDuration }}"

    - alert: CertificateExpired
      expr: |
        certmanager_certificate_expiration_timestamp_seconds < time()
      for: 0m
      labels:
        severity: critical
      annotations:
        summary: "Certificate expired"
        description: "Certificate {{ $labels.name }} has expired"
```

### Check Expiry Script

```bash
#!/bin/bash
# check-certs.sh

DOMAIN="aragora.company.com"
WARN_DAYS=30

EXPIRY=$(echo | openssl s_client -servername $DOMAIN -connect $DOMAIN:443 2>/dev/null | \
  openssl x509 -noout -enddate | cut -d= -f2)

EXPIRY_EPOCH=$(date -d "$EXPIRY" +%s)
NOW_EPOCH=$(date +%s)
DAYS_LEFT=$(( ($EXPIRY_EPOCH - $NOW_EPOCH) / 86400 ))

if [ $DAYS_LEFT -lt $WARN_DAYS ]; then
  echo "WARNING: Certificate expires in $DAYS_LEFT days"
  exit 1
fi

echo "OK: Certificate valid for $DAYS_LEFT days"
exit 0
```

---

## Troubleshooting

### Certificate Not Renewing

```bash
# Check cert-manager logs
kubectl logs -n cert-manager deployment/cert-manager

# Check certificate events
kubectl describe certificate aragora-tls -n aragora

# Check ACME challenges
kubectl get challenges -n aragora

# Force renewal
kubectl delete certificate aragora-tls -n aragora
# cert-manager will recreate it
```

### Invalid Certificate Chain

```bash
# Verify chain
openssl s_client -connect aragora.company.com:443 -showcerts

# Check intermediate certificates
curl https://aragora.company.com -v 2>&1 | grep -i "ssl\|tls\|cert"
```

---

## Rollback

If new certificate causes issues:

```bash
# Restore previous secret from backup
kubectl apply -f backup/aragora-tls-secret.yaml

# Or recreate from backup files
kubectl create secret tls aragora-tls \
  --cert=backup/aragora.crt \
  --key=backup/aragora.key \
  --namespace=aragora \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart pods
kubectl rollout restart deployment/aragora-api -n aragora
```

---

## See Also

- [Incident Response Runbook](incident-response.md)
- [Monitoring Setup Runbook](monitoring-setup.md)
