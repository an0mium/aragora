# Security Deployment Guide

Comprehensive guide for deploying Aragora securely in production environments.

## Table of Contents

- [Security Headers](#security-headers)
- [WAF Configuration](#waf-configuration)
- [TLS Configuration](#tls-configuration)
- [Network Security](#network-security)
- [Secrets Management](#secrets-management)
- [Monitoring and Alerting](#monitoring-and-alerting)
- [Application Security](#application-security)

---

## Security Headers

### Required Headers

Apply these headers to all responses in your reverse proxy (nginx, Cloudflare, etc.):

```nginx
# nginx.conf

# Prevent MIME type sniffing
add_header X-Content-Type-Options "nosniff" always;

# Prevent clickjacking
add_header X-Frame-Options "DENY" always;

# Enable XSS filter (legacy browsers)
add_header X-XSS-Protection "1; mode=block" always;

# Enforce HTTPS
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;

# Control referrer information
add_header Referrer-Policy "strict-origin-when-cross-origin" always;

# Restrict browser features
add_header Permissions-Policy "geolocation=(), microphone=(), camera=(), payment=()" always;
```

### Content Security Policy

Configure CSP based on your deployment:

```nginx
# Production CSP - strict
add_header Content-Security-Policy "
  default-src 'self';
  script-src 'self' 'unsafe-inline' https://js.stripe.com;
  style-src 'self' 'unsafe-inline';
  img-src 'self' data: https:;
  font-src 'self';
  connect-src 'self' wss://your-domain.com https://api.stripe.com;
  frame-src https://js.stripe.com;
  object-src 'none';
  base-uri 'self';
  form-action 'self';
  frame-ancestors 'none';
  upgrade-insecure-requests;
" always;
```

### CORS Headers

Configure CORS in your application or reverse proxy:

```bash
# Environment variable
ARAGORA_ALLOWED_ORIGINS=https://app.yourdomain.com,https://yourdomain.com
```

```nginx
# nginx.conf
location /api/ {
    if ($request_method = 'OPTIONS') {
        add_header 'Access-Control-Allow-Origin' '$http_origin';
        add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS';
        add_header 'Access-Control-Allow-Headers' 'Authorization, Content-Type, X-Request-ID';
        add_header 'Access-Control-Allow-Credentials' 'true';
        add_header 'Access-Control-Max-Age' '86400';
        add_header 'Content-Length' '0';
        return 204;
    }
}
```

---

## WAF Configuration

### Cloudflare WAF

Recommended Cloudflare WAF rules for Aragora:

#### 1. Rate Limiting Rules

```
Rule: API Rate Limit
Expression: (http.request.uri.path contains "/api/")
Action: Challenge
Rate: 100 requests per 10 seconds per IP
```

```
Rule: Auth Endpoint Protection
Expression: (http.request.uri.path contains "/api/auth/login" or http.request.uri.path contains "/api/auth/register")
Action: Challenge
Rate: 5 requests per minute per IP
```

#### 2. SQL Injection Protection

```
Rule: SQL Injection Block
Expression: (
  http.request.uri.query contains "UNION" or
  http.request.uri.query contains "SELECT" or
  http.request.uri.query contains "INSERT" or
  http.request.uri.query contains "UPDATE" or
  http.request.uri.query contains "DELETE" or
  http.request.uri.query contains "--" or
  http.request.uri.query contains "'" and http.request.uri.query contains "OR"
)
Action: Block
```

#### 3. XSS Protection

```
Rule: XSS Block
Expression: (
  http.request.uri.query contains "<script" or
  http.request.uri.query contains "javascript:" or
  http.request.uri.query contains "onerror=" or
  http.request.uri.query contains "onload="
)
Action: Block
```

#### 4. Path Traversal Protection

```
Rule: Path Traversal Block
Expression: (
  http.request.uri.path contains ".." or
  http.request.uri.path contains "..%2f" or
  http.request.uri.path contains "%2e%2e"
)
Action: Block
```

#### 5. Bot Protection

```
Rule: Known Bad Bots
Expression: (
  cf.client.bot or
  http.user_agent contains "sqlmap" or
  http.user_agent contains "nikto" or
  http.user_agent contains "nmap"
)
Action: Challenge
```

### AWS WAF Rules

For AWS deployments, use these WAF rule sets:

```yaml
# AWS WAF Web ACL
Resources:
  AragoraWAF:
    Type: AWS::WAFv2::WebACL
    Properties:
      Name: aragora-waf
      Scope: REGIONAL
      DefaultAction:
        Allow: {}
      Rules:
        # AWS Managed Rule Sets
        - Name: AWSManagedRulesCommonRuleSet
          Priority: 1
          Statement:
            ManagedRuleGroupStatement:
              VendorName: AWS
              Name: AWSManagedRulesCommonRuleSet
          OverrideAction:
            None: {}
          VisibilityConfig:
            SampledRequestsEnabled: true
            CloudWatchMetricsEnabled: true
            MetricName: CommonRuleSet

        - Name: AWSManagedRulesKnownBadInputsRuleSet
          Priority: 2
          Statement:
            ManagedRuleGroupStatement:
              VendorName: AWS
              Name: AWSManagedRulesKnownBadInputsRuleSet
          OverrideAction:
            None: {}
          VisibilityConfig:
            SampledRequestsEnabled: true
            CloudWatchMetricsEnabled: true
            MetricName: KnownBadInputsRuleSet

        - Name: AWSManagedRulesSQLiRuleSet
          Priority: 3
          Statement:
            ManagedRuleGroupStatement:
              VendorName: AWS
              Name: AWSManagedRulesSQLiRuleSet
          OverrideAction:
            None: {}
          VisibilityConfig:
            SampledRequestsEnabled: true
            CloudWatchMetricsEnabled: true
            MetricName: SQLiRuleSet

        # Custom Rate Limiting
        - Name: RateLimitRule
          Priority: 4
          Statement:
            RateBasedStatement:
              Limit: 2000
              AggregateKeyType: IP
          Action:
            Block: {}
          VisibilityConfig:
            SampledRequestsEnabled: true
            CloudWatchMetricsEnabled: true
            MetricName: RateLimitRule
```

### nginx ModSecurity

For self-hosted deployments with nginx:

```nginx
# /etc/nginx/modsecurity/aragora.conf

# Enable ModSecurity
SecRuleEngine On

# SQL Injection
SecRule ARGS|ARGS_NAMES|REQUEST_COOKIES|REQUEST_COOKIES_NAMES|REQUEST_BODY|REQUEST_HEADERS|XML:/*|XML://@* "@detectSQLi" \
    "id:1,\
    phase:2,\
    block,\
    capture,\
    t:none,t:urlDecodeUni,\
    msg:'SQL Injection Attack',\
    logdata:'Matched Data: %{TX.0} found within %{MATCHED_VAR_NAME}',\
    tag:'attack-sqli'"

# XSS
SecRule ARGS|ARGS_NAMES|REQUEST_COOKIES|REQUEST_COOKIES_NAMES|REQUEST_BODY|REQUEST_HEADERS|XML:/*|XML://@* "@detectXSS" \
    "id:2,\
    phase:2,\
    block,\
    capture,\
    t:none,t:urlDecodeUni,t:htmlEntityDecode,t:jsDecode,\
    msg:'XSS Attack',\
    logdata:'Matched Data: %{TX.0} found within %{MATCHED_VAR_NAME}',\
    tag:'attack-xss'"

# Path Traversal
SecRule REQUEST_URI "@contains .." \
    "id:3,\
    phase:1,\
    block,\
    msg:'Path Traversal Attempt',\
    tag:'attack-lfi'"

# Rate Limiting (use with nginx limit_req)
# See nginx rate limiting below
```

---

## TLS Configuration

### nginx TLS Configuration

```nginx
# /etc/nginx/conf.d/ssl.conf

# TLS 1.2 and 1.3 only
ssl_protocols TLSv1.2 TLSv1.3;

# Strong cipher suites
ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
ssl_prefer_server_ciphers on;

# OCSP stapling
ssl_stapling on;
ssl_stapling_verify on;
resolver 8.8.8.8 8.8.4.4 valid=300s;
resolver_timeout 5s;

# SSL session caching
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 1d;
ssl_session_tickets off;

# DH parameters (generate with: openssl dhparam -out /etc/nginx/dhparam.pem 2048)
ssl_dhparam /etc/nginx/dhparam.pem;
```

### Certificate Requirements

- Use certificates from trusted CAs (Let's Encrypt, DigiCert, etc.)
- Enable automatic renewal
- Monitor certificate expiration

```bash
# Let's Encrypt with certbot
certbot certonly --nginx -d api.yourdomain.com -d app.yourdomain.com

# Auto-renewal cron
0 0 * * * certbot renew --quiet --post-hook "systemctl reload nginx"
```

---

## Network Security

### Firewall Rules

```bash
# Allow HTTPS only
ufw allow 443/tcp

# Allow SSH from specific IPs
ufw allow from <admin-ip> to any port 22

# Block everything else
ufw default deny incoming
ufw enable
```

### Internal Network Isolation

```yaml
# kubernetes/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: aragora-network-policy
  namespace: aragora
spec:
  podSelector:
    matchLabels:
      app: aragora
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: nginx-ingress
      ports:
        - protocol: TCP
          port: 8080
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: redis
      ports:
        - protocol: TCP
          port: 6379
    - to:
        - podSelector:
            matchLabels:
              app: postgres
      ports:
        - protocol: TCP
          port: 5432
    # Allow external HTTPS for LLM APIs
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
      ports:
        - protocol: TCP
          port: 443
```

---

## Secrets Management

### Environment Variables

**Never commit secrets to version control.**

```bash
# Production secrets (set in deployment platform)
ARAGORA_JWT_SECRET=<random-256-bit-key>
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Database credentials
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...
```

### Kubernetes Secrets

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: aragora-secrets
  namespace: aragora
type: Opaque
stringData:
  aragora-jwt-secret: ${ARAGORA_JWT_SECRET}
  stripe-secret-key: ${STRIPE_SECRET_KEY}
  anthropic-api-key: ${ANTHROPIC_API_KEY}
```

### HashiCorp Vault

For advanced secret management:

```python
# Example Vault integration
import hvac

client = hvac.Client(url='https://vault.internal:8200')
client.auth.kubernetes.login(role='aragora')

secrets = client.secrets.kv.read_secret_version(path='aragora/production')
ARAGORA_JWT_SECRET = secrets['data']['data']['aragora_jwt_secret']
```

---

## Monitoring and Alerting

### Security Alerts

Configure alerts for:

1. **Authentication Failures**
   - >5 failed logins per user per hour
   - >20 failed logins per IP per hour

2. **Rate Limiting**
   - >100 requests blocked per minute

3. **WAF Blocks**
   - Any SQL injection attempt
   - Any XSS attempt
   - Path traversal attempts

4. **Certificate Expiration**
   - Alert at 30 days before expiry
   - Critical at 7 days before expiry

### Prometheus Alerts

```yaml
# prometheus/alerts.yaml
groups:
  - name: security
    rules:
      - alert: HighFailedLogins
        expr: sum(rate(aragora_auth_failures_total[5m])) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High rate of failed login attempts

      - alert: WAFBlocksSpike
        expr: sum(rate(aragora_waf_blocks_total[5m])) > 50
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: High rate of WAF blocks - possible attack

      - alert: CertificateExpiringSoon
        expr: probe_ssl_earliest_cert_expiry - time() < 86400 * 7
        for: 1h
        labels:
          severity: critical
        annotations:
          summary: TLS certificate expires in less than 7 days
```

### Audit Logging

Enable comprehensive audit logging:

```bash
# Environment variable
ARAGORA_AUDIT_ENABLED=true
ARAGORA_AUDIT_RETENTION_DAYS=90
```

Audit log includes:
- All authentication events (login, logout, MFA)
- Authorization failures
- Admin actions
- Data access patterns
- API key usage

---

## Application Security

Aragora includes several application-level security features that are enabled by default.

### OIDC Token Validation

Token validation is enforced strictly in production mode:

```bash
# Environment variable (default: true in production)
ARAGORA_STRICT_TOKEN_VALIDATION=true
```

When enabled:
- ID tokens must have valid signatures (no fallback to userinfo endpoint)
- Expired tokens are rejected without exception
- Token claims (iss, aud, exp) are fully validated

### Tenant Isolation

Multi-tenant deployments enforce strict resource isolation:

```python
# Shared resources are validated at startup
# Only explicitly allowed resources can be shared across tenants:
ALLOWED_SHARED_RESOURCES = frozenset([
    "system_agents",     # System-provided agents
    "public_templates",  # Public workflow templates
])
```

Key protections:
- Query filters automatically include tenant_id
- Cross-tenant data access is logged and audited
- Shared resources are immutable and defined at startup

### RBAC Permission Validation

All route permissions are validated at startup:

```python
# Environment variable (default: false, set to true for strict mode)
ARAGORA_RBAC_STRICT_MODE=true
```

Features:
- Route permissions validated against SYSTEM_PERMISSIONS registry
- Undefined permissions logged as warnings (errors in strict mode)
- Wildcard permissions (e.g., `admin.*`) validated against defined permission prefixes
- O(1) cache invalidation using version-based keys

### Rate Limiting

Configure per-client and per-endpoint rate limits:

```bash
# Global rate limits
ARAGORA_RATE_LIMIT_REQUESTS=1000
ARAGORA_RATE_LIMIT_WINDOW_SECONDS=60

# Per-tier overrides
ARAGORA_RATE_LIMIT_FREE_TIER=100
ARAGORA_RATE_LIMIT_PRO_TIER=1000
ARAGORA_RATE_LIMIT_ENTERPRISE_TIER=10000
```

---

## Security Checklist

### Pre-Deployment

- [ ] All secrets stored in secure vault/platform secrets
- [ ] TLS certificates configured and valid
- [ ] WAF rules deployed and tested
- [ ] Security headers configured
- [ ] CORS properly restricted
- [ ] Rate limiting configured
- [ ] Network policies applied
- [ ] Audit logging enabled

### Post-Deployment

- [ ] Penetration test completed
- [ ] Security monitoring active
- [ ] Alert thresholds configured
- [ ] Incident response plan documented
- [ ] Regular security updates scheduled

### Periodic Review

- [ ] Review access logs monthly
- [ ] Rotate secrets quarterly
- [ ] Update dependencies weekly
- [ ] Review WAF rules quarterly
- [ ] Test backup restoration quarterly
