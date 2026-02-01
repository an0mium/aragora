# Gateway Architecture

## 1. Overview

The Aragora Gateway is the unified entry point for all communication between Aragora's
internal debate engine and external AI agent frameworks. It enforces enterprise security
policies, routes decisions between debate and direct execution, and manages credentials,
audit trails, and multi-tenant isolation.

**Design goals:**
- Zero-trust security with deny-by-default action filtering
- Transparent multi-tenant isolation with per-tenant quotas and encryption keys
- Cryptographic audit trails for SOC 2 Type II compliance
- Resilient proxying with circuit breakers, bulkheads, and retry policies
- Pluggable framework federation via capability-based discovery

## 2. Architecture Diagram

```
                          Client Request
                               |
                    +----------v-----------+
                    |    Auth Bridge        |  OIDC / SAML / API Key
                    |  (auth_bridge.py)     |  Token exchange (RFC 8693)
                    +----------+-----------+
                               |
                    +----------v-----------+
                    |   Tenant Router       |  Per-tenant endpoints & quotas
                    |  (tenant_router.py)   |  Load balancing, isolation
                    +----------+-----------+
                               |
                    +----------v-----------+
                    |  Audit Interceptor    |  SHA-256 hash chain logging
                    | (audit_interceptor.py)|  PII redaction, SIEM export
                    +----------+-----------+
                               |
              +----------------v----------------+
              |        Decision Router           |  Financial / risk / compliance
              |      (decision_router.py)        |  thresholds determine route
              +-------+----------------+--------+
                      |                |
           +----------v---+    +------v-----------+
           | Debate Engine |    | Enterprise Proxy  |
           |   (Arena)     |    |   (proxy.py)      |
           +--------------+    +------+------------+
                                      |
                    +-----------------+-----------------+
                    |                 |                 |
            +-------v------+  +------v-------+  +-----v--------+
            | OpenClaw      |  | Federation   |  | Other        |
            | Adapter       |  | Registry     |  | Frameworks   |
            | - Action      |  | (registry.py)|  |              |
            |   Filter      |  +--------------+  +--------------+
            | - Credential  |
            |   Vault       |
            +---------------+
```

**Three layers:**

| Layer | Components | Responsibility |
|-------|-----------|----------------|
| Enterprise Security | AuthBridge, TenantRouter, AuditInterceptor | Authentication, isolation, compliance |
| Gateway Core | DecisionRouter, EnterpriseProxy | Routing logic, resilient outbound calls |
| External Frameworks | OpenClaw adapter, Federation registry | Protocol translation, discovery |

## 3. Security Model

### Authentication

| Method | Module | Details |
|--------|--------|---------|
| OIDC Access/ID Tokens | `auth_bridge.py` | JWT validation, JWKS key rotation |
| SAML Assertions | `auth_bridge.py` | XML signature verification |
| API Keys | `auth_bridge.py` | Header or query parameter extraction |
| Token Exchange | `auth_bridge.py` | RFC 8693 subject/actor token swap |

AuthBridge maps Aragora RBAC permissions to external framework actions via
bidirectional `PermissionMapping` rules with wildcard and conditional support
(tenant, role, and permission conditions).

### Tenant Isolation

- Requests carry `X-Tenant-ID` / `X-Aragora-Tenant` headers injected by TenantRouter
- Cross-tenant access detected and blocked (`CrossTenantAccessError`)
- Per-tenant encryption keys via KMS in CredentialVault
- Isolation levels propagated via `X-Isolation-Level` header

### Encryption

| Scope | Algorithm | Module |
|-------|-----------|--------|
| Credentials at rest | AES-256-GCM | `credential_vault.py` |
| Audit record signing | HMAC-SHA256 | `audit_interceptor.py` |
| Audit hash chain | SHA-256 | `audit_interceptor.py` |

### Action Filtering (Deny-by-Default)

ActionFilter evaluates every outbound action through a layered pipeline:

1. **Critical denylist** (non-overridable): `rm -rf`, `format`, `dd`, `mkfs`, `shutdown`, `sudo`, `chmod 777`, credential export, port scans, raw sockets, code injection
2. **Dangerous pattern scan**: regex detection of fork bombs, pipe-to-shell, reverse shells, base64 eval
3. **Tenant allowlist**: per-tenant rule overrides
4. **Custom rules**: category-based allow/deny with risk levels
5. **Risk assessment**: LOW/MEDIUM/HIGH/CRITICAL threshold check

### RBAC for Credentials

Credential operations require explicit permissions: `credentials:create`, `credentials:read`,
`credentials:update`, `credentials:delete`, `credentials:rotate`, `credentials:list`,
`credentials:admin`. Rate limited to 30 requests/minute, 200/hour with 300-second lockout.

## 4. Request Lifecycle

```
1. Client sends request
2. AuthBridge validates token (OIDC/SAML/API key)
   - Creates AuthContext with user identity, roles, permissions
   - Establishes or resumes BridgedSession
3. TenantRouter resolves tenant
   - Checks quotas (per-minute/hour/day, concurrent, bandwidth)
   - Selects endpoint via load balancing strategy
   - Injects tenant context headers
4. AuditInterceptor records inbound request
   - Redacts PII fields (mask/hash/remove/truncate/tokenize)
   - Appends to SHA-256 hash chain
5. DecisionRouter evaluates routing criteria
   - Checks: explicit intent > custom rules > financial ($10K threshold)
     > risk level > compliance flags > stakeholder count > defaults
   - Routes to: DEBATE | EXECUTE | HYBRID | REJECT
6. If EXECUTE or HYBRID:
   a. EnterpriseProxy prepares outbound request
      - Sanitizes headers and body
      - Runs pre-request hooks
   b. Circuit breaker check (per-framework, threshold=5, cooldown=60s)
   c. Bulkhead admission (max 50 concurrent per framework)
   d. Retry loop (max 3 attempts, retryable: 429/500/502/503/504)
   e. ActionFilter validates action (deny-by-default pipeline)
   f. CredentialVault injects framework credentials
   g. Request dispatched to external framework
   h. Post-request hooks run
7. AuditInterceptor records response
   - Signs record with HMAC-SHA256
   - Exports to webhook/SIEM if configured
8. Response returned to client
```

## 5. Component Reference

| Module | Key Classes | Purpose |
|--------|------------|---------|
| `enterprise/proxy.py` | `EnterpriseProxy`, `FrameworkCircuitBreaker`, `FrameworkBulkhead`, `RequestSanitizer` | Resilient outbound HTTP with circuit breakers, bulkheads, retry, connection pooling |
| `enterprise/auth_bridge.py` | `AuthBridge`, `AuthContext`, `PermissionMapping`, `BridgedSession` | SSO pass-through, token exchange, permission mapping, session management |
| `enterprise/tenant_router.py` | `TenantRouter`, `QuotaTracker`, `EndpointHealthTracker` | Multi-tenant routing, quota enforcement, load balancing |
| `enterprise/audit_interceptor.py` | `AuditInterceptor`, `AuditRecord`, `PIIRedactionRule` | Tamper-evident logging, PII redaction, SOC 2 evidence, SIEM export |
| `openclaw/adapter.py` | `OpenClawAdapter`, `OpenClawMessage`, `OpenClawAction`, `OpenClawSession` | Protocol translation, channel formatting (WhatsApp/Telegram/Slack/Discord), sandbox execution |
| `openclaw/credential_vault.py` | `CredentialVault`, `StoredCredential`, `RotationPolicy` | AES-256-GCM encrypted storage, per-tenant keys, rotation policies, rate limiting |
| `openclaw/action_filter.py` | `ActionFilter`, `ActionRule`, `FilterDecision` | Deny-by-default filtering, critical denylist, dangerous pattern regex, risk assessment |
| `federation/registry.py` | `FederationRegistry`, `ExternalFramework`, `FrameworkCapability` | Framework discovery, health monitoring, API version negotiation, capability indexing |
| `decision_router.py` | `DecisionRouter`, `RoutingCriteria`, `RouteDecision`, `SimpleAnomalyDetector` | Debate vs. execute routing, financial/risk/compliance thresholds, anomaly detection |

### Resilience Settings

| Setting | Default | Description |
|---------|---------|-------------|
| Circuit breaker threshold | 5 failures | Opens circuit after N consecutive failures |
| Circuit breaker cooldown | 60 seconds | Time before half-open probe |
| Bulkhead concurrency | 50 | Max concurrent requests per framework |
| Retry attempts | 3 | Max retries per request |
| Retryable status codes | 429, 500, 502, 503, 504 | HTTP codes that trigger retry |

### Load Balancing Strategies

| Strategy | Description |
|----------|-------------|
| `ROUND_ROBIN` | Sequential rotation across endpoints |
| `WEIGHTED_RANDOM` | Probabilistic selection by weight |
| `LEAST_CONNECTIONS` | Prefer endpoint with fewest active requests |
| `PRIORITY` | Strict priority ordering |
| `LATENCY` | Prefer lowest observed latency |

## 6. Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ARAGORA_AUDIT_SIGNING_KEY` | Production | HMAC key for audit record signatures |
| `ARAGORA_AUDIT_INTERCEPTOR_KEY` | Fallback | Alternative name for audit signing key |
| `ARAGORA_CREDENTIAL_VAULT_KEY` | Yes | Master encryption key for credential vault |
| `ARAGORA_POSTGRES_DSN` | If using Postgres audit | PostgreSQL connection string |
| `DATABASE_URL` | Fallback | Alternative Postgres connection string |
| `ARAGORA_ENV` | No | Environment name (enables strict checks in `production`) |

### Quota Defaults (per tenant)

| Quota | Default |
|-------|---------|
| Requests per minute | 60 |
| Requests per hour | 1,000 |
| Requests per day | 10,000 |
| Concurrent requests | 10 |
| Bandwidth per minute | 10 MB |

### Credential Rotation Policies

| Policy | Max Age | Description |
|--------|---------|-------------|
| Strict | 30 days | High-security credentials |
| Standard | 90 days | Default rotation cycle |
| Relaxed | 365 days | Low-risk credentials |

### Decision Routing Defaults

| Criterion | Threshold | Route |
|-----------|-----------|-------|
| Financial impact | >= $10,000 | DEBATE |
| Risk level | HIGH or CRITICAL | DEBATE |
| Compliance flags | pii, financial, hipaa, gdpr, sox | DEBATE |
| Stakeholder count | >= 3 | DEBATE |
| Explicit intent | User-specified | As requested |

## 7. Threat Model

| Threat | Boundary | Mitigation |
|--------|----------|------------|
| Token replay | Auth layer | Short-lived tokens, session binding, token exchange |
| Cross-tenant data leak | Tenant router | Header injection, isolation level enforcement, `CrossTenantAccessError` |
| Audit log tampering | Audit interceptor | SHA-256 hash chain, HMAC-SHA256 signatures, append-only storage |
| PII exposure in logs | Audit interceptor | Configurable redaction (mask/hash/remove/truncate/tokenize) for 8+ default patterns |
| Credential theft | Credential vault | AES-256-GCM encryption, per-tenant KMS keys, RBAC, rate limiting |
| Malicious actions | Action filter | Non-overridable critical denylist, regex pattern scanning, deny-by-default |
| Framework compromise | Proxy layer | Circuit breakers isolate failures, bulkheads limit blast radius |
| Denial of service | Tenant router | Per-tenant quotas (rate, concurrency, bandwidth), load balancing |
| Privilege escalation | Auth bridge | Bidirectional permission mapping with conditions, no implicit grants |
| Supply chain (rogue framework) | Federation registry | Health checks, capability verification, API version negotiation |

### SOC 2 Control Mapping

| Control | Implementation |
|---------|---------------|
| CC6.1 (Access Control) | AuthBridge + RBAC permission mapping |
| CC6.6 (Audit Logging) | AuditInterceptor hash chain with SIEM export |
| CC6.7 (Data Protection) | AES-256-GCM credentials, PII redaction |
| CC7.2 (Monitoring) | Prometheus metrics, anomaly detection, health probes |

## 8. Integration Guide

### Adding a New External Framework

1. **Register the framework** in FederationRegistry:
   ```python
   from aragora.gateway.federation.registry import FederationRegistry, ExternalFramework

   registry = FederationRegistry()
   framework = ExternalFramework(
       framework_id="my-framework",
       name="My Framework",
       base_url="https://api.myframework.io",
       api_version="1.0.0",
       capabilities=[FrameworkCapability(name="text_generation", version="1.0")],
   )
   result = await registry.register(framework)
   ```

2. **Configure proxy settings** for the framework:
   ```python
   from aragora.gateway.enterprise.proxy import ExternalFrameworkConfig

   config = ExternalFrameworkConfig(
       framework_id="my-framework",
       base_url="https://api.myframework.io",
       circuit_breaker=CircuitBreakerSettings(failure_threshold=5, cooldown_seconds=60),
       retry=RetrySettings(max_retries=3),
       bulkhead=BulkheadSettings(max_concurrent=50),
   )
   proxy.register_framework(config)
   ```

3. **Store credentials** in the vault:
   ```python
   from aragora.gateway.openclaw.credential_vault import CredentialVault

   vault = CredentialVault()
   await vault.store(
       credential_id="my-framework-key",
       credential_type=CredentialType.API_KEY,
       value="sk-...",
       framework=Framework.CUSTOM,
       tenant_id="tenant-123",
   )
   ```

4. **Define action filter rules** (optional):
   ```python
   from aragora.gateway.openclaw.action_filter import ActionFilter, ActionRule

   filter = ActionFilter()
   filter.add_rule(ActionRule(
       action_pattern="my_framework.*",
       category=ActionCategory.NETWORK,
       risk_level=RiskLevel.MEDIUM,
       decision=FilterDecision.ALLOW,
   ))
   ```

5. **Map permissions** in AuthBridge:
   ```python
   bridge.add_mapping(PermissionMapping(
       aragora_permission="frameworks:execute",
       external_action="my-framework:invoke",
       bidirectional=True,
   ))
   ```

### Framework Selection Strategies

The FederationRegistry supports three selection strategies when multiple frameworks
match a capability query:

| Strategy | Use Case |
|----------|----------|
| `healthiest` | Prefer the framework with best health status |
| `newest` | Prefer the most recently registered framework |
| `random` | Random selection for load distribution |

### Supported OpenClaw Channels

| Channel | Formatter | Notes |
|---------|-----------|-------|
| WhatsApp | `WhatsAppFormatter` | Media attachments, templates |
| Telegram | `TelegramFormatter` | Markdown, inline keyboards |
| Slack | `SlackFormatter` | Block Kit, threads |
| Discord | `DiscordFormatter` | Embeds, reactions |
| SMS, Email, Web, Voice, Teams, Matrix | Default | Basic text formatting |
