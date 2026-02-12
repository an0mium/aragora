# OpenClaw Gateway Security Audit

> Last audited: 2026-02-12
> Scope: `aragora/server/handlers/openclaw/`, `aragora/compat/openclaw/`
> Standard: OWASP Top 10 (2021)

## Summary

| Category | Status | Key Evidence |
|----------|--------|--------------|
| A01: Broken Access Control | PASS | RBAC decorators on all 15+ endpoints, user/tenant scoping |
| A02: Cryptographic Failures | PASS | Secrets encrypted via AES-256-GCM, never in responses |
| A03: Injection | PASS | Regex input validation, parameterized queries, shell metachar blocking |
| A04: Insecure Design | PASS | Skill malware scanner blocks DANGEROUS before execution |
| A05: Security Misconfiguration | PASS | No default credentials, rate limits on all endpoints |
| A06: Vulnerable Components | PASS | No hard external dependencies in gateway surface |
| A07: Auth Failures | PASS | Per-user rate limiting on credential rotation |
| A08: Data Integrity Failures | PASS | SHA-256 feedback hashes, HMAC receipt signatures |
| A09: Logging Failures | PASS | Full audit trail with actor accountability |
| A10: SSRF | PASS | Action type whitelist prevents arbitrary URL access |

**Overall: No OWASP Top 10 violations found.**

---

## A01: Broken Access Control

### RBAC Enforcement

All endpoints use `@require_permission` decorators at handler level:

| Endpoint | Method | Permission | File |
|----------|--------|-----------|------|
| Sessions list | GET | `gateway:sessions.read` | `orchestrator.py` |
| Session get | GET | `gateway:sessions.read` | `orchestrator.py` |
| Session create | POST | `gateway:sessions.create` | `orchestrator.py` |
| Session close | DELETE | `gateway:sessions.delete` | `orchestrator.py` |
| Actions list | GET | `gateway:actions.read` | `orchestrator.py` |
| Action execute | POST | `gateway:actions.execute` | `orchestrator.py` |
| Action cancel | POST | `gateway:actions.cancel` | `orchestrator.py` |
| Credentials list | GET | `gateway:credentials.read` | `credentials.py` |
| Credential store | POST | `gateway:credentials.create` | `credentials.py` |
| Credential rotate | POST | `gateway:credentials.rotate` | `credentials.py` |
| Credential delete | DELETE | `gateway:credentials.delete` | `credentials.py` |
| Policy rules get | GET | `gateway:policy.read` | `policies.py` |
| Policy rules set | POST | `gateway:policy.write` | `policies.py` |
| Approvals list | GET | `gateway:approvals.read` | `policies.py` |
| Approval submit | POST | `gateway:approvals.write` | `policies.py` |

### User & Tenant Isolation

Sessions are scoped to user/tenant for non-admin requests:

```python
# orchestrator.py — session access check
if not is_admin and session.user_id != user_id:
    return error_response("Access denied", 403)
```

This pattern is consistently applied on:
- Session close/end (ownership check)
- Action access (via session ownership)
- Credential rotation/deletion (ownership check)

### Impersonation Prevention

Approval endpoints ignore `approver_id` from the request body — only the authenticated user can approve:

```python
# policies.py — approver forced to authenticated user
approver_id = user_id  # From authenticated handler, NOT from request body
```

---

## A02: Cryptographic Failures

### Secret Storage

- Secrets stored in separate `_credential_secrets` dict, never in main model
- Encrypted via `aragora.security.encryption.encrypt_value()` (AES-256-GCM)
- Fallback to base64 encoding if encryption module unavailable
- Decryption only during rotation/retrieval operations

### API Response Safety

`Credential.to_dict()` explicitly excludes secret values — only metadata (id, name, type, timestamps) is serialized.

### Audit Log Safety

Audit entries log credential name and type but never the secret value.

---

## A03: Injection

### Input Validation (`validation.py`)

| Input | Pattern | Max Length |
|-------|---------|-----------|
| Credential name | `^[a-zA-Z][a-zA-Z0-9_-]{0,127}$` | 128 chars |
| Credential secret | No null bytes, type check | 8–8,192 bytes |
| Action type | `^[a-zA-Z][a-zA-Z0-9._-]{0,63}$` | 64 chars |
| Session config | JSON, type check | 8 KB, 50 keys, 5 levels deep |

### Shell Metacharacter Blocking

```python
SHELL_METACHARACTERS = re.compile(r'[;&|`$(){}[\]<>\\"\'\n\r\x00]')
```

Applied to action parameters before execution.

### SQL Injection Prevention

All database queries use parameterized `?` placeholders — no string concatenation.

---

## A04: Insecure Design — Skill Malware Scanner

The skill scanner (`aragora/compat/openclaw/skill_scanner.py`) blocks malicious skills before they can execute:

| Category | Patterns | Severity |
|----------|----------|----------|
| Shell commands | `curl \| bash`, `rm -rf /`, `netcat -e`, `/dev/tcp` | CRITICAL |
| Exfiltration | URL variable interpolation, base64-encoded `/etc/passwd` | CRITICAL |
| Prompt injection | "ignore previous instructions", system prompt override | HIGH |
| Credential access | `$API_KEY`, `$SECRET`, hardcoded AWS/GitHub tokens | CRITICAL |
| Obfuscation | Base64 commands, hex/octal encoding, command substitution | HIGH |

**Risk scoring**: CRITICAL=50pts, HIGH=30pts, MEDIUM=15pts, LOW=5pts. Score >= 70 = DANGEROUS (blocked).

`SkillPublisher` invokes the scanner before marketplace publication — DANGEROUS skills are rejected.

**Test coverage**: 31 tests in `tests/compat/openclaw/test_skill_scanner.py`.

---

## A05: Security Misconfiguration

- No default credentials or API keys
- Rate limits enforced on all endpoints (see A07)
- Error responses use `safe_error_message()` — no stack traces or internal details exposed
- Circuit breaker (threshold=5, cooldown=30s) prevents cascading failures

---

## A06: Vulnerable & Outdated Components

The OpenClaw gateway surface uses only standard library modules and internal Aragora packages. No third-party dependencies in the handler chain.

---

## A07: Identification & Authentication Failures

### Rate Limiting

| Endpoint | Limit | Type |
|----------|-------|------|
| List sessions/actions | 120 req/min | Standard |
| Get session/action | 120 req/min | Standard |
| Create session | 30 req/min | Standard |
| Close/cancel | 30 req/min | Standard |
| Execute action | 60 req/min | Auth rate limit |
| List credentials | 60 req/min | Standard |
| Store credential | 10 req/min | Auth rate limit |
| Rotate credential | 10 req/min | Auth rate limit + per-user sliding window |
| Delete credential | 20 req/min | Standard |
| Policy rules | 30–60 req/min | Standard |
| Approvals | 30–60 req/min | Standard |

### Credential Rotation Brute Force Protection

Per-user sliding window limiter:
- Max 10 rotations per hour
- Returns HTTP 429 with `Retry-After` header
- Thread-safe (Lock-protected)
- Rate limit events logged to audit trail

---

## A08: Software & Data Integrity Failures

- Decision receipts use HMAC-SHA256 signatures for tamper detection
- ERC-8004 feedback records include SHA-256 content hashes
- Skill scanner verifies skill integrity before marketplace publication

---

## A09: Security Logging & Monitoring Failures

### Audit Trail Coverage

Every state-changing operation produces an `AuditEntry`:

| Field | Description |
|-------|-------------|
| `id` | Unique audit log ID |
| `timestamp` | UTC timestamp |
| `action` | e.g., `session.create`, `credential.rotate` |
| `actor_id` | Authenticated user performing the action |
| `resource_type` | `session`, `credential`, `action`, `policy` |
| `resource_id` | ID of affected resource |
| `result` | `success`, `rate_limited`, `access_denied` |
| `details` | Additional context (no secrets) |

Logged actions include: session lifecycle, action execution/cancellation, credential CRUD, policy management, and approval decisions.

---

## A10: Server-Side Request Forgery (SSRF)

Action types are validated against a strict regex whitelist (`^[a-zA-Z][a-zA-Z0-9._-]{0,63}$`), preventing arbitrary URL construction. No user-controlled URLs are used in server-side requests within the gateway surface.

---

## Recommendations

1. **Low priority**: Add alerting on repeated 403 responses to detect access probing
2. **Low priority**: Consider Content-Security-Policy headers for any HTML-rendering endpoints
3. **Maintenance**: Re-run this audit after any new handler endpoint is added

## Files Audited

- `aragora/server/handlers/openclaw/` — gateway.py, orchestrator.py, credentials.py, policies.py, validation.py, store.py, models.py, _base.py
- `aragora/compat/openclaw/skill_scanner.py`, `skill_converter.py`
- `aragora/rbac/defaults/permissions/integrations.py`
- `tests/compat/openclaw/test_skill_scanner.py`
- `tests/server/handlers/openclaw/test_gateway.py`
