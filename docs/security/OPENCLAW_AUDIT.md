# OpenClaw Gateway Security Audit

**Date:** 2026-02-12
**Auditor:** Automated OWASP Top 10 audit (Claude Opus 4.6)
**Scope:** `aragora/compat/openclaw/`, `aragora/server/handlers/openclaw/`, `aragora/gateway/`
**Standard:** OWASP Top 10 (2021)
**Prior audit:** 2026-02-12 (initial)

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 1 |
| HIGH     | 3 |
| MEDIUM   | 4 |
| LOW      | 3 |
| INFO     | 2 |
| **Total** | **13** |

**Overall assessment:** The OpenClaw gateway has strong security fundamentals -- RBAC
decorators on all handler endpoints, parameterized SQL queries, input validation with
regex whitelisting, and comprehensive audit logging. However, the standalone server
(`standalone.py`) bypasses most of these controls, and the persistent store has a
cryptographic fallback that silently degrades to base64 encoding when the encryption
library is unavailable. These represent the two most significant findings.

## Findings

### [CRITICAL] F01: Standalone Server Has No Authentication

- **Category:** OWASP A01 (Broken Access Control), A07 (Authentication Failures)
- **Location:** `aragora/compat/openclaw/standalone.py:326-345`
- **Description:** The `_MinimalHandlerContext` extracts `user_id` from the
  `X-User-ID` header and defaults to `"anonymous"`. There is no authentication
  middleware -- any client can set `X-User-ID` to any value and impersonate any
  user. The standalone server has no API key check, no token verification, and no
  authentication middleware in its request pipeline.
- **Risk:** An attacker with network access to the standalone gateway can:
  - Create sessions as any user
  - Access, modify, or delete other users' sessions and credentials
  - Execute actions under any identity
  - Approve or deny actions while impersonating an admin
- **Recommendation:**
  1. Add API key authentication (e.g., check `Authorization: Bearer <key>` header)
     before routing to the handler, similar to `LocalGateway._auth_middleware`.
  2. Never trust `X-User-ID` from unauthenticated requests -- this header should
     only be populated by a trusted reverse proxy after authentication.
  3. At minimum, require `ARAGORA_OPENCLAW_API_KEY` environment variable and reject
     requests without a matching key.
- **Status:** Open

### [HIGH] F02: Credential Encryption Silently Falls Back to Base64

- **Category:** OWASP A02 (Cryptographic Failures)
- **Location:** `aragora/server/handlers/openclaw/store.py:458-479`
- **Description:** `OpenClawPersistentStore._encrypt_secret()` attempts to import
  `aragora.security.encryption.encrypt_value`, but on `ImportError` silently falls
  back to base64 encoding. Base64 is NOT encryption -- it is trivially reversible.
  Credential secrets stored under this fallback have zero confidentiality protection.
  There is no log warning, no runtime check, and no configuration flag to enforce
  real encryption.
- **Risk:** If the `cryptography` library is not installed (e.g., minimal Docker
  image, CI environment, or dependency resolution failure), all credential secrets
  are stored in plaintext-equivalent format. An attacker with database access can
  decode all secrets with a single `base64 -d` command.
- **Recommendation:**
  1. Log a CRITICAL-level warning when falling back to base64.
  2. Add a startup check: if `ARAGORA_ENV=production` and encryption is unavailable,
     refuse to start.
  3. Consider making the `cryptography` library a hard dependency of the gateway
     module.
- **Status:** Mitigated (logger.critical + RuntimeError in production)

### [HIGH] F03: Standalone Server Unbounded Request Body Size

- **Category:** OWASP A04 (Insecure Design)
- **Location:** `aragora/compat/openclaw/standalone.py:147-153`
- **Description:** The standalone server reads the request body based on the
  `Content-Length` header with no upper bound validation:
  ```python
  content_length = int(headers.get("content-length", "0"))
  if content_length > 0:
      body_bytes = await asyncio.wait_for(reader.readexactly(content_length), timeout=30.0)
  ```
  A malicious client can send `Content-Length: 2147483647` and the server will
  attempt to allocate gigabytes of memory, causing an out-of-memory condition.
- **Risk:** Denial of service. A single request can crash the standalone gateway
  process.
- **Recommendation:** Add a maximum body size check before reading:
  ```python
  MAX_BODY_SIZE = 1_048_576  # 1 MB
  if content_length > MAX_BODY_SIZE:
      await self._send_response(writer, 413, {"error": "Request too large"})
      return
  ```
- **Status:** Fixed (MAX_BODY_SIZE=1MB, HTTP 413)

### [HIGH] F04: Standalone Server Unbounded Header Count

- **Category:** OWASP A04 (Insecure Design)
- **Location:** `aragora/compat/openclaw/standalone.py:136-143`
- **Description:** The header parsing loop (`while True`) reads headers until an
  empty line is received, with no limit on the number of headers. Each header line
  has a 10-second timeout, but there is no maximum header count. An attacker can
  send thousands of headers to consume memory and tie up the connection for extended
  periods (`10s * N_headers`).
- **Risk:** Resource exhaustion / slowloris-style denial of service. With 1000
  headers, one connection can occupy the server for approximately 10,000 seconds.
- **Recommendation:** Replace `while True` with `for _hdr_idx in range(MAX_HEADER_COUNT):`
  and add a `for/else` clause that returns HTTP 431 when exceeded.
- **Status:** Fixed (MAX_HEADER_COUNT=100, HTTP 431)

### [MEDIUM] F05: CORS Wildcard in Standalone Server

- **Category:** OWASP A05 (Security Misconfiguration)
- **Location:** `aragora/compat/openclaw/standalone.py:58, 406`
- **Description:** `self.cors_origins` defaults to `["*"]` and the CLI `--cors`
  flag defaults to `"*"`. This means the standalone server allows cross-origin
  requests from ANY domain by default, which combined with the lack of
  authentication (F01), allows any website to make authenticated requests to the
  gateway if a user visits a malicious page.
- **Risk:** Cross-site request forgery from any origin. Combined with F01, this
  enables remote exploitation via a victim's browser.
- **Recommendation:** Change the default to an empty list or `["http://localhost"]`
  and require explicit configuration for production CORS origins.
- **Status:** Fixed (CLI default changed to `http://localhost:3000`)

### [MEDIUM] F06: OpenClaw Routes Missing from RBAC Middleware

- **Category:** OWASP A01 (Broken Access Control)
- **Location:** `aragora/rbac/middleware.py` (DEFAULT_ROUTE_PERMISSIONS)
- **Description:** The `DEFAULT_ROUTE_PERMISSIONS` list in the RBAC middleware
  does not include any routes matching `/api/gateway/openclaw/*`. While the
  handler-level `@require_permission` decorators provide per-endpoint protection,
  the absence of gateway routes in the middleware means the middleware's
  `check_request()` function will fall through to the default policy (which
  allows authenticated access without checking specific permissions). This
  creates a defense-in-depth gap -- if a developer adds a new OpenClaw endpoint
  without a `@require_permission` decorator, it would be unprotected at both
  layers.
- **Risk:** Defense-in-depth violation. New endpoints added without decorators
  would be accessible to any authenticated user regardless of role.
- **Recommendation:** Add `RoutePermission` entries for `/api/gateway/openclaw/`
  routes to the `DEFAULT_ROUTE_PERMISSIONS` list, mirroring the permissions used
  by the handler decorators.
- **Status:** Fixed (33 RoutePermission entries added for all OpenClaw routes)

### [MEDIUM] F07: NavigateAction Accepts Arbitrary URLs Without SSRF Validation

- **Category:** OWASP A10 (SSRF)
- **Location:** `aragora/compat/openclaw/computer_use_bridge.py:110-115`
- **Description:** The `ComputerUseBridge.from_openclaw()` method creates
  `NavigateAction` objects with a URL taken directly from user-supplied params:
  ```python
  return NavigateAction(
      url=params.get("url", ""),
      ...
  )
  ```
  The URL is not validated against the SSRF protection module
  (`aragora.security.ssrf_protection`). If this NavigateAction is later executed
  to perform an HTTP request (e.g., browser automation), it could target internal
  services, cloud metadata endpoints, or localhost.
- **Risk:** Server-side request forgery if the NavigateAction URL is used to make
  actual HTTP requests to the target. The severity depends on whether the
  downstream execution actually fetches the URL (currently the sandbox environment
  may limit this, but the bridge itself performs no validation).
- **Recommendation:** Add URL validation before creating the NavigateAction:
  ```python
  from aragora.security.ssrf_protection import validate_url
  result = validate_url(params.get("url", ""))
  if not result.is_safe:
      raise ValueError(f"Unsafe URL: {result.error}")
  ```
- **Status:** Open

### [MEDIUM] F08: Health Endpoint Leaks Exception Strings

- **Category:** OWASP A05 (Security Misconfiguration), A09 (Logging Failures)
- **Location:** `aragora/server/handlers/openclaw/policies.py` (health handler)
- **Description:** The health endpoint catches all exceptions and includes
  `str(e)` directly in the response body. Exception messages can contain internal
  details such as database connection strings, file paths, SQL errors, or
  configuration details.
- **Risk:** Information disclosure. Internal implementation details leaked to
  unauthenticated callers (health endpoints are typically public).
- **Recommendation:** Replace `str(e)` with a generic message like
  `"Internal health check failed"` and log the actual exception server-side.
- **Status:** Fixed (replaced `str(e)` with `safe_error_message()`)

### [LOW] F09: Query Parameter Values Not URL-Decoded

- **Category:** OWASP A03 (Injection)
- **Location:** `aragora/compat/openclaw/standalone.py:128-132`
- **Description:** Query parameters are parsed by splitting on `&` and `=`
  without URL-decoding. Values containing `%20`, `%3D`, etc., are passed through
  as-is. While this does not create a direct injection vulnerability, it can cause
  unexpected behavior when parameter values contain encoded characters.
- **Risk:** Low. May cause parameter misinterpretation but no direct
  exploitation path.
- **Recommendation:** Use `urllib.parse.unquote()` on both key and value after
  splitting.
- **Status:** Open

### [LOW] F10: In-Memory Store Has No Session Expiration

- **Category:** OWASP A07 (Authentication Failures)
- **Location:** `aragora/server/handlers/openclaw/store.py:29-109`
- **Description:** The `OpenClawGatewayStore` (in-memory) and
  `OpenClawPersistentStore` (SQLite) have no session expiration mechanism.
  Sessions created with status `ACTIVE` remain active indefinitely unless
  explicitly closed. There is no TTL, no idle timeout, and no periodic cleanup.
- **Risk:** Abandoned sessions accumulate, consuming memory. Long-lived sessions
  increase the window for session hijacking if session IDs are leaked.
- **Recommendation:** Add a configurable session idle timeout (e.g., 24 hours)
  and a periodic cleanup task that closes sessions past their TTL.
- **Status:** Open

### [LOW] F11: Audit Log Has Fixed 10,000-Entry Cap

- **Category:** OWASP A09 (Logging Failures)
- **Location:** `aragora/server/handlers/openclaw/store.py:266-268`
- **Description:** The in-memory audit log caps at 10,000 entries. Older entries
  are silently dropped when the cap is reached. Under heavy load (e.g., DDoS or
  audit-intensive operations), security-relevant entries could be lost.
- **Risk:** Loss of security audit trail during high-activity periods.
- **Recommendation:** For the in-memory store, log a warning when the cap is hit.
  For production, use the persistent store which has no cap. Consider adding a
  separate rate-limited alert when audit entries are being dropped.
- **Status:** Accepted Risk (persistent store used in production)

### [INFO] F12: YAML Parser Uses `yaml.safe_load` Correctly

- **Category:** OWASP A08 (Data Integrity)
- **Location:** `aragora/compat/openclaw/skill_parser.py`
- **Description:** The SKILL.md parser uses `yaml.safe_load()` (not the unsafe
  `yaml.load()`) to parse frontmatter. This correctly prevents YAML
  deserialization attacks.
- **Risk:** None.
- **Status:** Positive finding

### [INFO] F13: SQL Queries Use Parameterized Statements

- **Category:** OWASP A03 (Injection)
- **Location:** `aragora/server/handlers/openclaw/store.py` (all SQL methods)
- **Description:** All SQLite queries in `OpenClawPersistentStore` use `?`
  parameterized placeholders. No string concatenation or f-strings are used in
  SQL construction. The `WHERE` clause building uses only pre-validated column
  names, not user input.
- **Risk:** None.
- **Status:** Positive finding

## Positive Controls

The following security controls are already in place and function correctly:

1. **RBAC decorators** (`@require_permission`) on all 15+ handler endpoints in
   `orchestrator.py`, `credentials.py`, and `policies.py`.

2. **Ownership verification** on session access, action access, credential
   rotation, and credential deletion. Non-admin users can only access their own
   resources.

3. **Impersonation prevention** -- approval/deny endpoints force `approver_id`
   to the authenticated user, ignoring any value in the request body.

4. **Input validation** with strict regex whitelisting for credential names,
   action types, and shell metacharacter blocking on action parameters.

5. **Rate limiting** on all endpoints (30-120 req/min) plus per-user credential
   rotation limits (10/hour) with `Retry-After` headers.

6. **Circuit breaker** (threshold=5, cooldown=30s) to prevent cascading failures.

7. **Skill malware scanner** that blocks dangerous patterns (shell injection,
   data exfiltration, prompt injection, credential access, obfuscation) before
   skill conversion and marketplace publication.

8. **Credential data separation** -- `Credential.to_dict()` never includes
   secret values. Secrets are stored separately and never appear in API responses.

9. **Safe error messages** -- `safe_error_message()` sanitizes exception details
   before returning them to clients (used in all handler catch blocks except
   the health endpoint noted in F08).

10. **Comprehensive audit logging** with actor, resource, action, result, and
    timestamp for all state-changing operations.

11. **Sandbox isolation** -- `openclaw_sandbox.py` provides workspace scoping,
    resource limits, and path blocking for action execution.

12. **No dangerous function calls** -- no `eval()`, `exec()`, `pickle.loads()`,
    or unsafe `yaml.load()` in any gateway code. The `eval` string in
    `skill_scanner.py` is a detection pattern, not an invocation.

## Recommended Code Fixes

The following fixes were developed during this audit. They should be applied manually
since automated application was not possible in this session.

### Fix for F02 (store.py encryption fallback)

In `aragora/server/handlers/openclaw/store.py`, add `logger = logging.getLogger(__name__)`
at module level, then replace the `_encrypt_secret` and `_decrypt_secret` methods:

```python
def _encrypt_secret(self, value: str) -> str:
    try:
        from aragora.security.encryption import encrypt_value
        return encrypt_value(value)
    except ImportError:
        import base64
        import os
        logger.critical(
            "cryptography library unavailable - credential secrets stored "
            "with base64 only (NOT encrypted)."
        )
        if os.environ.get("ARAGORA_ENV", "").lower() in ("production", "prod"):
            raise RuntimeError(
                "Encryption library unavailable in production."
            )
        return base64.b64encode(value.encode()).decode()
```

### Fix for F03 and F04 (standalone.py request bounds)

In `aragora/compat/openclaw/standalone.py`, add constants and modify the request handler:

```python
MAX_BODY_SIZE = 1_048_576  # 1 MB
MAX_HEADER_COUNT = 100

# Replace `while True:` header loop with:
for _hdr_idx in range(MAX_HEADER_COUNT):
    ...
else:
    await self._send_response(writer, 431, {"error": "Too many headers"})
    return

# Before reading body:
if content_length > MAX_BODY_SIZE:
    await self._send_response(writer, 413, {"error": "Request body too large"})
    return
```

## Files Audited

| Path | Lines | Purpose |
|------|-------|---------|
| `aragora/compat/openclaw/standalone.py` | 419 | Standalone gateway server |
| `aragora/compat/openclaw/skill_scanner.py` | 341 | Skill malware scanning |
| `aragora/compat/openclaw/skill_parser.py` | 233 | SKILL.md parsing |
| `aragora/compat/openclaw/skill_converter.py` | 280 | Skill conversion |
| `aragora/compat/openclaw/computer_use_bridge.py` | 220 | Browser action bridge |
| `aragora/compat/openclaw/capability_mapper.py` | 132 | Capability mapping |
| `aragora/compat/openclaw/migration_utils.py` | 201 | Migration utilities |
| `aragora/server/handlers/openclaw/gateway.py` | 325 | Main handler class |
| `aragora/server/handlers/openclaw/orchestrator.py` | 433 | Session/action handlers |
| `aragora/server/handlers/openclaw/credentials.py` | 423 | Credential handlers |
| `aragora/server/handlers/openclaw/policies.py` | 391 | Policy/admin handlers |
| `aragora/server/handlers/openclaw/validation.py` | 320 | Input validation |
| `aragora/server/handlers/openclaw/store.py` | 1216 | Data store (memory + SQLite) |
| `aragora/server/handlers/openclaw/models.py` | 192 | Data models |
| `aragora/server/handlers/openclaw/_base.py` | 76 | Mixin base class |
| `aragora/gateway/server.py` | 943 | Local gateway server |
| `aragora/gateway/router.py` | 142 | Message routing |
| `aragora/security/ssrf_protection.py` | 595 | SSRF protection |
| `aragora/security/encryption.py` | ~80 | Encryption service |
| `aragora/rbac/middleware.py` | 904 | RBAC middleware |
