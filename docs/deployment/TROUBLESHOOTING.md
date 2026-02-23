# Troubleshooting Guide

Symptom-based troubleshooting for Aragora deployments. Follow each decision tree
from the symptom to the resolution.

---

## Table of Contents

1. [API Connectivity Issues](#1-api-connectivity-issues)
2. [Debate Failures / No Results](#2-debate-failures--no-results)
3. [High Latency](#3-high-latency)
4. [Memory Growth](#4-memory-growth)
5. [WebSocket Disconnects](#5-websocket-disconnects)
6. [Database Errors](#6-database-errors)
7. [Authentication Failures](#7-authentication-failures)
8. [General Diagnostics](#8-general-diagnostics)

---

## 1. API Connectivity Issues

**Symptom:** Clients receive connection refused, timeout, or CORS errors.

```
Can't connect to API
|
+-- Is the server running?
|   |
|   +-- NO --> Start it:
|   |          $ aragora serve --api-port 8080 --ws-port 8765
|   |
|   +-- YES --> Is the health endpoint reachable?
|               |
|               +-- Check liveness:
|               |   $ curl -s http://localhost:8080/healthz
|               |   Expected: {"status": "ok"}
|               |
|               +-- Check readiness:
|               |   $ curl -s http://localhost:8080/readyz
|               |   Expected: {"status": "ready"}
|               |   If {"status": "not_ready"} --> Server is still starting up. Wait.
|               |
|               +-- Neither responds?
|                   |
|                   +-- Check the port is bound:
|                   |   $ lsof -i :8080
|                   |   $ ss -tlnp | grep 8080   # Linux
|                   |
|                   +-- Check firewall / security groups:
|                       $ sudo iptables -L -n | grep 8080   # Linux
|                       Ensure cloud security groups allow inbound on 8080/8765.
```

### CORS errors in the browser

```
CORS error
|
+-- Set ARAGORA_ALLOWED_ORIGINS:
|   $ export ARAGORA_ALLOWED_ORIGINS="https://app.example.com"
|   Multiple origins: comma-separated.
|
+-- For development:
    $ export ARAGORA_ALLOWED_ORIGINS="*"
```

---

## 2. Debate Failures / No Results

**Symptom:** Debates start but produce no output, hang, or return errors.

```
No debate results
|
+-- Are API keys configured?
|   $ env | grep -E '(ANTHROPIC|OPENAI|OPENROUTER)_API_KEY'
|   At least one must be set.
|   |
|   +-- Missing --> Set a key:
|       $ export ANTHROPIC_API_KEY="sk-ant-..."
|       or
|       $ export OPENAI_API_KEY="sk-..."
|
+-- Is the provider reachable?
|   $ curl -s https://api.anthropic.com/v1/messages \
|       -H "x-api-key: $ANTHROPIC_API_KEY" \
|       -H "content-type: application/json" \
|       -d '{"model":"claude-sonnet-4-5-20250929","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}'
|   |
|   +-- 401 --> Invalid API key.
|   +-- 429 --> Rate limited. Wait, or set OPENROUTER_API_KEY for fallback.
|   +-- Connection error --> DNS / firewall issue to provider.
|
+-- Check circuit breaker state:
|   $ curl -s http://localhost:8080/api/v1/health | python3 -m json.tool
|   Look for "circuit_breaker" fields showing "open" state.
|   |
|   +-- Circuit breaker open --> The provider failed repeatedly.
|       Wait for the half-open recovery window (default: 60s),
|       or restart the server to reset breaker state.
|
+-- Check server logs for errors:
    $ journalctl -u aragora --since "10 min ago" --no-pager | grep -i error
    or
    $ tail -200 /var/log/aragora/server.log | grep -iE '(error|exception|traceback)'
```

---

## 3. High Latency

**Symptom:** Debates or API calls take much longer than expected.

```
High latency
|
+-- Which phase is slow?
|   |
|   +-- API response time (non-debate endpoints):
|   |   |
|   |   +-- Check database query time:
|   |   |   $ curl -w "\n%{time_total}s\n" -s http://localhost:8080/healthz
|   |   |   If > 1s, database may be the bottleneck.
|   |   |
|   |   +-- Check connection pool saturation:
|   |       $ curl -s http://localhost:8080/api/v1/metrics | grep pool
|   |       If pool is exhausted, increase pool size:
|   |       $ export ARAGORA_POOL_MAX_SIZE=50
|   |
|   +-- Debate round time:
|       |
|       +-- Check concurrency settings in ArenaConfig:
|       |   MAX_CONCURRENT_PROPOSALS (default varies)
|       |   MAX_CONCURRENT_CRITIQUES
|       |   MAX_CONCURRENT_REVISIONS
|       |   Increase for more parallelism if provider rate limits allow.
|       |
|       +-- Check rate limits:
|       |   $ curl -s http://localhost:8080/api/v1/rate-limits
|       |   Reduce per-agent rate if hitting provider 429s.
|       |
|       +-- Network latency to provider:
|           $ curl -o /dev/null -s -w "DNS: %{time_namelookup}s\nConnect: %{time_connect}s\nTTFB: %{time_starttransfer}s\nTotal: %{time_total}s\n" https://api.anthropic.com/
```

### Database performance tuning

```bash
# Check WAL mode is enabled (should be default)
$ sqlite3 /path/to/aragora.db "PRAGMA journal_mode;"
# Expected: wal

# Run ANALYZE to update query planner statistics
$ sqlite3 /path/to/aragora.db "ANALYZE;"

# Check for missing indexes
$ sqlite3 /path/to/aragora.db ".indices"
```

---

## 4. Memory Growth

**Symptom:** Server process memory usage increases over time.

```
Memory growth
|
+-- Check process memory:
|   $ ps aux | grep aragora | grep -v grep
|   or
|   $ python3 -c "import resource; print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)"
|
+-- Check memory tier TTLs:
|   Tier      | Default TTL | Purpose
|   ----------|-------------|--------
|   Fast      | 1 minute    | Immediate context
|   Medium    | 1 hour      | Session memory
|   Slow      | 1 day       | Cross-session learning
|   Glacial   | 1 week      | Long-term patterns
|   |
|   +-- If memory grows between debates, TTL eviction may not be running.
|       Check that ContinuumMemory cleanup is scheduled.
|
+-- Check cache sizes:
|   |
|   +-- KnowledgeMound cache:
|   |   $ curl -s http://localhost:8080/api/v1/knowledge/stats
|   |
|   +-- RBAC permission cache:
|   |   Cached permissions have a default TTL. If stale entries accumulate,
|   |   restart the server or call the cache-clear endpoint.
|   |
|   +-- Connection pool:
|       $ curl -s http://localhost:8080/api/v1/metrics | grep -i pool
|       Idle connections consume memory. Reduce ARAGORA_POOL_MAX_SIZE if needed.
|
+-- Python-level profiling:
    $ python3 -c "
    import tracemalloc
    tracemalloc.start()
    # ... import aragora modules ...
    snapshot = tracemalloc.take_snapshot()
    for stat in snapshot.statistics('lineno')[:10]:
        print(stat)
    "
```

---

## 5. WebSocket Disconnects

**Symptom:** WebSocket connections drop during debates or streaming.

```
WebSocket disconnects
|
+-- Is the WebSocket port accessible?
|   $ websocat ws://localhost:8765/
|   or
|   $ curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" \
|       -H "Sec-WebSocket-Version: 13" -H "Sec-WebSocket-Key: dGVzdA==" \
|       http://localhost:8765/
|   |
|   +-- Connection refused --> WebSocket server not running or wrong port.
|       $ aragora serve --ws-port 8765
|
+-- Behind a reverse proxy (nginx / ALB)?
|   |
|   +-- Check proxy timeout (common cause):
|   |   nginx: proxy_read_timeout 300s;
|   |   ALB:   idle timeout >= 300 seconds
|   |   Debates can take minutes; short timeouts will drop the connection.
|   |
|   +-- Check WebSocket upgrade is allowed:
|       nginx:
|         proxy_http_version 1.1;
|         proxy_set_header Upgrade $http_upgrade;
|         proxy_set_header Connection "upgrade";
|
+-- Heartbeat / ping-pong:
|   Default server heartbeat interval is configurable.
|   If clients don't respond to pings in time, the server disconnects.
|   Check client-side WebSocket keep-alive handling.
|
+-- Too many concurrent connections?
    $ ss -s | grep ESTAB
    Check ulimits:
    $ ulimit -n    # file descriptors
    Increase if needed:
    $ ulimit -n 65536
```

---

## 6. Database Errors

**Symptom:** SQLite or PostgreSQL errors in logs.

### SQLite issues

```
SQLite error
|
+-- "database is locked"
|   |
|   +-- WAL mode should prevent most lock contention.
|   |   Verify: $ sqlite3 db.sqlite3 "PRAGMA journal_mode;"
|   |   If not "wal", the DB was created without WAL.
|   |   Fix: $ sqlite3 db.sqlite3 "PRAGMA journal_mode=WAL;"
|   |
|   +-- Multiple processes writing to the same DB?
|   |   SQLite supports one writer at a time.
|   |   For multi-process: switch to PostgreSQL,
|   |   or ensure only one server process writes.
|   |
|   +-- Increase busy timeout:
|       Default is 30 seconds. Set DB_TIMEOUT env or in config.
|
+-- "disk I/O error"
|   |
|   +-- Check disk space: $ df -h
|   +-- Check filesystem health
|   +-- NFS / network filesystem? SQLite does not work reliably on NFS.
|       Use a local filesystem or switch to PostgreSQL.
|
+-- Schema version mismatch:
    |
    +-- Check: $ sqlite3 db.sqlite3 "SELECT * FROM _schema_versions;"
    +-- If version is higher than the code expects, you may be running
        an older version of the code against a newer database.
        Upgrade your code, or restore from backup.
```

### PostgreSQL issues

```
PostgreSQL error
|
+-- "could not connect to server"
|   $ pg_isready -h localhost -p 5432
|   |
|   +-- Not running --> Start PostgreSQL
|   +-- Wrong credentials --> Check DATABASE_URL env var
|
+-- Advisory lock timeout (migration runner):
|   The migration runner uses pg_try_advisory_lock(2089872453).
|   If a previous migration crashed, the lock may be held by a dead connection.
|   |
|   +-- Check: SELECT * FROM pg_locks WHERE locktype = 'advisory';
|   +-- Kill stale session: SELECT pg_terminate_backend(<pid>);
|
+-- Migration state inconsistency:
    The migration runner re-checks pending/applied state AFTER
    acquiring the advisory lock to handle crashed-pod edge cases.
    If a migration is partially applied:
    |
    +-- Check _schema_versions table for the current state
    +-- Manually fix the version if needed
    +-- Re-run the migration runner
```

---

## 7. Authentication Failures

**Symptom:** 401 Unauthorized or 403 Forbidden responses.

```
Auth failure
|
+-- API token auth:
|   $ curl -s -H "Authorization: Bearer $ARAGORA_API_TOKEN" \
|       http://localhost:8080/api/v1/health
|   |
|   +-- 401 --> Token missing or invalid.
|   |   Check: $ echo $ARAGORA_API_TOKEN
|   |   Ensure the server was started with the same token.
|   |
|   +-- 403 --> Token valid but insufficient permissions.
|       Check RBAC role assignments for the user/token.
|
+-- OIDC / SAML SSO:
|   |
|   +-- Token expired?
|   |   Decode the JWT: $ echo "$TOKEN" | cut -d. -f2 | base64 -d 2>/dev/null | python3 -m json.tool
|   |   Check "exp" claim against current time.
|   |
|   +-- Wrong issuer / audience?
|   |   Compare token claims against OIDC config:
|   |   ARAGORA_OIDC_ISSUER
|   |   ARAGORA_OIDC_CLIENT_ID
|   |   ARAGORA_OIDC_AUDIENCE
|   |
|   +-- JWKS fetch failing?
|       $ curl -s $ARAGORA_OIDC_ISSUER/.well-known/openid-configuration
|       Verify jwks_uri is reachable from the server.
|
+-- RBAC permission denied (403):
    |
    +-- Check user's roles:
    |   $ curl -s -H "Authorization: Bearer $TOKEN" \
    |       http://localhost:8080/api/v1/rbac/me
    |
    +-- Check required permission for the endpoint in the handler's
        @require_permission decorator.
    |
    +-- Permission cache stale?
        Restart the server or wait for cache TTL expiry.
```

---

## 8. General Diagnostics

### Quick health check

```bash
# Full status check
curl -s http://localhost:8080/healthz | python3 -m json.tool
curl -s http://localhost:8080/readyz | python3 -m json.tool

# Thread health (background workers, health checks)
curl -s http://localhost:8080/health/threads | python3 -m json.tool

# Metrics (Prometheus format)
curl -s http://localhost:8080/metrics
```

### Server startup issues

```bash
# Verify Python dependencies
pip check

# Verify aragora is importable
python3 -c "import aragora; print(aragora.__version__)"

# Check for port conflicts
lsof -i :8080
lsof -i :8765

# Start with verbose logging
ARAGORA_LOG_LEVEL=DEBUG aragora serve --api-port 8080 --ws-port 8765
```

### Configuration validation

```bash
# Check which env vars are set
env | grep ARAGORA_ | sort

# Verify .env file is loaded (if using direnv)
direnv status

# Check required keys
for key in ANTHROPIC_API_KEY OPENAI_API_KEY; do
  if [ -n "${!key}" ]; then
    echo "$key: set (${#!key} chars)"
  else
    echo "$key: NOT SET"
  fi
done
```

### Log analysis

```bash
# Recent errors
journalctl -u aragora --since "1 hour ago" --no-pager | grep -c ERROR

# Most common error types
journalctl -u aragora --since "1 hour ago" --no-pager \
  | grep ERROR | sed 's/.*ERROR//' | sort | uniq -c | sort -rn | head -10

# Slow requests (if access log enabled)
grep "took [0-9]*\.[0-9]*s" /var/log/aragora/access.log \
  | awk '{print $NF}' | sort -n | tail -10
```

### Database inspection

```bash
# SQLite: check schema version
sqlite3 /path/to/aragora.db "SELECT * FROM _schema_versions;"

# SQLite: check table sizes
sqlite3 /path/to/aragora.db "
SELECT name, SUM(pgsize) as size_bytes
FROM dbstat
GROUP BY name
ORDER BY size_bytes DESC
LIMIT 10;
"

# PostgreSQL: check connection count
psql "$DATABASE_URL" -c "SELECT count(*) FROM pg_stat_activity;"
```
