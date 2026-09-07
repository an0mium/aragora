# Aragora Deploy Truth Table

**Date:** March 10, 2026
**Campaign:** phase0a-bootstrap-governance
**Task:** phase0a-005
**Purpose:** Map every deploy surface to actual backend and worker commands,
identify drift between documented and real behavior, and record mismatches
for Phase 0B resolution.

---

## Environment Matrix

| Surface | Backend Command | Worker Command | Database | Redis | Frontend |
|---------|----------------|----------------|----------|-------|----------|
| **Local dev** | `python -m aragora.server --api-port 8080 --ws-port 8765` | `python scripts/queue_worker.py` | SQLite | Optional | `cd aragora/live && npm run dev` |
| **Docker dev** | `python -m aragora.server --host 0.0.0.0 --api-port 8080 --ws-port 8765` | N/A | SQLite or Postgres (profile) | Redis 7.2 | `node server.js` (port 3000) |
| **Docker quickstart** | Image default CMD | N/A | SQLite (`ARAGORA_OFFLINE=true`) | Disabled | `node server.js` |
| **Docker simple** | Image default CMD | N/A | SQLite (`ARAGORA_MODE=minimal`) | Disabled | N/A |
| **Docker demo** | `python -m aragora.server --offline --api-port 8080 --ws-port 8765` | N/A | SQLite | N/A | `node server.js` |
| **Docker production** | Via `docker-entrypoint.sh` → `python -m aragora.server --host 0.0.0.0 --http-port 8080 --ws-port 8765` | `python -m scripts.queue_worker --concurrency 3` | PostgreSQL 16.2 | Redis 7.2 (password) | N/A |
| **Kubernetes** | Image default CMD (no explicit command in Helm) | Image default CMD + `WORKER_MODE=true` | PostgreSQL (external secret) | Redis (if enabled) | `node server.js` |
| **EC2 production** | `aragora serve --api-port 8080 --ws-port 8765 --host 127.0.0.1` | N/A (no worker service) | PostgreSQL (env file) | N/A | N/A |
| **CI workflows** | `aragora serve [--demo] --api-port 8080 --ws-port 8765 --host 127.0.0.1` | N/A | SQLite | N/A | N/A |
| **OpenClaw gateway** | `aragora/gateway` image default CMD | N/A | PostgreSQL 15.6 | Redis 7.2 | N/A |

---

## Drift Register

Each drift item documents a specific mismatch between what is documented,
what is implemented, and the severity of the gap.

### DRIFT-001: Backend startup command inconsistency

| Aspect | Value |
|--------|-------|
| **Severity** | Medium |
| **Surfaces affected** | EC2, CI workflows |
| **Expected** | `python -m aragora.server` (module entrypoint) |
| **Actual** | `aragora serve` (CLI wrapper with lazy-load overhead) |
| **Impact** | Both paths work. CLI path adds ~0.5s startup overhead from lazy module loading. Risk: if CLI dispatch code has a bug, EC2 and CI break but Docker doesn't. |
| **Resolution** | Standardize EC2 systemd and CI workflows to use `python -m aragora.server` |

### DRIFT-002: Kubernetes worker has no explicit entrypoint

| Aspect | Value |
|--------|-------|
| **Severity** | High |
| **Surfaces affected** | Kubernetes debate-worker deployment |
| **Expected** | Explicit `command:` in Helm template starting a worker process |
| **Actual** | Template sets `WORKER_MODE=true` env var but no `command:` override; falls through to image CMD which starts the server |
| **Impact** | Worker pods may be running the full server instead of a dedicated worker process. If `WORKER_MODE=true` is not checked in the server startup code, workers are servers. |
| **Resolution** | Add explicit `command: ["python", "-m", "scripts.queue_worker"]` to debate-worker Helm template; see `docs/governance/worker-mode-env-vars.md` |

### DRIFT-003: Health check endpoint inconsistency

| Aspect | Value |
|--------|-------|
| **Severity** | Medium |
| **Surfaces affected** | All |
| **Endpoints in use** | `/healthz` (Makefile docs), `/api/v1/health` (Docker demo), `/health/live` + `/health/ready` (Kubernetes), `/api/health` (EC2 Nginx) |
| **Impact** | Monitoring, load balancers, and readiness probes hit different endpoints. If one endpoint is removed or renamed, some surfaces break silently. |
| **Resolution** | Standardize on `/api/v1/health` (liveness) and `/api/v1/health/ready` (readiness) across all surfaces |

### DRIFT-004: PostgreSQL version pinning inconsistency

| Aspect | Value |
|--------|-------|
| **Severity** | Low |
| **Surfaces affected** | Docker dev vs. Docker production |
| **Docker dev** | `postgres:15.6-alpine` |
| **Docker production** | `postgres:16.2-alpine` |
| **OpenClaw** | `postgres:15.6-alpine` |
| **Impact** | Minor — PostgreSQL 15→16 is backwards compatible. Risk: schema features used in production may not exist in dev. |
| **Resolution** | Pin all surfaces to 16.2-alpine |

### DRIFT-005: Docker entrypoint script only used in production

| Aspect | Value |
|--------|-------|
| **Severity** | Medium |
| **Surfaces affected** | Docker dev, quickstart, simple |
| **Production** | Uses `/app/scripts/docker-entrypoint.sh` (runs migrations, loads secrets) |
| **Dev/quickstart** | Skips entrypoint script entirely; uses image default CMD |
| **Impact** | Database migrations run automatically in production but not in dev containers. Dev containers may have schema drift from production. |
| **Resolution** | Use entrypoint script in all Docker surfaces; skip migration step via env var when using SQLite |

### DRIFT-006: EC2 has no worker service

| Aspect | Value |
|--------|-------|
| **Severity** | High |
| **Surfaces affected** | EC2 production |
| **Expected** | Queue worker process for async debate processing |
| **Actual** | Only the server process runs; no separate worker service in systemd |
| **Impact** | Async debate jobs dispatched via Redis Streams have no consumer on EC2. Works only if debate processing is synchronous or if Redis/worker is hosted elsewhere. |
| **Resolution** | Add a `aragora-worker.service` systemd unit or confirm that EC2 runs in single-instance synchronous mode |

### DRIFT-007: `ARAGORA_SINGLE_INSTANCE=true` undocumented

| Aspect | Value |
|--------|-------|
| **Severity** | Low |
| **Surfaces affected** | EC2 production |
| **Context** | EC2 deploy uses `ARAGORA_SINGLE_INSTANCE=true` to run without Redis/workers |
| **Impact** | Not documented in environment variable reference. Developers may not know this mode exists. |
| **Resolution** | Document in `docs/governance/worker-mode-env-vars.md` |

### DRIFT-008: Dead or unclear Docker Compose files

| Aspect | Value |
|--------|-------|
| **Severity** | Low |
| **Surfaces affected** | Repository root |
| **Files** | `docker-compose.dev.yml`, `docker-compose.sme.yml` |
| **Impact** | Unclear if these are used. May confuse operators choosing a compose file. |
| **Resolution** | Delete if unused or document purpose |

---

## Port Assignment Table

| Surface | HTTP | WebSocket | Nginx/Proxy | Frontend | Metrics |
|---------|------|-----------|-------------|----------|---------|
| Local dev | 8080 | 8765 | — | 3000 | — |
| Docker dev | 8080 | 8765 | — | 3000 | — |
| Docker production | 8080 (internal) | 8765 (internal) | 80/443 | — | — |
| Kubernetes | 8080 (ClusterIP) | 8765 (ClusterIP) | Istio | 3000 (ClusterIP) | 9090 |
| EC2 | 8080 (127.0.0.1) | 8765 (127.0.0.1) | 80 (Nginx) | — | — |
| CI | 8080 or 8090 | 8765 | — | — | — |

**CI drift:** `test.yml` uses port 8090 for the backend; all other surfaces use 8080.

---

## Database Backend Selection

| Surface | Default | Configurable | Version |
|---------|---------|-------------|---------|
| Local dev | SQLite | Yes (`ARAGORA_DB_BACKEND=auto`) | N/A |
| Docker dev | SQLite | Yes (Postgres profile) | 15.6 |
| Docker quickstart | SQLite | No | N/A |
| Docker simple | SQLite | No | N/A |
| Docker demo | SQLite | No | N/A |
| Docker production | PostgreSQL | No (mandatory) | 16.2 |
| Kubernetes | PostgreSQL | Yes (Helm values) | External |
| EC2 | PostgreSQL | Via env file | External |

---

## Secrets Management

| Surface | Method | Location |
|---------|--------|----------|
| Local dev | `.env` file (gitignored, loaded by direnv) | `.env` |
| Docker dev | Environment variables in compose | `docker-compose.yml` |
| Docker production | Environment variables + `.env` file | Compose env_file |
| Kubernetes | External Secrets Operator | `aragora-postgres-secret` etc. |
| EC2 | AWS Secrets Manager → env file | `/etc/aragora/env` (600 perms) |
| CI | GitHub Secrets | Workflow env blocks |

---

## Resource Limits

| Surface | Backend CPU | Backend RAM | Worker CPU | Worker RAM |
|---------|-------------|-------------|------------|------------|
| Local dev | Unbounded | Unbounded | Unbounded | Unbounded |
| Docker dev | Unbounded | Unbounded | N/A | N/A |
| Docker production | 4 CPU | 4 GB | 2 CPU | 2 GB |
| Kubernetes | 0.5–2 CPU | 1–4 GB | 1–4 CPU | 2–8 GB |
| EC2 | Instance-bound | Instance-bound | N/A | N/A |

---

## Recommended Fixes (Phase 0B Scope)

| Priority | Drift | Fix |
|----------|-------|-----|
| P0 | DRIFT-002 | Add explicit worker command to Kubernetes Helm template |
| P0 | DRIFT-006 | Add worker systemd unit to EC2 or document single-instance mode |
| P1 | DRIFT-001 | Standardize all surfaces on `python -m aragora.server` |
| P1 | DRIFT-003 | Unify health check endpoints to `/api/v1/health` |
| P1 | DRIFT-005 | Use entrypoint script in all Docker surfaces |
| P2 | DRIFT-004 | Pin PostgreSQL to 16.2 everywhere |
| P2 | DRIFT-007 | Document `ARAGORA_SINGLE_INSTANCE` env var |
| P2 | DRIFT-008 | Clean up dead compose files |
