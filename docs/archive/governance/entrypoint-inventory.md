# Aragora Runtime Entrypoint Inventory

**Date:** March 10, 2026
**Campaign:** phase0a-bootstrap-governance
**Task:** phase0a-004
**Purpose:** Enumerate all execution surfaces in the repository, classify
each by status, and provide the canonical invocation for each.

## Classification Key

| Status | Meaning |
|--------|---------|
| **canonical** | Primary production path; documented and tested |
| **alternate** | Works but is not the recommended path |
| **deprecated** | Superseded; should be removed or redirected |
| **broken** | Known to be non-functional |
| **unknown** | Untested; status not verified |

---

## 1. Backend Server

| Entrypoint | Command | Status |
|-----------|---------|--------|
| Server `__main__` | `python -m aragora.server --api-port 8080 --ws-port 8765` | **canonical** |
| CLI wrapper | `aragora serve --api-port 8080 --ws-port 8765` | alternate |
| Package `__main__` | `python -m aragora` | alternate (dispatches to CLI) |
| Makefile | `make serve` | alternate (calls `python -m aragora.server`) |

**Canonical command:** `python -m aragora.server`

**Ports:** HTTP 8080, WebSocket 8765 (configurable via `--api-port`, `--ws-port`)

**Drift note:** EC2 systemd and CI workflows use `aragora serve` (CLI wrapper)
instead of `python -m aragora.server`. Both work but the CLI path adds lazy-load
overhead. Recommend standardizing on the module path.

---

## 2. CLI

| Entrypoint | Command | Status |
|-----------|---------|--------|
| CLI main | `python -m aragora.cli.main` or `aragora` | **canonical** |
| Legacy GT | `aragora/cli/gt.py` | deprecated |
| Legacy REPL | `aragora/cli/repl.py` | deprecated (moved to commands/) |

**Registered subcommands (40+):**

| Command | Module | Category |
|---------|--------|----------|
| `ask` / `debate` | `debate.py` | Core |
| `decide` | `decide.py` | Core |
| `serve` | `server.py` | Core |
| `status` | `status.py` | Core |
| `stats` | `stats.py` | Core |
| `doctor` | `doctor.py` | Core |
| `explain` | `explain.py` | Core |
| `consensus` | `consensus.py` | Core |
| `verify` | `verify.py` | Core |
| `receipt` | `receipt.py` | Core |
| `workflow` | `workflow.py` | Expansion |
| `pipeline` | `pipeline.py` | Expansion |
| `knowledge` | `knowledge.py` | Core |
| `compliance` | `compliance.py` | Expansion |
| `skills` | `skills.py` | Expansion |
| `agents` | `delegated.py` | Core |
| `swarm` | `swarm.py` | Core |
| `nomic` | `nomic.py` | Core |
| `self-improve` | `self_improve.py` | Core |
| `triage` | `triage.py` | Expansion |
| `mcp` | `delegated.py` | Expansion |
| `marketplace` | `delegated.py` | Expansion |
| `billing` | `delegated.py` | Expansion |
| `control-plane` | `delegated.py` | Core |
| `gauntlet` | `delegated.py` | Expansion |
| `verticals` | `verticals.py` | Expansion |
| `analytics` | `analytics.py` | Expansion |
| `handlers` | `handlers.py` | Internal |
| `tools` | `tools.py` | Internal |
| `config` | `delegated.py` | Core |
| `replay` | `delegated.py` | Expansion |
| `export` | `delegated.py` | Expansion |
| `init` / `setup` | `delegated.py` | Core |
| `demo` | `delegated.py` | Demo |
| `playbook` | `playbook.py` | Expansion |
| `worktree` | `worktree.py` | Core |
| `coordinate` | `coordinate.py` | Expansion |
| `healthcare` | `healthcare.py` | Vertical |
| `inbox-wedge` | `inbox_wedge.py` | Expansion |

---

## 3. Workers

| Entrypoint | Command | Status |
|-----------|---------|--------|
| Queue worker | `python scripts/queue_worker.py --worker-id ID --concurrency N` | **canonical** |
| Control plane worker | `python scripts/control_plane_deliberation_worker.py` | alternate |
| Docker worker | `python -m scripts.queue_worker --concurrency 3` | **canonical** (production) |

**Canonical command:** `python scripts/queue_worker.py`

**Drift note:** Docker Compose production uses `python -m scripts.queue_worker`
while Kubernetes Helm chart sets `WORKER_MODE=true` env var but provides no
explicit entrypoint command — it relies on the Docker image default CMD, which
is the *server*, not the worker. This is a real gap.

---

## 4. Self-Improvement / Nomic

| Entrypoint | Command | Status |
|-----------|---------|--------|
| Nomic loop | `python scripts/nomic_loop.py --cycle N` | **canonical** |
| Staged execution | `python scripts/nomic_staged.py [phase]` | alternate |
| Goal-driven | `python scripts/self_develop.py --goal "..."` | **canonical** |
| Campaign execution | `aragora swarm campaign run --manifest path.yaml` | **canonical** |
| Nomic eval | `python scripts/nomic_eval.py` | alternate |
| Nomic live fire | `python scripts/nomic_live_fire.py` | alternate |

---

## 5. Frontend (Next.js)

| Entrypoint | Command | Status |
|-----------|---------|--------|
| Dev server | `cd aragora/live && npm run dev` | **canonical** (dev) |
| Production build | `cd aragora/live && npm run build:standalone` | **canonical** (prod) |
| Production serve | `cd aragora/live && npm start` | **canonical** (prod) |
| Docker | `node server.js` (standalone Next.js) | **canonical** (container) |

**Ports:** 3000 (dev and prod)

---

## 6. WebSocket

| Entrypoint | Command | Status |
|-----------|---------|--------|
| Integrated WS | Started by `python -m aragora.server --ws-port 8765` | **canonical** |

WebSocket is not a separate process — it runs inside the unified server.
Event types: `debate_start`, `round_start`, `agent_message`, `critique`,
`vote`, `consensus`, `debate_end` (190+ total event types).

---

## 7. Scheduled Jobs

| Entrypoint | Command | Status |
|-----------|---------|--------|
| Scheduler module | `aragora/scheduler/` | **expansion** |
| Worktree maintainer | `python3 scripts/codex_worktree_autopilot.py maintain` | **canonical** (macOS launchd) |
| PR watch daemon | `aragora openclaw watch` | **canonical** (3 Macs) |

No crontab or OS-level scheduled jobs beyond the launchd agents.

---

## 8. Python Module Entrypoints (`__main__.py`)

| Module | Command | Status |
|--------|---------|--------|
| `aragora` | `python -m aragora` | **canonical** (CLI dispatcher) |
| `aragora.server` | `python -m aragora.server` | **canonical** (server) |
| `aragora.gauntlet` | `python -m aragora.gauntlet spec.md` | **canonical** |
| `aragora.mcp` | `python -m aragora.mcp` | **canonical** |
| `aragora.migrations` | `python -m aragora.migrations [cmd]` | **canonical** |

---

## 9. Docker / Container Entrypoints

| Surface | Backend CMD | Worker CMD |
|---------|------------|------------|
| Root `Dockerfile` | `python -m aragora.server --host 0.0.0.0 --http-port 8080 --ws-port 8765` | N/A |
| `Dockerfile.backend` | `python -m aragora.server --host 0.0.0.0 --api-port 8080 --ws-port 8765` | N/A |
| `Dockerfile.frontend` | `node server.js` | N/A |
| `docker-compose.production.yml` | Via `/app/scripts/docker-entrypoint.sh` | `python -m scripts.queue_worker --concurrency 3` |
| `docker-compose.quickstart.yml` | Image default CMD | N/A |
| `docker-compose.simple.yml` | Image default CMD | N/A |

**Entrypoint script** (`deploy/scripts/docker-entrypoint.sh`): Runs migrations,
loads secrets, then starts server. Only used in production compose.

---

## 10. Kubernetes / Helm

| Component | Replicas | Autoscaling | Explicit CMD |
|-----------|----------|-------------|--------------|
| Backend | 3 | 2–20 (CPU 70%, mem 80%) | **None** (uses image default) |
| Debate worker | 2 | 2–10 (CPU 60%) | **None** (uses `WORKER_MODE=true` env) |
| Frontend | 2 | Manual | **None** (uses image default) |

**Drift note:** No Helm template sets an explicit `command:` or `args:`.
All rely on Docker image defaults. The debate-worker template sets
`WORKER_MODE=true` but the image CMD is the server, not a worker — this is
likely a bug or relies on an undocumented startup-path branch.

---

## 11. Key Shell Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/demo.sh` | Full-stack demo startup | canonical (demo) |
| `scripts/install.sh` | Initial setup | canonical |
| `scripts/quick-start.sh` | Fast startup | alternate |
| `scripts/codex_session.sh` | Codex worktree bootstrap | canonical |
| `scripts/healthcheck.sh` | Health verification | canonical |
| `scripts/smoke_test.sh` | Quick smoke test | canonical |
| `scripts/docker-entrypoint.sh` | Docker production init | canonical |
| `scripts/aragora-deploy.sh` | EC2 deployment | canonical |
| `scripts/pre_deploy_check.sh` | Pre-deploy validation | canonical |
| `scripts/cleanup_worktrees.sh` | Worktree cleanup | canonical |
| `scripts/codex_worktree_autopilot.py` | Fleet worktree management | canonical |
| `scripts/gmail_oauth_setup.py` | Gmail OAuth (one-time) | canonical |

---

## 12. Makefile Targets (Process-Starting)

| Target | Command | Purpose |
|--------|---------|---------|
| `make serve` | `python -m aragora.server --api-port 8080 --ws-port 8765` | Dev server |
| `make repl` | `python -m aragora.cli.main repl` | Interactive REPL |
| `make doctor` | `python -m aragora.cli.doctor` | Health check |
| `make demo` | `bash scripts/demo.sh` | Full-stack demo |
| `make demo-docker` | `docker compose -f deploy/demo/docker-compose.yml up` | Docker demo |
| `make quickstart` | `docker compose -f docker-compose.quickstart.yml up` | Zero-config |
| `make quickstart-live` | `docker compose -f docker-compose.simple.yml up` | SQLite mode |
| `make db-migrate` | `python -m aragora.migrations.run` | Run migrations |
| `make test` | `pytest tests/ -v --timeout=120` | Full test suite |
| `make test-fast` | `pytest` with marker filters | Fast tests |
| `make lint` | `ruff check aragora/ tests/` | Linting |
| `make typecheck` | `mypy aragora/` | Type checking |
| `make ci-required` | All 5 required CI checks | Pre-push gate |

---

## Summary

| Category | Count | Canonical | Alternate | Deprecated |
|----------|-------|-----------|-----------|------------|
| Backend server | 4 | 1 | 3 | 0 |
| CLI commands | 40+ | 40+ | 0 | 2 |
| Workers | 3 | 2 | 1 | 0 |
| Self-improvement | 6 | 3 | 3 | 0 |
| Frontend | 4 | 3 | 1 | 0 |
| WebSocket | 1 | 1 | 0 | 0 |
| Scheduled | 3 | 2 | 0 | 1 |
| Module `__main__` | 5 | 5 | 0 | 0 |
| Docker containers | 6+ | 4 | 2 | 0 |
| Kubernetes | 3 | 0 | 3 | 0 |
| Shell scripts | 12+ | 9 | 3 | 0 |
| Makefile targets | 12+ | 12 | 0 | 0 |
| **Total** | **~100** | **~82** | **~16** | **~3** |

## Critical Findings

1. **Kubernetes worker entrypoint is undefined** — relies on image default
   which is the server, not a worker process
2. **EC2 and CI use `aragora serve`** (CLI) while Docker uses
   `python -m aragora.server` (module) — recommend standardizing
3. **Health check endpoints inconsistent** across surfaces: `/healthz`,
   `/api/v1/health`, `/health/live`, `/api/health`
4. **40+ CLI commands** but only ~15 are in the canonical core loop;
   the rest are expansion features
5. **No explicit worker command in Helm templates** — fragile implicit
   dependency on Docker image CMD
