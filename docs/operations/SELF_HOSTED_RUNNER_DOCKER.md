# Self-Hosted Runner Docker Provisioning

The `aragora`-labeled self-hosted GitHub Actions runners host VPC-resident
workflows that need Docker — notably:

- `load-tests.yml` — uses `services: redis:7-alpine` and runs k6 through
  the pinned `grafana/k6` Docker image

This document captures how Docker is provisioned on those runners and how to
provision new ones consistently.

## The `docker-ready` label convention

Workflows that need Docker MUST target the `docker-ready` custom label
in addition to `aragora`:

```yaml
runs-on: [self-hosted, Linux, X64, aragora, docker-ready]
```

The `aragora` label alone is insufficient because the fleet is heterogeneous
— see the `aragora` fleet table below — and not every runner has Docker.

The `docker-ready` label is added to a runner ONLY after the SSM Docker
install dance (Steps 1-3 below) has completed AND
`sudo -u ec2-user docker ps` returns successfully.

### Adding the label to a runner

```bash
# Find the runner ID:
gh api repos/synaptent/aragora/actions/runners --jq \
  '.runners[] | select(.name == "<runner-name>") | .id'

# Add the label:
gh api -X POST /repos/synaptent/aragora/actions/runners/<id>/labels \
  -f 'labels[]=docker-ready'

# Verify:
gh api /repos/synaptent/aragora/actions/runners/<id>/labels
```

### Removing the label (e.g. if Docker is broken on that runner)

```bash
gh api -X DELETE \
  /repos/synaptent/aragora/actions/runners/<id>/labels/docker-ready
```

## Current `aragora` fleet (as of 2026-04-24)

| Runner Name | Type | Status | Docker? | `docker-ready` label? |
|---|---|---|---|---|
| `aragora-hetzner-cpu1` | Hetzner CPU | online | unknown | no |
| `aragora-hetzner-cpu2` | Hetzner CPU | online | unknown | no |
| `aragora-hetzner-cpu3` | Hetzner CPU | online | unknown | no |
| `i-07e538fafbe61696d` | EC2 AL2023 | online | no | no |
| `i-0823e60c7c4b924e1` | EC2 AL2023 | online | no | no |
| `ip-10-50-1-235` | unknown | online | unknown | no |
| `ip-172-31-11-203` | EC2 AL2023 | online | no | no |
| `ip-172-31-24-39` (id=26) | EC2 AL2023 (i-0aae2ccd2f68b94d2) | online | **yes (25.0.14)** | **yes** |
| `ip-172-31-7-189` | EC2 AL2023 | online | no | no |
| `macbook-intel-64gb` | Mac Intel | online | unknown | no |
| `macbook-m1-16gb` | Mac M1 | online | unknown | no |
| `mac-studio-m3ultra` | Mac M3 Ultra | online | unknown | no |

The GitHub Actions runner service on EC2 AL2023 runs as `ec2-user` (systemd
unit `actions.runner.synaptent-aragora.<hostname>.service`). Mac runners use
`launchd` and Hetzner runners use systemd as well.

### Outstanding fleet work (tracked separately)

- Provision Docker on the remaining 4 EC2 AL2023 runners using the SSM
  procedure below; add `docker-ready` to each as `docker ps` succeeds.
- Audit Hetzner runner Docker state; if Docker is present, add `docker-ready`.
- Audit Mac runner Docker state (Docker Desktop or colima); if present, add
  `docker-ready`.
- Bake Docker into a custom AMI so newly-rotated EC2 runners inherit it.

## Required packages

- `docker` (Amazon Linux 2023 package, currently 25.x)
- The runner user (`ec2-user`) must be in the `docker` group
- The runner service must be restarted after group membership changes so
  that new supplementary groups take effect in the runner process

The workflow intentionally does **not** install k6 with OS packages. Amazon
Linux 2023 does not have `apt-get`, and the previous GitHub-hosted runner
version used Debian package commands. Instead, `load-tests.yml` pulls the
pinned `grafana/k6` image and runs k6 with Docker host networking so
`localhost` continues to reach the Aragora server started by the job.

Because self-hosted runners are persistent and may host resident Aragora
services on the default ports, `load-tests.yml` uses isolated job-local ports
instead of `8080`/`8765`/`9090`:

- HTTP API: `18080`
- WebSocket family: `18765`-`18768`
- Prometheus metrics: `19090`

The debate WebSocket burst test connects with the `aragora-v1` subprotocol,
matching `DebateStreamServer`'s handshake contract.

Before startup the workflow checks only those isolated ports and stops stale
Aragora listeners left by prior interrupted load-test jobs. Cleanup sends
`TERM`, waits for listeners to drain, escalates to `KILL` only for remaining
Aragora server listeners, and fails closed if another service owns one of the
isolated ports. The workflow also asserts that the server process it started is
still alive after health/readiness checks, so a stale listener cannot mask a
failed fresh startup.

The workflow pre-cleans `$GITHUB_WORKSPACE` before checkout. This is scoped to
the self-hosted runner job workspace and prevents generated files from prior
runs from causing `actions/checkout` to leave the working tree partially dirty.

Note: `docker-compose-plugin` is NOT available in the default AL2023 repo.
Workflows that rely on `docker compose` should either install it via the
upstream docker-ce repo or use `services:` containers instead.

## Provisioning (via AWS SSM)

Runners can be provisioned without SSH using AWS Systems Manager
Session Manager (the `aragora-ec2-ssm` instance profile grants
`AmazonSSMManagedInstanceCore`).

### Step 1 — Install Docker

```bash
aws ssm send-command \
  --instance-ids <instance-id> \
  --document-name "AWS-RunShellScript" \
  --comment "Install Docker for aragora self-hosted runner" \
  --parameters 'commands=[
    "set -euo pipefail",
    "sudo dnf install -y docker",
    "sudo systemctl enable --now docker",
    "sudo docker --version",
    "sudo usermod -aG docker ec2-user",
    "id ec2-user"
  ]'
```

### Step 2 — Restart the runner service so group membership takes effect

```bash
aws ssm send-command \
  --instance-ids <instance-id> \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=[
    "sudo systemctl restart actions.runner.*",
    "sleep 3",
    "sudo systemctl status actions.runner.* --no-pager | head -20"
  ]'
```

### Step 3 — Verify

```bash
# Docker daemon reachable by ec2-user without sudo:
aws ssm send-command \
  --instance-ids <instance-id> \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["sudo -u ec2-user docker ps"]'

# Runner online on GitHub:
gh api repos/synaptent/aragora/actions/runners --jq '.runners[] | select(.name == "<runner-name>")'
```

## History

- **2026-04-24** — Docker installed on `i-0aae2ccd2f68b94d2` to restore
  Load Tests to self-hosted (reverts #6554 which had temporarily moved
  Load Tests to `ubuntu-latest` because Docker was missing). See
  `.github/workflows/load-tests.yml`.
- **2026-04-24 (later)** — `docker-ready` custom label added to runner
  `ip-172-31-24-39` (id=26). `load-tests.yml` updated to target
  `[self-hosted, Linux, X64, aragora, docker-ready]` so the workflow
  no longer races against runners that lack Docker.
- **2026-04-25** — k6 execution moved from Ubuntu/Debian `apt-get`
  installation to the pinned `grafana/k6` Docker image, making the workflow
  compatible with the AL2023 runner targeted by the `docker-ready` label.
- **2026-04-25 (later)** — Added stale-port cleanup for persistent self-hosted
  runners after a dispatch proved stale listeners on `8080`/`8765` can survive
  previous interrupted jobs and break a fresh load-test run.
- **2026-04-25 (cleanup hardening)** — Stale-port cleanup now deduplicates
  listener PIDs and escalates from `TERM` to `KILL` if an interrupted
  `aragora.server` process does not release the fixed load-test ports.
- **2026-04-25 (startup verification)** — The workflow now verifies the
  newly-started server PID is alive before running k6, preventing a resident or
  stale listener from masking a failed fresh startup.
- **2026-04-25 (port isolation)** — Load tests moved from the default
  `8080`/`8765`/`9090` ports to isolated `18080`/`18765+`/`19090` ports so the
  workflow does not fight resident services on persistent self-hosted runners.
- **2026-04-25 (auxiliary WS ports)** — `run_unified_server()` now honors
  `ARAGORA_CONTROL_PLANE_WS_PORT`, `ARAGORA_NOMIC_LOOP_WS_PORT`, and
  `ARAGORA_CANVAS_WS_PORT`, allowing CI to isolate the full WebSocket family.
- **2026-04-25 (workspace isolation)** — Added a pre-checkout workspace clean
  so generated files from prior self-hosted runs cannot block checkout updates.
- **2026-04-25 (WebSocket handshake)** — k6 WebSocket load scripts now request
  the `aragora-v1` subprotocol required by the debate stream server.

## When to add ubuntu-latest fallback

If you spin up a new workflow that needs Docker AND:

- The workflow doesn't need VPC access to private AWS resources
- The workflow doesn't need GPU or high-memory instances

…then `runs-on: ubuntu-latest` is a simpler choice — GitHub-hosted runners
come with Docker pre-installed and are included in your Actions minute
allowance. Use self-hosted only when VPC access or specific hardware
is required.
