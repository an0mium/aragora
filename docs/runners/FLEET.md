# Self-Hosted Runner Fleet

Canonical roster of self-hosted GitHub Actions runners registered on
`synaptent/aragora`. Sourced from the live
`gh api repos/synaptent/aragora/actions/runners` endpoint cross-checked
against on-host `~/actions-runner/.runner` configs.

Last confirmed: **2026-04-23**.

## Registered and online

| Runner name (GH) | Host | IP | OS | Labels | Notes |
|---|---|---|---|---|---|
| `aragora-hetzner-cpu1` | Hetzner CX52 cloud | — | Linux x64 | `self-hosted, Linux, X64, aragora, hetzner` | Background CPU work |
| `aragora-hetzner-cpu2` | Hetzner CX52 cloud | — | Linux x64 | `self-hosted, Linux, X64, aragora, hetzner` | Background CPU work |
| `aragora-hetzner-cpu3` | Hetzner CX52 cloud | — | Linux x64 | `self-hosted, Linux, X64, aragora, hetzner` | Background CPU work |
| `i-07e538fafbe61696d` | AWS EC2 (production) | — | Linux x64 | `self-hosted, Linux, X64, aragora` | Production canary — also hosts the Aragora service |
| `i-0823e60c7c4b924e1` | AWS EC2 (production) | — | Linux x64 | `self-hosted, Linux, X64, aragora` | Production canary — also hosts the Aragora service |
| `ip-10-50-1-235` | AWS EC2 (staging) | 10.50.1.235 | Linux x64 | `self-hosted, Linux, X64, aragora` | Staging tier |
| `ip-172-31-7-189` | AWS EC2 (staging) | 172.31.7.189 | Linux x64 | `self-hosted, Linux, X64, aragora` | Staging tier |
| `ip-172-31-11-203` | AWS EC2 (staging) | 172.31.11.203 | Linux x64 | `self-hosted, Linux, X64, aragora` | Staging tier |
| `ip-172-31-24-39` | AWS EC2 (staging) | 172.31.24.39 | Linux x64 | `self-hosted, Linux, X64, aragora` | Staging tier |
| `mac-studio-m3ultra` | Mac Studio (local LAN) | 10.0.0.62 / 10.0.0.90 | macOS ARM64 | `self-hosted, aragora, macOS, ARM64, mac-studio` | Apple-silicon workloads |

**Total online: 10**

## Locally installed but not phoning home

Runners that have an `~/actions-runner` install + LaunchAgent + live `Runner.Listener` process but do NOT appear in the GitHub runner API. Root cause: IPv6 DNS resolution failure for `pipelinesghubeus7.actions.githubusercontent.com` on the local LAN (DNS served by Tailscale at `100.100.100.100` returns an A record for IPv4 but AAAA lookup fails, and the .NET HTTP client happy-eyeballs path hits "Can't assign requested address"). See issue #6474 for the full diagnostic trail.

| Local config name | Host | IP | Agent ID | Status |
|---|---|---|---|---|
| `macbook-m1-16gb` | MacBook-Pro16GB.local | 10.0.0.170 | 33 | Listener retrying since 2026-03-20; IPv6 DNS error |
| `macbook-intel-64gb` | MacBook-Pro-3.local | 10.0.0.193 | 34 | Listener just restarted 2026-04-23T17:08Z |

**Fix recipe** (apply to each Mac):

```bash
ssh armand@<host>.local

# Option A — append IPv4 pinning to /etc/hosts for the actions endpoint
sudo tee -a /etc/hosts <<'HOSTS'
# Pin GH Actions pipelines to IPv4 until IPv6 DNS via Tailscale is fixed.
20.246.184.240 pipelinesghubeus7.actions.githubusercontent.com
HOSTS

# Option B — force .NET to prefer IPv4 via the runner .env
echo "DOTNET_SYSTEM_NET_DISABLEIPV6=1" >> ~/actions-runner/.env

# Either way, restart the runner
launchctl kickstart -k gui/$(id -u)/com.github.actions-runner

# Verify — should appear in `gh api repos/synaptent/aragora/actions/runners` within 60s
gh api repos/synaptent/aragora/actions/runners --jq '.runners | map(select(.name | test("macbook"))) | .[].name'
```

## How to add a runner

1. On the host, create `~/actions-runner`, download the action-runner tarball, extract.
2. Generate a registration token from **Settings → Actions → Runners → New self-hosted runner** on the repo page.
3. Run `./config.sh --url https://github.com/synaptent/aragora --token <token>`. Pick a memorable name matching `<arch>-<variant>`, e.g. `macbook-m3-96gb`, `hetzner-gpu1`.
4. Install as a service: `./svc.sh install && ./svc.sh start` (Linux) or use `install-and-run.sh` + LaunchAgent (macOS).
5. Add the runner to this file. Reviewer of the FLEET.md change confirms the new headcount matches the live API.

## How to re-register a stale runner

If a host has a local install but GH API doesn't see it:

```bash
cd ~/actions-runner
./svc.sh stop || true
./svc.sh uninstall || true
./config.sh remove --token <removal-token>   # may fail if token invalid; OK to proceed
rm -f .credentials .runner .credentials_rsaparams

# Fresh registration
TOKEN=$(gh api -X POST repos/synaptent/aragora/actions/runners/registration-token --jq .token)
./config.sh --url https://github.com/synaptent/aragora --token "$TOKEN" \
  --name <canonical-name> --labels self-hosted,aragora,<platform-tags> --unattended
./svc.sh install && ./svc.sh start
```

## Monitoring

A scheduled workflow at `.github/workflows/runner-headcount-monitor.yml` polls
the runner API daily and alerts when the count drifts from the committed
baseline in this file. See that workflow for threshold + notification
configuration.
