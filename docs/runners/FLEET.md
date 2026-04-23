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
| `macbook-m1-16gb` | MacBook-Pro16GB.local | 10.0.0.170 | macOS ARM64 | `self-hosted, aragora, macOS, ARM64` | Apple-silicon MacBook |
| `macbook-intel-64gb` | MacBook-Pro-3.local | 10.0.0.193 | macOS x64 | `self-hosted, aragora, macOS, X64` | Intel MacBook, more cores |

**Total online: 12**

## Past incidents

### 2026-04-23 — TCP port exhaustion locked out two Mac runners

**Symptom:** `macbook-m1-16gb` and `macbook-intel-64gb` had local `~/actions-runner` installs, active LaunchAgents, and live `Runner.Listener` processes, but never appeared in the GitHub runner API. Log showed `Can't assign requested address (pipelinesghubeus7.actions.githubusercontent.com:443)` on every retry.

**Diagnosis** (after two misdiagnoses that blamed IPv6 DNS and Tailscale routing):

- **Uptime: 93 days** on both Macs
- **TIME_WAIT entries: 31,857** on Pro16GB (ephemeral port range 49152–65535, only 16,384 ports)
- `connect()` to **any** address — including `127.0.0.1` — failed with `EADDRNOTAVAIL` at source-port allocation
- ping worked (no ephemeral port needed); TCP did not

**Fix:** reboot both Macs. `sysctl -w net.inet.tcp.msl=1000` was tried live to accelerate TIME_WAIT drain but only affects new entries — the 32K already on 15-second timers refilled the port range before draining.

**After reboot** (both Macs rebooted 2026-04-23 by the founder), both runners registered within ~60s and have been online since.

**Monitoring suggestion** (not yet implemented): per-host daily `launchd` job that alerts when `netstat | grep TIME_WAIT | wc -l` crosses ~5,000. Catches the condition before the stack locks.

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
