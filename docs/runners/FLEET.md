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

Runners that have an `~/actions-runner` install + LaunchAgent + live `Runner.Listener` process but do NOT appear in the GitHub runner API.

| Local config name | Host | IP | Agent ID | Diagnosis |
|---|---|---|---|---|
| `macbook-m1-16gb` | MacBook-Pro16GB.local | 10.0.0.170 | 33 | Blocked by Tailscale secondary default route (see below) |
| `macbook-intel-64gb` | MacBook-Pro-3.local | 10.0.0.193 | 34 | Same — both on the same LAN / Tailscale config |

### Root cause: Tailscale secondary default route captures outbound TCP

On the founder's local LAN, Tailscale has installed a **second default route** via `utun4` that sits alongside the real `10.0.0.1 → en0` default:

```
Internet:
Destination        Gateway            Flags               Netif
default            10.0.0.1           UGScg                 en0
default            link#33            UCSIg               utun4    <-- Tailscale
```

Symptoms (verified on both Macs, 2026-04-23):

- `ping 10.0.0.1` and `ping 8.8.8.8` work — **ICMP traverses the en0 default correctly**
- `curl https://...` to ANY host returns `http=000` — **TCP connect() fails with `Can't assign requested address` (EADDRNOTAVAIL, errno 49)**
- `nc -vz 20.246.184.240 443` reports the same EADDRNOTAVAIL
- DNS resolution is healthy (IPv4 A records resolve; AAAA lookups fail but that's a symptom, not the cause)

The kernel selects utun4 for the TCP source address because it's the second default route; utun4 only has a link-local IPv6 address (`fe80::...%utun4`) so there's no valid IPv4 source to bind, and connect() fails before the first SYN.

The runner's earlier log entries that blamed IPv6 (`Can't assign requested address (pipelinesghubeus7.actions.githubusercontent.com:443)`) are a downstream manifestation of the same routing issue — the error text includes the hostname but the actual failure is at TCP bind, not DNS.

### Fix (requires founder intervention)

This is a Tailscale configuration issue, NOT a runner configuration issue. The two Macs cannot make outbound TCP connections to ANY host until the Tailscale routing is corrected. Options, in increasing order of scope:

1. **Disable Tailscale on the affected Macs** (`tailscale down` or via the menubar UI). Default route via utun4 disappears; runners immediately work. Cost: loses Tailscale overlay on those Macs.
2. **Reconfigure Tailscale to not claim a default route.** If Tailscale is running with `--accept-routes` + an exit-node advertisement that's installing the default, either disable the exit-node use or remove the advertisement. Typical command: `tailscale set --exit-node=` (unset exit node).
3. **Advertise specific subnet routes only** instead of a default. Keeps Tailscale overlay for its intended purpose without capturing the whole default.

Once the secondary default is removed, the existing runner installs will phone home within 60s of the next retry (30s retry interval already configured).

### After applying the fix

```bash
# Verify the extra default is gone:
netstat -rn -f inet | grep default
# Should show only: default  10.0.0.1  UGScg  en0

# Restart the runner (otherwise it'll wait up to 30s for the next retry):
launchctl unload ~/Library/LaunchAgents/com.github.actions-runner.plist
launchctl load ~/Library/LaunchAgents/com.github.actions-runner.plist

# Check GH API registration within 60s:
gh api repos/synaptent/aragora/actions/runners --jq '.runners | map(select(.name | test("macbook"))) | .[].name'
# Expected: macbook-m1-16gb and macbook-intel-64gb
```

Then bump `BASELINE_COUNT` in `.github/workflows/runner-headcount-monitor.yml` from 10 → 12 in a follow-up PR.

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
