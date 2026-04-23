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
| `macbook-m1-16gb` | MacBook-Pro16GB.local | 10.0.0.170 | 33 | TCP port exhaustion — needs reboot |
| `macbook-intel-64gb` | MacBook-Pro-3.local | 10.0.0.193 | 34 | Same class; not separately verified |

### Root cause: kernel ephemeral-port exhaustion via TIME_WAIT accumulation

Verified on MacBook-Pro16GB.local on 2026-04-23:

- **Uptime: 93 days** (no reboot since January)
- **TCP table total entries: 33,204**
- **TIME_WAIT entries: 31,857**
- **Ephemeral port range: 49152–65535** (16,384 ports)
- **`connect() from 127.0.0.1` fails** — even Tailscale's own CLI gets `EADDRNOTAVAIL` hitting its local daemon at `127.0.0.1:57246`
- `ping 10.0.0.1` works (ICMP doesn't need a source port)

The kernel has burned through the ephemeral port range with 32K stuck TIME_WAITs. Any new outbound TCP call — local or remote — fails with `EADDRNOTAVAIL` at the source-port-allocation step, before a SYN ever goes out.

What was previously attributed to "IPv6 DNS" and "Tailscale default route" was misdiagnosis: the `Can't assign requested address` error message in the runner's log IS a port-allocation failure, and the two default-route entries in `netstat` are unrelated (the en0 default is used for the packets that actually succeed, like ping).

### Why TIME_WAITs accumulated

Not definitively diagnosed. Candidates:

1. The `Runner.Listener` retry loop has been active ~2 months; each retry may complete a TCP 4-way handshake before aborting, leaving a TIME_WAIT each time. 2 months × 2-per-minute × 30s MSL window = possible backlog if many connections pile up.
2. Another process on the Mac churns local connections faster than MSL drains them.

Identifying the exact source requires cleared state — easier to reboot and re-measure over a week than to forensically unwind 93 days of cruft.

### Fix: reboot

Nothing short of a reboot clears the TCP state table. `sysctl -w net.inet.tcp.msl=1000` was attempted live (lowering MSL from 15s to 1s) but only affects **new** TIME_WAITs — existing ones sit on their original 15-second timers and, with ~32K of them, refill the port range before they drain.

```bash
# On each affected Mac:
ssh armand@<host>.local
sudo shutdown -r now
# Wait ~2 min for reboot; if auto-login is disabled, log in manually so the LaunchAgent runs
```

### After reboot

```bash
# Verify TIME_WAIT count is sane (should be < a few hundred):
ssh armand@<host>.local "netstat -an -p tcp | awk '\$NF==\"TIME_WAIT\"' | wc -l"

# The LaunchAgent + Runner.Listener autostart on login. Verify via GH API within 60s:
gh api repos/synaptent/aragora/actions/runners --jq '.runners | map(select(.name | test("macbook"))) | .[].name'
# Expected: macbook-m1-16gb and macbook-intel-64gb
```

`BASELINE_COUNT` in `.github/workflows/runner-headcount-monitor.yml` is now 12 to match the restored fleet.

### Mitigation for next time (not-yet-opened follow-up)

Add a per-host daily `launchd` job that alerts if `netstat | grep TIME_WAIT | wc -l` crosses a threshold (say 5,000). Catches the condition early before it locks the stack.

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
