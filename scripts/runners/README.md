# Runner health monitoring

Per-host daily monitoring for Mac self-hosted runners. Addresses issue #6478 following the incident post-mortem in `docs/runners/FLEET.md` "Past incidents".

## Files

- `mac_timewait_check.sh` — bash script that counts TIME_WAIT TCP entries and writes a structured log line. Alerts via a flag file when count exceeds threshold.
- `com.aragora.runner-health.plist` — LaunchAgent that runs the check daily at 06:00 local. Writes to `~/Library/Logs/aragora-runner-health.log`.

## Installation (per Mac runner)

```bash
# One-time on each Mac (mac-studio, macbook-m1-16gb, macbook-intel-64gb):

# 1. Copy the check script + plist
mkdir -p ~/actions-runner/runner-health
scp scripts/runners/mac_timewait_check.sh armand@<magicdns-host>:~/actions-runner/runner-health/
scp scripts/runners/com.aragora.runner-health.plist armand@<magicdns-host>:~/Library/LaunchAgents/
ssh armand@<magicdns-host> "chmod +x ~/actions-runner/runner-health/mac_timewait_check.sh"

# 2. Load the LaunchAgent
ssh armand@<magicdns-host> "launchctl load ~/Library/LaunchAgents/com.aragora.runner-health.plist"

# 3. Verify first run produced a log line
ssh armand@<magicdns-host> "tail -1 ~/Library/Logs/aragora-runner-health.log"
# Expected: ts=... host=... uptime=... tcp_total=... timewait=... established=... threshold=5000
```

## Alert paths

**Local (always works, even if network is broken):**
- Structured line appended to `~/Library/Logs/aragora-runner-health.log` daily
- Alert flag file at `~/Library/Logs/aragora-runner-health.alert` when threshold crossed
- Review with `tail -f ~/Library/Logs/aragora-runner-health.log` after any mysterious runner downtime

**Remote (GH Actions weekly poll):**
- `.github/workflows/mac-runner-health-poll.yml` runs Mondays 14:00 UTC
- SSHes into each Mac via `secrets.MAC_RUNNER_SSH_KEY`
- Reads the last log line; alerts if `timewait > 5000` or log is > 48h stale
- Opens / comments on a `runner-health`-labeled issue on drift

## Threshold rationale

- macOS ephemeral port range: 49152–65535 (16,384 ports)
- 5,000 = ~30% of the pool — well before exhaustion at ~15K, leaving headroom to investigate
- Actual exhaustion in #6474 was at 31,857 (~200% of the pool, kernel cycling)

## Secrets setup (one-time)

Add a GH secret `MAC_RUNNER_SSH_KEY` containing the SSH private key that can reach the Macs. For this repo's setup, the Tailscale ACL lets a GH-hosted runner SSH to MagicDNS names; adjust the workflow's host list if routing differs.

## Related

- Incident: #6474 (TIME_WAIT exhaustion locked 2 Mac runners for 2 months)
- Issue: #6478 (this monitor's tracking issue)
- Runbook: `docs/runners/FLEET.md` "Past incidents" section
