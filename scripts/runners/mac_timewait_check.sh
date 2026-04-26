#!/bin/bash
# TIME_WAIT monitor for Mac runners.
#
# Counts TCP connections in TIME_WAIT state and emits a single line of
# structured log output. Alerts via a flag file when the count exceeds
# the threshold so the weekly GH Actions check can detect it.
#
# Intended to run under launchd daily. Writes to a persistent log so a
# human SSHing in later sees the forensic trail, even if the alert
# path (which depends on outbound TCP) itself is what's broken.
#
# Background: issue #6474 traced two Mac runners offline for 2 months
# due to TIME_WAIT exhaustion (31K+ entries filled the ephemeral port
# range). This monitor catches that condition early.

set -euo pipefail

THRESHOLD="${THRESHOLD:-5000}"
LOG_FILE="${LOG_FILE:-$HOME/Library/Logs/aragora-runner-health.log}"
ALERT_FILE="${ALERT_FILE:-$HOME/Library/Logs/aragora-runner-health.alert}"

# Count TIME_WAIT entries. Use -p tcp so we don't include unix sockets.
TIMEWAIT_COUNT=$(netstat -an -p tcp 2>/dev/null | awk '$NF == "TIME_WAIT"' | wc -l | tr -d ' ')

# Also count totals for context.
TCP_TOTAL=$(netstat -an -p tcp 2>/dev/null | wc -l | tr -d ' ')
ESTABLISHED=$(netstat -an -p tcp 2>/dev/null | awk '$NF == "ESTABLISHED"' | wc -l | tr -d ' ')
UPTIME=$(uptime | awk -F'up ' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
HOST=$(hostname)
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Single structured log line (parse-friendly).
LINE="ts=$TS host=$HOST uptime=$UPTIME tcp_total=$TCP_TOTAL timewait=$TIMEWAIT_COUNT established=$ESTABLISHED threshold=$THRESHOLD"

# Append to log (create if missing). This runs as a LaunchAgent in user
# context, so avoid sudo: a password prompt under launchd would drop the
# forensic trail this script exists to preserve.
mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true
printf '%s\n' "$LINE" >> "$LOG_FILE" 2>/dev/null || echo "$LINE" >&2

# Write alert flag if we're over threshold.
if [ "$TIMEWAIT_COUNT" -gt "$THRESHOLD" ]; then
    cat > "$ALERT_FILE" <<ALERT || echo "ALERT: $LINE" >&2
host=$HOST
ts=$TS
timewait=$TIMEWAIT_COUNT
threshold=$THRESHOLD
uptime=$UPTIME
message=TIME_WAIT count exceeded threshold; reboot likely needed (see docs/runners/FLEET.md).
ALERT
    echo "ALERT: $LINE" >&2
else
    # Clear any stale alert file if we've recovered.
    rm -f "$ALERT_FILE" 2>/dev/null || true
fi

# Emit the line to stdout too so interactive runs show it.
echo "$LINE"
