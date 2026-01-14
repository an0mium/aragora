#!/bin/bash
# Watchdog script to restart services if health check fails
# Designed to run as a systemd service

HEALTH_URL="http://localhost:8080/healthz"
MAX_FAILURES=3
FAILURE_COUNT=0
CHECK_INTERVAL=30

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [watchdog] $1"
}

check_health() {
    if curl -sf "$HEALTH_URL" -o /dev/null --connect-timeout 5; then
        return 0
    else
        return 1
    fi
}

restart_services() {
    log "Health check failed $MAX_FAILURES times, restarting services..."

    # Try aragora-staging first (EC2), then aragora (Lightsail)
    if sudo systemctl is-active --quiet aragora-staging 2>/dev/null; then
        log "Restarting aragora-staging..."
        sudo systemctl restart aragora-staging
    elif sudo systemctl is-active --quiet aragora 2>/dev/null; then
        log "Restarting aragora..."
        sudo systemctl restart aragora
    else
        log "No aragora service found to restart"
    fi

    sleep 10

    log "Restarting cloudflared..."
    sudo systemctl restart cloudflared

    sleep 5

    # Verify recovery
    if check_health; then
        log "Services recovered successfully"
    else
        log "Services still unhealthy after restart"
    fi
}

log "Starting health watchdog (checking every ${CHECK_INTERVAL}s, max failures: ${MAX_FAILURES})"

while true; do
    if check_health; then
        if [ $FAILURE_COUNT -gt 0 ]; then
            log "Health check passed, resetting failure count"
        fi
        FAILURE_COUNT=0
    else
        ((FAILURE_COUNT++))
        log "Health check failed ($FAILURE_COUNT/$MAX_FAILURES)"

        if [ $FAILURE_COUNT -ge $MAX_FAILURES ]; then
            restart_services
            FAILURE_COUNT=0
        fi
    fi

    sleep $CHECK_INTERVAL
done
