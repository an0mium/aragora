#!/bin/bash
# Aragora Docker Entrypoint
# Runs database migrations before starting the server.
#
# Usage: Set as ENTRYPOINT in Dockerfile, or call directly.
# Environment:
#   SKIP_MIGRATIONS=1  - Skip migration step (e.g., for worker containers)
#   DATABASE_URL       - PostgreSQL connection string (required for migrations)

set -e

echo "[entrypoint] Aragora server starting..."

# Run migrations unless explicitly skipped
if [ "${SKIP_MIGRATIONS}" != "1" ]; then
    if [ -n "${DATABASE_URL}" ] || [ -n "${ARAGORA_POSTGRES_DSN}" ]; then
        echo "[entrypoint] Running database migrations..."
        python -m aragora.migrations.runner upgrade 2>&1 || {
            echo "[entrypoint] WARNING: Migration failed. Server will start but may use degraded mode."
            echo "[entrypoint] Check DATABASE_URL and database connectivity."
        }
        echo "[entrypoint] Migrations complete."
    else
        echo "[entrypoint] No DATABASE_URL set, skipping migrations (using SQLite)."
    fi
else
    echo "[entrypoint] SKIP_MIGRATIONS=1, skipping migrations."
fi

# Seed demo data if requested
if [ "${ARAGORA_SEED_DEMO}" = "true" ]; then
    echo "[entrypoint] Seeding demo data..."
    if [ -f "scripts/seed_demo.py" ]; then
        python scripts/seed_demo.py 2>&1 || {
            echo "[entrypoint] WARNING: Demo seeding failed (non-fatal)."
        }
    else
        echo "[entrypoint] seed_demo.py not found, skipping demo seed."
    fi
fi

# Execute the main command (default: start the server)
exec "$@"
