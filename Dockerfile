# Aragora Production Dockerfile
# Multi-stage build for smaller final image
#
# Build:
#   docker build -t aragora .
#
# Run:
#   docker run -e ANTHROPIC_API_KEY=sk-ant-... -p 8080:8080 -p 8765:8765 aragora

# ---------------------------------------------------------------------------
# Build stage: install Python dependencies only
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies (needed for native extensions like asyncpg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only project metadata for dependency resolution
COPY pyproject.toml README.md ./
COPY aragora/__init__.py ./aragora/__init__.py

# Install the package and its optional dependency groups.
# This pulls all transitive deps (asyncpg, redis, prometheus-client, etc.)
# into /usr/local/lib/python3.11/site-packages.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[postgres,redis,monitoring]"

# Remove the stub aragora package from site-packages so it does not
# shadow the full source tree that we COPY in the production stage.
RUN rm -rf /usr/local/lib/python3.11/site-packages/aragora \
           /usr/local/lib/python3.11/site-packages/aragora-*.dist-info \
           /usr/local/lib/python3.11/site-packages/aragora-*.egg-info

# ---------------------------------------------------------------------------
# Production stage: slim runtime image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS production

WORKDIR /app

# Install runtime-only OS packages (curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python dependencies from builder (aragora stub already removed)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy full application source and entrypoint
COPY aragora/ ./aragora/
COPY pyproject.toml README.md ./
COPY deploy/scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create non-root user for security
RUN useradd -m -u 1000 aragora && \
    mkdir -p /app/data /app/logs && \
    chown -R aragora:aragora /app
USER aragora

# Environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    ARAGORA_ENV=production \
    ARAGORA_BIND_HOST=0.0.0.0 \
    ARAGORA_API_PORT=8080 \
    ARAGORA_WS_PORT=8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=5 \
    CMD curl -fsS http://localhost:8080/healthz || exit 1

# Expose ports (HTTP API + WebSocket)
EXPOSE 8080 8765

# Entrypoint runs database migrations (if DATABASE_URL set), then starts server
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "-m", "aragora.server", \
     "--host", "0.0.0.0", \
     "--http-port", "8080", \
     "--ws-port", "8765"]
