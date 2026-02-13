# Aragora Production Dockerfile
# Multi-stage build for smaller final image

# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata and source for install
COPY pyproject.toml README.md ./
COPY aragora/__init__.py ./aragora/__init__.py

# Install dependencies (non-editable, only needs __init__.py for metadata)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[postgres,redis,monitoring]"

# Production stage
FROM python:3.11-slim AS production

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY aragora/ ./aragora/
COPY pyproject.toml README.md ./
COPY deploy/scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create non-root user for security
RUN useradd -m -u 1000 aragora && \
    mkdir -p /app/data && \
    chown -R aragora:aragora /app
USER aragora

# Environment defaults
# ARAGORA_BIND_HOST is the env var read by the server for bind address
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ARAGORA_ENV=production \
    ARAGORA_BIND_HOST=0.0.0.0 \
    ARAGORA_API_PORT=8080 \
    ARAGORA_WS_PORT=8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${ARAGORA_API_PORT}/healthz || exit 1

# Expose ports
EXPOSE 8080 8765

# Entrypoint runs migrations, then starts the server
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "-m", "aragora.server", \
     "--host", "0.0.0.0", \
     "--http-port", "8080", \
     "--ws-port", "8765"]
