# Aragora Server Docker Image
# Multi-stage build for production deployment
#
# Usage:
#   docker build -t aragora:latest -f deploy/Dockerfile .
#   docker run -p 8080:8080 -e ANTHROPIC_API_KEY=... aragora:latest
#
# Build variants:
#   --build-arg VARIANT=full      # All dependencies (default)
#   --build-arg VARIANT=minimal   # Core only
#   --build-arg VARIANT=postgres  # Core + PostgreSQL support

ARG PYTHON_VERSION=3.11
ARG VARIANT=full

# =============================================================================
# Stage 1: Builder - Install dependencies
# =============================================================================
FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Copy package metadata
COPY pyproject.toml README.md ./
COPY aragora/ ./aragora/

# Install based on variant
ARG VARIANT
RUN if [ "$VARIANT" = "minimal" ]; then \
        pip install --no-cache-dir .; \
    elif [ "$VARIANT" = "postgres" ]; then \
        pip install --no-cache-dir ".[postgres,redis]"; \
    else \
        pip install --no-cache-dir ".[persistence,redis,monitoring,observability,postgres,rlm]"; \
    fi

# =============================================================================
# Stage 2: Production - Minimal runtime image
# =============================================================================
FROM python:${PYTHON_VERSION}-slim AS production

# Security: Run as non-root user
RUN groupadd -r aragora && useradd -r -g aragora aragora

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=aragora:aragora aragora/ ./aragora/
COPY --chown=aragora:aragora scripts/ ./scripts/

# Create directories for state
RUN mkdir -p /app/.nomic /app/data && chown -R aragora:aragora /app

# Switch to non-root user
USER aragora

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ARAGORA_NOMIC_DIR=/app/.nomic \
    ARAGORA_LOG_FORMAT=json \
    ARAGORA_LOG_LEVEL=INFO \
    # Metrics and observability
    METRICS_ENABLED=true \
    TRACING_ENABLED=true \
    DECISION_METRICS_ENABLED=true \
    # OpenTelemetry defaults (override via env)
    OTEL_SERVICE_NAME=aragora \
    OTEL_TRACES_EXPORTER=otlp \
    OTEL_METRICS_EXPORTER=prometheus \
    OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317 \
    # Decision router settings
    DECISION_CACHE_ENABLED=true \
    DECISION_CACHE_TTL=3600 \
    DECISION_DEDUP_ENABLED=true

# Expose ports (HTTP API, WebSocket, Prometheus metrics)
EXPOSE 8080 8765 9090

# Health check using the v2 health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/api/v2/health || exit 1

# Labels for container metadata
LABEL org.opencontainers.image.title="Aragora" \
      org.opencontainers.image.description="Control plane for multi-agent vetted decisionmaking across org knowledge and channels" \
      org.opencontainers.image.version="2.1.10" \
      org.opencontainers.image.vendor="Aragora" \
      org.opencontainers.image.source="https://github.com/aragora/aragora"

# Default command: Run unified server
CMD ["python", "-m", "aragora.server.unified_server", "--host", "0.0.0.0", "--port", "8080"]

# =============================================================================
# Stage 3: Development - Includes dev tools
# =============================================================================
FROM production AS development

USER root

# Install dev dependencies
RUN pip install --no-cache-dir pytest pytest-asyncio black ruff mypy

# Copy all code including tests
COPY --chown=aragora:aragora . .

USER aragora

# Override health check for faster dev iteration
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=2 \
    CMD curl -f http://localhost:8080/api/v2/health || exit 1

CMD ["python", "-m", "aragora.server.unified_server", "--host", "0.0.0.0", "--port", "8080"]
