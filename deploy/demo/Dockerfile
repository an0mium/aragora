FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libffi-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Install aragora
COPY pyproject.toml ./
COPY aragora/ aragora/
COPY scripts/ scripts/
COPY deploy/scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh && \
    pip install --no-cache-dir -e ".[server]" 2>/dev/null || pip install --no-cache-dir -e .

# Create data directory and non-root user
RUN groupadd -r aragora && useradd -r -g aragora -d /app -s /sbin/nologin aragora && \
    mkdir -p /app/data && \
    chown -R aragora:aragora /app

# Switch to non-root user
USER aragora

EXPOSE 8080 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "-m", "aragora.server", "--offline", "--api-port", "8080", "--ws-port", "8765"]
