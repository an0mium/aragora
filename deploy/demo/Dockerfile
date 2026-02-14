FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# Install aragora
COPY pyproject.toml ./
COPY aragora/ aragora/
COPY scripts/ scripts/
COPY deploy/scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh && \
    pip install --no-cache-dir -e ".[server]" 2>/dev/null || pip install --no-cache-dir -e .

# Create data directory
RUN mkdir -p /app/data

EXPOSE 8080 8765

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "-m", "aragora.server", "--offline", "--api-port", "8080", "--ws-port", "8765"]
