# Aragora Multi-Agent Debate Framework
# Production Dockerfile

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1000 aragora && \
    useradd --uid 1000 --gid aragora --shell /bin/bash --create-home aragora

# Copy requirements first for layer caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY aragora/ ./aragora/
COPY scripts/ ./scripts/
COPY pyproject.toml README.md ./

# Create data directories with proper permissions
RUN mkdir -p /app/data /app/.nomic /app/logs && \
    chown -R aragora:aragora /app

# Switch to non-root user
USER aragora

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Expose port
EXPOSE 8080

# Default command
CMD ["python", "-m", "aragora.server.unified_server", "--host", "0.0.0.0", "--port", "8080"]
