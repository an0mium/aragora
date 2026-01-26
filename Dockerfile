# Aragora Production Dockerfile
# Multi-stage build for smaller final image

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching
COPY pyproject.toml ./
COPY README.md ./

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[postgres,redis,monitoring]"

# Production stage
FROM python:3.11-slim as production

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

# Create non-root user for security
RUN useradd -m -u 1000 aragora && \
    chown -R aragora:aragora /app
USER aragora

# Environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ARAGORA_ENV=production \
    ARAGORA_HOST=0.0.0.0 \
    ARAGORA_API_PORT=8080 \
    ARAGORA_WS_PORT=8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${ARAGORA_API_PORT}/api/health || exit 1

# Expose ports
EXPOSE 8080 8765

# Default command
CMD ["python", "-m", "aragora.server.unified_server", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--ws-port", "8765"]
