# Aragora Server Docker Image
# Runs the unified server (HTTP API + WebSocket)

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create .nomic directory for state
RUN mkdir -p /app/.nomic

# Expose ports
EXPOSE 8080 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

# Run server
CMD ["python", "-m", "aragora.server", "--host", "0.0.0.0", "--http-port", "8080", "--port", "8765"]
