FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# Install aragora
COPY pyproject.toml setup.cfg ./
COPY aragora/ aragora/
COPY scripts/ scripts/
RUN pip install --no-cache-dir -e ".[server]" 2>/dev/null || pip install --no-cache-dir -e .

# Create data directory
RUN mkdir -p /app/data

EXPOSE 8080 8765

CMD ["python", "-m", "aragora.server", "--offline", "--api-port", "8080", "--ws-port", "8765"]
