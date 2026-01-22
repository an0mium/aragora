---
title: Installation
description: Install and configure Aragora
sidebar_position: 3
---

# Installation

This guide covers all installation methods for Aragora.

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Node.js | 18.x | 20.x |
| Python | 3.10 | 3.11+ |
| RAM | 2GB | 4GB+ |
| Storage | 1GB | 5GB+ |

## Installation Methods

### Docker (Recommended)

The fastest way to get started:

```bash
# Pull and run
docker run -d \
  -p 8080:8080 \
  -e ANTHROPIC_API_KEY=your_key \
  ghcr.io/aragora/aragora:latest
```

With Docker Compose:

```yaml
# docker-compose.yml
version: '3.8'
services:
  aragora:
    image: ghcr.io/aragora/aragora:latest
    ports:
      - "8080:8080"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - aragora_data:/data
volumes:
  aragora_data:
```

### From Source

```bash
# Clone the repository
git clone https://github.com/aragora/aragora.git
cd aragora

# Install Python dependencies
pip install -e .

# Install Node dependencies (for frontend)
cd aragora/live
npm install

# Start the server
python -m aragora.server.unified_server
```

### Python Package

```bash
pip install aragora
```

### Kubernetes

See the [Kubernetes deployment guide](/docs/deployment/kubernetes) for Helm charts and manifests.

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# Required: At least one AI provider
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...

# Optional: Database (defaults to SQLite)
DATABASE_URL=postgresql://user:pass@host:5432/aragora

# Optional: Redis (for caching/queues)
REDIS_URL=redis://localhost:6379

# Optional: Security
ARAGORA_API_TOKEN=your_secure_token
ARAGORA_ALLOWED_ORIGINS=https://your-domain.com
```

### Configuration File

For advanced configuration, create `aragora.yaml`:

```yaml
server:
  host: 0.0.0.0
  port: 8080
  workers: 4

agents:
  defaults:
    temperature: 0.7
    max_tokens: 4096

  pool:
    - name: claude
      provider: anthropic
      model: claude-3-5-sonnet-latest
    - name: gpt4
      provider: openai
      model: gpt-4o

debate:
  default_rounds: 3
  consensus_threshold: 0.75

memory:
  backend: redis  # or sqlite
  ttl:
    fast: 60
    medium: 3600
    slow: 86400
```

## Verify Installation

```bash
# Check health endpoint
curl http://localhost:8080/health

# Expected response
{"status": "healthy", "version": "2.1.0"}
```

## Next Steps

- [First Debate Tutorial](/docs/getting-started/first-debate)
- [SDK Quickstart](/docs/guides/sdk)
- [API Reference](/docs/api-reference)
