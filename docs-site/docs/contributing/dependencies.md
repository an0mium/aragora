---
title: Aragora Dependencies Guide
description: Aragora Dependencies Guide
---

# Aragora Dependencies Guide

This guide helps you choose the right dependencies for your use case.

## Quick Start

```bash
# Minimal install (CLI debates only)
pip install aragora

# Development install (includes testing tools)
pip install aragora[dev]

# Full install (all features)
pip install aragora[all]
```

## Feature-to-Extras Mapping

| Feature | Extras | Size Impact |
|---------|--------|-------------|
| CLI debates | (none) | ~50MB |
| Development/Testing | `dev` | +100MB |
| Monitoring (Prometheus/Sentry) | `monitoring` | +20MB |
| Distributed tracing (OpenTelemetry) | `observability` | +30MB |
| Redis caching | `redis` | +5MB |
| Database (Supabase/PostgreSQL) | `persistence,postgres` | +40MB |
| PDF/DOCX processing | `documents` | +30MB |
| Text-to-speech (basic) | `broadcast` | +50MB |
| Text-to-speech (premium) | `broadcast-premium` | +500MB |
| Web research | `research` | +20MB |
| ML/Embeddings | `ml` | +2GB |

## Installation Profiles

### Profile 1: CLI User
For running debates from the command line:

```bash
pip install aragora
```

**Includes**: Core debate engine, SQLite storage, all agent providers

### Profile 2: API Server
For running the REST/WebSocket server:

```bash
pip install aragora[monitoring,observability]
```

**Includes**: Core + Prometheus metrics + OpenTelemetry tracing

### Profile 3: Production Deployment
For full production environment:

```bash
pip install aragora[monitoring,observability,persistence,postgres,redis]
```

**Includes**: Core + metrics + tracing + Supabase + PostgreSQL + Redis caching

### Profile 4: Development
For contributing to Aragora:

```bash
pip install aragora[dev,monitoring]
```

**Includes**: Core + pytest + mypy + ruff + bandit + metrics

### Profile 5: Research/Evidence
For debates with web research and evidence collection:

```bash
pip install aragora[research,documents]
```

**Includes**: Core + DuckDuckGo search + PDF/DOCX parsing

### Profile 6: Broadcast/Audio
For generating debate audio/video:

```bash
# Basic TTS (free, edge-tts)
pip install aragora[broadcast]

# Premium TTS (ElevenLabs, AWS Polly, Coqui XTTS)
pip install aragora[broadcast-premium]
```

**Note**: `broadcast-premium` requires ~500MB and includes PyTorch for XTTS

### Profile 7: ML/Semantic Search
For semantic similarity and embeddings:

```bash
pip install aragora[ml]
```

**Warning**: This adds ~2GB for sentence-transformers and dependencies

## Dependency Details

### Core Dependencies (always installed)

| Package | Purpose | Security |
|---------|---------|----------|
| aiohttp>=3.13.3 | Async HTTP client | CVE fixes in 3.13.3 |
| websockets>=12.0 | WebSocket support | - |
| pyyaml>=6.0 | Config parsing | - |
| pydantic>=2.0 | Data validation | - |
| bcrypt>=4.0 | Password hashing | - |
| markupsafe>=2.1.0 | XSS prevention | - |
| pyotp>=2.9 | MFA support | - |
| jinja2>=3.1.6 | Templating | CVE-2024-56326 fix |
| urllib3>=2.6.3 | HTTP utilities | CVE fix |

### Optional Dependencies

#### `dev` - Development Tools
```
pytest, pytest-asyncio, pytest-cov, pytest-timeout
black, ruff, bandit, mypy, mutmut
```

#### `monitoring` - Production Metrics
```
prometheus-client - Metrics export
sentry-sdk - Error tracking
```

#### `observability` - Distributed Tracing
```
opentelemetry-api, opentelemetry-sdk
opentelemetry-exporter-otlp
opentelemetry-instrumentation-logging
prometheus-client
```

#### `persistence` - Database Backends
```
supabase - Supabase client
sqlalchemy - ORM support
```

#### `postgres` - PostgreSQL Support
```
asyncpg - Async PostgreSQL driver
```

#### `documents` - Document Processing
```
pypdf>=6.6 - PDF parsing (CVE fixes)
python-docx - DOCX parsing
```

#### `broadcast` - Text-to-Speech
```
edge-tts - Microsoft Edge TTS (free)
pydub - Audio processing
pyttsx3 - Local TTS (macOS/Windows)
```

#### `broadcast-elevenlabs` - ElevenLabs TTS
```
elevenlabs - Premium voice synthesis
```

#### `broadcast-polly` - AWS Polly TTS
```
boto3 - AWS SDK for Polly
```

#### `broadcast-xtts` - Coqui XTTS
```
TTS - Local neural TTS (requires PyTorch)
```

#### `research` - Web Research
```
duckduckgo-search - Web search API
httpx - HTTP client
beautifulsoup4 - HTML parsing
```

#### `ml` - Machine Learning
```
numpy>=2.0 - Numerical computing
scipy>=1.14.0 - Scientific computing
scikit-learn>=1.5.0 - ML algorithms
sentence-transformers>=3.0.0 - Text embeddings
```

## Environment Variables

Dependencies may require these environment variables:

| Variable | Required For |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude agents |
| `OPENAI_API_KEY` | GPT agents |
| `OPENROUTER_API_KEY` | Fallback agents |
| `SUPABASE_URL` | Supabase persistence |
| `SUPABASE_KEY` | Supabase persistence |
| `ELEVENLABS_API_KEY` | ElevenLabs TTS |
| `AWS_ACCESS_KEY_ID` | AWS Polly TTS |
| `AWS_SECRET_ACCESS_KEY` | AWS Polly TTS |

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`, install the required extras:

```bash
# Missing prometheus_client
pip install aragora[monitoring]

# Missing sentence_transformers
pip install aragora[ml]

# Missing pypdf
pip install aragora[documents]
```

### Large Downloads

The `ml` and `broadcast-premium` extras download large models:

- `ml`: ~2GB (sentence-transformers + torch)
- `broadcast-xtts`: ~500MB (Coqui TTS + torch)

Use `--no-cache-dir` to avoid caching:

```bash
pip install --no-cache-dir aragora[ml]
```

### Conflicts

If you have dependency conflicts:

```bash
# Create fresh environment
python -m venv .venv
source .venv/bin/activate

# Install with constraints
pip install aragora[your-extras]
```
