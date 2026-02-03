# Config Module

Centralized configuration management with validated Pydantic settings and legacy constants.

## Quick Start

```python
# Preferred: Pydantic settings (validated, type-safe)
from aragora.config.settings import get_settings

settings = get_settings()
timeout = settings.database.timeout_seconds
rate_limit = settings.rate_limit.default_limit

# Legacy: Module constants (backward compatibility)
from aragora.config import DEFAULT_ROUNDS, AGENT_TIMEOUT_SECONDS
```

## Key Components

| File | Purpose |
|------|---------|
| `settings.py` | Pydantic settings classes with validation |
| `legacy.py` | Legacy module-level constants |
| `feature_flags.py` | Feature flag definitions and checks |
| `performance_slos.py` | SLO targets by operation type |
| `timeouts.py` | Timeout configurations |
| `redis.py` | Redis connection settings |
| `secrets.py` | Secret management utilities |
| `validator.py` | Configuration validation |
| `env_helpers.py` | Environment variable parsing |

## Architecture

The config system has two layers:

```
                    ┌─────────────────────┐
  New Code ────────►│  settings.py        │  Pydantic-validated
                    │  get_settings()     │  Type-safe, env vars
                    └─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
  Legacy Code ─────►│  legacy.py          │  Module constants
                    │  DEFAULT_ROUNDS=3   │  Backward compatible
                    └─────────────────────┘
```

## Settings Classes

| Class | Purpose | Env Prefix |
|-------|---------|------------|
| `ArgoraSettings` | Root settings container | `ARAGORA_` |
| `AuthSettings` | Authentication (token TTL, rate limits) | `ARAGORA_` |
| `RateLimitSettings` | Rate limiting (default, burst, Redis) | `ARAGORA_` |
| `APILimitSettings` | API pagination limits | `ARAGORA_` |
| `DatabaseSettings` | Database paths and timeouts | `ARAGORA_` |
| `DebateSettings` | Debate defaults (rounds, consensus) | `ARAGORA_` |
| `FeatureSettings` | Feature toggles | `ARAGORA_` |

## Environment Variables

All settings can be overridden via environment variables:

```bash
# Rate limiting
export ARAGORA_RATE_LIMIT=100
export ARAGORA_IP_RATE_LIMIT=200
export ARAGORA_BURST_MULTIPLIER=2.0

# Database
export ARAGORA_DB_PATH=/data/aragora.db
export ARAGORA_DB_TIMEOUT=30

# Authentication
export ARAGORA_TOKEN_TTL=7200
export ARAGORA_SHAREABLE_LINK_TTL=86400

# Redis (for distributed rate limiting)
export ARAGORA_REDIS_URL=redis://localhost:6379/0
```

## Performance SLOs

The `performance_slos.py` module defines operation-specific latency targets:

```python
from aragora.config.performance_slos import DebateSLO, KnowledgeMoundSLO

# Debate latency targets
DebateSLO.p50  # 50ms
DebateSLO.p95  # 200ms
DebateSLO.p99  # 500ms

# Knowledge Mound query targets
KnowledgeMoundSLO.query_p99  # 100ms
```

## Feature Flags

```python
from aragora.config.feature_flags import FeatureFlags

flags = FeatureFlags()
if flags.is_enabled("supermemory"):
    # Use supermemory integration
    pass

# Check with default
enabled = flags.is_enabled("experimental_feature", default=False)
```

## Legacy Constants

Common legacy constants (from `legacy.py`):

| Constant | Default | Description |
|----------|---------|-------------|
| `DEFAULT_ROUNDS` | 3 | Default debate rounds |
| `DEFAULT_CONSENSUS` | "majority" | Default consensus method |
| `DEFAULT_AGENTS` | list | Default agent list |
| `AGENT_TIMEOUT_SECONDS` | 60 | Agent response timeout |
| `CACHE_TTL_*` | varies | Cache TTL for various operations |
| `DB_*_PATH` | varies | Database file paths |

## Adding New Settings

1. Add to appropriate settings class in `settings.py`:

```python
class MyFeatureSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ARAGORA_")

    my_setting: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="My feature setting",
        alias="ARAGORA_MY_SETTING"
    )
```

2. Include in `ArgoraSettings`:

```python
class ArgoraSettings(BaseSettings):
    my_feature: MyFeatureSettings = Field(default_factory=MyFeatureSettings)
```

3. Add to `legacy.py` for backward compatibility if needed:

```python
MY_SETTING = 100  # Legacy constant
```

## Validation

Configuration is validated at startup:

```python
from aragora.config.validator import validate_config

# Raises ConfigurationError if invalid
validate_config()
```

## Related

- `aragora/observability/slo.py` - Runtime SLO monitoring
- `aragora/resilience/` - Circuit breakers, retries
- `docs/ENVIRONMENT.md` - Full environment variable reference
