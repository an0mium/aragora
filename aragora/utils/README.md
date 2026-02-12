# Utils Module

Shared utility functions and patterns used throughout Aragora.

## Quick Start

```python
# Datetime utilities
from aragora.utils import utc_now, to_iso_timestamp, parse_timestamp

now = utc_now()
iso_str = to_iso_timestamp(now)
parsed = parse_timestamp("2024-01-15T10:30:00Z")

# Safe path handling
from aragora.utils import safe_path, is_safe_path

safe = safe_path("/data", user_input)  # Prevents path traversal

# Caching with TTL
from aragora.utils.cache import lru_cache_with_ttl

@lru_cache_with_ttl(ttl_seconds=300, maxsize=100)
def expensive_query(key: str) -> dict:
    ...
```

## Key Components

| File | Purpose |
|------|---------|
| `datetime_helpers.py` | Timezone-aware datetime utilities |
| `json_helpers.py` | Safe JSON parsing |
| `paths.py` | Path traversal protection |
| `sql_helpers.py` | SQL escaping utilities |
| `cache.py` | LRU cache with TTL expiry |
| `redis_cache.py` | Distributed Redis caching |
| `async_utils.py` | Async/await helpers |
| `timeouts.py` | Timeout decorators |
| `optional_imports.py` | Lazy import handling |
| `logging_utils.py` | Structured logging helpers |
| `error_sanitizer.py` | Error message sanitization |
| `token_helpers.py` | Token generation/validation |
| `subprocess_runner.py` | Safe subprocess execution |
| `cache_registry.py` | Global cache management |

## Architecture

```
utils/
├── datetime_helpers.py    # UTC now, ISO timestamps, parsing
├── json_helpers.py        # safe_json_loads (handles NaN, Infinity)
├── paths.py               # Path traversal prevention
├── sql_helpers.py         # LIKE pattern escaping
├── cache.py               # LRU + TTL caching
├── redis_cache.py         # Distributed cache layer
├── async_utils.py         # run_sync, gather_with_concurrency
├── timeouts.py            # @timeout decorator
├── optional_imports.py    # try_import, LazyImport
├── logging_utils.py       # Structured log formatting
├── error_sanitizer.py     # Remove sensitive data from errors
├── token_helpers.py       # Secure token generation
├── subprocess_runner.py   # Safe shell execution
└── cache_registry.py      # Central cache management
```

## Usage Examples

### Datetime Helpers

```python
from aragora.utils import (
    utc_now,
    to_iso_timestamp,
    from_iso_timestamp,
    ensure_timezone_aware,
    timestamp_ms,
)

# Current UTC time
now = utc_now()

# To/from ISO format
iso = to_iso_timestamp(now)  # "2024-01-15T10:30:00.000000Z"
dt = from_iso_timestamp(iso)

# Unix timestamps
ms = timestamp_ms()  # Milliseconds since epoch

# Ensure timezone awareness
aware = ensure_timezone_aware(naive_dt)
```

### Path Security

```python
from aragora.utils import safe_path, is_safe_path, PathTraversalError

# Validate and resolve path
try:
    resolved = safe_path("/data/uploads", user_filename)
except PathTraversalError:
    # User tried "../../../etc/passwd" or similar
    pass

# Check without exception
if is_safe_path("/data", user_path):
    # Path is safe to use
    pass
```

### Caching

```python
from aragora.utils.cache import lru_cache_with_ttl, cached_property_ttl

# Method caching with TTL
class MyService:
    @lru_cache_with_ttl(ttl_seconds=300, maxsize=100)
    def get_data(self, key: str) -> dict:
        # Cached for 5 minutes
        return expensive_query(key)

    @cached_property_ttl(ttl_seconds=600)
    def config(self) -> dict:
        # Property cached for 10 minutes
        return load_config()
```

### Async Utilities

```python
from aragora.utils.async_utils import run_sync, gather_with_concurrency

# Run async in sync context
result = run_sync(async_function())

# Limit concurrent tasks
results = await gather_with_concurrency(
    10,  # Max 10 concurrent
    [fetch_url(url) for url in urls]
)
```

### Optional Imports

```python
from aragora.utils import try_import, LazyImport

# Try importing optional dependency
numpy = try_import("numpy")
if numpy:
    arr = numpy.array([1, 2, 3])

# Lazy import (deferred until first use)
heavy_module = LazyImport("heavy_dependency")
```

### Error Sanitization

```python
from aragora.utils.error_sanitizer import sanitize_error_message

# Remove sensitive data from error messages
safe_msg = sanitize_error_message(exception)
# Removes API keys, passwords, tokens from stack traces
```

## Re-exported Functions

The `__init__.py` re-exports commonly used functions:

```python
from aragora.utils import (
    # Datetime
    utc_now, to_iso_timestamp, from_iso_timestamp,
    ensure_timezone_aware, format_timestamp, parse_timestamp,
    timestamp_ms, timestamp_s,
    # JSON
    safe_json_loads,
    # Imports
    try_import, try_import_class, LazyImport,
    # Paths
    safe_path, validate_path_component, is_safe_path, PathTraversalError,
    # SQL
    escape_like_pattern,
)
```

## Integration Points

| Module | Integration |
|--------|-------------|
| `aragora.server` | Request handling, caching |
| `aragora.storage` | Database operations, SQL escaping |
| `aragora.security` | Token handling, sanitization |
| `aragora.observability` | Logging, metrics |

## Related

- `aragora/config/` - Configuration settings
- `aragora/resilience/` - Circuit breakers, retries
- `aragora/server/handlers/base.py` - Handler-specific utilities
