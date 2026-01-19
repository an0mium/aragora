# Connector Troubleshooting Guide

This guide helps diagnose and resolve common issues with Aragora's evidence connectors.

## Quick Diagnostics

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| "401 Unauthorized" | Missing or invalid API key | Check environment variables |
| "429 Too Many Requests" | Rate limit exceeded | Wait and retry, or reduce request frequency |
| "Connection refused" | Service unreachable | Check network/firewall settings |
| "Timeout" | Slow response or network issues | Increase timeout, check connectivity |
| "JSON decode error" | Invalid response format | API may have changed or returned error HTML |

## Exception Reference

All connectors use a standardized exception hierarchy:

```
ConnectorError (base)
├── ConnectorAuthError       - Authentication failures (NOT retryable)
├── ConnectorRateLimitError  - Rate limit exceeded (retryable after delay)
├── ConnectorTimeoutError    - Request timeout (retryable)
├── ConnectorNetworkError    - Connection issues (retryable)
├── ConnectorAPIError        - API errors (5xx retryable, 4xx not)
├── ConnectorValidationError - Invalid input (NOT retryable)
├── ConnectorNotFoundError   - Resource not found (NOT retryable)
├── ConnectorQuotaError      - Quota exhausted (NOT retryable)
└── ConnectorParseError      - Response parsing failed (NOT retryable)
```

### Checking Retryability

```python
from aragora.connectors.exceptions import is_retryable_error, get_retry_delay

try:
    results = await connector.search("query")
except ConnectorError as e:
    if is_retryable_error(e):
        delay = get_retry_delay(e, default=5.0)
        print(f"Retrying in {delay}s...")
        await asyncio.sleep(delay)
        # Retry the operation
    else:
        print(f"Non-retryable error: {e}")
```

## Common Issues by Error Type

### ConnectorAuthError (401/403)

**Symptoms:**
- "401 Unauthorized" or "403 Forbidden"
- "Invalid API key"
- "Authentication required"

**Causes:**
1. API key not set in environment
2. API key expired or revoked
3. Insufficient permissions for operation

**Solutions:**

```bash
# Check if API key is set
echo $GITHUB_TOKEN
echo $NEWSAPI_KEY
echo $TWITTER_BEARER_TOKEN

# Set missing keys
export GITHUB_TOKEN="ghp_..."
export NEWSAPI_KEY="..."
```

**Per-connector requirements:**

| Connector | Required Variable | Notes |
|-----------|------------------|-------|
| GitHub | `GITHUB_TOKEN` | Personal access token with `repo` scope |
| NewsAPI | `NEWSAPI_KEY` | Get from newsapi.org |
| Twitter | `TWITTER_BEARER_TOKEN` | From Twitter Developer Portal |
| Reddit | `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` | OAuth app credentials |
| arXiv | None | No authentication required |
| Wikipedia | None | No authentication required |
| HackerNews | None | No authentication required |

### ConnectorRateLimitError (429)

**Symptoms:**
- "429 Too Many Requests"
- "Rate limit exceeded"
- Operations slow down or fail in bursts

**Causes:**
1. Too many requests in short time window
2. Concurrent operations from multiple instances
3. Provider-specific rate limits

**Solutions:**

```python
# The connector handles retries automatically, but you can configure:
from aragora.connectors.github import GitHubConnector

connector = GitHubConnector(
    max_retries=5,          # Increase retry attempts
    base_delay=2.0,         # Start with longer delay
    max_delay=60.0,         # Cap maximum wait time
)

# Or implement your own rate limiting
import asyncio
from contextlib import asynccontextmanager

class RateLimiter:
    def __init__(self, requests_per_minute: int = 30):
        self.interval = 60.0 / requests_per_minute
        self.last_request = 0

    @asynccontextmanager
    async def limit(self):
        now = time.time()
        elapsed = now - self.last_request
        if elapsed < self.interval:
            await asyncio.sleep(self.interval - elapsed)
        self.last_request = time.time()
        yield
```

**Provider rate limits:**

| Provider | Limit | Notes |
|----------|-------|-------|
| GitHub | 5000/hour (authenticated) | 60/hour unauthenticated |
| NewsAPI | 100/day (free), 500/day (paid) | Resets at midnight UTC |
| Twitter | Varies by endpoint | Check developer portal |
| Reddit | 60/minute | Per OAuth client |
| arXiv | 3/second | Be gentle, it's free |
| HackerNews | ~100/minute | Unofficial limit |

### ConnectorTimeoutError

**Symptoms:**
- "Request timed out"
- Operations hang then fail
- Intermittent failures

**Causes:**
1. Network latency
2. Slow API response
3. Large response payloads
4. DNS resolution delays

**Solutions:**

```python
# Increase timeout for slow APIs
import httpx

async with httpx.AsyncClient(timeout=60.0) as client:
    # Custom timeout per request
    response = await client.get(url, timeout=120.0)

# Configure connector timeout
connector = WebConnector(
    request_timeout=30.0,  # Seconds
)
```

### ConnectorNetworkError

**Symptoms:**
- "Connection refused"
- "Network unreachable"
- "DNS lookup failed"
- "SSL certificate error"

**Causes:**
1. Firewall blocking outbound connections
2. DNS resolution issues
3. Provider service outage
4. SSL/TLS certificate problems

**Solutions:**

```bash
# Test connectivity
curl -v https://api.github.com

# Check DNS
dig api.github.com

# Test SSL
openssl s_client -connect api.github.com:443

# Check firewall (macOS)
sudo pfctl -sr
```

For SSL issues:

```python
# If you must disable SSL verification (NOT recommended for production)
import ssl
import certifi

# Use system certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Or disable verification (development only!)
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE
```

### ConnectorAPIError

**Symptoms:**
- "HTTP 500" or other 5xx errors
- "API error" messages
- Unexpected response format

**Causes:**
1. Provider service issues
2. Invalid request parameters
3. API version mismatch
4. Provider maintenance

**Solutions:**

```python
# Check provider status
# - GitHub: https://www.githubstatus.com/
# - Twitter: https://api.twitterstat.us/
# - Reddit: https://www.redditstatus.com/

# Enable debug logging
import logging
logging.getLogger("aragora.connectors").setLevel(logging.DEBUG)

# Inspect the actual response
from aragora.connectors.exceptions import ConnectorAPIError

try:
    await connector.search("query")
except ConnectorAPIError as e:
    print(f"Status: {e.status_code}")
    print(f"Message: {e}")
```

### ConnectorParseError

**Symptoms:**
- "JSON decode error"
- "Failed to parse response"
- "Unexpected response format"

**Causes:**
1. API returned HTML error page instead of JSON
2. Response encoding issues
3. Malformed API response
4. API schema changed

**Solutions:**

```python
# Log raw response for debugging
import httpx

async with httpx.AsyncClient() as client:
    response = await client.get(url)
    print(f"Status: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type')}")
    print(f"Body preview: {response.text[:500]}")
```

### ConnectorValidationError

**Symptoms:**
- "Invalid parameter"
- "Missing required field"
- "Value out of range"

**Causes:**
1. Query string too long/short
2. Invalid search parameters
3. Unsupported filter options

**Solutions:**

```python
# Check parameter constraints
from aragora.connectors.github import GitHubConnector

# Valid search
results = await connector.search(
    query="python testing",
    limit=10,  # Must be 1-100
)

# Invalid - will raise ValidationError
results = await connector.search(
    query="",  # Empty query not allowed
    limit=1000,  # Exceeds max
)
```

### ConnectorNotFoundError

**Symptoms:**
- "404 Not Found"
- "Resource not found"
- Empty results unexpectedly

**Causes:**
1. Resource deleted or moved
2. Incorrect resource ID
3. Permission to view resource revoked

**Solutions:**

```python
from aragora.connectors.exceptions import ConnectorNotFoundError

try:
    evidence = await connector.fetch("nonexistent-id")
except ConnectorNotFoundError as e:
    print(f"Resource not found: {e.resource_id}")
    # Handle gracefully - maybe search for alternatives
```

### ConnectorQuotaError

**Symptoms:**
- "Quota exceeded"
- "Daily limit reached"
- "Monthly usage exhausted"

**Causes:**
1. Free tier limits exceeded
2. Billing issues with paid tier
3. Shared API key across many users

**Solutions:**

```python
from aragora.connectors.exceptions import ConnectorQuotaError

try:
    results = await connector.search("query")
except ConnectorQuotaError as e:
    if e.quota_reset:
        print(f"Quota resets in {e.quota_reset} seconds")
    # Switch to fallback connector
    results = await fallback_connector.search("query")
```

## Debugging Techniques

### Enable Debug Logging

```python
import logging

# Log all connector activity
logging.getLogger("aragora.connectors").setLevel(logging.DEBUG)

# Log HTTP requests
logging.getLogger("httpx").setLevel(logging.DEBUG)

# Log to file
handler = logging.FileHandler("connector_debug.log")
handler.setLevel(logging.DEBUG)
logging.getLogger("aragora.connectors").addHandler(handler)
```

### Inspect Cache State

```python
# Check connector cache
stats = connector._cache_stats()
print(f"Cache entries: {stats['total_entries']}")
print(f"Active: {stats['active_entries']}")
print(f"Expired: {stats['expired_entries']}")

# Clear cache if needed
connector._cache.clear()

# Or clear only expired
cleared = connector._cache_clear_expired()
print(f"Cleared {cleared} expired entries")
```

### Test Connector Manually

```python
import asyncio
from aragora.connectors.github import GitHubConnector

async def test_connector():
    connector = GitHubConnector()

    # Test search
    print("Testing search...")
    results = await connector.search("python asyncio", limit=3)
    print(f"Found {len(results)} results")

    # Test fetch if you have an ID
    if results:
        print(f"\nTesting fetch for {results[0].id}...")
        evidence = await connector.fetch(results[0].id)
        if evidence:
            print(f"Fetched: {evidence.title}")

asyncio.run(test_connector())
```

### Using connector_error_handler

```python
from aragora.connectors.exceptions import connector_error_handler

# Wrap any code to convert exceptions
async with connector_error_handler("my_connector"):
    response = await client.get(url)
    data = response.json()  # Errors converted to ConnectorError
```

## Connector-Specific Tips

### GitHub Connector

```python
# Use search qualifiers for better results
results = await github.search(
    "language:python stars:>100",
    limit=20,
)

# Search specific repo
results = await github.search(
    "repo:anthropics/claude-code exception handling",
)
```

### arXiv Connector

```python
# arXiv is rate-sensitive - be patient
arxiv = ArxivConnector(
    base_delay=3.0,  # Longer delays between requests
)

# Use category prefixes
results = await arxiv.search("cat:cs.AI neural networks")
```

### NewsAPI Connector

```python
# Free tier is limited - cache aggressively
newsapi = NewsAPIConnector(
    cache_ttl_seconds=7200,  # 2 hour cache
)

# Use date filters to reduce result size
results = await newsapi.search(
    "AI safety",
    from_date="2024-01-01",
    to_date="2024-01-31",
)
```

### Wikipedia Connector

```python
# Wikipedia is generally reliable - increase cache
wiki = WikipediaConnector(
    cache_ttl_seconds=86400,  # 24 hour cache
)

# Use exact article titles when possible
evidence = await wiki.fetch("Artificial_intelligence")
```

## Error Recovery Patterns

### Automatic Retry with Backoff

```python
async def resilient_search(connector, query, max_attempts=3):
    """Search with automatic retry and exponential backoff."""
    for attempt in range(max_attempts):
        try:
            return await connector.search(query)
        except ConnectorError as e:
            if not e.is_retryable or attempt == max_attempts - 1:
                raise
            delay = get_retry_delay(e, default=2 ** attempt)
            await asyncio.sleep(delay)
    return []
```

### Fallback Connectors

```python
async def search_with_fallback(query, connectors):
    """Try connectors in order until one succeeds."""
    for connector in connectors:
        try:
            results = await connector.search(query)
            if results:
                return results
        except ConnectorError as e:
            logger.warning(f"{connector.name} failed: {e}")
            continue
    return []

# Usage
connectors = [
    GitHubConnector(),
    ArxivConnector(),
    WikipediaConnector(),
]
results = await search_with_fallback("quantum computing", connectors)
```

### Circuit Breaker Pattern

```python
from aragora.resilience import CircuitBreaker

breaker = CircuitBreaker(
    failure_threshold=5,  # Open after 5 failures
    recovery_timeout=60,  # Try again after 60s
)

async def protected_search(connector, query):
    if not breaker.can_proceed():
        logger.warning("Circuit open - using cached data")
        return get_cached_results(query)

    try:
        results = await connector.search(query)
        breaker.record_success()
        return results
    except ConnectorError as e:
        breaker.record_failure()
        raise
```

## Health Checks

### Check All Connectors

```python
async def health_check():
    """Check health of all configured connectors."""
    connectors = [
        ("github", GitHubConnector()),
        ("arxiv", ArxivConnector()),
        ("wikipedia", WikipediaConnector()),
    ]

    results = {}
    for name, connector in connectors:
        try:
            # Simple search to verify connectivity
            await connector.search("test", limit=1)
            results[name] = {"status": "healthy"}
        except ConnectorError as e:
            results[name] = {
                "status": "unhealthy",
                "error": str(e),
                "retryable": e.is_retryable,
            }

    return results
```

## Environment Setup Checklist

1. **Required API keys configured:**
   ```bash
   env | grep -E "(GITHUB|NEWSAPI|TWITTER|REDDIT)"
   ```

2. **Network connectivity:**
   ```bash
   curl -s -o /dev/null -w "%{http_code}" https://api.github.com
   curl -s -o /dev/null -w "%{http_code}" https://export.arxiv.org
   ```

3. **Python dependencies:**
   ```bash
   pip show httpx aiohttp
   ```

4. **Test imports:**
   ```python
   from aragora.connectors import (
       GitHubConnector,
       ArxivConnector,
       WikipediaConnector,
   )
   ```

## Getting Help

If issues persist:

1. Check connector logs with DEBUG level
2. Verify API key permissions on provider's dashboard
3. Test the API directly with curl
4. Check provider status pages
5. Report issues at https://github.com/anthropics/aragora/issues

## See Also

- [Evidence API Guide](EVIDENCE_API_GUIDE.md) - Using the evidence system
- [API Usage](API_USAGE.md) - Server API reference
- [Observability](OBSERVABILITY.md) - Monitoring and logging
