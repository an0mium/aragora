# Plugin Development Guide

This guide explains how to build, test, and publish plugins (skills) for Aragora.

## Overview

Aragora's plugin system is built on the **Skills Architecture**, allowing you to extend the platform with custom capabilities. Skills can:

- Fetch data from external APIs
- Execute code in sandboxed environments
- Query and store knowledge in the Knowledge Mound
- Participate in debates with custom evidence collection
- Integrate with third-party services

## Quick Start

### 1. Create a Basic Skill

```python
from aragora.skills import Skill, SkillManifest, SkillResult, SkillContext, SkillCapability

class WeatherSkill(Skill):
    """A simple skill that fetches weather data."""

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="weather",
            version="1.0.0",
            description="Get current weather for a location",
            author="your-name",
            capabilities=[SkillCapability.EXTERNAL_API],
            input_schema={
                "location": {
                    "type": "string",
                    "required": True,
                    "description": "City name or coordinates",
                },
            },
            required_permissions=["skills:weather"],
            tags=["weather", "external-api"],
        )

    async def execute(
        self,
        input_data: dict,
        context: SkillContext,
    ) -> SkillResult:
        location = input_data["location"]

        # Your implementation here
        weather_data = await self._fetch_weather(location)

        return SkillResult.create_success(weather_data)

    async def _fetch_weather(self, location: str) -> dict:
        # Implementation details
        pass
```

### 2. Register Your Skill

```python
from aragora.skills import get_skill_registry

registry = get_skill_registry()
registry.register(WeatherSkill())
```

### 3. Invoke Your Skill

```python
from aragora.skills import SkillContext

context = SkillContext(
    user_id="user-123",
    permissions=["skills:weather"],
)

result = await registry.invoke(
    "weather",
    {"location": "San Francisco"},
    context,
)

if result.success:
    print(result.data)
else:
    print(f"Error: {result.error_message}")
```

## Skill Manifest Reference

The `SkillManifest` declares your skill's metadata and requirements:

```python
@dataclass
class SkillManifest:
    # Required fields
    name: str              # Unique identifier (lowercase, no spaces)
    version: str           # Semantic version (e.g., "1.0.0")
    capabilities: list[SkillCapability]  # What your skill can do
    input_schema: dict     # JSON Schema for input validation

    # Optional metadata
    description: str       # Human-readable description
    author: str            # Author name or organization
    tags: list[str]        # Searchable tags

    # Requirements
    required_permissions: list[str]  # RBAC permissions needed
    required_env_vars: list[str]     # Environment variables
    required_packages: list[str]     # Python packages

    # Execution constraints
    max_execution_time_seconds: float = 60.0
    max_retries: int = 3
    rate_limit_per_minute: int | None = None

    # Debate integration
    debate_compatible: bool = True
    requires_debate_context: bool = False
```

## Capabilities

Skills declare capabilities to enable RBAC enforcement:

| Capability | Description | Permission Required |
|------------|-------------|---------------------|
| `READ_LOCAL` | Read local files | `files:read` |
| `WRITE_LOCAL` | Write local files | `files:write` |
| `READ_DATABASE` | Query databases | `database:read` |
| `WRITE_DATABASE` | Modify databases | `database:write` |
| `EXTERNAL_API` | Call external APIs | `external:api` |
| `WEB_SEARCH` | Search the web | `web:search` |
| `WEB_FETCH` | Fetch web pages | `web:fetch` |
| `CODE_EXECUTION` | Execute code | `code:execute` |
| `SHELL_EXECUTION` | Run shell commands | `shell:execute` |
| `LLM_INFERENCE` | Call LLM APIs | `llm:inference` |
| `EMBEDDING` | Generate embeddings | `embedding:generate` |
| `DEBATE_CONTEXT` | Access debate context | `debate:read` |
| `EVIDENCE_COLLECTION` | Collect evidence | `evidence:collect` |
| `KNOWLEDGE_QUERY` | Query knowledge mound | `knowledge:read` |

## Input Schema

Define your skill's expected inputs using JSON Schema:

```python
input_schema={
    "query": {
        "type": "string",
        "required": True,
        "description": "The search query",
        "minLength": 1,
        "maxLength": 500,
    },
    "limit": {
        "type": "number",
        "required": False,
        "default": 10,
        "minimum": 1,
        "maximum": 100,
    },
    "filters": {
        "type": "object",
        "required": False,
        "properties": {
            "date_from": {"type": "string", "format": "date"},
            "date_to": {"type": "string", "format": "date"},
        },
    },
}
```

## Execution Context

The `SkillContext` provides runtime information:

```python
@dataclass
class SkillContext:
    # Identity
    user_id: str | None
    tenant_id: str | None
    session_id: str | None

    # Permissions (for RBAC checks)
    permissions: list[str]

    # Debate context (when invoked during a debate)
    debate_id: str | None
    debate_context: dict | None
    agent_name: str | None

    # Environment
    environment: str  # "development", "staging", "production"
    config: dict      # Skill-specific configuration

    # Previous results (for skill chaining)
    previous_results: dict[str, SkillResult]
```

## Result Handling

Always return a `SkillResult`:

```python
# Success with data
return SkillResult.create_success(
    data={"temperature": 72, "conditions": "sunny"},
    source="weather-api",
    cache_ttl_seconds=300,
)

# Failure with details
return SkillResult.create_failure(
    error_message="API rate limit exceeded",
    error_code="rate_limited",
    status=SkillStatus.RATE_LIMITED,
)

# Timeout
return SkillResult.create_timeout(timeout_seconds=30)

# Permission denied
return SkillResult.create_permission_denied(permission="external:api")
```

## Debate Integration

Skills can participate in debates for evidence collection:

```python
class ResearchSkill(Skill):
    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="research",
            version="1.0.0",
            capabilities=[
                SkillCapability.WEB_SEARCH,
                SkillCapability.EVIDENCE_COLLECTION,
            ],
            debate_compatible=True,
            requires_debate_context=True,  # Needs active debate
            input_schema={"topic": {"type": "string", "required": True}},
        )

    async def execute(self, input_data: dict, context: SkillContext) -> SkillResult:
        if not context.debate_id:
            return SkillResult.create_failure("Requires active debate context")

        # Access debate context
        current_arguments = context.debate_context.get("arguments", [])

        # Collect evidence
        evidence = await self._research(input_data["topic"])

        return SkillResult.create_success({
            "evidence": evidence,
            "debate_id": context.debate_id,
            "relevant_to": [arg["id"] for arg in current_arguments[:3]],
        })
```

## Testing Your Skill

### Unit Tests

```python
import pytest
from aragora.skills import SkillContext, SkillStatus

@pytest.fixture
def weather_skill():
    return WeatherSkill()

@pytest.fixture
def context():
    return SkillContext(
        user_id="test-user",
        permissions=["skills:weather"],
    )

@pytest.mark.asyncio
async def test_weather_skill_success(weather_skill, context):
    result = await weather_skill.execute(
        {"location": "San Francisco"},
        context,
    )

    assert result.status == SkillStatus.SUCCESS
    assert "temperature" in result.data

@pytest.mark.asyncio
async def test_weather_skill_invalid_input(weather_skill, context):
    is_valid, error = await weather_skill.validate_input({})

    assert not is_valid
    assert "location" in error

@pytest.mark.asyncio
async def test_weather_skill_permission_check(weather_skill):
    context = SkillContext(
        user_id="test-user",
        permissions=[],  # No permissions
    )

    has_permission, missing = await weather_skill.check_permissions(context)

    assert not has_permission
    assert missing == "skills:weather"
```

### Integration Tests

```python
from aragora.skills import get_skill_registry

@pytest.mark.asyncio
async def test_skill_registration_and_invocation():
    registry = get_skill_registry()
    skill = WeatherSkill()

    # Register
    registry.register(skill)
    assert registry.has_skill("weather")

    # Invoke
    context = SkillContext(
        user_id="test-user",
        permissions=["skills:weather"],
    )
    result = await registry.invoke("weather", {"location": "NYC"}, context)

    assert result.success
```

## Publishing to Marketplace

### 1. Package Your Skill

Create a `skill.yaml` manifest:

```yaml
name: weather-skill
version: 1.0.0
description: Get current weather for any location
author: your-name
repository: https://github.com/your-name/weather-skill
license: MIT

tier: free  # free, standard, premium, enterprise
category: web_tools

entrypoint: weather_skill.WeatherSkill

dependencies:
  - requests>=2.28.0
  - aiohttp>=3.8.0

permissions_required:
  - skills:weather

tags:
  - weather
  - api
  - external
```

### 2. Publish

```python
from aragora.skills.marketplace import get_marketplace

marketplace = get_marketplace()

# Publish your skill
result = await marketplace.publish(
    skill=WeatherSkill(),
    manifest_path="skill.yaml",
    readme_path="README.md",
)

print(f"Published: {result.skill_id}")
```

### 3. Install From Marketplace

```python
# Search for skills
results = await marketplace.search("weather")

# Get details
listing = await marketplace.get_skill("weather-skill")
print(f"Rating: {listing.average_rating}/5")
print(f"Installs: {listing.install_count}")

# Install
result = await marketplace.install(
    skill_id="weather-skill",
    tenant_id="my-tenant",
)
```

## Best Practices

### 1. Handle Errors Gracefully

```python
async def execute(self, input_data: dict, context: SkillContext) -> SkillResult:
    try:
        result = await self._do_work(input_data)
        return SkillResult.create_success(result)
    except RateLimitError:
        return SkillResult.create_failure(
            "Rate limit exceeded",
            status=SkillStatus.RATE_LIMITED,
        )
    except TimeoutError:
        return SkillResult.create_timeout(self.manifest.max_execution_time_seconds)
    except Exception as e:
        logger.exception("Skill execution failed")
        return SkillResult.create_failure(str(e), error_code="internal_error")
```

### 2. Respect Rate Limits

```python
manifest = SkillManifest(
    name="api_skill",
    rate_limit_per_minute=60,  # Enforce rate limiting
    # ...
)
```

### 3. Use Caching

```python
from aragora.skills import SkillResult

async def execute(self, input_data: dict, context: SkillContext) -> SkillResult:
    # Check cache first
    cache_key = f"weather:{input_data['location']}"
    cached = await self._get_cached(cache_key)
    if cached:
        return SkillResult.create_success(cached, cache_hit=True)

    # Fetch fresh data
    data = await self._fetch_data(input_data)
    await self._cache(cache_key, data, ttl=300)

    return SkillResult.create_success(data, cache_hit=False)
```

### 4. Log Appropriately

```python
import logging

logger = logging.getLogger(__name__)

async def execute(self, input_data: dict, context: SkillContext) -> SkillResult:
    logger.info(
        "Executing weather skill",
        extra={
            "user_id": context.user_id,
            "location": input_data.get("location"),
        },
    )
    # ...
```

### 5. Validate Inputs Thoroughly

```python
async def execute(self, input_data: dict, context: SkillContext) -> SkillResult:
    # Use built-in validation
    is_valid, error = await self.validate_input(input_data)
    if not is_valid:
        return SkillResult.create_failure(
            error,
            status=SkillStatus.INVALID_INPUT,
        )

    # Additional custom validation
    if len(input_data.get("query", "")) > 1000:
        return SkillResult.create_failure(
            "Query too long (max 1000 chars)",
            status=SkillStatus.INVALID_INPUT,
        )

    # Proceed with execution
    # ...
```

## Synchronous Skills

For skills with synchronous code, use `SyncSkill`:

```python
from aragora.skills import SyncSkill, SkillResult

class LegacyApiSkill(SyncSkill):
    @property
    def manifest(self) -> SkillManifest:
        # ...

    def execute_sync(self, input_data: dict, context: SkillContext) -> SkillResult:
        # Synchronous code here - will be wrapped in async executor
        response = requests.get(f"https://api.example.com?q={input_data['query']}")
        return SkillResult.create_success(response.json())
```

## Environment Variables

Access configuration via environment:

```python
import os

class ApiSkill(Skill):
    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="api_skill",
            required_env_vars=["API_KEY", "API_SECRET"],
            # ...
        )

    async def execute(self, input_data: dict, context: SkillContext) -> SkillResult:
        api_key = os.environ.get("API_KEY")
        if not api_key:
            return SkillResult.create_failure(
                "API_KEY environment variable not set",
                error_code="missing_config",
            )
        # ...
```

## See Also

- [ADAPTER_GUIDE.md](../guides/ADAPTER_GUIDE.md) - Creating Knowledge Mound adapters
- [API Reference](../api/API_REFERENCE.md) - Full API documentation
- [aragora/plugins/builtin/](../../aragora/plugins/builtin/) - Example built-in plugins
