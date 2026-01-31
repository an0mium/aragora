# Python SDK Breaking Changes

This document tracks breaking changes specific to the Aragora Python SDK. For core API breaking changes, see the main [BREAKING_CHANGES.md](../../docs/BREAKING_CHANGES.md).

---

## Version 2.x

### v2.4.0 (2026-01-25)

**No breaking changes.** Added new namespace resources.

**New Features:**
- Added resources: `orgs`, `tenants`, `policies`, `codebase`, `costs`, `decisions`, `onboarding`, `notifications`, `gmail`, `explainability`
- Fixed payloads for billing and RBAC endpoints

---

### v2.2.0 (2026-01-24)

**No breaking changes.** Version aligned with core package.

---

### v2.0.0 (2026-01-17)

#### Breaking Changes

| Change | Before | After | Migration |
|--------|--------|-------|-----------|
| API version default | v1 | v2 | Explicit `api_version="v1"` to keep old behavior |
| Method naming | Camel case methods | Namespace-based methods | `client.getDebates()` becomes `client.debates.list()` |
| Response format | Direct data | Wrapped in `data`/`meta` | Access via `response["data"]` |

#### Method Renames

| Old Method | New Method |
|------------|------------|
| `client.getDebates()` | `client.debates.list()` |
| `client.createDebate(...)` | `client.debates.create(...)` |
| `client.getDebate(id)` | `client.debates.get(id)` |
| `client.getAgents()` | `client.agents.list()` |
| `client.getAgent(name)` | `client.agents.get(name)` |
| `client.submitVote(...)` | `client.debates.vote(debate_id, ...)` |
| `client.getConsensus(id)` | `client.consensus.get(id)` |

#### Migration Example

```python
# Before (v1.x)
from aragora.client import AragoraClient

client = AragoraClient(base_url="https://api.aragora.ai")
debates = client.getDebates()
debate = client.createDebate(topic="Should we use GraphQL?", max_rounds=3)

# After (v2.x)
from aragora import AragoraClient

client = AragoraClient(
    base_url="https://api.aragora.ai",
    api_version="v2"  # Optional, v2 is now default
)
response = client.debates.list()
debates = response["data"]["debates"]

response = client.debates.create(
    task="Should we use GraphQL?",  # 'topic' renamed to 'task'
    rounds=3  # 'max_rounds' renamed to 'rounds'
)
debate = response["data"]
```

#### Response Format Change

```python
# Before (v1.x) - Direct data access
response = client.debates.list()
debates = response["debates"]
count = response["count"]

# After (v2.x) - Wrapped response
response = client.debates.list()
debates = response["data"]["debates"]
count = response["data"]["count"]
meta = response["meta"]  # Contains version, timestamp, request_id
```

---

## Version 1.x

### v1.0.0 (2026-01-14)

**Initial stable release.** No breaking changes from pre-1.0 beta versions.

---

## Upcoming Breaking Changes

### Scheduled for v3.0.0

No breaking changes currently scheduled.

---

## Migration Guides

- [API v1 to v2 Migration](../../docs/MIGRATION_V1_TO_V2.md) - Complete guide for API migration
- [SDK Guide](../../docs/SDK_GUIDE.md) - Full SDK documentation

---

## Deprecation Warnings

The SDK emits `DeprecationWarning` for deprecated methods. Enable warnings to see them:

```python
import warnings
warnings.filterwarnings("default", category=DeprecationWarning, module="aragora")
```

Or run Python with `-W default`:

```bash
python -W default your_script.py
```

---

*Last updated: 2026-01-31*
