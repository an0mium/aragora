---
title: Python SDK Migration (aragora-client -> aragora-sdk)
description: Python SDK Migration (aragora-client -> aragora-sdk)
---

# Python SDK Migration (`aragora-client` -> `aragora-sdk`)

This is the canonical migration path for Python SDK users.

## Policy

- Use `aragora-sdk` for all new Python integrations.
- Keep `aragora-client` only for legacy compatibility during migration.
- Prefer `/api/v1` endpoints for SDK integrations.

## Quick Migration

```bash
pip uninstall aragora-client
pip install aragora-sdk
```

```python
# Before (deprecated)
from aragora_client import AragoraClient

# After (canonical)
from aragora_sdk import AragoraClient
```

## Detailed Guide

For method-level mapping and advanced examples, see:

- `docs/guides/MIGRATION_GUIDE.md#python-aragora-client--aragora-sdk`
