# Deprecated Packages

This directory contains archived SDK packages that have been deprecated in favor of newer, consolidated alternatives.

## Why These Packages Are Deprecated

As part of Aragora's SDK consolidation effort (v2.4.0+), we've unified our SDKs:

- **Python SDKs** consolidated into `aragora-client` (published from `aragora-py/`)
- **TypeScript SDKs** consolidated into `@aragora/sdk` (published from `sdk/typescript/`)

## Archived Packages

### sdk-python (formerly `sdk/python/`)

**Package Name:** `aragora`
**Replacement:** `pip install aragora-client`

### aragora-sdk (formerly `aragora-sdk/`)

**Package Name:** `aragora-sdk`
**Replacement:** `pip install aragora-client`

## Migration

```python
# OLD
from aragora import AragoraClient

# NEW
from aragora_client import AragoraClient
```

## Documentation

- [SDK Comparison](../docs/SDK_COMPARISON.md)
- [SDK Consolidation Timeline](../docs/SDK_CONSOLIDATION.md)
