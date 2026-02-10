# Python SDK Consolidation

**Recommended (new integrations):** `aragora-sdk` (import `aragora_sdk`) in `sdk/python/`.

**Legacy (deprecated):** `aragora-client` (import `aragora_client`) in `aragora-py/`.

## What To Use

- Use `aragora-sdk` for all new Python integrations and remote API usage.
- If you are currently using `aragora-client`, migrate to `aragora-sdk`.

## References

- `sdk/python/README.md` (official Python SDK: `aragora-sdk`)
- `docs/guides/MIGRATION_GUIDE.md` (`aragora-client` -> `aragora-sdk`)
- `aragora-py/README.md` (legacy client notes)

## Deprecation Plan (High Level)

- `aragora-client` remains for backwards compatibility and receives only critical fixes.
- Feature work ships in `aragora-sdk`.
