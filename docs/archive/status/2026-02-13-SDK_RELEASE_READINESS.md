# SDK Release Readiness - February 13, 2026

Version target: `2.6.3`

## Alignment Status

- Canonical version file: `aragora/__version__.py` -> `2.6.3`
- Python SDK version: `sdk/python/pyproject.toml` -> `2.6.3`
- TypeScript SDK version: `sdk/typescript/package.json` -> `2.6.3`
- Version alignment check: `python scripts/check_version_alignment.py` -> **pass**

## Contract and Parity Gates

- `python scripts/check_sdk_parity.py --strict --baseline scripts/baselines/check_sdk_parity.json --budget scripts/baselines/check_sdk_parity_budget.json` -> **pass** (no regression)
- `python scripts/check_sdk_namespace_parity.py --strict --baseline scripts/baselines/check_sdk_namespace_parity.json` -> **pass** (no regression)

## Package Validation

- `pytest -q sdk/python/tests` -> **pass**
- `python -m build sdk/python` -> **pass**
- `cd sdk/typescript && npm run -s typecheck && npm run -s test && npm run -s build` -> **pass**

## Publish Commands

Python (`aragora-sdk`):

1. Run workflow: `.github/workflows/publish-python-sdk.yml`
2. Inputs:
   - `version=2.6.3`
   - `confirm=PUBLISH`

TypeScript (`@aragora/sdk`):

1. Push tag: `sdk-v2.6.3` (or run workflow manually)
2. Workflow: `.github/workflows/publish-sdk.yml`

## Release Decision

SDK artifacts are release-ready for `2.6.3` based on alignment, parity, and package build/test gates.
