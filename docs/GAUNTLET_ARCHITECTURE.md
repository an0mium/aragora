# Gauntlet Architecture

This document explains the Gauntlet package structure and provides a migration guide for users transitioning from the deprecated `aragora.modes.gauntlet` module.

## Package Structure

Gauntlet has two orchestration options, both accessible from the canonical `aragora.gauntlet` package:

| Class | Description | Use Case |
|-------|-------------|----------|
| `GauntletRunner` | Simple 3-phase runner with templates | Quick validations, CI/CD |
| `GauntletOrchestrator` | Full 5-phase orchestrator with deep audit | Thorough compliance audits |

### Directory Layout

```
aragora/gauntlet/
├── __init__.py       # Public API (canonical import location)
├── config.py         # GauntletConfig for Runner
├── runner.py         # GauntletRunner (3-phase)
├── result.py         # Result types
├── receipt.py        # DecisionReceipt generation
├── heatmap.py        # RiskHeatmap visualization
├── storage.py        # Persistent storage
├── templates.py      # Pre-built validation templates
├── types.py          # Shared types (InputType, Verdict, etc.)
└── personas/         # Regulatory personas (GDPR, HIPAA, etc.)
```

## Import Migration

### Deprecated (will show DeprecationWarning)

```python
# OLD - Deprecated
from aragora.modes.gauntlet import GauntletOrchestrator, GauntletConfig
```

### Recommended (canonical)

```python
# NEW - Canonical import location
from aragora.gauntlet import GauntletOrchestrator, OrchestratorConfig

# For the simpler Runner API
from aragora.gauntlet import GauntletRunner, GauntletConfig
```

### Name Changes

| Old Name | New Name | Notes |
|----------|----------|-------|
| `aragora.modes.gauntlet.GauntletConfig` | `aragora.gauntlet.OrchestratorConfig` | Renamed to avoid conflict with Runner's config |
| `aragora.modes.gauntlet.GauntletResult` | `aragora.gauntlet.OrchestratorResult` | Renamed for clarity |
| `aragora.modes.gauntlet.GauntletOrchestrator` | `aragora.gauntlet.GauntletOrchestrator` | Same name, new location |
| `aragora.modes.gauntlet.GauntletProgress` | `aragora.gauntlet.GauntletProgress` | Same name, new location |

## Choosing Between Runner and Orchestrator

### GauntletRunner (Recommended for most cases)

3-phase execution:
1. Attack Phase - Red team attacks
2. Probe Phase - Capability probing
3. Aggregation - Risk scoring

```python
from aragora.gauntlet import GauntletRunner, GauntletConfig, AttackCategory

config = GauntletConfig(
    attack_categories=[AttackCategory.SECURITY, AttackCategory.COMPLIANCE],
    agents=["anthropic-api", "openai-api"],
)
runner = GauntletRunner(config)
result = await runner.run("Your specification here")
receipt = result.to_receipt()
```

Best for:
- CI/CD integration
- Quick security checks
- Template-based validations

### GauntletOrchestrator (Full 5-phase)

5-phase execution:
1. Red Team - Security attacks
2. Probe - Capability testing
3. Deep Audit - Multi-round debate
4. Verification - Formal proofs (optional)
5. Risk Assessment - Final scoring

```python
from aragora.gauntlet import (
    GauntletOrchestrator,
    OrchestratorConfig,
    InputType,
)

config = OrchestratorConfig(
    input_type=InputType.POLICY,
    input_content="Your policy document here",
    persona="gdpr",  # Regulatory persona
    enable_verification=True,
    deep_audit_rounds=4,
)
orchestrator = GauntletOrchestrator(agents)
result = await orchestrator.run(config)
```

Best for:
- Compliance audits (SOC2, GDPR, HIPAA)
- Thorough architecture reviews
- High-stakes decision validation

## Shared Types

All types are defined in `aragora.gauntlet.types` and re-exported from the main package:

```python
from aragora.gauntlet import (
    InputType,      # SPEC, CODE, POLICY, ARCHITECTURE, etc.
    Verdict,        # APPROVED, NEEDS_REVIEW, REJECTED
    SeverityLevel,  # CRITICAL, HIGH, MEDIUM, LOW
    GauntletPhase,  # Phase enum
    BaseFinding,    # Base class for findings
    RiskSummary,    # Risk aggregation
)
```

## Pre-configured Profiles

Available from the canonical package:

```python
from aragora.gauntlet import (
    QUICK_GAUNTLET,        # 2 rounds, no verification
    THOROUGH_GAUNTLET,     # 6 rounds, verification enabled
    CODE_REVIEW_GAUNTLET,  # Security-focused code review
    POLICY_GAUNTLET,       # Policy document validation
)
```

## Decision Receipts

Both Runner and Orchestrator produce compatible results that can be converted to receipts:

```python
from aragora.gauntlet import DecisionReceipt

# From Runner result
receipt = result.to_receipt()

# From Orchestrator result (manual)
receipt = DecisionReceipt.from_mode_result(result, input_hash=...)

# Export formats
html = receipt.to_html()
markdown = receipt.to_markdown()
dict_data = receipt.to_dict()
```

## API Endpoints

The server exposes Gauntlet functionality via REST API:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/gauntlet/run` | POST | Start a gauntlet run |
| `/api/gauntlet/{id}` | GET | Get status/results |
| `/api/gauntlet/{id}/receipt` | GET | Get decision receipt |
| `/api/gauntlet/{id}/heatmap` | GET | Get risk heatmap |
| `/api/gauntlet/personas` | GET | List regulatory personas |
| `/api/gauntlet/results` | GET | List recent results |

See [API_REFERENCE.md](API_REFERENCE.md) for full documentation.

## Deprecation Timeline

| Date | Action |
|------|--------|
| 2026-01 | `aragora.modes.gauntlet` shows DeprecationWarning |
| 2026-04 | Warning upgraded to FutureWarning |
| 2026-07 | Module removed, imports will fail |

Migrate now to avoid breaking changes.
