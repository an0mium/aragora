"""
Gauntlet - Adversarial Validation Engine.

Stress-tests high-stakes decisions through multi-agent adversarial debate.

The Gauntlet orchestrates:
- Red Team attacks (logical, security, scalability)
- Capability probing (hallucination, sycophancy, consistency)
- Scenario matrix testing (scale, risk, time horizon)
- Risk aggregation and Decision Receipts

Two orchestration options:
1. GauntletRunner - Simple 3-phase runner with templates (recommended)
2. GauntletOrchestrator - Full 5-phase orchestrator with deep audit

Usage (Runner - recommended):
    from aragora.gauntlet import GauntletRunner, GauntletConfig

    config = GauntletConfig(
        attack_categories=[AttackCategory.SECURITY, AttackCategory.COMPLIANCE],
        agents=["anthropic-api", "openai-api", "gemini"],
    )
    runner = GauntletRunner(config)
    result = await runner.run("spec.md content here")
    receipt = result.to_receipt()

Usage (Orchestrator - full 5-phase):
    from aragora.gauntlet import GauntletOrchestrator, OrchestratorConfig

    config = OrchestratorConfig(input_content="spec.md content")
    orchestrator = GauntletOrchestrator(agents)
    result = await orchestrator.run(config)
"""

# Shared types (canonical source)
from .types import (
    InputType,
    Verdict,
    SeverityLevel,
    GauntletSeverity,
    GauntletPhase,
    BaseFinding,
    RiskSummary,
)

# Config and categories
from .config import GauntletConfig, AttackCategory, ProbeCategory

# Result types
from .result import GauntletResult, Vulnerability
from .result import RiskSummary as ResultRiskSummary  # Alias for backward compat

# Runner
from .runner import GauntletRunner

# Output formats
from .receipt import DecisionReceipt
from .heatmap import RiskHeatmap, HeatmapCell

# Re-export orchestrator classes from modes (full 5-phase implementation)
# These are imported here to provide a single canonical import location
# NOTE: Import is deferred to avoid circular imports and deprecation warning at import time
def _get_orchestrator_classes():
    """Lazy import of orchestrator classes."""
    import warnings
    # Suppress deprecation warning during internal re-export
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from aragora.modes.gauntlet import (
            GauntletOrchestrator as _Orchestrator,
            GauntletConfig as _OrchestratorConfig,
            GauntletProgress as _Progress,
            GauntletResult as _OrchestratorResult,
            Finding,
            run_gauntlet,
            QUICK_GAUNTLET,
            THOROUGH_GAUNTLET,
            CODE_REVIEW_GAUNTLET,
            POLICY_GAUNTLET,
        )
    return {
        "GauntletOrchestrator": _Orchestrator,
        "OrchestratorConfig": _OrchestratorConfig,
        "GauntletProgress": _Progress,
        "OrchestratorResult": _OrchestratorResult,
        "Finding": Finding,
        "run_gauntlet": run_gauntlet,
        "QUICK_GAUNTLET": QUICK_GAUNTLET,
        "THOROUGH_GAUNTLET": THOROUGH_GAUNTLET,
        "CODE_REVIEW_GAUNTLET": CODE_REVIEW_GAUNTLET,
        "POLICY_GAUNTLET": POLICY_GAUNTLET,
    }

# Lazy attribute access for orchestrator classes
def __getattr__(name: str):
    """Lazy loading for orchestrator classes."""
    orchestrator_names = {
        "GauntletOrchestrator",
        "OrchestratorConfig",
        "GauntletProgress",
        "OrchestratorResult",
        "Finding",
        "run_gauntlet",
        "QUICK_GAUNTLET",
        "THOROUGH_GAUNTLET",
        "CODE_REVIEW_GAUNTLET",
        "POLICY_GAUNTLET",
    }
    if name in orchestrator_names:
        classes = _get_orchestrator_classes()
        return classes[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Shared types (canonical)
    "InputType",
    "Verdict",
    "SeverityLevel",
    "GauntletSeverity",
    "GauntletPhase",
    "BaseFinding",
    "RiskSummary",
    # Config (Runner)
    "GauntletConfig",
    "AttackCategory",
    "ProbeCategory",
    # Result (Runner)
    "GauntletResult",
    "Vulnerability",
    # Runner
    "GauntletRunner",
    # Receipt
    "DecisionReceipt",
    # Heatmap
    "RiskHeatmap",
    "HeatmapCell",
    # Orchestrator (full 5-phase) - lazy loaded
    "GauntletOrchestrator",
    "OrchestratorConfig",
    "GauntletProgress",
    "OrchestratorResult",
    "Finding",
    "run_gauntlet",
    "QUICK_GAUNTLET",
    "THOROUGH_GAUNTLET",
    "CODE_REVIEW_GAUNTLET",
    "POLICY_GAUNTLET",
]
