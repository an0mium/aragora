"""
Gauntlet - Adversarial Validation Engine.

Stress-tests high-stakes decisions through multi-agent adversarial debate.

The Gauntlet orchestrates:
- Red Team attacks (logical, security, scalability)
- Capability probing (hallucination, sycophancy, consistency)
- Scenario matrix testing (scale, risk, time horizon)
- Risk aggregation and Decision Receipts

Two orchestration options:
1. GauntletRunner (this package) - Simple 3-phase runner with templates
2. GauntletOrchestrator (aragora.modes.gauntlet) - Full 5-phase orchestrator

Usage:
    from aragora.gauntlet import GauntletRunner, GauntletConfig

    config = GauntletConfig(
        attack_categories=[AttackCategory.SECURITY, AttackCategory.COMPLIANCE],
        agents=["anthropic-api", "openai-api", "gemini"],
    )
    runner = GauntletRunner(config)
    result = await runner.run("spec.md content here")
    receipt = result.to_receipt()
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

__all__ = [
    # Shared types (canonical)
    "InputType",
    "Verdict",
    "SeverityLevel",
    "GauntletSeverity",
    "GauntletPhase",
    "BaseFinding",
    "RiskSummary",
    # Config
    "GauntletConfig",
    "AttackCategory",
    "ProbeCategory",
    # Result
    "GauntletResult",
    "Vulnerability",
    # Runner
    "GauntletRunner",
    # Receipt
    "DecisionReceipt",
    # Heatmap
    "RiskHeatmap",
    "HeatmapCell",
]
