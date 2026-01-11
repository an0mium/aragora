"""
Gauntlet - Adversarial Validation Engine.

Stress-tests high-stakes decisions through multi-agent adversarial debate.

The Gauntlet orchestrates:
- Red Team attacks (logical, security, scalability)
- Capability probing (hallucination, sycophancy, consistency)
- Scenario matrix testing (scale, risk, time horizon)
- Risk aggregation and Decision Receipts

Usage:
    from aragora.gauntlet import GauntletRunner, GauntletConfig

    config = GauntletConfig(
        attack_types=[AttackCategory.SECURITY, AttackCategory.COMPLIANCE],
        agents=["anthropic-api", "openai-api", "gemini"],
    )
    runner = GauntletRunner(config)
    result = await runner.run("spec.md content here")
    receipt = result.to_receipt()
"""

from .config import GauntletConfig, AttackCategory, ProbeCategory
from .result import GauntletResult, Vulnerability, RiskSummary
from .runner import GauntletRunner
from .receipt import DecisionReceipt
from .heatmap import RiskHeatmap, HeatmapCell

__all__ = [
    # Config
    "GauntletConfig",
    "AttackCategory",
    "ProbeCategory",
    # Result
    "GauntletResult",
    "Vulnerability",
    "RiskSummary",
    # Runner
    "GauntletRunner",
    # Receipt
    "DecisionReceipt",
    # Heatmap
    "RiskHeatmap",
    "HeatmapCell",
]
