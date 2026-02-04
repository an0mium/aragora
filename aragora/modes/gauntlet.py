"""
Gauntlet Mode - Adversarial Validation Engine.

.. deprecated:: 2.0
    This module is a backward-compatibility shim. Use ``aragora.gauntlet`` instead.

Migration::

    # Old (deprecated)
    from aragora.modes.gauntlet import GauntletOrchestrator

    # New (recommended)
    from aragora.gauntlet import GauntletOrchestrator
"""

from __future__ import annotations

import warnings

warnings.warn(
    "aragora.modes.gauntlet is deprecated. Use aragora.gauntlet instead. "
    "See docs/GAUNTLET_ARCHITECTURE.md for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical location
from aragora.gauntlet.orchestrator import (  # noqa: F401, E402
    AI_ACT_GAUNTLET,
    CODE_REVIEW_GAUNTLET,
    Finding,
    GDPR_GAUNTLET,
    GauntletConfig,
    GauntletOrchestrator,
    GauntletProgress,
    GauntletResult,
    HIPAA_GAUNTLET,
    POLICY_GAUNTLET,
    QUICK_GAUNTLET,
    SECURITY_GAUNTLET,
    SOX_GAUNTLET,
    THOROUGH_GAUNTLET,
    VerifiedClaim,
    get_compliance_gauntlet,
    run_gauntlet,
)

__all__ = [
    "GauntletOrchestrator",
    "GauntletConfig",
    "GauntletProgress",
    "GauntletResult",
    "Finding",
    "VerifiedClaim",
    "run_gauntlet",
    "get_compliance_gauntlet",
    "QUICK_GAUNTLET",
    "THOROUGH_GAUNTLET",
    "CODE_REVIEW_GAUNTLET",
    "POLICY_GAUNTLET",
    "GDPR_GAUNTLET",
    "HIPAA_GAUNTLET",
    "AI_ACT_GAUNTLET",
    "SECURITY_GAUNTLET",
    "SOX_GAUNTLET",
]
