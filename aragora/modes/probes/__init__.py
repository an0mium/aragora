"""
Capability probing system for adversarial testing.

This package provides tools for probing agent capabilities to find:
- Self-contradictions
- Hallucinated evidence
- Sycophantic behavior
- Premature concession
- Calibration issues
- Reasoning depth gaps
- Edge case failures
- Prompt injection vulnerabilities
- Capability exaggeration
"""

from .models import (
    ProbeResult,
    ProbeType,
    VulnerabilityReport,
    VulnerabilitySeverity,
)
from .strategies import (
    STRATEGIES,
    CapabilityExaggerationProbe,
    ConfidenceCalibrationProbe,
    ContradictionTrap,
    EdgeCaseProbe,
    HallucinationBait,
    InstructionInjectionProbe,
    PersistenceChallenge,
    ProbeStrategy,
    ReasoningDepthProbe,
    SycophancyTest,
)

__all__ = [
    # Models
    "ProbeType",
    "VulnerabilitySeverity",
    "ProbeResult",
    "VulnerabilityReport",
    # Strategy base
    "ProbeStrategy",
    # Concrete strategies
    "ContradictionTrap",
    "HallucinationBait",
    "SycophancyTest",
    "PersistenceChallenge",
    "ConfidenceCalibrationProbe",
    "ReasoningDepthProbe",
    "EdgeCaseProbe",
    "InstructionInjectionProbe",
    "CapabilityExaggerationProbe",
    # Registry
    "STRATEGIES",
]
