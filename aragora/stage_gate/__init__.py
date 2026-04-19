"""Stage-Gate Conductor support modules.

The Stage-Gate Conductor runs as a scheduled trigger that watches roadmap
tiers, proof surfaces, and queue labels for drift.  When it detects drift it
creates a ``stage-gate-drift`` labeled GitHub issue.

This sub-package contains post-detection helpers that operate on those drift
issues.  ``auto_remediation`` dispatches a small, conservative repair for
recognised drift patterns so a human does not have to pick up every drift
ticket by hand.
"""

from __future__ import annotations

from aragora.stage_gate.auto_remediation import (
    AUTO_REMEDIATION_FLAG_ENV,
    RemediationAction,
    RemediationResult,
    auto_remediation_enabled,
    remediate_drift,
)

__all__ = [
    "AUTO_REMEDIATION_FLAG_ENV",
    "RemediationAction",
    "RemediationResult",
    "auto_remediation_enabled",
    "remediate_drift",
]
