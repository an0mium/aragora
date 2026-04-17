"""Aragora metrics package.

Currently exposes the AGT-06 VIAH (verifiable improvements per agent-hour)
helper. See ``docs/plans/AGENT_CIVILIZATION_SUBSTRATE.md`` §4 and
issue #6067 for the rationale.
"""

from __future__ import annotations

from aragora.metrics.viah import ViahReport, compute_viah, viah_score

__all__ = ["ViahReport", "compute_viah", "viah_score"]
