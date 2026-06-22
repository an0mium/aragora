"""Aragora metrics package.

Exposes:
- AGT-06 VIAH (verifiable improvements per agent-hour): :class:`ViahReport`,
  :func:`compute_viah`, :func:`viah_score`.
- AGT-03 Manifold Brier scorer: :class:`ManifoldBrierScorer`,
  :class:`ManifoldPrediction`, :class:`BrierWindowSummary`,
  :class:`CalibrationBin`, :func:`brier_score`, :func:`manifold_brier_enabled`.

Imports are lazy (PEP 562 ``__getattr__``) so importing
``aragora.metrics.manifold_brier`` does not trigger the heavy transitive
dependency chain in ``aragora.metrics.viah``.
"""

from __future__ import annotations

__all__ = [
    # AGT-06
    "ViahReport",
    "compute_viah",
    "viah_score",
    # AGT-03
    "ManifoldBrierScorer",
    "ManifoldPrediction",
    "BrierWindowSummary",
    "CalibrationBin",
    "brier_score",
    "manifold_brier_enabled",
]

_VIAH_NAMES = {"ViahReport", "compute_viah", "viah_score"}
_BRIER_NAMES = {
    "ManifoldBrierScorer",
    "ManifoldPrediction",
    "BrierWindowSummary",
    "CalibrationBin",
    "brier_score",
    "manifold_brier_enabled",
}


def __getattr__(name: str):  # noqa: ANN001, ANN201 — PEP 562 module __getattr__
    if name in _VIAH_NAMES:
        from aragora.metrics.viah import ViahReport, compute_viah, viah_score

        _globals = globals()
        _globals["ViahReport"] = ViahReport
        _globals["compute_viah"] = compute_viah
        _globals["viah_score"] = viah_score
        return _globals[name]
    if name in _BRIER_NAMES:
        from aragora.metrics.manifold_brier import (
            BrierWindowSummary,
            CalibrationBin,
            ManifoldBrierScorer,
            ManifoldPrediction,
            brier_score,
            manifold_brier_enabled,
        )

        _globals = globals()
        _globals["BrierWindowSummary"] = BrierWindowSummary
        _globals["CalibrationBin"] = CalibrationBin
        _globals["ManifoldBrierScorer"] = ManifoldBrierScorer
        _globals["ManifoldPrediction"] = ManifoldPrediction
        _globals["brier_score"] = brier_score
        _globals["manifold_brier_enabled"] = manifold_brier_enabled
        return _globals[name]
    raise AttributeError(f"module 'aragora.metrics' has no attribute {name!r}")
