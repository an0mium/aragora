"""Flag-gated CruxSet emission for the production debate path.

This module is the seam between the existing :class:`CruxDetector`
(which currently runs as a post-analysis byproduct in the debate
phases) and the new :class:`CruxSet` contract. It is **dormant** unless
``ARAGORA_CRUXSET_EMISSION_ENABLED`` is set, satisfying the AGT-*
"planning truth, no live behavior change" rule from
``docs/status/NEXT_STEPS_CANONICAL.md``.

When enabled, callers in the debate path (e.g. winner_selector,
analytics_phase) can call :func:`maybe_emit_cruxset` after the
BeliefNetwork has been populated, and a CruxSet will be returned for
downstream persistence via the receipt path. When disabled (the
default), the function returns ``None`` and the debate path is
unchanged.

Activation gate: open this flag only after the substrate-first gate in
``docs/status/NEXT_STEPS_CANONICAL.md`` permits the AGT-* upper-layer
tranche, and DIC-15 (#6025) and DIC-16 (#6026) are landed.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from aragora.reasoning.cruxset import CruxSet, build_cruxset_from_analysis

if TYPE_CHECKING:
    from aragora.reasoning.belief import BeliefNetwork

logger = logging.getLogger(__name__)

CRUXSET_EMISSION_ENV_VAR = "ARAGORA_CRUXSET_EMISSION_ENABLED"


def cruxset_emission_enabled() -> bool:
    """Return True when the AGT-01 CruxSet emission surface is enabled.

    Reads :data:`CRUXSET_EMISSION_ENV_VAR` from the process environment.
    Default is False; emission is dormant unless explicitly enabled.
    """
    raw = str(os.environ.get(CRUXSET_EMISSION_ENV_VAR) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def enable_cruxset_emission() -> None:
    """Enable the AGT-01 CruxSet emission surface for the current process.

    Sets ``ARAGORA_CRUXSET_EMISSION_ENABLED=1``. Mirror of
    :func:`aragora.markets.enable_synthetic_markets`.
    """
    os.environ[CRUXSET_EMISSION_ENV_VAR] = "1"


def maybe_emit_cruxset(
    *,
    question: str,
    network: "BeliefNetwork | None" = None,
    analysis_payload: dict[str, Any] | None = None,
    decision: str | None = None,
    receipt_id: str = "",
    provenance: dict[str, Any] | None = None,
    top_k: int = 5,
    min_score: float = 0.1,
) -> CruxSet | None:
    """Build a CruxSet for the given debate context if emission is enabled.

    Two input modes:

    - Pass ``network`` (a populated :class:`BeliefNetwork`) and the
      function will instantiate :class:`CruxDetector` and run
      ``detect_cruxes(top_k=top_k, min_score=min_score)`` itself.
    - Pass ``analysis_payload`` (the dict returned by
      ``CruxAnalysisResult.to_dict()``) to skip the detector step. This
      is the test-friendly path because it removes the BeliefNetwork
      dependency.

    Returns ``None`` when emission is disabled, when neither input is
    provided, or when the analysis surfaces no cruxes. Errors during
    detector invocation are logged at warning level and swallowed
    rather than re-raised — the function is a soft enrichment layer
    and must not cause debate failures.
    """
    if not cruxset_emission_enabled():
        return None

    payload = analysis_payload
    if payload is None:
        if network is None:
            logger.warning(
                "cruxset emission enabled but neither network nor analysis_payload "
                "supplied for question=%r",
                question[:80],
            )
            return None
        try:
            from aragora.reasoning.crux_detector import CruxDetector

            detector = CruxDetector(network=network)
            analysis = detector.detect_cruxes(top_k=top_k, min_score=min_score)
            payload = analysis.to_dict()
        except Exception as exc:  # noqa: BLE001 - soft enrichment must not crash debate
            logger.warning("cruxset emission failed for question=%r: %s", question[:80], exc)
            return None

    if not payload or not payload.get("cruxes"):
        logger.debug("cruxset emission produced no cruxes for question=%r", question[:80])
        return None

    try:
        return build_cruxset_from_analysis(
            question=question,
            analysis_payload=payload,
            decision=decision,
            receipt_id=receipt_id,
            provenance=provenance,
            max_cruxes=top_k,
        )
    except (ValueError, KeyError) as exc:
        logger.warning(
            "cruxset emission could not build CruxSet for question=%r: %s",
            question[:80],
            exc,
        )
        return None


__all__ = [
    "CRUXSET_EMISSION_ENV_VAR",
    "cruxset_emission_enabled",
    "enable_cruxset_emission",
    "maybe_emit_cruxset",
]
