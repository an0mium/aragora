"""Auditable calibration report aggregation (issue #8229, ODR-5).

Builds a structured, interpretable calibration report for a single agent from
the EXISTING calibration stores — the ELO ratings table (overall tournament
prediction calibration) and the domain calibration tables maintained by
:class:`aragora.ranking.calibration_engine.DomainCalibrationEngine`.

Two rules drive every figure emitted here:

1. **Read-only aggregation** — no new computation paths beyond summarizing
   what the calibration engines already recorded. Nothing is re-scored.
2. **Honest absence** — agents with no calibration data get an explicit
   ``{"status": "absent", "reason": ...}`` report, never invented numbers,
   and every numeric figure carries a mandatory ``sample_size`` disclosure.

The report backs ``GET /api/v1/agents/{id}/calibration-report`` and the
calibrated-confidence provenance block in Open Decision Receipt (ODR)
exports (:mod:`aragora.gauntlet.odr_export`).
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from aragora.config import ELO_CALIBRATION_MIN_COUNT

if TYPE_CHECKING:
    from aragora.ranking.elo import EloSystem

logger = logging.getLogger(__name__)

__all__ = [
    "CALIBRATION_REPORT_ENDPOINT_TEMPLATE",
    "build_calibration_report",
    "build_odr_calibration_provenance",
]

#: Canonical endpoint template the report is served from. Provenance refs in
#: ODR receipts point at this path so "confidence: 0.84" is auditable.
CALIBRATION_REPORT_ENDPOINT_TEMPLATE = "/api/v1/agents/{agent}/calibration-report"


def _absent(reason: str) -> dict[str, Any]:
    """Explicit absent marker — same contract as ODR's honest-absence rule."""
    return {"status": "absent", "reason": reason, "sample_size": 0}


def _get_elo_system(elo_system: EloSystem | None) -> EloSystem:
    if elo_system is not None:
        return elo_system
    from aragora.ranking.elo import EloSystem as _EloSystem

    return _EloSystem()


def _query_prediction_window(elo: EloSystem, agent_name: str) -> dict[str, Any] | None:
    """Read first/last tournament-prediction timestamps for the agent.

    Returns None when the table is unreadable or holds no rows for the agent;
    callers translate that into an explicit absent marker.
    """
    try:
        from aragora.ranking.calibration_database import CalibrationDatabase

        db = CalibrationDatabase(elo.db_path)
        row = db.fetch_one(
            """
            SELECT MIN(created_at), MAX(created_at), COUNT(*)
            FROM calibration_predictions
            WHERE predictor_agent = ?
            """,
            (agent_name,),
        )
    except sqlite3.Error as e:
        logger.debug("calibration_predictions window query failed for %s: %s", agent_name, e)
        return None
    if not row or not row[2]:
        return None
    return {"first_recorded_at": row[0], "last_recorded_at": row[1], "sample_size": int(row[2])}


def _query_domain_window(elo: EloSystem, agent_name: str) -> str | None:
    """Most recent domain-calibration update timestamp, if any."""
    try:
        from aragora.ranking.calibration_database import CalibrationDatabase

        db = CalibrationDatabase(elo.db_path)
        row = db.fetch_one(
            "SELECT MAX(updated_at) FROM domain_calibration WHERE agent_name = ?",
            (agent_name,),
        )
    except sqlite3.Error as e:
        logger.debug("domain_calibration window query failed for %s: %s", agent_name, e)
        return None
    return row[0] if row else None


def build_calibration_report(
    agent_name: str,
    *,
    elo_system: EloSystem | None = None,
    domain: str | None = None,
) -> dict[str, Any]:
    """Build the auditable calibration report for one agent.

    Args:
        agent_name: Agent to report on.
        elo_system: Optional shared :class:`EloSystem`; a default instance is
            created when omitted (read-only usage either way).
        domain: Optional domain filter for the per-domain breakdown and curve.

    Returns:
        A JSON-serializable report. ``status`` is ``"ok"`` when any calibration
        data exists, else ``"absent"`` with an explicit reason. Every numeric
        block carries a ``sample_size`` disclosure.
    """
    from aragora.ranking.calibration_engine import DomainCalibrationEngine

    elo = _get_elo_system(elo_system)
    generated_at = datetime.now(timezone.utc).isoformat()

    # --- Overall calibration (ratings table; aggregate across all sources) ---
    has_rating = elo.has_rating(agent_name)
    rating = elo.get_rating(agent_name) if has_rating else None
    overall_total = int(rating.calibration_total) if rating is not None else 0

    # --- Domain calibration (domain_calibration / calibration_buckets) ---
    domain_engine = DomainCalibrationEngine(db_path=elo.db_path, elo_system=None)
    domain_stats = domain_engine.get_domain_stats(agent_name, domain=domain)
    domain_total = int(domain_stats.get("total", 0))
    curve = domain_engine.get_calibration_curve(agent_name, domain=domain)
    curve_total = sum(b.predictions for b in curve)

    if overall_total == 0 and domain_total == 0:
        return {
            "agent": agent_name,
            "status": "absent",
            "reason": (
                "no calibration data recorded for this agent "
                "(no resolved predictions in the ratings, domain_calibration, "
                "or calibration_buckets stores)"
            ),
            "sample_size": 0,
            "generated_at": generated_at,
            "endpoint": CALIBRATION_REPORT_ENDPOINT_TEMPLATE.format(agent=agent_name),
        }

    # Overall block — only from real rating rows, never synthesized defaults.
    overall: dict[str, Any]
    if rating is not None and overall_total > 0:
        overall = {
            "status": "ok",
            "sample_size": overall_total,
            "predictions": overall_total,
            "correct": int(rating.calibration_correct),
            "accuracy": rating.calibration_accuracy,
            "brier_score": rating.calibration_brier_score,
            "calibration_score": rating.calibration_score,
            "meets_minimum_sample": overall_total >= ELO_CALIBRATION_MIN_COUNT,
            "minimum_sample_threshold": ELO_CALIBRATION_MIN_COUNT,
        }
    else:
        overall = _absent(
            "agent has no resolved predictions in the ratings store "
            "(overall accuracy/Brier cannot be reported)"
        )

    # Per-domain block.
    domains: dict[str, Any]
    if domain_total > 0:
        domains = {
            "status": "ok",
            "sample_size": domain_total,
            "predictions": domain_total,
            "correct": int(domain_stats.get("correct", 0)),
            "accuracy": domain_stats.get("accuracy", 0.0),
            "brier_score": domain_stats.get("brier_score"),
            "by_domain": {
                name: {
                    "sample_size": int(stats["predictions"]),
                    "predictions": int(stats["predictions"]),
                    "correct": int(stats["correct"]),
                    "accuracy": stats["accuracy"],
                    "brier_score": stats["brier_score"],
                }
                for name, stats in domain_stats.get("domains", {}).items()
            },
        }
    else:
        domains = _absent(
            "agent has no domain-tagged predictions recorded"
            + (f" for domain '{domain}'" if domain else "")
        )

    # Calibration curve + ECE — only when bucket data exists. The engine's
    # get_expected_calibration_error returns 1.0 for empty data, which would
    # read as "maximally miscalibrated"; honest absence instead.
    curve_block: dict[str, Any]
    ece_block: dict[str, Any]
    if curve_total > 0:
        curve_block = {
            "status": "ok",
            "sample_size": curve_total,
            "buckets": [
                {
                    "bucket": b.bucket_key,
                    "confidence_range": [b.bucket_start, b.bucket_end],
                    "sample_size": int(b.predictions),
                    "predictions": int(b.predictions),
                    "correct": int(b.correct),
                    "accuracy": b.accuracy,
                    "expected_accuracy": b.expected_accuracy,
                    "brier_score": b.brier_score,
                }
                for b in curve
            ],
        }
        ece_block = {
            "status": "ok",
            "value": domain_engine.get_expected_calibration_error(agent_name),
            "sample_size": curve_total,
        }
    else:
        curve_block = _absent("no confidence-bucket data recorded for this agent")
        ece_block = _absent(
            "expected calibration error requires confidence-bucket data, none recorded"
        )

    # Data window — when the raw timestamps were recorded.
    prediction_window = _query_prediction_window(elo, agent_name)
    domain_updated_at = _query_domain_window(elo, agent_name)
    data_window: dict[str, Any] = {
        "tournament_predictions": (
            prediction_window
            if prediction_window is not None
            else _absent("no timestamped tournament predictions recorded")
        ),
        "domain_calibration_last_updated_at": domain_updated_at,
        "rating_last_updated_at": rating.updated_at if rating is not None else None,
    }

    return {
        "agent": agent_name,
        "status": "ok",
        "generated_at": generated_at,
        "endpoint": CALIBRATION_REPORT_ENDPOINT_TEMPLATE.format(agent=agent_name),
        "domain_filter": domain,
        "sample_size": max(overall_total, domain_total),
        "overall": overall,
        "domains": domains,
        "calibration_curve": curve_block,
        "expected_calibration_error": ece_block,
        "data_window": data_window,
        "sources": [
            "aragora.ranking.elo:ratings.calibration_*",
            "aragora.ranking.calibration_engine:domain_calibration",
            "aragora.ranking.calibration_engine:calibration_buckets",
            "aragora.ranking.calibration_engine:calibration_predictions",
        ],
    }


def build_odr_calibration_provenance(
    agent_names: list[str],
    *,
    elo_system: EloSystem | None = None,
) -> dict[str, Any] | None:
    """Build the ODR ``confidence.calibration.provenance_ref`` block.

    Checks which of the given agents actually have recorded calibration data
    and, when at least one does, returns a provenance reference pointing at
    the per-agent calibration-report endpoint with per-agent sample sizes.

    Returns ``None`` when no agent has calibration data — callers must then
    emit an explicit absent marker (never fabricate provenance).
    """
    if not agent_names:
        return None

    elo = _get_elo_system(elo_system)

    rows: list[dict[str, Any]] = []
    for name in sorted({n.strip() for n in agent_names if n and n.strip()}):
        try:
            if not elo.has_rating(name):
                continue
            rating = elo.get_rating(name)
        except ValueError:
            # Invalid agent name per EloSystem validation — skip, never guess.
            continue
        total = int(rating.calibration_total)
        if total <= 0:
            continue
        rows.append(
            {
                "agent": name,
                "sample_size": total,
                "accuracy": rating.calibration_accuracy,
                "brier_score": rating.calibration_brier_score,
                "report_ref": CALIBRATION_REPORT_ENDPOINT_TEMPLATE.format(agent=name),
            }
        )

    if not rows:
        return None

    return {
        "type": "aragora.calibration_report",
        "endpoint_template": CALIBRATION_REPORT_ENDPOINT_TEMPLATE,
        "agents": rows,
    }
