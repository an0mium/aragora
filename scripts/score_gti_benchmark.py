"""Score the Ground-Truth Integrity benchmark and publish a scorecard + status surface.

Usage:
    python scripts/score_gti_benchmark.py \
        --corpus docs/status/generated/gti/scenarios.json \
        --scorecard-out docs/status/generated/gti/scorecard-$(date -u +%Y%m%dT%H%M%SZ).json \
        --status-out docs/status/GTI_GROUND_TRUTH_INTEGRITY_STATUS.md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aragora.gti.scenarios import load_scenarios  # noqa: E402
from aragora.gti.scorer import ScoreResult, score_corpus  # noqa: E402


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _render_status(result: ScoreResult, generated_at: str) -> str:
    n, g, d = result.naive, result.gated, result.delta
    return (
        "# Ground-Truth Integrity (GTI) Status\n\n"
        f"Last updated: {generated_at}\n\n"
        f"Benchmark: gti-ground-truth-integrity-v1 | scenarios: {result.scenario_count}\n\n"
        "| metric | naive | gated | delta (gated improvement) |\n"
        "|---|---|---|---|\n"
        f"| stale_belief_action_rate | {n.stale_belief_action_rate:.3f} | "
        f"{g.stale_belief_action_rate:.3f} | {d.stale_belief_action_rate:+.3f} |\n"
        f"| detection_rate | {n.detection_rate:.3f} | {g.detection_rate:.3f} | "
        f"{d.detection_rate:+.3f} |\n"
        f"| correction_rate | {n.correction_rate:.3f} | {g.correction_rate:.3f} | "
        f"{d.correction_rate:+.3f} |\n"
        f"| false_green_rate | {n.false_green_rate:.3f} | {g.false_green_rate:.3f} | "
        f"{d.false_green_rate:+.3f} |\n\n"
        "Project scale metrics are not duplicated here; see `docs/METRICS.md` "
        "(`python scripts/regenerate_metrics.py --check`).\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the GTI benchmark.")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--scorecard-out", required=True, type=Path)
    parser.add_argument("--status-out", required=True, type=Path)
    parser.add_argument("--now", default=None, help="ISO8601 override for deterministic runs")
    args = parser.parse_args(argv)

    generated_at = args.now or _now_iso()
    scenarios = load_scenarios(args.corpus)
    result = score_corpus(scenarios)

    scorecard = {
        "benchmark": "gti-ground-truth-integrity-v1",
        "generated_at": generated_at,
        "scenario_count": result.scenario_count,
        "naive": dataclasses.asdict(result.naive),
        "gated": dataclasses.asdict(result.gated),
        "delta": dataclasses.asdict(result.delta),
        "scenario_ids": [s.id for s in scenarios],
    }
    args.scorecard_out.parent.mkdir(parents=True, exist_ok=True)
    args.scorecard_out.write_text(json.dumps(scorecard, indent=2) + "\n", encoding="utf-8")
    args.status_out.parent.mkdir(parents=True, exist_ok=True)
    args.status_out.write_text(_render_status(result, generated_at), encoding="utf-8")
    print(json.dumps({"ok": True, "scorecard": str(args.scorecard_out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
