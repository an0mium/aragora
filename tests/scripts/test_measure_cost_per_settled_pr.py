"""Tests for ``scripts/measure_cost_per_settled_pr.py`` (#8233 phase 1).

All inputs are synthetic fixtures (routing-record / receipt JSON files in
tmp dirs plus literal merged-PR lists); no test touches gh or the network.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_module(script_name: str) -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


meas = _load_module("measure_cost_per_settled_pr.py")

WINDOW_START = datetime(2026, 6, 5, tzinfo=timezone.utc)
WINDOW_END = datetime(2026, 6, 12, tzinfo=timezone.utc)
IN_WINDOW = "2026-06-10T12:00:00Z"
OUT_WINDOW = "2026-05-01T12:00:00Z"


def _routing_record(
    pr: int,
    *,
    generated_at: str = IN_WINDOW,
    recorded: bool = False,
    total_usd: Any = None,
    repo: str = "synaptent/aragora",
) -> dict[str, Any]:
    return {
        "record_type": "routing_rationale",
        "schema": "aragora.routing_rationale/v1",
        "generated_at": generated_at,
        "repo": repo,
        "pr": pr,
        "cost": {"recorded": recorded, "total_usd": total_usd},
    }


def _receipt(*, timestamp: str | None = IN_WINDOW, total_cost_usd: Any = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"receipt_id": "r1", "verdict": "PASS"}
    if timestamp is not None:
        payload["timestamp"] = timestamp
    if total_cost_usd is not None:
        payload["cost_summary"] = {"total_cost_usd": total_cost_usd}
    return payload


def _write(directory: Path, name: str, payload: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(payload))
    return path


def _compute(
    merged: list[dict[str, Any]],
    routing_dirs: list[Path],
    receipts_dirs: list[Path],
) -> dict[str, Any]:
    return meas.compute_cost_per_settled_pr(
        merged_prs=merged,
        routing_artifacts=meas.load_json_artifacts(routing_dirs),
        receipt_artifacts=meas.load_json_artifacts(receipts_dirs),
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        window_days=7,
        repo="synaptent/aragora",
    )


# --- Aggregation ---------------------------------------------------------------


def test_zero_coverage_baseline_is_honest(tmp_path: Path) -> None:
    routing = tmp_path / "routing"
    _write(routing, "r1.json", _routing_record(1))  # no cost recorded
    result = _compute([{"number": 1}, {"number": 2}], [routing], [tmp_path / "none"])
    assert result["settled_prs_total"] == 2
    assert result["settled_prs_with_cost_record"] == 0
    assert result["coverage_ratio"] == 0.0
    assert result["total_recorded_cost_usd"] == 0.0
    assert result["cost_per_settled_pr_usd"] == 0.0
    assert result["cost_is_lower_bound"] is True
    assert result["routing_records_without_cost"] == 1


def test_attributed_cost_and_coverage(tmp_path: Path) -> None:
    routing = tmp_path / "routing"
    _write(routing, "r1.json", _routing_record(1, recorded=True, total_usd=0.50))
    _write(routing, "r2.json", _routing_record(2, recorded=True, total_usd=0.25))
    _write(routing, "r3.json", _routing_record(3))  # settled, no cost
    result = _compute(
        [{"number": 1}, {"number": 2}, {"number": 3}, {"number": 4}],
        [routing],
        [tmp_path / "none"],
    )
    assert result["settled_prs_with_cost_record"] == 2
    assert result["covered_pr_numbers"] == [1, 2]
    assert result["attributed_recorded_cost_usd"] == 0.75
    assert result["cost_per_settled_pr_usd"] == round(0.75 / 4, 6)


def test_receipt_cost_summary_is_unattributed(tmp_path: Path) -> None:
    receipts = tmp_path / "receipts"
    _write(receipts, "a.json", _receipt(total_cost_usd="0.30"))
    result = _compute([{"number": 1}], [tmp_path / "none"], [receipts])
    assert result["unattributed_recorded_cost_usd"] == 0.30
    assert result["settled_prs_with_cost_record"] == 0  # receipts carry no PR number
    assert result["total_recorded_cost_usd"] == 0.30
    assert result["receipts_with_cost"] == 1


def test_no_settled_prs_publishes_null_ratio(tmp_path: Path) -> None:
    result = _compute([], [tmp_path / "none"], [tmp_path / "none"])
    assert result["settled_prs_total"] == 0
    assert result["cost_per_settled_pr_usd"] is None
    assert result["coverage_ratio"] is None


# --- Double-count guards ---------------------------------------------------------


def test_same_file_in_two_scanned_dirs_counts_once(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    _write(shared, "r1.json", _routing_record(1, recorded=True, total_usd=1.0))
    # Same directory passed twice: load_json_artifacts dedupes by resolved path.
    result = _compute([{"number": 1}], [shared, shared], [tmp_path / "none"])
    assert result["attributed_recorded_cost_usd"] == 1.0
    assert len(result["cost_sources"]) == 1


def test_routing_record_in_receipts_dir_is_not_double_counted(tmp_path: Path) -> None:
    # One dir scanned as BOTH routing-records dir and receipts dir: the routing
    # pass owns routing_rationale files; the receipt pass must skip them.
    both = tmp_path / "both"
    _write(both, "r1.json", _routing_record(1, recorded=True, total_usd=2.0))
    result = _compute([{"number": 1}], [both], [both])
    assert result["total_recorded_cost_usd"] == 2.0
    assert result["receipts_scanned"] == 0
    assert len(result["cost_sources"]) == 1


def test_cost_sources_re_add_to_total(tmp_path: Path) -> None:
    routing = tmp_path / "routing"
    receipts = tmp_path / "receipts"
    _write(routing, "r1.json", _routing_record(1, recorded=True, total_usd=0.10))
    _write(receipts, "a.json", _receipt(total_cost_usd=0.20))
    result = _compute([{"number": 1}], [routing], [receipts])
    assert (
        round(sum(s["usd"] for s in result["cost_sources"]), 6) == result["total_recorded_cost_usd"]
    )


# --- Honesty: nothing fabricated, gaps disclosed ---------------------------------


def test_malformed_recorded_cost_is_excluded_and_disclosed(tmp_path: Path) -> None:
    routing = tmp_path / "routing"
    path = _write(routing, "bad.json", _routing_record(1, recorded=True, total_usd="not-a-price"))
    result = _compute([{"number": 1}], [routing], [tmp_path / "none"])
    assert result["total_recorded_cost_usd"] == 0.0
    assert result["routing_records_malformed_cost"] == [str(path.resolve())]


def test_negative_cost_is_never_counted(tmp_path: Path) -> None:
    routing = tmp_path / "routing"
    _write(routing, "neg.json", _routing_record(1, recorded=True, total_usd=-5))
    result = _compute([{"number": 1}], [routing], [tmp_path / "none"])
    assert result["total_recorded_cost_usd"] == 0.0


def test_out_of_window_artifacts_are_excluded(tmp_path: Path) -> None:
    routing = tmp_path / "routing"
    receipts = tmp_path / "receipts"
    _write(
        routing,
        "old.json",
        _routing_record(1, generated_at=OUT_WINDOW, recorded=True, total_usd=9.0),
    )
    _write(receipts, "old.json", _receipt(timestamp=OUT_WINDOW, total_cost_usd=9.0))
    result = _compute([{"number": 1}], [routing], [receipts])
    assert result["total_recorded_cost_usd"] == 0.0
    assert result["routing_records_outside_window"] == 1
    assert result["receipts_outside_window"] == 1


def test_receipt_without_timestamp_is_skipped_and_disclosed(tmp_path: Path) -> None:
    receipts = tmp_path / "receipts"
    _write(receipts, "nots.json", _receipt(timestamp=None, total_cost_usd=3.0))
    result = _compute([{"number": 1}], [tmp_path / "none"], [receipts])
    assert result["total_recorded_cost_usd"] == 0.0
    assert result["receipts_skipped_no_timestamp"] == 1


def test_unreadable_receipt_is_counted_not_hidden(tmp_path: Path) -> None:
    receipts = tmp_path / "receipts"
    receipts.mkdir(parents=True)
    (receipts / "junk.json").write_text("{not json")
    result = _compute([{"number": 1}], [tmp_path / "none"], [receipts])
    assert result["receipts_unreadable"] == 1


# --- Publication ------------------------------------------------------------------


def test_render_block_marks_lower_bound_and_coverage(tmp_path: Path) -> None:
    routing = tmp_path / "routing"
    _write(routing, "r1.json", _routing_record(1, recorded=True, total_usd=0.5))
    result = _compute([{"number": 1}, {"number": 2}], [routing], [tmp_path / "none"])
    block = meas.render_cost_block(result, "2026-06-12T00:00:00Z")
    assert "lower bound" in block
    assert "50% coverage" in block
    assert "Coverage caveat" in block


def test_publish_preserves_text_outside_own_region(tmp_path: Path) -> None:
    doc = tmp_path / "LEVERAGE.md"
    doc.write_text(
        "# Leverage & Waste Status\n\n"
        "<!-- leverage-managed:begin -->\nLR table here\n<!-- leverage-managed:end -->\n\n"
        "## Blind-Period Log\n\n- manual entry\n"
    )
    meas.update_status_doc(doc, "BLOCK-ONE")
    text = doc.read_text()
    assert "LR table here" in text
    assert "- manual entry" in text
    assert "BLOCK-ONE" in text
    # Re-publish replaces only its own region (idempotent, no duplication).
    meas.update_status_doc(doc, "BLOCK-TWO")
    text = doc.read_text()
    assert "BLOCK-ONE" not in text
    assert text.count(meas.COST_BEGIN) == 1
    assert "LR table here" in text
    assert "- manual entry" in text


def test_publish_to_empty_doc_creates_header(tmp_path: Path) -> None:
    doc = tmp_path / "LEVERAGE.md"
    meas.update_status_doc(doc, "BLOCK")
    text = doc.read_text()
    assert text.startswith("# Leverage & Waste Status")
    assert "BLOCK" in text
