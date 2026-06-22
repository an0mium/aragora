"""Tests for ``scripts/tier4_merge_train.py`` — Tier-4 contention serializer.

Pure-core, fixture-driven: the lane decision is exercised against explicit
open-PR lists, so no ``gh`` subprocess is ever invoked. A drift guard asserts
the vendored serialized-surface list stays in sync with the canonical
``review_queue.TIER_4_PREFIXES`` (skipped when that import's heavy dependency
closure is unavailable, e.g. in a minimal sandbox).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_module() -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / "tier4_merge_train.py"
    spec = importlib.util.spec_from_file_location("tier4_merge_train_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train = _load_module()


def _pr(number: int, files: list[str], *, created: str = "", title: str = "") -> dict[str, Any]:
    return {
        "number": number,
        "title": title or f"PR {number}",
        "createdAt": created,
        "files": [{"path": f} for f in files],
    }


# ---------------------------------------------------------------------------
# Path classification
# ---------------------------------------------------------------------------


def test_matches_exact_file_surface() -> None:
    assert (
        train.matches_serialized_path("scripts/settle_tier4_pr.py") == "scripts/settle_tier4_pr.py"
    )


def test_matches_directory_prefix_surface() -> None:
    assert (
        train.matches_serialized_path(".github/workflows/aragora-merge-quorum.yml")
        == ".github/workflows/"
    )


def test_non_serialized_path_returns_none() -> None:
    assert train.matches_serialized_path("scripts/run_boss_cycle.sh") is None
    assert train.matches_serialized_path("aragora/debate/orchestrator.py") is None


def test_serialized_paths_for_dedupes_by_surface() -> None:
    touched = train.serialized_paths_for(
        ["scripts/settle_tier4_pr.py", "tests/scripts/test_settle_tier4_pr.py", "README.md"]
    )
    assert touched == {"scripts/settle_tier4_pr.py"}


# ---------------------------------------------------------------------------
# Lane decision
# ---------------------------------------------------------------------------


def test_empty_lane_allows_candidate() -> None:
    result = train.evaluate_merge_train(
        ["scripts/settle_tier4_pr.py"],
        open_prs=[],
        cap=1,
    )
    assert result["ok"] is True
    assert result["decision"] == train.ALLOW
    assert result["blocking_prs"] == []


def test_full_lane_queues_candidate() -> None:
    open_prs = [_pr(8405, ["scripts/settle_tier4_pr.py"], created="2026-06-14T04:00:00Z")]
    result = train.evaluate_merge_train(
        ["scripts/settle_tier4_pr.py", "tests/scripts/test_settle_tier4_pr.py"],
        open_prs=open_prs,
        cap=1,
    )
    assert result["ok"] is False
    assert result["decision"] == train.QUEUE
    assert result["blocking_prs"] == [8405]
    assert result["contended_surfaces"]["scripts/settle_tier4_pr.py"]["head_pr"] == 8405
    assert result["contended_surfaces"]["scripts/settle_tier4_pr.py"]["lane_full"] is True


def test_cap_two_allows_second_pr() -> None:
    open_prs = [_pr(8405, ["scripts/settle_tier4_pr.py"], created="2026-06-14T04:00:00Z")]
    result = train.evaluate_merge_train(
        ["scripts/settle_tier4_pr.py"],
        open_prs=open_prs,
        cap=2,
    )
    assert result["ok"] is True


def test_candidate_excluded_from_its_own_lane_count() -> None:
    # Re-evaluating an already-open PR must not block on itself.
    open_prs = [_pr(8408, ["scripts/settle_tier4_pr.py"], created="2026-06-14T05:00:00Z")]
    result = train.evaluate_merge_train(
        ["scripts/settle_tier4_pr.py"],
        open_prs=open_prs,
        cap=1,
        candidate_pr=8408,
    )
    assert result["ok"] is True
    assert result["blocking_prs"] == []


def test_non_serialized_candidate_always_allowed() -> None:
    open_prs = [_pr(8405, ["scripts/settle_tier4_pr.py"])]
    result = train.evaluate_merge_train(
        ["aragora/debate/orchestrator.py", "README.md"],
        open_prs=open_prs,
        cap=1,
    )
    assert result["ok"] is True
    assert result["candidate_surfaces"] == []
    assert "no serialized Tier-4 surface" in result["reason"]


def test_distinct_surfaces_do_not_cross_block() -> None:
    # An open PR on settle_one_pr.py must not block a candidate on review_queue.py.
    open_prs = [_pr(8396, ["scripts/settle_one_pr.py"])]
    result = train.evaluate_merge_train(
        ["aragora/cli/commands/review_queue.py"],
        open_prs=open_prs,
        cap=1,
    )
    assert result["ok"] is True


def test_train_ordered_oldest_first_then_number() -> None:
    open_prs = [
        _pr(8412, ["scripts/settle_tier4_pr.py"], created="2026-06-14T07:39:00Z"),
        _pr(8405, ["scripts/settle_tier4_pr.py"], created="2026-06-14T04:56:00Z"),
        _pr(8408, ["scripts/settle_tier4_pr.py"], created="2026-06-14T05:32:00Z"),
    ]
    lanes = train.build_train(open_prs)
    order = [entry["number"] for entry in lanes["scripts/settle_tier4_pr.py"]]
    assert order == [8405, 8408, 8412]


def test_blocking_prs_unions_across_surfaces() -> None:
    open_prs = [
        _pr(8405, ["scripts/settle_tier4_pr.py"], created="2026-06-14T04:00:00Z"),
        _pr(8396, ["scripts/settle_one_pr.py"], created="2026-06-14T03:00:00Z"),
    ]
    result = train.evaluate_merge_train(
        ["scripts/settle_tier4_pr.py", "scripts/settle_one_pr.py"],
        open_prs=open_prs,
        cap=1,
    )
    assert result["ok"] is False
    assert result["blocking_prs"] == [8396, 8405]


# ---------------------------------------------------------------------------
# CLI exit codes (pure path via injected open-PR JSON)
# ---------------------------------------------------------------------------


def test_cli_check_allow_exit_zero(tmp_path: Path, capsys: Any) -> None:
    prs = tmp_path / "open.json"
    prs.write_text("[]", encoding="utf-8")
    code = train.main(
        [
            "--check",
            "--changed-files",
            "scripts/settle_tier4_pr.py",
            "--open-prs-json",
            str(prs),
            "--json",
        ]
    )
    assert code == 0


def test_cli_check_queue_exit_two(tmp_path: Path) -> None:
    prs = tmp_path / "open.json"
    prs.write_text(
        '[{"number": 8405, "title": "x", "createdAt": "2026-06-14T04:00:00Z",'
        ' "files": [{"path": "scripts/settle_tier4_pr.py"}]}]',
        encoding="utf-8",
    )
    code = train.main(
        [
            "--check",
            "--changed-files",
            "scripts/settle_tier4_pr.py",
            "--open-prs-json",
            str(prs),
            "--json",
        ]
    )
    assert code == 2


def test_cli_requires_exactly_one_mode(tmp_path: Path) -> None:
    prs = tmp_path / "open.json"
    prs.write_text("[]", encoding="utf-8")
    # neither --check nor --status
    assert train.main(["--open-prs-json", str(prs)]) == 1
    # both
    assert train.main(["--check", "--status", "--open-prs-json", str(prs)]) == 1


# ---------------------------------------------------------------------------
# Drift guard: vendored list must match the canonical classifier
# ---------------------------------------------------------------------------


def test_serialized_prefixes_match_canonical_tier4() -> None:
    review_queue = pytest.importorskip(
        "aragora.cli.commands.review_queue",
        reason="review_queue import closure unavailable in this environment",
    )
    assert set(train.SERIALIZED_TIER4_PREFIXES) == set(review_queue.TIER_4_PREFIXES), (
        "scripts/tier4_merge_train.py SERIALIZED_TIER4_PREFIXES has drifted from "
        "review_queue.TIER_4_PREFIXES — re-sync the vendored copy."
    )
