"""Tests for ``scripts/pr_value_classifier.py``.

The ``gh pr list`` boundary is injected; no test touches the network.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest


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


clf = _load_module("pr_value_classifier.py")

NOW = datetime(2026, 6, 13, 12, 0, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _pr(
    number: int,
    title: str,
    *,
    labels: list[str] | None = None,
    is_draft: bool = False,
    created: datetime | None = None,
) -> dict[str, Any]:
    return {
        "number": number,
        "title": title,
        "labels": [{"name": name} for name in (labels or [])],
        "isDraft": is_draft,
        "createdAt": _iso(created or NOW),
        "updatedAt": _iso(created or NOW),
    }


def _runner(prs: list[dict[str, Any]]):
    return lambda: list(prs)


def test_classify_maintenance_by_label() -> None:
    assert clf.classify_pr(_pr(1, "anything", labels=["codex-automation"])) == "maintenance"


@pytest.mark.parametrize(
    "title",
    [
        "outbox-harvest sweep",
        "Refresh generated module_tiers",
        "regenerate fixtures",
        "resync convoy store",
        "metrics drift report",
        "stale-quorum cleanup",
        "salvage queue drain",
        "repair broken lane",
        "drift sentinel update",
        "preflight gate check",
        "reconcile leases",
        "boss janitor pass",
        "backpressure tuning",
        "quorum gate adjust",
    ],
)
def test_classify_maintenance_by_title(title: str) -> None:
    assert clf.classify_pr(_pr(1, title)) == "maintenance"


@pytest.mark.parametrize(
    "title",
    [
        "Add ODR workflow",
        "crux extraction",
        "calibration tracker",
        "decision receipt store",
        "healthcare vertical",
        "new API endpoint",
        "Python SDK client",
        "fix handler bug",
        "debate orchestrator",
        "consensus proof",
    ],
)
def test_classify_product_by_title(title: str) -> None:
    assert clf.classify_pr(_pr(1, title)) == "product"


def test_classify_product_by_feature_label() -> None:
    assert clf.classify_pr(_pr(1, "opaque title", labels=["feature"])) == "product"


@pytest.mark.parametrize(
    "title",
    [
        "add tests for foo",
        "mypy fixes",
        "lint cleanup",
        "ruff autofix",
        "ci pipeline tweak",
        "packaging metadata",
        "docs update",
        "fix import order",
    ],
)
def test_classify_infra_by_title(title: str) -> None:
    assert clf.classify_pr(_pr(1, title)) == "infra"


def test_classify_unknown() -> None:
    assert clf.classify_pr(_pr(1, "wibble wobble flooble")) == "unknown"


def test_precedence_label_beats_product_title() -> None:
    pr = _pr(1, "feat(api): new endpoint", labels=["codex-automation"])
    assert clf.classify_pr(pr) == "maintenance"


def test_precedence_maintenance_title_beats_product_title() -> None:
    assert clf.classify_pr(_pr(1, "API gate refactor")) == "maintenance"


def test_precedence_product_beats_infra() -> None:
    assert clf.classify_pr(_pr(1, "add test for new API endpoint")) == "product"


def test_maintenance_ratio_and_breach_exit3(capsys: pytest.CaptureFixture[str]) -> None:
    prs = [
        _pr(1, "drift repair", labels=["codex-automation"]),
        _pr(2, "reconcile leases"),
        _pr(3, "new API endpoint"),
    ]
    rc = clf.run_classifier(list_prs=_runner(prs), now=NOW, max_maintenance_ratio=0.5)
    out = json.loads(capsys.readouterr().out)
    assert out["generated_at"] == "2026-06-13T12:00:00Z"
    assert out["total"] == 3
    assert out["by_class"]["maintenance"] == 2
    assert out["by_class"]["product"] == 1
    assert out["maintenance_ratio"] == pytest.approx(0.6667, abs=1e-3)
    assert out["product_ratio"] == pytest.approx(0.3333, abs=1e-3)
    assert rc == clf.EXIT_BREACH


def test_no_breach_at_threshold_boundary(capsys: pytest.CaptureFixture[str]) -> None:
    prs = [_pr(1, "reconcile leases"), _pr(2, "new API endpoint")]
    rc = clf.run_classifier(list_prs=_runner(prs), now=NOW, max_maintenance_ratio=0.5)
    out = json.loads(capsys.readouterr().out)
    assert out["maintenance_ratio"] == 0.5
    assert rc == clf.EXIT_OK


def test_stale_and_drafts_counted(capsys: pytest.CaptureFixture[str]) -> None:
    old = NOW - timedelta(days=10)
    fresh = NOW - timedelta(hours=2)
    prs = [
        _pr(1, "new API endpoint", is_draft=True, created=old),
        _pr(2, "another API endpoint", created=fresh),
        _pr(3, "third API endpoint", created=old),
    ]
    clf.run_classifier(list_prs=_runner(prs), now=NOW, stale_days=4)
    out = json.loads(capsys.readouterr().out)
    assert out["drafts"] == 1
    assert out["stale_count"] == 2


def test_empty_list_zero_ratios_exit0(capsys: pytest.CaptureFixture[str]) -> None:
    rc = clf.run_classifier(list_prs=_runner([]), now=NOW)
    out = json.loads(capsys.readouterr().out)
    assert out["total"] == 0
    assert out["maintenance_ratio"] == 0.0
    assert out["product_ratio"] == 0.0
    assert rc == clf.EXIT_OK


def test_sample_capped_at_five(capsys: pytest.CaptureFixture[str]) -> None:
    prs = [_pr(i, "new API endpoint") for i in range(1, 9)]
    clf.run_classifier(list_prs=_runner(prs), now=NOW)
    out = json.loads(capsys.readouterr().out)
    assert len(out["sample"]["product"]) == 5
    assert out["by_class"]["product"] == 8


def test_summary_one_line(capsys: pytest.CaptureFixture[str]) -> None:
    clf.run_classifier(
        list_prs=_runner([_pr(1, "drift"), _pr(2, "new API endpoint")]), now=NOW, summary=True
    )
    out = capsys.readouterr().out.strip()
    assert "\n" not in out
    assert out.startswith("PR value:")
    assert "maint=1" in out
    assert "product=1" in out


@pytest.mark.parametrize("bad", ["not-a-repo", "owner/", "/name", "a/b/c", "owner name"])
def test_repo_arg_rejects_malformed(bad: str) -> None:
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        clf.repo_arg(bad)


def test_atomic_write_no_temp_left_on_success(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    rc = clf.run_classifier(
        list_prs=_runner([_pr(1, "new API endpoint")]),
        now=NOW,
        out_file=str(out),
    )
    assert rc == clf.EXIT_OK
    assert json.loads(out.read_text(encoding="utf-8"))["total"] == 1
    assert [path.name for path in tmp_path.iterdir() if path.name != "report.json"] == []


def test_atomic_write_failure_leaves_no_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "report.json"

    def boom(src: str, dst: str) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(clf.os, "replace", boom)
    rc = clf.run_classifier(
        list_prs=_runner([_pr(1, "new API endpoint")]),
        now=NOW,
        out_file=str(out),
    )
    assert rc == clf.EXIT_FAILURE
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []


def test_runner_failure_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    def boom() -> list[dict[str, Any]]:
        raise RuntimeError("gh exploded")

    rc = clf.run_classifier(list_prs=boom, now=NOW)
    assert rc == clf.EXIT_FAILURE
    assert "gh exploded" in capsys.readouterr().err


def test_truncation_annotation(capsys: pytest.CaptureFixture[str]) -> None:
    prs = [_pr(i, "new API endpoint") for i in range(1, 4)]
    clf.run_classifier(list_prs=_runner(prs), now=NOW, limit=3)
    out = json.loads(capsys.readouterr().out)
    assert "list_truncated:>=3" in out["annotations"]
