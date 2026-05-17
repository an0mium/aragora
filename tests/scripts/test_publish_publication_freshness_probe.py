"""Tests for ``scripts/publish_publication_freshness_probe.py``.

The publisher invokes other scripts via ``subprocess.run``. All tests
inject a fake ``runner`` callable to avoid spawning real processes,
so the suite is fast and deterministic.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import publish_publication_freshness_probe as publisher  # noqa: E402


class _Proc:
    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _fake_runner(proc: _Proc) -> Any:
    def runner(*_args: Any, **_kwargs: Any) -> _Proc:
        return proc

    return runner


def _make_script(repo_root: Path, name: str) -> Path:
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_dir / name
    script_path.write_text("# stub", encoding="utf-8")
    return script_path


def _now() -> dt.datetime:
    return dt.datetime(2026, 5, 17, 6, 0, 0, tzinfo=dt.UTC)


def test_run_canonical_metrics_probe_handles_missing_script(tmp_path: Path) -> None:
    out = publisher.run_canonical_metrics_probe(repo_root=tmp_path)
    assert out["available"] is False


def test_run_canonical_metrics_probe_parses_results(tmp_path: Path) -> None:
    _make_script(tmp_path, "check_canonical_metrics.py")
    payload = {
        "manifest_id": "canonical_metrics",
        "results": [
            {"claim_id": "a", "status": "pass", "observed": "1", "claimed": "1"},
            {"claim_id": "b", "status": "warn", "observed": "10", "claimed": "1"},
            {"claim_id": "c", "status": "fail", "observed": "0", "claimed": "1"},
        ],
    }
    out = publisher.run_canonical_metrics_probe(
        repo_root=tmp_path,
        runner=_fake_runner(_Proc(stdout=json.dumps(payload), returncode=0)),
    )
    assert out["available"] is True
    assert out["summary"] == {"pass": 1, "warn": 1, "fail": 1, "skip": 0}
    assert out["drift_count"] == 2
    assert {r["claim_id"] for r in out["drift_records"]} == {"b", "c"}


def test_run_reconcile_status_docs_probe_parses_findings(tmp_path: Path) -> None:
    _make_script(tmp_path, "reconcile_status_docs.py")
    payload = {
        "generated": "2026-05-17T01:21:27Z",
        "findings": [
            {"severity": "info", "source": "x", "message": "ok"},
            {"severity": "warning", "source": "y", "message": "aging"},
            {"severity": "critical", "source": "z", "message": "broken"},
        ],
        "summary": {"critical": 1, "warning": 1, "info": 1, "total": 3},
    }
    out = publisher.run_reconcile_status_docs_probe(
        repo_root=tmp_path,
        runner=_fake_runner(_Proc(stdout=json.dumps(payload), returncode=0)),
    )
    assert out["available"] is True
    assert out["drift_count"] == 2
    assert {f["source"] for f in out["drift_records"]} == {"y", "z"}


def test_scan_benchmark_truth_artifacts_flags_stale(tmp_path: Path) -> None:
    truth_root = tmp_path / "tracked" / "benchmark_truth_artifacts"
    fresh_dir = truth_root / "tw-fresh"
    stale_dir = truth_root / "tw-stale"
    fresh_dir.mkdir(parents=True)
    stale_dir.mkdir(parents=True)
    now = _now()
    (fresh_dir / "latest.json").write_text(
        json.dumps(
            {
                "generated_at": (now - dt.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "coverage": {"status": "complete"},
            }
        ),
        encoding="utf-8",
    )
    (stale_dir / "latest.json").write_text(
        json.dumps(
            {
                "generated_at": (now - dt.timedelta(hours=72)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "coverage": {"status": "complete"},
            }
        ),
        encoding="utf-8",
    )
    out = publisher.scan_benchmark_truth_artifacts(truth_root=truth_root, stale_hours=48.0, now=now)
    assert out["available"] is True
    assert {row["corpus_id"] for row in out["corpora"]} == {"tw-fresh", "tw-stale"}
    fresh = next(r for r in out["corpora"] if r["corpus_id"] == "tw-fresh")
    stale = next(r for r in out["corpora"] if r["corpus_id"] == "tw-stale")
    assert fresh["is_stale"] is False
    assert stale["is_stale"] is True
    assert out["drift_count"] == 1


def test_scan_benchmark_truth_artifacts_missing_root(tmp_path: Path) -> None:
    out = publisher.scan_benchmark_truth_artifacts(
        truth_root=tmp_path / "absent", stale_hours=48.0, now=_now()
    )
    assert out["available"] is False
    assert out["drift_count"] == 0


def test_build_published_report_verdict_fresh(tmp_path: Path) -> None:
    report = publisher.build_published_report(
        repo_root=tmp_path,
        truth_root=tmp_path / "absent",
        stale_hours=48.0,
        now=_now(),
    )
    assert report["schema_version"] == publisher.SCHEMA_VERSION
    assert report["generated_at"] == "2026-05-17T06:00:00Z"
    assert report["verdict"] == "fresh"
    assert report["total_drift"] == 0
    assert set(report["sources"].keys()) == {
        "canonical_metrics",
        "reconcile_status_docs",
        "benchmark_truth_artifacts",
    }


def test_build_published_report_drift(tmp_path: Path) -> None:
    _make_script(tmp_path, "check_canonical_metrics.py")
    _make_script(tmp_path, "reconcile_status_docs.py")
    truth_root = tmp_path / "tracked" / "benchmark_truth_artifacts"
    stale_dir = truth_root / "tw-stale"
    stale_dir.mkdir(parents=True)
    (stale_dir / "latest.json").write_text(
        json.dumps(
            {
                "generated_at": (_now() - dt.timedelta(hours=72)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "coverage": {"status": "complete"},
            }
        ),
        encoding="utf-8",
    )
    canonical_payload = {
        "results": [
            {"claim_id": "a", "status": "fail", "observed": "0", "claimed": "1"},
            {"claim_id": "b", "status": "fail", "observed": "0", "claimed": "1"},
            {"claim_id": "c", "status": "fail", "observed": "0", "claimed": "1"},
        ]
    }
    reconcile_payload = {
        "findings": [
            {"severity": "warning", "source": "x", "message": "m"},
            {"severity": "critical", "source": "y", "message": "m"},
        ],
        "summary": {"critical": 1, "warning": 1, "info": 0, "total": 2},
    }
    report = publisher.build_published_report(
        repo_root=tmp_path,
        truth_root=truth_root,
        stale_hours=48.0,
        now=_now(),
        canonical_runner=_fake_runner(_Proc(stdout=json.dumps(canonical_payload))),
        reconcile_runner=_fake_runner(_Proc(stdout=json.dumps(reconcile_payload))),
    )
    assert report["verdict"] == "drift"
    assert report["total_drift"] == 6


def test_publish_report_bundle_writes_latest_and_snapshot(tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    paths = publisher.publish_report_bundle(
        {
            "schema_version": 1,
            "generated_at": "2026-05-17T06:00:00Z",
            "verdict": "fresh",
        },
        out_root=out_root,
    )
    assert paths["latest"].exists()
    assert paths["snapshot"].exists()
    payload = json.loads(paths["latest"].read_text())
    assert payload["generated_at"] == "2026-05-17T06:00:00Z"
    assert paths["snapshot"].name.startswith("probe-")


def test_render_status_markdown_emits_sections() -> None:
    sample_report = {
        "generated_at": "2026-05-17T06:00:00Z",
        "verdict": "drift",
        "total_drift": 4,
        "stale_threshold_hours": 48.0,
        "sources": {
            "canonical_metrics": {
                "available": True,
                "summary": {"pass": 6, "warn": 1, "fail": 2, "skip": 0},
                "drift_count": 3,
                "drift_records": [
                    {
                        "status": "fail",
                        "claim_id": "x",
                        "observed": "0",
                        "claimed": "1",
                    }
                ],
            },
            "reconcile_status_docs": {
                "available": True,
                "summary": {"critical": 1, "warning": 1, "info": 0, "total": 2},
                "drift_count": 1,
                "drift_records": [
                    {"severity": "warning", "source": "ROADMAP.md", "message": "old"}
                ],
            },
            "benchmark_truth_artifacts": {
                "available": True,
                "stale_threshold_hours": 48.0,
                "drift_count": 0,
                "corpora": [
                    {
                        "corpus_id": "tw-01",
                        "age_hours": 10.0,
                        "coverage_status": "complete",
                        "is_stale": False,
                    }
                ],
            },
        },
    }
    md = publisher.render_status_markdown(sample_report)
    assert "# Publication Freshness Probe Status" in md
    assert "Verdict: **drift**" in md
    assert "## Canonical Metrics" in md
    assert "## Status-doc Reconciliation" in md
    assert "## Benchmark Truth Artifacts" in md
    assert "ROADMAP.md" in md


def test_main_dry_run_writes_nothing(tmp_path: Path, capsys: Any) -> None:
    out_root = tmp_path / "out"
    status_md = tmp_path / "status.md"
    rc = publisher.main(
        [
            "--repo-root",
            str(tmp_path),
            "--out-root",
            str(out_root),
            "--status-md",
            str(status_md),
            "--truth-root",
            str(tmp_path / "absent"),
            "--json",
            "--dry-run",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["verdict"] == "fresh"
    assert not out_root.exists()
    assert not status_md.exists()


def test_main_default_publishes_to_out_root(tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    status_md = tmp_path / "status.md"
    rc = publisher.main(
        [
            "--repo-root",
            str(tmp_path),
            "--out-root",
            str(out_root),
            "--status-md",
            str(status_md),
            "--truth-root",
            str(tmp_path / "absent"),
            "--render-markdown",
        ]
    )
    assert rc == 0
    assert (out_root / "latest.json").exists()
    snapshots = list(out_root.glob("probe-*.json"))
    assert snapshots, "expected at least one timestamped snapshot"
    assert status_md.exists()
