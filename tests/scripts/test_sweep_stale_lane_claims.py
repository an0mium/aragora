"""Tests for scripts/sweep_stale_lane_claims.py."""

from __future__ import annotations

import datetime as dt
import json
import sys
from collections.abc import Generator
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


@pytest.fixture(autouse=True)
def _setup_path() -> Generator[None, None, None]:
    sys.path.insert(0, str(SCRIPTS_DIR))
    yield
    sys.path.remove(str(SCRIPTS_DIR))


def _write_registry(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True))


def test_dry_run_reports_stale_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sweep_stale_lane_claims as mod

    registry = tmp_path / "lanes.json"
    _write_registry(
        registry,
        [
            {
                "lane_id": "old-stale",
                "owner_session": "ghost-001",
                "branch": "missing-branch",
                "status": "active",
                "updated_at": "2026-05-15T12:00:00Z",
            },
            {
                "lane_id": "fresh",
                "owner_session": "live-002",
                "branch": "still-here",
                "status": "active",
                "updated_at": "2026-05-18T17:30:00Z",
            },
        ],
    )

    monkeypatch.setattr(mod, "branch_exists_locally", lambda _r, b, **_k: b == "still-here")
    monkeypatch.setattr(mod, "branch_exists_remotely", lambda _r, b, **_k: b == "still-here")

    now = dt.datetime(2026, 5, 18, 18, 0, tzinfo=dt.UTC)

    report = mod.sweep(
        registry_path=registry,
        repo=tmp_path,
        max_age_hours=24.0,
        apply=False,
        check_branches=True,
        check_remote=True,
        now=now,
    )

    assert report["total_rows"] == 2
    assert report["active_rows"] == 2
    assert report["stale_rows"] == 1
    assert report["stale_records"][0]["lane_id"] == "old-stale"
    assert "branch_missing" in report["stale_records"][0]["reasons"]
    assert "stale_updated_at" in report["stale_records"][0]["reasons"]
    assert report["applied"] is False

    # Dry-run must not touch the file.
    persisted = json.loads(registry.read_text())
    assert persisted[0]["status"] == "active"


def test_apply_expires_stale_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sweep_stale_lane_claims as mod

    registry = tmp_path / "lanes.json"
    _write_registry(
        registry,
        [
            {
                "lane_id": "old-stale",
                "owner_session": "ghost-001",
                "branch": "missing-branch",
                "status": "active",
                "updated_at": "2026-05-15T12:00:00Z",
            },
            {
                "lane_id": "live",
                "owner_session": "live-002",
                "branch": "still-here",
                "status": "active",
                "updated_at": "2026-05-18T17:55:00Z",
            },
        ],
    )

    monkeypatch.setattr(mod, "branch_exists_locally", lambda _r, b, **_k: b == "still-here")
    monkeypatch.setattr(mod, "branch_exists_remotely", lambda _r, b, **_k: b == "still-here")

    now = dt.datetime(2026, 5, 18, 18, 0, tzinfo=dt.UTC)

    report = mod.sweep(
        registry_path=registry,
        repo=tmp_path,
        max_age_hours=24.0,
        apply=True,
        check_branches=True,
        check_remote=True,
        now=now,
    )

    assert report["applied"] is True
    assert report["stale_rows"] == 1

    persisted = json.loads(registry.read_text())
    by_lane = {r["lane_id"]: r for r in persisted}
    assert by_lane["old-stale"]["status"] == "expired"
    assert "stale" in by_lane["old-stale"]["conflict_reason"]
    assert "branch_missing" in by_lane["old-stale"]["conflict_reason"]
    assert by_lane["live"]["status"] == "active"


def test_new_lifecycle_statuses_participate_in_stale_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sweep_stale_lane_claims as mod

    registry = tmp_path / "lanes.json"
    _write_registry(
        registry,
        [
            {
                "lane_id": "old-working",
                "owner_session": "ghost-working",
                "branch": "missing-branch",
                "status": "working",
                "updated_at": "2026-05-15T12:00:00Z",
            },
            {
                "lane_id": "already-superseded",
                "owner_session": "ghost-superseded",
                "branch": "missing-branch",
                "status": "superseded",
                "updated_at": "2026-05-15T12:00:00Z",
            },
        ],
    )

    monkeypatch.setattr(mod, "branch_exists_locally", lambda *_a, **_k: False)
    monkeypatch.setattr(mod, "branch_exists_remotely", lambda *_a, **_k: False)

    report = mod.sweep(
        registry_path=registry,
        repo=tmp_path,
        max_age_hours=24.0,
        apply=False,
        check_branches=True,
        check_remote=True,
        now=dt.datetime(2026, 5, 18, 18, 0, tzinfo=dt.UTC),
    )

    assert report["active_rows"] == 1
    assert report["stale_rows"] == 1
    assert report["stale_records"][0]["lane_id"] == "old-working"


def test_worktree_missing_signals_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sweep_stale_lane_claims as mod

    registry = tmp_path / "lanes.json"
    missing_wt = tmp_path / "nonexistent"
    _write_registry(
        registry,
        [
            {
                "lane_id": "wt-gone",
                "owner_session": "ghost",
                "branch": "live-branch",
                "worktree": str(missing_wt),
                "status": "active",
                "updated_at": "2026-05-18T17:55:00Z",
            },
        ],
    )

    monkeypatch.setattr(mod, "branch_exists_locally", lambda *_a, **_k: True)
    monkeypatch.setattr(mod, "branch_exists_remotely", lambda *_a, **_k: True)

    now = dt.datetime(2026, 5, 18, 18, 0, tzinfo=dt.UTC)
    report = mod.sweep(
        registry_path=registry,
        repo=tmp_path,
        max_age_hours=24.0,
        apply=False,
        check_branches=True,
        check_remote=True,
        now=now,
    )

    assert report["stale_rows"] == 1
    assert report["stale_records"][0]["reasons"] == ["worktree_missing"]


def test_skip_branch_check_ignores_branch_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sweep_stale_lane_claims as mod

    registry = tmp_path / "lanes.json"
    _write_registry(
        registry,
        [
            {
                "lane_id": "stale-old",
                "owner_session": "ghost",
                "branch": "missing-branch",
                "status": "active",
                "updated_at": "2026-05-18T17:55:00Z",
            },
        ],
    )

    monkeypatch.setattr(mod, "branch_exists_locally", lambda *_a, **_k: False)
    monkeypatch.setattr(mod, "branch_exists_remotely", lambda *_a, **_k: False)

    now = dt.datetime(2026, 5, 18, 18, 0, tzinfo=dt.UTC)
    report = mod.sweep(
        registry_path=registry,
        repo=tmp_path,
        max_age_hours=24.0,
        apply=False,
        check_branches=False,
        check_remote=False,
        now=now,
    )

    assert report["stale_rows"] == 0


def test_non_active_rows_are_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sweep_stale_lane_claims as mod

    registry = tmp_path / "lanes.json"
    _write_registry(
        registry,
        [
            {
                "lane_id": "released-row",
                "owner_session": "anyone",
                "branch": "missing",
                "status": "released",
                "updated_at": "2026-05-01T00:00:00Z",
            },
            {
                "lane_id": "completed-row",
                "owner_session": "anyone",
                "branch": "missing",
                "status": "completed",
                "updated_at": "2026-05-01T00:00:00Z",
            },
        ],
    )

    monkeypatch.setattr(mod, "branch_exists_locally", lambda *_a, **_k: False)
    monkeypatch.setattr(mod, "branch_exists_remotely", lambda *_a, **_k: False)

    report = mod.sweep(
        registry_path=registry,
        repo=tmp_path,
        max_age_hours=1.0,
        apply=True,
        check_branches=True,
        check_remote=True,
        now=dt.datetime(2026, 5, 18, 18, 0, tzinfo=dt.UTC),
    )

    assert report["active_rows"] == 0
    assert report["stale_rows"] == 0


def test_build_parser_defaults() -> None:
    import sweep_stale_lane_claims as mod

    parser = mod.build_parser()
    args = parser.parse_args([])
    assert args.apply is False
    assert args.dry_run is False
    assert args.max_active_age_hours == 24.0
    assert args.branch_grace_hours == 1.0
    assert args.skip_branch_check is False
    assert args.skip_remote_check is False


def test_dry_run_flag_is_accepted_as_noop_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--dry-run is an explicit no-op alias for the default behavior.

    Used by Makefile/cron invocations so the intent is self-documenting.
    It must NOT change the canonical --apply semantics (still default off).
    """
    import sweep_stale_lane_claims as mod

    parser = mod.build_parser()
    args = parser.parse_args(["--dry-run"])
    assert args.dry_run is True
    assert args.apply is False


def test_dry_run_and_apply_are_mutually_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Combining --dry-run and --apply must fail closed (parser error)."""
    import sweep_stale_lane_claims as mod

    registry = tmp_path / "lanes.json"
    _write_registry(registry, [])

    with pytest.raises(SystemExit) as exc:
        mod.main(
            [
                "--registry-path",
                str(registry),
                "--repo",
                str(tmp_path),
                "--dry-run",
                "--apply",
            ]
        )
    assert exc.value.code != 0
    captured = capsys.readouterr()
    assert "mutually exclusive" in captured.err


def test_dry_run_flag_does_not_write_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An invocation with --dry-run must leave the registry untouched even when
    stale rows are detected (sanity guard against future regression where the
    flag accidentally toggles apply semantics)."""
    import sweep_stale_lane_claims as mod

    registry = tmp_path / "lanes.json"
    _write_registry(
        registry,
        [
            {
                "lane_id": "old-stale",
                "owner_session": "ghost-001",
                "branch": "missing-branch",
                "status": "active",
                "updated_at": "2026-05-15T12:00:00Z",
            },
        ],
    )

    monkeypatch.setattr(mod, "branch_exists_locally", lambda *_a, **_k: False)
    monkeypatch.setattr(mod, "branch_exists_remotely", lambda *_a, **_k: False)

    rc = mod.main(
        [
            "--registry-path",
            str(registry),
            "--repo",
            str(tmp_path),
            "--dry-run",
        ]
    )
    assert rc == 0
    persisted = json.loads(registry.read_text())
    assert persisted[0]["status"] == "active"


def test_branch_grace_period_protects_fresh_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sweep_stale_lane_claims as mod

    registry = tmp_path / "lanes.json"
    _write_registry(
        registry,
        [
            {
                "lane_id": "freshly-claimed",
                "owner_session": "live-001",
                "branch": "droid/just-claimed-branch-not-yet-pushed",
                "status": "active",
                "updated_at": "2026-05-18T17:55:00Z",
            },
        ],
    )

    monkeypatch.setattr(mod, "branch_exists_locally", lambda *_a, **_k: False)
    monkeypatch.setattr(mod, "branch_exists_remotely", lambda *_a, **_k: False)

    now = dt.datetime(2026, 5, 18, 18, 0, tzinfo=dt.UTC)
    report = mod.sweep(
        registry_path=registry,
        repo=tmp_path,
        max_age_hours=24.0,
        apply=False,
        check_branches=True,
        check_remote=True,
        now=now,
        branch_grace_hours=1.0,
    )

    assert report["stale_rows"] == 0


def test_resolve_registry_path_prefers_repo(tmp_path: Path) -> None:
    import sweep_stale_lane_claims as mod

    (tmp_path / ".aragora" / "agent-bridge").mkdir(parents=True)
    resolved = mod.resolve_registry_path(repo_root=tmp_path, explicit=None)
    assert resolved == tmp_path / ".aragora" / "agent-bridge" / "lanes.json"


def test_resolve_registry_path_respects_explicit(tmp_path: Path) -> None:
    import sweep_stale_lane_claims as mod

    explicit = tmp_path / "custom-lanes.json"
    resolved = mod.resolve_registry_path(repo_root=tmp_path, explicit=explicit)
    assert resolved == explicit
