"""Tests for ``scripts/sweep_lane_contact_methods.py`` (R05 — reach plan)."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sweep_module = importlib.import_module("sweep_lane_contact_methods")


@pytest.fixture()
def tmp_registry(tmp_path: Path) -> Path:
    p = tmp_path / "lanes.json"
    p.write_text("[]", encoding="utf-8")
    return p


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_empty_registry_returns_zero_counts(tmp_registry: Path) -> None:
    report = sweep_module.sweep(
        registry_path=tmp_registry, status_filter=None, tmux_session="aragora"
    )
    assert report["counts"] == {"total": 0, "considered": 0, "inferred": 0, "skipped": 0}
    assert report["results"] == []


def test_row_with_existing_contact_method_is_not_overwritten(
    tmp_registry: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_rows(
        tmp_registry,
        [
            {
                "lane_id": "L1",
                "owner_session": "claude-A",
                "status": "active",
                "contact_method": "tmux:claude-A",
            }
        ],
    )
    monkeypatch.setattr(sweep_module, "list_tmux_window_names", lambda *, session: ["claude-A"])
    report = sweep_module.sweep(
        registry_path=tmp_registry, status_filter=None, tmux_session="aragora"
    )
    entry = report["results"][0]
    assert entry["existing_contact_method"] == "tmux:claude-A"
    assert entry["inferred_contact_method"] is None
    assert entry["reason"] == "already-set"


def test_owner_equals_window_infers_tmux(
    tmp_registry: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_rows(
        tmp_registry, [{"lane_id": "L1", "owner_session": "claude-p52", "status": "active"}]
    )
    monkeypatch.setattr(
        sweep_module, "list_tmux_window_names", lambda *, session: ["claude-p52", "_control"]
    )
    report = sweep_module.sweep(
        registry_path=tmp_registry, status_filter=None, tmux_session="aragora"
    )
    entry = report["results"][0]
    assert entry["inferred_contact_method"] == "tmux:claude-p52"
    assert entry["reason"] == "owner-equals-window"
    assert report["counts"]["inferred"] == 1


def test_substring_match_infers_window_name(
    tmp_registry: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_rows(
        tmp_registry,
        [{"lane_id": "L1", "owner_session": "claude-79AAF84B-extra-suffix", "status": "active"}],
    )
    monkeypatch.setattr(sweep_module, "list_tmux_window_names", lambda *, session: ["claude-79"])
    report = sweep_module.sweep(
        registry_path=tmp_registry, status_filter=None, tmux_session="aragora"
    )
    entry = report["results"][0]
    assert entry["inferred_contact_method"] == "tmux:claude-79"
    assert "substring" in entry["reason"]


def test_no_match_leaves_unset(tmp_registry: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_rows(
        tmp_registry, [{"lane_id": "L1", "owner_session": "droid-9999X", "status": "active"}]
    )
    monkeypatch.setattr(sweep_module, "list_tmux_window_names", lambda *, session: ["claude-p52"])
    report = sweep_module.sweep(
        registry_path=tmp_registry, status_filter=None, tmux_session="aragora"
    )
    entry = report["results"][0]
    assert entry["inferred_contact_method"] is None
    assert entry["reason"] == "no-live-match"


def test_status_filter_skips_completed(tmp_registry: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_rows(
        tmp_registry,
        [
            {"lane_id": "L1", "owner_session": "claude-p52", "status": "active"},
            {"lane_id": "L2", "owner_session": "claude-p53", "status": "completed"},
        ],
    )
    monkeypatch.setattr(
        sweep_module, "list_tmux_window_names", lambda *, session: ["claude-p52", "claude-p53"]
    )
    report = sweep_module.sweep(
        registry_path=tmp_registry, status_filter={"active"}, tmux_session="aragora"
    )
    assert report["counts"]["considered"] == 1
    assert report["counts"]["skipped"] == 1
    assert len(report["results"]) == 1
    assert report["results"][0]["lane_id"] == "L1"


def test_no_owner_session_returns_no_owner_reason(
    tmp_registry: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_rows(tmp_registry, [{"lane_id": "L1", "owner_session": "", "status": "active"}])
    monkeypatch.setattr(sweep_module, "list_tmux_window_names", lambda *, session: ["any"])
    report = sweep_module.sweep(
        registry_path=tmp_registry, status_filter=None, tmux_session="aragora"
    )
    assert report["results"][0]["reason"] == "no-owner"


def test_tmux_unavailable_returns_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("tmux not on PATH")

    monkeypatch.setattr(sweep_module.subprocess, "run", fake_run)
    assert sweep_module.list_tmux_window_names() == []


def test_tmux_nonzero_returncode_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args, **kwargs):
        return sweep_module.subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="no session"
        )

    monkeypatch.setattr(sweep_module.subprocess, "run", fake_run)
    assert sweep_module.list_tmux_window_names() == []


def test_main_dry_run_outputs_json(tmp_registry: Path, capsys: pytest.CaptureFixture) -> None:
    _write_rows(tmp_registry, [])
    rc = sweep_module.main(["--registry-path", str(tmp_registry), "--pretty"])
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["mode"] == "dry-run"
    assert data["counts"]["total"] == 0


def test_main_unknown_status_errors(tmp_registry: Path) -> None:
    rc = sweep_module.main(["--registry-path", str(tmp_registry), "--status", "bogus", "--pretty"])
    assert rc == 1


def test_main_active_only_convenience(
    tmp_registry: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_rows(
        tmp_registry,
        [
            {"lane_id": "L1", "owner_session": "a", "status": "active"},
            {"lane_id": "L2", "owner_session": "b", "status": "completed"},
            {"lane_id": "L3", "owner_session": "c", "status": "running"},
        ],
    )
    monkeypatch.setattr(sweep_module, "list_tmux_window_names", lambda *, session: [])
    rc = sweep_module.main(["--registry-path", str(tmp_registry), "--active-only"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["counts"]["considered"] == 2
    assert data["counts"]["skipped"] == 1
