"""Focused tests for `scripts/publish_worktree_value_inventory.py`."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import publish_worktree_value_inventory as publisher  # noqa: E402


def _inventory_payload(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "base": "origin/main",
        "base_sha": "deadbeef" * 5,
        "roots": ["/path/to/.worktrees/codex-auto"],
        "candidates": candidates,
    }


def _candidate(
    *,
    path: str = "/.worktrees/codex-auto/abc",
    branch: str | None = "feature/foo",
    classification: str = "unique_unharvested",
    decision: str = "harvest_candidate",
    active_session: bool = False,
    registered_worktree: bool = True,
    dirty: bool = False,
    ahead: int | None = 5,
    behind: int | None = 1,
    open_prs: list[str] | None = None,
    mtime: str = "2026-05-16T00:00:00+00:00",
) -> dict[str, Any]:
    return {
        "path": path,
        "candidate_id": "x",
        "classification": classification,
        "cleanup_candidate": decision == "cleanup_candidate",
        "decision": decision,
        "active_session": active_session,
        "git": {
            "branch": branch,
            "registered_worktree": registered_worktree,
            "dirty": dirty,
            "ahead": ahead,
            "behind": behind,
        },
        "links": {
            "open_prs": list(open_prs or []),
            "outbox_files": [],
            "receipt_files": [],
        },
        "lock_files": [],
        "mtime": mtime,
    }


def test_load_inventory_payload_rejects_non_dict(tmp_path: Path) -> None:
    path = tmp_path / "inv.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError):
        publisher.load_inventory_payload(path)


def test_load_inventory_payload_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        publisher.load_inventory_payload(tmp_path / "missing.json")


def test_summarize_candidates_counts_three_decision_buckets() -> None:
    summary = publisher._summarize_candidates(
        [
            _candidate(decision="harvest_candidate", classification="unique_unharvested"),
            _candidate(decision="harvest_candidate", classification="unique_unharvested"),
            _candidate(
                decision="cleanup_candidate",
                classification="patch_equivalent_or_merged",
                ahead=0,
                behind=10,
            ),
            _candidate(
                decision="preserve",
                classification="active_or_dirty",
                active_session=True,
                dirty=True,
                open_prs=["https://github.com/synaptent/aragora/pull/9999"],
            ),
        ]
    )

    assert summary["total_candidates"] == 4
    assert summary["active_sessions"] == 1
    assert summary["dirty_count"] == 1
    assert summary["candidates_with_open_pr"] == 1
    assert summary["registered_worktrees"] == 4
    assert summary["decisions"] == {
        "harvest_candidate": 2,
        "cleanup_candidate": 1,
        "preserve": 1,
    }
    assert summary["classifications"] == {
        "unique_unharvested": 2,
        "patch_equivalent_or_merged": 1,
        "active_or_dirty": 1,
    }
    assert len(summary["harvest_candidates"]) == 2
    assert len(summary["cleanup_candidates"]) == 1
    assert len(summary["preserves"]) == 1


def test_build_published_report_writes_canonical_keys(tmp_path: Path) -> None:
    inv = tmp_path / "inv.json"
    inv.write_text(
        json.dumps(_inventory_payload([_candidate(branch="feature/x")])),
        encoding="utf-8",
    )

    payload = publisher.build_published_report(
        inventory_path=inv, generated_at="2026-05-17T05:00:00Z"
    )

    assert payload["schema_version"] == publisher.SCHEMA_VERSION
    assert payload["generated_at"] == "2026-05-17T05:00:00Z"
    assert payload["base"] == "origin/main"
    assert payload["base_sha"].startswith("deadbeef")
    assert payload["roots"] == ["/path/to/.worktrees/codex-auto"]
    assert payload["summary"]["total_candidates"] == 1
    # The full inventory is preserved verbatim for downstream diffability.
    assert payload["inventory"]["candidates"][0]["git"]["branch"] == "feature/x"


def test_build_published_report_accepts_previous_published_report(tmp_path: Path) -> None:
    inv = tmp_path / "inv.json"
    inv.write_text(
        json.dumps(_inventory_payload([_candidate(branch="feature/x")])),
        encoding="utf-8",
    )
    first_payload = publisher.build_published_report(
        inventory_path=inv, generated_at="2026-05-17T05:00:00Z"
    )
    latest = tmp_path / "latest.json"
    latest.write_text(json.dumps(first_payload), encoding="utf-8")

    republished = publisher.build_published_report(
        inventory_path=latest,
        generated_at="2026-05-17T06:00:00Z",
    )

    assert republished["source_inventory_path"].endswith("latest.json")
    assert republished["summary"]["total_candidates"] == 1
    assert republished["inventory"]["candidates"][0]["git"]["branch"] == "feature/x"


def test_resolve_paths_round_trip(tmp_path: Path) -> None:
    ts = "2026-05-17T05:00:00Z"
    pub_path = publisher.resolve_published_report_path(publish_dir=tmp_path, generated_at=ts)
    latest_path = publisher.resolve_latest_report_path(publish_dir=tmp_path)
    assert pub_path.name == "worktree-value-inventory-20260517T050000Z.json"
    assert latest_path.name == "latest.json"


def test_publish_report_bundle_writes_two_artifacts(tmp_path: Path) -> None:
    inv = tmp_path / "inv.json"
    inv.write_text(
        json.dumps(_inventory_payload([_candidate()])),
        encoding="utf-8",
    )
    payload = publisher.build_published_report(
        inventory_path=inv, generated_at="2026-05-17T05:00:00Z"
    )
    published = publisher.publish_report_bundle(
        publish_dir=tmp_path / "publish",
        payload=payload,
    )

    assert published["timestamped"].exists()
    assert published["latest"].exists()
    assert published["timestamped"].read_text(encoding="utf-8") == published["latest"].read_text(
        encoding="utf-8"
    )
    # latest.json must be valid JSON matching schema
    on_disk = json.loads(published["latest"].read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == publisher.SCHEMA_VERSION
    assert on_disk["summary"]["total_candidates"] == 1


def test_render_status_markdown_includes_key_sections(tmp_path: Path) -> None:
    inv = tmp_path / "inv.json"
    inv.write_text(
        json.dumps(
            _inventory_payload(
                [
                    _candidate(
                        path="/x/.worktrees/codex-auto/feature-a",
                        branch="feature/a",
                        decision="harvest_candidate",
                        classification="unique_unharvested",
                    ),
                    _candidate(
                        path="/x/.worktrees/codex-auto/feature-b",
                        branch="feature/b",
                        decision="cleanup_candidate",
                        classification="patch_equivalent_or_merged",
                    ),
                    _candidate(
                        path="/x/.worktrees/codex-auto/active",
                        branch="droid/x",
                        decision="preserve",
                        classification="active_or_dirty",
                        active_session=True,
                    ),
                ]
            )
        ),
        encoding="utf-8",
    )
    payload = publisher.build_published_report(
        inventory_path=inv, generated_at="2026-05-17T05:00:00Z"
    )
    md = publisher.render_status_markdown(payload)

    assert md.startswith("# Worktree Value Inventory Status")
    assert "Last updated: 2026-05-17T05:00:00Z" in md
    assert "Total candidates: `3`" in md
    assert "Active sessions: `1`" in md
    assert "## Harvest Candidates" in md
    assert "## Cleanup Candidates" in md
    assert "## Preserved (active or dirty)" in md
    assert "feature/a" in md
    assert "feature/b" in md
    assert "droid/x" in md
    # No trailing whitespace bloat / always ends with newline
    assert md.endswith("\n")


def test_main_dry_run_writes_no_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inv = tmp_path / "inv.json"
    inv.write_text(
        json.dumps(_inventory_payload([_candidate()])),
        encoding="utf-8",
    )
    publish_dir = tmp_path / "publish"
    status_doc = tmp_path / "WORKTREE_VALUE_INVENTORY_STATUS.md"

    rc = publisher.main(
        [
            "--input",
            str(inv),
            "--publish-dir",
            str(publish_dir),
            "--status-doc",
            str(status_doc),
            "--dry-run",
            "--generated-at",
            "2026-05-17T05:00:00Z",
        ]
    )
    assert rc == 0
    assert not publish_dir.exists()
    assert not status_doc.exists()

    out = capsys.readouterr().out
    assert "worktree-value-inventory:" in out
    assert "dry-run: artifacts not written" in out


def test_main_writes_artifacts_and_status_doc(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inv = tmp_path / "inv.json"
    inv.write_text(
        json.dumps(
            _inventory_payload(
                [
                    _candidate(decision="harvest_candidate"),
                    _candidate(decision="cleanup_candidate"),
                ]
            )
        ),
        encoding="utf-8",
    )
    publish_dir = tmp_path / "publish"
    status_doc = tmp_path / "WORKTREE_VALUE_INVENTORY_STATUS.md"

    rc = publisher.main(
        [
            "--input",
            str(inv),
            "--publish-dir",
            str(publish_dir),
            "--status-doc",
            str(status_doc),
            "--generated-at",
            "2026-05-17T05:00:00Z",
            "--json",
        ]
    )
    assert rc == 0
    assert (publish_dir / "latest.json").exists()
    assert (publish_dir / "worktree-value-inventory-20260517T050000Z.json").exists()
    assert status_doc.exists()
    md = status_doc.read_text(encoding="utf-8")
    assert "Total candidates: `2`" in md

    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["schema_version"] == publisher.SCHEMA_VERSION
    assert parsed["summary"]["total_candidates"] == 2
