"""Tests for ``aragora review-queue observe-outcomes`` (Round 30g phase A).

Synthetic fixtures only — every test uses ``timeline_provider`` to inject
timeline events. No network calls.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from aragora.cli.parser import build_parser
from aragora.cli.commands.review_queue import cmd_review_queue
from aragora.review import observe_outcomes_cli as observe_module
from aragora.review.observe_outcomes_cli import (
    DEFAULT_PER_RECEIPT_EVENT_CAP,
    DEFAULT_WINDOW_DAYS,
    run_observe_outcomes,
)

UTC = timezone.utc
RECEIPTS_SUBDIR = "receipts"


class TestCliParser:
    def test_review_queue_observe_outcomes_json_routes_to_command(self, monkeypatch) -> None:
        import aragora.cli.commands.observe_outcomes_cmd as command_module

        called = {}

        def fake_cmd_observe_outcomes(args) -> int:
            called["args"] = args
            return 0

        monkeypatch.setattr(command_module, "cmd_observe_outcomes", fake_cmd_observe_outcomes)
        args = build_parser().parse_args(["review-queue", "observe-outcomes", "--json"])

        assert args.review_queue_command == "observe-outcomes"
        assert args.json is True
        assert args.write is False
        assert cmd_review_queue(args) == 0
        assert called["args"] is args

    def test_command_resolves_repo_root_instead_of_nested_cwd(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        import aragora.cli.commands.observe_outcomes_cmd as command_module

        repo = tmp_path / "repo"
        nested = repo / "tools" / "subdir"
        (repo / ".git").mkdir(parents=True)
        nested.mkdir(parents=True)
        monkeypatch.chdir(nested)

        called: dict[str, Any] = {}

        def fake_run_observe_outcomes(**kwargs):
            called.update(kwargs)
            return {"mode": "dry-run"}

        monkeypatch.setattr(command_module, "resolve_repo_root", lambda path: repo)
        monkeypatch.setattr(command_module, "run_observe_outcomes", fake_run_observe_outcomes)
        args = build_parser().parse_args(["review-queue", "observe-outcomes", "--json"])

        assert command_module.cmd_observe_outcomes(args) == 0
        assert called["repo_root"] == repo


def _write_receipt(receipts_dir: Path, name: str, payload: dict) -> Path:
    receipts_dir.mkdir(parents=True, exist_ok=True)
    path = receipts_dir / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _base_payload(
    *,
    pr_number: int = 1234,
    reviewed_at: str = "2026-04-25T12:00:00+00:00",
    head_sha: str = "abc1234567890",
) -> dict:
    return {
        "session_id": "sess-1",
        "reviewed_at": reviewed_at,
        "actor": "armand",
        "action": "settle",
        "reason": "test",
        "pr_number": pr_number,
        "pr_url": f"https://github.com/org/repo/pull/{pr_number}",
        "head_sha": head_sha,
        "base_sha": "000",
        "packet_sha": "p",
        "queue_bucket": "ready",
        "machine_recommendation": "fire_and_forget",
        "github_event": "merged",
    }


def _silent_provider(
    pr_number: int, head_sha: str, event_cap: int
) -> tuple[list[Mapping[str, Any]], str | None]:
    """A provider that returns no events (clean window)."""
    return [], None


def _revert_provider_for(head_sha: str):
    def _p(pr_number: int, sha: str, cap: int) -> tuple[list[Mapping[str, Any]], str | None]:
        return (
            [
                {
                    "type": "commit",
                    "at": "2026-04-30T08:00:00+00:00",
                    "message": f'Revert "feat: thing" — refs {head_sha[:7]}',
                }
            ],
            None,
        )

    return _p


def _failing_provider(
    pr_number: int, head_sha: str, cap: int
) -> tuple[list[Mapping[str, Any]], str | None]:
    return [], "gh: 503 Service Unavailable"


# --- run_observe_outcomes ----------------------------------------------


class TestEmptyStore:
    def test_dry_run_no_receipts_proposes_insufficiency_receipt_without_writing(
        self, tmp_path: Path
    ) -> None:
        repo_root = tmp_path
        summary = run_observe_outcomes(
            store_root=tmp_path,
            repo_root=repo_root,
            window_end=datetime(2026, 4, 30, 12, tzinfo=UTC),
            window_days=DEFAULT_WINDOW_DAYS,
            max_receipts=20,
            per_receipt_event_cap=DEFAULT_PER_RECEIPT_EVENT_CAP,
            write=False,
            timeline_provider=_silent_provider,
        )
        assert summary["mode"] == "dry-run"
        assert summary["receipts_examined"] == 0
        assert summary["insufficiency_receipt_path"] is not None
        path = Path(summary["insufficiency_receipt_path"])
        assert not path.exists()
        body = summary["insufficiency_receipt"]
        assert body["kind"] == "phase-a-observe-outcomes-insufficiency-receipt"
        assert "no_receipts_in_window" in " ".join(body["remaining_blockers"])

    def test_write_mode_no_receipts_writes_insufficiency_receipt(self, tmp_path: Path) -> None:
        summary = run_observe_outcomes(
            store_root=tmp_path,
            repo_root=tmp_path,
            window_end=datetime(2026, 4, 30, 12, tzinfo=UTC),
            window_days=DEFAULT_WINDOW_DAYS,
            max_receipts=20,
            per_receipt_event_cap=DEFAULT_PER_RECEIPT_EVENT_CAP,
            write=True,
            timeline_provider=_silent_provider,
        )
        assert summary["mode"] == "write"
        assert summary["receipts_examined"] == 0
        path = Path(summary["insufficiency_receipt_path"])
        assert path.exists()
        body = json.loads(path.read_text())
        assert body == summary["insufficiency_receipt"]
        assert body["kind"] == "phase-a-observe-outcomes-insufficiency-receipt"
        assert "no_receipts_in_window" in " ".join(body["remaining_blockers"])


class TestDryRunDoesNotMutate:
    def test_dry_run_does_not_change_receipt_files(self, tmp_path: Path) -> None:
        receipts_dir = tmp_path / RECEIPTS_SUBDIR
        path = _write_receipt(receipts_dir, "r1", _base_payload())
        original_mtime = path.stat().st_mtime
        original_text = path.read_text()
        summary = run_observe_outcomes(
            store_root=tmp_path,
            repo_root=tmp_path,
            window_end=datetime(2026, 4, 30, 12, tzinfo=UTC),
            window_days=DEFAULT_WINDOW_DAYS,
            max_receipts=20,
            per_receipt_event_cap=DEFAULT_PER_RECEIPT_EVENT_CAP,
            write=False,
            timeline_provider=_revert_provider_for(_base_payload()["head_sha"]),
        )
        assert summary["mode"] == "dry-run"
        assert summary["receipts_examined"] == 1
        assert summary["receipts_written"] == 0
        # File on disk unchanged.
        assert path.read_text() == original_text
        assert path.stat().st_mtime == original_mtime
        # Summary still records signals_after for preview.
        result = summary["results"][0]
        assert result["signals_after"]["outcome_revert_within_window"] is True
        assert result["written"] is False

    def test_dry_run_no_signals_proposes_insufficiency_without_writing(
        self, tmp_path: Path
    ) -> None:
        receipts_dir = tmp_path / RECEIPTS_SUBDIR
        _write_receipt(receipts_dir, "r1", _base_payload())
        summary = run_observe_outcomes(
            store_root=tmp_path,
            repo_root=tmp_path,
            window_end=datetime(2026, 4, 30, 12, tzinfo=UTC),
            window_days=DEFAULT_WINDOW_DAYS,
            max_receipts=20,
            per_receipt_event_cap=DEFAULT_PER_RECEIPT_EVENT_CAP,
            write=False,
            timeline_provider=_silent_provider,
        )
        assert summary["mode"] == "dry-run"
        assert summary["receipts_examined"] == 1
        assert summary["receipts_written"] == 0
        assert summary["insufficiency_receipt"] is not None
        path = Path(summary["insufficiency_receipt_path"])
        assert not path.exists()
        joined = " ".join(summary["insufficiency_receipt"]["remaining_blockers"])
        assert "no_signals_fired" in joined


class TestWriteModeMutates:
    def test_write_mode_persists_v2_fields(self, tmp_path: Path) -> None:
        receipts_dir = tmp_path / RECEIPTS_SUBDIR
        path = _write_receipt(receipts_dir, "r1", _base_payload())
        summary = run_observe_outcomes(
            store_root=tmp_path,
            repo_root=tmp_path,
            window_end=datetime(2026, 4, 30, 12, tzinfo=UTC),
            window_days=DEFAULT_WINDOW_DAYS,
            max_receipts=20,
            per_receipt_event_cap=DEFAULT_PER_RECEIPT_EVENT_CAP,
            write=True,
            timeline_provider=_revert_provider_for(_base_payload()["head_sha"]),
        )
        assert summary["mode"] == "write"
        assert summary["receipts_written"] == 1
        body = json.loads(path.read_text())
        assert body["outcome_revert_within_window"] is True
        assert body["outcome_observed_at"] is not None
        # Other v2 fields are explicit False, not None.
        assert body["outcome_post_merge_incident"] is False
        assert body["outcome_human_override_redo"] is False
        assert body["outcome_rollback"] is False
        assert body["outcome_reopened_pr"] is False

    def test_atomic_write_preserves_existing_receipt_permissions(self, tmp_path: Path) -> None:
        receipts_dir = tmp_path / RECEIPTS_SUBDIR
        path = _write_receipt(receipts_dir, "r1", _base_payload())
        path.chmod(0o644)

        run_observe_outcomes(
            store_root=tmp_path,
            repo_root=tmp_path,
            window_end=datetime(2026, 4, 30, 12, tzinfo=UTC),
            window_days=DEFAULT_WINDOW_DAYS,
            max_receipts=20,
            per_receipt_event_cap=DEFAULT_PER_RECEIPT_EVENT_CAP,
            write=True,
            timeline_provider=_revert_provider_for(_base_payload()["head_sha"]),
        )

        assert path.stat().st_mode & 0o777 == 0o644

    def test_write_mode_no_signals_emits_insufficiency(self, tmp_path: Path) -> None:
        receipts_dir = tmp_path / RECEIPTS_SUBDIR
        _write_receipt(receipts_dir, "r1", _base_payload())
        summary = run_observe_outcomes(
            store_root=tmp_path,
            repo_root=tmp_path,
            window_end=datetime(2026, 4, 30, 12, tzinfo=UTC),
            window_days=DEFAULT_WINDOW_DAYS,
            max_receipts=20,
            per_receipt_event_cap=DEFAULT_PER_RECEIPT_EVENT_CAP,
            write=True,
            timeline_provider=_silent_provider,
        )
        assert summary["receipts_examined"] == 1
        assert summary["receipts_with_signals_fired"] == 0
        # v2 fields ARE now present (we wrote False values), but no
        # signals fired -> insufficiency receipt with sharp note.
        assert summary["insufficiency_receipt_path"] is not None
        body = json.loads(Path(summary["insufficiency_receipt_path"]).read_text())
        joined = " ".join(body["remaining_blockers"])
        assert "no_signals_fired" in joined


def _rate_limit_provider(
    pr_number: int, head_sha: str, cap: int
) -> tuple[list[Mapping[str, Any]], str | None]:
    return (
        [],
        "['gh', 'api', '-X'] returned 1: gh: API rate limit exceeded for user ID 33477136",
    )


class TestFetchErrorsFlagged:
    def test_fetch_errors_recorded_and_skipped(self, tmp_path: Path) -> None:
        receipts_dir = tmp_path / RECEIPTS_SUBDIR
        path = _write_receipt(receipts_dir, "r1", _base_payload())
        summary = run_observe_outcomes(
            store_root=tmp_path,
            repo_root=tmp_path,
            window_end=datetime(2026, 4, 30, 12, tzinfo=UTC),
            window_days=DEFAULT_WINDOW_DAYS,
            max_receipts=20,
            per_receipt_event_cap=DEFAULT_PER_RECEIPT_EVENT_CAP,
            write=True,
            timeline_provider=_failing_provider,
        )
        assert summary["github_fetch_errors"] == 1
        assert summary["github_other_fetch_errors"] == 1
        assert summary["github_rate_limit_fetch_errors"] == 0
        assert summary["receipts_written"] == 0
        # File on disk untouched even in write mode when fetch failed.
        body = json.loads(path.read_text())
        assert body.get("outcome_revert_within_window") is None
        # Insufficiency receipt names the fetch errors (non-rate-limit).
        body_ins = json.loads(Path(summary["insufficiency_receipt_path"]).read_text())
        joined = " ".join(body_ins["remaining_blockers"])
        assert "github_fetch_errors" in joined
        assert "github_rate_limit_fetch_errors" not in joined
        # Per-result entry carries the classification.
        assert summary["results"][0]["fetch_error_class"] == "other"

    def test_rate_limit_fetch_errors_classified_separately(self, tmp_path: Path) -> None:
        receipts_dir = tmp_path / RECEIPTS_SUBDIR
        _write_receipt(receipts_dir, "r1", _base_payload())
        summary = run_observe_outcomes(
            store_root=tmp_path,
            repo_root=tmp_path,
            window_end=datetime(2026, 4, 30, 12, tzinfo=UTC),
            window_days=DEFAULT_WINDOW_DAYS,
            max_receipts=20,
            per_receipt_event_cap=DEFAULT_PER_RECEIPT_EVENT_CAP,
            write=True,
            timeline_provider=_rate_limit_provider,
        )
        # Totals: 1 fetch error, classified as rate_limit.
        assert summary["github_fetch_errors"] == 1
        assert summary["github_rate_limit_fetch_errors"] == 1
        assert summary["github_other_fetch_errors"] == 0
        assert summary["results"][0]["fetch_error_class"] == "rate_limit"
        # Insufficiency receipt advises waiting for rate limit, not debugging
        # network connectivity.
        body_ins = json.loads(Path(summary["insufficiency_receipt_path"]).read_text())
        joined = " ".join(body_ins["remaining_blockers"])
        assert "github_rate_limit_fetch_errors" in joined
        assert "rate limit window to reset" in joined
        assert "github_fetch_errors:" not in joined  # the generic blocker (not rate-limit)
        # Insufficiency body also exposes the classified counts as fields.
        assert body_ins["github_rate_limit_fetch_errors"] == 1
        assert body_ins["github_other_fetch_errors"] == 0

    def test_mixed_fetch_errors_classified_per_receipt(self, tmp_path: Path) -> None:
        """A batch with both rate-limit and non-rate-limit failures should
        produce both blocker reasons and split per-class counts."""
        receipts_dir = tmp_path / RECEIPTS_SUBDIR
        _write_receipt(receipts_dir, "r1", _base_payload(pr_number=4001))
        _write_receipt(receipts_dir, "r2", _base_payload(pr_number=4002))

        def _mixed_provider(
            pr_number: int, head_sha: str, cap: int
        ) -> tuple[list[Mapping[str, Any]], str | None]:
            if pr_number == 4001:
                return [], "gh: 503 Service Unavailable"
            return (
                [],
                "['gh', 'api'] returned 1: gh: secondary rate limit exceeded",
            )

        summary = run_observe_outcomes(
            store_root=tmp_path,
            repo_root=tmp_path,
            window_end=datetime(2026, 4, 30, 12, tzinfo=UTC),
            window_days=DEFAULT_WINDOW_DAYS,
            max_receipts=20,
            per_receipt_event_cap=DEFAULT_PER_RECEIPT_EVENT_CAP,
            write=True,
            timeline_provider=_mixed_provider,
        )
        assert summary["github_fetch_errors"] == 2
        assert summary["github_rate_limit_fetch_errors"] == 1
        assert summary["github_other_fetch_errors"] == 1
        body_ins = json.loads(Path(summary["insufficiency_receipt_path"]).read_text())
        joined = " ".join(body_ins["remaining_blockers"])
        assert "github_rate_limit_fetch_errors" in joined
        assert "github_fetch_errors:" in joined


class TestBoundedFanout:
    def test_max_receipts_cap_is_enforced(self, tmp_path: Path) -> None:
        receipts_dir = tmp_path / RECEIPTS_SUBDIR
        for i in range(5):
            _write_receipt(receipts_dir, f"r{i}", _base_payload(pr_number=1000 + i))
        summary = run_observe_outcomes(
            store_root=tmp_path,
            repo_root=tmp_path,
            window_end=datetime(2026, 4, 30, 12, tzinfo=UTC),
            window_days=DEFAULT_WINDOW_DAYS,
            max_receipts=3,
            per_receipt_event_cap=DEFAULT_PER_RECEIPT_EVENT_CAP,
            write=False,
            timeline_provider=_silent_provider,
        )
        assert summary["receipts_examined"] == 3

    def test_per_receipt_event_cap_passed_to_provider(self, tmp_path: Path) -> None:
        receipts_dir = tmp_path / RECEIPTS_SUBDIR
        _write_receipt(receipts_dir, "r1", _base_payload())
        observed_caps: list[int] = []

        def _capturing(
            pr_number: int, head_sha: str, cap: int
        ) -> tuple[list[Mapping[str, Any]], str | None]:
            observed_caps.append(cap)
            return [], None

        run_observe_outcomes(
            store_root=tmp_path,
            repo_root=tmp_path,
            window_end=datetime(2026, 4, 30, 12, tzinfo=UTC),
            window_days=DEFAULT_WINDOW_DAYS,
            max_receipts=20,
            per_receipt_event_cap=42,
            write=False,
            timeline_provider=_capturing,
        )
        assert observed_caps == [42]


class TestWindowFiltering:
    def test_receipts_outside_window_skipped(self, tmp_path: Path) -> None:
        receipts_dir = tmp_path / RECEIPTS_SUBDIR
        _write_receipt(
            receipts_dir,
            "old",
            _base_payload(reviewed_at="2025-12-01T12:00:00+00:00"),
        )
        _write_receipt(receipts_dir, "in", _base_payload())
        summary = run_observe_outcomes(
            store_root=tmp_path,
            repo_root=tmp_path,
            window_end=datetime(2026, 4, 30, 12, tzinfo=UTC),
            window_days=DEFAULT_WINDOW_DAYS,
            max_receipts=20,
            per_receipt_event_cap=DEFAULT_PER_RECEIPT_EVENT_CAP,
            write=False,
            timeline_provider=_silent_provider,
        )
        assert summary["receipts_examined"] == 1


class TestValidationErrors:
    def test_invalid_window_days(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="window_days must be positive"):
            run_observe_outcomes(
                store_root=tmp_path,
                repo_root=tmp_path,
                window_end=datetime(2026, 4, 30, 12, tzinfo=UTC),
                window_days=0,
                max_receipts=20,
                per_receipt_event_cap=DEFAULT_PER_RECEIPT_EVENT_CAP,
                write=False,
                timeline_provider=_silent_provider,
            )

    def test_invalid_max_receipts(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="max_receipts must be positive"):
            run_observe_outcomes(
                store_root=tmp_path,
                repo_root=tmp_path,
                window_end=datetime(2026, 4, 30, 12, tzinfo=UTC),
                window_days=14,
                max_receipts=0,
                per_receipt_event_cap=DEFAULT_PER_RECEIPT_EVENT_CAP,
                write=False,
                timeline_provider=_silent_provider,
            )

    def test_invalid_event_cap(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="per_receipt_event_cap must be positive"):
            run_observe_outcomes(
                store_root=tmp_path,
                repo_root=tmp_path,
                window_end=datetime(2026, 4, 30, 12, tzinfo=UTC),
                window_days=14,
                max_receipts=20,
                per_receipt_event_cap=0,
                write=False,
                timeline_provider=_silent_provider,
            )


class TestMalformedReceipts:
    def test_missing_pr_number_skipped(self, tmp_path: Path) -> None:
        receipts_dir = tmp_path / RECEIPTS_SUBDIR
        bad = _base_payload()
        bad["pr_number"] = 0
        _write_receipt(receipts_dir, "bad", bad)
        summary = run_observe_outcomes(
            store_root=tmp_path,
            repo_root=tmp_path,
            window_end=datetime(2026, 4, 30, 12, tzinfo=UTC),
            window_days=DEFAULT_WINDOW_DAYS,
            max_receipts=20,
            per_receipt_event_cap=DEFAULT_PER_RECEIPT_EVENT_CAP,
            write=True,
            timeline_provider=_revert_provider_for(bad["head_sha"]),
        )
        assert summary["receipts_examined"] == 1
        result = summary["results"][0]
        assert result["skipped_reason"] == "malformed receipt"
        assert result["written"] is False

    def test_partial_receipt_skipped_before_network_fetch(self, tmp_path: Path) -> None:
        receipts_dir = tmp_path / RECEIPTS_SUBDIR
        _write_receipt(
            receipts_dir,
            "partial",
            {
                "reviewed_at": "2026-04-25T12:00:00+00:00",
                "pr_number": 1234,
                "head_sha": "abc1234567890",
            },
        )

        def _provider_should_not_run(pr_number: int, head_sha: str, cap: int):
            raise AssertionError("network provider should not run for malformed receipts")

        summary = run_observe_outcomes(
            store_root=tmp_path,
            repo_root=tmp_path,
            window_end=datetime(2026, 4, 30, 12, tzinfo=UTC),
            window_days=DEFAULT_WINDOW_DAYS,
            max_receipts=20,
            per_receipt_event_cap=DEFAULT_PER_RECEIPT_EVENT_CAP,
            write=True,
            timeline_provider=_provider_should_not_run,
        )

        assert summary["receipts_examined"] == 1
        assert summary["results"][0]["skipped_reason"] == "malformed receipt"
        assert "missing" in str(summary["results"][0]["fetch_error"])


class TestLiveProviderNormalization:
    def test_issue_search_items_emit_pr_opened_for_follow_up_prs(self) -> None:
        event = observe_module._normalize_issue_search_item(
            {
                "created_at": "2026-04-30T08:00:00Z",
                "number": 9999,
                "title": "Revert feature",
                "body": "Supersedes #1234",
                "pull_request": {"url": "https://api.github.com/repos/o/r/pulls/9999"},
                "labels": [{"name": "rollback"}],
            }
        )

        assert event == {
            "type": "pr_opened",
            "at": "2026-04-30T08:00:00Z",
            "labels": ["rollback"],
            "title": "Revert feature",
            "body": "Supersedes #1234",
            "number": 9999,
        }

    def test_issue_search_items_preserve_incident_body(self) -> None:
        event = observe_module._normalize_issue_search_item(
            {
                "created_at": "2026-04-30T08:00:00Z",
                "number": 42,
                "title": "Incident after #1234",
                "body": "The regression references abc1234.",
                "labels": [{"name": "incident"}],
            }
        )

        assert event == {
            "type": "issue_opened",
            "at": "2026-04-30T08:00:00Z",
            "labels": ["incident"],
            "title": "Incident after #1234",
            "body": "The regression references abc1234.",
            "number": 42,
        }

    def test_timeline_labeled_entry_preserves_source_issue_body(self) -> None:
        event = observe_module._normalize_timeline_entry(
            {
                "event": "labeled",
                "created_at": "2026-04-30T08:00:00Z",
                "label": {"name": "incident"},
                "source": {"issue": {"title": "Incident #1234", "body": "mentions abc1234"}},
            },
            pr_number=1234,
        )

        assert event == {
            "type": "issue_opened",
            "at": "2026-04-30T08:00:00Z",
            "labels": ["incident"],
            "title": "Incident #1234",
            "body": "mentions abc1234",
        }

    def test_gh_provider_combines_timeline_issue_pr_and_commit_search(self, monkeypatch) -> None:
        monkeypatch.setattr(observe_module.shutil, "which", lambda name: "/usr/bin/gh")

        def fake_run_gh_json(args, *, timeout=20):
            if args[:4] == ["gh", "repo", "view", "--json"]:
                return {"nameWithOwner": "owner/repo"}, None
            if any("timeline" in str(part) for part in args):
                return [{"event": "reopened", "created_at": "2026-04-30T08:00:00Z"}], None
            if "search/issues" in args:
                return {
                    "items": [
                        {
                            "node_id": "PR_1",
                            "created_at": "2026-04-30T09:00:00Z",
                            "number": 9999,
                            "title": "Revert feature",
                            "body": "Supersedes #1234",
                            "pull_request": {"url": "https://api.github.com/repos/o/r/pulls/9999"},
                            "labels": [{"name": "rollback"}],
                        },
                        {
                            "node_id": "I_1",
                            "created_at": "2026-04-30T10:00:00Z",
                            "number": 77,
                            "title": "Incident #1234",
                            "body": "mentions abc1234",
                            "labels": [{"name": "incident"}],
                        },
                    ]
                }, None
            if "search/commits" in args:
                return {
                    "items": [
                        {
                            "node_id": "C_1",
                            "sha": "def456",
                            "commit": {
                                "message": 'Revert "feature" refs abc1234',
                                "author": {"date": "2026-04-30T11:00:00Z"},
                            },
                        }
                    ]
                }, None
            raise AssertionError(f"unexpected gh args: {args}")

        monkeypatch.setattr(observe_module, "_run_gh_json", fake_run_gh_json)

        events, error = observe_module._gh_timeline_provider(1234, "abc1234567890", 100)

        assert error is None
        assert {event["type"] for event in events} == {
            "pr_reopened",
            "pr_opened",
            "issue_opened",
            "commit",
        }

    def test_gh_provider_fails_closed_when_search_fetch_fails(self, monkeypatch) -> None:
        monkeypatch.setattr(observe_module.shutil, "which", lambda name: "/usr/bin/gh")

        def fake_run_gh_json(args, *, timeout=20):
            if args[:4] == ["gh", "repo", "view", "--json"]:
                return {"nameWithOwner": "owner/repo"}, None
            if any("timeline" in str(part) for part in args):
                return [], None
            if "search/issues" in args:
                return None, "search/issues returned 502"
            raise AssertionError(f"unexpected gh args: {args}")

        monkeypatch.setattr(observe_module, "_run_gh_json", fake_run_gh_json)

        events, error = observe_module._gh_timeline_provider(1234, "abc1234567890", 100)

        assert events == []
        assert error == "search/issues returned 502"


class TestGhJsonRateLimitHandling:
    def setup_method(self) -> None:
        observe_module._GH_LAST_CALL_AT_BY_BUCKET.clear()

    def test_search_api_uses_slower_default_throttle(self) -> None:
        assert (
            observe_module._gh_throttle_seconds_for(["gh", "api", "-X", "GET", "search/issues"])
            == observe_module.DEFAULT_GH_SEARCH_API_THROTTLE_SECONDS
        )
        assert (
            observe_module._gh_throttle_seconds_for(
                [
                    "gh",
                    "api",
                    "-H",
                    "Accept: application/vnd.github+json",
                    "/repos/:owner/:repo/issues/123/timeline",
                ]
            )
            == observe_module.DEFAULT_GH_API_THROTTLE_SECONDS
        )

    def test_rate_limit_errors_retry_with_backoff(self, monkeypatch) -> None:
        calls = []
        sleeps: list[float] = []
        responses = [
            SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="API rate limit exceeded for user ID 33477136",
            ),
            SimpleNamespace(returncode=0, stdout='{"ok": true}', stderr=""),
        ]

        def fake_run(*args, **kwargs):
            calls.append((args, kwargs))
            return responses.pop(0)

        monkeypatch.setattr(observe_module.subprocess, "run", fake_run)

        payload, error = observe_module._run_gh_json(
            ["gh", "api", "search/issues"],
            throttle_seconds=0.1,
            sleep=sleeps.append,
            clock=lambda: 0.0,
        )

        assert error is None
        assert payload == {"ok": True}
        assert len(calls) == 2
        assert sleeps == [0.1, 0.1]

    def test_non_rate_limit_errors_do_not_retry(self, monkeypatch) -> None:
        calls = []
        sleeps: list[float] = []

        def fake_run(*args, **kwargs):
            calls.append((args, kwargs))
            return SimpleNamespace(returncode=1, stdout="", stderr="Not Found")

        monkeypatch.setattr(observe_module.subprocess, "run", fake_run)

        payload, error = observe_module._run_gh_json(
            ["gh", "api", "search/issues"],
            attempts=3,
            throttle_seconds=0.1,
            sleep=sleeps.append,
            clock=lambda: 0.0,
        )

        assert payload is None
        assert error is not None
        assert "returned 1" in error
        assert len(calls) == 1
        assert sleeps == []

    def test_successful_first_call_does_not_pay_throttle_delay(self, monkeypatch) -> None:
        sleeps: list[float] = []

        def fake_run(*args, **kwargs):
            return SimpleNamespace(returncode=0, stdout='{"ok": true}', stderr="")

        monkeypatch.setattr(observe_module.subprocess, "run", fake_run)

        payload, error = observe_module._run_gh_json(
            ["gh", "api", "search/issues"],
            throttle_seconds=0.1,
            sleep=sleeps.append,
            clock=lambda: 10.0,
        )

        assert error is None
        assert payload == {"ok": True}
        assert sleeps == []

    def test_consecutive_calls_are_spaced_by_throttle_delay(self, monkeypatch) -> None:
        sleeps: list[float] = []

        def fake_run(*args, **kwargs):
            return SimpleNamespace(returncode=0, stdout='{"ok": true}', stderr="")

        monkeypatch.setattr(observe_module.subprocess, "run", fake_run)

        for _ in range(2):
            payload, error = observe_module._run_gh_json(
                ["gh", "api", "search/issues"],
                throttle_seconds=0.1,
                sleep=sleeps.append,
                clock=lambda: 10.0,
            )
            assert error is None
            assert payload == {"ok": True}

        assert sleeps == [0.1]


class TestInsufficiencyReceiptShape:
    def test_v2_now_present_in_window_after_write(self, tmp_path: Path) -> None:
        receipts_dir = tmp_path / RECEIPTS_SUBDIR
        _write_receipt(receipts_dir, "r1", _base_payload())
        summary = run_observe_outcomes(
            store_root=tmp_path,
            repo_root=tmp_path,
            window_end=datetime(2026, 4, 30, 12, tzinfo=UTC),
            window_days=DEFAULT_WINDOW_DAYS,
            max_receipts=20,
            per_receipt_event_cap=DEFAULT_PER_RECEIPT_EVENT_CAP,
            write=True,
            timeline_provider=_revert_provider_for(_base_payload()["head_sha"]),
        )
        assert summary["v2_outcome_fields_now_present_in_window"] is True
        # Signals fired => no insufficiency receipt expected.
        assert summary["insufficiency_receipt_path"] is None
