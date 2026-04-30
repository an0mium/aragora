"""Tests for aragora.markets.resolver — deterministic GitHub-event resolution."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta

import pytest

from aragora.markets.resolver import GitHubMarketResolver, ResolutionError, resolve_market
from aragora.markets.types import Market


def _completed(
    stdout: str, *, returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["gh"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _expired_pr_market(*, number: int = 5959) -> Market:
    created = datetime(2026, 4, 1, tzinfo=UTC)
    return Market.create(
        question_kind="pr_merge",
        target={"repo": "synaptent/aragora", "number": number},
        description="will it merge",
        resolution_window_days=7,
        created_at=created,
    )


def _expired_issue_market(*, number: int = 6068) -> Market:
    return Market.create(
        question_kind="issue_close",
        target={"repo": "synaptent/aragora", "number": number},
        description="will it close",
        resolution_window_days=14,
        created_at=datetime(2026, 4, 1, tzinfo=UTC),
    )


def _expired_ci_market(*, ref: str = "abc123") -> Market:
    return Market.create(
        question_kind="ci_pass",
        target={"repo": "synaptent/aragora", "ref": ref},
        description="will ci pass",
        resolution_window_days=3,
        created_at=datetime(2026, 4, 1, tzinfo=UTC),
    )


class TestExpiryGuard:
    def test_unexpired_market_raises(self) -> None:
        market = Market.create(
            question_kind="pr_merge",
            target={"repo": "owner/repo", "number": 1},
            description="x",
            resolution_window_days=7,
            created_at=datetime(2026, 4, 17, tzinfo=UTC),
        )

        def _runner(*args, **kwargs):  # pragma: no cover - never called
            raise AssertionError("runner must not be invoked for unexpired markets")

        resolver = GitHubMarketResolver(gh_runner=_runner)
        with pytest.raises(ResolutionError):
            resolver.resolve(market, now=datetime(2026, 4, 18, tzinfo=UTC))

    def test_require_expiry_false_resolves_immediately(self) -> None:
        market = Market.create(
            question_kind="pr_merge",
            target={"repo": "owner/repo", "number": 1},
            description="x",
            resolution_window_days=7,
            created_at=datetime(2026, 4, 17, tzinfo=UTC),
        )

        def _runner(args, **kwargs):
            return _completed(json.dumps({"state": "MERGED", "merged": True}))

        resolver = GitHubMarketResolver(gh_runner=_runner, require_expiry=False)
        event = resolver.resolve(market, now=datetime(2026, 4, 18, tzinfo=UTC))
        assert event.outcome == "yes"


class TestPrMergeResolution:
    def test_merged_pr_resolves_yes(self) -> None:
        market = _expired_pr_market()
        captured: list[list[str]] = []

        def _runner(args, **kwargs):
            captured.append(list(args))
            return _completed(
                json.dumps({"state": "MERGED", "merged": True, "mergedAt": "2026-04-05T12:00:00Z"})
            )

        event = resolve_market(market, gh_runner=_runner)
        assert event.outcome == "yes"
        assert event.resolution_source == "github_pr_state"
        assert event.evidence["merged"] is True
        assert captured[0][0] == "pr"

    def test_closed_unmerged_pr_resolves_no(self) -> None:
        market = _expired_pr_market(number=42)

        def _runner(args, **kwargs):
            return _completed(
                json.dumps({"state": "CLOSED", "merged": False, "closedAt": "2026-04-06T08:00:00Z"})
            )

        event = resolve_market(market, gh_runner=_runner)
        assert event.outcome == "no"

    def test_open_pr_at_expiry_resolves_inconclusive(self) -> None:
        market = _expired_pr_market(number=43)

        def _runner(args, **kwargs):
            return _completed(json.dumps({"state": "OPEN", "merged": False}))

        event = resolve_market(market, gh_runner=_runner)
        assert event.outcome == "inconclusive"

    def test_gh_failure_raises_resolution_error(self) -> None:
        market = _expired_pr_market(number=44)

        def _runner(args, **kwargs):
            return _completed("", returncode=1, stderr="server error")

        with pytest.raises(ResolutionError):
            resolve_market(market, gh_runner=_runner)

    def test_real_world_gh_payload_no_merged_field_resolves_yes(self) -> None:
        """Regression: round 30c-Phase-E discovered that ``gh pr view --json``
        does NOT expose a ``merged`` field — only ``mergedAt`` and ``state``.

        Previous resolver versions queried ``state,merged,...`` and read
        ``bool(payload.get("merged"))``, which always evaluated to False for
        real ``gh`` output. Live MERGED PRs would never resolve YES.

        This test pins the fix: even when ``merged`` is absent from the
        payload, ``state == "MERGED"`` still produces the YES outcome."""
        market = _expired_pr_market(number=6828)
        captured: list[list[str]] = []

        def _runner(args, **kwargs):
            captured.append(list(args))
            # Real gh output — note ``merged`` is NOT present.
            return _completed(
                json.dumps(
                    {
                        "state": "MERGED",
                        "mergedAt": "2026-04-30T03:16:56Z",
                        "closedAt": "2026-04-30T03:16:56Z",
                    }
                )
            )

        event = resolve_market(market, gh_runner=_runner)
        assert event.outcome == "yes"
        assert event.evidence["merged"] is True
        # And the field-list arg must NOT include the invalid ``merged`` field.
        json_arg_idx = captured[0].index("--json")
        fields = captured[0][json_arg_idx + 1].split(",")
        assert "merged" not in fields, (
            f"`merged` is not a valid gh JSON field — found in {fields!r}"
        )

    def test_real_world_gh_payload_no_merged_field_open_pr(self) -> None:
        """Companion regression: an OPEN PR with no ``merged`` key still
        resolves to inconclusive (not YES) under the fixed logic."""
        market = _expired_pr_market(number=99)

        def _runner(args, **kwargs):
            return _completed(json.dumps({"state": "OPEN"}))

        event = resolve_market(market, gh_runner=_runner)
        assert event.outcome == "inconclusive"
        assert event.evidence["merged"] is False


class TestIssueResolution:
    def test_closed_issue_resolves_yes(self) -> None:
        market = _expired_issue_market(number=6068)

        def _runner(args, **kwargs):
            return _completed(json.dumps({"state": "CLOSED", "closedAt": "2026-04-10T12:00:00Z"}))

        event = resolve_market(market, gh_runner=_runner)
        assert event.outcome == "yes"

    def test_open_issue_at_expiry_resolves_inconclusive(self) -> None:
        market = _expired_issue_market(number=6069)

        def _runner(args, **kwargs):
            return _completed(json.dumps({"state": "OPEN", "closedAt": None}))

        event = resolve_market(market, gh_runner=_runner)
        assert event.outcome == "inconclusive"


class TestCiResolution:
    def test_all_success_resolves_yes(self) -> None:
        market = _expired_ci_market(ref="abcd")
        suites = [
            {"status": "completed", "conclusion": "success", "app_slug": "github-actions"},
            {"status": "completed", "conclusion": "neutral", "app_slug": "github-actions"},
        ]

        def _runner(args, **kwargs):
            return _completed(json.dumps(suites))

        event = resolve_market(market, gh_runner=_runner)
        assert event.outcome == "yes"

    def test_any_failure_resolves_no(self) -> None:
        market = _expired_ci_market(ref="bad1")
        suites = [
            {"status": "completed", "conclusion": "success"},
            {"status": "completed", "conclusion": "failure"},
        ]

        def _runner(args, **kwargs):
            return _completed(json.dumps(suites))

        event = resolve_market(market, gh_runner=_runner)
        assert event.outcome == "no"

    def test_no_completed_runs_resolves_inconclusive(self) -> None:
        market = _expired_ci_market(ref="pending")
        suites = [
            {"status": "in_progress", "conclusion": None},
            {"status": "queued", "conclusion": None},
        ]

        def _runner(args, **kwargs):
            return _completed(json.dumps(suites))

        event = resolve_market(market, gh_runner=_runner)
        assert event.outcome == "inconclusive"


class TestResolveBatch:
    def test_resolve_batch_skips_transient_failures(self) -> None:
        good = _expired_pr_market(number=1)
        bad = _expired_pr_market(number=2)
        call_count = {"n": 0}

        def _runner(args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _completed(json.dumps({"state": "MERGED", "merged": True}))
            return _completed("", returncode=1, stderr="boom")

        resolver = GitHubMarketResolver(gh_runner=_runner)
        events = resolver.resolve_batch([good, bad])
        assert len(events) == 1
        assert events[0].outcome == "yes"
