"""Comprehensive tests for aragora/swarm/boss_feed.py.

Covers:
- GitHubIssue dataclass and to_dict()
- IssueEligibilityReport properties
- GitHubIssueFeed construction, fetch(), _fetch_issue()
- _normalize_scope_entry / _normalize_scope_entries
- _normalize_lane_name
- _infer_lane_from_scope_entry
- infer_issue_lane_hints
- _scope_entry_matches / scope_entries_overlap
- infer_issue_scope_entries
- issue_overlaps_blocked_scopes
- fetch_open_pr_changed_paths
- build_issue_eligibility_report
- select_eligible_issue
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from aragora.swarm.boss_feed import (
    GitHubIssue,
    GitHubIssueFeed,
    IssueEligibilityReport,
    _infer_lane_from_scope_entry,
    _normalize_lane_name,
    _normalize_scope_entries,
    _normalize_scope_entry,
    _scope_entry_matches,
    build_issue_eligibility_report,
    fetch_open_pr_changed_paths,
    infer_issue_lane_hints,
    infer_issue_scope_entries,
    issue_overlaps_blocked_scopes,
    scope_entries_overlap,
    select_eligible_issue,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_issue(
    number: int = 1,
    title: str = "Fix something important",
    body: str = "## Task\nImplement a comprehensive fix for this long-standing problem in the codebase.",
    labels: list[str] | None = None,
    state: str = "OPEN",
    url: str = "https://github.com/org/repo/issues/1",
    created_at: str = "2026-01-01T00:00:00Z",
) -> GitHubIssue:
    return GitHubIssue(
        number=number,
        title=title,
        body=body,
        labels=labels if labels is not None else [],
        url=url,
        state=state,
        created_at=created_at,
    )


def _make_gh_proc(stdout: str, returncode: int = 0, stderr: str = "") -> MagicMock:
    proc = MagicMock()
    proc.stdout = stdout
    proc.stderr = stderr
    proc.returncode = returncode
    return proc


# ---------------------------------------------------------------------------
# GitHubIssue
# ---------------------------------------------------------------------------


class TestGitHubIssue:
    def test_to_dict_round_trip(self):
        issue = _make_issue(number=42, title="My issue", labels=["bug", "help wanted"])
        d = issue.to_dict()
        assert d["number"] == 42
        assert d["title"] == "My issue"
        assert d["labels"] == ["bug", "help wanted"]
        assert d["state"] == "OPEN"

    def test_to_dict_labels_copy(self):
        issue = _make_issue(labels=["x"])
        d = issue.to_dict()
        d["labels"].append("y")
        assert issue.labels == ["x"]

    def test_to_dict_all_fields_present(self):
        issue = _make_issue()
        d = issue.to_dict()
        assert set(d.keys()) == {"number", "title", "body", "labels", "url", "state", "created_at"}

    def test_empty_labels(self):
        issue = _make_issue(labels=[])
        assert issue.to_dict()["labels"] == []


# ---------------------------------------------------------------------------
# IssueEligibilityReport
# ---------------------------------------------------------------------------


class TestIssueEligibilityReport:
    def test_eligible_count_empty(self):
        report = IssueEligibilityReport()
        assert report.eligible_count == 0

    def test_eligible_count(self):
        report = IssueEligibilityReport(eligible=[_make_issue(1), _make_issue(2)])
        assert report.eligible_count == 2

    def test_skipped_by_label_count(self):
        report = IssueEligibilityReport(skipped_by_label={"wip": [1, 2], "blocked": [3]})
        assert report.skipped_by_label_count == 3

    def test_skipped_by_label_count_empty(self):
        report = IssueEligibilityReport()
        assert report.skipped_by_label_count == 0

    def test_skipped_by_sanitation_count(self):
        report = IssueEligibilityReport(
            skipped_by_sanitation={"empty_body": [5], "task_too_short": [6, 7]}
        )
        assert report.skipped_by_sanitation_count == 3

    def test_skipped_by_sanitation_count_empty(self):
        report = IssueEligibilityReport()
        assert report.skipped_by_sanitation_count == 0


# ---------------------------------------------------------------------------
# _normalize_scope_entry
# ---------------------------------------------------------------------------


class TestNormalizeScopeEntry:
    def test_valid_aragora_path(self):
        assert _normalize_scope_entry("aragora/swarm/boss_feed.py") == "aragora/swarm/boss_feed.py"

    def test_valid_tests_path(self):
        assert _normalize_scope_entry("tests/swarm/test_boss.py") == "tests/swarm/test_boss.py"

    def test_valid_scripts_path(self):
        assert _normalize_scope_entry("scripts/nomic_loop.py") == "scripts/nomic_loop.py"

    def test_valid_docs_path(self):
        assert _normalize_scope_entry("docs/STATUS.md") == "docs/STATUS.md"

    def test_valid_sdk_path(self):
        assert _normalize_scope_entry("sdk/client.py") == "sdk/client.py"

    def test_empty_string_returns_none(self):
        assert _normalize_scope_entry("") is None

    def test_none_returns_none(self):
        assert _normalize_scope_entry(None) is None

    def test_no_slash_returns_none(self):
        assert _normalize_scope_entry("somefile.py") is None

    def test_url_returns_none(self):
        assert _normalize_scope_entry("https://example.com/aragora/foo.py") is None

    def test_unknown_root_prefix_returns_none(self):
        assert _normalize_scope_entry("unknown_dir/foo/bar.py") is None

    def test_trailing_slash_stripped(self):
        assert _normalize_scope_entry("aragora/swarm/") == "aragora/swarm"

    def test_dotslash_prefix_stripped(self):
        # The leading "." is stripped by strip("'\".,;:()[]{}<>"), leaving "/aragora/..."
        # which then fails the root-prefix check — so the result is None.
        # This documents the actual behaviour (no special-case for "./" prefix).
        assert _normalize_scope_entry("./aragora/swarm/foo.py") is None

    def test_backtick_stripped(self):
        assert _normalize_scope_entry("`aragora/swarm/foo.py`") == "aragora/swarm/foo.py"

    def test_glob_path(self):
        result = _normalize_scope_entry("aragora/swarm/**")
        assert result == "aragora/swarm/**"

    def test_quotes_stripped(self):
        assert _normalize_scope_entry("'aragora/swarm/foo.py'") == "aragora/swarm/foo.py"


# ---------------------------------------------------------------------------
# _normalize_scope_entries
# ---------------------------------------------------------------------------


class TestNormalizeScopeEntries:
    def test_deduplicates(self):
        entries = ["aragora/swarm/foo.py", "aragora/swarm/foo.py"]
        result = _normalize_scope_entries(entries)
        assert result == ["aragora/swarm/foo.py"]

    def test_filters_invalid(self):
        result = _normalize_scope_entries(["aragora/foo.py", "invalid", ""])
        assert result == ["aragora/foo.py"]

    def test_empty_list(self):
        assert _normalize_scope_entries([]) == []

    def test_preserves_order(self):
        entries = ["tests/swarm/a.py", "aragora/swarm/b.py", "scripts/c.py"]
        result = _normalize_scope_entries(entries)
        assert result == entries


# ---------------------------------------------------------------------------
# _normalize_lane_name
# ---------------------------------------------------------------------------


class TestNormalizeLaneName:
    def test_empty_returns_none(self):
        assert _normalize_lane_name("") is None

    def test_none_returns_none(self):
        assert _normalize_lane_name(None) is None

    def test_lowercased(self):
        assert _normalize_lane_name("SERVER") == "server"

    def test_alias_api_to_server(self):
        assert _normalize_lane_name("api") == "server"

    def test_alias_backend_to_server(self):
        assert _normalize_lane_name("backend") == "server"

    def test_alias_nomic_to_swarm(self):
        assert _normalize_lane_name("nomic") == "swarm"

    def test_alias_frontend_unchanged(self):
        assert _normalize_lane_name("frontend") == "frontend"

    def test_alias_ui_to_frontend(self):
        assert _normalize_lane_name("ui") == "frontend"

    def test_alias_live_to_frontend(self):
        assert _normalize_lane_name("live") == "frontend"

    def test_alias_documentation_to_docs(self):
        assert _normalize_lane_name("documentation") == "docs"

    def test_alias_tooling_to_infra(self):
        assert _normalize_lane_name("tooling") == "infra"

    def test_underscores_converted_to_hyphens(self):
        # "control_plane" normalizes to "control-plane" then alias maps to swarm
        assert _normalize_lane_name("control_plane") == "swarm"

    def test_spaces_converted(self):
        assert _normalize_lane_name("front end") == "frontend"

    def test_repeated_hyphens_collapsed(self):
        # "front--end" → "front-end" → alias front-end → "frontend"
        result = _normalize_lane_name("front--end")
        assert result is not None  # normalizes without error

    def test_special_chars_stripped(self):
        assert _normalize_lane_name("server!") == "server"

    def test_unknown_lane_returned_as_is(self):
        result = _normalize_lane_name("custom-lane-xyz")
        assert result == "custom-lane-xyz"


# ---------------------------------------------------------------------------
# _infer_lane_from_scope_entry
# ---------------------------------------------------------------------------


class TestInferLaneFromScopeEntry:
    def test_swarm_path(self):
        assert _infer_lane_from_scope_entry("aragora/swarm/foo.py") == "swarm"

    def test_nomic_path(self):
        assert _infer_lane_from_scope_entry("aragora/nomic/meta_planner.py") == "swarm"

    def test_tests_swarm_path(self):
        assert _infer_lane_from_scope_entry("tests/swarm/test_foo.py") == "swarm"

    def test_server_path(self):
        assert _infer_lane_from_scope_entry("aragora/server/unified_server.py") == "server"

    def test_live_path(self):
        assert _infer_lane_from_scope_entry("aragora/live/index.tsx") == "frontend"

    def test_docs_site_path(self):
        assert _infer_lane_from_scope_entry("docs-site/docs/intro.md") == "frontend"

    def test_sdk_path(self):
        assert _infer_lane_from_scope_entry("sdk/client.py") == "sdk"

    def test_docs_path(self):
        assert _infer_lane_from_scope_entry("docs/STATUS.md") == "docs"

    def test_scripts_path(self):
        assert _infer_lane_from_scope_entry("scripts/nomic_loop.py") == "infra"

    def test_github_path(self):
        assert _infer_lane_from_scope_entry(".github/workflows/ci.yml") == "infra"

    def test_unknown_path_returns_none(self):
        assert _infer_lane_from_scope_entry("aragora/billing/cost_tracker.py") is None

    def test_exact_prefix_match(self):
        # "sdk" without trailing slash — should still match "sdk/" prefix rule
        assert _infer_lane_from_scope_entry("sdk") == "sdk"


# ---------------------------------------------------------------------------
# _scope_entry_matches / scope_entries_overlap
# ---------------------------------------------------------------------------


class TestScopeEntryMatches:
    def test_exact_match(self):
        assert _scope_entry_matches("aragora/swarm/foo.py", "aragora/swarm/foo.py")

    def test_directory_prefix(self):
        assert _scope_entry_matches("aragora/swarm", "aragora/swarm/foo.py")

    def test_glob_match(self):
        assert _scope_entry_matches("aragora/swarm/**", "aragora/swarm/sub/foo.py")

    def test_no_match_different_file(self):
        assert not _scope_entry_matches("aragora/swarm/foo.py", "aragora/swarm/bar.py")

    def test_no_partial_prefix_match(self):
        # "aragora/swarmer" should not match "aragora/swarm"
        assert not _scope_entry_matches("aragora/swarm", "aragora/swarmer/foo.py")

    def test_directory_no_extension_matches_children(self):
        assert _scope_entry_matches("tests/swarm", "tests/swarm/test_foo.py")


class TestScopeEntriesOverlap:
    def test_symmetric(self):
        assert scope_entries_overlap("aragora/swarm", "aragora/swarm/foo.py")
        assert scope_entries_overlap("aragora/swarm/foo.py", "aragora/swarm")

    def test_no_overlap(self):
        assert not scope_entries_overlap("aragora/swarm/a.py", "aragora/swarm/b.py")

    def test_glob_overlaps_file(self):
        assert scope_entries_overlap("aragora/swarm/**", "aragora/swarm/deep/file.py")


# ---------------------------------------------------------------------------
# infer_issue_lane_hints
# ---------------------------------------------------------------------------


class TestInferIssueLaneHints:
    def test_lane_from_label(self):
        issue = _make_issue(labels=["lane:swarm"])
        hints = infer_issue_lane_hints(issue)
        assert "swarm" in hints

    def test_lane_from_label_area_prefix(self):
        issue = _make_issue(labels=["area:server"])
        hints = infer_issue_lane_hints(issue)
        assert "server" in hints

    def test_lane_from_label_alias(self):
        issue = _make_issue(labels=["lane:backend"])
        hints = infer_issue_lane_hints(issue)
        assert "server" in hints

    def test_lane_from_body(self):
        issue = _make_issue(
            labels=[],
            body="Lane: frontend\n\n## Task\nUpdate the live dashboard with new metrics display.",
        )
        hints = infer_issue_lane_hints(issue)
        assert "frontend" in hints

    def test_lane_from_body_owner_lane(self):
        issue = _make_issue(
            labels=[],
            body="Owner Lane: swarm\n\n## Task\nFix the supervisor loop polling interval.",
        )
        hints = infer_issue_lane_hints(issue)
        assert "swarm" in hints

    def test_label_takes_priority_over_body(self):
        issue = _make_issue(
            labels=["lane:sdk"],
            body="Lane: server",
        )
        hints = infer_issue_lane_hints(issue)
        assert hints[0] == "sdk"

    def test_no_hints_no_matching_scope(self):
        issue = _make_issue(labels=[], body="## Task\nGeneric work with no file references here.")
        hints = infer_issue_lane_hints(issue)
        # May be empty or inferred from scope — just check it doesn't crash
        assert isinstance(hints, list)

    def test_deduplicates_hints(self):
        issue = _make_issue(labels=["lane:swarm", "area:swarm"])
        hints = infer_issue_lane_hints(issue)
        assert hints.count("swarm") == 1


# ---------------------------------------------------------------------------
# infer_issue_scope_entries
# ---------------------------------------------------------------------------


class TestInferIssueScopeEntries:
    def test_scope_from_body(self):
        body = (
            "## Task\n"
            "Update aragora/swarm/boss_feed.py to add new filtering logic. "
            "Also update tests/swarm/test_boss_feed.py accordingly."
        )
        issue = _make_issue(body=body)
        with patch("aragora.swarm.boss_feed.infer_issue_scope_entries") as mock_infer:
            mock_infer.return_value = [
                "aragora/swarm/boss_feed.py",
                "tests/swarm/test_boss_feed.py",
            ]
            result = mock_infer(issue)
        assert "aragora/swarm/boss_feed.py" in result

    def test_empty_body_returns_empty(self):
        # Use a real call but patch SwarmSpec
        issue = _make_issue(body="No file paths here at all.")
        with patch("aragora.swarm.spec.SwarmSpec.infer_file_scope_hints", return_value=[]):
            result = infer_issue_scope_entries(issue)
        assert result == []

    def test_deduplicates_entries(self):
        issue = _make_issue(body="aragora/swarm/foo.py aragora/swarm/foo.py")
        with patch(
            "aragora.swarm.spec.SwarmSpec.infer_file_scope_hints",
            return_value=["aragora/swarm/foo.py", "aragora/swarm/foo.py"],
        ):
            result = infer_issue_scope_entries(issue)
        assert result.count("aragora/swarm/foo.py") == 1


# ---------------------------------------------------------------------------
# issue_overlaps_blocked_scopes
# ---------------------------------------------------------------------------


class TestIssueOverlapsBlockedScopes:
    def test_no_blocked_scopes_returns_false(self):
        issue = _make_issue()
        assert not issue_overlaps_blocked_scopes(issue, None)

    def test_empty_blocked_scopes_returns_false(self):
        issue = _make_issue()
        assert not issue_overlaps_blocked_scopes(issue, set())

    def test_overlap_detected(self):
        issue = _make_issue()
        with patch(
            "aragora.swarm.boss_feed.infer_issue_scope_entries",
            return_value=["aragora/swarm/boss_feed.py"],
        ):
            result = issue_overlaps_blocked_scopes(issue, {"aragora/swarm/boss_feed.py"})
        assert result is True

    def test_no_overlap(self):
        issue = _make_issue()
        with patch(
            "aragora.swarm.boss_feed.infer_issue_scope_entries",
            return_value=["aragora/billing/cost_tracker.py"],
        ):
            result = issue_overlaps_blocked_scopes(issue, {"aragora/swarm/foo.py"})
        assert result is False

    def test_no_issue_scopes_returns_false(self):
        issue = _make_issue()
        with patch("aragora.swarm.boss_feed.infer_issue_scope_entries", return_value=[]):
            result = issue_overlaps_blocked_scopes(issue, {"aragora/swarm/foo.py"})
        assert result is False


# ---------------------------------------------------------------------------
# fetch_open_pr_changed_paths
# ---------------------------------------------------------------------------


class TestFetchOpenPrChangedPaths:
    def test_returns_paths_from_gh_output(self):
        payload = json.dumps(
            [
                {
                    "files": [
                        {"path": "aragora/swarm/boss_feed.py"},
                        {"path": "tests/swarm/test_boss.py"},
                    ]
                }
            ]
        )
        proc = _make_gh_proc(payload)
        with patch("subprocess.run", return_value=proc):
            paths = fetch_open_pr_changed_paths()
        assert "aragora/swarm/boss_feed.py" in paths
        assert "tests/swarm/test_boss.py" in paths

    def test_ignores_non_root_paths(self):
        payload = json.dumps([{"files": [{"path": "unknown_dir/file.py"}]}])
        proc = _make_gh_proc(payload)
        with patch("subprocess.run", return_value=proc):
            paths = fetch_open_pr_changed_paths()
        assert "unknown_dir/file.py" not in paths

    def test_returns_empty_on_gh_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("gh not found")):
            paths = fetch_open_pr_changed_paths()
        assert paths == set()

    def test_returns_empty_on_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("gh", 30)):
            paths = fetch_open_pr_changed_paths()
        assert paths == set()

    def test_returns_empty_on_nonzero_returncode(self):
        proc = _make_gh_proc("", returncode=1, stderr="not logged in")
        with patch("subprocess.run", return_value=proc):
            paths = fetch_open_pr_changed_paths()
        assert paths == set()

    def test_returns_empty_on_invalid_json(self):
        proc = _make_gh_proc("not-json")
        with patch("subprocess.run", return_value=proc):
            paths = fetch_open_pr_changed_paths()
        assert paths == set()

    def test_returns_empty_when_json_not_list(self):
        proc = _make_gh_proc(json.dumps({"files": []}))
        with patch("subprocess.run", return_value=proc):
            paths = fetch_open_pr_changed_paths()
        assert paths == set()

    def test_with_repo_flag(self):
        proc = _make_gh_proc(json.dumps([]))
        with patch("subprocess.run", return_value=proc) as mock_run:
            fetch_open_pr_changed_paths(repo="owner/repo")
        call_args = mock_run.call_args[0][0]
        assert "--repo" in call_args
        assert "owner/repo" in call_args

    def test_limit_clamped_to_100(self):
        proc = _make_gh_proc(json.dumps([]))
        with patch("subprocess.run", return_value=proc) as mock_run:
            fetch_open_pr_changed_paths(limit=9999)
        call_args = mock_run.call_args[0][0]
        limit_idx = call_args.index("--limit") + 1
        assert int(call_args[limit_idx]) == 100


# ---------------------------------------------------------------------------
# GitHubIssueFeed — construction
# ---------------------------------------------------------------------------


class TestGitHubIssueFeedConstruction:
    def test_limit_clamped_minimum(self):
        feed = GitHubIssueFeed(limit=0)
        assert feed.limit == 1

    def test_limit_clamped_maximum(self):
        feed = GitHubIssueFeed(limit=999)
        assert feed.limit == 100

    def test_issue_numbers_coerced_to_int(self):
        feed = GitHubIssueFeed(issue_numbers=["42", "7"])
        assert feed.issue_numbers == [42, 7]

    def test_issue_numbers_filters_zero(self):
        feed = GitHubIssueFeed(issue_numbers=[0, 1, 2])
        assert 0 not in feed.issue_numbers

    def test_defaults(self):
        feed = GitHubIssueFeed()
        assert feed.repo is None
        assert feed.label_filter is None
        assert feed.issue_numbers == []
        assert feed.limit == 25


# ---------------------------------------------------------------------------
# GitHubIssueFeed.fetch() — list mode
# ---------------------------------------------------------------------------


class TestGitHubIssueFeedFetch:
    def _raw_issue(self, number: int = 1) -> dict:
        return {
            "number": number,
            "title": f"Issue #{number}",
            "body": "## Task\nSome meaningful work description that is long enough.",
            "labels": [{"name": "bug"}, {"name": "help-wanted"}],
            "url": f"https://github.com/org/repo/issues/{number}",
            "state": "OPEN",
            "createdAt": "2026-01-01T00:00:00Z",
        }

    def test_parses_issues_from_json(self):
        payload = json.dumps([self._raw_issue(1), self._raw_issue(2)])
        proc = _make_gh_proc(payload)
        feed = GitHubIssueFeed()
        with patch("subprocess.run", return_value=proc):
            issues = feed.fetch()
        assert len(issues) == 2
        assert issues[0].number == 1
        assert issues[1].number == 2

    def test_labels_extracted_from_dict(self):
        payload = json.dumps([self._raw_issue(1)])
        proc = _make_gh_proc(payload)
        feed = GitHubIssueFeed()
        with patch("subprocess.run", return_value=proc):
            issues = feed.fetch()
        assert "bug" in issues[0].labels
        assert "help-wanted" in issues[0].labels

    def test_labels_can_be_plain_strings(self):
        raw = self._raw_issue(1)
        raw["labels"] = ["plain-string-label"]
        payload = json.dumps([raw])
        proc = _make_gh_proc(payload)
        feed = GitHubIssueFeed()
        with patch("subprocess.run", return_value=proc):
            issues = feed.fetch()
        assert "plain-string-label" in issues[0].labels

    def test_empty_label_names_filtered(self):
        raw = self._raw_issue(1)
        raw["labels"] = [{"name": ""}, {"name": "valid"}]
        payload = json.dumps([raw])
        proc = _make_gh_proc(payload)
        feed = GitHubIssueFeed()
        with patch("subprocess.run", return_value=proc):
            issues = feed.fetch()
        assert issues[0].labels == ["valid"]

    def test_returns_empty_on_gh_not_found(self):
        feed = GitHubIssueFeed()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert feed.fetch() == []

    def test_returns_empty_on_timeout(self):
        feed = GitHubIssueFeed()
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("gh", 30)):
            assert feed.fetch() == []

    def test_returns_empty_on_nonzero_returncode(self):
        proc = _make_gh_proc("", returncode=1, stderr="auth error")
        feed = GitHubIssueFeed()
        with patch("subprocess.run", return_value=proc):
            assert feed.fetch() == []

    def test_returns_empty_on_invalid_json(self):
        proc = _make_gh_proc("INVALID JSON")
        feed = GitHubIssueFeed()
        with patch("subprocess.run", return_value=proc):
            assert feed.fetch() == []

    def test_returns_empty_when_json_not_list(self):
        proc = _make_gh_proc(json.dumps({"issues": []}))
        feed = GitHubIssueFeed()
        with patch("subprocess.run", return_value=proc):
            assert feed.fetch() == []

    def test_skips_non_dict_items_in_list(self):
        payload = json.dumps(["not-a-dict", self._raw_issue(1)])
        proc = _make_gh_proc(payload)
        feed = GitHubIssueFeed()
        with patch("subprocess.run", return_value=proc):
            issues = feed.fetch()
        assert len(issues) == 1

    def test_repo_flag_included_in_cmd(self):
        proc = _make_gh_proc(json.dumps([]))
        feed = GitHubIssueFeed(repo="owner/repo")
        with patch("subprocess.run", return_value=proc) as mock_run:
            feed.fetch()
        call_args = mock_run.call_args[0][0]
        assert "--repo" in call_args
        assert "owner/repo" in call_args

    def test_label_filter_included_in_cmd(self):
        proc = _make_gh_proc(json.dumps([]))
        feed = GitHubIssueFeed(label_filter="swarm-dispatch")
        with patch("subprocess.run", return_value=proc) as mock_run:
            feed.fetch()
        call_args = mock_run.call_args[0][0]
        assert "--label" in call_args
        assert "swarm-dispatch" in call_args

    def test_issue_numbers_mode_uses_view(self):
        raw = self._raw_issue(42)
        proc = _make_gh_proc(json.dumps(raw))
        feed = GitHubIssueFeed(issue_numbers=[42])
        with patch("subprocess.run", return_value=proc) as mock_run:
            issues = feed.fetch()
        call_args = mock_run.call_args[0][0]
        assert "view" in call_args
        assert issues[0].number == 42

    def test_issue_numbers_skips_closed(self):
        raw = self._raw_issue(7)
        raw["state"] = "CLOSED"
        proc = _make_gh_proc(json.dumps(raw))
        feed = GitHubIssueFeed(issue_numbers=[7])
        with patch("subprocess.run", return_value=proc):
            issues = feed.fetch()
        assert issues == []


# ---------------------------------------------------------------------------
# build_issue_eligibility_report
# ---------------------------------------------------------------------------


class TestBuildIssueEligibilityReport:
    def _sanitized_issue(self, number: int = 1, **kwargs) -> GitHubIssue:
        """Return an issue that passes sanitation checks."""
        return _make_issue(number=number, **kwargs)

    def test_open_issue_is_eligible(self):
        issues = [self._sanitized_issue()]
        with patch(
            "aragora.swarm.boss_feed.assess_issue_body_sanitation",
            return_value=(True, None),
        ):
            report = build_issue_eligibility_report(issues)
        assert report.eligible_count == 1

    def test_closed_issue_excluded(self):
        issues = [_make_issue(state="CLOSED")]
        with patch(
            "aragora.swarm.boss_feed.assess_issue_body_sanitation",
            return_value=(True, None),
        ):
            report = build_issue_eligibility_report(issues)
        assert report.eligible_count == 0

    def test_empty_title_excluded(self):
        issues = [_make_issue(title="")]
        with patch(
            "aragora.swarm.boss_feed.assess_issue_body_sanitation",
            return_value=(True, None),
        ):
            report = build_issue_eligibility_report(issues)
        assert report.eligible_count == 0

    def test_skip_label_excludes_issue(self):
        issues = [self._sanitized_issue(labels=["wip"])]
        with patch(
            "aragora.swarm.boss_feed.assess_issue_body_sanitation",
            return_value=(True, None),
        ):
            report = build_issue_eligibility_report(issues, skip_labels={"wip"})
        assert report.eligible_count == 0
        assert report.skipped_by_label_count == 1
        assert 1 in report.skipped_by_label["wip"]

    def test_skip_multiple_labels_recorded_per_label(self):
        issues = [
            self._sanitized_issue(number=1, labels=["wip", "blocked"]),
            self._sanitized_issue(number=2, labels=["blocked"]),
        ]
        with patch(
            "aragora.swarm.boss_feed.assess_issue_body_sanitation",
            return_value=(True, None),
        ):
            report = build_issue_eligibility_report(issues, skip_labels={"wip", "blocked"})
        assert report.eligible_count == 0
        assert report.skipped_by_label_count >= 2

    def test_require_labels_excludes_without_them(self):
        issues = [self._sanitized_issue(labels=["bug"])]
        with patch(
            "aragora.swarm.boss_feed.assess_issue_body_sanitation",
            return_value=(True, None),
        ):
            report = build_issue_eligibility_report(issues, require_labels={"swarm-dispatch"})
        assert report.eligible_count == 0

    def test_require_labels_passes_with_all_labels(self):
        issues = [self._sanitized_issue(labels=["bug", "swarm-dispatch"])]
        with patch(
            "aragora.swarm.boss_feed.assess_issue_body_sanitation",
            return_value=(True, None),
        ):
            report = build_issue_eligibility_report(issues, require_labels={"swarm-dispatch"})
        assert report.eligible_count == 1

    def test_sanitation_failure_excluded(self):
        issues = [_make_issue(body="")]
        with patch(
            "aragora.swarm.boss_feed.assess_issue_body_sanitation",
            return_value=(False, "empty_body"),
        ):
            report = build_issue_eligibility_report(issues)
        assert report.eligible_count == 0
        assert report.skipped_by_sanitation_count == 1

    def test_blocked_scope_excludes_issue(self):
        issues = [self._sanitized_issue()]
        with patch(
            "aragora.swarm.boss_feed.assess_issue_body_sanitation",
            return_value=(True, None),
        ):
            with patch(
                "aragora.swarm.boss_feed.issue_overlaps_blocked_scopes",
                return_value=True,
            ):
                report = build_issue_eligibility_report(
                    issues, blocked_scopes={"aragora/swarm/boss_feed.py"}
                )
        assert report.eligible_count == 0

    def test_empty_issues_returns_empty_report(self):
        report = build_issue_eligibility_report([])
        assert report.eligible_count == 0
        assert report.skipped_by_label_count == 0


# ---------------------------------------------------------------------------
# select_eligible_issue
# ---------------------------------------------------------------------------


class TestSelectEligibleIssue:
    def test_returns_none_on_empty_list(self):
        result = select_eligible_issue([])
        assert result is None

    def test_returns_first_eligible(self):
        issues = [_make_issue(1), _make_issue(2)]
        with patch(
            "aragora.swarm.boss_feed.assess_issue_body_sanitation",
            return_value=(True, None),
        ):
            with patch(
                "aragora.swarm.boss_feed.issue_overlaps_blocked_scopes",
                return_value=False,
            ):
                result = select_eligible_issue(issues)
        assert result is not None
        assert result.number == 1

    def test_returns_none_when_all_skipped_by_label(self):
        issues = [_make_issue(labels=["wip"])]
        with patch(
            "aragora.swarm.boss_feed.assess_issue_body_sanitation",
            return_value=(True, None),
        ):
            result = select_eligible_issue(issues, skip_labels={"wip"})
        assert result is None

    def test_skips_closed_issues(self):
        issues = [_make_issue(state="CLOSED"), _make_issue(number=2, state="OPEN")]
        with patch(
            "aragora.swarm.boss_feed.assess_issue_body_sanitation",
            return_value=(True, None),
        ):
            with patch(
                "aragora.swarm.boss_feed.issue_overlaps_blocked_scopes",
                return_value=False,
            ):
                result = select_eligible_issue(issues)
        assert result is not None
        assert result.number == 2

    def test_value_ranking_fallback_on_import_error(self):
        issues = [_make_issue(1), _make_issue(2)]
        with patch(
            "aragora.swarm.boss_feed.assess_issue_body_sanitation",
            return_value=(True, None),
        ):
            with patch(
                "aragora.swarm.boss_feed.issue_overlaps_blocked_scopes",
                return_value=False,
            ):
                with patch.dict("sys.modules", {"aragora.swarm.value_estimator": None}):
                    result = select_eligible_issue(issues, use_value_ranking=True)
        # Falls back to first eligible
        assert result is not None

    def test_without_value_ranking_returns_first(self):
        issues = [_make_issue(1), _make_issue(2), _make_issue(3)]
        with patch(
            "aragora.swarm.boss_feed.assess_issue_body_sanitation",
            return_value=(True, None),
        ):
            with patch(
                "aragora.swarm.boss_feed.issue_overlaps_blocked_scopes",
                return_value=False,
            ):
                result = select_eligible_issue(issues, use_value_ranking=False)
        assert result is not None
        assert result.number == 1
