from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aragora.ralph.github_control import GitHubControl, GitHubControlError

_HEAD_SHA = "a" * 40


def _completed_process(
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
):
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


class TestGitHubControlBranchDiscovery:
    @patch("aragora.ralph.github_control.subprocess.run")
    def test_find_pr_for_branch_returns_url(self, mock_run, tmp_path: Path) -> None:
        mock_run.return_value = _completed_process(
            stdout=json.dumps([{"url": "https://github.com/org/repo/pull/42"}])
        )

        control = GitHubControl(repo_root=tmp_path)
        assert control.find_pr_for_branch("codex/test") == "https://github.com/org/repo/pull/42"

    @patch("aragora.ralph.github_control.subprocess.run")
    def test_find_pr_for_branch_returns_none_when_absent(self, mock_run, tmp_path: Path) -> None:
        mock_run.return_value = _completed_process(stdout="[]")

        control = GitHubControl(repo_root=tmp_path)
        assert control.find_pr_for_branch("codex/test") is None


class TestGitHubControlPRCreation:
    @patch("aragora.ralph.github_control.subprocess.run")
    def test_create_pr_for_branch_returns_url(self, mock_run, tmp_path: Path) -> None:
        mock_run.return_value = _completed_process(stdout="https://github.com/org/repo/pull/77\n")

        control = GitHubControl(repo_root=tmp_path)
        pr_url = control.create_pr_for_branch("codex/test", "main")

        assert pr_url == "https://github.com/org/repo/pull/77"

    @patch("aragora.ralph.github_control.subprocess.run")
    def test_create_pr_for_branch_raises_on_error(self, mock_run, tmp_path: Path) -> None:
        mock_run.return_value = _completed_process(returncode=1, stderr="auth failed")

        control = GitHubControl(repo_root=tmp_path)
        with pytest.raises(GitHubControlError, match="auth failed"):
            control.create_pr_for_branch("codex/test", "main")


class TestGitHubControlIssueComments:
    @patch("aragora.ralph.github_control.subprocess.run")
    def test_upsert_issue_comment_creates_new_comment(self, mock_run, tmp_path: Path) -> None:
        mock_run.side_effect = [
            _completed_process(stdout=json.dumps([])),
            _completed_process(
                stdout=json.dumps(
                    {
                        "id": 91,
                        "html_url": "https://github.com/org/repo/issues/42#issuecomment-91",
                    }
                )
            ),
        ]

        control = GitHubControl(repo_root=tmp_path)
        result = control.upsert_issue_comment(
            repo="org/repo",
            issue_number=42,
            body="Boss loop published a PR.",
            marker="<!-- aragora-boss-loop-publish -->",
        )

        assert result["commented"] is True
        assert result["action"] == "created"
        assert result["comment_id"] == 91
        create_cmd = mock_run.call_args_list[1].args[0]
        assert create_cmd[:4] == ["gh", "api", "--method", "POST"]
        assert create_cmd[4] == "repos/org/repo/issues/42/comments"
        assert any("aragora-boss-loop-publish" in arg for arg in create_cmd)

    @patch("aragora.ralph.github_control.subprocess.run")
    def test_upsert_issue_comment_updates_existing_marker_comment(
        self, mock_run, tmp_path: Path
    ) -> None:
        mock_run.side_effect = [
            _completed_process(
                stdout=json.dumps(
                    [
                        {
                            "id": 77,
                            "body": "Prior update\n\n<!-- aragora-boss-loop-publish -->",
                            "html_url": "https://github.com/org/repo/issues/42#issuecomment-77",
                        }
                    ]
                )
            ),
            _completed_process(
                stdout=json.dumps(
                    {
                        "id": 77,
                        "html_url": "https://github.com/org/repo/issues/42#issuecomment-77",
                    }
                )
            ),
        ]

        control = GitHubControl(repo_root=tmp_path)
        result = control.upsert_issue_comment(
            repo="org/repo",
            issue_number=42,
            body="Boss loop reused the existing PR.",
            marker="<!-- aragora-boss-loop-publish -->",
        )

        assert result["commented"] is True
        assert result["action"] == "updated"
        assert result["comment_id"] == 77
        update_cmd = mock_run.call_args_list[1].args[0]
        assert update_cmd[:4] == ["gh", "api", "--method", "PATCH"]
        assert update_cmd[4] == "repos/org/repo/issues/comments/77"

    @patch("aragora.ralph.github_control.subprocess.run")
    def test_upsert_issue_comment_returns_failure_on_create_error(
        self, mock_run, tmp_path: Path
    ) -> None:
        mock_run.side_effect = [
            _completed_process(stdout=json.dumps([])),
            _completed_process(returncode=1, stderr="comment write failed"),
        ]

        control = GitHubControl(repo_root=tmp_path)
        result = control.upsert_issue_comment(
            repo="org/repo",
            issue_number=42,
            body="Boss loop published a PR.",
            marker="<!-- aragora-boss-loop-publish -->",
        )

        assert result["commented"] is False
        assert result["action"] == "comment_failed"
        assert "comment write failed" in result["detail"]


class TestGitHubControlGateSnapshots:
    @patch("aragora.ralph.github_control.subprocess.run")
    def test_fetch_gate_snapshot_detects_merged_pr(self, mock_run, tmp_path: Path) -> None:
        mock_run.side_effect = [
            _completed_process(
                stdout=json.dumps(
                    {
                        "url": "https://github.com/org/repo/pull/55",
                        "state": "MERGED",
                        "isDraft": False,
                        "headRefName": "codex/test",
                        "baseRefName": "main",
                        "reviewDecision": "APPROVED",
                        "mergeStateStatus": "CLEAN",
                        "mergeCommit": {"oid": "merge-sha"},
                        "statusCheckRollup": [],
                    }
                )
            ),
            _completed_process(stdout=json.dumps([])),
        ]

        control = GitHubControl(repo_root=tmp_path)
        snapshot = control.fetch_gate_snapshot("https://github.com/org/repo/pull/55")

        assert snapshot.disposition == "merged"
        assert snapshot.merge_commit_sha == "merge-sha"

    @patch("aragora.ralph.github_control.subprocess.run")
    def test_fetch_gate_snapshot_waits_for_review(self, mock_run, tmp_path: Path) -> None:
        mock_run.side_effect = [
            _completed_process(
                stdout=json.dumps(
                    {
                        "url": "https://github.com/org/repo/pull/55",
                        "state": "OPEN",
                        "isDraft": False,
                        "headRefName": "codex/test",
                        "baseRefName": "main",
                        "reviewDecision": "REVIEW_REQUIRED",
                        "mergeStateStatus": "BLOCKED",
                        "mergeCommit": None,
                        "statusCheckRollup": [],
                    }
                )
            ),
            _completed_process(stdout=json.dumps([])),
        ]

        control = GitHubControl(repo_root=tmp_path)
        snapshot = control.fetch_gate_snapshot("https://github.com/org/repo/pull/55")

        assert snapshot.disposition == "wait_for_review"

    @patch("aragora.ralph.github_control.subprocess.run")
    def test_fetch_gate_snapshot_waits_for_required_checks(self, mock_run, tmp_path: Path) -> None:
        mock_run.side_effect = [
            _completed_process(
                stdout=json.dumps(
                    {
                        "url": "https://github.com/org/repo/pull/55",
                        "state": "OPEN",
                        "isDraft": False,
                        "headRefName": "codex/test",
                        "baseRefName": "main",
                        "reviewDecision": "APPROVED",
                        "mergeStateStatus": "BLOCKED",
                        "mergeCommit": None,
                        "statusCheckRollup": [
                            {"context": "ci/unit", "state": "PENDING"},
                            {"context": "lint", "state": "SUCCESS"},
                        ],
                    }
                )
            ),
            _completed_process(
                stdout=json.dumps(
                    [
                        {
                            "parameters": {
                                "required_status_checks": [
                                    {"context": "ci/unit"},
                                ]
                            }
                        }
                    ]
                )
            ),
        ]

        control = GitHubControl(repo_root=tmp_path)
        snapshot = control.fetch_gate_snapshot("https://github.com/org/repo/pull/55")

        assert snapshot.disposition == "wait_for_required_checks"
        assert snapshot.required_checks_green is False
        assert [check.name for check in snapshot.required_checks] == ["ci/unit"]

    @patch("aragora.ralph.github_control.subprocess.run")
    def test_fetch_gate_snapshot_ignores_advisory_failures_when_required_green(
        self, mock_run, tmp_path: Path
    ) -> None:
        mock_run.side_effect = [
            _completed_process(
                stdout=json.dumps(
                    {
                        "url": "https://github.com/org/repo/pull/55",
                        "state": "OPEN",
                        "isDraft": False,
                        "headRefName": "codex/test",
                        "baseRefName": "main",
                        "reviewDecision": "APPROVED",
                        "mergeStateStatus": "CLEAN",
                        "mergeCommit": None,
                        "statusCheckRollup": [
                            {"context": "ci/unit", "state": "SUCCESS"},
                            {"context": "lint", "state": "FAILURE"},
                        ],
                    }
                )
            ),
            _completed_process(
                stdout=json.dumps(
                    [
                        {
                            "parameters": {
                                "required_status_checks": [
                                    {"context": "ci/unit"},
                                ]
                            }
                        }
                    ]
                )
            ),
        ]

        control = GitHubControl(repo_root=tmp_path)
        snapshot = control.fetch_gate_snapshot("https://github.com/org/repo/pull/55")

        assert snapshot.disposition == "merge_now"
        assert snapshot.required_checks_green is True
        assert [check.name for check in snapshot.advisory_checks] == ["lint"]

    @patch("aragora.ralph.github_control.subprocess.run")
    def test_fetch_gate_snapshot_fails_closed_when_required_truth_unknown(
        self, mock_run, tmp_path: Path
    ) -> None:
        mock_run.side_effect = [
            _completed_process(
                stdout=json.dumps(
                    {
                        "url": "https://github.com/org/repo/pull/55",
                        "state": "OPEN",
                        "isDraft": False,
                        "headRefName": "codex/test",
                        "baseRefName": "main",
                        "reviewDecision": "APPROVED",
                        "mergeStateStatus": "CLEAN",
                        "mergeCommit": None,
                        "statusCheckRollup": [{"context": "ci/unit", "state": "SUCCESS"}],
                    }
                )
            ),
            _completed_process(returncode=1, stderr="rules api unavailable"),
            _completed_process(returncode=1, stderr="protection api unavailable"),
        ]

        control = GitHubControl(repo_root=tmp_path)
        snapshot = control.fetch_gate_snapshot("https://github.com/org/repo/pull/55")

        assert snapshot.disposition == "blocked_nonreviewable"
        assert snapshot.required_checks_known is False


class TestGitHubControlTaskCoverage:
    @patch("aragora.ralph.github_control.subprocess.run")
    def test_find_pr_for_branch_found(self, mock_run, tmp_path: Path) -> None:
        mock_run.return_value = _completed_process(
            stdout=json.dumps([{"number": 42, "url": "https://github.com/org/repo/pull/42"}])
        )

        control = GitHubControl(repo_root=tmp_path)

        assert control.find_pr_for_branch("codex/test") == "https://github.com/org/repo/pull/42"

    @patch("aragora.ralph.github_control.subprocess.run")
    def test_find_pr_for_branch_not_found(self, mock_run, tmp_path: Path) -> None:
        mock_run.return_value = _completed_process(stdout="[]")

        control = GitHubControl(repo_root=tmp_path)

        assert control.find_pr_for_branch("codex/test") is None

    @patch("aragora.ralph.github_control.subprocess.run")
    def test_create_pr_for_branch_success(self, mock_run, tmp_path: Path) -> None:
        mock_run.return_value = _completed_process(
            stdout=json.dumps({"url": "https://github.com/org/repo/pull/77 "})
        )

        control = GitHubControl(repo_root=tmp_path)

        assert control.create_pr_for_branch("codex/test", "main") == (
            "https://github.com/org/repo/pull/77"
        )

    @patch("aragora.ralph.github_control.subprocess.run")
    def test_create_pr_for_branch_failure(self, mock_run, tmp_path: Path) -> None:
        mock_run.return_value = _completed_process(returncode=1, stderr="auth failed")

        control = GitHubControl(repo_root=tmp_path)

        with pytest.raises(GitHubControlError, match="auth failed"):
            control.create_pr_for_branch("codex/test", "main")

    @patch("aragora.ralph.github_control.subprocess.run")
    def test_fetch_gate_snapshot_parses_rulesets(self, mock_run, tmp_path: Path) -> None:
        mock_run.side_effect = [
            _completed_process(
                stdout=json.dumps(
                    {
                        "url": "https://github.com/org/repo/pull/55",
                        "state": "OPEN",
                        "isDraft": False,
                        "headRefName": "codex/test",
                        "baseRefName": "main",
                        "reviewDecision": "APPROVED",
                        "mergeStateStatus": "BLOCKED",
                        "mergeCommit": None,
                        "statusCheckRollup": [
                            {"context": "ci/unit", "state": "SUCCESS"},
                            {"context": "lint", "state": "PENDING"},
                            {"context": "coverage", "state": "SUCCESS"},
                        ],
                    }
                )
            ),
            _completed_process(
                stdout=json.dumps(
                    [
                        {
                            "parameters": {
                                "required_status_checks": [
                                    {"context": "ci/unit"},
                                    {"context": "lint"},
                                ]
                            }
                        }
                    ]
                )
            ),
        ]

        control = GitHubControl(repo_root=tmp_path)
        snapshot = control.fetch_gate_snapshot("https://github.com/org/repo/pull/55")

        assert [check.name for check in snapshot.required_checks] == ["ci/unit", "lint"]
        assert snapshot.required_checks_source == "ruleset"
        assert snapshot.required_checks_known is True

    @patch("aragora.ralph.github_control.subprocess.run")
    def test_fetch_gate_snapshot_draft_disposition(self, mock_run, tmp_path: Path) -> None:
        mock_run.side_effect = [
            _completed_process(
                stdout=json.dumps(
                    {
                        "url": "https://github.com/org/repo/pull/56",
                        "state": "OPEN",
                        "isDraft": True,
                        "headRefName": "codex/test",
                        "baseRefName": "main",
                        "reviewDecision": "APPROVED",
                        "mergeStateStatus": "CLEAN",
                        "mergeCommit": None,
                        "statusCheckRollup": [{"context": "ci/unit", "state": "SUCCESS"}],
                    }
                )
            ),
            _completed_process(
                stdout=json.dumps(
                    [
                        {
                            "parameters": {
                                "required_status_checks": [
                                    {"context": "ci/unit"},
                                ]
                            }
                        }
                    ]
                )
            ),
        ]

        control = GitHubControl(repo_root=tmp_path)
        snapshot = control.fetch_gate_snapshot("https://github.com/org/repo/pull/56")

        assert snapshot.draft is True
        assert snapshot.disposition == "wait_for_review"


class TestGitHubControlMerge:
    @patch("aragora.ralph.github_control.subprocess.run")
    def test_merge_pr_uses_normal_merge_first(self, mock_run, tmp_path: Path) -> None:
        mock_run.return_value = _completed_process(stdout="merged")

        control = GitHubControl(repo_root=tmp_path)
        result = control.merge_pr(
            "https://github.com/org/repo/pull/88",
            required_checks_green=True,
            allow_admin=True,
            head_sha=_HEAD_SHA,
        )

        assert result.merged is True
        assert result.used_admin is False
        called = mock_run.call_args.args[0]
        assert called == [
            "gh",
            "pr",
            "merge",
            "https://github.com/org/repo/pull/88",
            "--squash",
            "--match-head-commit",
            _HEAD_SHA,
        ]

    @patch("aragora.ralph.github_control.subprocess.run")
    def test_merge_pr_falls_back_to_admin_when_needed(self, mock_run, tmp_path: Path) -> None:
        mock_run.side_effect = [
            _completed_process(
                returncode=1, stderr="Repository rules require administrator override"
            ),
            _completed_process(stdout="merged with admin"),
        ]

        control = GitHubControl(repo_root=tmp_path)
        result = control.merge_pr(
            "https://github.com/org/repo/pull/88",
            required_checks_green=True,
            allow_admin=True,
            head_sha=_HEAD_SHA,
        )

        assert result.merged is True
        assert result.used_admin is True
        assert mock_run.call_args_list[1].args[0] == [
            "gh",
            "pr",
            "merge",
            "https://github.com/org/repo/pull/88",
            "--squash",
            "--admin",
            "--match-head-commit",
            _HEAD_SHA,
        ]

    @patch("aragora.ralph.github_control.subprocess.run")
    def test_merge_pr_does_not_attempt_admin_without_signal(self, mock_run, tmp_path: Path) -> None:
        mock_run.return_value = _completed_process(returncode=1, stderr="merge conflict")

        control = GitHubControl(repo_root=tmp_path)
        result = control.merge_pr(
            "https://github.com/org/repo/pull/88",
            required_checks_green=True,
            allow_admin=True,
            head_sha=_HEAD_SHA,
        )

        assert result.merged is False
        assert result.used_admin is False
        assert mock_run.call_count == 1

    @patch("aragora.ralph.github_control.subprocess.run")
    def test_merge_pr_blocks_when_required_checks_not_green(self, mock_run, tmp_path: Path) -> None:
        control = GitHubControl(repo_root=tmp_path)
        result = control.merge_pr(
            "https://github.com/org/repo/pull/88",
            required_checks_green=False,
            allow_admin=True,
        )

        assert result.merged is False
        assert result.action == "blocked"
        assert mock_run.call_count == 0


class TestMergeIsBoundToTheAuthorizedHead:
    """#9216: the guard authorizes a specific head, so the merge must pin to it.

    Without --match-head-commit the head can change between assert_merge_allowed
    and the merge itself, letting an unwaived head land under an armed halt.
    """

    @patch("aragora.ralph.github_control.subprocess.run")
    def test_merge_pins_the_head_it_was_authorized_for(self, mock_run, tmp_path: Path) -> None:
        mock_run.return_value = _completed_process(returncode=0, stdout="merged")
        head = "a" * 40

        control = GitHubControl(repo_root=tmp_path)
        control.merge_pr(
            "https://github.com/org/repo/pull/42",
            required_checks_green=True,
            allow_admin=False,
            head_sha=head,
        )

        argv = [c for c in mock_run.call_args[0][0]]
        assert "--match-head-commit" in argv, f"merge argv is not head-bound: {argv}"
        assert argv[argv.index("--match-head-commit") + 1] == head

    @patch("aragora.ralph.github_control.subprocess.run")
    def test_head_change_rejection_does_not_fall_back_to_admin(
        self, mock_run, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "aragora.ralph.github_control.evaluate",
            lambda *_a, **_k: SimpleNamespace(allowed=True, reason="no halt"),
        )
        mock_run.return_value = _completed_process(
            returncode=1,
            stderr="Pull request head branch was modified",
        )

        control = GitHubControl(repo_root=tmp_path)
        result = control.merge_pr(
            "https://github.com/org/repo/pull/42",
            required_checks_green=True,
            allow_admin=True,
            head_sha=_HEAD_SHA,
        )

        assert result.merged is False
        assert result.action == "merge_failed"
        assert mock_run.call_count == 1
        assert list(mock_run.call_args.args[0])[-2:] == ["--match-head-commit", _HEAD_SHA]

    @patch("aragora.ralph.github_control.subprocess.run")
    @pytest.mark.parametrize("head_sha", [None, "", "abc123", "g" * 40, "a" * 39, "a" * 41])
    def test_merge_without_a_full_snapshot_head_fails_closed(
        self, mock_run, tmp_path: Path, head_sha: str | None, monkeypatch
    ) -> None:
        evaluate = MagicMock()
        monkeypatch.setattr("aragora.ralph.github_control.evaluate", evaluate)

        control = GitHubControl(repo_root=tmp_path)
        result = control.merge_pr(
            "https://github.com/org/repo/pull/42",
            required_checks_green=True,
            allow_admin=True,
            head_sha=head_sha,
        )

        assert result.merged is False
        assert result.action == "blocked"
        assert "full 40-character" in result.detail
        evaluate.assert_not_called()
        mock_run.assert_not_called()


def test_halt_and_disarm_markers_share_one_root() -> None:
    """A disarm placed beside the halt must actually stop a linked-worktree run.

    Moving DEFAULT_HALT_FILE to the shared checkout while DEFAULT_DISARM_FILE
    stayed worktree-local made the strictest control fail open: operators are told
    the markers live at the primary checkout, but the disarm would be looked for
    somewhere else. Both must resolve identically.
    """
    import scripts.merge_executor as merge_executor

    assert merge_executor.DEFAULT_DISARM_FILE.parent == merge_executor.DEFAULT_HALT_FILE.parent


class TestSnapshotHeadAuthorization:
    """#9216: checks, halt authorization, and merge share one exact head."""

    @patch("aragora.ralph.github_control.subprocess.run")
    def test_armed_exact_head_waiver_merges_that_head(
        self, mock_run, tmp_path: Path, monkeypatch
    ) -> None:
        head = "b" * 40
        seen: list[str] = []

        def fake_evaluate(pr: int, head_sha: str, **_kw):
            seen.append(head_sha)
            allowed = head_sha == head
            return SimpleNamespace(allowed=allowed, reason="halted" if not allowed else "ok")

        monkeypatch.setattr("aragora.ralph.github_control.evaluate", fake_evaluate)
        mock_run.return_value = _completed_process(returncode=0, stdout="merged")

        control = GitHubControl(repo_root=tmp_path)
        result = control.merge_pr(
            "https://github.com/org/repo/pull/42",
            required_checks_green=True,
            allow_admin=False,
            head_sha=head,
        )

        assert seen == [head]
        assert result.merged is True
        argv = list(mock_run.call_args[0][0])
        assert "--match-head-commit" in argv and head in argv

    @patch("aragora.ralph.github_control.subprocess.run")
    def test_unhalted_merge_is_still_pinned_to_the_snapshot_head(
        self, mock_run, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "aragora.ralph.github_control.evaluate",
            lambda *_a, **_k: SimpleNamespace(allowed=True, reason="no halt"),
        )
        mock_run.return_value = _completed_process(returncode=0, stdout="merged")

        control = GitHubControl(repo_root=tmp_path)
        control.merge_pr(
            "https://github.com/org/repo/pull/42",
            required_checks_green=True,
            allow_admin=False,
            head_sha=_HEAD_SHA,
        )

        assert mock_run.call_count == 1
        assert list(mock_run.call_args.args[0])[-2:] == ["--match-head-commit", _HEAD_SHA]

    @patch("aragora.ralph.github_control.subprocess.run")
    def test_normal_and_admin_attempts_use_the_same_snapshot_head(
        self, mock_run, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "aragora.ralph.github_control.evaluate",
            lambda *_a, **_k: SimpleNamespace(allowed=True, reason="no halt"),
        )
        mock_run.side_effect = [
            _completed_process(returncode=1, stderr="repository rules require admin"),
            _completed_process(returncode=0, stdout="merged"),
        ]

        control = GitHubControl(repo_root=tmp_path)
        result = control.merge_pr(
            "https://github.com/org/repo/pull/42",
            required_checks_green=True,
            allow_admin=True,
            head_sha=_HEAD_SHA,
        )

        assert result.merged is True
        assert result.used_admin is True
        for call in mock_run.call_args_list:
            argv = list(call.args[0])
            assert argv[-2:] == ["--match-head-commit", _HEAD_SHA]
