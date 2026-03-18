from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.ralph.github_control import GitHubControl, GitHubControlError


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


class TestGitHubControlMerge:
    @patch("aragora.ralph.github_control.subprocess.run")
    def test_merge_pr_uses_normal_merge_first(self, mock_run, tmp_path: Path) -> None:
        mock_run.return_value = _completed_process(stdout="merged")

        control = GitHubControl(repo_root=tmp_path)
        result = control.merge_pr(
            "https://github.com/org/repo/pull/88",
            required_checks_green=True,
            allow_admin=True,
        )

        assert result.merged is True
        assert result.used_admin is False
        called = mock_run.call_args.args[0]
        assert called == ["gh", "pr", "merge", "https://github.com/org/repo/pull/88", "--squash"]

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
        ]

    @patch("aragora.ralph.github_control.subprocess.run")
    def test_merge_pr_does_not_attempt_admin_without_signal(self, mock_run, tmp_path: Path) -> None:
        mock_run.return_value = _completed_process(returncode=1, stderr="merge conflict")

        control = GitHubControl(repo_root=tmp_path)
        result = control.merge_pr(
            "https://github.com/org/repo/pull/88",
            required_checks_green=True,
            allow_admin=True,
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
