"""Regression tests for Boss-loop deliverable qualification (#891).

Covers the bug where Boss loop reported an issue as completed even though
the only artifact was a dirty local worktree with no pushed branch or PR.

The fix requires _dispatch_issue to check for a concrete deliverable
(PR URL, pushed branch with commits, or adopted PR reference) before
accepting a run as completed.
"""

from __future__ import annotations

from typing import Any

import pytest

from aragora.swarm.boss_loop import _extract_deliverable


class TestExtractDeliverable:
    """Unit tests for _extract_deliverable gate function."""

    def test_no_work_orders_returns_none(self) -> None:
        run_dict: dict[str, Any] = {"work_orders": []}
        assert _extract_deliverable(run_dict) is None

    def test_dirty_worktree_only_returns_none(self) -> None:
        """Exact shape from boss-7e2ae05d15b3: completed work order with
        changed paths but no branch, no PR, no commits."""
        run_dict: dict[str, Any] = {
            "work_orders": [
                {
                    "work_order_id": "wo-1",
                    "status": "completed",
                    "worktree_path": "/tmp/swarm-work-49d",
                    "changed_paths": ["aragora/live/package.json"],
                    "commit_shas": [],
                    "branch": "",
                    "pr_url": "",
                },
            ],
        }
        assert _extract_deliverable(run_dict) is None

    def test_pr_url_is_concrete_deliverable(self) -> None:
        run_dict: dict[str, Any] = {
            "work_orders": [
                {
                    "work_order_id": "wo-1",
                    "status": "completed",
                    "pr_url": "https://github.com/synaptent/aragora/pull/857",
                    "branch": "fix/eslintrc",
                    "commit_shas": ["abc123"],
                },
            ],
        }
        result = _extract_deliverable(run_dict)
        assert result is not None
        assert result["type"] == "pr"
        assert result["pr_url"] == "https://github.com/synaptent/aragora/pull/857"

    def test_branch_with_commits_is_concrete_deliverable(self) -> None:
        run_dict: dict[str, Any] = {
            "work_orders": [
                {
                    "work_order_id": "wo-1",
                    "status": "completed",
                    "branch": "codex/fix-eslintrc",
                    "commit_shas": ["abc123", "def456"],
                    "pr_url": "",
                },
            ],
        }
        result = _extract_deliverable(run_dict)
        assert result is not None
        assert result["type"] == "branch"
        assert result["branch"] == "codex/fix-eslintrc"
        assert result["commit_shas"] == ["abc123", "def456"]

    def test_adopted_pr_is_concrete_deliverable(self) -> None:
        run_dict: dict[str, Any] = {
            "work_orders": [
                {
                    "work_order_id": "wo-1",
                    "status": "completed",
                    "adopted_pr": "https://github.com/synaptent/aragora/pull/857",
                },
            ],
        }
        result = _extract_deliverable(run_dict)
        assert result is not None
        assert result["type"] == "adopted_pr"

    def test_branch_without_commits_returns_none(self) -> None:
        """A branch name alone is not a deliverable — commits prove work was pushed."""
        run_dict: dict[str, Any] = {
            "work_orders": [
                {
                    "work_order_id": "wo-1",
                    "status": "completed",
                    "branch": "codex/fix-eslintrc",
                    "commit_shas": [],
                    "pr_url": "",
                },
            ],
        }
        assert _extract_deliverable(run_dict) is None

    def test_failed_work_order_ignored(self) -> None:
        """Only completed/merged work orders count as deliverables."""
        run_dict: dict[str, Any] = {
            "work_orders": [
                {
                    "work_order_id": "wo-1",
                    "status": "failed",
                    "pr_url": "https://github.com/synaptent/aragora/pull/857",
                },
            ],
        }
        assert _extract_deliverable(run_dict) is None

    def test_merged_work_order_with_pr_is_deliverable(self) -> None:
        run_dict: dict[str, Any] = {
            "work_orders": [
                {
                    "work_order_id": "wo-1",
                    "status": "merged",
                    "pr_url": "https://github.com/synaptent/aragora/pull/857",
                },
            ],
        }
        result = _extract_deliverable(run_dict)
        assert result is not None
        assert result["type"] == "pr"

    def test_multiple_work_orders_first_deliverable_wins(self) -> None:
        """If multiple work orders have deliverables, the first one is returned."""
        run_dict: dict[str, Any] = {
            "work_orders": [
                {
                    "work_order_id": "wo-1",
                    "status": "completed",
                    "branch": "",
                    "commit_shas": [],
                    "pr_url": "",
                },
                {
                    "work_order_id": "wo-2",
                    "status": "completed",
                    "pr_url": "https://github.com/synaptent/aragora/pull/999",
                },
            ],
        }
        result = _extract_deliverable(run_dict)
        assert result is not None
        assert result["work_order_id"] == "wo-2"

    def test_missing_work_orders_key_returns_none(self) -> None:
        assert _extract_deliverable({}) is None

    def test_non_dict_work_orders_skipped(self) -> None:
        run_dict: dict[str, Any] = {"work_orders": ["not_a_dict", 42, None]}
        assert _extract_deliverable(run_dict) is None
