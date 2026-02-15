"""
Tests for Autonomous DevOps Agent.
"""

import json
import os
import pytest
from unittest.mock import patch, MagicMock

from aragora.agents.devops.agent import (
    DevOpsAgent,
    DevOpsAgentConfig,
    DevOpsTask,
    TaskResult,
    AuditEntry,
    ALLOWED_COMMANDS,
    BLOCKED_COMMANDS,
    DESTRUCTIVE_COMMANDS,
)


# ── Config Tests ───────────────────────────────────────────────────


class TestDevOpsAgentConfig:
    def test_defaults(self):
        config = DevOpsAgentConfig()
        assert config.repo == ""
        assert config.poll_interval == 300
        assert config.max_prs_per_run == 5
        assert config.allow_destructive is False
        assert config.dry_run is False

    def test_from_env(self):
        env = {
            "ARAGORA_DEVOPS_REPO": "an0mium/aragora",
            "ARAGORA_DEVOPS_POLL_INTERVAL": "600",
            "ARAGORA_DEVOPS_AGENTS": "anthropic-api",
            "ARAGORA_DEVOPS_ALLOW_DESTRUCTIVE": "true",
            "ARAGORA_DEVOPS_DRY_RUN": "true",
            "GITHUB_TOKEN": "ghp_test123",
        }
        with patch.dict(os.environ, env, clear=False):
            config = DevOpsAgentConfig.from_env()
        assert config.repo == "an0mium/aragora"
        assert config.poll_interval == 600
        assert config.review_agents == "anthropic-api"
        assert config.allow_destructive is True
        assert config.dry_run is True
        assert config.github_token == "ghp_test123"

    def test_from_env_defaults(self):
        env = {}
        with patch.dict(os.environ, env, clear=True):
            config = DevOpsAgentConfig.from_env()
        assert config.repo == ""
        assert config.poll_interval == 300
        assert config.allow_destructive is False


# ── Command Validation Tests ──────────────────────────────────────


class TestCommandValidation:
    def setup_method(self):
        self.config = DevOpsAgentConfig(repo="test/repo", dry_run=True)
        self.agent = DevOpsAgent(self.config)

    def test_allowed_commands(self):
        """All ALLOWED_COMMANDS prefixes should pass validation."""
        for cmd in ALLOWED_COMMANDS:
            ok, reason = self.agent._validate_command(cmd + "something")
            # Destructive commands fail without allow_destructive
            if any(cmd.startswith(d) for d in DESTRUCTIVE_COMMANDS):
                continue
            assert ok, f"Expected '{cmd}' to be allowed, got: {reason}"

    def test_blocked_commands(self):
        """All BLOCKED_COMMANDS should be rejected."""
        for cmd in BLOCKED_COMMANDS:
            ok, reason = self.agent._validate_command(cmd + "something")
            assert not ok, f"Expected '{cmd}' to be blocked"
            assert "blocked" in reason.lower()

    def test_unknown_command_rejected(self):
        ok, reason = self.agent._validate_command("unknown_binary --flag")
        assert not ok
        assert "not in allowlist" in reason

    def test_destructive_without_flag(self):
        ok, reason = self.agent._validate_command("twine upload dist/*")
        assert not ok
        assert "allow-destructive" in reason

    def test_destructive_with_flag(self):
        self.config.allow_destructive = True
        ok, reason = self.agent._validate_command("twine upload dist/*")
        assert ok
        assert reason == "allowed"

    def test_gh_pr_list_allowed(self):
        ok, _ = self.agent._validate_command("gh pr list -R test/repo --state open --json number")
        assert ok

    def test_gh_pr_merge_blocked_without_destructive(self):
        ok, reason = self.agent._validate_command("gh pr merge 123 -R test/repo")
        assert not ok
        assert "allow-destructive" in reason

    def test_aragora_review_allowed(self):
        ok, _ = self.agent._validate_command(
            "aragora review --diff-file /tmp/pr.diff --output-format json"
        )
        assert ok

    def test_rm_always_blocked(self):
        ok, _ = self.agent._validate_command("rm -rf /")
        assert not ok

    def test_sudo_always_blocked(self):
        ok, _ = self.agent._validate_command("sudo anything")
        assert not ok

    def test_curl_blocked(self):
        ok, _ = self.agent._validate_command("curl https://evil.com")
        assert not ok

    def test_git_diff_allowed(self):
        ok, _ = self.agent._validate_command("git diff HEAD~1")
        assert ok

    def test_git_push_blocked_without_destructive(self):
        ok, reason = self.agent._validate_command("git push origin main")
        assert not ok
        assert "allow-destructive" in reason


# ── Dry Run Tests ─────────────────────────────────────────────────


class TestDryRun:
    def setup_method(self):
        self.config = DevOpsAgentConfig(repo="test/repo", dry_run=True)
        self.agent = DevOpsAgent(self.config)

    def test_dry_run_does_not_execute(self):
        ok, output = self.agent._execute("gh pr list -R test/repo --state open", "test_action")
        assert ok
        assert output == "[dry run]"

    def test_dry_run_logs_audit(self):
        self.agent._execute("gh pr list -R test/repo", "test_action")
        assert len(self.agent.audit_log) == 1
        assert self.agent.audit_log[0].outcome == "dry_run"

    def test_blocked_commands_still_blocked_in_dry_run(self):
        ok, _ = self.agent._execute("rm -rf /", "test_blocked")
        assert not ok
        assert self.agent.audit_log[0].outcome == "blocked"


# ── Audit Logging Tests ──────────────────────────────────────────


class TestAuditLogging:
    def setup_method(self):
        self.config = DevOpsAgentConfig(repo="test/repo", dry_run=True)
        self.agent = DevOpsAgent(self.config)

    def test_audit_on_allowed(self):
        self.agent._execute("gh auth status", "check_auth")
        assert len(self.agent.audit_log) == 1
        entry = self.agent.audit_log[0]
        assert entry.action == "check_auth"
        assert entry.outcome == "dry_run"
        assert entry.timestamp

    def test_audit_on_blocked(self):
        self.agent._execute("rm -rf /", "dangerous")
        entry = self.agent.audit_log[0]
        assert entry.outcome == "blocked"
        assert "blocked" in entry.detail.lower()

    def test_export_audit_log(self):
        self.agent._execute("gh auth status", "auth")
        self.agent._execute("rm evil", "evil")
        exported = self.agent.export_audit_log()
        assert len(exported) == 2
        assert exported[0]["action"] == "auth"
        assert exported[1]["action"] == "evil"
        assert all(isinstance(e, dict) for e in exported)

    def test_command_truncated_in_audit(self):
        long_cmd = "gh pr list " + "x" * 300
        self.agent._execute(long_cmd, "long_cmd")
        assert len(self.agent.audit_log[0].command) <= 200


# ── PR Review Tests ───────────────────────────────────────────────


class TestReviewPRs:
    def setup_method(self):
        self.config = DevOpsAgentConfig(repo="test/repo", dry_run=True)
        self.agent = DevOpsAgent(self.config)

    def test_review_prs_dry_run(self):
        result = self.agent.review_prs()
        assert isinstance(result, TaskResult)
        assert result.task == "review-prs"
        assert result.duration_seconds >= 0
        assert result.completed_at is not None

    def test_review_prs_skips_reviewed(self):
        """PRs with aragora-reviewed label should be skipped."""
        prs_json = json.dumps(
            [
                {"number": 1, "title": "Test PR", "labels": [{"name": "aragora-reviewed"}]},
                {"number": 2, "title": "New PR", "labels": []},
            ]
        )
        with patch.object(self.agent, "_execute") as mock_exec:
            mock_exec.side_effect = [
                (True, prs_json),  # list PRs
                (True, "diff content"),  # get diff for PR #2
                (True, "{}"),  # run review
                (True, ""),  # post comment
                (True, ""),  # add label
            ]
            self.config.dry_run = False
            result = self.agent.review_prs()
        assert result.items_skipped == 1

    def test_format_review_comment(self):
        review_data = json.dumps(
            {
                "findings": [
                    {"severity": "high", "message": "SQL injection risk"},
                    {"severity": "low", "message": "Consider renaming variable"},
                ]
            }
        )
        comment = self.agent._format_review_comment(review_data, 42)
        assert "Aragora AI Review" in comment
        assert "SQL injection risk" in comment
        assert "HIGH" in comment
        assert "PR #42" in comment
        assert "OpenClaw" in comment

    def test_format_review_comment_no_findings(self):
        comment = self.agent._format_review_comment("{}", 1)
        assert "No significant findings" in comment

    def test_format_review_comment_invalid_json(self):
        comment = self.agent._format_review_comment("not json", 1)
        assert "No significant findings" in comment


# ── Issue Triage Tests ────────────────────────────────────────────


class TestTriageIssues:
    def setup_method(self):
        self.config = DevOpsAgentConfig(repo="test/repo", dry_run=True)
        self.agent = DevOpsAgent(self.config)

    def test_classify_issue_bug(self):
        labels = self.agent._classify_issue(
            "Error when clicking submit", "The app crashes with a traceback"
        )
        assert "bug" in labels

    def test_classify_issue_enhancement(self):
        labels = self.agent._classify_issue(
            "Feature request: dark mode", "Please add dark mode support"
        )
        assert "enhancement" in labels

    def test_classify_issue_security(self):
        labels = self.agent._classify_issue(
            "XSS vulnerability in form", "Found an injection exploit"
        )
        assert "security" in labels

    def test_classify_issue_docs(self):
        labels = self.agent._classify_issue("README typo", "The documentation has a typo")
        assert "documentation" in labels

    def test_classify_issue_question(self):
        labels = self.agent._classify_issue(
            "How to configure agents?", "Help needed: how do I set up?"
        )
        assert "question" in labels

    def test_classify_issue_max_3_labels(self):
        labels = self.agent._classify_issue(
            "Bug: slow error crash feature request question docs security",
            "Everything at once",
        )
        assert len(labels) <= 3

    def test_classify_issue_no_match(self):
        labels = self.agent._classify_issue("Miscellaneous", "Nothing specific here")
        assert labels == []

    def test_triage_issues_dry_run(self):
        result = self.agent.triage_issues()
        assert result.task == "triage-issues"
        assert result.success

    def test_triage_skips_already_triaged(self):
        issues_json = json.dumps(
            [
                {"number": 1, "title": "Old issue", "body": "", "labels": [{"name": "triaged"}]},
            ]
        )
        with patch.object(self.agent, "_execute") as mock_exec:
            mock_exec.return_value = (True, issues_json)
            self.config.dry_run = False
            result = self.agent.triage_issues()
        assert result.items_skipped == 1
        assert result.items_processed == 0


# ── Health Check Tests ────────────────────────────────────────────


class TestHealthCheck:
    def setup_method(self):
        self.config = DevOpsAgentConfig(repo="test/repo", dry_run=True)
        self.agent = DevOpsAgent(self.config)

    def test_health_check_dry_run(self):
        result = self.agent.health_check()
        assert result.task == "health-check"
        assert result.success
        # Should have attempted multiple checks
        assert len(self.agent.audit_log) >= 1


# ── Release Tests ─────────────────────────────────────────────────


class TestPrepareRelease:
    def setup_method(self):
        self.config = DevOpsAgentConfig(repo="test/repo", dry_run=True)
        self.agent = DevOpsAgent(self.config)

    def test_prepare_release_dry_run(self):
        result = self.agent.prepare_release(version="1.0.0")
        assert result.task == "prepare-release"
        assert result.success

    def test_prepare_release_test_failure_stops_pipeline(self):
        """If tests fail, build should not run."""
        call_count = 0

        def mock_execute(cmd, action):
            nonlocal call_count
            call_count += 1
            if "pytest" in cmd:
                return False, "FAILED: 3 errors"
            return True, "ok"

        with patch.object(self.agent, "_execute", side_effect=mock_execute):
            self.config.dry_run = False
            result = self.agent.prepare_release(version="1.0.0")

        assert not result.success
        assert any("Tests failed" in e for e in result.errors)


# ── Task Dispatcher Tests ─────────────────────────────────────────


class TestRunTask:
    def setup_method(self):
        self.config = DevOpsAgentConfig(repo="test/repo", dry_run=True)
        self.agent = DevOpsAgent(self.config)

    def test_dispatch_review_prs(self):
        result = self.agent.run_task(DevOpsTask.REVIEW_PRS)
        assert result.task == "review-prs"

    def test_dispatch_triage_issues(self):
        result = self.agent.run_task(DevOpsTask.TRIAGE_ISSUES)
        assert result.task == "triage-issues"

    def test_dispatch_health_check(self):
        result = self.agent.run_task(DevOpsTask.HEALTH_CHECK)
        assert result.task == "health-check"

    def test_dispatch_prepare_release(self):
        result = self.agent.run_task(DevOpsTask.PREPARE_RELEASE, version="1.0.0")
        assert result.task == "prepare-release"


# ── TaskResult Tests ──────────────────────────────────────────────


class TestTaskResult:
    def test_defaults(self):
        result = TaskResult(task="test", success=True)
        assert result.items_processed == 0
        assert result.items_skipped == 0
        assert result.errors == []
        assert result.details == []
        assert result.completed_at is None

    def test_started_at_auto_set(self):
        result = TaskResult(task="test", success=True)
        assert result.started_at is not None


# ── Integration-Style Tests ───────────────────────────────────────


class TestEndToEnd:
    """End-to-end tests using mocked subprocess."""

    def test_full_review_workflow(self):
        """Simulate a full PR review cycle with mocked gh/aragora calls."""
        config = DevOpsAgentConfig(repo="test/repo", dry_run=False)
        agent = DevOpsAgent(config)

        pr_list = json.dumps(
            [
                {"number": 42, "title": "Add feature X", "labels": []},
            ]
        )
        review_output = json.dumps(
            {
                "findings": [
                    {"severity": "medium", "message": "Consider input validation"},
                ]
            }
        )

        responses = [
            (True, pr_list),  # list PRs
            (True, "diff content\n"),  # get diff
            (True, review_output),  # run review
            (True, ""),  # post comment
            (True, ""),  # add label
        ]
        call_idx = 0

        def mock_execute(cmd, action):
            nonlocal call_idx
            if call_idx < len(responses):
                r = responses[call_idx]
                call_idx += 1
                return r
            return True, ""

        with patch.object(agent, "_execute", side_effect=mock_execute):
            result = agent.review_prs()

        assert result.success
        assert result.items_processed == 1

    def test_full_triage_workflow(self):
        """Simulate issue triage with mocked gh calls."""
        config = DevOpsAgentConfig(repo="test/repo", dry_run=False)
        agent = DevOpsAgent(config)

        issue_list = json.dumps(
            [
                {
                    "number": 10,
                    "title": "Bug: crash on startup",
                    "body": "App crashes with error",
                    "labels": [],
                },
                {
                    "number": 11,
                    "title": "Add dark mode",
                    "body": "Feature request for UI",
                    "labels": [],
                },
            ]
        )

        call_idx = 0
        responses = [
            (True, issue_list),  # list issues
            (True, ""),  # label issue 10
            (True, ""),  # label issue 11
        ]

        def mock_execute(cmd, action):
            nonlocal call_idx
            if call_idx < len(responses):
                r = responses[call_idx]
                call_idx += 1
                return r
            return True, ""

        with patch.object(agent, "_execute", side_effect=mock_execute):
            result = agent.triage_issues()

        assert result.success
        assert result.items_processed == 2
        assert result.details[0]["labels"] == ["bug"]
        assert "enhancement" in result.details[1]["labels"]
