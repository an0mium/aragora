"""Tests for PlanExecutor GitHub integration (issue/PR creation)."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from aragora.pipeline.executor import PlanExecutor
from aragora.pipeline.decision_plan import DecisionPlan, PlanStatus


def _make_plan(
    *,
    status: PlanStatus = PlanStatus.APPROVED,
    task: str = "Refactor authentication module",
    debate_id: str = "debate-123",
    plan_id: str = "plan-456",
    with_tasks: bool = True,
    with_risks: bool = True,
    with_verification: bool = True,
) -> DecisionPlan:
    """Create a minimal DecisionPlan for testing."""
    plan = MagicMock(spec=DecisionPlan)
    plan.id = plan_id
    plan.debate_id = debate_id
    plan.task = task
    plan.status = status
    plan.metadata = {}

    # Implementation tasks
    if with_tasks:
        task1 = MagicMock()
        task1.description = "Extract auth logic into separate module"
        task2 = MagicMock()
        task2.description = "Add unit tests for auth module"
        impl = MagicMock()
        impl.tasks = [task1, task2]
        plan.implement_plan = impl
    else:
        plan.implement_plan = None

    # Risk register
    if with_risks:
        risk1 = MagicMock()
        risk1.title = "Breaking API changes"
        risk1.level = MagicMock()
        risk1.level.value = "HIGH"
        risk1.mitigation = "Version the API endpoints"
        risk2 = MagicMock()
        risk2.title = "Test coverage regression"
        risk2.level = MagicMock()
        risk2.level.value = "MEDIUM"
        risk2.mitigation = "Run full test suite before merge"
        register = MagicMock()
        register.risks = [risk1, risk2]
        plan.risk_register = register
    else:
        plan.risk_register = None

    # Verification plan
    if with_verification:
        case1 = MagicMock()
        case1.description = "Auth endpoints return 200"
        case2 = MagicMock()
        case2.description = "Rate limiting still works"
        vp = MagicMock()
        vp.cases = [case1, case2]
        plan.verification_plan = vp
    else:
        plan.verification_plan = None

    return plan


class TestBuildIssueBody:
    """Test _build_issue_body generates correct markdown."""

    def test_includes_debate_and_plan_ids(self):
        executor = PlanExecutor()
        plan = _make_plan()
        body = executor._build_issue_body(plan)
        assert "debate-123" in body
        assert "plan-456" in body

    def test_includes_task_description(self):
        executor = PlanExecutor()
        plan = _make_plan(task="Improve error handling")
        body = executor._build_issue_body(plan)
        assert "Improve error handling" in body

    def test_includes_implementation_tasks_as_checkboxes(self):
        executor = PlanExecutor()
        plan = _make_plan()
        body = executor._build_issue_body(plan)
        assert "- [ ] Extract auth logic into separate module" in body
        assert "- [ ] Add unit tests for auth module" in body

    def test_includes_risk_checklist_with_severity(self):
        executor = PlanExecutor()
        plan = _make_plan()
        body = executor._build_issue_body(plan)
        assert "**[HIGH]** Breaking API changes" in body
        assert "**[MEDIUM]** Test coverage regression" in body

    def test_includes_risk_mitigation(self):
        executor = PlanExecutor()
        plan = _make_plan()
        body = executor._build_issue_body(plan)
        assert "Version the API endpoints" in body

    def test_includes_verification_plan(self):
        executor = PlanExecutor()
        plan = _make_plan()
        body = executor._build_issue_body(plan)
        assert "- [ ] Auth endpoints return 200" in body
        assert "- [ ] Rate limiting still works" in body

    def test_omits_sections_when_empty(self):
        executor = PlanExecutor()
        plan = _make_plan(with_tasks=False, with_risks=False, with_verification=False)
        body = executor._build_issue_body(plan)
        assert "Implementation Tasks" not in body
        assert "Risk Checklist" not in body
        assert "Verification Plan" not in body

    def test_includes_footer(self):
        executor = PlanExecutor()
        plan = _make_plan()
        body = executor._build_issue_body(plan)
        assert "Aragora Decision Pipeline" in body


class TestExecuteToGithubIssue:
    """Test execute_to_github_issue validates status and calls gh CLI."""

    def test_rejects_rejected_plan(self):
        executor = PlanExecutor()
        plan = _make_plan(status=PlanStatus.REJECTED)
        with pytest.raises(ValueError, match="APPROVED or CREATED"):
            executor.execute_to_github_issue(plan)

    def test_rejects_executing_plan(self):
        executor = PlanExecutor()
        plan = _make_plan(status=PlanStatus.EXECUTING)
        with pytest.raises(ValueError, match="APPROVED or CREATED"):
            executor.execute_to_github_issue(plan)

    def test_accepts_approved_plan(self):
        executor = PlanExecutor()
        plan = _make_plan(status=PlanStatus.APPROVED)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="https://github.com/org/repo/issues/42\n",
                stderr="",
            )
            result = executor.execute_to_github_issue(plan)
            assert result["url"] == "https://github.com/org/repo/issues/42"
            assert result["number"] == 42
            assert result["plan_id"] == "plan-456"

    def test_accepts_created_plan(self):
        executor = PlanExecutor()
        plan = _make_plan(status=PlanStatus.CREATED)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="https://github.com/org/repo/issues/1\n",
                stderr="",
            )
            result = executor.execute_to_github_issue(plan)
            assert result["url"] == "https://github.com/org/repo/issues/1"

    def test_passes_repo_labels_assignees(self):
        executor = PlanExecutor()
        plan = _make_plan()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="https://github.com/org/repo/issues/10\n",
                stderr="",
            )
            executor.execute_to_github_issue(
                plan,
                repo="org/repo",
                labels=["aragora", "decision"],
                assignees=["alice"],
            )
            cmd = mock_run.call_args[0][0]
            assert "--repo" in cmd
            assert "org/repo" in cmd
            assert cmd.count("--label") == 2
            assert "--assignee" in cmd

    def test_handles_gh_not_installed(self):
        executor = PlanExecutor()
        plan = _make_plan()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = executor.execute_to_github_issue(plan)
            assert result["error"] == "gh CLI not installed"
            assert result["url"] is None

    def test_handles_timeout(self):
        executor = PlanExecutor()
        plan = _make_plan()
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="gh", timeout=30),
        ):
            result = executor.execute_to_github_issue(plan)
            assert result["error"] == "Timed out"

    def test_handles_gh_error(self):
        executor = PlanExecutor()
        plan = _make_plan()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="GraphQL: Resource not found",
            )
            result = executor.execute_to_github_issue(plan)
            assert "Resource not found" in result["error"]
            assert result["url"] is None


class TestExecuteToGithubPR:
    """Test execute_to_github_pr validates status and calls gh CLI."""

    def test_rejects_non_approved_plan(self):
        executor = PlanExecutor()
        plan = _make_plan(status=PlanStatus.CREATED)
        with pytest.raises(ValueError, match="APPROVED"):
            executor.execute_to_github_pr(plan)

    def test_creates_pr_for_approved_plan(self):
        executor = PlanExecutor()
        plan = _make_plan(status=PlanStatus.APPROVED)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="https://github.com/org/repo/pull/99\n",
                stderr="",
            )
            result = executor.execute_to_github_pr(plan)
            assert result["url"] == "https://github.com/org/repo/pull/99"
            assert result["number"] == 99
            assert result["plan_id"] == "plan-456"

    def test_passes_base_head_draft(self):
        executor = PlanExecutor()
        plan = _make_plan()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="https://github.com/org/repo/pull/5\n",
                stderr="",
            )
            executor.execute_to_github_pr(
                plan,
                base="develop",
                head="feature/auth",
                draft=True,
            )
            cmd = mock_run.call_args[0][0]
            assert "--base" in cmd
            idx = cmd.index("--base")
            assert cmd[idx + 1] == "develop"
            assert "--head" in cmd
            assert "--draft" in cmd

    def test_handles_gh_not_installed(self):
        executor = PlanExecutor()
        plan = _make_plan()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = executor.execute_to_github_pr(plan)
            assert result["error"] == "gh CLI not installed"


class TestAutoExecutePlanExtension:
    """Test auto_execute_plan wiring in ArenaExtensions."""

    def test_auto_execute_plan_disabled_by_default(self):
        from aragora.debate.extensions import ArenaExtensions

        ext = ArenaExtensions()
        assert ext.auto_execute_plan is False
        assert ext.has_plan_executor is False

    def test_auto_execute_plan_enabled(self):
        from aragora.debate.extensions import ArenaExtensions

        ext = ArenaExtensions(auto_execute_plan=True)
        assert ext.auto_execute_plan is True
        assert ext.has_plan_executor is True

    def test_has_plan_executor_with_executor(self):
        from aragora.debate.extensions import ArenaExtensions

        executor = MagicMock()
        ext = ArenaExtensions(plan_executor=executor)
        assert ext.has_plan_executor is True

    def test_skips_when_disabled(self):
        from aragora.debate.extensions import ArenaExtensions

        ext = ArenaExtensions(auto_execute_plan=False)
        ctx = MagicMock()
        result = MagicMock()
        ext._auto_execute_plan(ctx, result)
        # No error, no action

    def test_skips_when_no_plan_found(self):
        from aragora.debate.extensions import ArenaExtensions

        ext = ArenaExtensions(auto_execute_plan=True)
        ctx = MagicMock(spec=[])
        ctx.metadata = {}
        result = MagicMock(spec=[])
        result.metadata = {}
        ext._auto_execute_plan(ctx, result)
        # No error, no issue created

    def test_calls_execute_to_github_issue(self):
        from aragora.debate.extensions import ArenaExtensions

        plan = _make_plan()
        mock_executor = MagicMock()
        mock_executor.execute_to_github_issue.return_value = {
            "url": "https://github.com/org/repo/issues/1",
            "number": 1,
            "plan_id": plan.id,
        }
        ext = ArenaExtensions(
            auto_execute_plan=True,
            plan_executor=mock_executor,
        )
        ctx = MagicMock()
        result = MagicMock()
        result.decision_plan = plan
        result.metadata = {}

        ext._auto_execute_plan(ctx, result)
        mock_executor.execute_to_github_issue.assert_called_once_with(plan)
        assert result.metadata["github_issue_url"] == "https://github.com/org/repo/issues/1"

    def test_finds_plan_in_context_metadata(self):
        from aragora.debate.extensions import ArenaExtensions

        plan = _make_plan()
        mock_executor = MagicMock()
        mock_executor.execute_to_github_issue.return_value = {
            "url": "https://github.com/org/repo/issues/2",
            "number": 2,
            "plan_id": plan.id,
        }
        ext = ArenaExtensions(
            auto_execute_plan=True,
            plan_executor=mock_executor,
        )
        ctx = MagicMock(spec=[])
        ctx.metadata = {"decision_plan": plan}
        result = MagicMock(spec=[])
        result.metadata = {}

        ext._auto_execute_plan(ctx, result)
        mock_executor.execute_to_github_issue.assert_called_once_with(plan)

    def test_exception_does_not_propagate(self):
        from aragora.debate.extensions import ArenaExtensions

        plan = _make_plan()
        mock_executor = MagicMock()
        mock_executor.execute_to_github_issue.side_effect = RuntimeError("gh crashed")
        ext = ArenaExtensions(
            auto_execute_plan=True,
            plan_executor=mock_executor,
        )
        ctx = MagicMock()
        result = MagicMock()
        result.decision_plan = plan

        # Should not raise
        ext._auto_execute_plan(ctx, result)

    def test_on_debate_complete_calls_auto_execute_plan(self):
        from aragora.debate.extensions import ArenaExtensions

        ext = ArenaExtensions(auto_execute_plan=True)
        ctx = MagicMock()
        ctx.debate_id = "d-1"
        result = MagicMock()
        result.messages = []

        with patch.object(ext, "_auto_execute_plan") as mock:
            ext.on_debate_complete(ctx, result, [])
            mock.assert_called_once_with(ctx, result)


class TestExtensionsConfigPlanExecutor:
    """Test ExtensionsConfig includes plan_executor fields."""

    def test_config_has_auto_execute_plan(self):
        from aragora.debate.extensions import ExtensionsConfig

        config = ExtensionsConfig(auto_execute_plan=True)
        assert config.auto_execute_plan is True

    def test_config_has_plan_executor(self):
        from aragora.debate.extensions import ExtensionsConfig

        executor = MagicMock()
        config = ExtensionsConfig(plan_executor=executor)
        assert config.plan_executor is executor

    def test_create_extensions_passes_auto_execute_plan(self):
        from aragora.debate.extensions import ExtensionsConfig

        config = ExtensionsConfig(auto_execute_plan=True)
        ext = config.create_extensions()
        assert ext.auto_execute_plan is True

    def test_create_extensions_passes_plan_executor(self):
        from aragora.debate.extensions import ExtensionsConfig

        executor = MagicMock()
        config = ExtensionsConfig(plan_executor=executor)
        ext = config.create_extensions()
        assert ext.plan_executor is executor

    def test_create_extensions_defaults(self):
        from aragora.debate.extensions import ExtensionsConfig

        config = ExtensionsConfig()
        ext = config.create_extensions()
        assert ext.auto_execute_plan is False
        assert ext.plan_executor is None


class TestArenaConfigAutoExecutePlan:
    """Test ArenaConfig includes auto_execute_plan parameter."""

    def test_arena_config_has_auto_execute_plan(self):
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig(auto_execute_plan=True)
        assert config.auto_execute_plan is True

    def test_arena_config_default_false(self):
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig()
        assert config.auto_execute_plan is False
