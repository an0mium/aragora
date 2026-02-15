"""Tests for CI Feedback Loop infrastructure.

Covers:
- CITestFailure, CITestSummary, CIResult dataclasses
- CIResultCollector with mocked gh CLI
- VerifyPhase CI integration
- FeedbackLoop ci_failure error type
- PlanningContext CI fields
- nomic_ci_test_selector script
- SemanticConflictDetector (basic wiring tests)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.ci_feedback import (
    CIResult,
    CIResultCollector,
    CITestFailure,
    CITestSummary,
)
from aragora.nomic.autonomous_orchestrator import (
    AgentAssignment,
    FeedbackLoop,
)
from aragora.nomic.meta_planner import PlanningContext, MetaPlanner, Track
from aragora.nomic.branch_coordinator import BranchCoordinator, BranchCoordinatorConfig


# ============================================================
# CITestFailure
# ============================================================

class TestCITestFailure:
    def test_creation(self):
        f = CITestFailure(test_name="test_foo", error_message="assert failed")
        assert f.test_name == "test_foo"
        assert f.error_message == "assert failed"
        assert f.file_path == ""

    def test_creation_with_file_path(self):
        f = CITestFailure(test_name="test_bar", error_message="err", file_path="tests/test_bar.py")
        assert f.file_path == "tests/test_bar.py"


# ============================================================
# CITestSummary
# ============================================================

class TestCITestSummary:
    def test_creation_defaults(self):
        s = CITestSummary()
        assert s.total == 0
        assert s.passed == 0
        assert s.failed == 0
        assert s.skipped == 0
        assert s.failure_details == []

    def test_creation_with_values(self):
        failures = [CITestFailure("t1", "e1")]
        s = CITestSummary(total=10, passed=8, failed=1, skipped=1, failure_details=failures)
        assert s.total == 10
        assert len(s.failure_details) == 1

    def test_empty_summary(self):
        s = CITestSummary()
        assert s.failure_details == []
        assert s.total == 0


# ============================================================
# CIResult
# ============================================================

class TestCIResult:
    def test_creation(self):
        r = CIResult(
            workflow_run_id=123,
            branch="dev/test",
            commit_sha="abc123",
            conclusion="success",
        )
        assert r.workflow_run_id == 123
        assert r.branch == "dev/test"
        assert r.conclusion == "success"
        assert r.test_summary is None

    def test_with_summary(self):
        summary = CITestSummary(total=5, passed=5)
        r = CIResult(
            workflow_run_id=456,
            branch="main",
            commit_sha="def456",
            conclusion="success",
            test_summary=summary,
        )
        assert r.test_summary is not None
        assert r.test_summary.total == 5

    def test_without_summary(self):
        r = CIResult(
            workflow_run_id=789,
            branch="feature",
            commit_sha="ghi789",
            conclusion="failure",
        )
        assert r.test_summary is None
        assert r.duration_seconds == 0.0


# ============================================================
# CIResultCollector
# ============================================================

class TestCIResultCollector:
    def test_gh_available_with_gh(self):
        with patch("aragora.nomic.ci_feedback.shutil.which", return_value="/usr/bin/gh"):
            assert CIResultCollector._gh_available() is True

    def test_gh_not_available(self):
        with patch("aragora.nomic.ci_feedback.shutil.which", return_value=None):
            assert CIResultCollector._gh_available() is False

    def test_gh_available_returns_bool(self):
        with patch("aragora.nomic.ci_feedback.shutil.which", return_value=None):
            result = CIResultCollector._gh_available()
            assert isinstance(result, bool)

    def test_poll_gh_unavailable(self):
        with patch("aragora.nomic.ci_feedback.shutil.which", return_value=None):
            collector = CIResultCollector(repo_owner="test", repo_name="repo")
            result = collector.poll_for_result("main", "abc123", timeout=1, poll_interval=0.1)
            assert result is None

    @patch("aragora.nomic.ci_feedback.subprocess.run")
    @patch("aragora.nomic.ci_feedback.shutil.which", return_value="/usr/bin/gh")
    def test_poll_success(self, mock_which, mock_run):
        run_data = {
            "workflow_runs": [{
                "id": 100,
                "head_branch": "dev/test",
                "head_sha": "abc123",
                "status": "completed",
                "conclusion": "success",
                "run_started_at": "2026-02-15T10:00:00Z",
                "updated_at": "2026-02-15T10:05:00Z",
            }]
        }
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(run_data),
        )
        collector = CIResultCollector(repo_owner="test", repo_name="repo")
        result = collector.poll_for_result("dev/test", "abc123", timeout=5, poll_interval=0.1)
        assert result is not None
        assert result.conclusion == "success"
        assert result.workflow_run_id == 100

    @patch("aragora.nomic.ci_feedback.time.sleep")
    @patch("aragora.nomic.ci_feedback.subprocess.run")
    @patch("aragora.nomic.ci_feedback.shutil.which", return_value="/usr/bin/gh")
    def test_poll_timeout(self, mock_which, mock_run, mock_sleep):
        # Return in-progress (no completed runs)
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"workflow_runs": []}),
        )
        collector = CIResultCollector(repo_owner="test", repo_name="repo")
        result = collector.poll_for_result("dev/test", "abc123", timeout=0.1, poll_interval=0.05)
        assert result is None

    @patch("aragora.nomic.ci_feedback.subprocess.run")
    @patch("aragora.nomic.ci_feedback.shutil.which", return_value="/usr/bin/gh")
    def test_get_latest_result_success(self, mock_which, mock_run):
        run_data = {
            "id": 200,
            "head_branch": "dev/feature",
            "head_sha": "xyz789",
            "status": "completed",
            "conclusion": "failure",
            "run_started_at": "",
            "updated_at": "",
        }
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(run_data),
        )
        collector = CIResultCollector(repo_owner="test", repo_name="repo")
        result = collector.get_latest_result("dev/feature")
        assert result is not None
        assert result.conclusion == "failure"

    @patch("aragora.nomic.ci_feedback.subprocess.run")
    @patch("aragora.nomic.ci_feedback.shutil.which", return_value="/usr/bin/gh")
    def test_get_latest_no_results(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        collector = CIResultCollector(repo_owner="test", repo_name="repo")
        result = collector.get_latest_result("dev/empty")
        assert result is None

    def test_get_latest_gh_unavailable(self):
        with patch("aragora.nomic.ci_feedback.shutil.which", return_value=None):
            collector = CIResultCollector(repo_owner="test", repo_name="repo")
            result = collector.get_latest_result("main")
            assert result is None

    def test_parse_run_to_result_success(self):
        collector = CIResultCollector(repo_owner="test", repo_name="repo")
        run_data = {
            "id": 300,
            "head_branch": "main",
            "head_sha": "abc",
            "conclusion": "success",
            "run_started_at": "2026-02-15T10:00:00Z",
            "updated_at": "2026-02-15T10:10:00Z",
        }
        result = collector._parse_run_to_result(run_data)
        assert result.workflow_run_id == 300
        assert result.conclusion == "success"
        assert result.duration_seconds == 600.0

    def test_parse_run_to_result_failure(self):
        collector = CIResultCollector(repo_owner="test", repo_name="repo")
        run_data = {
            "id": 301,
            "head_branch": "dev/x",
            "head_sha": "def",
            "conclusion": "failure",
            "run_started_at": "",
            "updated_at": "",
        }
        result = collector._parse_run_to_result(run_data)
        assert result.conclusion == "failure"
        assert result.duration_seconds == 0.0

    def test_parse_run_to_result_cancelled(self):
        collector = CIResultCollector(repo_owner="test", repo_name="repo")
        run_data = {
            "id": 302,
            "head_branch": "dev/y",
            "head_sha": "ghi",
            "conclusion": "cancelled",
        }
        result = collector._parse_run_to_result(run_data)
        assert result.conclusion == "cancelled"

    def test_parse_run_missing_fields(self):
        collector = CIResultCollector(repo_owner="test", repo_name="repo")
        result = collector._parse_run_to_result({})
        assert result.workflow_run_id == 0
        assert result.branch == ""
        assert result.conclusion == "unknown"

    def test_to_feedback_error_format(self):
        result = CIResult(
            workflow_run_id=400,
            branch="dev/test",
            commit_sha="abc12345",
            conclusion="failure",
            test_summary=CITestSummary(
                total=10, passed=8, failed=2,
                failure_details=[
                    CITestFailure("test_a", "assert failed"),
                    CITestFailure("test_b", "timeout"),
                ],
            ),
        )
        feedback = CIResultCollector.to_feedback_error(result)
        assert feedback["type"] == "ci_failure"
        assert "abc12345" in feedback["message"]
        assert len(feedback["ci_failures"]) == 2

    def test_to_feedback_error_no_summary(self):
        result = CIResult(
            workflow_run_id=401,
            branch="main",
            commit_sha="def456",
            conclusion="failure",
        )
        feedback = CIResultCollector.to_feedback_error(result)
        assert feedback["type"] == "ci_failure"
        assert feedback["ci_failures"] == []


# ============================================================
# VerifyPhase CI Integration
# ============================================================

class TestVerifyPhaseCI:
    @pytest.mark.asyncio
    async def test_ci_collector_not_set(self):
        """VerifyPhase works fine without ci_collector."""
        from scripts.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=Path("/tmp/fake"),
            ci_collector=None,
        )
        assert phase.ci_collector is None

    @pytest.mark.asyncio
    async def test_ci_collector_set(self):
        """VerifyPhase accepts ci_collector parameter."""
        from scripts.nomic.phases.verify import VerifyPhase

        mock_collector = MagicMock()
        phase = VerifyPhase(
            aragora_path=Path("/tmp/fake"),
            ci_collector=mock_collector,
        )
        assert phase.ci_collector is mock_collector

    @pytest.mark.asyncio
    async def test_check_ci_results_no_branch(self):
        """_check_ci_results handles git failure gracefully."""
        from scripts.nomic.phases.verify import VerifyPhase

        mock_collector = MagicMock()
        phase = VerifyPhase(
            aragora_path=Path("/tmp/fake"),
            ci_collector=mock_collector,
        )

        # Mock git rev-parse to fail
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (b"", b"")
            mock_proc.returncode = 1
            mock_exec.return_value = mock_proc

            result = await phase._check_ci_results()
            # Should return None or a passed result when branch/sha unavailable
            assert result is None or result.get("passed", True)

    @pytest.mark.asyncio
    async def test_check_ci_results_success(self):
        """_check_ci_results returns passed when CI succeeded."""
        from scripts.nomic.phases.verify import VerifyPhase

        mock_collector = MagicMock()
        mock_collector.get_latest_result.return_value = CIResult(
            workflow_run_id=500,
            branch="dev/test",
            commit_sha="abc",
            conclusion="success",
        )
        phase = VerifyPhase(
            aragora_path=Path("/tmp/fake"),
            ci_collector=mock_collector,
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (b"dev/test\n", b"")
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc

            result = await phase._check_ci_results()
            assert result is not None
            assert result["passed"] is True
            assert result["check"] == "ci"

    @pytest.mark.asyncio
    async def test_check_ci_results_failure(self):
        """_check_ci_results returns not passed when CI failed."""
        from scripts.nomic.phases.verify import VerifyPhase

        mock_collector = MagicMock()
        mock_collector.get_latest_result.return_value = CIResult(
            workflow_run_id=501,
            branch="dev/test",
            commit_sha="abc",
            conclusion="failure",
            test_summary=CITestSummary(total=10, passed=5, failed=5),
        )
        phase = VerifyPhase(
            aragora_path=Path("/tmp/fake"),
            ci_collector=mock_collector,
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (b"dev/test\n", b"")
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc

            result = await phase._check_ci_results()
            assert result is not None
            assert result["passed"] is False
            assert "failure" in result["output"]


# ============================================================
# FeedbackLoop CI Integration
# ============================================================

class TestFeedbackLoopCI:
    def _make_assignment(self):
        from aragora.nomic.task_decomposer import SubTask
        subtask = SubTask(
            id="test_1",
            title="Test task",
            description="A test",
            estimated_complexity="low",
        )
        return AgentAssignment(
            subtask=subtask,
            track=MagicMock(),
            agent_type="claude",
        )

    def test_ci_failure_returns_retry_implement(self):
        loop = FeedbackLoop()
        assignment = self._make_assignment()
        result = loop.analyze_failure(
            assignment,
            {"type": "ci_failure", "message": "CI failed", "ci_failures": ["test_a"]},
        )
        assert result["action"] == "retry_implement"
        assert "CI" in result["reason"]
        assert result["hints"] == ["test_a"]

    def test_ci_failure_empty_failures(self):
        loop = FeedbackLoop()
        assignment = self._make_assignment()
        result = loop.analyze_failure(
            assignment,
            {"type": "ci_failure", "message": "CI failed"},
        )
        assert result["action"] == "retry_implement"
        assert result["hints"] == []

    def test_ci_failure_escalates_after_max(self):
        loop = FeedbackLoop(max_iterations=1)
        assignment = self._make_assignment()
        # First call uses up the iteration
        loop.analyze_failure(assignment, {"type": "ci_failure"})
        # Second call exceeds max
        result = loop.analyze_failure(assignment, {"type": "ci_failure"})
        assert result["action"] == "escalate"

    def test_ci_failure_preserves_hints_list(self):
        loop = FeedbackLoop()
        assignment = self._make_assignment()
        hints = ["test_x: assert 1 == 2", "test_y: timeout"]
        result = loop.analyze_failure(
            assignment,
            {"type": "ci_failure", "ci_failures": hints},
        )
        assert result["hints"] == hints


# ============================================================
# PlanningContext CI Fields
# ============================================================

class TestPlanningContextCI:
    def test_ci_failures_field(self):
        ctx = PlanningContext(ci_failures=["test_foo failed"])
        assert ctx.ci_failures == ["test_foo failed"]

    def test_ci_flaky_tests_field(self):
        ctx = PlanningContext(ci_flaky_tests=["test_bar"])
        assert ctx.ci_flaky_tests == ["test_bar"]

    def test_ci_fields_default_empty(self):
        ctx = PlanningContext()
        assert ctx.ci_failures == []
        assert ctx.ci_flaky_tests == []

    def test_debate_topic_includes_ci_failures(self):
        planner = MetaPlanner()
        ctx = PlanningContext(ci_failures=["test_foo: assertion error"])
        topic = planner._build_debate_topic(
            "Improve test reliability",
            [Track.QA],
            [],
            ctx,
        )
        assert "CI FAILURES" in topic
        assert "test_foo" in topic

    def test_debate_topic_includes_flaky_tests(self):
        planner = MetaPlanner()
        ctx = PlanningContext(ci_flaky_tests=["test_bar_intermittent"])
        topic = planner._build_debate_topic(
            "Fix flaky tests",
            [Track.QA],
            [],
            ctx,
        )
        assert "FLAKY TESTS" in topic
        assert "test_bar_intermittent" in topic


# ============================================================
# Nomic CI Test Selector
# ============================================================

class TestNomicCITestSelector:
    def test_infer_test_paths_source_file(self, tmp_path):
        """Test mapping aragora/foo/bar.py -> tests/foo/test_bar.py."""
        # Need to import from scripts
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from nomic_ci_test_selector import infer_test_paths

        # Create the target test file so it passes the exists() check
        test_dir = tmp_path / "tests" / "foo"
        test_dir.mkdir(parents=True)
        (test_dir / "test_bar.py").touch()

        with patch("nomic_ci_test_selector.Path.exists", return_value=True):
            result = infer_test_paths(["aragora/foo/bar.py"])
        assert "tests/foo/test_bar.py" in result

    def test_infer_test_paths_test_file(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from nomic_ci_test_selector import infer_test_paths

        result = infer_test_paths(["tests/foo/test_bar.py"])
        assert result == ["tests/foo/test_bar.py"]

    def test_infer_test_paths_empty(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from nomic_ci_test_selector import infer_test_paths

        result = infer_test_paths([])
        assert result == []

    def test_infer_test_paths_blank_entries(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from nomic_ci_test_selector import infer_test_paths

        result = infer_test_paths(["", "  "])
        assert result == []

    def test_infer_test_paths_non_python(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from nomic_ci_test_selector import infer_test_paths

        result = infer_test_paths(["aragora/foo/bar.txt"])
        assert result == []

    def test_infer_test_paths_root_module(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from nomic_ci_test_selector import infer_test_paths

        with patch("nomic_ci_test_selector.Path.exists", return_value=True):
            result = infer_test_paths(["aragora/utils.py"])
        assert "tests/test_utils.py" in result

    def test_infer_test_paths_dedup(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from nomic_ci_test_selector import infer_test_paths

        result = infer_test_paths(["tests/foo/test_bar.py", "tests/foo/test_bar.py"])
        assert len(result) == 1

    def test_infer_test_paths_non_aragora(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from nomic_ci_test_selector import infer_test_paths

        result = infer_test_paths(["scripts/foo.py"])
        assert result == []


# ============================================================
# SemanticConflictDetector (basic wiring)
# ============================================================

class TestSemanticConflictDetectorBasic:
    def test_import(self):
        from aragora.nomic.semantic_conflict_detector import SemanticConflictDetector
        assert SemanticConflictDetector is not None

    def test_creation(self, tmp_path):
        from aragora.nomic.semantic_conflict_detector import SemanticConflictDetector
        detector = SemanticConflictDetector(tmp_path)
        assert detector.repo_path == tmp_path
        assert detector.enable_debate is True

    def test_detect_no_branches(self, tmp_path):
        from aragora.nomic.semantic_conflict_detector import SemanticConflictDetector
        detector = SemanticConflictDetector(tmp_path)
        result = detector.detect([], "main")
        assert result == []

    def test_extract_signatures(self):
        from aragora.nomic.semantic_conflict_detector import SemanticConflictDetector
        detector = SemanticConflictDetector(Path("/tmp"))
        code = """
def foo(a, b, c=1):
    pass

async def bar(x, *args, **kwargs):
    pass
"""
        sigs = detector._extract_signatures(code)
        assert len(sigs) == 2
        foo_sig = next(s for s in sigs if s.name == "foo")
        assert foo_sig.args == ["a", "b", "c"]
        assert foo_sig.defaults_count == 1
        assert foo_sig.is_async is False

        bar_sig = next(s for s in sigs if s.name == "bar")
        assert bar_sig.has_varargs is True
        assert bar_sig.has_kwargs is True
        assert bar_sig.is_async is True

    def test_extract_signatures_invalid_syntax(self):
        from aragora.nomic.semantic_conflict_detector import SemanticConflictDetector
        detector = SemanticConflictDetector(Path("/tmp"))
        sigs = detector._extract_signatures("def foo(:")
        assert sigs == []

    def test_signatures_conflict_different_args(self):
        from aragora.nomic.semantic_conflict_detector import (
            SemanticConflictDetector,
            FunctionSignature,
        )
        sig_a = FunctionSignature("foo", ["a", "b"], 0, False, False, False)
        sig_b = FunctionSignature("foo", ["a", "b", "c"], 0, False, False, False)
        assert SemanticConflictDetector._signatures_conflict(sig_a, sig_b) is True

    def test_signatures_no_conflict(self):
        from aragora.nomic.semantic_conflict_detector import (
            SemanticConflictDetector,
            FunctionSignature,
        )
        sig_a = FunctionSignature("foo", ["a", "b"], 0, False, False, False)
        sig_b = FunctionSignature("foo", ["a", "b"], 0, False, False, False)
        assert SemanticConflictDetector._signatures_conflict(sig_a, sig_b) is False

    def test_extract_imports(self):
        from aragora.nomic.semantic_conflict_detector import SemanticConflictDetector
        detector = SemanticConflictDetector(Path("/tmp"))
        code = """
import os
import json
from pathlib import Path
from aragora.nomic import ci_feedback
"""
        imports = detector._extract_imports(code)
        assert "os" in imports
        assert "json" in imports
        assert "pathlib" in imports
        assert "aragora.nomic" in imports


# ============================================================
# BranchCoordinator semantic conflict wiring
# ============================================================

class TestBranchCoordinatorSemanticConflicts:
    def test_config_has_enable_flag(self):
        config = BranchCoordinatorConfig(enable_semantic_conflicts=True)
        assert config.enable_semantic_conflicts is True

    def test_default_config_disabled(self):
        config = BranchCoordinatorConfig()
        assert config.enable_semantic_conflicts is False

    def test_coordinator_accepts_detector(self, tmp_path):
        mock_detector = MagicMock()
        coord = BranchCoordinator(
            repo_path=tmp_path,
            semantic_conflict_detector=mock_detector,
        )
        assert coord.semantic_conflict_detector is mock_detector

    def test_coordinator_auto_creates_detector(self, tmp_path):
        config = BranchCoordinatorConfig(enable_semantic_conflicts=True)
        coord = BranchCoordinator(repo_path=tmp_path, config=config)
        assert coord.semantic_conflict_detector is not None


# ============================================================
# Auto-detect repo
# ============================================================

class TestAutoDetectRepo:
    @patch("aragora.nomic.ci_feedback.shutil.which", return_value=None)
    def test_no_gh(self, mock_which):
        collector = CIResultCollector()
        assert collector._repo_owner == ""
        assert collector._repo_name == ""

    @patch("aragora.nomic.ci_feedback.subprocess.run")
    @patch("aragora.nomic.ci_feedback.shutil.which", return_value="/usr/bin/gh")
    def test_auto_detect_success(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"owner": {"login": "acme"}, "name": "aragora"}),
        )
        collector = CIResultCollector()
        assert collector._repo_owner == "acme"
        assert collector._repo_name == "aragora"

    @patch("aragora.nomic.ci_feedback.subprocess.run")
    @patch("aragora.nomic.ci_feedback.shutil.which", return_value="/usr/bin/gh")
    def test_auto_detect_failure(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        collector = CIResultCollector()
        assert collector._repo_owner == ""

    def test_repo_slug(self):
        with patch("aragora.nomic.ci_feedback.shutil.which", return_value=None):
            collector = CIResultCollector(repo_owner="owner", repo_name="repo")
        assert collector.repo_slug == "owner/repo"
