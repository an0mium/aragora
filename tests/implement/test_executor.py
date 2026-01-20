"""Tests for hybrid multi-model executor."""

import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from aragora.implement.executor import (
    HybridExecutor,
    TASK_PROMPT_TEMPLATE,
)
from aragora.implement.types import ImplementTask, TaskResult


class TestHybridExecutorInit:
    """Tests for HybridExecutor initialization."""

    def test_creates_with_repo_path(self, tmp_path):
        """Should initialize with repo path."""
        executor = HybridExecutor(repo_path=tmp_path)
        assert executor.repo_path == tmp_path

    def test_default_timeouts(self, tmp_path):
        """Should have default timeout values."""
        executor = HybridExecutor(repo_path=tmp_path)
        assert executor.claude_timeout == 1200
        assert executor.codex_timeout == 1200

    def test_custom_timeouts(self, tmp_path):
        """Should accept custom timeout values."""
        executor = HybridExecutor(
            repo_path=tmp_path,
            claude_timeout=600,
            codex_timeout=900,
        )
        assert executor.claude_timeout == 600
        assert executor.codex_timeout == 900

    def test_default_max_retries(self, tmp_path):
        """Should have default max retries."""
        executor = HybridExecutor(repo_path=tmp_path)
        assert executor.max_retries == 2

    def test_custom_max_retries(self, tmp_path):
        """Should accept custom max retries."""
        executor = HybridExecutor(repo_path=tmp_path, max_retries=5)
        assert executor.max_retries == 5

    def test_agents_initialized_lazily(self, tmp_path):
        """Agents should not be created until accessed."""
        executor = HybridExecutor(repo_path=tmp_path)
        assert executor._claude is None
        assert executor._codex is None


class TestGetTaskTimeout:
    """Tests for task timeout calculation."""

    @pytest.fixture
    def executor(self, tmp_path):
        """Create executor with standard settings."""
        return HybridExecutor(repo_path=tmp_path)

    def test_simple_task_base_timeout(self, executor):
        """Simple tasks should have 5 min base timeout."""
        task = ImplementTask(
            id="t1",
            description="Test",
            files=["a.py"],
            complexity="simple",
        )
        with patch.dict(os.environ, {"IMPL_COMPLEXITY_TIMEOUT": "1"}):
            timeout = executor._get_task_timeout(task)
        assert timeout == 300  # 5 min

    def test_moderate_task_base_timeout(self, executor):
        """Moderate tasks should have 10 min base timeout."""
        task = ImplementTask(
            id="t1",
            description="Test",
            files=["a.py"],
            complexity="moderate",
        )
        with patch.dict(os.environ, {"IMPL_COMPLEXITY_TIMEOUT": "1"}):
            timeout = executor._get_task_timeout(task)
        assert timeout == 600  # 10 min

    def test_complex_task_base_timeout(self, executor):
        """Complex tasks should have 20 min base timeout."""
        task = ImplementTask(
            id="t1",
            description="Test",
            files=["a.py"],
            complexity="complex",
        )
        with patch.dict(os.environ, {"IMPL_COMPLEXITY_TIMEOUT": "1"}):
            timeout = executor._get_task_timeout(task)
        assert timeout == 1200  # 20 min

    def test_file_count_bonus(self, executor):
        """Additional files should add timeout."""
        task = ImplementTask(
            id="t1",
            description="Test",
            files=["a.py", "b.py", "c.py"],
            complexity="simple",
        )
        with patch.dict(os.environ, {"IMPL_COMPLEXITY_TIMEOUT": "1"}):
            timeout = executor._get_task_timeout(task)
        # 300 base + 2*120 for extra files = 540
        assert timeout == 540

    def test_timeout_capped_at_30_min(self, executor):
        """Timeout should not exceed 30 minutes."""
        task = ImplementTask(
            id="t1",
            description="Test",
            files=[f"file{i}.py" for i in range(20)],
            complexity="complex",
        )
        with patch.dict(os.environ, {"IMPL_COMPLEXITY_TIMEOUT": "1"}):
            timeout = executor._get_task_timeout(task)
        assert timeout == 1800  # 30 min cap

    def test_empty_files_uses_default(self, executor):
        """Empty files list should use default file count of 1."""
        task = ImplementTask(
            id="t1",
            description="Test",
            files=[],
            complexity="simple",
        )
        with patch.dict(os.environ, {"IMPL_COMPLEXITY_TIMEOUT": "1"}):
            timeout = executor._get_task_timeout(task)
        assert timeout == 300  # No file bonus

    def test_feature_flag_disabled(self, executor):
        """Should return default timeout when flag is disabled."""
        task = ImplementTask(
            id="t1",
            description="Test",
            files=["a.py"],
            complexity="simple",
        )
        # Patch the module-level constant directly
        with patch("aragora.implement.executor.COMPLEXITY_TIMEOUT", False):
            timeout = executor._get_task_timeout(task)
        assert timeout == executor.claude_timeout


class TestBuildPrompt:
    """Tests for prompt building."""

    @pytest.fixture
    def executor(self, tmp_path):
        """Create executor."""
        return HybridExecutor(repo_path=tmp_path)

    def test_includes_description(self, executor):
        """Prompt should include task description."""
        task = ImplementTask(
            id="t1",
            description="Add error handling to the API",
            files=["api.py"],
            complexity="simple",
        )
        prompt = executor._build_prompt(task)
        assert "Add error handling to the API" in prompt

    def test_includes_files(self, executor):
        """Prompt should list files to modify."""
        task = ImplementTask(
            id="t1",
            description="Test",
            files=["src/main.py", "src/utils.py"],
            complexity="simple",
        )
        prompt = executor._build_prompt(task)
        assert "src/main.py" in prompt
        assert "src/utils.py" in prompt

    def test_includes_repo_path(self, executor, tmp_path):
        """Prompt should include repo path."""
        task = ImplementTask(
            id="t1",
            description="Test",
            files=["a.py"],
            complexity="simple",
        )
        prompt = executor._build_prompt(task)
        assert str(tmp_path) in prompt

    def test_handles_empty_files(self, executor):
        """Should handle tasks with no specific files."""
        task = ImplementTask(
            id="t1",
            description="Test",
            files=[],
            complexity="simple",
        )
        prompt = executor._build_prompt(task)
        assert "determine from description" in prompt


class TestSelectAgent:
    """Tests for agent selection."""

    @pytest.fixture
    def executor(self, tmp_path):
        """Create executor."""
        return HybridExecutor(repo_path=tmp_path)

    def test_always_selects_claude(self, executor):
        """Should always select Claude for implementation."""
        for complexity in ("simple", "moderate", "complex"):
            agent, name = executor._select_agent(complexity)
            assert name == "claude"

    def test_returns_claude_agent(self, executor):
        """Should return the Claude agent instance."""
        agent, _ = executor._select_agent("simple")
        # Agent should be the claude property
        assert agent == executor.claude


class TestTaskPromptTemplate:
    """Tests for the task prompt template."""

    def test_has_description_placeholder(self):
        """Template should have description placeholder."""
        assert "{description}" in TASK_PROMPT_TEMPLATE

    def test_has_files_placeholder(self):
        """Template should have files placeholder."""
        assert "{files}" in TASK_PROMPT_TEMPLATE

    def test_has_repo_path_placeholder(self):
        """Template should have repo_path placeholder."""
        assert "{repo_path}" in TASK_PROMPT_TEMPLATE

    def test_formats_correctly(self):
        """Template should format without errors."""
        result = TASK_PROMPT_TEMPLATE.format(
            description="Test task",
            files="- file.py",
            repo_path="/test/path",
        )
        assert "Test task" in result
        assert "file.py" in result
        assert "/test/path" in result

    def test_includes_safety_instructions(self):
        """Template should include safety instructions."""
        assert "safe and reversible" in TASK_PROMPT_TEMPLATE.lower()


class TestGetGitDiff:
    """Tests for git diff retrieval."""

    @pytest.fixture
    def executor(self, tmp_path):
        """Create executor with temp repo."""
        # Initialize a git repo
        import subprocess

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            capture_output=True,
        )
        return HybridExecutor(repo_path=tmp_path)

    def test_returns_empty_for_no_changes(self, executor, tmp_path):
        """Should return empty string when no changes."""
        # Create initial commit
        (tmp_path / "test.txt").write_text("initial")
        import subprocess

        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=tmp_path,
            capture_output=True,
        )
        
        diff = executor._get_git_diff()
        assert diff == ""

    def test_returns_diff_for_changes(self, executor, tmp_path):
        """Should return diff for uncommitted changes."""
        # Create initial commit
        test_file = tmp_path / "test.txt"
        test_file.write_text("initial")
        import subprocess

        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=tmp_path,
            capture_output=True,
        )

        # Make changes
        test_file.write_text("modified content")

        diff = executor._get_git_diff()
        assert "test.txt" in diff

    def test_handles_non_git_directory(self, tmp_path):
        """Should handle non-git directories gracefully."""
        executor = HybridExecutor(repo_path=tmp_path)
        diff = executor._get_git_diff()
        # Should return empty string, not raise
        assert diff == ""


class TestTaskResult:
    """Tests for TaskResult creation in executor context."""

    def test_success_result_structure(self):
        """Success result should have all fields."""
        result = TaskResult(
            task_id="t1",
            success=True,
            diff="2 files changed",
            model_used="claude",
            duration_seconds=45.0,
        )
        assert result.task_id == "t1"
        assert result.success is True
        assert result.diff == "2 files changed"
        assert result.model_used == "claude"
        assert result.duration_seconds == 45.0
        assert result.error is None

    def test_failure_result_structure(self):
        """Failure result should include error."""
        result = TaskResult(
            task_id="t1",
            success=False,
            error="Timeout: task took too long",
            model_used="claude",
            duration_seconds=120.0,
        )
        assert result.task_id == "t1"
        assert result.success is False
        assert "Timeout" in result.error
        assert result.diff == ""  # Default empty string


class TestExecutorIntegration:
    """Integration tests for executor (without actual API calls)."""

    @pytest.fixture
    def executor(self, tmp_path):
        """Create executor."""
        return HybridExecutor(repo_path=tmp_path)

    def test_multiple_tasks_timeout_calculation(self, executor):
        """Should calculate different timeouts for different tasks."""
        tasks = [
            ImplementTask(id="t1", description="Simple", files=["a.py"], complexity="simple"),
            ImplementTask(id="t2", description="Moderate", files=["b.py", "c.py"], complexity="moderate"),
            ImplementTask(id="t3", description="Complex", files=["d.py", "e.py", "f.py", "g.py"], complexity="complex"),
        ]

        with patch.dict(os.environ, {"IMPL_COMPLEXITY_TIMEOUT": "1"}):
            timeouts = [executor._get_task_timeout(t) for t in tasks]

        # Simple < Moderate < Complex
        assert timeouts[0] < timeouts[1] < timeouts[2]

    def test_build_prompt_for_multiple_tasks(self, executor):
        """Should build distinct prompts for each task."""
        tasks = [
            ImplementTask(id="t1", description="Add feature A", files=["a.py"], complexity="simple"),
            ImplementTask(id="t2", description="Add feature B", files=["b.py"], complexity="simple"),
        ]

        prompts = [executor._build_prompt(t) for t in tasks]

        assert prompts[0] != prompts[1]
        assert "feature A" in prompts[0]
        assert "feature B" in prompts[1]
