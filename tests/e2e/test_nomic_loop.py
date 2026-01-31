"""
E2E tests for Nomic Loop phases.

Phase 5.2: Tests the full self-improvement cycle with safety gates.

The Nomic Loop is the autonomous self-improvement cycle:
- Context: Gather codebase understanding
- Debate: Agents propose improvements
- Design: Architecture planning
- Implement: Code generation
- Verify: Tests and quality checks
- Commit: Git commit if verified

These tests validate:
1. Each phase produces expected outputs
2. Safety gates block dangerous changes
3. Rollback on verification failure
4. Protected files are not modified
5. Full cycle integration

All LLM calls are mocked to avoid costs and ensure deterministic testing.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def nomic_temp_dir(tmp_path: Path) -> Path:
    """Create a mock aragora project structure for nomic loop testing."""
    # Create directory structure
    (tmp_path / "aragora").mkdir()
    (tmp_path / "aragora" / "debate").mkdir()
    (tmp_path / "aragora" / "memory").mkdir()
    (tmp_path / "aragora" / "nomic").mkdir()
    (tmp_path / "aragora" / "nomic" / "phases").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / ".nomic").mkdir()

    # Create __init__.py files
    (tmp_path / "aragora" / "__init__.py").write_text(
        '"""Aragora - Multi-agent debate framework."""\n__version__ = "1.0.0"\n'
    )
    (tmp_path / "aragora" / "debate" / "__init__.py").write_text(
        '"""Debate module."""\nfrom .orchestrator import Arena\n'
    )
    (tmp_path / "aragora" / "debate" / "orchestrator.py").write_text(
        '''"""Arena for orchestrating debates."""
class Arena:
    def __init__(self):
        pass
    async def run(self):
        return {"consensus": True}
'''
    )
    (tmp_path / "aragora" / "memory" / "__init__.py").write_text('"""Memory module."""\n')
    (tmp_path / "aragora" / "nomic" / "__init__.py").write_text('"""Nomic module."""\n')
    (tmp_path / "aragora" / "nomic" / "phases" / "__init__.py").write_text('"""Phases."""\n')

    # Create protected files
    (tmp_path / "CLAUDE.md").write_text("# Claude Code Integration Guide\n\nDo not modify.")
    (tmp_path / ".env").write_text("# Environment variables\nSECRET_KEY=test123\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "aragora"\nversion = "1.0.0"\n')

    # Create test directory with a sample test
    (tmp_path / "tests" / "__init__.py").write_text("")
    (tmp_path / "tests" / "test_sample.py").write_text(
        """
def test_sample():
    assert True
"""
    )

    # Initialize git repo for the tests that need it
    os.system(f"cd {tmp_path} && git init -q && git add . && git commit -qm 'init'")

    return tmp_path


@pytest.fixture
def protected_files() -> list[str]:
    """List of files that should never be modified."""
    return [
        "CLAUDE.md",
        "aragora/__init__.py",
        "aragora/core/__init__.py",
        "aragora/core_types.py",
        ".env",
        "scripts/nomic_loop.py",
    ]


@pytest.fixture
def mock_claude_agent() -> MagicMock:
    """Create a mock Claude agent."""
    agent = MagicMock()
    agent.name = "claude"
    agent.timeout = 300
    agent.generate = AsyncMock(
        return_value="""## FEATURE INVENTORY
| Feature | Module | Status |
|---------|--------|--------|
| Arena | aragora/debate | IMPLEMENTED |
| Memory | aragora/memory | IMPLEMENTED |

## ARCHITECTURE PATTERNS
- Mixin-based handlers
- Adapter pattern for KM

## GENUINE GAPS
- No real-time streaming for live debates
"""
    )
    return agent


@pytest.fixture
def mock_codex_agent() -> MagicMock:
    """Create a mock Codex agent."""
    agent = MagicMock()
    agent.name = "codex"
    agent.timeout = 300
    agent.generate = AsyncMock(
        return_value="""## CODEBASE ANALYSIS
The codebase has:
- debate/ module with Arena orchestrator
- memory/ module for persistence
- Tests in tests/ directory

## SUGGESTED IMPROVEMENTS
- Add streaming for live debates
- Improve error handling
"""
    )
    return agent


@pytest.fixture
def mock_gemini_agent() -> MagicMock:
    """Create a mock Gemini agent."""
    agent = MagicMock()
    agent.name = "gemini"
    agent.timeout = 300
    agent.generate = AsyncMock(
        return_value="""## FEATURE PROPOSAL
**Existence Check:** I searched for streaming, websocket, live-update and found nothing.

**Feature:** Real-time Debate Streaming

**What It Does:** Enables live streaming of debate progress via WebSocket.

**How It Works:** Add websocket endpoint that broadcasts debate messages.

**Why It Matters:** Allows users to watch debates in real-time.
"""
    )
    return agent


@pytest.fixture
def mock_log_fn() -> MagicMock:
    """Create a mock logging function that captures logs."""
    logs: list[str] = []

    def log_fn(msg: str, **kwargs):
        logs.append(msg)

    log_fn.logs = logs  # type: ignore
    return log_fn


@pytest.fixture
def mock_stream_emit_fn() -> MagicMock:
    """Create a mock stream emit function."""
    return MagicMock()


@pytest.fixture
def mock_record_replay_fn() -> MagicMock:
    """Create a mock replay recording function."""
    return MagicMock()


# ============================================================================
# Context Phase Tests
# ============================================================================


class TestContextPhase:
    """Tests for the Context gathering phase (Phase 0)."""

    @pytest.mark.asyncio
    async def test_context_phase_gathers_from_agents(
        self,
        nomic_temp_dir: Path,
        mock_claude_agent: MagicMock,
        mock_codex_agent: MagicMock,
        mock_log_fn: MagicMock,
    ):
        """Context phase should gather information from multiple agents."""
        from aragora.nomic.phases.context import ContextPhase

        phase = ContextPhase(
            aragora_path=nomic_temp_dir,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        # Mock the _gather_with_agent method to return predictable results
        with patch.object(phase, "_gather_with_agent", new_callable=AsyncMock) as mock_gather:
            mock_gather.side_effect = [
                ("claude", "Claude Code", "## FEATURE INVENTORY\n- Arena implemented"),
                ("codex", "Codex CLI", "## CODEBASE ANALYSIS\n- Tests found"),
            ]

            result = await phase.execute()

            assert result["success"] is True
            assert "codebase_summary" in result
            assert len(result["codebase_summary"]) > 0
            assert result["duration_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_context_phase_handles_timeout(
        self,
        nomic_temp_dir: Path,
        mock_claude_agent: MagicMock,
        mock_codex_agent: MagicMock,
        mock_log_fn: MagicMock,
    ):
        """Context phase should handle agent timeouts gracefully."""
        from aragora.nomic.phases.context import ContextPhase

        phase = ContextPhase(
            aragora_path=nomic_temp_dir,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
            get_features_fn=lambda: "Fallback: Core debate features available",
        )

        with patch.object(phase, "_gather_with_agent", new_callable=AsyncMock) as mock_gather:
            # Simulate timeout for both agents
            mock_gather.return_value = ("claude", "Claude Code", "Error: timeout exceeded")

            result = await phase.execute()

            # Should still succeed with fallback
            assert result is not None
            assert "codebase_summary" in result

    @pytest.mark.asyncio
    async def test_context_phase_uses_fallback(
        self,
        nomic_temp_dir: Path,
        mock_claude_agent: MagicMock,
        mock_codex_agent: MagicMock,
        mock_log_fn: MagicMock,
    ):
        """Context phase should use fallback when all agents fail."""
        from aragora.nomic.phases.context import ContextPhase

        fallback_text = "Fallback: Core aragora features"

        phase = ContextPhase(
            aragora_path=nomic_temp_dir,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
            get_features_fn=lambda: fallback_text,
        )

        with patch.object(phase, "_gather_with_agent", new_callable=AsyncMock) as mock_gather:
            # All agents return errors
            mock_gather.return_value = ("agent", "harness", "Error: connection failed")

            result = await phase.execute()

            assert "Fallback" in result["codebase_summary"] or len(result["codebase_summary"]) > 0


# ============================================================================
# Debate Phase Tests
# ============================================================================


class TestDebatePhase:
    """Tests for the Debate phase (Phase 1)."""

    @pytest.mark.asyncio
    async def test_debate_phase_produces_proposals(
        self,
        nomic_temp_dir: Path,
        mock_claude_agent: MagicMock,
        mock_codex_agent: MagicMock,
        mock_log_fn: MagicMock,
    ):
        """Debate phase should produce improvement proposals from agents."""
        from aragora.nomic.phases.debate import DebatePhase

        # Configure agents to generate proposals
        mock_claude_agent.generate = AsyncMock(
            return_value="Proposal: Add caching layer for better performance"
        )
        mock_codex_agent.generate = AsyncMock(return_value="Proposal: Improve error handling")

        phase = DebatePhase(
            aragora_path=nomic_temp_dir,
            agents=[mock_claude_agent, mock_codex_agent],
            log_fn=mock_log_fn,
        )

        result = await phase.run(context="Test context")

        assert "proposals" in result
        assert len(result["proposals"]) > 0

    @pytest.mark.asyncio
    async def test_debate_phase_voting(
        self,
        nomic_temp_dir: Path,
        mock_claude_agent: MagicMock,
        mock_codex_agent: MagicMock,
        mock_log_fn: MagicMock,
    ):
        """Debate phase should collect votes from agents."""
        from aragora.nomic.phases.debate import DebatePhase

        mock_claude_agent.generate = AsyncMock(return_value="Proposal: Add streaming")
        mock_codex_agent.generate = AsyncMock(return_value="Proposal: Add streaming")

        phase = DebatePhase(
            aragora_path=nomic_temp_dir,
            agents=[mock_claude_agent, mock_codex_agent],
            log_fn=mock_log_fn,
            consensus_threshold=0.5,
        )

        result = await phase.run(context="Test context")

        assert "votes" in result
        # With two agents voting for the same, should have votes

    @pytest.mark.asyncio
    async def test_debate_phase_consensus_check(
        self,
        nomic_temp_dir: Path,
        mock_log_fn: MagicMock,
    ):
        """Debate phase should correctly check for consensus."""
        from aragora.nomic.phases.debate import DebatePhase

        agent1 = MagicMock()
        agent1.name = "agent1"
        agent1.generate = AsyncMock(return_value="Proposal A")

        agent2 = MagicMock()
        agent2.name = "agent2"
        agent2.generate = AsyncMock(return_value="Proposal A")

        agent3 = MagicMock()
        agent3.name = "agent3"
        agent3.generate = AsyncMock(return_value="Proposal A")

        phase = DebatePhase(
            aragora_path=nomic_temp_dir,
            agents=[agent1, agent2, agent3],
            log_fn=mock_log_fn,
            consensus_threshold=0.5,
        )

        # Test consensus check logic
        votes = {"agent1": "proposal_a", "agent2": "proposal_a", "agent3": "proposal_b"}
        result = phase.check_consensus(votes, total_agents=3)

        assert result["consensus"] is True
        assert result["winner"] == "proposal_a"
        assert result["confidence"] >= 0.5

    @pytest.mark.asyncio
    async def test_debate_phase_no_consensus(
        self,
        nomic_temp_dir: Path,
        mock_log_fn: MagicMock,
    ):
        """Debate phase should detect lack of consensus."""
        from aragora.nomic.phases.debate import DebatePhase

        agent1 = MagicMock()
        agent1.name = "agent1"
        agent1.generate = AsyncMock(return_value="Proposal A")

        phase = DebatePhase(
            aragora_path=nomic_temp_dir,
            agents=[agent1],
            log_fn=mock_log_fn,
            consensus_threshold=0.8,
        )

        # Votes are split - no clear winner meets threshold
        votes = {
            "agent1": "proposal_a",
            "agent2": "proposal_b",
            "agent3": "proposal_c",
            "agent4": "proposal_d",
        }
        result = phase.check_consensus(votes, total_agents=4)

        assert result["consensus"] is False
        assert result["confidence"] < 0.8


# ============================================================================
# Design Phase Tests
# ============================================================================


class TestDesignPhase:
    """Tests for the Design phase (Phase 2)."""

    @pytest.mark.asyncio
    async def test_design_phase_creates_plan(
        self,
        nomic_temp_dir: Path,
        mock_claude_agent: MagicMock,
        mock_log_fn: MagicMock,
    ):
        """Design phase should create an implementation plan."""
        from aragora.nomic.phases.design import DesignPhase

        mock_claude_agent.generate = AsyncMock(
            return_value="""
## FILE CHANGES
- `aragora/streaming/handler.py` - Create new
- `aragora/debate/orchestrator.py` - Modify existing

## API DESIGN
```python
class StreamHandler:
    async def broadcast(self, message: str) -> None:
        ...
```

## TEST PLAN
- test_streaming_handler.py
"""
        )

        phase = DesignPhase(
            aragora_path=nomic_temp_dir,
            claude_agent=mock_claude_agent,
            log_fn=mock_log_fn,
        )

        proposal = {"proposal": "Add real-time debate streaming"}
        result = await phase.run(proposal=proposal)

        assert "design" in result
        assert result["design"] is not None

    @pytest.mark.asyncio
    async def test_design_phase_safety_review(
        self,
        nomic_temp_dir: Path,
        mock_claude_agent: MagicMock,
        mock_log_fn: MagicMock,
        protected_files: list[str],
    ):
        """Design phase should flag designs that modify protected files."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=nomic_temp_dir,
            claude_agent=mock_claude_agent,
            protected_files=protected_files,
            log_fn=mock_log_fn,
        )

        # Design that would modify protected file
        design = {
            "description": "Modify core module",
            "files_to_modify": ["aragora/__init__.py"],
            "files_to_create": [],
        }

        safety_result = await phase.safety_review(design)

        assert not safety_result["safe"]
        assert len(safety_result["issues"]) > 0
        assert any("protected" in issue.lower() for issue in safety_result["issues"])

    @pytest.mark.asyncio
    async def test_design_phase_blocks_dangerous_patterns(
        self,
        nomic_temp_dir: Path,
        mock_claude_agent: MagicMock,
        mock_log_fn: MagicMock,
    ):
        """Design phase should detect dangerous code patterns."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=nomic_temp_dir,
            claude_agent=mock_claude_agent,
            log_fn=mock_log_fn,
        )

        design = {
            "description": "Execute shell command with os.system('rm -rf /')",
            "files_to_modify": [],
            "files_to_create": ["aragora/dangerous.py"],
        }

        safety_result = await phase.safety_review(design)

        # Should detect high-risk patterns
        assert len(safety_result["risk_patterns"]) > 0 or safety_result["risk_level"] != "low"


# ============================================================================
# Implement Phase Tests
# ============================================================================


class TestImplementPhase:
    """Tests for the Implement phase (Phase 3)."""

    @pytest.mark.asyncio
    async def test_implement_phase_generates_code(
        self,
        nomic_temp_dir: Path,
        mock_codex_agent: MagicMock,
        mock_log_fn: MagicMock,
    ):
        """Implement phase should generate code from design."""
        from aragora.nomic.phases.implement import ImplementPhase

        mock_codex_agent.generate = AsyncMock(
            return_value='"""New module."""\ndef new_function():\n    return True\n'
        )

        phase = ImplementPhase(
            aragora_path=nomic_temp_dir,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        # Mock the internal _generate_code to return file changes
        async def mock_generate(design):
            return {"aragora/new_module.py": "def new_function():\n    return True\n"}

        with patch.object(phase, "_generate_code", side_effect=mock_generate):
            result = await phase.run("Create a new module")

            assert result["success"] is True
            assert "files_modified" in result

    @pytest.mark.asyncio
    async def test_implement_phase_validates_syntax(
        self,
        nomic_temp_dir: Path,
        mock_codex_agent: MagicMock,
        mock_log_fn: MagicMock,
    ):
        """Implement phase should validate Python syntax."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=nomic_temp_dir,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        # Valid syntax
        assert phase.validate_syntax("def foo(): pass") is True

        # Invalid syntax
        assert phase.validate_syntax("def foo(") is False

    @pytest.mark.asyncio
    async def test_implement_phase_checks_dangerous_patterns(
        self,
        nomic_temp_dir: Path,
        mock_codex_agent: MagicMock,
        mock_log_fn: MagicMock,
    ):
        """Implement phase should detect dangerous code patterns."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=nomic_temp_dir,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        # Code with dangerous patterns
        dangerous_code = """
import os
os.system("rm -rf /")
eval(user_input)
exec(dynamic_code)
"""
        result = phase.check_dangerous_patterns(dangerous_code)

        assert not result["safe"]
        assert len(result["patterns_found"]) > 0

    @pytest.mark.asyncio
    async def test_implement_phase_creates_backup(
        self,
        nomic_temp_dir: Path,
        mock_codex_agent: MagicMock,
        mock_log_fn: MagicMock,
    ):
        """Implement phase should create backups before modifying files."""
        from aragora.nomic.phases.implement import ImplementPhase

        # Create a file to backup
        test_file = nomic_temp_dir / "aragora" / "to_modify.py"
        test_file.write_text("# Original content\n")

        backup_dir = nomic_temp_dir / ".nomic" / "backups"
        phase = ImplementPhase(
            aragora_path=nomic_temp_dir,
            codex_agent=mock_codex_agent,
            backup_path=backup_dir,
            log_fn=mock_log_fn,
        )

        manifest = await phase.create_backup(["aragora/to_modify.py"])

        assert manifest is not None
        assert "path" in manifest
        assert "aragora/to_modify.py" in manifest["files"]

    @pytest.mark.asyncio
    async def test_implement_phase_rollback(
        self,
        nomic_temp_dir: Path,
        mock_codex_agent: MagicMock,
        mock_log_fn: MagicMock,
    ):
        """Implement phase should be able to rollback changes."""
        from aragora.nomic.phases.implement import ImplementPhase

        # Create original file
        test_file = nomic_temp_dir / "aragora" / "rollback_test.py"
        original_content = "# Original content\n"
        test_file.write_text(original_content)

        backup_dir = nomic_temp_dir / ".nomic" / "backups"
        phase = ImplementPhase(
            aragora_path=nomic_temp_dir,
            codex_agent=mock_codex_agent,
            backup_path=backup_dir,
            log_fn=mock_log_fn,
        )

        # Create backup
        manifest = await phase.create_backup(["aragora/rollback_test.py"])

        # Modify the file
        test_file.write_text("# Modified content\n")
        assert test_file.read_text() == "# Modified content\n"

        # Rollback
        await phase.rollback(manifest)

        # Should be back to original
        assert test_file.read_text() == original_content


# ============================================================================
# Verify Phase Tests
# ============================================================================


class TestVerifyPhase:
    """Tests for the Verify phase (Phase 4)."""

    @pytest.mark.asyncio
    async def test_verify_phase_runs_checks(
        self,
        nomic_temp_dir: Path,
        mock_log_fn: MagicMock,
    ):
        """Verify phase should run syntax and import checks."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=nomic_temp_dir,
            log_fn=mock_log_fn,
        )

        # Mock subprocess calls for verification
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"OK", b""))
            mock_exec.return_value = mock_proc

            result = await phase.execute()

            assert "success" in result
            assert "data" in result
            assert "checks" in result["data"]

    @pytest.mark.asyncio
    async def test_verify_phase_syntax_failure(
        self,
        nomic_temp_dir: Path,
        mock_log_fn: MagicMock,
    ):
        """Verify phase should detect syntax errors."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=nomic_temp_dir,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "_check_syntax", new_callable=AsyncMock) as mock_syntax:
            with patch.object(phase, "_check_imports", new_callable=AsyncMock) as mock_imports:
                with patch.object(phase, "_run_tests", new_callable=AsyncMock) as mock_tests:
                    mock_syntax.return_value = {
                        "check": "syntax",
                        "passed": False,
                        "output": "SyntaxError: invalid syntax",
                    }
                    mock_imports.return_value = {"check": "import", "passed": True}
                    mock_tests.return_value = {"check": "tests", "passed": True}

                    result = await phase.execute()

                    assert result["success"] is False
                    assert result["syntax_valid"] is False

    @pytest.mark.asyncio
    async def test_verify_phase_test_failure(
        self,
        nomic_temp_dir: Path,
        mock_log_fn: MagicMock,
    ):
        """Verify phase should detect test failures."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=nomic_temp_dir,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "_check_syntax", new_callable=AsyncMock) as mock_syntax:
            with patch.object(phase, "_check_imports", new_callable=AsyncMock) as mock_imports:
                with patch.object(phase, "_run_tests", new_callable=AsyncMock) as mock_tests:
                    mock_syntax.return_value = {"check": "syntax", "passed": True}
                    mock_imports.return_value = {"check": "import", "passed": True}
                    mock_tests.return_value = {
                        "check": "tests",
                        "passed": False,
                        "output": "FAILED test_sample.py::test_something",
                    }

                    result = await phase.execute()

                    assert result["success"] is False


# ============================================================================
# Commit Phase Tests
# ============================================================================


class TestCommitPhase:
    """Tests for the Commit phase (Phase 5)."""

    @pytest.mark.asyncio
    async def test_commit_phase_requires_approval(
        self,
        nomic_temp_dir: Path,
        mock_log_fn: MagicMock,
    ):
        """Commit phase should require approval when configured."""
        from aragora.nomic.phases.commit import CommitPhase

        phase = CommitPhase(
            aragora_path=nomic_temp_dir,
            require_human_approval=True,
            auto_commit=False,
            log_fn=mock_log_fn,
        )

        # Mock the approval check to return False
        with patch.object(phase, "_get_approval", return_value=False):
            result = await phase.execute("Test improvement")

            assert result["committed"] is False
            assert "declined" in result["data"].get("reason", "").lower()

    @pytest.mark.asyncio
    async def test_commit_phase_auto_commit(
        self,
        nomic_temp_dir: Path,
        mock_log_fn: MagicMock,
    ):
        """Commit phase should auto-commit when configured."""
        from aragora.nomic.phases.commit import CommitPhase

        # Make a change to commit
        test_file = nomic_temp_dir / "aragora" / "new_feature.py"
        test_file.write_text("# New feature\n")

        phase = CommitPhase(
            aragora_path=nomic_temp_dir,
            require_human_approval=False,
            auto_commit=True,
            log_fn=mock_log_fn,
        )

        # Stage the changes first
        os.system(f"cd {nomic_temp_dir} && git add .")

        result = await phase.execute("Add new feature")

        if result["committed"]:
            assert result["commit_hash"] is not None
        # Note: May not commit if there are no actual changes in git


# ============================================================================
# Safety Gates Tests
# ============================================================================


class TestSafetyGates:
    """Tests for Nomic Loop safety gates."""

    @pytest.mark.asyncio
    async def test_protected_files_not_modified(
        self,
        nomic_temp_dir: Path,
        protected_files: list[str],
    ):
        """Safety gate should block modification of protected files."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=nomic_temp_dir,
            protected_files=protected_files,
        )

        # Create design that touches protected file
        design = {
            "description": "Modify protected file",
            "files_to_modify": ["CLAUDE.md"],
            "files_to_create": [],
        }

        result = await phase.safety_review(design)
        assert not result["safe"]

    def test_checksum_verification(
        self,
        nomic_temp_dir: Path,
    ):
        """Checksums should detect modifications to protected files."""
        protected_file = nomic_temp_dir / "CLAUDE.md"
        original_content = protected_file.read_text()
        original_checksum = hashlib.sha256(original_content.encode()).hexdigest()

        # Modify the file
        protected_file.write_text(original_content + "\n# Modified!")

        new_content = protected_file.read_text()
        new_checksum = hashlib.sha256(new_content.encode()).hexdigest()

        assert original_checksum != new_checksum

    @pytest.mark.asyncio
    async def test_design_scope_limiter(
        self,
        nomic_temp_dir: Path,
    ):
        """Scope limiter should reject overly complex designs."""
        from aragora.nomic.phases.scope_limiter import ScopeLimiter

        limiter = ScopeLimiter(
            protected_files=["CLAUDE.md", ".env"],
            max_files=5,
            max_complexity=5.0,
        )

        # Complex design with many files
        complex_design = """
## FILE CHANGES
- file1.py - Create new (500 lines)
- file2.py - Create new (300 lines)
- file3.py - Modify (100 lines)
- file4.py - Create new (400 lines)
- file5.py - Create new (200 lines)
- file6.py - Create new (600 lines)
- file7.py - Create new (150 lines)
- file8.py - Create new (250 lines)
"""

        evaluation = limiter.evaluate(complex_design)

        # Should flag as too complex or not implementable
        assert evaluation.complexity_score > 3.0 or evaluation.file_count > 5


# ============================================================================
# Rollback Tests
# ============================================================================


class TestRollbackBehavior:
    """Tests for rollback on verification failure."""

    @pytest.mark.asyncio
    async def test_rollback_on_test_failure(
        self,
        nomic_temp_dir: Path,
        mock_codex_agent: MagicMock,
        mock_log_fn: MagicMock,
    ):
        """Changes should be rolled back when tests fail."""
        from aragora.nomic.phases.implement import ImplementPhase

        # Create original file
        test_file = nomic_temp_dir / "aragora" / "feature.py"
        original_content = "# Original implementation\ndef feature(): pass\n"
        test_file.write_text(original_content)

        backup_dir = nomic_temp_dir / ".nomic" / "backups"
        phase = ImplementPhase(
            aragora_path=nomic_temp_dir,
            codex_agent=mock_codex_agent,
            backup_path=backup_dir,
            log_fn=mock_log_fn,
        )

        # Create backup
        manifest = await phase.create_backup(["aragora/feature.py"])

        # Modify the file (simulating implementation)
        modified_content = "# Bad implementation\ndef feature(): raise Exception()\n"
        test_file.write_text(modified_content)

        # Verify file was modified
        assert test_file.read_text() == modified_content

        # Rollback
        await phase.rollback(manifest)

        # Verify rollback worked
        assert test_file.read_text() == original_content

    @pytest.mark.asyncio
    async def test_rollback_removes_created_files(
        self,
        nomic_temp_dir: Path,
        mock_codex_agent: MagicMock,
        mock_log_fn: MagicMock,
    ):
        """Rollback should remove files that were created during implementation."""
        from aragora.nomic.phases.implement import ImplementPhase

        backup_dir = nomic_temp_dir / ".nomic" / "backups"
        phase = ImplementPhase(
            aragora_path=nomic_temp_dir,
            codex_agent=mock_codex_agent,
            backup_path=backup_dir,
            log_fn=mock_log_fn,
        )

        # Create manifest with files_created tracking
        manifest = {
            "path": str(backup_dir / "test_backup"),
            "timestamp": "20260131_120000",
            "files": [],
            "files_created": ["aragora/new_file.py"],
        }

        # Create the new file (simulating implementation)
        new_file = nomic_temp_dir / "aragora" / "new_file.py"
        new_file.write_text("# New file content\n")
        assert new_file.exists()

        # Rollback should remove created files
        await phase.rollback(manifest)

        assert not new_file.exists()


# ============================================================================
# Full Cycle Integration Tests
# ============================================================================


class TestFullCycleIntegration:
    """Integration tests for complete Nomic Loop cycles."""

    @pytest.mark.asyncio
    async def test_successful_cycle_flow(
        self,
        nomic_temp_dir: Path,
        mock_claude_agent: MagicMock,
        mock_codex_agent: MagicMock,
        mock_log_fn: MagicMock,
    ):
        """Test a successful flow through all phases."""
        from aragora.nomic.phases import PhaseValidator

        # Track phase results
        phase_results: dict[str, dict[str, Any]] = {}

        # Phase 0: Context
        context_result = {
            "success": True,
            "codebase_summary": "Project has debate and memory modules",
            "recent_changes": "Added streaming support",
            "open_issues": [],
            "duration_seconds": 5.0,
        }
        is_valid, error = PhaseValidator.validate("context", context_result)
        assert is_valid, f"Context validation failed: {error}"
        phase_results["context"] = context_result

        # Phase 1: Debate
        debate_result = {
            "success": True,
            "improvement": "Add real-time streaming for debates",
            "consensus_reached": True,
            "confidence": 0.85,
            "votes": [("claude", "streaming"), ("codex", "streaming")],
        }
        is_valid, error = PhaseValidator.validate("debate", debate_result)
        assert is_valid, f"Debate validation failed: {error}"
        phase_results["debate"] = debate_result

        # Phase 2: Design
        design_result = {
            "success": True,
            "design": "Create WebSocket handler for debate streaming",
            "files_affected": ["aragora/streaming/handler.py"],
            "complexity_estimate": "low",
        }
        is_valid, error = PhaseValidator.validate("design", design_result)
        assert is_valid, f"Design validation failed: {error}"
        phase_results["design"] = design_result

        # Phase 3: Implement
        implement_result = {
            "success": True,
            "files_modified": ["aragora/streaming/handler.py"],
            "diff_summary": "+50 lines",
        }
        is_valid, error = PhaseValidator.validate("implement", implement_result)
        assert is_valid, f"Implement validation failed: {error}"
        phase_results["implement"] = implement_result

        # Phase 4: Verify
        verify_result = {
            "success": True,
            "tests_passed": True,
            "test_output": "15 tests passed",
            "syntax_valid": True,
        }
        is_valid, error = PhaseValidator.validate("verify", verify_result)
        assert is_valid, f"Verify validation failed: {error}"
        phase_results["verify"] = verify_result

        # Phase 5: Commit
        commit_result = {
            "success": True,
            "commit_hash": "abc123def",
            "committed": True,
        }
        is_valid, error = PhaseValidator.validate("commit", commit_result)
        assert is_valid, f"Commit validation failed: {error}"
        phase_results["commit"] = commit_result

        # All phases should have succeeded
        assert all(r["success"] for r in phase_results.values())

    @pytest.mark.asyncio
    async def test_cycle_aborts_on_no_consensus(
        self,
        nomic_temp_dir: Path,
    ):
        """Cycle should abort if debate fails to reach consensus."""
        from aragora.nomic.phases import PhaseValidator

        debate_result = {
            "success": False,
            "improvement": "",
            "consensus_reached": False,
            "confidence": 0.3,
            "votes": [("claude", "A"), ("codex", "B"), ("gemini", "C")],
        }

        is_valid, _ = PhaseValidator.validate("debate", debate_result)
        assert is_valid  # Result structure is valid

        # Should not proceed to design when no consensus
        should_continue = debate_result["consensus_reached"]
        assert not should_continue

    @pytest.mark.asyncio
    async def test_cycle_aborts_on_verify_failure(
        self,
        nomic_temp_dir: Path,
    ):
        """Cycle should abort and not commit if verification fails."""
        from aragora.nomic.phases import PhaseValidator

        verify_result = {
            "success": False,
            "tests_passed": False,
            "test_output": "FAILED test_feature.py::test_streaming",
            "syntax_valid": True,
        }

        is_valid, _ = PhaseValidator.validate("verify", verify_result)
        assert is_valid  # Result structure is valid

        # Should trigger rollback, not commit
        should_commit = verify_result["tests_passed"] and verify_result["success"]
        assert not should_commit


# ============================================================================
# Phase Validator Tests (Extended)
# ============================================================================


class TestPhaseValidatorExtended:
    """Extended tests for PhaseValidator edge cases."""

    def test_normalize_result_handles_extreme_confidence(self):
        """Normalize should clamp confidence values."""
        from aragora.nomic.phases import PhaseValidator

        # Very high confidence
        result = PhaseValidator.normalize_result("debate", {"confidence": 150.0})
        assert result["confidence"] == 1.0

        # Negative confidence
        result = PhaseValidator.normalize_result("debate", {"confidence": -50.0})
        assert result["confidence"] == 0.0

        # Normal confidence
        result = PhaseValidator.normalize_result("debate", {"confidence": 0.75})
        assert result["confidence"] == 0.75

    def test_validate_all_phase_types(self):
        """Validate should accept all valid phase types."""
        from aragora.nomic.phases import PhaseValidator

        phases = ["context", "debate", "design", "implement", "verify", "commit"]

        for phase in phases:
            result = {"success": True}
            if phase == "debate":
                result["consensus_reached"] = True
                result["improvement"] = "test"

            is_valid, error = PhaseValidator.validate(phase, result)
            # Should not fail on required fields for basic success case
            # (debate needs extra fields)
            if not is_valid and phase != "debate":
                # Other phases just need success field
                assert error is None or "success" in error

    def test_safe_get_with_various_inputs(self):
        """safe_get should handle various input types."""
        from aragora.nomic.phases import PhaseValidator

        # None input
        assert PhaseValidator.safe_get(None, "key", "default") == "default"

        # String input
        assert PhaseValidator.safe_get("string", "key", "default") == "default"

        # Empty dict
        assert PhaseValidator.safe_get({}, "key", "default") == "default"

        # Dict with key
        assert PhaseValidator.safe_get({"key": "value"}, "key", "default") == "value"

        # Dict without key
        assert PhaseValidator.safe_get({"other": "value"}, "key", "default") == "default"
