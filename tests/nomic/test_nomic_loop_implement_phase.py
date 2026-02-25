"""Tests for the NomicLoop._create_implement_phase wiring and ImplementPhase behavior."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.convoy_executor import GastownConvoyExecutor
from aragora.nomic.phases.implement import (
    DEFAULT_PROTECTED_FILES,
    ImplementPhase,
    SAFETY_PREAMBLE,
)
from scripts.nomic_loop import NomicLoop


class DummyAgent:
    def __init__(self, name: str) -> None:
        self.name = name


# ===========================================================================
# Wiring: NomicLoop._create_implement_phase
# ===========================================================================


@pytest.mark.asyncio
async def test_create_implement_phase_prefers_gastown_executor(tmp_path):
    """NomicLoop should wire a GastownConvoyExecutor into the ImplementPhase."""
    with patch.object(NomicLoop, "_init_agents", lambda self: None):
        loop = NomicLoop(aragora_path=str(tmp_path))
    loop.claude = DummyAgent("anthropic-api")
    loop.codex = DummyAgent("openai-api")

    phase = loop._create_implement_phase()
    assert isinstance(phase._executor, GastownConvoyExecutor)


@pytest.mark.asyncio
async def test_create_implement_phase_returns_implement_phase(tmp_path):
    """The returned object should be an ImplementPhase instance."""
    with patch.object(NomicLoop, "_init_agents", lambda self: None):
        loop = NomicLoop(aragora_path=str(tmp_path))
    loop.claude = DummyAgent("anthropic-api")
    loop.codex = DummyAgent("openai-api")

    phase = loop._create_implement_phase()
    assert isinstance(phase, ImplementPhase)


# ===========================================================================
# ImplementPhase initialization
# ===========================================================================


def test_init_default_protected_files(tmp_path):
    """Default protected files should include critical project files."""
    phase = ImplementPhase(aragora_path=tmp_path)
    assert "CLAUDE.md" in phase.protected_files
    assert "scripts/nomic_loop.py" in phase.protected_files


def test_init_custom_protected_files(tmp_path):
    """Custom protected files should override the default list."""
    custom = ["my_file.py"]
    phase = ImplementPhase(aragora_path=tmp_path, protected_files=custom)
    assert phase.protected_files == custom


def test_init_defaults_for_optional_args(tmp_path):
    """Optional arguments should have sensible defaults."""
    phase = ImplementPhase(aragora_path=tmp_path)
    assert phase.cycle_count == 0
    assert phase._executor is None
    assert phase._plan_generator is None
    assert phase.codex is None
    assert phase.backup_path == tmp_path / ".nomic" / "backups"


def test_init_with_all_args(tmp_path):
    """Providing all arguments should store them correctly."""
    mock_gen = MagicMock()
    mock_exec = MagicMock()
    mock_log = MagicMock()
    mock_codex = DummyAgent("codex")

    phase = ImplementPhase(
        aragora_path=tmp_path,
        plan_generator=mock_gen,
        executor=mock_exec,
        cycle_count=5,
        log_fn=mock_log,
        codex_agent=mock_codex,
    )
    assert phase._plan_generator is mock_gen
    assert phase._executor is mock_exec
    assert phase.cycle_count == 5
    assert phase.codex is mock_codex


# ===========================================================================
# Syntax validation
# ===========================================================================


def test_validate_syntax_valid_code(tmp_path):
    """Valid Python code should pass validation."""
    phase = ImplementPhase(aragora_path=tmp_path)
    assert phase.validate_syntax("x = 1 + 2\ndef foo(): pass") is True


def test_validate_syntax_invalid_code(tmp_path):
    """Invalid Python code should fail validation."""
    phase = ImplementPhase(aragora_path=tmp_path)
    assert phase.validate_syntax("def broken(:\n  pass") is False


def test_validate_syntax_empty_string(tmp_path):
    """Empty code string should be valid (no syntax error)."""
    phase = ImplementPhase(aragora_path=tmp_path)
    assert phase.validate_syntax("") is True


# ===========================================================================
# Dangerous pattern detection
# ===========================================================================


def test_check_dangerous_patterns_safe_code(tmp_path):
    """Safe code should report no dangerous patterns."""
    phase = ImplementPhase(aragora_path=tmp_path)
    result = phase.check_dangerous_patterns("x = 1 + 2\ndef hello(): return 'hi'")
    assert result["safe"] is True
    assert result["patterns_found"] == []


def test_check_dangerous_patterns_unsafe_code(tmp_path):
    """Code with eval/exec/os.system should be flagged."""
    phase = ImplementPhase(aragora_path=tmp_path)
    result = phase.check_dangerous_patterns("eval(user_input)\nos.system('rm -rf /')")
    assert result["safe"] is False
    assert len(result["patterns_found"]) >= 2
    patterns = [p["pattern"] for p in result["patterns_found"]]
    assert "eval(" in patterns
    assert "os.system" in patterns


def test_check_dangerous_patterns_subprocess(tmp_path):
    """subprocess.call should be flagged as dangerous."""
    phase = ImplementPhase(aragora_path=tmp_path)
    result = phase.check_dangerous_patterns("subprocess.call(['ls'])")
    assert result["safe"] is False
    assert any(p["pattern"] == "subprocess.call" for p in result["patterns_found"])


def test_check_dangerous_patterns_open(tmp_path):
    """open() should be flagged as dangerous."""
    phase = ImplementPhase(aragora_path=tmp_path)
    result = phase.check_dangerous_patterns("f = open('secret.txt')")
    assert result["safe"] is False


# ===========================================================================
# Legacy run() method
# ===========================================================================


@pytest.mark.asyncio
async def test_run_with_no_codex_returns_empty(tmp_path):
    """Without codex agent, generate_code returns empty dict."""
    phase = ImplementPhase(aragora_path=tmp_path)
    result = await phase.run("add error handling")
    # No code changes => success with no files
    assert result["success"] is True
    assert result.get("files_modified", []) == []


@pytest.mark.asyncio
async def test_run_with_codex_writes_file(tmp_path):
    """With codex agent, run should generate and write code."""
    codex = AsyncMock()
    codex.generate = AsyncMock(return_value="print('hello')")
    phase = ImplementPhase(aragora_path=tmp_path, codex_agent=codex)

    result = await phase.run("add greeting")
    assert result["success"] is True
    assert "generated.py" in result["files_modified"]
    assert (tmp_path / "generated.py").exists()


@pytest.mark.asyncio
async def test_run_with_invalid_syntax_fails(tmp_path):
    """If generated code has syntax errors, run should fail."""
    codex = AsyncMock()
    codex.generate = AsyncMock(return_value="def broken(:\n  pass")
    phase = ImplementPhase(aragora_path=tmp_path, codex_agent=codex)

    result = await phase.run("add feature")
    assert result["success"] is False
    assert "Syntax error" in result["error"]


# ===========================================================================
# Backup and rollback
# ===========================================================================


@pytest.mark.asyncio
async def test_create_backup_copies_existing_files(tmp_path):
    """create_backup should copy existing files to a backup directory."""
    (tmp_path / "foo.py").write_text("original")
    phase = ImplementPhase(aragora_path=tmp_path)

    manifest = await phase.create_backup(["foo.py"])
    assert "foo.py" in manifest["files"]
    assert Path(manifest["path"]).exists()


@pytest.mark.asyncio
async def test_create_backup_skips_missing_files(tmp_path):
    """Backup should skip files that don't exist yet."""
    phase = ImplementPhase(aragora_path=tmp_path)
    manifest = await phase.create_backup(["nonexistent.py"])
    assert manifest["files"] == []


@pytest.mark.asyncio
async def test_rollback_restores_backed_up_files(tmp_path):
    """rollback should restore files from backup and remove created files."""
    original_content = "original content"
    (tmp_path / "existing.py").write_text(original_content)

    phase = ImplementPhase(aragora_path=tmp_path)
    manifest = await phase.create_backup(["existing.py"])

    # Modify the original
    (tmp_path / "existing.py").write_text("modified content")
    # Create a new file tracked in manifest
    (tmp_path / "new_file.py").write_text("new")
    manifest["files_created"] = ["new_file.py"]

    await phase.rollback(manifest)

    assert (tmp_path / "existing.py").read_text() == original_content
    assert not (tmp_path / "new_file.py").exists()


# ===========================================================================
# Write files
# ===========================================================================


@pytest.mark.asyncio
async def test_write_files_creates_new_files(tmp_path):
    """write_files should create new files and report them."""
    phase = ImplementPhase(aragora_path=tmp_path)
    result = await phase.write_files({"subdir/new.py": "print('hello')"})
    assert result["success"] is True
    assert "subdir/new.py" in result["files_written"]
    assert "subdir/new.py" in result["files_created"]
    assert (tmp_path / "subdir" / "new.py").read_text() == "print('hello')"


@pytest.mark.asyncio
async def test_write_files_overwrites_existing(tmp_path):
    """write_files should overwrite existing files."""
    (tmp_path / "target.py").write_text("old")
    phase = ImplementPhase(aragora_path=tmp_path)
    result = await phase.write_files({"target.py": "new"})
    assert result["success"] is True
    assert "target.py" not in result["files_created"]  # Not new
    assert (tmp_path / "target.py").read_text() == "new"


# ===========================================================================
# Constitution / protected files verification
# ===========================================================================


@pytest.mark.asyncio
async def test_verify_constitution_flags_protected_files(tmp_path):
    """Modifying protected files should be flagged as a violation."""
    phase = ImplementPhase(aragora_path=tmp_path)
    violation = await phase._verify_constitution_compliance(
        modified_files=["CLAUDE.md", "aragora/feature.py"],
        diff="some diff",
    )
    assert violation is not None
    assert "CLAUDE.md" in violation


@pytest.mark.asyncio
async def test_verify_constitution_allows_safe_files(tmp_path):
    """Non-protected files should not trigger violations (no verifier)."""
    phase = ImplementPhase(aragora_path=tmp_path)
    violation = await phase._verify_constitution_compliance(
        modified_files=["aragora/new_feature.py"],
        diff="some diff",
    )
    assert violation is None


# ===========================================================================
# Safety preamble and defaults
# ===========================================================================


def test_safety_preamble_contains_rules():
    """SAFETY_PREAMBLE should include critical safety instructions."""
    assert "NEVER delete" in SAFETY_PREAMBLE
    assert "NEVER remove existing functionality" in SAFETY_PREAMBLE


def test_default_protected_files_list():
    """DEFAULT_PROTECTED_FILES should be a non-empty list."""
    assert len(DEFAULT_PROTECTED_FILES) >= 4
    assert "CLAUDE.md" in DEFAULT_PROTECTED_FILES
    assert ".env" in DEFAULT_PROTECTED_FILES
