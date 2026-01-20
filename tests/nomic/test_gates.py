"""
Tests for Nomic Loop Safety Gates.

Safety gates prevent dangerous operations:
- Protected file modifications
- Excessive code changes
- Dangerous patterns
- Resource limits
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestProtectedFileGate:
    """Tests for protected file detection."""

    def test_detects_claude_md_modification(self):
        """Should block CLAUDE.md modifications."""
        from aragora.nomic.gates import is_protected_file

        assert is_protected_file("CLAUDE.md") is True
        assert is_protected_file("./CLAUDE.md") is True
        assert is_protected_file("aragora/../CLAUDE.md") is True

    def test_detects_core_py_modification(self):
        """Should block core.py modifications."""
        from aragora.nomic.gates import is_protected_file

        assert is_protected_file("core.py") is True
        assert is_protected_file("aragora/core.py") is True

    def test_detects_init_modification(self):
        """Should block __init__.py modifications in root."""
        from aragora.nomic.gates import is_protected_file

        assert is_protected_file("aragora/__init__.py") is True

    def test_allows_regular_files(self):
        """Should allow regular file modifications."""
        from aragora.nomic.gates import is_protected_file

        assert is_protected_file("aragora/utils.py") is False
        assert is_protected_file("tests/test_utils.py") is False
        assert is_protected_file("aragora/debate/orchestrator.py") is False

    def test_detects_env_files(self):
        """Should block .env file modifications."""
        from aragora.nomic.gates import is_protected_file

        assert is_protected_file(".env") is True
        assert is_protected_file(".env.local") is True


class TestChangeVolumeGate:
    """Tests for change volume limits."""

    def test_blocks_excessive_file_changes(self):
        """Should block changes to too many files."""
        from aragora.nomic.gates import check_change_volume

        files = [f"file{i}.py" for i in range(100)]

        result = check_change_volume(files_changed=files, max_files=20)

        assert result["allowed"] is False
        assert "too many files" in result["reason"].lower()

    def test_allows_reasonable_file_changes(self):
        """Should allow reasonable number of file changes."""
        from aragora.nomic.gates import check_change_volume

        files = ["file1.py", "file2.py", "file3.py"]

        result = check_change_volume(files_changed=files, max_files=20)

        assert result["allowed"] is True

    def test_blocks_excessive_line_changes(self):
        """Should block changes with too many line modifications."""
        from aragora.nomic.gates import check_change_volume

        result = check_change_volume(
            files_changed=["file.py"],
            lines_added=5000,
            lines_removed=100,
            max_lines=1000,
        )

        assert result["allowed"] is False
        assert "too many lines" in result["reason"].lower()


class TestDangerousPatternGate:
    """Tests for dangerous code pattern detection."""

    def test_detects_eval_usage(self):
        """Should detect eval() usage in changes."""
        from aragora.nomic.gates import check_dangerous_patterns

        code = """
def process(data):
    return eval(data)
"""
        result = check_dangerous_patterns(code)

        assert result["safe"] is False
        assert "eval" in result["patterns_found"]

    def test_detects_exec_usage(self):
        """Should detect exec() usage in changes."""
        from aragora.nomic.gates import check_dangerous_patterns

        code = """
def run_code(code_str):
    exec(code_str)
"""
        result = check_dangerous_patterns(code)

        assert result["safe"] is False
        assert "exec" in result["patterns_found"]

    def test_detects_os_system_usage(self):
        """Should detect os.system() usage in changes."""
        from aragora.nomic.gates import check_dangerous_patterns

        code = """
import os
def run_command(cmd):
    os.system(cmd)
"""
        result = check_dangerous_patterns(code)

        assert result["safe"] is False

    def test_allows_safe_code(self):
        """Should allow safe code patterns."""
        from aragora.nomic.gates import check_dangerous_patterns

        code = """
def safe_function(data):
    return data.upper()
"""
        result = check_dangerous_patterns(code)

        assert result["safe"] is True
        assert len(result.get("patterns_found", [])) == 0


class TestResourceLimitGate:
    """Tests for resource limit checking."""

    def test_blocks_long_running_operations(self):
        """Should enforce time limits on operations."""
        from aragora.nomic.gates import check_resource_limits

        result = check_resource_limits(
            estimated_duration_seconds=3600,  # 1 hour
            max_duration_seconds=600,  # 10 minutes
        )

        assert result["allowed"] is False
        assert "time" in result["reason"].lower()

    def test_allows_quick_operations(self):
        """Should allow quick operations."""
        from aragora.nomic.gates import check_resource_limits

        result = check_resource_limits(
            estimated_duration_seconds=30,
            max_duration_seconds=600,
        )

        assert result["allowed"] is True


class TestGateIntegration:
    """Integration tests for all gates combined."""

    def test_all_gates_pass(self):
        """Should pass when all gates allow the change."""
        from aragora.nomic.gates import check_all_gates

        changes = {
            "files_changed": ["aragora/utils.py"],
            "lines_added": 50,
            "lines_removed": 10,
            "code_content": "def helper(): return True",
            "estimated_duration": 30,
        }

        result = check_all_gates(changes)

        assert result["allowed"] is True
        assert len(result.get("blocked_by", [])) == 0

    def test_multiple_gates_can_fail(self):
        """Should report all failing gates."""
        from aragora.nomic.gates import check_all_gates

        changes = {
            "files_changed": ["CLAUDE.md", "core.py"] + [f"f{i}.py" for i in range(50)],
            "lines_added": 10000,
            "lines_removed": 0,
            "code_content": "eval(input())",
            "estimated_duration": 7200,
        }

        result = check_all_gates(changes)

        assert result["allowed"] is False
        assert len(result["blocked_by"]) >= 2  # Multiple gates should fail


class TestGateConfiguration:
    """Tests for gate configuration."""

    def test_custom_protected_files(self):
        """Should use custom protected files list."""
        from aragora.nomic.gates import GateConfig, is_protected_file

        config = GateConfig(
            protected_files=["custom_protected.py", "secret.key"],
        )

        with patch("aragora.nomic.gates._gate_config", config):
            assert is_protected_file("custom_protected.py") is True
            assert is_protected_file("secret.key") is True

    def test_custom_dangerous_patterns(self):
        """Should use custom dangerous patterns."""
        from aragora.nomic.gates import GateConfig, check_dangerous_patterns

        config = GateConfig(
            dangerous_patterns=["dangerous_func", "unsafe_call"],
        )

        code = "result = dangerous_func(data)"

        with patch("aragora.nomic.gates._gate_config", config):
            result = check_dangerous_patterns(code)
            assert result["safe"] is False
