"""Tests for the secrets_manager CLI tool."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Path to the script
SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "secrets_manager.py"

# Import from scripts directory for direct unit tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from secrets_manager import SECRETS, SECRETS_BY_ENV_VAR, LocalEnvBackend, mask_secret


class TestSecretsManagerCLI:
    """Tests for secrets_manager.py CLI via subprocess."""

    def run_cli(self, *args):
        """Run the secrets_manager CLI with given arguments."""
        cmd = [sys.executable, str(SCRIPT_PATH)] + list(args)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env={**os.environ},
        )
        return result

    def test_help_flag(self):
        """Test that --help returns usage information."""
        result = self.run_cli("--help")
        assert result.returncode == 0
        assert "usage" in result.stdout.lower()

    def test_explain_flag(self):
        """Test that --explain prints rotation explanation."""
        result = self.run_cli("--explain")
        assert result.returncode == 0
        assert "WHY MANUAL ROTATION" in result.stdout

    def test_no_args_shows_usage(self):
        """Test that running with no args shows usage and exits with error."""
        result = self.run_cli()
        assert result.returncode != 0


class TestMaskSecret:
    """Tests for the mask_secret utility function."""

    def test_mask_none(self):
        """Test masking None returns '(not set)'."""
        assert mask_secret(None) == "(not set)"

    def test_mask_empty(self):
        """Test masking empty string returns '(not set)'."""
        assert mask_secret("") == "(not set)"

    def test_mask_short(self):
        """Test masking a short value returns '****'."""
        assert mask_secret("abcd") == "****"

    def test_mask_exactly_eight(self):
        """Test masking an 8-character value returns '****'."""
        assert mask_secret("12345678") == "****"

    def test_mask_normal(self):
        """Test masking a normal-length secret shows first 4 and last 4."""
        result = mask_secret("sk-ant-abcdef1234")
        assert result == "sk-a...1234"


class TestSecretDefinitions:
    """Tests for the SECRETS registry definitions."""

    def test_all_secrets_have_env_var(self):
        """Every secret definition must have a non-empty env_var."""
        for secret in SECRETS:
            assert secret.env_var, f"Secret '{secret.name}' has empty env_var"

    def test_secrets_by_env_var_matches(self):
        """SECRETS_BY_ENV_VAR should have the same count as SECRETS (no duplicates)."""
        assert len(SECRETS_BY_ENV_VAR) == len(SECRETS)

    def test_all_secrets_have_name(self):
        """Every secret definition must have a non-empty name."""
        for secret in SECRETS:
            assert secret.name, f"Secret with env_var '{secret.env_var}' has empty name"


class TestLocalEnvBackend:
    """Tests for the LocalEnvBackend .env file reader."""

    def test_load_from_env_file(self, tmp_path):
        """Test loading key-value pairs from an .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("FOO=bar\nBAZ=qux\n")

        backend = LocalEnvBackend(env_path=env_file)
        assert backend.get("FOO") == "bar"
        assert backend.get("BAZ") == "qux"

    def test_handles_comments(self, tmp_path):
        """Test that comments and blank lines are skipped."""
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\nKEY=value\n\n# another\n")

        backend = LocalEnvBackend(env_path=env_file)
        assert backend.get("KEY") == "value"
        assert backend.get("# comment") is None

    def test_nonexistent_file(self, tmp_path):
        """Test that a nonexistent .env file returns None for any key."""
        env_file = tmp_path / "nonexistent.env"

        backend = LocalEnvBackend(env_path=env_file)
        assert backend.get("KEY") is None
