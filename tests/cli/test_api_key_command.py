"""Tests for secure CLI LLM API key management."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aragora.cli.api_keys import (
    get_provider_key,
    hydrate_env_from_secure_store,
    list_provider_statuses,
    validate_provider_key,
)
from aragora.cli.commands.api_key import cmd_api_key
from aragora.cli.parser import build_parser


def _file_store_env(tmp_path: Path) -> dict[str, str]:
    return {
        "ARAGORA_API_KEY_BACKEND": "file",
        "HOME": str(tmp_path),
    }


class TestApiKeyParser:
    def test_set_parses(self):
        args = build_parser().parse_args(["api-key", "set", "openai", "sk-test-1234"])
        assert args.command == "api-key"
        assert args.api_key_command == "set"
        assert args.provider == "openai"
        assert args.key == "sk-test-1234"

    def test_set_allows_hidden_prompt_without_cli_key(self):
        args = build_parser().parse_args(["api-key", "set", "openai"])
        assert args.command == "api-key"
        assert args.api_key_command == "set"
        assert args.provider == "openai"
        assert args.key is None

    def test_list_parses(self):
        args = build_parser().parse_args(["api-key", "list"])
        assert args.command == "api-key"
        assert args.api_key_command == "list"

    def test_validate_parses(self):
        args = build_parser().parse_args(["api-key", "validate", "anthropic"])
        assert args.command == "api-key"
        assert args.api_key_command == "validate"
        assert args.provider == "anthropic"


class TestValidateParserHelp:
    """The top-level ``validate`` command must document what it actually does."""

    def _validate_subparser(self):
        parser = build_parser()
        # argparse stores subparsers under the _SubParsersAction choices.
        for action in parser._actions:  # noqa: SLF001 - introspecting argparse
            choices = getattr(action, "choices", None)
            if choices and "validate" in choices:
                return choices["validate"]
        raise AssertionError("validate subparser not found")

    def test_validate_has_a_description(self):
        subparser = self._validate_subparser()
        assert subparser.description, "validate subparser must set a description"
        # Description should mention the actual behavior (full health check).
        assert "health" in subparser.description.lower()

    def test_validate_help_output_includes_description(self, capsys):
        subparser = self._validate_subparser()
        help_text = subparser.format_help()
        # The empty-description bug emitted only usage + the -h line.
        assert "health" in help_text.lower()


class TestApiKeyCommands:
    def test_set_stores_key_in_encrypted_file_backend(self, tmp_path, monkeypatch, capsys):
        for key, value in _file_store_env(tmp_path).items():
            monkeypatch.setenv(key, value)

        args = SimpleNamespace(api_key_command="set", provider="openai", key="sk-test-1234")
        cmd_api_key(args)

        store_path = tmp_path / ".aragora" / "api_keys.json"
        store_contents = store_path.read_text(encoding="utf-8")
        assert "sk-test-1234" not in store_contents
        assert '"backend": "file"' in store_contents

        resolved_key, source = get_provider_key("openai")
        assert resolved_key == "sk-test-1234"
        assert source == "secure-store"

        output = capsys.readouterr().out
        assert "Stored OpenAI API key" in output
        assert "Backend:  file" in output

    def test_set_prompts_securely_when_key_argument_missing(self, tmp_path, monkeypatch, capsys):
        for key, value in _file_store_env(tmp_path).items():
            monkeypatch.setenv(key, value)

        with patch("aragora.cli.commands.api_key.getpass.getpass", return_value="sk-prompt-5678"):
            cmd_api_key(SimpleNamespace(api_key_command="set", provider="openai", key=None))

        resolved_key, source = get_provider_key("openai")
        assert resolved_key == "sk-prompt-5678"
        assert source == "secure-store"
        output = capsys.readouterr().out
        assert "Stored OpenAI API key" in output

    def test_list_includes_secure_store_and_env_override(self, tmp_path, monkeypatch, capsys):
        for key, value in _file_store_env(tmp_path).items():
            monkeypatch.setenv(key, value)

        cmd_api_key(SimpleNamespace(api_key_command="set", provider="openai", key="sk-test-1234"))
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-override")

        cmd_api_key(SimpleNamespace(api_key_command="list"))

        output = capsys.readouterr().out
        assert "openai" in output
        assert "environment override (OPENAI_API_KEY)" in output
        assert "sk-e...ride" in output

    def test_hydrate_env_from_secure_store_preserves_existing_env(self, tmp_path, monkeypatch):
        for key, value in _file_store_env(tmp_path).items():
            monkeypatch.setenv(key, value)

        cmd_api_key(SimpleNamespace(api_key_command="set", provider="anthropic", key="sk-ant-1234"))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-override")

        hydrated = hydrate_env_from_secure_store()

        assert hydrated == {}
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-env-override"

    def test_validate_uses_stored_key(self, tmp_path, monkeypatch, capsys):
        for key, value in _file_store_env(tmp_path).items():
            monkeypatch.setenv(key, value)

        cmd_api_key(SimpleNamespace(api_key_command="set", provider="openai", key="sk-test-1234"))
        response = MagicMock(status_code=200)

        with patch("aragora.security.safe_http.safe_get", return_value=response):
            with pytest.raises(SystemExit) as exc_info:
                cmd_api_key(SimpleNamespace(api_key_command="validate", provider="openai"))

        assert exc_info.value.code == 0
        output = capsys.readouterr().out
        assert "OpenAI API key validation" in output
        assert "Remote check: valid" in output
        assert "API key is valid" in output

    def test_validate_fails_for_missing_key(self, tmp_path, monkeypatch, capsys):
        for key, value in _file_store_env(tmp_path).items():
            monkeypatch.setenv(key, value)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            cmd_api_key(SimpleNamespace(api_key_command="validate", provider="mistral"))

        assert exc_info.value.code == 1
        output = capsys.readouterr().out
        assert "No API key configured for Mistral" in output

    def test_deepseek_validation_makes_a_real_test_call(self, tmp_path, monkeypatch):
        """A DeepSeek key must be live-probed, not rubber-stamped as a pass.

        Regression: ``_probe_provider_key`` previously returned ``"skipped"``
        for DeepSeek with no network call, and ``"skipped"`` was mapped to
        ``is_valid=True`` — so a fabricated key reported a verified pass.
        """
        for key, value in _file_store_env(tmp_path).items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-totallyfakekey1234567890abcdef")

        # 200 → key really is valid.
        ok_response = MagicMock(status_code=200)
        with patch("aragora.security.safe_http.safe_get", return_value=ok_response) as mock_get:
            report = validate_provider_key("deepseek")

        # A real test call was made against DeepSeek.
        mock_get.assert_called_once()
        assert "deepseek.com" in mock_get.call_args.args[0]
        assert report.remote_status == "valid"
        assert report.is_valid is True
        assert report.unverified is False

    def test_bogus_deepseek_key_is_not_reported_valid(self, tmp_path, monkeypatch):
        """A fabricated DeepSeek key rejected by the provider must report as
        invalid (is_valid False), not as a passing/verified check."""
        for key, value in _file_store_env(tmp_path).items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-totallyfakekey1234567890abcdef")

        rejected = MagicMock(status_code=401)
        with patch("aragora.security.safe_http.safe_get", return_value=rejected):
            report = validate_provider_key("deepseek")

        assert report.is_valid is False
        assert report.unverified is False
        assert report.remote_status == "invalid"

    def test_skipped_remote_status_is_unverified_not_valid(self, tmp_path, monkeypatch):
        """When live validation genuinely cannot run (probe returns 'skipped'),
        the key is reported as present-but-unverified, never a verified pass."""
        for key, value in _file_store_env(tmp_path).items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv("MISTRAL_API_KEY", "sk-mistral-fakekey1234567890")

        with patch(
            "aragora.cli.api_keys._probe_provider_key",
            return_value=("skipped", "live validation is unavailable"),
        ):
            report = validate_provider_key("mistral")

        assert report.remote_status == "skipped"
        # The crux: skipped must NOT be a verified pass.
        assert report.is_valid is False
        assert report.unverified is True

    def test_list_provider_statuses_shows_environment_alias(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "google-test-key")

        statuses = {status.provider: status for status in list_provider_statuses()}

        assert statuses["gemini"].configured is True
        assert statuses["gemini"].source == "environment (GOOGLE_API_KEY)"
        assert statuses["gemini"].masked_value == "goog...-key"
