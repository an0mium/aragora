"""Tests for presence-only secrets CLI commands."""

from __future__ import annotations

import json
from types import SimpleNamespace

from aragora.cli.commands.secrets import cmd_secrets_health
from aragora.cli.parser import build_parser
from aragora.config.secrets import clear_secret_cache, reset_secret_manager


class TestSecretsParser:
    def test_health_parses(self):
        args = build_parser().parse_args(
            ["secrets", "health", "--json", "--name", "GEMINI_API_KEY"]
        )

        assert args.command == "secrets"
        assert args.secrets_command == "health"
        assert args.json is True
        assert args.name == ["GEMINI_API_KEY"]

    def test_hydrate_parses(self):
        args = build_parser().parse_args(["secrets", "hydrate", "--overwrite"])

        assert args.command == "secrets"
        assert args.secrets_command == "hydrate"
        assert args.overwrite is True


class TestSecretsHealthCommand:
    def setup_method(self):
        reset_secret_manager()
        clear_secret_cache()

    def teardown_method(self):
        reset_secret_manager()
        clear_secret_cache()

    def test_health_json_reports_presence_without_value(self, monkeypatch, capsys):
        monkeypatch.setenv("OPENROUTER_API_KEY", "do-not-print")
        monkeypatch.setenv("ARAGORA_SECRETS_STRICT", "false")

        args = SimpleNamespace(name=["OPENROUTER_API_KEY"], json=True, require_all=False)
        assert cmd_secrets_health(args) == 0

        output = capsys.readouterr().out
        payload = json.loads(output)
        assert payload["secrets"][0]["source"] == "env"
        assert "do-not-print" not in output

    def test_require_all_fails_for_missing_secret(self, monkeypatch, capsys):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("ARAGORA_SECRETS_STRICT", "false")

        args = SimpleNamespace(name=["GEMINI_API_KEY"], json=True, require_all=True)
        assert cmd_secrets_health(args) == 1

        output = capsys.readouterr().out
        payload = json.loads(output)
        assert payload["secrets"][0]["source"] == "missing"
