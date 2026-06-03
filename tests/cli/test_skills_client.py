"""Tests for the skills CLI client helpers (base URL, auth headers, hints).

Regression coverage for three defects:
- _get_auth_headers ignored ARAGORA_API_TOKEN (read a non-existent settings attr).
- _get_api_base targeted localhost:8000 and ignored ARAGORA_API_URL.
- The empty-list hint pointed at the non-existent 'aragora marketplace install'.
"""

from __future__ import annotations

import inspect

import pytest

from aragora.cli.commands import skills as skills_mod
from aragora.cli.commands.skills import _get_api_base, _get_auth_headers


class TestGetApiBase:
    """Tests for _get_api_base."""

    def test_defaults_to_documented_port_8080(self, monkeypatch):
        """With no env override, base resolves to the documented 8080 port."""
        monkeypatch.delenv("ARAGORA_API_URL", raising=False)
        assert _get_api_base() == "http://localhost:8080"

    def test_honors_aragora_api_url(self, monkeypatch):
        """ARAGORA_API_URL overrides the default, matching other CLI clients."""
        monkeypatch.setenv("ARAGORA_API_URL", "https://api.example.com:9443")
        assert _get_api_base() == "https://api.example.com:9443"

    def test_never_uses_legacy_8000_port(self, monkeypatch):
        """The old localhost:8000 fallback must not reappear."""
        monkeypatch.delenv("ARAGORA_API_URL", raising=False)
        assert "8000" not in _get_api_base()


class TestGetAuthHeaders:
    """Tests for _get_auth_headers."""

    def test_returns_bearer_when_token_set(self, monkeypatch):
        """ARAGORA_API_TOKEN must be sent as a Bearer Authorization header."""
        monkeypatch.setenv("ARAGORA_API_TOKEN", "test-token-12345")
        monkeypatch.delenv("ARAGORA_API_KEY", raising=False)
        assert _get_auth_headers() == {"Authorization": "Bearer test-token-12345"}

    def test_falls_back_to_api_key(self, monkeypatch):
        """ARAGORA_API_KEY is honored when ARAGORA_API_TOKEN is unset."""
        monkeypatch.delenv("ARAGORA_API_TOKEN", raising=False)
        monkeypatch.setenv("ARAGORA_API_KEY", "key-abc")
        assert _get_auth_headers() == {"Authorization": "Bearer key-abc"}

    def test_token_takes_precedence_over_key(self, monkeypatch):
        """ARAGORA_API_TOKEN wins over ARAGORA_API_KEY when both are set."""
        monkeypatch.setenv("ARAGORA_API_TOKEN", "tok")
        monkeypatch.setenv("ARAGORA_API_KEY", "key")
        assert _get_auth_headers() == {"Authorization": "Bearer tok"}

    def test_empty_when_no_token(self, monkeypatch):
        """No credentials -> no Authorization header."""
        monkeypatch.delenv("ARAGORA_API_TOKEN", raising=False)
        monkeypatch.delenv("ARAGORA_API_KEY", raising=False)
        assert _get_auth_headers() == {}


class TestCommandHints:
    """The empty-list / usage hints must reference working commands."""

    def test_no_marketplace_install_hint(self):
        """Source must not point users at the non-existent 'marketplace install'."""
        src = inspect.getsource(skills_mod)
        assert "aragora marketplace install" not in src
        assert "aragora marketplace uninstall" not in src
        assert "aragora marketplace info" not in src

    def test_uses_skills_install_hint(self):
        """The working 'aragora skills install' guidance is present."""
        src = inspect.getsource(skills_mod)
        assert "aragora skills install <skill_id>" in src


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
