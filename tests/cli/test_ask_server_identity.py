"""Tests for the `aragora ask` server-identity trust check (issue #8805).

`aragora ask` auto-discovers a local API server by probing
``localhost:8080/api/health``. Port 8080 is the most commonly squatted dev
port, so a bare HTTP 200 must NOT be treated as an Aragora server: the health
payload has to positively identify as Aragora (``status`` in
{"healthy", "degraded"} plus a ``timestamp`` field — the exact shape the
public /api/health endpoint returns). An explicitly configured URL
(ARAGORA_API_URL or --api-url) is trusted as-is because the user opted in.
"""

from __future__ import annotations

import io
import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from aragora.cli.commands import debate as debate_cmd

ARAGORA_MINIMAL_HEALTH = json.dumps(
    {"status": "healthy", "timestamp": "2026-07-02T12:00:00+00:00Z"}
).encode()

ARAGORA_FULL_HEALTH = json.dumps(
    {
        "status": "healthy",
        "version": "2.8.0",
        "uptime_seconds": 12,
        "demo_mode": False,
        "db_mode": "sqlite",
        "checks": {"degraded_mode": {"healthy": True}},
        "timestamp": "2026-07-02T12:00:00+00:00Z",
        "response_time_ms": 1.2,
    }
).encode()


class _FakeResponse(io.BytesIO):
    """Minimal stand-in for the urlopen response object."""

    def __init__(self, body: bytes, status: int = 200):
        super().__init__(body)
        self.status = status

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


@contextmanager
def _serving(body: bytes, status: int = 200):
    """Patch urlopen so every request gets the given response."""
    with patch(
        "urllib.request.urlopen",
        side_effect=lambda *a, **k: _FakeResponse(body, status),
    ):
        yield


@pytest.fixture(autouse=True)
def _no_env_api_url(monkeypatch):
    """Default: no explicit opt-in via environment."""
    monkeypatch.delenv("ARAGORA_API_URL", raising=False)


class TestIdentifiesAsAragora:
    def test_minimal_health_payload_accepted(self):
        assert debate_cmd._identifies_as_aragora(ARAGORA_MINIMAL_HEALTH) is True

    def test_full_health_payload_accepted(self):
        assert debate_cmd._identifies_as_aragora(ARAGORA_FULL_HEALTH) is True

    def test_degraded_status_accepted(self):
        body = json.dumps({"status": "degraded", "timestamp": "2026-07-02T12:00:00Z"}).encode()
        assert debate_cmd._identifies_as_aragora(body) is True

    def test_generic_ok_status_rejected(self):
        # Common non-Aragora health shape ({"status": "ok"}) must not match.
        body = json.dumps({"status": "ok", "timestamp": "2026-07-02T12:00:00Z"}).encode()
        assert debate_cmd._identifies_as_aragora(body) is False

    def test_missing_timestamp_rejected(self):
        assert debate_cmd._identifies_as_aragora(b'{"status": "healthy"}') is False

    def test_html_rejected(self):
        assert debate_cmd._identifies_as_aragora(b"<html><body>Jenkins</body></html>") is False

    def test_non_dict_json_rejected(self):
        assert debate_cmd._identifies_as_aragora(b'["healthy"]') is False

    def test_empty_body_rejected(self):
        assert debate_cmd._identifies_as_aragora(b"") is False

    def test_non_utf8_rejected(self):
        assert debate_cmd._identifies_as_aragora(b"\xff\xfe\x00garbage") is False


class TestProbeServerIdentity:
    URL = "http://localhost:8080"

    def test_aragora_health_response(self):
        with _serving(ARAGORA_MINIMAL_HEALTH):
            assert debate_cmd._probe_server_identity(self.URL) == "aragora"

    def test_foreign_json_response(self):
        with _serving(b'{"app": "someone-elses-dashboard"}'):
            assert debate_cmd._probe_server_identity(self.URL) == "foreign"

    def test_foreign_html_response(self):
        with _serving(b"<html>Tomcat manager</html>"):
            assert debate_cmd._probe_server_identity(self.URL) == "foreign"

    def test_connection_refused(self):
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            assert debate_cmd._probe_server_identity(self.URL) == "unavailable"

    def test_non_200_status(self):
        with _serving(b"{}", status=204):
            assert debate_cmd._probe_server_identity(self.URL) == "unavailable"


class TestTrustedServerAvailable:
    URL = debate_cmd.DEFAULT_API_URL  # the auto-discovery default

    def test_real_aragora_server_is_used(self):
        with _serving(ARAGORA_MINIMAL_HEALTH):
            assert debate_cmd._trusted_server_available(self.URL) is True

    def test_non_aragora_200_is_skipped_with_notice(self, capsys):
        with _serving(b'{"status": "ok"}'):
            assert debate_cmd._trusted_server_available(self.URL) is False
        err = capsys.readouterr().err
        assert "does not identify as an Aragora API server" in err
        assert "running the debate locally" in err

    def test_no_server_is_unavailable_without_notice(self, capsys):
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            assert debate_cmd._trusted_server_available(self.URL) is False
        assert capsys.readouterr().err == ""

    def test_explicit_env_url_trusted_without_identity_check(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_API_URL", "http://localhost:9999")
        with (
            patch.object(debate_cmd, "_is_server_available", return_value=True),
            patch.object(
                debate_cmd,
                "_probe_server_identity",
                side_effect=AssertionError("identity probe must be skipped for explicit URLs"),
            ),
        ):
            assert debate_cmd._trusted_server_available("http://localhost:9999") is True

    def test_explicit_flag_url_trusted_without_identity_check(self):
        # A URL differing from the default means the user passed --api-url.
        with (
            patch.object(debate_cmd, "_is_server_available", return_value=True),
            patch.object(
                debate_cmd,
                "_probe_server_identity",
                side_effect=AssertionError("identity probe must be skipped for explicit URLs"),
            ),
        ):
            assert debate_cmd._trusted_server_available("http://intranet:8080") is True

    def test_flag_passed_with_default_url_trusted_without_identity_check(self):
        # Round-1 review [P2]: `--api-url http://localhost:8080` equals the
        # default value, but the user typed the flag — that is an explicit
        # opt-in and must skip the identity probe. Flag presence comes from
        # argparse (parser default is None), threaded through flag_passed.
        with (
            patch.object(debate_cmd, "_is_server_available", return_value=True),
            patch.object(
                debate_cmd,
                "_probe_server_identity",
                side_effect=AssertionError("identity probe must be skipped for explicit URLs"),
            ),
        ):
            assert (
                debate_cmd._trusted_server_available(debate_cmd.DEFAULT_API_URL, flag_passed=True)
                is True
            )

    def test_explicit_url_still_requires_reachability(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_API_URL", "http://localhost:9999")
        with patch.object(debate_cmd, "_is_server_available", return_value=False):
            assert debate_cmd._trusted_server_available("http://localhost:9999") is False


class TestExplicitConfigDetection:
    def test_default_url_is_not_explicit(self):
        assert debate_cmd._is_explicitly_configured_api_url(debate_cmd.DEFAULT_API_URL) is False

    def test_env_var_makes_url_explicit(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_API_URL", debate_cmd.DEFAULT_API_URL)
        assert debate_cmd._is_explicitly_configured_api_url(debate_cmd.DEFAULT_API_URL) is True

    def test_non_default_url_is_explicit(self):
        assert debate_cmd._is_explicitly_configured_api_url("http://example.com:8080") is True

    def test_flag_presence_makes_default_url_explicit(self):
        assert (
            debate_cmd._is_explicitly_configured_api_url(
                debate_cmd.DEFAULT_API_URL, flag_passed=True
            )
            is True
        )

    def test_trailing_slash_still_matches_default(self):
        assert (
            debate_cmd._is_explicitly_configured_api_url(debate_cmd.DEFAULT_API_URL + "/") is False
        )
