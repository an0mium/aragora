"""Tests for the triage CLI command wiring."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from aragora.cli.commands import triage as triage_cmd
from aragora.inbox.trust_wedge import InboxWedgeAction, TriageDecision


def test_add_triage_parser_supports_auth_and_dry_run():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    triage_cmd.add_triage_parser(subparsers)

    run_args = parser.parse_args(["triage", "run", "--batch", "3", "--dry-run"])
    auth_args = parser.parse_args(["triage", "auth"])

    assert run_args.triage_command == "run"
    assert run_args.batch == 3
    assert run_args.dry_run is True
    assert auth_args.triage_command == "auth"


@pytest.mark.asyncio
async def test_run_triage_uses_receipt_review_loop():
    decision = TriageDecision.create(
        final_action="ignore",
        confidence=0.4,
        dissent_summary="",
        receipt_id="receipt-1",
    )
    fake_runner = SimpleNamespace(run_triage=AsyncMock(return_value=[decision]))
    fake_service = SimpleNamespace(review_receipt=object())
    captured: dict[str, object] = {}

    class _FakeLoop:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def review_batch(self, decisions):
            captured["decisions"] = decisions
            return []

    with (
        patch.object(triage_cmd, "_get_gmail_connector", return_value=object()),
        patch(
            "aragora.inbox.triage_runner.InboxTriageRunner",
            return_value=fake_runner,
        ),
        patch(
            "aragora.inbox.trust_wedge.get_inbox_trust_wedge_service",
            return_value=fake_service,
        ),
        patch(
            "aragora.inbox.cli_review.CLIReviewLoop",
            _FakeLoop,
        ),
    ):
        await triage_cmd._run_triage(batch_size=1, auto_approve=False)

    assert captured["review_fn"] is fake_service.review_receipt
    assert captured["decisions"] == [decision]


@pytest.mark.asyncio
async def test_run_triage_dry_run_disables_action_execution(capsys):
    decision = TriageDecision.create(
        final_action="archive",
        confidence=0.92,
        dissent_summary="",
        receipt_id="receipt-2",
    )
    fake_runner = SimpleNamespace(run_triage=AsyncMock(return_value=[decision]))
    fake_service = SimpleNamespace(review_receipt=object())

    with (
        patch.object(triage_cmd, "_get_gmail_connector", return_value=object()),
        patch(
            "aragora.inbox.triage_runner.InboxTriageRunner",
            return_value=fake_runner,
        ),
        patch(
            "aragora.inbox.trust_wedge.get_inbox_trust_wedge_service",
            return_value=fake_service,
        ),
    ):
        await triage_cmd._run_triage(batch_size=2, auto_approve=True, dry_run=True)

    fake_runner.run_triage.assert_awaited_once_with(batch_size=2, auto_approve=False)
    out = capsys.readouterr().out
    assert "[DRY RUN] Fetching up to 2 unread messages" in out
    assert "[DRY RUN] Proposed triage decisions" in out
    assert "archive" in out


def test_print_decisions_formats_enum_values(capsys):
    decision = TriageDecision.create(
        final_action=InboxWedgeAction.IGNORE,
        confidence=0.4,
        dissent_summary="",
        receipt_id="receipt-1",
    )
    decision.intent = SimpleNamespace(_subject="Subject line")

    triage_cmd._print_decisions([decision])

    out = capsys.readouterr().out
    assert "ignore" in out
    assert "InboxWedgeAction" not in out


def test_get_gmail_connector_loads_refresh_token_from_home_file(tmp_path, monkeypatch):
    class _FakeConnector:
        def __init__(self):
            self._refresh_token = None

    token_dir = Path(tmp_path) / ".aragora"
    token_dir.mkdir()
    (token_dir / "gmail_refresh_token").write_text("refresh-from-file\n")

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GMAIL_CLIENT_ID", "client-id")
    monkeypatch.delenv("GMAIL_REFRESH_TOKEN", raising=False)

    with patch(
        "aragora.connectors.enterprise.communication.gmail.GmailConnector",
        _FakeConnector,
    ):
        connector = triage_cmd._get_gmail_connector()

    assert connector is not None
    assert connector._refresh_token == "refresh-from-file"


def test_show_status_reports_refresh_token(tmp_path, monkeypatch, capsys):
    token_dir = Path(tmp_path) / ".aragora"
    token_dir.mkdir()
    (token_dir / "gmail_refresh_token").write_text("refresh-from-file\n")
    (token_dir / "signing.key").write_text("dummy-signing-key")

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GMAIL_CLIENT_ID", "client-id")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    triage_cmd._show_status()

    out = capsys.readouterr().out
    assert "Gmail configured:     yes" in out
    assert "Durable signing key:  yes" in out
    assert "Gmail refresh token:  yes" in out
    assert "OpenRouter fallback:  yes" in out


@pytest.mark.asyncio
async def test_run_gmail_auth_saves_refresh_token(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GMAIL_CLIENT_ID", "client-id")
    monkeypatch.setenv("GMAIL_CLIENT_SECRET", "client-secret")

    opened_urls: list[str] = []

    class _FakeConnector:
        def __init__(self):
            self._refresh_token = "refresh-token-123"

        def get_oauth_url(self, redirect_uri: str) -> str:
            assert redirect_uri == "http://localhost:8089/callback"
            return "https://accounts.google.test/oauth"

        async def authenticate(self, *, code: str, redirect_uri: str) -> bool:
            assert code == "oauth-code"
            assert redirect_uri == "http://localhost:8089/callback"
            return True

    class _FakeHTTPServer:
        def __init__(self, server_address, handler_cls):
            self.server_address = server_address
            self.handler_cls = handler_cls

        def handle_request(self):
            handler = object.__new__(self.handler_cls)
            handler.path = "/callback?code=oauth-code"
            handler.wfile = io.BytesIO()
            handler.send_response = lambda _code: None
            handler.send_header = lambda *_args, **_kwargs: None
            handler.end_headers = lambda: None
            self.handler_cls.do_GET(handler)

        def server_close(self):
            return None

    with (
        patch(
            "aragora.connectors.enterprise.communication.gmail.GmailConnector",
            _FakeConnector,
        ),
        patch("webbrowser.open", side_effect=opened_urls.append),
        patch("http.server.HTTPServer", _FakeHTTPServer),
    ):
        await triage_cmd._run_gmail_auth()

    token_path = Path(tmp_path) / ".aragora" / "gmail_refresh_token"
    assert token_path.read_text() == "refresh-token-123"
    assert token_path.stat().st_mode & 0o777 == 0o600
    assert opened_urls == ["https://accounts.google.test/oauth"]

    out = capsys.readouterr().out
    assert "Opening browser for Gmail authorization..." in out
    assert "Gmail authenticated successfully!" in out
    assert str(token_path) in out
