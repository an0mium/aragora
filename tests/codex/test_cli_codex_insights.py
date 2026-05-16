"""CLI surface tests for ``aragora codex insights {summary,anomalies,crossref,digest}``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from aragora.cli.commands import codex_insights as cli


def _args(**kwargs) -> argparse.Namespace:  # type: ignore[no-untyped-def]
    base = {
        "codex_home": None,
        "json": False,
        "since": "4h",
        "include_archived": False,
    }
    base.update(kwargs)
    return argparse.Namespace(**base)


# -- summary ------------------------------------------------------------------


def test_cli_summary_text(fake_codex_home, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    rc = cli.cmd_codex_insights_summary(_args())
    out = capsys.readouterr().out
    assert rc == 0
    assert "Window:" in out
    assert "Tokens:" in out
    assert "Tool calls" in out


def test_cli_summary_json(fake_codex_home, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    rc = cli.cmd_codex_insights_summary(_args(json=True))
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert "patterns" in payload
    assert payload["thread_count"] >= 1


def test_cli_summary_bad_since(fake_codex_home, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    rc = cli.cmd_codex_insights_summary(_args(since="nonsense"))
    assert rc == 2
    assert "invalid duration" in capsys.readouterr().err


# -- anomalies ----------------------------------------------------------------


def test_cli_anomalies_text(fake_codex_home, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    rc = cli.cmd_codex_insights_anomalies(
        _args(token_cap=100_000, runaway_tool_calls=200, stuck_turn_minutes=5)
    )
    out = capsys.readouterr().out
    assert rc == 0
    # No anomalies expected in the small synthetic fixture, but exit code should be 0.
    assert "anomalies" in out or "(no anomalies" in out


def test_cli_anomalies_json(fake_codex_home, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    rc = cli.cmd_codex_insights_anomalies(
        _args(json=True, token_cap=100_000, runaway_tool_calls=200, stuck_turn_minutes=5)
    )
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert isinstance(payload, list)


# -- crossref -----------------------------------------------------------------


def test_cli_crossref_text(fake_codex_home, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    rc = cli.cmd_codex_insights_crossref(_args())
    out = capsys.readouterr().out
    assert rc == 0
    assert "Cross-references" in out


def test_cli_crossref_json(fake_codex_home, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    rc = cli.cmd_codex_insights_crossref(_args(json=True))
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert isinstance(payload, list)


# -- digest -------------------------------------------------------------------


def test_cli_digest_to_stdout(fake_codex_home, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    rc = cli.cmd_codex_insights_digest(
        _args(json=True, emit_receipt=False, receipt_dir=None, ingest_km=False)
    )
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert "sha256" in payload
    assert "patterns" in payload


def test_cli_digest_emit_receipt_writes_file(
    fake_codex_home,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    receipt_dir = tmp_path / "receipts"
    rc = cli.cmd_codex_insights_digest(
        _args(
            json=False,
            emit_receipt=True,
            receipt_dir=str(receipt_dir),
            ingest_km=False,
        )
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "wrote" in out
    files = list(receipt_dir.glob("digest-*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["schema_version"]
    assert payload["sha256"]


def test_cli_digest_emit_receipt_json_is_machine_readable(
    fake_codex_home,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    receipt_dir = tmp_path / "receipts"
    rc = cli.cmd_codex_insights_digest(
        _args(
            json=True,
            emit_receipt=True,
            receipt_dir=str(receipt_dir),
            ingest_km=False,
        )
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["schema_version"]
    assert payload["sha256"]
    assert payload["receipt_path"].startswith(str(receipt_dir))
    assert "signing_note" in payload
    assert list(receipt_dir.glob("digest-*.json"))
