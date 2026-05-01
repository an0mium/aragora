"""Tests for ``aragora swarm harness-status`` (Round 30f phase 2)."""

from __future__ import annotations

import argparse
import json

import pytest

from aragora.cli.commands.harness_status import cmd_harness_status
from aragora.swarm.harness_health import (
    get_harness_health_registry,
    reset_harness_health_registry,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_harness_health_registry()
    yield
    reset_harness_health_registry()


def _ns(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


class TestTableRendering:
    def test_renders_known_harnesses_when_idle(self, capsys) -> None:
        cmd_harness_status(_ns(as_json=False))
        out = capsys.readouterr().out
        assert "claude-code" in out
        assert "codex" in out
        # Round 30g: aider intentionally absent; no real harness yet.
        assert "aider" not in out
        assert "Fallback ladders:" in out
        assert "implementation" in out
        assert "review" in out

    def test_shows_pin_reason_after_auth_failure(self, capsys) -> None:
        reg = get_harness_health_registry()
        reg.record_failure("claude-code", reason="api key invalid", status_code=401)
        cmd_harness_status(_ns(as_json=False))
        out = capsys.readouterr().out
        assert "claude-code" in out
        assert "no" in out  # not available
        assert "auth" in out

    def test_shows_chosen_fallback_when_primary_pinned(self, capsys) -> None:
        reg = get_harness_health_registry()
        reg.record_failure("claude-code", reason="api key invalid", status_code=401)
        cmd_harness_status(_ns(as_json=False))
        out = capsys.readouterr().out
        # implementation ladder should now skip claude-code -> codex
        assert "chosen=codex" in out


class TestJsonRendering:
    def test_emits_valid_json(self, capsys) -> None:
        cmd_harness_status(_ns(as_json=True))
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert "harnesses" in payload
        assert "ladders" in payload

    def test_json_shape(self, capsys) -> None:
        reg = get_harness_health_registry()
        reg.record_success("claude-code")
        cmd_harness_status(_ns(as_json=True))
        payload = json.loads(capsys.readouterr().out)
        names = {h["harness"] for h in payload["harnesses"]}
        assert "claude-code" in names
        cc = next(h for h in payload["harnesses"] if h["harness"] == "claude-code")
        assert cc["available"] is True
        assert cc["last_outcome"] == "success"
        ladder_names = {ladder["ladder"] for ladder in payload["ladders"]}
        assert ladder_names == {"implementation", "review"}

    def test_json_pin_propagates(self, capsys) -> None:
        reg = get_harness_health_registry()
        reg.record_failure("codex", reason="rate limit", status_code=429)
        cmd_harness_status(_ns(as_json=True))
        payload = json.loads(capsys.readouterr().out)
        codex_entry = next(h for h in payload["harnesses"] if h["harness"] == "codex")
        assert codex_entry["available"] is False
        assert "quota" in (codex_entry["permanent_pin_reason"] or "")
