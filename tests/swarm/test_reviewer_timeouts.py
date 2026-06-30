"""Reviewer timeout policy for slow Claude reviews and bounded other transports."""

from __future__ import annotations

import subprocess

import aragora.swarm.quorum_evidence as qe


def test_review_ceiling_is_generous_only_for_claude():
    # Claude gets the generous ceiling; unprobed transports keep the old default.
    assert qe._CLAUDE_TIMEOUT >= 600
    assert qe._CODEX_TIMEOUT == 300
    assert qe._REVIEWER_TIMEOUT == 300


def test_probe_timeout_does_not_block_real_review(monkeypatch):
    def _timeout(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="claude", timeout=90)

    monkeypatch.setattr(qe.subprocess, "run", _timeout)
    assert qe._cli_liveness_probe("claude", ["claude", "-p"]) is None


def test_probe_proceeds_when_responsive(monkeypatch):
    monkeypatch.setattr(
        qe.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="OK", stderr=""
        ),
    )
    assert qe._cli_liveness_probe("claude", ["claude", "-p"]) is None


def test_probe_reports_fast_nonzero_exit(monkeypatch):
    monkeypatch.setattr(
        qe.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="login required"
        ),
    )

    err = qe._cli_liveness_probe("claude", ["claude", "-p"])

    assert err == "claude CLI liveness probe exit 1: login required"


def test_probe_proceeds_on_missing_binary(monkeypatch):
    # A missing binary is already a fast failure; let the real call surface it.
    def _missing(*_a, **_k):
        raise FileNotFoundError("claude")

    monkeypatch.setattr(qe.subprocess, "run", _missing)
    assert qe._cli_liveness_probe("claude", ["claude", "-p"]) is None


def test_probe_disabled_via_env(monkeypatch):
    monkeypatch.setenv(qe._CLI_PROBE_TIMEOUT_ENV, "0.0")

    def _boom(*_a, **_k):
        raise AssertionError("probe must not run when disabled")

    monkeypatch.setattr(qe.subprocess, "run", _boom)
    assert qe._cli_liveness_probe("claude", ["claude", "-p"]) is None


def test_run_claude_cli_runs_real_review_after_probe_timeout(monkeypatch):
    calls: list[str] = []

    def _timeout(*_a, **_k):
        calls.append("run")
        raise subprocess.TimeoutExpired(cmd="claude", timeout=90)

    monkeypatch.setattr(qe.subprocess, "run", _timeout)
    result = qe._run_claude_cli("review this diff")
    assert not result.ok
    assert result.error == "claude CLI timed out after 600s"
    assert calls == ["run", "run"]


def test_run_claude_cli_fast_fails_on_probe_nonzero(monkeypatch):
    calls: list[str] = []

    def _nonzero(*_a, **_k):
        calls.append("run")
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="auth required")

    monkeypatch.setattr(qe.subprocess, "run", _nonzero)

    result = qe._run_claude_cli("review this diff")

    assert not result.ok
    assert result.error == "claude CLI liveness probe exit 1: auth required"
    assert calls == ["run"]
