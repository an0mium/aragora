"""Two-tier reviewer timeouts: a generous ceiling for genuinely slow reviews,
plus a short liveness probe that fails fast when no valid response is coming
(wedged/contended/unauthed CLI)."""

from __future__ import annotations

import subprocess

import aragora.swarm.quorum_evidence as qe


def test_review_ceiling_is_generous():
    # A real review of a large diff can take minutes; the default ceiling is 10m.
    assert qe._CLAUDE_TIMEOUT >= 600
    assert qe._CODEX_TIMEOUT >= 600
    assert qe._REVIEWER_TIMEOUT >= 600


def test_probe_reports_unresponsive_on_timeout(monkeypatch):
    def _timeout(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="claude", timeout=90)

    monkeypatch.setattr(qe.subprocess, "run", _timeout)
    err = qe._cli_liveness_probe("claude", ["claude", "-p"])
    assert err is not None
    assert "liveness probe" in err


def test_probe_proceeds_when_responsive(monkeypatch):
    monkeypatch.setattr(
        qe.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="OK", stderr=""
        ),
    )
    assert qe._cli_liveness_probe("claude", ["claude", "-p"]) is None


def test_probe_proceeds_on_missing_binary(monkeypatch):
    # A missing binary is already a fast failure; let the real call surface it.
    def _missing(*_a, **_k):
        raise FileNotFoundError("claude")

    monkeypatch.setattr(qe.subprocess, "run", _missing)
    assert qe._cli_liveness_probe("claude", ["claude", "-p"]) is None


def test_probe_disabled_via_env(monkeypatch):
    monkeypatch.setenv(qe._CLI_PROBE_TIMEOUT_ENV, "0")

    def _boom(*_a, **_k):
        raise AssertionError("probe must not run when disabled")

    monkeypatch.setattr(qe.subprocess, "run", _boom)
    assert qe._cli_liveness_probe("claude", ["claude", "-p"]) is None


def test_run_claude_cli_fast_fails_when_probe_times_out(monkeypatch):
    # subprocess.run always times out → the probe catches it and the real review
    # is never attempted; the error names the probe, not the full review ceiling.
    def _timeout(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="claude", timeout=90)

    monkeypatch.setattr(qe.subprocess, "run", _timeout)
    result = qe._run_claude_cli("review this diff")
    assert not result.ok
    assert "liveness probe" in result.error
