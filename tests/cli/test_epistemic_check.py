"""Tests for the ``aragora epistemic-check`` CLI command (DIC-14 / #6024).

Runs without network access, subprocess calls, pydantic, or real YAML
parsing.  ClaimVerifier is injected as a mock — YAML parsing is covered by
``tests/epistemic/test_claim_verifier.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

for _stub in ["yaml", "pydantic", "pydantic.fields", "pydantic_settings", "pydantic_settings.main"]:
    if _stub not in sys.modules:
        sys.modules[_stub] = MagicMock()

from aragora.cli.commands.epistemic_check import _enabled, cmd_epistemic_check  # noqa: E402
from aragora.epistemic.claim_verifier import ClaimResult, ClaimStatus, ClaimVerifier  # noqa: E402

_PATCH = "aragora.epistemic.claim_verifier.ClaimVerifier"
_real_report_json = ClaimVerifier.report_json  # captured before any patching


def _ns(path=None, *, json_output=False, dry_run=False, execute=False, repo_root=None):
    ns = argparse.Namespace()
    ns.path = path
    ns.json = json_output
    ns.dry_run = dry_run
    ns.execute = execute
    ns.repo_root = repo_root
    return ns


def _result(claim_id="c", status=ClaimStatus.PASS, severity="info"):
    return ClaimResult(
        claim_id=claim_id,
        status=status,
        message="ok",
        severity=severity,
        allowed_action="report_only",
        elapsed_ms=1.0,
    )


def _mock(results):
    v = MagicMock()
    v.verify_manifest.return_value = results
    v.report_json = _real_report_json
    return v


def _run(tmp_path, results, *, json_output=True):
    p = tmp_path / "m.yaml"
    p.write_text("placeholder")
    with (
        patch.dict("os.environ", {"ARAGORA_EPISTEMIC_CLAIMS_ENABLED": "1"}),
        patch(_PATCH, return_value=_mock(results)),
    ):
        return cmd_epistemic_check(_ns(str(p), json_output=json_output))


# ---------------------------------------------------------------------------
# Flag-gating
# ---------------------------------------------------------------------------


def test_disabled_exits_zero_with_stderr(tmp_path, capsys):
    with patch.dict("os.environ", {}, clear=True):
        rc = cmd_epistemic_check(_ns(str(tmp_path)))
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ARAGORA_EPISTEMIC_CLAIMS_ENABLED" in captured.err


def test_enabled_values():
    for v in ("1", "true", "yes", "on", "TRUE"):
        with patch.dict("os.environ", {"ARAGORA_EPISTEMIC_CLAIMS_ENABLED": v}):
            assert _enabled() is True


def test_disabled_values():
    for v in ("0", "false", "no", "off", ""):
        with patch.dict("os.environ", {"ARAGORA_EPISTEMIC_CLAIMS_ENABLED": v}):
            assert _enabled() is False


# ---------------------------------------------------------------------------
# JSON output shape
# ---------------------------------------------------------------------------


def test_json_schema_keys(tmp_path, capsys):
    rc = _run(tmp_path, [_result()])
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert "results" in payload and "summary" in payload
    assert rc == 0


def test_json_result_status_and_id(tmp_path, capsys):
    _run(tmp_path, [_result("my.claim", ClaimStatus.STALE)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0] == {
        "claim_id": "my.claim",
        "status": "stale",
        "message": "ok",
        "severity": "info",
        "allowed_action": "report_only",
        "elapsed_ms": 1.0,
        "detail": {},
    }


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------


def test_blocking_fail_exits_one(tmp_path):
    assert _run(tmp_path, [_result("x", ClaimStatus.FAIL, "blocking")]) == 1


def test_blocking_error_exits_one(tmp_path):
    assert _run(tmp_path, [_result("x", ClaimStatus.ERROR, "blocking")]) == 1


def test_non_blocking_fail_exits_zero(tmp_path):
    assert _run(tmp_path, [_result("x", ClaimStatus.FAIL, "warning")]) == 0


def test_missing_path_exits_one(tmp_path, capsys):
    with patch.dict("os.environ", {"ARAGORA_EPISTEMIC_CLAIMS_ENABLED": "1"}):
        rc = cmd_epistemic_check(_ns(str(tmp_path / "absent.yaml")))
    assert rc == 1


# ---------------------------------------------------------------------------
# Directory scanning and dry_run forwarding
# ---------------------------------------------------------------------------


def test_empty_dir_exits_zero(tmp_path):
    with patch.dict("os.environ", {"ARAGORA_EPISTEMIC_CLAIMS_ENABLED": "1"}):
        assert cmd_epistemic_check(_ns(str(tmp_path))) == 0


def _capture_verifier_kwargs(tmp_path, ns):
    """Run the command with ClaimVerifier mocked and return its ctor kwargs."""
    p = tmp_path / "m.yaml"
    p.write_text("placeholder")
    ns.path = str(p)
    captured_kwargs: dict = {}

    def _ctor(**kw):
        captured_kwargs.update(kw)
        return _mock([_result()])

    with (
        patch.dict("os.environ", {"ARAGORA_EPISTEMIC_CLAIMS_ENABLED": "1"}),
        patch(_PATCH, _ctor),
    ):
        cmd_epistemic_check(ns)
    return captured_kwargs


def test_dry_run_forwarded(tmp_path):
    kwargs = _capture_verifier_kwargs(tmp_path, _ns(dry_run=True, json_output=True))
    assert kwargs.get("dry_run") is True


def test_default_invocation_is_read_only(tmp_path):
    """Read-only by default: without --execute the verifier runs dry-run, so
    manifest-provided commands are never executed even for untrusted paths."""
    kwargs = _capture_verifier_kwargs(tmp_path, _ns(json_output=True))
    assert kwargs.get("dry_run") is True


def test_execute_flag_enables_command_execution(tmp_path):
    """--execute opts in to running commands (dry_run forwarded as False)."""
    kwargs = _capture_verifier_kwargs(tmp_path, _ns(execute=True, json_output=True))
    assert kwargs.get("dry_run") is False


def test_dry_run_overrides_execute(tmp_path):
    """If both flags are passed, --dry-run wins (fail safe, no execution)."""
    kwargs = _capture_verifier_kwargs(tmp_path, _ns(execute=True, dry_run=True, json_output=True))
    assert kwargs.get("dry_run") is True


def test_default_does_not_run_command_kind_claims(tmp_path):
    """End-to-end: a manifest whose command would mutate state is NOT run by
    default. We use a real ClaimVerifier with an injected command runner and
    confirm the runner is never invoked unless --execute is set."""
    import yaml as _yaml

    if isinstance(_yaml, MagicMock) or isinstance(getattr(_yaml, "safe_load", None), MagicMock):
        pytest.skip("real PyYAML required to parse the manifest in this end-to-end test")

    from aragora.epistemic.claim_verifier import ClaimVerifier

    manifest = tmp_path / "danger.yaml"
    manifest.write_text(
        "claims:\n"
        "  - claim_id: danger.command\n"
        "    verification:\n"
        "      kind: command\n"
        "      command: echo pwned\n"
    )

    runner_calls: list = []

    def _spy_runner(args):
        runner_calls.append(args)
        return 0, "", ""

    real_ctor = ClaimVerifier

    def _ctor(**kw):
        kw.pop("command_runner", None)
        return real_ctor(command_runner=_spy_runner, **kw)

    # Default (no --execute): runner must never be called.
    with (
        patch.dict("os.environ", {"ARAGORA_EPISTEMIC_CLAIMS_ENABLED": "1"}),
        patch(_PATCH, _ctor),
    ):
        cmd_epistemic_check(_ns(str(manifest), json_output=True))
    assert runner_calls == []

    # With --execute: the command runs.
    with (
        patch.dict("os.environ", {"ARAGORA_EPISTEMIC_CLAIMS_ENABLED": "1"}),
        patch(_PATCH, _ctor),
    ):
        cmd_epistemic_check(_ns(str(manifest), execute=True, json_output=True))
    assert runner_calls == [["echo", "pwned"]]
