"""Regression coverage for scripts/swarm_host_preflight.sh."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "swarm_host_preflight.sh"


def _run_provider_check(
    *, worker_model: str, review_model: str, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-lc", f'source "{SCRIPT}"; require_provider_keys'],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "WORKER_MODEL": worker_model,
            "REVIEW_MODEL": review_model,
            **env,
        },
    )


def _capture_worker_preflight_command(*, worker_model: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-lc",
            'source "$SCRIPT_PATH"; WORKER_MODEL="$WORKER_MODEL_UNDER_TEST"; python3() { printf "%s\\n" "$*"; }; run_worker_preflight',
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "SCRIPT_PATH": str(SCRIPT),
            "WORKER_MODEL_UNDER_TEST": worker_model,
        },
    )


def test_provider_check_requires_both_keys_for_mixed_claude_codex_models() -> None:
    result = _run_provider_check(
        worker_model="claude",
        review_model="codex",
        env={"ANTHROPIC_API_KEY": "anthropic-test-key", "OPENAI_API_KEY": ""},
    )

    assert result.returncode == 1
    assert "anthropic: ok" in result.stdout
    assert "openai: NOT CONFIGURED" in result.stderr


def test_provider_check_passes_when_both_mixed_model_keys_are_present() -> None:
    result = _run_provider_check(
        worker_model="claude",
        review_model="codex",
        env={
            "ANTHROPIC_API_KEY": "anthropic-test-key",
            "OPENAI_API_KEY": "openai-test-key",
        },
    )

    assert result.returncode == 0
    assert "anthropic: ok" in result.stdout
    assert "openai: ok" in result.stdout


def test_worker_preflight_uses_swarm_cli_entrypoint() -> None:
    result = _capture_worker_preflight_command(worker_model="codex")

    assert result.returncode == 0
    assert result.stdout.strip() == "-m aragora.cli.main swarm preflight --worker-model codex"
