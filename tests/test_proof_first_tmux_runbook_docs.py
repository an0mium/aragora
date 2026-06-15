from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = REPO_ROOT / "docs" / "runbooks" / "RUNBOOK_PROOF_FIRST_TMUX_OPERATOR.md"
REFRESH_SCRIPT = REPO_ROOT / "scripts" / "refresh_proof_surfaces.sh"


def test_proof_first_tmux_runbook_routes_surface_checks_through_wrapper() -> None:
    text = RUNBOOK.read_text()
    help_result = subprocess.run(
        ["bash", str(REFRESH_SCRIPT), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert help_result.returncode == 0
    assert "--surface {all,b0,tw03}" in help_result.stdout
    assert "--check" in help_result.stdout
    assert "bash scripts/refresh_proof_surfaces.sh --check" in text
    assert "bash scripts/refresh_proof_surfaces.sh --surface b0 --check" in text
    assert "bash scripts/refresh_proof_surfaces.sh --surface tw03 --check" in text


def test_proof_first_tmux_runbook_assigns_wrapper_to_benchmark_lane() -> None:
    text = RUNBOOK.read_text()
    ownership = text.split("Recommended ownership:", 1)[1].split("## Setup", 1)[0]

    assert "`benchmark-proof`" in ownership
    assert "`scripts/refresh_proof_surfaces.sh`" in ownership
