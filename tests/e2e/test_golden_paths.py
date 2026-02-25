import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"),
    reason="golden_paths script requires real API keys (ANTHROPIC_API_KEY or OPENAI_API_KEY)",
)
def test_golden_paths_script(tmp_path: Path) -> None:
    output_dir = tmp_path / "golden_paths"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/golden_paths.py",
            "--mode",
            "fast",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )

    assert (output_dir / "ask_result.json").exists()
    assert (output_dir / "gauntlet_result.json").exists()
    assert (output_dir / "review_result.json").exists()
    assert (output_dir / "summary.json").exists()

    ask_payload = json.loads((output_dir / "ask_result.json").read_text())
    assert "final_answer" in ask_payload
    assert "consensus_reached" in ask_payload

    gauntlet_payload = json.loads((output_dir / "gauntlet_result.json").read_text())
    assert "verdict" in gauntlet_payload
    assert "attack_summary" in gauntlet_payload

    review_payload = json.loads((output_dir / "review_result.json").read_text())
    assert "findings" in review_payload
    assert "summary" in review_payload

    summary_payload = json.loads((output_dir / "summary.json").read_text())
    assert summary_payload.get("mode") == "fast"

    # Ensure script emitted a completion line
    assert "golden-paths" in result.stdout.lower()
