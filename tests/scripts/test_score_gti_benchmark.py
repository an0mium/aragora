import json
import subprocess
import sys
from pathlib import Path


def test_cli_writes_scorecard_and_status(tmp_path):
    corpus = Path("docs/status/generated/gti/scenarios.json")
    scorecard = tmp_path / "scorecard.json"
    status = tmp_path / "GTI_STATUS.md"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/score_gti_benchmark.py",
            "--corpus",
            str(corpus),
            "--scorecard-out",
            str(scorecard),
            "--status-out",
            str(status),
            "--now",
            "2026-06-06T12:00:00+00:00",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(scorecard.read_text())
    assert data["benchmark"] == "gti-ground-truth-integrity-v1"
    assert data["generated_at"] == "2026-06-06T12:00:00+00:00"
    assert data["naive"]["stale_belief_action_rate"] > data["gated"]["stale_belief_action_rate"]
    text = status.read_text()
    assert "Last updated: 2026-06-06T12:00:00+00:00" in text
    assert "stale_belief_action_rate" in text
