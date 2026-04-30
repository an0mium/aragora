from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_heterogeneity_probe.py"


def test_probe_script_dry_run_selects_pilot_subset() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["prompt_count"] == 21
    assert payload["class_counts"]["null_negative"] == 2


def test_probe_script_writes_synthetic_fixture_receipt(tmp_path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--synthetic-fixture",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "fixture",
            "--json",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(proc.stdout)
    receipt_path = Path(payload["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == "heterogeneity_probe_receipt.v1"
    assert receipt["judge_model"] == "synthetic-fixture"
    assert "synthetic fixture only" in receipt["scope_caveats"][0]


def test_probe_script_writes_receipt_from_classifications_file(tmp_path) -> None:
    fixture = ROOT / "tests" / "heterogeneity" / "fixtures" / "classifications_minimal.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--limit",
            "1",
            "--classifications-file",
            str(fixture),
            "--output-root",
            str(tmp_path),
            "--run-id",
            "external",
            "--json",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(proc.stdout)
    receipt = json.loads(Path(payload["receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["judge_model"] == "external-judge-fixture"
    assert receipt["verdict"] == "insufficient_pilot"
    assert "external judged classifications" in receipt["scope_caveats"][0]
