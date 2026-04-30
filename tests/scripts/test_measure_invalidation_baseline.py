from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/measure_invalidation_baseline.py")


def test_measure_invalidation_baseline_dry_run_emits_insufficiency_json(tmp_path: Path) -> None:
    review_queue_root = tmp_path / ".aragora" / "review-queue"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--calibration-db",
            str(tmp_path / "calibration.db"),
            "--review-queue-root",
            str(review_queue_root),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["schema_version"] == "insufficiency_receipt.v1"
    assert "below_min_human_settled" in payload["insufficiency"]["reasons"]
    assert not (tmp_path / ".aragora" / "review-queue" / "thresholds").exists()


def test_measure_invalidation_baseline_write_receipt(tmp_path: Path) -> None:
    receipt_dir = tmp_path / ".aragora" / "review-queue" / "thresholds"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--calibration-db",
            str(tmp_path / "calibration.db"),
            "--review-queue-root",
            str(tmp_path / ".aragora" / "review-queue"),
            "--receipt-dir",
            str(receipt_dir),
            "--write-receipt",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    path = Path(payload["receipt_path"])
    assert path.exists()
    assert path.parent == receipt_dir
    assert json.loads(path.read_text(encoding="utf-8"))["receipt_id"] == payload["receipt_id"]
