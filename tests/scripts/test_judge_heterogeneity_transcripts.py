from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "judge_heterogeneity_transcripts.py"
RECEIPT_SCRIPT = ROOT / "scripts" / "run_heterogeneity_probe.py"


def _write_transcript(path: Path) -> None:
    rows = [
        {
            "type": "round",
            "round_id": "fixture-cn_01_invalidation_signals",
            "metadata": {"prompt_id": "cn_01_invalidation_signals", "class": "clean_neutral"},
            "prompt": "fixture prompt",
        },
        {
            "type": "turn",
            "agent": "solo",
            "returncode": 0,
            "timed_out": False,
            "error": None,
            "stdout": "NO_CONCRETE_ERROR: I checked the facts.",
            "stderr": "",
        },
        {
            "type": "turn",
            "agent": "failed",
            "returncode": 124,
            "timed_out": True,
            "error": "timeout",
            "stdout": "",
            "stderr": "timeout",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_fake_judge(path: Path) -> None:
    path.write_text(
        """
from __future__ import annotations

import json
import sys

prompt = sys.argv[-1]
assert "No seeded error." in prompt
print("```json")
print(json.dumps({"results": [{"agent": "solo", "verdict": "missed", "rationale": "fixture judged"}]}))
print("```")
""".lstrip(),
        encoding="utf-8",
    )


def _write_fake_no_seed_flagged_correctly_judge(path: Path) -> None:
    path.write_text(
        """
from __future__ import annotations

import json
import sys

prompt = sys.argv[-1]
assert "No seeded error." in prompt
print(json.dumps({"results": [{"agent": "solo", "verdict": "flagged_correctly", "rationale": "correctly found no concrete error"}]}))
""".lstrip(),
        encoding="utf-8",
    )


def test_judge_bridge_writes_classifications_and_receipt_input(tmp_path: Path) -> None:
    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir()
    _write_transcript(transcript_dir / "dialog-fixture-cn_01_invalidation_signals.jsonl")
    fake_judge = tmp_path / "fake_judge.py"
    _write_fake_judge(fake_judge)
    output = tmp_path / "classifications.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--transcript-dir",
            str(transcript_dir),
            "--limit",
            "1",
            "--output",
            str(output),
            "--judge-command",
            f"{sys.executable} {fake_judge}",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    summary = json.loads(proc.stdout)
    assert summary["prompt_count"] == 1
    assert summary["classification_count"] == 2

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["panel_models"] == ["solo", "failed"]
    classifications = payload["results"][0]["classifications"]
    assert classifications == [
        {"agent": "solo", "rationale": "fixture judged", "verdict": "missed"},
        {"agent": "failed", "rationale": "timeout", "verdict": "dispatch_failed"},
    ]

    receipt_proc = subprocess.run(
        [
            sys.executable,
            str(RECEIPT_SCRIPT),
            "--limit",
            "1",
            "--classifications-file",
            str(output),
            "--output-root",
            str(tmp_path / "receipts"),
            "--run-id",
            "fixture",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    receipt_summary = json.loads(receipt_proc.stdout)
    assert receipt_summary["receipt_verdict"] == "insufficient_pilot"


def test_judge_bridge_normalizes_no_seeded_flagged_correctly(tmp_path: Path) -> None:
    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir()
    _write_transcript(transcript_dir / "dialog-fixture-cn_01_invalidation_signals.jsonl")
    fake_judge = tmp_path / "fake_no_seed_judge.py"
    _write_fake_no_seed_flagged_correctly_judge(fake_judge)
    output = tmp_path / "classifications.json"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--transcript-dir",
            str(transcript_dir),
            "--limit",
            "1",
            "--output",
            str(output),
            "--judge-command",
            f"{sys.executable} {fake_judge}",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    classifications = payload["results"][0]["classifications"]
    assert classifications[0]["agent"] == "solo"
    assert classifications[0]["verdict"] == "missed"
    assert classifications[0]["rationale"].startswith(
        "normalized no-seeded-error flagged_correctly to missed:"
    )


def test_judge_bridge_rejects_missing_transcript(tmp_path: Path) -> None:
    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir()
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--transcript-dir",
            str(transcript_dir),
            "--limit",
            "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "expected exactly one transcript" in proc.stderr
