from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_epic_debate_receipt.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("generate_epic_debate_receipt", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_debate_transcript_rejects_blank_input(tmp_path: Path) -> None:
    module = _load_module()
    transcript = tmp_path / "debate_transcript.txt"
    transcript.write_text("\n\t\n", encoding="utf-8")

    try:
        module.parse_debate_transcript(transcript)
    except module.TranscriptParseError as exc:
        assert "Debate transcript is empty" in str(exc)
        assert str(transcript) in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("blank transcript should fail closed")


def test_main_rejects_blank_transcript_without_writing_receipts(tmp_path: Path) -> None:
    debate_dir = tmp_path / ".nomic" / "epic_strategic_debate"
    debate_dir.mkdir(parents=True)
    (debate_dir / "debate_transcript.txt").write_text("\n\t\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "Debate transcript is empty" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert not (debate_dir / "receipts").exists()
