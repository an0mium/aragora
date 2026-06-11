from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import scripts.capability_matrix_delta as capability_matrix_delta


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "capability_matrix_delta.py"


def _matrix_text(*, mapped: int = 3, total: int = 5) -> str:
    return "\n".join(
        [
            "| Surface | Coverage |",
            "|---|---|",
            "| **HTTP API** | 10 paths / 12 operations |",
            "| **CLI** | 4 commands |",
            "| **SDK (Python)** | 2 namespaces |",
            "| **SDK (TypeScript)** | 2 namespaces |",
            f"| **Capability Catalog** | {mapped}/{total} mapped |",
            "",
        ]
    )


def test_parse_matrix_rejects_catalog_mapped_count_above_total() -> None:
    try:
        capability_matrix_delta._parse_matrix(_matrix_text(mapped=9, total=3))
    except capability_matrix_delta.MatrixParseError as exc:
        assert "mapped count 9 exceeds total count 3" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("impossible catalog coverage should fail closed")


def test_parse_matrix_rejects_zero_catalog_total() -> None:
    try:
        capability_matrix_delta._parse_matrix(_matrix_text(mapped=0, total=0))
    except capability_matrix_delta.MatrixParseError as exc:
        assert "total must be greater than zero" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("zero catalog total should fail closed")


def test_cli_rejects_invalid_current_matrix_without_writing_output(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "CAPABILITY_MATRIX.md").write_text(
        _matrix_text(mapped=9, total=3),
        encoding="utf-8",
    )
    output = tmp_path / "summary.md"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--root",
            str(tmp_path),
            "--out",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "mapped count 9 exceeds total count 3" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert not output.exists()
