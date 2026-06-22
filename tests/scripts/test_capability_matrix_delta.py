from __future__ import annotations

from pathlib import Path

import scripts.capability_matrix_delta as capability_matrix_delta


HEAD_MATRIX = """# Capability Matrix

| Surface | Coverage |
|---|---:|
| **HTTP API** | 10 paths / 20 operations |
| **CLI** | 3 commands |
| **SDK (Python)** | 4 namespaces |
| **SDK (TypeScript)** | 5 namespaces |
| **Capability Catalog** | 6/7 mapped |
"""

BASE_MATRIX = """# Capability Matrix

| Surface | Coverage |
|---|---:|
| **HTTP API** | 8 paths / 17 operations |
| **CLI** | 2 commands |
| **SDK (Python)** | 4 namespaces |
| **SDK (TypeScript)** | 3 namespaces |
| **Capability Catalog** | 5/7 mapped |
"""


def _write_head_matrix(repo_root: Path) -> None:
    matrix_path = repo_root / "docs" / "CAPABILITY_MATRIX.md"
    matrix_path.parent.mkdir(parents=True)
    matrix_path.write_text(HEAD_MATRIX, encoding="utf-8")


def test_main_without_base_ref_writes_head_snapshot(tmp_path: Path) -> None:
    _write_head_matrix(tmp_path)
    out_path = tmp_path / "summary.md"

    rc = capability_matrix_delta.main(["--root", str(tmp_path), "--out", str(out_path)])

    assert rc == 0
    body = out_path.read_text(encoding="utf-8")
    assert "| HTTP paths | n/a | 10 | n/a |" in body
    assert "| Total capabilities | n/a | 7 | n/a |" in body
    assert "Coverage: 85.7%" in body


def test_main_with_base_ref_writes_measured_deltas(tmp_path: Path, monkeypatch) -> None:
    _write_head_matrix(tmp_path)
    out_path = tmp_path / "summary.md"

    def fake_load_base_text(repo_root: Path, base_ref: str) -> str:
        assert repo_root == tmp_path.resolve()
        assert base_ref == "origin/main"
        return BASE_MATRIX

    monkeypatch.setattr(capability_matrix_delta, "_load_base_text", fake_load_base_text)

    rc = capability_matrix_delta.main(
        [
            "--root",
            str(tmp_path),
            "--base-ref",
            "origin/main",
            "--out",
            str(out_path),
        ]
    )

    assert rc == 0
    body = out_path.read_text(encoding="utf-8")
    assert "Base ref: `origin/main`" in body
    assert "| HTTP paths | 8 | 10 | +2 |" in body
    assert "| HTTP operations | 17 | 20 | +3 |" in body
    assert "| TypeScript SDK namespaces | 3 | 5 | +2 |" in body
    assert "Coverage: 71.4% -> 85.7% (+14.3pp)" in body


def test_main_with_unavailable_base_ref_fails_closed(tmp_path: Path, monkeypatch, capsys) -> None:
    _write_head_matrix(tmp_path)
    out_path = tmp_path / "summary.md"

    def fake_load_base_text(repo_root: Path, base_ref: str) -> str:
        raise capability_matrix_delta.BaseMatrixUnavailableError(
            "Base matrix unavailable for ref 'missing': fatal: bad revision"
        )

    monkeypatch.setattr(capability_matrix_delta, "_load_base_text", fake_load_base_text)

    rc = capability_matrix_delta.main(
        [
            "--root",
            str(tmp_path),
            "--base-ref",
            "missing",
            "--out",
            str(out_path),
        ]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert not out_path.exists()
    assert "Base matrix unavailable for ref 'missing'" in captured.err
