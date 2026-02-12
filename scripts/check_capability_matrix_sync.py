#!/usr/bin/env python3
"""Fail if generated capability matrix files are out of date.

Checks:
- docs/CAPABILITY_MATRIX.md
- docs-site/docs/contributing/capability-matrix.md
"""

from __future__ import annotations

import difflib
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_SCRIPT = REPO_ROOT / "scripts" / "generate_capability_matrix.py"

TARGETS = [
    REPO_ROOT / "docs" / "CAPABILITY_MATRIX.md",
    REPO_ROOT / "docs-site" / "docs" / "contributing" / "capability-matrix.md",
]


def _diff(a: Path, b: Path, *, label_a: str, label_b: str) -> str:
    a_text = a.read_text(encoding="utf-8").splitlines(keepends=True)
    b_text = b.read_text(encoding="utf-8").splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            a_text,
            b_text,
            fromfile=label_a,
            tofile=label_b,
        )
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="aragora-capability-matrix-") as tmp:
        tmp_path = Path(tmp)
        diffs: list[str] = []

        for target in TARGETS:
            if not target.exists():
                print(f"ERROR: Missing target file: {target}", file=sys.stderr)
                return 1

            generated_tmp = tmp_path / target.name
            subprocess.run(
                [
                    sys.executable,
                    str(GEN_SCRIPT),
                    "--root",
                    str(REPO_ROOT),
                    "--out",
                    str(generated_tmp),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            rel_label = str(target.relative_to(REPO_ROOT))
            diff = _diff(
                target,
                generated_tmp,
                label_a=rel_label,
                label_b=f"{rel_label} (generated)",
            )
            if diff:
                diffs.append(diff)

        if diffs:
            print("Capability matrix files are out of date.", file=sys.stderr)
            print("Run: python scripts/generate_capability_matrix.py", file=sys.stderr)
            print(
                "Also run: python scripts/generate_capability_matrix.py --out docs-site/docs/contributing/capability-matrix.md",
                file=sys.stderr,
            )
            print("", file=sys.stderr)
            for d in diffs:
                print(d, file=sys.stderr)
            return 1

    print("Capability matrix files are up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
