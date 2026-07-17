from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNBOOK = REPO_ROOT / "docs" / "runbooks" / "RUNBOOK_MAIN_HEAD_HEALTH.md"


def test_generate_validate_proxy_uses_one_unique_spec_path() -> None:
    row = next(
        line
        for line in RUNBOOK.read_text(encoding="utf-8").splitlines()
        if line.startswith("| `Generate & Validate` |")
    )

    assert "/tmp/openapi_ci_required.json" not in row
    assert row.count("mktemp -d") == 1
    assert row.count('SPEC="$SPEC_DIR/openapi.json"') == 1
    assert row.count('--output "$SPEC"') == 1
    assert row.count('--spec "$SPEC"') == 4
    assert row.count('--extra-spec "$SPEC"') == 1
