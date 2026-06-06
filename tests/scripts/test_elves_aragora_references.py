from __future__ import annotations

from pathlib import Path


def test_tier4_reference_uses_live_settle_tier4_modes() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    references_root = repo_root / ".agents/skills/elves-aragora/references"
    reference_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(references_root.glob("*.md"))
    )

    assert "settle_tier4_pr.py --check/--apply" not in reference_text
    assert "scripts/settle_tier4_pr.py --check" in reference_text
    assert "--settle-only" in reference_text
    assert "--merge-apply" in reference_text
