from __future__ import annotations

import sys
from pathlib import Path


def test_aragora_debate_src_is_bootstrapped_for_repo_tests() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    expected = str(repo_root / "aragora-debate" / "src")
    assert expected in sys.path
