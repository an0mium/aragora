from __future__ import annotations

from pathlib import Path

import scripts.generate_capability_matrix as generate_capability_matrix


def test_static_cli_count_matches_runtime_parser() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    assert generate_capability_matrix._count_cli_commands_static(
        repo_root
    ) == generate_capability_matrix._count_cli_commands(repo_root)
