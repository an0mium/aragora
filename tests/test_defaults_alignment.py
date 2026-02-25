from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from aragora.config import DEFAULT_AGENTS, DEFAULT_CONSENSUS, DEFAULT_ROUNDS, MAX_ROUNDS


def _extract_default(pattern: str, text: str, name: str) -> str:
    match = re.search(pattern, text)
    assert match, f"Missing {name} in frontend config"
    return match.group(1)


def test_frontend_defaults_match_server_defaults() -> None:
    # Skip if server defaults are overridden via environment
    assert not any(
        os.getenv(key)
        for key in (
            "ARAGORA_DEFAULT_ROUNDS",
            "ARAGORA_MAX_ROUNDS",
            "ARAGORA_DEFAULT_CONSENSUS",
            "ARAGORA_DEFAULT_AGENTS",
        )
    ), "Server defaults overridden via environment"

    config_path = Path(__file__).resolve().parents[1] / "aragora" / "live" / "src" / "config.ts"
    text = config_path.read_text(encoding="utf-8")

    rounds = int(_extract_default(r"DEFAULT_ROUNDS.*\|\|\s*'(\d+)'", text, "DEFAULT_ROUNDS"))
    max_rounds = int(_extract_default(r"MAX_ROUNDS.*\|\|\s*'(\d+)'", text, "MAX_ROUNDS"))
    consensus = _extract_default(
        r"DEFAULT_CONSENSUS\s*=\s*.*\|\|\s*'([^']+)'", text, "DEFAULT_CONSENSUS"
    )
    agents = _extract_default(r"DEFAULT_AGENTS\s*=.*\|\|\s*'([^']+)'", text, "DEFAULT_AGENTS")

    assert rounds == DEFAULT_ROUNDS
    assert max_rounds == MAX_ROUNDS
    assert consensus == DEFAULT_CONSENSUS
    assert agents == DEFAULT_AGENTS
