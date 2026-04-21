from __future__ import annotations

import os

import pytest


@pytest.mark.skipif(
    os.environ.get("ARAGORA_LIVE_AGENT_BRIDGE") != "1",
    reason="Set ARAGORA_LIVE_AGENT_BRIDGE=1 to run live agent bridge smoke tests",
)
def test_live_smoke_skeleton() -> None:
    pytest.skip("Live smoke skeleton only; real subprocess coverage is opt-in.")
