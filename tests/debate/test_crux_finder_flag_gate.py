"""Flag-gate tests for the crux_finder consensus mode (DIC-15 / #6025).

These tests verify that ``ARAGORA_CRUX_FINDER_ENABLED`` (default off) guards
the crux_finder consensus path and that the consensus_phase falls back to
majority with a machine-readable metadata reason when the flag is absent.

All tests are deterministic and require no live agents, no Arena, and no
network access.
"""

from __future__ import annotations

import pytest

from aragora.debate.crux_mode import (
    CRUX_FINDER_ENV_VAR,
    crux_finder_enabled,
    enable_crux_finder,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure each test starts with the flag absent from the environment."""
    monkeypatch.delenv(CRUX_FINDER_ENV_VAR, raising=False)


# ---------------------------------------------------------------------------
# 1. Flag helpers
# ---------------------------------------------------------------------------


def test_flag_off_by_default() -> None:
    """``crux_finder_enabled()`` must return False when the env var is absent."""
    assert not crux_finder_enabled()


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", "YES", "ON"])
def test_flag_truthy_values(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CRUX_FINDER_ENV_VAR, value)
    assert crux_finder_enabled()


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "  ", "2"])
def test_flag_falsy_values(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CRUX_FINDER_ENV_VAR, value)
    assert not crux_finder_enabled()


def test_enable_function_activates_flag() -> None:
    """``enable_crux_finder()`` must flip ``crux_finder_enabled()`` to True."""
    assert not crux_finder_enabled()
    enable_crux_finder()
    assert crux_finder_enabled()


def test_env_var_name_is_stable() -> None:
    """Constant must not drift — downstream tooling hard-codes this name."""
    assert CRUX_FINDER_ENV_VAR == "ARAGORA_CRUX_FINDER_ENABLED"


def test_flag_symbols_exported_in_all() -> None:
    """All three gate symbols must be in ``__all__`` for clean star-imports."""
    import aragora.debate.crux_mode as mod

    assert "CRUX_FINDER_ENV_VAR" in mod.__all__
    assert "crux_finder_enabled" in mod.__all__
    assert "enable_crux_finder" in mod.__all__


# ---------------------------------------------------------------------------
# 2. Consensus-phase flag-off fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_off_falls_back_to_majority_with_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the flag is absent the handler must fall back to majority and record
    a machine-readable ``crux_finder_skipped_reason`` in result.metadata.
    """
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from aragora.debate.phases.consensus_phase import ConsensusPhase
    from aragora.protocols.debate import DebateProtocol

    protocol = DebateProtocol(consensus="crux_finder")
    result = SimpleNamespace(
        debate_id="d-flag-off",
        rounds_used=0,
        consensus_proof=None,
        consensus_reached=None,
        final_answer=None,
        consensus_strength=None,
        formal_verification=None,
        metadata={},
    )
    ctx = SimpleNamespace(
        env=SimpleNamespace(task="Should the flag gate work?"),
        agents=[SimpleNamespace(name="demo")],
        result=result,
        debate_id="d-flag-off",
        belief_network=None,
    )

    phase = ConsensusPhase.__new__(ConsensusPhase)
    phase.protocol = protocol
    phase._notify_spectator = None
    phase.hooks = {}
    phase._handle_majority_consensus = AsyncMock()  # type: ignore[method-assign]

    await phase._execute_consensus(ctx, "crux_finder")  # type: ignore[arg-type]

    assert result.metadata["crux_finder_skipped_reason"] == "flag_disabled"
    assert result.metadata["crux_finder_fallback_consensus"] == "majority"
    phase._handle_majority_consensus.assert_awaited_once_with(ctx)


@pytest.mark.asyncio
async def test_flag_off_leaves_consensus_proof_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the flag off, the handler must not write a crux_finder proof."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from aragora.debate.phases.consensus_phase import ConsensusPhase
    from aragora.protocols.debate import DebateProtocol

    protocol = DebateProtocol(consensus="crux_finder")
    result = SimpleNamespace(
        debate_id="d-no-proof",
        rounds_used=0,
        consensus_proof=None,
        consensus_reached=None,
        final_answer=None,
        consensus_strength=None,
        formal_verification=None,
        metadata={},
    )
    ctx = SimpleNamespace(
        env=SimpleNamespace(task="q"),
        agents=[],
        result=result,
        debate_id="d-no-proof",
        belief_network=None,
    )

    phase = ConsensusPhase.__new__(ConsensusPhase)
    phase.protocol = protocol
    phase._notify_spectator = None
    phase.hooks = {}
    phase._handle_majority_consensus = AsyncMock()  # type: ignore[method-assign]

    await phase._execute_consensus(ctx, "crux_finder")  # type: ignore[arg-type]

    assert result.consensus_proof is None, (
        "Flag-off must not write a crux_finder proof; "
        "downstream consumers must not see a __CRUX_MAP__ sentinel."
    )
