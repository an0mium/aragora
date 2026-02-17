"""Tests for GenesisLedger event emissions."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from aragora.genesis.ledger import GenesisLedger, GenesisEventType


@pytest.fixture
def ledger(tmp_path):
    db_path = str(tmp_path / "test_genesis.db")
    return GenesisLedger(db_path=db_path)


class TestGenesisLedgerEvents:
    """Tests for genesis event emission."""

    def test_debate_start_emits_event(self, ledger) -> None:
        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            ledger.record_debate_start(
                debate_id="d1",
                task="Test debate",
                agents=["claude", "gemini"],
            )

        mock_dispatch.assert_called_once()
        data = mock_dispatch.call_args[0][1]
        assert data["event_type"] == "debate_start"
        assert "d1" in str(data["data"])

    def test_agent_birth_emits_event(self, ledger) -> None:
        genome = MagicMock()
        genome.genome_id = "gen_001"
        genome.name = "evolved_claude"
        genome.generation = 3
        genome.traits = {"reasoning": 0.9}
        genome.expertise = ["logic", "math"]
        genome.model_preference = "claude"

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            ledger.record_agent_birth(
                genome=genome,
                parents=["parent_1", "parent_2"],
                birth_type="crossover",
            )

        mock_dispatch.assert_called_once()
        data = mock_dispatch.call_args[0][1]
        assert data["event_type"] == "agent_birth"

    def test_agent_death_emits_event(self, ledger) -> None:
        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            ledger.record_agent_death(
                genome_id="gen_001",
                reason="low_fitness",
                final_fitness=0.2,
            )

        mock_dispatch.assert_called_once()
        data = mock_dispatch.call_args[0][1]
        assert data["event_type"] == "agent_death"

    def test_consensus_emits_event(self, ledger) -> None:
        proof = MagicMock()
        proof.proof_id = "proof_1"
        proof.confidence = 0.85
        proof.supporting_agents = ["claude", "gemini"]
        proof.dissenting_agents = ["grok"]
        proof.final_claim = "We agree on option A"

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            ledger.record_consensus(debate_id="d1", proof=proof)

        mock_dispatch.assert_called_once()
        data = mock_dispatch.call_args[0][1]
        assert data["event_type"] == "consensus_reached"

    def test_debate_spawn_emits_event(self, ledger) -> None:
        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            ledger.record_debate_spawn(
                parent_id="d1",
                child_id="d2",
                trigger="tension",
                tension_description="Disagreement on approach",
            )

        mock_dispatch.assert_called_once()
        data = mock_dispatch.call_args[0][1]
        assert data["event_type"] == "debate_spawn"

    def test_event_type_is_genesis_event(self, ledger) -> None:
        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            ledger.record_debate_start(
                debate_id="d1",
                task="test",
                agents=["a1"],
            )

        assert mock_dispatch.call_args[0][0] == "genesis_event"

    def test_handles_import_error(self, ledger) -> None:
        with patch(
            "aragora.events.dispatcher.dispatch_event",
            side_effect=ImportError("no module"),
        ):
            # Should not raise
            event = ledger.record_debate_start(
                debate_id="d1",
                task="test",
                agents=["a1"],
            )

        # Event still recorded in ledger
        assert event is not None

    def test_multiple_events_each_emit(self, ledger) -> None:
        events = []
        with patch(
            "aragora.events.dispatcher.dispatch_event",
            side_effect=lambda name, data: events.append(data),
        ):
            ledger.record_debate_start("d1", "task1", ["a1", "a2"])
            ledger.record_debate_spawn("d1", "d2", "tension", "Disagreement")
            ledger.record_debate_merge("d1", "d2", True, "Resolved")

        assert len(events) == 3
        types = [e["event_type"] for e in events]
        assert "debate_start" in types
        assert "debate_spawn" in types
        assert "debate_merge" in types

    def test_event_data_is_serializable(self, ledger) -> None:
        """Ensure event data contains only JSON-serializable types."""
        import json

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            ledger.record_debate_start(
                debate_id="d1",
                task="test task",
                agents=["claude", "gemini"],
            )

        data = mock_dispatch.call_args[0][1]
        # Should not raise
        json.dumps(data)
