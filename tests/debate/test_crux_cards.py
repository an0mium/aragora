"""Crux cards on standard debates (#8227 / #9046 phase 1).

`enable_crux_cards` attaches the load-bearing disagreements of a *standard*
debate to its result metadata — without changing the debate goal (that is
`consensus="crux_finder"`, covered by test_crux_mode.py). Flag OFF (the
default) must leave results and receipts byte-identical to pre-flag behavior.
"""

from __future__ import annotations

from types import SimpleNamespace

from aragora.core_types import DebateResult, Message
from aragora.debate.crux_cards import CRUX_CARDS_METADATA_KEY, build_crux_cards
from aragora.debate.protocol import DebateProtocol
from aragora.reasoning.belief import BeliefNetwork
from aragora.reasoning.claims import RelationType


def _contested_network() -> BeliefNetwork:
    """A small network with one clearly load-bearing contested claim."""
    network = BeliefNetwork()
    network.add_claim("c1", "Load-bearing contested claim", "agent-alpha", 0.55)
    network.add_claim("c2", "Depends on c1", "agent-alpha", 0.7)
    network.add_claim("c3", "Also depends on c1", "agent-alpha", 0.4)
    network.add_claim("c4", "Counter-evidence against c1", "agent-beta", 0.6)
    network.add_claim("c5", "Supporting evidence for c1", "agent-gamma", 0.5)
    network.add_factor("c1", "c2", RelationType.SUPPORTS)
    network.add_factor("c1", "c3", RelationType.SUPPORTS)
    network.add_factor("c4", "c1", RelationType.CONTRADICTS)
    network.add_factor("c5", "c1", RelationType.SUPPORTS)
    return network


class TestProtocolFlag:
    def test_default_off(self) -> None:
        assert DebateProtocol().enable_crux_cards is False

    def test_opt_in(self) -> None:
        assert DebateProtocol(enable_crux_cards=True).enable_crux_cards is True


class TestBuildCruxCards:
    def test_from_belief_network(self) -> None:
        cards = build_crux_cards(belief_network=_contested_network(), top_k=5, min_score=0.05)
        assert cards is not None
        assert cards["items"], "contested network must yield at least one crux"
        first = cards["items"][0]
        # Per-crux dissent attribution (the work order's acceptance criterion).
        assert first["statement"]
        assert "contesting_agents" in first
        assert "author" in first
        assert "crux_score" in first
        assert "convergence_barrier" in cards
        assert cards["detector"] == "belief_network"

    def test_no_material_returns_none(self) -> None:
        assert build_crux_cards(belief_network=None, messages=[]) is None

    def test_messages_fallback_does_not_raise(self) -> None:
        messages = [
            Message(role="proposer", agent="agent-a", content="X is clearly true."),
            Message(role="critic", agent="agent-b", content="X is clearly false."),
            Message(role="synthesizer", agent="agent-c", content="ignored role"),
        ]
        cards = build_crux_cards(messages=messages, min_score=0.0)
        assert cards is None or cards["items"]

    def test_high_min_score_filters_to_none(self) -> None:
        cards = build_crux_cards(belief_network=_contested_network(), min_score=1.1)
        assert cards is None


class TestConsensusPhaseAttach:
    def _phase(self, protocol: DebateProtocol):
        from aragora.debate.phases.consensus_phase import ConsensusPhase

        return ConsensusPhase(protocol=protocol)

    def _ctx(self, network: BeliefNetwork | None) -> SimpleNamespace:
        result = DebateResult(
            debate_id="d-crux",
            task="Should we ship?",
            final_answer="Yes, with a canary rollout.",
            confidence=0.8,
            consensus_reached=True,
            rounds_used=2,
            participants=["agent-alpha", "agent-beta"],
        )
        return SimpleNamespace(result=result, belief_network=network)

    def test_attach_sets_metadata(self) -> None:
        phase = self._phase(DebateProtocol(enable_crux_cards=True, crux_finder_min_score=0.05))
        ctx = self._ctx(_contested_network())
        phase._attach_crux_cards(ctx)
        cards = ctx.result.metadata.get(CRUX_CARDS_METADATA_KEY)
        assert cards is not None
        assert cards["items"]
        assert cards["items"][0]["contesting_agents"] is not None

    def test_attach_without_material_leaves_metadata_untouched(self) -> None:
        phase = self._phase(DebateProtocol(enable_crux_cards=True, crux_finder_min_score=0.05))
        ctx = self._ctx(network=None)
        phase._attach_crux_cards(ctx)
        assert CRUX_CARDS_METADATA_KEY not in ctx.result.metadata

    def test_attach_never_raises(self) -> None:
        phase = self._phase(DebateProtocol(enable_crux_cards=True))
        broken_ctx = SimpleNamespace(result=None, belief_network=None)
        phase._attach_crux_cards(broken_ctx)  # must swallow, not raise
