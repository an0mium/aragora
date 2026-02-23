"""Tests for aragora.debate.event_bridge.EventEmitterBridge."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, call, patch

import pytest

from aragora.debate.event_bridge import EventEmitterBridge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bridge(spectator=None, event_emitter=None, cartographer=None, loop_id=""):
    """Construct an EventEmitterBridge with optional mocks."""
    return EventEmitterBridge(
        spectator=spectator,
        event_emitter=event_emitter,
        cartographer=cartographer,
        loop_id=loop_id,
    )


def _make_stream_types():
    """Return a (StreamEvent, StreamEventType) mock pair compatible with the bridge.

    The bridge uses::

        stream_type = getattr(StreamEventType, stream_type_name, None)
        if not stream_type:
            return

    so we need each attribute access on StreamEventType to return a truthy value.
    MagicMock attribute access already does this by default (child mocks are truthy),
    so a plain MagicMock() works fine as long as we don't accidentally spec it.
    """
    StreamEventType = MagicMock(name="StreamEventType")
    # Child attribute mocks are truthy by default — no extra setup needed.

    StreamEvent = MagicMock(name="StreamEvent")

    # When StreamEvent(...) is called, return a predictable object.
    event_instance = MagicMock(name="stream_event_instance")
    StreamEvent.return_value = event_instance

    return StreamEvent, StreamEventType, event_instance


def _patch_lazy_imports(StreamEvent, StreamEventType, push_spectator_event=None):
    """
    Return a context manager that patches both lazy import targets used by the bridge.

    Usage::

        with _patch_lazy_imports(SE, SET) as push_mock:
            ...
    """
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        events_module = ModuleType("aragora.events.types")
        events_module.StreamEvent = StreamEvent
        events_module.StreamEventType = StreamEventType

        spectate_module = ModuleType("aragora.server.handlers.debates.spectate")
        push_fn = push_spectator_event or MagicMock(name="push_spectator_event")
        spectate_module.push_spectator_event = push_fn

        patched_sys_modules = dict(sys.modules)
        patched_sys_modules["aragora.events.types"] = events_module
        patched_sys_modules["aragora.server.handlers.debates.spectate"] = spectate_module

        with patch.dict(sys.modules, patched_sys_modules):
            yield push_fn

    return _ctx()


# ===========================================================================
# 1. EVENT_TYPE_MAPPING
# ===========================================================================


class TestEventTypeMapping:
    """Verify the completeness and values of EVENT_TYPE_MAPPING."""

    EXPECTED = {
        "debate_start": "DEBATE_START",
        "debate_end": "DEBATE_END",
        "round": "ROUND_START",
        "round_start": "ROUND_START",
        "propose": "AGENT_MESSAGE",
        "proposal": "AGENT_MESSAGE",
        "critique": "CRITIQUE",
        "vote": "VOTE",
        "consensus": "CONSENSUS",
        "convergence": "CONSENSUS",
        "judge": "AGENT_MESSAGE",
        "memory_recall": "MEMORY_RECALL",
        "audience_drain": "AUDIENCE_DRAIN",
        "audience_summary": "AUDIENCE_SUMMARY",
        "insight_extracted": "INSIGHT_EXTRACTED",
        "token_start": "TOKEN_START",
        "token_delta": "TOKEN_DELTA",
        "token_end": "TOKEN_END",
        "graph_update": "GRAPH_UPDATE",
        "claim_verification": "CLAIM_VERIFICATION_RESULT",
        "memory_tier_promotion": "MEMORY_TIER_PROMOTION",
        "memory_tier_demotion": "MEMORY_TIER_DEMOTION",
        "agent_elo_updated": "AGENT_ELO_UPDATED",
    }

    def test_all_expected_keys_present(self):
        for key in self.EXPECTED:
            assert key in EventEmitterBridge.EVENT_TYPE_MAPPING, f"Missing key: {key}"

    def test_all_values_correct(self):
        for key, expected_value in self.EXPECTED.items():
            actual = EventEmitterBridge.EVENT_TYPE_MAPPING[key]
            assert actual == expected_value, (
                f"Mapping for {key!r}: expected {expected_value!r}, got {actual!r}"
            )

    def test_no_extra_unmapped_keys(self):
        """The mapping should contain exactly the expected entries."""
        assert set(EventEmitterBridge.EVENT_TYPE_MAPPING.keys()) == set(self.EXPECTED.keys())


# ===========================================================================
# 2. __init__
# ===========================================================================


class TestInit:
    def test_default_construction(self):
        bridge = EventEmitterBridge()
        assert bridge.spectator is None
        assert bridge.event_emitter is None
        assert bridge.cartographer is None
        assert bridge.loop_id == ""

    def test_custom_construction(self):
        spec = MagicMock(name="spectator")
        ee = MagicMock(name="event_emitter")
        cart = MagicMock(name="cartographer")
        bridge = EventEmitterBridge(
            spectator=spec, event_emitter=ee, cartographer=cart, loop_id="loop-99"
        )

        assert bridge.spectator is spec
        assert bridge.event_emitter is ee
        assert bridge.cartographer is cart
        assert bridge.loop_id == "loop-99"


# ===========================================================================
# 3. notify() — spectator param filtering
# ===========================================================================


class TestNotifySpectator:
    """notify() should call spectator.emit with ONLY supported kwargs."""

    SUPPORTED = ("agent", "details", "metric", "round_number")

    def test_supported_params_forwarded(self):
        spec = MagicMock(name="spectator")
        bridge = _make_bridge(spectator=spec)
        bridge.notify("proposal", agent="claude", details="A proposal", round_number=2, metric=0.9)
        spec.emit.assert_called_once_with(
            "proposal",
            agent="claude",
            details="A proposal",
            round_number=2,
            metric=0.9,
        )

    def test_unsupported_params_stripped(self):
        spec = MagicMock(name="spectator")
        bridge = _make_bridge(spectator=spec)
        bridge.notify(
            "proposal",
            agent="gpt4",
            details="Hello",
            unsupported_key="should_be_stripped",
            another_extra=42,
        )
        _, call_kwargs = spec.emit.call_args
        assert "unsupported_key" not in call_kwargs
        assert "another_extra" not in call_kwargs
        assert call_kwargs.get("agent") == "gpt4"

    def test_no_spectator_no_error(self):
        bridge = _make_bridge(spectator=None)
        # Should not raise
        bridge.notify("proposal", agent="claude", details="Hello")

    def test_all_supported_params_passed_individually(self):
        for param in self.SUPPORTED:
            spec = MagicMock(name="spectator")
            bridge = _make_bridge(spectator=spec)
            bridge.notify("vote", **{param: "test_value"})
            call_kwargs = spec.emit.call_args[1]
            assert param in call_kwargs


# ===========================================================================
# 4. notify() — SSE / push_spectator_event
# ===========================================================================


class TestNotifySSE:
    def test_push_called_when_loop_id_set(self):
        push_fn = MagicMock(name="push_spectator_event")
        SE, SET, _ = _make_stream_types()
        with _patch_lazy_imports(SE, SET, push_fn):
            bridge = _make_bridge(loop_id="debate-42")
            bridge.notify("proposal", agent="claude", details="Hi")
            push_fn.assert_called_once()
            args = push_fn.call_args
            assert args[0][0] == "debate-42"  # first positional arg
            assert args[0][1] == "proposal"  # second positional arg

    def test_push_not_called_when_no_loop_id(self):
        push_fn = MagicMock(name="push_spectator_event")
        SE, SET, _ = _make_stream_types()
        with _patch_lazy_imports(SE, SET, push_fn):
            bridge = _make_bridge(loop_id="")
            bridge.notify("proposal", agent="claude", details="Hi")
            push_fn.assert_not_called()

    def test_import_error_silently_caught(self):
        """If push_spectator_event cannot be imported, notify must not raise."""
        bridge = _make_bridge(loop_id="debate-1")
        with patch.dict(sys.modules, {"aragora.server.handlers.debates.spectate": None}):
            # Should not raise
            bridge.notify("proposal", agent="claude")

    def test_sse_uses_filtered_spectator_kwargs(self):
        """push_spectator_event should receive only filtered kwargs (no extras)."""
        push_fn = MagicMock(name="push_spectator_event")
        SE, SET, _ = _make_stream_types()
        with _patch_lazy_imports(SE, SET, push_fn):
            bridge = _make_bridge(loop_id="loop-7")
            bridge.notify("proposal", agent="alice", details="D", extra_param="NOPE")
            call_kwargs = push_fn.call_args[1]
            assert "extra_param" not in call_kwargs


# ===========================================================================
# 5. notify() → _emit_to_websocket
# ===========================================================================


class TestNotifyWebSocket:
    def test_emit_called_on_event_emitter(self):
        SE, SET, event_inst = _make_stream_types()
        ee = MagicMock(name="event_emitter")
        bridge = _make_bridge(event_emitter=ee, loop_id="loop-1")

        with _patch_lazy_imports(SE, SET):
            bridge.notify("proposal", agent="claude", details="Proposed", round_number=1)

        ee.emit.assert_called()

    def test_no_event_emitter_no_websocket_call(self):
        SE, SET, _ = _make_stream_types()
        bridge = _make_bridge(event_emitter=None)
        with _patch_lazy_imports(SE, SET):
            bridge.notify("proposal", agent="claude", details="Proposed")
        # Should just not raise


# ===========================================================================
# 6. _emit_to_websocket
# ===========================================================================


class TestEmitToWebSocket:
    def test_skips_unmapped_event_type(self):
        SE, SET, _ = _make_stream_types()
        ee = MagicMock(name="event_emitter")
        bridge = _make_bridge(event_emitter=ee)
        with _patch_lazy_imports(SE, SET):
            bridge._emit_to_websocket("totally_unknown_event", agent="x")
        ee.emit.assert_not_called()

    def test_skips_when_streamtype_attr_missing(self):
        """If StreamEventType does not have the mapped attr, skip emission."""
        SE = MagicMock(name="StreamEvent")
        SET = MagicMock(name="StreamEventType")
        # Simulate missing attribute by returning None via spec
        SET.__class__ = type  # make getattr work normally
        SET.AGENT_MESSAGE = None  # mapped value for "proposal" is AGENT_MESSAGE

        ee = MagicMock(name="event_emitter")
        bridge = _make_bridge(event_emitter=ee)

        events_mod = ModuleType("aragora.events.types")
        events_mod.StreamEvent = SE
        events_mod.StreamEventType = SET
        with patch.dict(sys.modules, {"aragora.events.types": events_mod}):
            bridge._emit_to_websocket("proposal", agent="x")

        ee.emit.assert_not_called()

    def test_stream_event_constructed_with_correct_fields(self):
        SE, SET, event_inst = _make_stream_types()
        # Make getattr return a concrete mock so the branch doesn't skip
        stream_type_value = MagicMock(name="AGENT_MESSAGE")
        type(SET).AGENT_MESSAGE = MagicMock(return_value=stream_type_value)
        SET.AGENT_MESSAGE = stream_type_value

        ee = MagicMock(name="event_emitter")
        bridge = _make_bridge(event_emitter=ee, loop_id="loop-5")

        with _patch_lazy_imports(SE, SET):
            bridge._emit_to_websocket(
                "proposal", agent="alice", details="My idea", round_number=3, metric=0.8
            )

        SE.assert_called()
        call_kwargs = SE.call_args[1]
        assert call_kwargs["agent"] == "alice"
        assert call_kwargs["loop_id"] == "loop-5"
        assert call_kwargs["round"] == 3
        assert call_kwargs["data"]["details"] == "My idea"
        assert call_kwargs["data"]["metric"] == 0.8
        assert call_kwargs["data"]["event_source"] == "spectator"

    def test_emission_exception_does_not_propagate(self):
        """Errors inside _emit_to_websocket should be swallowed."""
        SE, SET, event_inst = _make_stream_types()
        ee = MagicMock(name="event_emitter")
        ee.emit.side_effect = RuntimeError("broken emitter")

        bridge = _make_bridge(event_emitter=ee)
        with _patch_lazy_imports(SE, SET):
            # Should not raise
            bridge._emit_to_websocket("proposal", agent="x", details="y")

    def test_no_event_emitter_returns_early(self):
        bridge = _make_bridge(event_emitter=None)
        # No import needed — should return immediately
        bridge._emit_to_websocket("proposal", agent="x")

    def test_cartographer_updated_after_websocket_emit(self):
        SE, SET, _ = _make_stream_types()
        ee = MagicMock(name="event_emitter")
        cart = MagicMock(name="cartographer")

        bridge = _make_bridge(event_emitter=ee, cartographer=cart)
        with _patch_lazy_imports(SE, SET):
            bridge._emit_to_websocket("proposal", agent="alice", details="A", round_number=1)

        cart.update_from_message.assert_called_once()


# ===========================================================================
# 7. _update_cartographer — event routing
# ===========================================================================


class TestUpdateCartographer:
    def test_no_cartographer_returns_early(self):
        bridge = _make_bridge(cartographer=None)
        # Should not raise
        bridge._update_cartographer("proposal", agent="x")

    def test_proposal_calls_update_from_message(self):
        SE, SET, _ = _make_stream_types()
        cart = MagicMock(name="cartographer")
        ee = MagicMock(name="event_emitter")
        bridge = _make_bridge(event_emitter=ee, cartographer=cart)

        with _patch_lazy_imports(SE, SET):
            bridge._update_cartographer(
                "proposal", agent="alice", details="My plan", round_number=2
            )

        cart.update_from_message.assert_called_once_with(
            agent="alice",
            content="My plan",
            role="proposer",
            round_num=2,
        )

    def test_propose_alias_calls_update_from_message(self):
        SE, SET, _ = _make_stream_types()
        cart = MagicMock(name="cartographer")
        ee = MagicMock(name="event_emitter")
        bridge = _make_bridge(event_emitter=ee, cartographer=cart)

        with _patch_lazy_imports(SE, SET):
            bridge._update_cartographer("propose", agent="bob", details="Concept", round_number=1)

        cart.update_from_message.assert_called_once()

    def test_critique_calls_update_from_critique(self):
        SE, SET, _ = _make_stream_types()
        cart = MagicMock(name="cartographer")
        ee = MagicMock(name="event_emitter")
        bridge = _make_bridge(event_emitter=ee, cartographer=cart)

        details = "Critiqued alice: weak argument"
        with _patch_lazy_imports(SE, SET):
            bridge._update_cartographer(
                "critique", agent="bob", details=details, metric=0.7, round_number=3
            )

        cart.update_from_critique.assert_called_once_with(
            critic_agent="bob",
            target_agent="alice",  # extracted from "Critiqued alice: ..."
            severity=0.7,
            round_num=3,
            critique_text=details,
        )

    def test_critique_non_numeric_metric_defaults_to_0_5(self):
        SE, SET, _ = _make_stream_types()
        cart = MagicMock(name="cartographer")
        ee = MagicMock(name="event_emitter")
        bridge = _make_bridge(event_emitter=ee, cartographer=cart)

        with _patch_lazy_imports(SE, SET):
            bridge._update_cartographer(
                "critique", agent="x", details="Some critique", metric="bad_value"
            )

        call_kwargs = cart.update_from_critique.call_args[1]
        assert call_kwargs["severity"] == 0.5

    def test_vote_calls_update_from_vote_with_colon(self):
        SE, SET, _ = _make_stream_types()
        cart = MagicMock(name="cartographer")
        ee = MagicMock(name="event_emitter")
        bridge = _make_bridge(event_emitter=ee, cartographer=cart)

        with _patch_lazy_imports(SE, SET):
            bridge._update_cartographer(
                "vote", agent="carol", details="choice: agree", round_number=1
            )

        cart.update_from_vote.assert_called_once_with(
            agent="carol",
            vote_value="agree",
            round_num=1,
        )

    def test_vote_calls_update_from_vote_without_colon(self):
        SE, SET, _ = _make_stream_types()
        cart = MagicMock(name="cartographer")
        ee = MagicMock(name="event_emitter")
        bridge = _make_bridge(event_emitter=ee, cartographer=cart)

        with _patch_lazy_imports(SE, SET):
            bridge._update_cartographer("vote", agent="dave", details="abstain", round_number=2)

        cart.update_from_vote.assert_called_once_with(
            agent="dave",
            vote_value="abstain",
            round_num=2,
        )

    def test_consensus_calls_update_from_consensus_with_colon(self):
        SE, SET, _ = _make_stream_types()
        cart = MagicMock(name="cartographer")
        ee = MagicMock(name="event_emitter")
        bridge = _make_bridge(event_emitter=ee, cartographer=cart)

        with _patch_lazy_imports(SE, SET):
            bridge._update_cartographer(
                "consensus", agent="", details="result: approved", round_number=4
            )

        cart.update_from_consensus.assert_called_once_with(
            result="approved",
            round_num=4,
        )

    def test_consensus_calls_update_from_consensus_without_colon(self):
        SE, SET, _ = _make_stream_types()
        cart = MagicMock(name="cartographer")
        ee = MagicMock(name="event_emitter")
        bridge = _make_bridge(event_emitter=ee, cartographer=cart)

        with _patch_lazy_imports(SE, SET):
            bridge._update_cartographer("consensus", agent="", details="rejected", round_number=5)

        cart.update_from_consensus.assert_called_once_with(
            result="rejected",
            round_num=5,
        )

    def test_unknown_event_does_not_call_any_cartographer_method(self):
        SE, SET, _ = _make_stream_types()
        cart = MagicMock(name="cartographer")
        ee = MagicMock(name="event_emitter")
        bridge = _make_bridge(event_emitter=ee, cartographer=cart)

        with _patch_lazy_imports(SE, SET):
            bridge._update_cartographer("memory_recall", agent="x", details="y")

        cart.update_from_message.assert_not_called()
        cart.update_from_critique.assert_not_called()
        cart.update_from_vote.assert_not_called()
        cart.update_from_consensus.assert_not_called()

    def test_cartographer_exception_does_not_propagate(self):
        SE, SET, _ = _make_stream_types()
        cart = MagicMock(name="cartographer")
        cart.update_from_message.side_effect = RuntimeError("cart exploded")
        ee = MagicMock(name="event_emitter")
        bridge = _make_bridge(event_emitter=ee, cartographer=cart)

        with _patch_lazy_imports(SE, SET):
            # Should not raise
            bridge._update_cartographer("proposal", agent="x", details="y")

    def test_graph_update_emitted_after_proposal(self):
        SE, SET, _ = _make_stream_types()
        cart = MagicMock(name="cartographer")
        cart.to_dict.return_value = {"nodes": [], "edges": []}
        ee = MagicMock(name="event_emitter")
        bridge = _make_bridge(event_emitter=ee, cartographer=cart)

        with _patch_lazy_imports(SE, SET):
            bridge._update_cartographer("proposal", agent="x", details="y", round_number=1)

        # _emit_graph_update calls event_emitter.emit once (for GRAPH_UPDATE)
        ee.emit.assert_called()


# ===========================================================================
# 8. _emit_graph_update
# ===========================================================================


class TestEmitGraphUpdate:
    def test_requires_both_event_emitter_and_cartographer(self):
        SE, SET, _ = _make_stream_types()
        ee = MagicMock(name="event_emitter")
        cart = MagicMock(name="cartographer")
        cart.to_dict.return_value = {"nodes": []}

        # Missing cartographer — should not emit
        bridge = _make_bridge(event_emitter=ee, cartographer=None)
        with _patch_lazy_imports(SE, SET):
            bridge._emit_graph_update()
        ee.emit.assert_not_called()

        # Missing event_emitter — should not emit
        bridge2 = _make_bridge(event_emitter=None, cartographer=cart)
        with _patch_lazy_imports(SE, SET):
            bridge2._emit_graph_update()

    def test_emits_stream_event_with_graph_data(self):
        SE, SET, event_inst = _make_stream_types()
        ee = MagicMock(name="event_emitter")
        cart = MagicMock(name="cartographer")
        graph_data = {"nodes": ["a", "b"], "edges": [("a", "b")]}
        cart.to_dict.return_value = graph_data

        bridge = _make_bridge(event_emitter=ee, cartographer=cart, loop_id="loop-X")
        with _patch_lazy_imports(SE, SET):
            bridge._emit_graph_update()

        ee.emit.assert_called_once()
        SE.assert_called()
        # The StreamEvent should have been constructed with data=graph_data
        call_kwargs = SE.call_args[1]
        assert call_kwargs["data"] == graph_data
        assert call_kwargs["loop_id"] == "loop-X"

    def test_exception_in_graph_update_does_not_propagate(self):
        SE, SET, _ = _make_stream_types()
        ee = MagicMock(name="event_emitter")
        ee.emit.side_effect = RuntimeError("boom")
        cart = MagicMock(name="cartographer")
        cart.to_dict.return_value = {}

        bridge = _make_bridge(event_emitter=ee, cartographer=cart)
        with _patch_lazy_imports(SE, SET):
            # Must not raise
            bridge._emit_graph_update()


# ===========================================================================
# 9. _extract_critique_target (static method)
# ===========================================================================


class TestExtractCritiqueTarget:
    def test_extracts_target_when_critiqued_present(self):
        target = EventEmitterBridge._extract_critique_target(
            "Critiqued alice: the argument is weak"
        )
        assert target == "alice"

    def test_extracts_target_with_extra_spaces(self):
        target = EventEmitterBridge._extract_critique_target("Critiqued bob smith: something else")
        assert target == "bob smith"

    def test_returns_empty_when_critiqued_absent(self):
        target = EventEmitterBridge._extract_critique_target(
            "General critique text without the keyword"
        )
        assert target == ""

    def test_returns_empty_for_empty_string(self):
        target = EventEmitterBridge._extract_critique_target("")
        assert target == ""

    def test_returns_empty_for_partial_match(self):
        # "critiqued" lowercase — no match (case-sensitive)
        target = EventEmitterBridge._extract_critique_target("critiqued carol: bad")
        assert target == ""

    def test_handles_multiple_colons_correctly(self):
        # Should split only on the first colon after the agent name
        target = EventEmitterBridge._extract_critique_target("Critiqued dave: reason: more details")
        assert target == "dave"


# ===========================================================================
# 10. emit_moment
# ===========================================================================


class TestEmitMoment:
    def test_emits_moment_detected_event(self):
        SE, SET, event_inst = _make_stream_types()
        ee = MagicMock(name="event_emitter")
        moment = MagicMock(name="moment")
        moment.to_dict.return_value = {"moment_type": "breakthrough", "agent_name": "claude"}
        moment.moment_type = "breakthrough"
        moment.agent_name = "claude"

        bridge = _make_bridge(event_emitter=ee, loop_id="loop-M")
        with _patch_lazy_imports(SE, SET):
            bridge.emit_moment(moment)

        ee.emit.assert_called_once()
        SE.assert_called()
        call_kwargs = SE.call_args[1]
        assert call_kwargs["data"] == {"moment_type": "breakthrough", "agent_name": "claude"}
        assert call_kwargs["loop_id"] == "loop-M"

    def test_no_event_emitter_returns_early(self):
        moment = MagicMock(name="moment")
        bridge = _make_bridge(event_emitter=None)
        # Should not raise, and should not import anything
        bridge.emit_moment(moment)
        moment.to_dict.assert_not_called()

    def test_loop_id_defaults_to_unknown_when_empty(self):
        SE, SET, _ = _make_stream_types()
        ee = MagicMock(name="event_emitter")
        moment = MagicMock(name="moment")
        moment.to_dict.return_value = {}
        moment.moment_type = "x"
        moment.agent_name = "y"

        bridge = _make_bridge(event_emitter=ee, loop_id="")
        with _patch_lazy_imports(SE, SET):
            bridge.emit_moment(moment)

        call_kwargs = SE.call_args[1]
        assert call_kwargs["loop_id"] == "unknown"

    def test_exception_does_not_propagate(self):
        SE, SET, _ = _make_stream_types()
        ee = MagicMock(name="event_emitter")
        ee.emit.side_effect = ValueError("bad emit")
        moment = MagicMock(name="moment")
        moment.to_dict.return_value = {}
        moment.moment_type = "x"
        moment.agent_name = "y"

        bridge = _make_bridge(event_emitter=ee, loop_id="loop-Z")
        with _patch_lazy_imports(SE, SET):
            # Must not raise
            bridge.emit_moment(moment)


# ===========================================================================
# 11. Full notify() integration scenarios
# ===========================================================================


class TestNotifyIntegration:
    """End-to-end notify() scenarios covering multiple listeners at once."""

    def test_notify_fires_all_three_listeners(self):
        SE, SET, _ = _make_stream_types()
        spec = MagicMock(name="spectator")
        ee = MagicMock(name="event_emitter")
        cart = MagicMock(name="cartographer")
        push_fn = MagicMock(name="push_fn")

        bridge = _make_bridge(
            spectator=spec, event_emitter=ee, cartographer=cart, loop_id="loop-ALL"
        )
        with _patch_lazy_imports(SE, SET, push_fn):
            bridge.notify("proposal", agent="alice", details="My idea", round_number=1)

        spec.emit.assert_called_once()
        push_fn.assert_called_once()
        ee.emit.assert_called()  # called for WebSocket + graph update

    def test_notify_debate_start_no_cartographer_update(self):
        """debate_start is not a cartographer event — cartographer methods not called."""
        SE, SET, _ = _make_stream_types()
        ee = MagicMock(name="event_emitter")
        cart = MagicMock(name="cartographer")

        bridge = _make_bridge(event_emitter=ee, cartographer=cart)
        with _patch_lazy_imports(SE, SET):
            bridge.notify("debate_start", agent="", details="starting")

        cart.update_from_message.assert_not_called()
        cart.update_from_critique.assert_not_called()
        cart.update_from_vote.assert_not_called()
        cart.update_from_consensus.assert_not_called()

    def test_notify_convergence_maps_to_consensus_stream_type(self):
        """convergence maps to CONSENSUS in EVENT_TYPE_MAPPING."""
        SE, SET, _ = _make_stream_types()
        ee = MagicMock(name="event_emitter")
        bridge = _make_bridge(event_emitter=ee)

        with _patch_lazy_imports(SE, SET):
            bridge.notify("convergence", agent="", details="converged")

        # Stream event should have been constructed (mapped to CONSENSUS)
        SE.assert_called()
        call_kwargs = SE.call_args[1]
        # The type should be whatever getattr(StreamEventType, "CONSENSUS") returned
        assert call_kwargs["type"] is not None

    def test_notify_without_any_listener_is_no_op(self):
        bridge = _make_bridge()
        # Should not raise even with no spectator, no emitter, no cartographer
        bridge.notify("proposal", agent="x", details="y", round_number=1)

    def test_notify_with_all_feedback_loop_events(self):
        """Ensure feedback loop event types are emitted to WebSocket."""
        SE, SET, _ = _make_stream_types()
        ee = MagicMock(name="event_emitter")
        bridge = _make_bridge(event_emitter=ee)

        feedback_events = [
            "claim_verification",
            "memory_tier_promotion",
            "memory_tier_demotion",
            "agent_elo_updated",
        ]
        with _patch_lazy_imports(SE, SET):
            for evt in feedback_events:
                ee.reset_mock()
                SE.reset_mock()
                bridge._emit_to_websocket(evt, agent="x", details="y")
                SE.assert_called(), f"StreamEvent not constructed for {evt}"
