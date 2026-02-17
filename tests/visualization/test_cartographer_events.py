"""Tests for ArgumentCartographer event emissions."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.visualization.mapper import ArgumentCartographer


class TestCartographerEventEmission:
    """Tests for _emit_graph_update."""

    def test_message_emits_event(self) -> None:
        cart = ArgumentCartographer()
        cart.set_debate_context("debate_1", "Test topic")

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            node_id = cart.update_from_message(
                agent="agent_a",
                content="I propose we consider option A",
                role="proposer",
                round_num=1,
            )

        mock_dispatch.assert_called_once()
        data = mock_dispatch.call_args[0][1]
        assert data["debate_id"] == "debate_1"
        assert data["action"] == "message"
        assert data["node_id"] == node_id
        assert data["total_nodes"] == 1
        assert data["total_edges"] >= 0

    def test_vote_emits_event(self) -> None:
        cart = ArgumentCartographer()
        cart.set_debate_context("debate_2", "Vote topic")

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            node_id = cart.update_from_vote("agent_b", "agree", round_num=1)

        mock_dispatch.assert_called_once()
        data = mock_dispatch.call_args[0][1]
        assert data["action"] == "vote"
        assert data["node_id"] == node_id

    def test_consensus_emits_event(self) -> None:
        cart = ArgumentCartographer()
        cart.set_debate_context("debate_3", "Consensus topic")

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            node_id = cart.update_from_consensus(
                result="majority",
                round_num=2,
                vote_counts={"agree": 3, "disagree": 1},
            )

        mock_dispatch.assert_called_once()
        data = mock_dispatch.call_args[0][1]
        assert data["action"] == "consensus"
        assert data["total_nodes"] == 1

    def test_multiple_messages_accumulate(self) -> None:
        cart = ArgumentCartographer()
        cart.set_debate_context("debate_4", "Multi topic")

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            cart.update_from_message("a1", "First message", "proposer", 1)
            cart.update_from_message("a2", "Second message", "critic", 1)

        assert mock_dispatch.call_count == 2
        last_data = mock_dispatch.call_args_list[-1][0][1]
        assert last_data["total_nodes"] == 2

    def test_event_type_is_argument_map_updated(self) -> None:
        cart = ArgumentCartographer()

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            cart.update_from_message("agent", "content", "proposer", 1)
            assert mock_dispatch.call_args[0][0] == "argument_map_updated"

    def test_handles_import_error(self) -> None:
        cart = ArgumentCartographer()

        with patch(
            "aragora.events.dispatcher.dispatch_event",
            side_effect=ImportError("no module"),
        ):
            # Should not raise
            node_id = cart.update_from_message("agent", "content", "proposer", 1)

        assert node_id  # Still returns the node ID
        assert len(cart.nodes) == 1

    def test_no_debate_id_uses_empty_string(self) -> None:
        cart = ArgumentCartographer()  # No context set

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            cart.update_from_message("agent", "content", "proposer", 1)

        data = mock_dispatch.call_args[0][1]
        assert data["debate_id"] == ""

    def test_critique_does_not_emit_directly(self) -> None:
        """Critiques create edges, not primary graph events."""
        cart = ArgumentCartographer()
        cart.set_debate_context("debate_5", "Critique topic")

        # Add messages first so critique has nodes to link
        with patch("aragora.events.dispatcher.dispatch_event"):
            cart.update_from_message("a1", "Proposal", "proposer", 1)
            cart.update_from_message("a2", "Response", "critic", 1)

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            cart.update_from_critique("a2", "a1", 0.8, 1, "Too risky")
            # Critiques don't call _emit_graph_update
            mock_dispatch.assert_not_called()

    def test_full_debate_flow_events(self) -> None:
        cart = ArgumentCartographer()
        cart.set_debate_context("debate_6", "Full flow")
        events = []

        with patch(
            "aragora.events.dispatcher.dispatch_event",
            side_effect=lambda name, data: events.append(data),
        ):
            cart.update_from_message("a1", "My proposal", "proposer", 1)
            cart.update_from_message("a2", "I disagree", "critic", 1)
            cart.update_from_vote("a1", "agree", 1)
            cart.update_from_vote("a2", "disagree", 1)
            cart.update_from_consensus("split", 1, {"agree": 1, "disagree": 1})

        assert len(events) == 5
        actions = [e["action"] for e in events]
        assert actions == ["message", "message", "vote", "vote", "consensus"]
        # Final event should have all nodes
        assert events[-1]["total_nodes"] == 5
