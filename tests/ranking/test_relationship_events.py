"""Tests for RelationshipTracker event emissions."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from aragora.ranking.relationships import (
    RelationshipTracker,
    AgentRelationship,
)


class TestRelationshipEventEmission:
    """Tests for _emit_relationship_event."""

    def test_emits_event_for_relationship(self) -> None:
        tracker = RelationshipTracker.__new__(RelationshipTracker)
        tracker._db = MagicMock()

        rel = AgentRelationship(
            agent_a="claude",
            agent_b="gemini",
            debate_count=5,
            agreement_count=4,
            critique_count_a_to_b=2,
            critique_count_b_to_a=3,
            critique_accepted_a_to_b=1,
            critique_accepted_b_to_a=2,
            position_changes_a_after_b=1,
            position_changes_b_after_a=0,
            a_wins_over_b=3,
            b_wins_over_a=2,
        )

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            tracker._emit_relationship_event("claude", "gemini", rel)

        mock_dispatch.assert_called_once()
        data = mock_dispatch.call_args[0][1]
        assert data["agent_a"] == "claude"
        assert data["agent_b"] == "gemini"
        assert data["debate_count"] == 5

    def test_classifies_ally_relationship(self) -> None:
        tracker = RelationshipTracker.__new__(RelationshipTracker)
        tracker._db = MagicMock()

        rel = AgentRelationship(
            agent_a="claude",
            agent_b="gpt4",
            debate_count=10,
            agreement_count=9,  # High agreement
            critique_count_a_to_b=5,
            critique_count_b_to_a=5,
            critique_accepted_a_to_b=4,
            critique_accepted_b_to_a=4,
            position_changes_a_after_b=0,
            position_changes_b_after_a=0,
            a_wins_over_b=5,
            b_wins_over_a=5,
        )

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            tracker._emit_relationship_event("claude", "gpt4", rel)

        data = mock_dispatch.call_args[0][1]
        assert data["relationship_type"] == "ally"
        assert data["alliance_score"] > 0.5

    def test_classifies_rival_relationship(self) -> None:
        tracker = RelationshipTracker.__new__(RelationshipTracker)
        tracker._db = MagicMock()

        rel = AgentRelationship(
            agent_a="claude",
            agent_b="grok",
            debate_count=20,
            agreement_count=2,  # Low agreement
            critique_count_a_to_b=10,
            critique_count_b_to_a=10,
            critique_accepted_a_to_b=1,
            critique_accepted_b_to_a=1,
            position_changes_a_after_b=0,
            position_changes_b_after_a=0,
            a_wins_over_b=10,
            b_wins_over_a=10,  # Balanced wins
        )

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            tracker._emit_relationship_event("claude", "grok", rel)

        data = mock_dispatch.call_args[0][1]
        assert data["relationship_type"] == "rival"
        assert data["rivalry_score"] > 0.5

    def test_classifies_neutral_relationship(self) -> None:
        tracker = RelationshipTracker.__new__(RelationshipTracker)
        tracker._db = MagicMock()

        # Too few debates for meaningful scores
        rel = AgentRelationship(
            agent_a="claude",
            agent_b="mistral",
            debate_count=1,
            agreement_count=0,
        )

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            tracker._emit_relationship_event("claude", "mistral", rel)

        data = mock_dispatch.call_args[0][1]
        assert data["relationship_type"] == "neutral"

    def test_event_type_is_correct(self) -> None:
        tracker = RelationshipTracker.__new__(RelationshipTracker)
        tracker._db = MagicMock()

        rel = AgentRelationship(agent_a="a", agent_b="b", debate_count=5, agreement_count=3)

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            tracker._emit_relationship_event("a", "b", rel)

        assert mock_dispatch.call_args[0][0] == "relationship_updated"

    def test_handles_import_error(self) -> None:
        tracker = RelationshipTracker.__new__(RelationshipTracker)
        tracker._db = MagicMock()

        rel = AgentRelationship(agent_a="a", agent_b="b")

        with patch(
            "aragora.events.dispatcher.dispatch_event",
            side_effect=ImportError("no module"),
        ):
            # Should not raise
            tracker._emit_relationship_event("a", "b", rel)

    def test_includes_influence_scores(self) -> None:
        tracker = RelationshipTracker.__new__(RelationshipTracker)
        tracker._db = MagicMock()

        rel = AgentRelationship(
            agent_a="claude",
            agent_b="gemini",
            debate_count=10,
            agreement_count=5,
            position_changes_a_after_b=3,
            position_changes_b_after_a=1,
        )

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            tracker._emit_relationship_event("claude", "gemini", rel)

        data = mock_dispatch.call_args[0][1]
        assert data["influence_a_on_b"] == 0.1  # 1/10
        assert data["influence_b_on_a"] == 0.3  # 3/10
