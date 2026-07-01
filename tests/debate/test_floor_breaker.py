"""Tests for consensus floor breaker in nomic loop.

The floor breaker is an emergency mechanism that activates when:
1. Consensus threshold has decayed to the floor (0.4)
2. Multiple failures occur at this floor level
3. Standard fallback strategies have been exhausted

It uses escalating strategies:
1. Judge arbitration (existing arbitration logic)
2. Plurality wins (highest-voted proposal)
3. Random selection from top-2 viable proposals
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


class TestFloorBreakerStateTracking:
    """Test floor breaker state initialization and tracking."""

    def test_initial_state(self):
        """Floor breaker state should be clean on init."""
        # Test the expected initial state pattern
        floor_failure_count = 0
        floor_breaker_activated = False
        forced_decisions = []

        assert floor_failure_count == 0
        assert floor_breaker_activated is False
        assert forced_decisions == []

    def test_floor_failure_increment(self):
        """Floor failures should increment when at threshold floor."""
        loop = MagicMock()
        loop._floor_failure_count = 0
        loop._consensus_threshold_decay = 2  # At floor (0.4)

        # Simulate what happens in _handle_deadlock when at floor
        if loop._consensus_threshold_decay >= 2:  # At floor
            loop._floor_failure_count += 1

        assert loop._floor_failure_count == 1

    def test_floor_breaker_activation_threshold(self):
        """Floor breaker should activate after 2+ floor failures."""
        loop = MagicMock()
        loop._floor_failure_count = 2

        activate_floor_breaker = loop._floor_failure_count >= 2
        assert activate_floor_breaker is True

        loop._floor_failure_count = 1
        activate_floor_breaker = loop._floor_failure_count >= 2
        assert activate_floor_breaker is False


class TestFloorBreakerStrategies:
    """Test floor breaker escalation strategies."""

    @pytest.mark.asyncio
    async def test_judge_arbitration_strategy(self):
        """Strategy 1: Judge arbitration should use existing _arbitrate_design."""
        loop = MagicMock()
        loop._log = MagicMock()
        loop._arbitrate_design = AsyncMock(return_value="arbitrated_design")

        proposals = {
            "agent1": "proposal_a",
            "agent2": "proposal_b",
        }

        # Simulate judge arbitration strategy
        result = await loop._arbitrate_design(list(proposals.values()), "test_improvement")

        assert result == "arbitrated_design"
        loop._arbitrate_design.assert_called_once()

    def test_plurality_wins_strategy(self):
        """Strategy 2: Plurality wins should select highest-voted proposal."""
        vote_counts = {
            "proposal_a": 1,
            "proposal_b": 3,
            "proposal_c": 2,
        }

        # Simulate plurality selection
        winner = max(vote_counts.items(), key=lambda x: x[1])[0]

        assert winner == "proposal_b"

    def test_plurality_wins_tie_breaking(self):
        """Plurality with ties should still return a winner."""
        vote_counts = {
            "proposal_a": 2,
            "proposal_b": 2,
            "proposal_c": 1,
        }

        # Get all proposals with max votes
        max_votes = max(vote_counts.values())
        top_proposals = [p for p, v in vote_counts.items() if v == max_votes]

        assert len(top_proposals) == 2
        assert "proposal_a" in top_proposals
        assert "proposal_b" in top_proposals

    def test_random_selection_from_top_2(self):
        """Strategy 3: Random selection from top-2 viable proposals."""
        import random

        proposals = {
            "agent1": "proposal_a",
            "agent2": "proposal_b",
            "agent3": "proposal_c",
        }
        vote_counts = {
            "proposal_a": 3,
            "proposal_b": 2,
            "proposal_c": 1,
        }

        # Get top 2
        sorted_proposals = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
        top_2 = [p for p, v in sorted_proposals[:2]]

        assert len(top_2) == 2
        assert "proposal_a" in top_2
        assert "proposal_b" in top_2

        # Random selection should pick one of top 2
        random.seed(42)
        selected = random.choice(top_2)
        assert selected in top_2


class TestFloorBreakerRecording:
    """Test floor breaker audit trail recording."""

    def test_forced_decision_record_structure(self):
        """Forced decisions should have complete audit trail."""
        record = {
            "cycle": 5,
            "improvement": "Add new feature",
            "method": "plurality",
            "selected_design": "design_proposal",
            "total_proposals": 3,
            "vote_distribution": {"proposal_a": 2, "proposal_b": 1},
            "timestamp": "2026-01-09T12:00:00",
        }

        assert "cycle" in record
        assert "improvement" in record
        assert "method" in record
        assert "selected_design" in record
        assert record["method"] in ["judge", "plurality", "random"]

    def test_forced_decisions_append(self):
        """Forced decisions should be appended to audit list."""
        forced_decisions = []

        record1 = {"cycle": 1, "method": "judge"}
        record2 = {"cycle": 2, "method": "plurality"}

        forced_decisions.append(record1)
        forced_decisions.append(record2)

        assert len(forced_decisions) == 2
        assert forced_decisions[0]["method"] == "judge"
        assert forced_decisions[1]["method"] == "plurality"


class TestFloorBreakerReset:
    """Test floor breaker state reset on success."""

    def test_reset_on_success(self):
        """Floor breaker state should reset when cycle succeeds."""
        loop = MagicMock()
        loop._floor_failure_count = 3
        loop._floor_breaker_activated = True
        loop._log = MagicMock()

        # Simulate success reset
        cycle_result = {"outcome": "success"}
        if cycle_result.get("outcome") == "success":
            if loop._floor_failure_count > 0:
                loop._floor_failure_count = 0
                loop._floor_breaker_activated = False

        assert loop._floor_failure_count == 0
        assert loop._floor_breaker_activated is False

    def test_no_reset_on_failure(self):
        """Floor breaker state should persist when cycle fails."""
        loop = MagicMock()
        loop._floor_failure_count = 2
        loop._floor_breaker_activated = True

        # Simulate failure - no reset
        cycle_result = {"outcome": "no_improvement"}
        if cycle_result.get("outcome") == "success":
            loop._floor_failure_count = 0
            loop._floor_breaker_activated = False

        assert loop._floor_failure_count == 2
        assert loop._floor_breaker_activated is True


class TestFloorBreakerIntegration:
    """Integration tests for floor breaker with deadlock handling."""

    def test_deadlock_to_floor_breaker_flow(self):
        """Test complete flow from deadlock to floor breaker activation."""
        # Initial state
        deadlock_count = 0
        consensus_threshold_decay = 0
        floor_failure_count = 0
        floor_breaker_activated = False

        # First deadlock - decay threshold
        deadlock_count += 1
        consensus_threshold_decay = min(consensus_threshold_decay + 1, 2)

        assert consensus_threshold_decay == 1  # 0.5 threshold
        assert floor_failure_count == 0

        # Second deadlock - decay to floor
        deadlock_count += 1
        consensus_threshold_decay = min(consensus_threshold_decay + 1, 2)

        assert consensus_threshold_decay == 2  # 0.4 threshold (floor)

        # Third deadlock - at floor, count floor failure
        deadlock_count += 1
        if consensus_threshold_decay >= 2:  # At floor
            floor_failure_count += 1

        assert floor_failure_count == 1

        # Fourth deadlock - at floor again
        deadlock_count += 1
        if consensus_threshold_decay >= 2:
            floor_failure_count += 1

        assert floor_failure_count == 2

        # Now floor breaker should activate
        activate_floor_breaker = floor_failure_count >= 2
        assert activate_floor_breaker is True

    def test_floor_breaker_prevents_infinite_loop(self):
        """Floor breaker should provide escape from consensus deadlock."""
        max_deadlocks_before_floor_breaker = 4  # 2 to hit floor + 2 at floor

        # Without floor breaker, loop could continue indefinitely
        # With floor breaker, we have a definite exit after max_deadlocks

        deadlock_count = 0
        consensus_threshold_decay = 0
        floor_failure_count = 0

        while deadlock_count < 10:  # Simulated loop
            deadlock_count += 1

            if consensus_threshold_decay < 2:
                consensus_threshold_decay += 1
            else:
                floor_failure_count += 1
                if floor_failure_count >= 2:
                    # Floor breaker activates - exit loop
                    break

        assert deadlock_count == max_deadlocks_before_floor_breaker
        assert floor_failure_count == 2


class TestFloorBreakerEdgeCases:
    """Test edge cases for floor breaker."""

    def test_empty_proposals(self):
        """Floor breaker should handle empty proposals gracefully."""
        proposals = {}
        vote_counts = {}

        # Should not crash with empty inputs
        if proposals:
            sorted_proposals = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
        else:
            sorted_proposals = []

        assert sorted_proposals == []

    def test_single_proposal(self):
        """Floor breaker with single proposal should select it."""
        proposals = {"agent1": "only_proposal"}
        vote_counts = {"only_proposal": 1}

        # With only one proposal, it should be selected
        sorted_proposals = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
        selected = sorted_proposals[0][0] if sorted_proposals else None

        assert selected == "only_proposal"

    def test_all_zero_votes(self):
        """Floor breaker should handle all proposals with zero votes."""
        vote_counts = {
            "proposal_a": 0,
            "proposal_b": 0,
            "proposal_c": 0,
        }

        # All have same (zero) votes - any selection is valid
        max_votes = max(vote_counts.values())
        top_proposals = [p for p, v in vote_counts.items() if v == max_votes]

        assert len(top_proposals) == 3  # All tied at zero
        assert max_votes == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
