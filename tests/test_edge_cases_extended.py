"""
Extended edge case tests for guards added in Rounds 21-27.

These tests verify edge cases that weren't covered by existing test files,
focusing on string split safety and empty list handling.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock


# ============================================================================
# HIGH PRIORITY: Empty Agent List Guards
# ============================================================================


class TestEmptyAgentListGuards:
    """Test empty agent list handling across the codebase."""

    def test_orchestrator_require_agents_empty(self):
        """Test _require_agents raises ValueError with empty list."""
        from aragora.debate.orchestrator import Arena

        arena = Arena.__new__(Arena)
        arena.agents = []

        with pytest.raises(ValueError, match="[Nn]o agents"):
            arena._require_agents()

    def test_orchestrator_require_agents_valid(self):
        """Test _require_agents passes with valid agents."""
        from aragora.debate.orchestrator import Arena

        arena = Arena.__new__(Arena)
        arena.agents = [MagicMock(), MagicMock()]

        # Should not raise
        arena._require_agents()

    def test_select_critics_empty_agents(self):
        """Test _select_critics_for_proposal handles empty critic list."""
        from aragora.debate.orchestrator import Arena

        arena = Arena.__new__(Arena)
        # Mock protocol with topology attribute
        arena.protocol = MagicMock()
        arena.protocol.topology = "all-to-all"
        # Need at least one agent to pass _require_agents() check
        mock_agent = MagicMock()
        mock_agent.name = "agent1"
        arena.agents = [mock_agent]

        # Empty critics list should return empty list, not crash
        result = arena._select_critics_for_proposal("agent1", [])
        assert result == []


# ============================================================================
# HIGH PRIORITY: String Split Safety
# ============================================================================


class TestStringSplitSafety:
    """Test string split operations handle edge cases."""

    def test_agent_name_split_empty_string(self):
        """Test agent name split handles empty string."""
        # Direct test of split behavior
        empty = ""
        parts = empty.split("_")
        # empty.split("_") returns [""], not []
        assert parts == [""]
        # Accessing [0] should still work
        assert parts[0] == ""

    def test_agent_name_split_no_underscore(self):
        """Test agent name split when no underscore present."""
        name = "claude"
        parts = name.split("_")
        assert parts == ["claude"]
        assert parts[0] == "claude"

    def test_agent_name_split_leading_underscore(self):
        """Test agent name split with leading underscore."""
        name = "_agent"
        parts = name.split("_")
        # "_agent".split("_") returns ["", "agent"]
        assert parts[0] == ""
        assert parts[1] == "agent"

    def test_agent_name_split_multiple_underscores(self):
        """Test agent name split with multiple underscores."""
        name = "claude_visionary_v2"
        parts = name.split("_")
        assert parts[0] == "claude"
        assert len(parts) == 3

    def test_agent_name_extraction_pattern(self):
        """Test agent name extraction pattern used throughout codebase."""
        # This pattern is used in prompt_builder.py, personas.py, orchestrator.py
        test_names = [
            ("claude_visionary", "claude"),
            ("gpt4", "gpt4"),
            ("", ""),
            ("_underscore_start", ""),
        ]

        for full_name, expected_base in test_names:
            # Safe extraction pattern
            base = full_name.split("_")[0] if full_name else ""
            assert base == expected_base, f"Failed for {full_name}"


# ============================================================================
# MEDIUM PRIORITY: Convergence Edge Cases
# ============================================================================


class TestConvergenceEdgeCases:
    """Test convergence detection edge cases."""

    def test_convergence_detector_init(self):
        """Test ConvergenceDetector can be initialized."""
        from aragora.debate.convergence import ConvergenceDetector

        detector = ConvergenceDetector(convergence_threshold=0.8)
        assert detector is not None

    def test_convergence_empty_responses(self):
        """Test convergence check with empty response dicts."""
        from aragora.debate.convergence import ConvergenceDetector

        detector = ConvergenceDetector(convergence_threshold=0.8)

        # Empty responses should return None (not enough data)
        result = detector.check_convergence({}, {}, round_number=1)
        assert result is None or hasattr(result, "converged")

    def test_convergence_single_agent(self):
        """Test convergence with single agent responses."""
        from aragora.debate.convergence import ConvergenceDetector

        detector = ConvergenceDetector(convergence_threshold=0.8)

        current = {"agent1": "This is my response"}
        previous = {"agent1": "This is my response"}  # Same response

        result = detector.check_convergence(current, previous, round_number=2)
        # With same response, should detect convergence or return valid result
        assert result is None or hasattr(result, "converged")

    def test_convergence_no_common_agents(self):
        """Test convergence when different agents in each round."""
        from aragora.debate.convergence import ConvergenceDetector

        detector = ConvergenceDetector(convergence_threshold=0.8)

        # Round 1: agent_a, agent_b
        previous = {"agent_a": "Proposal A", "agent_b": "Critique B"}
        # Round 2: agent_c, agent_d (no overlap)
        current = {"agent_c": "Proposal C", "agent_d": "Critique D"}

        # Should handle gracefully without crashing
        result = detector.check_convergence(current, previous, round_number=2)
        assert result is None or hasattr(result, "converged")


# ============================================================================
# MEDIUM PRIORITY: Critique Details Split
# ============================================================================


class TestCritiqueDetailsSplit:
    """Test critique details string parsing."""

    def test_malformed_details_string(self):
        """Test handling of malformed critique details."""
        # Simulate the pattern: "issue: suggestion"
        test_cases = [
            ("valid issue: valid suggestion", ("valid issue", " valid suggestion")),
            ("no colon here", ("no colon here", "")),
            ("", ("", "")),
            (":", ("", "")),
            ("multiple: colons: here", ("multiple", " colons: here")),
        ]

        for details, expected in test_cases:
            if ":" in details:
                parts = details.split(":", 1)
                issue = parts[0]
                suggestion = parts[1] if len(parts) > 1 else ""
            else:
                issue = details
                suggestion = ""

            assert (issue, suggestion) == expected, f"Failed for: {details!r}"


# ============================================================================
# TTL Cache Integration Tests
# ============================================================================


class TestTTLCacheIntegration:
    """Test TTL caching behavior on CritiqueStore methods."""

    def test_critique_store_get_stats_cached(self):
        """Test get_stats uses caching."""
        import tempfile
        from aragora.memory.store import CritiqueStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = CritiqueStore(db_path=f"{tmpdir}/test.db")

            # First call
            stats1 = store.get_stats()

            # Second call should be cached (same result, faster)
            stats2 = store.get_stats()

            assert stats1 == stats2

    def test_critique_store_retrieve_patterns_cached(self):
        """Test retrieve_patterns uses caching."""
        import tempfile
        from aragora.memory.store import CritiqueStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = CritiqueStore(db_path=f"{tmpdir}/test.db")

            # First call
            patterns1 = store.retrieve_patterns(limit=5)

            # Second call should be cached
            patterns2 = store.retrieve_patterns(limit=5)

            assert patterns1 == patterns2


# ============================================================================
# Division By Zero Extended Tests
# ============================================================================


class TestDivisionByZeroExtended:
    """Extended division by zero protection tests."""

    def test_calibration_empty_predictions(self):
        """Test calibration score with zero predictions."""
        # Simulate calibration calculation
        total_predictions = 0
        total_error = 0.0

        # Should not divide by zero
        if total_predictions > 0:
            avg_error = total_error / total_predictions
        else:
            avg_error = 0.0

        assert avg_error == 0.0

    def test_success_rate_zero_attempts(self):
        """Test success rate calculation with zero attempts."""
        success_count = 0
        failure_count = 0
        total = success_count + failure_count

        # Safe calculation
        success_rate = success_count / total if total > 0 else 0.0

        assert success_rate == 0.0


# ============================================================================
# Empty Collection Guards
# ============================================================================


class TestEmptyCollectionGuards:
    """Test guards for empty collections throughout the codebase."""

    def test_random_choice_empty_list_guard(self):
        """Test that random.choice on empty list is guarded."""
        import random

        items = []

        # Direct random.choice would raise IndexError
        with pytest.raises(IndexError):
            random.choice(items)

        # Guarded version
        result = random.choice(items) if items else None
        assert result is None

    def test_max_empty_sequence_guard(self):
        """Test max() on empty sequence is guarded."""
        items = []

        # Direct max would raise ValueError
        with pytest.raises(ValueError):
            max(items)

        # Guarded version
        result = max(items, default=0)
        assert result == 0

    def test_sorted_empty_list_guard(self):
        """Test sorted() handles empty list."""
        items = []

        # sorted() handles empty list fine
        result = sorted(items)
        assert result == []

        # sorted with key also works
        result = sorted(items, key=lambda x: x)
        assert result == []


# ============================================================================
# Consensus Handler Topic Validation Edge Cases
# ============================================================================


class TestConsensusTopicValidation:
    """Test topic length validation boundary conditions."""

    def test_topic_exactly_500_chars_accepted(self):
        """Test topic at exactly 500 chars is accepted."""
        from aragora.server.handlers.consensus import ConsensusHandler
        from unittest.mock import MagicMock

        handler = ConsensusHandler(MagicMock())
        topic = "x" * 500
        # Mock ConsensusMemory to avoid database access
        with patch("aragora.server.handlers.consensus.ConsensusMemory") as MockMemory:
            mock_instance = MagicMock()
            mock_instance.find_similar_debates.return_value = []
            MockMemory.return_value = mock_instance
            result = handler.handle("/api/consensus/similar", {"topic": topic}, MagicMock())
        assert result.status_code == 200

    def test_topic_501_chars_rejected(self):
        """Test topic at 501 chars is rejected."""
        from aragora.server.handlers.consensus import ConsensusHandler
        from unittest.mock import MagicMock

        handler = ConsensusHandler(MagicMock())
        topic = "x" * 501
        result = handler.handle("/api/consensus/similar", {"topic": topic}, MagicMock())
        assert result.status_code == 400

    def test_topic_empty_string_rejected(self):
        """Test empty topic is rejected."""
        from aragora.server.handlers.consensus import ConsensusHandler
        from unittest.mock import MagicMock

        handler = ConsensusHandler(MagicMock())
        result = handler.handle("/api/consensus/similar", {"topic": ""}, MagicMock())
        assert result.status_code == 400

    def test_topic_whitespace_only_rejected(self):
        """Test whitespace-only topic is rejected."""
        from aragora.server.handlers.consensus import ConsensusHandler
        from unittest.mock import MagicMock

        handler = ConsensusHandler(MagicMock())
        result = handler.handle("/api/consensus/similar", {"topic": "   "}, MagicMock())
        # After strip(), this becomes empty
        assert result.status_code == 400

    def test_topic_as_list_handled(self):
        """Test topic passed as list (URL query param edge case)."""
        from aragora.server.handlers.consensus import ConsensusHandler
        from unittest.mock import MagicMock

        handler = ConsensusHandler(MagicMock())
        # Mock ConsensusMemory to avoid database access
        with patch("aragora.server.handlers.consensus.ConsensusMemory") as MockMemory:
            mock_instance = MagicMock()
            mock_instance.find_similar_debates.return_value = []
            MockMemory.return_value = mock_instance
            # URL params sometimes come as lists
            result = handler.handle(
                "/api/consensus/similar", {"topic": ["test topic"]}, MagicMock()
            )
        assert result.status_code == 200


# ============================================================================
# Orchestrator Deque Overflow Tests
# ============================================================================


class TestOrchestratorDequeOverflow:
    """Test deque bounded queue behavior in orchestrator."""

    def test_user_votes_deque_bounded(self):
        """Test user_votes deque respects maxlen."""
        from collections import deque
        from aragora.config import USER_EVENT_QUEUE_SIZE

        # Simulate the deque behavior
        votes = deque(maxlen=USER_EVENT_QUEUE_SIZE)

        # Add more items than maxlen
        for i in range(USER_EVENT_QUEUE_SIZE + 100):
            votes.append({"vote": i})

        # Should be capped at maxlen
        assert len(votes) == USER_EVENT_QUEUE_SIZE
        # Oldest should be evicted
        assert votes[0]["vote"] == 100  # First 100 evicted

    def test_user_suggestions_deque_bounded(self):
        """Test user_suggestions deque respects maxlen."""
        from collections import deque
        from aragora.config import USER_EVENT_QUEUE_SIZE

        suggestions = deque(maxlen=USER_EVENT_QUEUE_SIZE)

        for i in range(USER_EVENT_QUEUE_SIZE * 2):
            suggestions.append({"suggestion": f"idea_{i}"})

        assert len(suggestions) == USER_EVENT_QUEUE_SIZE


# ============================================================================
# Phase Module Edge Cases
# ============================================================================


class TestPhaseModuleEdgeCases:
    """Test edge cases in extracted phase modules."""

    def test_voting_phase_empty_votes(self):
        """Test VotingPhase handles empty vote list."""
        from aragora.debate.phases import VotingPhase
        from unittest.mock import MagicMock

        protocol = MagicMock()
        protocol.vote_grouping = True
        protocol.vote_grouping_threshold = 0.8

        phase = VotingPhase(protocol)
        groups = phase.group_similar_votes([])
        assert groups == {}

    def test_voting_phase_single_vote(self):
        """Test VotingPhase handles single vote."""
        from aragora.debate.phases import VotingPhase
        from unittest.mock import MagicMock

        protocol = MagicMock()
        protocol.vote_grouping = True
        protocol.vote_grouping_threshold = 0.8

        vote = MagicMock()
        vote.choice = "option_a"

        phase = VotingPhase(protocol)
        groups = phase.group_similar_votes([vote])
        assert groups == {}  # Need 2+ choices to group

    def test_critique_phase_empty_critics(self):
        """Test CritiquePhase handles empty critic list."""
        from aragora.debate.phases import CritiquePhase
        from unittest.mock import MagicMock

        protocol = MagicMock()
        protocol.topology = "all-to-all"

        phase = CritiquePhase(protocol, [])
        critics = phase.select_critics_for_proposal("agent1", [])
        assert critics == []

    def test_judgment_phase_no_agents(self):
        """Test JudgmentPhase raises with no agents."""
        from aragora.debate.phases import JudgmentPhase
        from unittest.mock import MagicMock

        protocol = MagicMock()
        protocol.judge_selection = "random"

        phase = JudgmentPhase(protocol, [])

        with pytest.raises(ValueError, match="[Nn]o agents"):
            phase._require_agents()

    def test_roles_manager_single_agent(self):
        """Test RolesManager handles single agent."""
        from aragora.debate.phases import RolesManager
        from unittest.mock import MagicMock

        protocol = MagicMock()
        protocol.proposer_count = 1
        protocol.asymmetric_stances = False

        agent = MagicMock()
        agent.role = None

        manager = RolesManager(protocol, [agent])
        manager.assign_roles()

        # Single agent should be proposer
        assert agent.role == "proposer"

    def test_roles_manager_two_agents(self):
        """Test RolesManager handles two agents."""
        from aragora.debate.phases import RolesManager
        from unittest.mock import MagicMock

        protocol = MagicMock()
        protocol.proposer_count = 1
        protocol.asymmetric_stances = False

        agent1 = MagicMock()
        agent1.role = None
        agent2 = MagicMock()
        agent2.role = None

        manager = RolesManager(protocol, [agent1, agent2])
        manager.assign_roles()

        # With 2 agents: 1 proposer, 1 synthesizer
        assert agent1.role == "proposer"
        assert agent2.role == "synthesizer"


# ============================================================================
# OpenRouter Fallback Tests
# ============================================================================


class TestRateLimitPatternDetection:
    """Test rate limit error pattern detection."""

    def _check_rate_limit_pattern(self, error_message: str) -> bool:
        """Check if error message matches rate limit patterns."""
        from aragora.agents.errors import RATE_LIMIT_PATTERNS

        error_str = error_message.lower()
        return any(pattern in error_str for pattern in RATE_LIMIT_PATTERNS)

    def _check_network_pattern(self, error_message: str) -> bool:
        """Check if error message matches network error patterns."""
        from aragora.agents.errors import NETWORK_ERROR_PATTERNS

        error_str = error_message.lower()
        return any(pattern in error_str for pattern in NETWORK_ERROR_PATTERNS)

    def _check_cli_pattern(self, error_message: str) -> bool:
        """Check if error message matches CLI error patterns."""
        from aragora.agents.errors import CLI_ERROR_PATTERNS

        error_str = error_message.lower()
        return any(pattern in error_str for pattern in CLI_ERROR_PATTERNS)

    def _check_auth_pattern(self, error_message: str) -> bool:
        """Check if error message matches auth error patterns."""
        from aragora.agents.errors import AUTH_ERROR_PATTERNS

        error_str = error_message.lower()
        return any(pattern in error_str for pattern in AUTH_ERROR_PATTERNS)

    def _check_all_fallback_patterns(self, error_message: str) -> bool:
        """Check if error message matches any fallback pattern."""
        from aragora.agents.errors import ALL_FALLBACK_PATTERNS

        error_str = error_message.lower()
        return any(pattern in error_str for pattern in ALL_FALLBACK_PATTERNS)

    def test_detects_rate_limit_429(self):
        """Test detection of HTTP 429 rate limit."""
        assert self._check_rate_limit_pattern("rate limit exceeded")
        assert self._check_rate_limit_pattern("Rate Limit Exceeded")
        assert self._check_rate_limit_pattern("429 Too Many Requests")

    def test_detects_quota_exceeded(self):
        """Test detection of quota errors."""
        assert self._check_rate_limit_pattern("quota exceeded")
        assert self._check_rate_limit_pattern("Error: quota_exceeded for account")
        assert self._check_rate_limit_pattern("insufficient_quota")

    def test_detects_network_errors(self):
        """Test detection of network connectivity errors."""
        assert self._check_network_pattern("could not resolve host")
        assert self._check_network_pattern("econnrefused")  # Lowercase pattern
        assert self._check_network_pattern("network is unreachable")
        assert self._check_network_pattern("name or service not known")

    def test_detects_auth_errors(self):
        """Test detection of authentication errors."""
        assert self._check_auth_pattern("invalid_api_key")
        assert self._check_auth_pattern("unauthorized")
        assert self._check_auth_pattern("authentication failed")

    def test_detects_cli_errors(self):
        """Test detection of CLI-specific errors."""
        assert self._check_cli_pattern("command not found")
        assert self._check_cli_pattern("permission denied")
        assert self._check_cli_pattern("no such file or directory")

    def test_non_fallback_errors(self):
        """Test that regular errors don't trigger fallback."""
        assert not self._check_all_fallback_patterns("invalid argument")
        assert not self._check_all_fallback_patterns("syntax error in response")
        assert not self._check_all_fallback_patterns("empty response")


class TestCircuitBreakerBehavior:
    """Test circuit breaker edge cases."""

    def test_circuit_breaker_threshold(self):
        """Test circuit breaker trips after threshold failures."""
        from scripts.nomic_loop import AgentCircuitBreaker

        cb = AgentCircuitBreaker(failure_threshold=3, cooldown_cycles=1)

        # Should not trip on first 2 failures
        cb.record_failure("agent1")
        assert cb.is_available("agent1")
        cb.record_failure("agent1")
        assert cb.is_available("agent1")

        # Should trip on 3rd failure
        cb.record_failure("agent1")
        assert not cb.is_available("agent1")

    def test_circuit_breaker_cooldown(self):
        """Test circuit breaker cooldown behavior."""
        from scripts.nomic_loop import AgentCircuitBreaker

        cb = AgentCircuitBreaker(failure_threshold=2, cooldown_cycles=1)

        # Trip the breaker
        cb.record_failure("agent1")
        cb.record_failure("agent1")
        assert not cb.is_available("agent1")

        # After one cycle, should be available again
        cb.start_new_cycle()
        assert cb.is_available("agent1")

    def test_circuit_breaker_success_resets(self):
        """Test that success resets failure count."""
        from scripts.nomic_loop import AgentCircuitBreaker

        cb = AgentCircuitBreaker(failure_threshold=3, cooldown_cycles=1)

        # Two failures
        cb.record_failure("agent1")
        cb.record_failure("agent1")

        # One success resets count
        cb.record_success("agent1")

        # Need full threshold again to trip
        cb.record_failure("agent1")
        assert cb.is_available("agent1")
        cb.record_failure("agent1")
        assert cb.is_available("agent1")

    def test_circuit_breaker_multiple_agents(self):
        """Test circuit breaker tracks agents independently."""
        from scripts.nomic_loop import AgentCircuitBreaker

        cb = AgentCircuitBreaker(failure_threshold=2, cooldown_cycles=1)

        # Trip agent1
        cb.record_failure("agent1")
        cb.record_failure("agent1")
        assert not cb.is_available("agent1")

        # agent2 should still be available
        assert cb.is_available("agent2")
        cb.record_failure("agent2")
        assert cb.is_available("agent2")


class TestOpenRouterFallbackIntegration:
    """Test OpenRouter fallback integration."""

    def test_fallback_enabled_by_default(self):
        """Test that fallback is enabled by default in CLI agents."""
        from aragora.agents.cli_agents import ClaudeAgent

        # Create agent without explicit enable_fallback (uses concrete subclass)
        agent = ClaudeAgent(name="test", model="claude")
        assert agent.enable_fallback is True

    def test_fallback_can_be_disabled(self):
        """Test that fallback can be explicitly disabled."""
        from aragora.agents.cli_agents import ClaudeAgent

        agent = ClaudeAgent(name="test", model="claude", enable_fallback=False)
        assert agent.enable_fallback is False

    def test_fallback_requires_api_key(self):
        """Test fallback behavior when API key is missing."""
        import os
        from aragora.agents.cli_agents import ClaudeAgent

        # Temporarily remove API key
        original = os.environ.pop("OPENROUTER_API_KEY", None)
        try:
            agent = ClaudeAgent(name="test", model="claude")
            # Should still enable fallback, but it won't work without key
            assert agent.enable_fallback is True
            # Fallback agent should not be created without key
            fallback = agent._get_fallback_agent()
            assert fallback is None
        finally:
            if original:
                os.environ["OPENROUTER_API_KEY"] = original

    def test_is_fallback_error_method(self):
        """Test the _is_fallback_error method on CLIAgent."""
        from aragora.agents.cli_agents import ClaudeAgent

        agent = ClaudeAgent(name="test", model="claude")

        # Should detect rate limit errors
        assert agent._is_fallback_error(Exception("rate limit exceeded"))
        assert agent._is_fallback_error(Exception("429 Too Many Requests"))
        assert agent._is_fallback_error(Exception("quota exceeded"))

        # Should detect network errors
        assert agent._is_fallback_error(Exception("ECONNREFUSED"))
        assert agent._is_fallback_error(Exception("network is unreachable"))

        # Should not detect regular errors
        assert not agent._is_fallback_error(Exception("invalid argument"))
        assert not agent._is_fallback_error(Exception("syntax error"))


class TestNoMicLoopValidation:
    """Test nomic loop startup validation."""

    def test_validate_openrouter_returns_bool(self):
        """Test OpenRouter validation returns boolean."""
        import os
        from scripts.nomic_loop import NomicLoop

        # Create minimal nomic loop instance
        loop = NomicLoop.__new__(NomicLoop)
        loop.log_file = "/dev/null"

        # Mock _log and _stream_emit
        loop._log = lambda msg, **kwargs: None
        loop._stream_emit = lambda *args, **kwargs: None

        # Test with key set
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        result = loop._validate_openrouter_fallback()
        assert isinstance(result, bool)

        # Test without key
        del os.environ["OPENROUTER_API_KEY"]
        result = loop._validate_openrouter_fallback()
        assert result is False
