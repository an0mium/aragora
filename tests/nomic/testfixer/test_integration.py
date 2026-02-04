"""
Integration tests for TestFixer LLM components.

Tests the interaction between:
- LLMFailureAnalyzer
- ArenaValidator
- RedTeamValidator
- PatternLearner

These tests verify the components work together correctly.
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from aragora.nomic.testfixer.analyzers.llm_analyzer import (
    LLMFailureAnalyzer,
    LLMAnalyzerConfig,
)
from aragora.nomic.testfixer.validators.arena_validator import (
    ArenaValidator,
    ArenaValidatorConfig,
)
from aragora.nomic.testfixer.validators.redteam_validator import (
    RedTeamValidator,
    RedTeamValidatorConfig,
)
from aragora.nomic.testfixer.learning.pattern_learner import (
    PatternLearner,
)
from aragora.nomic.testfixer.runner import TestFailure
from aragora.nomic.testfixer.analyzer import FailureCategory


# ---------------------------------------------------------------------------
# Mock Objects
# ---------------------------------------------------------------------------


@dataclass
class MockPatch:
    """Mock patch for testing."""

    file_path: str
    original_content: str = ""
    patched_content: str = ""


@dataclass
class MockProposal:
    """Mock proposal for testing."""

    id: str = "test-proposal-001"
    description: str = "Fix authentication flag"
    post_debate_confidence: float = 0.8
    patches: list = field(default_factory=list)

    def as_diff(self) -> str:
        return "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new"


@dataclass
class MockAnalysis:
    """Mock analysis for testing."""

    failure: TestFailure = None
    root_cause: str = "Missing authentication flag"
    fix_target: FailureCategory = FailureCategory.IMPL_BUG
    category: FailureCategory = FailureCategory.IMPL_BUG
    root_cause_file: str = "src/auth.py"


@dataclass
class MockFixAttempt:
    """Mock fix attempt for testing."""

    failure: TestFailure = None
    analysis: MockAnalysis = None
    proposal: MockProposal = None
    applied: bool = True
    success: bool = True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_test_failure() -> TestFailure:
    """Create a realistic test failure."""
    return TestFailure(
        test_name="test_user_authentication",
        test_file="tests/test_auth.py",
        error_type="AssertionError",
        error_message="assert user.is_authenticated == True, but got False",
        stack_trace="""
Traceback (most recent call last):
  File "tests/test_auth.py", line 42, in test_user_authentication
    user = User("testuser")
    user.authenticate("correct_password")
    assert user.is_authenticated == True
AssertionError: assert False == True
  +  where False = <User testuser>.is_authenticated
""",
        line_number=42,
        involved_files=["src/auth/user.py", "src/auth/session.py"],
        involved_functions=["authenticate", "verify_password"],
    )


@pytest.fixture
def sample_code_context() -> dict[str, str]:
    """Create realistic code context."""
    return {
        "src/auth/user.py": """
class User:
    def __init__(self, username: str):
        self.username = username
        self.is_authenticated = False

    def authenticate(self, password: str) -> bool:
        if self._verify_password(password):
            # BUG: Missing self.is_authenticated = True
            return True
        return False

    def _verify_password(self, password: str) -> bool:
        return password == "correct_password"
""",
        "tests/test_auth.py": """
from src.auth.user import User

def test_user_authentication():
    user = User("testuser")
    result = user.authenticate("correct_password")
    assert result == True
    assert user.is_authenticated == True  # Fails here
""",
    }


@pytest.fixture
def sample_proposal() -> MockProposal:
    """Create a realistic proposal."""
    return MockProposal(
        id="prop-001",
        description="Set is_authenticated = True in authenticate() after password verification succeeds",
        post_debate_confidence=0.9,
        patches=[
            MockPatch(
                file_path="src/auth/user.py",
                original_content="return True",
                patched_content="self.is_authenticated = True\n            return True",
            )
        ],
    )


@pytest.fixture
def sample_analysis(sample_test_failure) -> MockAnalysis:
    """Create a realistic analysis."""
    return MockAnalysis(
        failure=sample_test_failure,
        root_cause="The authenticate() method returns True but doesn't set self.is_authenticated = True",
        fix_target=FailureCategory.IMPL_BUG,
        category=FailureCategory.IMPL_BUG,
        root_cause_file="src/auth/user.py",
    )


def create_mock_agent(name: str, responses: list[str] | str):
    """Create a mock agent with configurable responses."""
    agent = MagicMock()
    agent.name = name

    if isinstance(responses, str):
        agent.generate = AsyncMock(return_value=responses)
    else:
        response_iter = iter(responses)

        async def generate(prompt):
            try:
                return next(response_iter)
            except StopIteration:
                return responses[-1]

        agent.generate = AsyncMock(side_effect=generate)

    return agent


# ---------------------------------------------------------------------------
# End-to-End Analysis Flow Tests
# ---------------------------------------------------------------------------


class TestAnalysisFlow:
    """Test the full analysis flow."""

    @pytest.mark.asyncio
    async def test_llm_analyzer_produces_actionable_output(
        self, sample_test_failure, sample_code_context
    ):
        """Test LLM analyzer produces output usable by validators."""
        with patch("aragora.nomic.testfixer.analyzers.llm_analyzer.create_agent") as mock_create:
            agent = create_mock_agent(
                "test_agent",
                """
<category>implementation_bug</category>
<root_cause>
The authenticate() method verifies the password and returns True on success,
but it never sets self.is_authenticated = True. This means the authentication
check passes but the user object's state is not updated.
</root_cause>
<approach>
1. In src/auth/user.py, locate the authenticate() method
2. Add self.is_authenticated = True before the return True statement
3. This ensures the user's authentication state is properly tracked
</approach>
<confidence>0.95</confidence>
""",
            )
            mock_create.return_value = agent

            config = LLMAnalyzerConfig(agent_types=["test"])
            analyzer = LLMFailureAnalyzer(config)

            root_cause, approach, confidence = await analyzer.analyze(
                sample_test_failure, sample_code_context
            )

            # Verify actionable output
            assert "authenticate" in root_cause.lower()
            assert "is_authenticated" in root_cause.lower()
            assert len(approach) > 50  # Substantial approach
            assert confidence >= 0.9


# ---------------------------------------------------------------------------
# Validation Pipeline Tests
# ---------------------------------------------------------------------------


class TestValidationPipeline:
    """Test the validation pipeline."""

    @pytest.mark.asyncio
    async def test_arena_then_redteam_validation(self, sample_proposal, sample_analysis):
        """Test running Arena validation followed by RedTeam validation."""
        with (
            patch("aragora.nomic.testfixer.validators.arena_validator.create_agent") as arena_mock,
            patch(
                "aragora.nomic.testfixer.validators.redteam_validator.create_agent"
            ) as redteam_mock,
        ):
            # Arena agents approve
            arena_agent = create_mock_agent(
                "arena_agent",
                """
VERDICT: APPROVE
CONFIDENCE: 0.9
REASONING: The fix correctly addresses the root cause by setting is_authenticated.
CONCERNS: None
IMPROVEMENTS: Consider adding a log statement for debugging.
""",
            )
            arena_mock.return_value = arena_agent

            # RedTeam agents find minor issues
            attacker = create_mock_agent(
                "attacker",
                """
---
ATTACK_TYPE: EDGE_CASE
DESCRIPTION: What happens if authenticate is called twice?
SEVERITY: LOW
EXPLOIT: user.authenticate(); user.authenticate() - is_authenticated still correct?
EVIDENCE: No idempotency issue, just noting the behavior.
---
""",
            )
            defender = create_mock_agent(
                "defender",
                """
---
ATTACK_ID: edge_case
DEFENSE: REFUTE
EXPLANATION: Calling authenticate twice is fine - it will just set is_authenticated = True again.
RESIDUAL_RISK: 0
---
""",
            )
            redteam_mock.side_effect = [attacker, defender]

            # Run Arena validation
            arena_config = ArenaValidatorConfig(agent_types=["test"])
            arena_validator = ArenaValidator(arena_config)
            arena_result = await arena_validator.validate(sample_proposal, sample_analysis)

            # Run RedTeam validation
            redteam_config = RedTeamValidatorConfig(
                attacker_types=["test"],
                attack_rounds=1,
            )
            redteam_validator = RedTeamValidator(redteam_config)
            redteam_result = await redteam_validator.validate(sample_proposal, sample_analysis)

            # Both should pass
            assert arena_result.is_valid is True
            assert arena_result.confidence >= 0.8
            assert redteam_result.passes is True
            assert redteam_result.robustness_score >= 0.5


# ---------------------------------------------------------------------------
# Learning Integration Tests
# ---------------------------------------------------------------------------


class TestLearningIntegration:
    """Test pattern learning integration."""

    @pytest.mark.asyncio
    async def test_learn_from_validated_fix(
        self, sample_test_failure, sample_code_context, sample_proposal, sample_analysis, tmp_path
    ):
        """Test learning from a fix that passed validation."""
        with patch(
            "aragora.nomic.testfixer.validators.arena_validator.create_agent"
        ) as mock_create:
            agent = create_mock_agent(
                "agent",
                "VERDICT: APPROVE\nCONFIDENCE: 0.9\nREASONING: Good fix",
            )
            mock_create.return_value = agent

            # Validate the fix
            validator = ArenaValidator(ArenaValidatorConfig(agent_types=["test"]))
            result = await validator.validate(sample_proposal, sample_analysis)

            assert result.is_valid is True

            # Learn from the successful fix
            learner = PatternLearner(tmp_path / "patterns.json")

            attempt = MockFixAttempt(
                failure=sample_test_failure,
                analysis=sample_analysis,
                proposal=sample_proposal,
                applied=True,
                success=True,
            )

            pattern = learner.learn_from_attempt(attempt)

            assert pattern is not None
            assert pattern.success_count == 1
            assert (
                "is_authenticated" in pattern.root_cause.lower()
                or "authentication" in pattern.error_pattern.lower()
            )

    def test_suggest_fix_from_learned_pattern(self, sample_test_failure, sample_analysis, tmp_path):
        """Test suggesting fixes based on learned patterns."""
        learner = PatternLearner(tmp_path / "patterns.json")

        # Pre-populate with a learned pattern
        from aragora.nomic.testfixer.learning.pattern_learner import FixPattern

        learner.patterns["auth-pattern"] = FixPattern(
            id="auth-pattern",
            category=sample_analysis.category.value,
            error_pattern="is_authenticated == True but got False",
            fix_pattern="Add self.is_authenticated = True after password verification",
            fix_diff="+ self.is_authenticated = True",
            fix_file="user.py",
            error_type="AssertionError",
            root_cause="Authentication flag not set",
            success_count=5,
            failure_count=0,
        )

        # Find similar patterns
        matches = learner.find_similar_patterns(sample_analysis, min_similarity=0.3)

        assert len(matches) >= 1
        assert matches[0].pattern.id == "auth-pattern"

        # Get suggestion
        suggestion = learner.suggest_heuristic(sample_analysis)

        if suggestion:  # May be None if confidence too low
            assert "is_authenticated" in suggestion or "authentication" in suggestion.lower()


# ---------------------------------------------------------------------------
# Cross-Component Communication Tests
# ---------------------------------------------------------------------------


class TestCrossComponentCommunication:
    """Test communication between components."""

    @pytest.mark.asyncio
    async def test_analysis_used_by_validators(self, sample_test_failure, sample_code_context):
        """Test that analysis output is correctly used by validators."""
        # First, run analysis
        with patch("aragora.nomic.testfixer.analyzers.llm_analyzer.create_agent") as analyzer_mock:
            analyzer_agent = create_mock_agent(
                "analyzer",
                """
<category>implementation_bug</category>
<root_cause>Missing is_authenticated = True in authenticate method</root_cause>
<approach>Add self.is_authenticated = True before return True</approach>
<confidence>0.9</confidence>
""",
            )
            analyzer_mock.return_value = analyzer_agent

            analyzer = LLMFailureAnalyzer(LLMAnalyzerConfig(agent_types=["test"]))
            root_cause, approach, confidence = await analyzer.analyze(
                sample_test_failure, sample_code_context
            )

        # Create analysis and proposal based on LLM output
        analysis = MockAnalysis(
            failure=sample_test_failure,
            root_cause=root_cause,
            category=FailureCategory.IMPL_BUG,
            root_cause_file="src/auth/user.py",
        )

        proposal = MockProposal(
            id="prop-from-analysis",
            description=approach[:200],
            post_debate_confidence=confidence,
            patches=[
                MockPatch(
                    file_path="src/auth/user.py",
                    patched_content="self.is_authenticated = True\nreturn True",
                )
            ],
        )

        # Now validate using Arena
        with patch(
            "aragora.nomic.testfixer.validators.arena_validator.create_agent"
        ) as validator_mock:
            validator_agent = create_mock_agent(
                "validator",
                f"""
VERDICT: APPROVE
CONFIDENCE: 0.85
REASONING: The proposed fix addresses the root cause: {root_cause[:100]}
CONCERNS: None
IMPROVEMENTS: None
""",
            )
            validator_mock.return_value = validator_agent

            validator = ArenaValidator(ArenaValidatorConfig(agent_types=["test"]))
            result = await validator.validate(proposal, analysis)

            assert result.is_valid is True


# ---------------------------------------------------------------------------
# Error Recovery Tests
# ---------------------------------------------------------------------------


class TestErrorRecovery:
    """Test error recovery across components."""

    @pytest.mark.asyncio
    async def test_validator_fallback_on_analyzer_failure(
        self, sample_test_failure, sample_proposal, sample_analysis
    ):
        """Test that validation can proceed even if analysis was partial."""
        # Simulate a partial analysis (low confidence)
        sample_analysis.root_cause = "Unknown - analysis failed"

        with patch(
            "aragora.nomic.testfixer.validators.arena_validator.create_agent"
        ) as mock_create:
            # Validator should still be able to evaluate the proposal
            agent = create_mock_agent(
                "agent",
                """
VERDICT: APPROVE
CONFIDENCE: 0.6
REASONING: The fix looks correct based on code review, but analysis was unclear.
CONCERNS: Would benefit from clearer root cause analysis.
IMPROVEMENTS: Add more context to the commit message.
""",
            )
            mock_create.return_value = agent

            validator = ArenaValidator(ArenaValidatorConfig(agent_types=["test"]))
            result = await validator.validate(sample_proposal, sample_analysis)

            # Should still produce a result, even with uncertain analysis
            assert result is not None
            assert 0 <= result.confidence <= 1

    @pytest.mark.asyncio
    async def test_pattern_learner_handles_validation_failure(
        self, sample_test_failure, sample_analysis, sample_proposal, tmp_path
    ):
        """Test pattern learner handles failed validation correctly."""
        learner = PatternLearner(tmp_path / "patterns.json")

        # First, learn a successful pattern
        success_attempt = MockFixAttempt(
            failure=sample_test_failure,
            analysis=sample_analysis,
            proposal=sample_proposal,
            applied=True,
            success=True,
        )
        pattern = learner.learn_from_attempt(success_attempt)
        assert pattern.success_count == 1
        assert pattern.failure_count == 0

        # Now learn from a failed attempt
        fail_attempt = MockFixAttempt(
            failure=sample_test_failure,
            analysis=sample_analysis,
            proposal=sample_proposal,
            applied=True,
            success=False,  # Failed
        )
        pattern = learner.learn_from_attempt(fail_attempt)

        # Pattern should be updated with failure
        assert pattern.success_count == 1
        assert pattern.failure_count == 1
        assert pattern.confidence == 0.5  # 1/2


# ---------------------------------------------------------------------------
# Performance and Timeout Tests
# ---------------------------------------------------------------------------


class TestPerformanceAndTimeouts:
    """Test performance and timeout handling."""

    @pytest.mark.asyncio
    async def test_parallel_validation(self, sample_proposal, sample_analysis):
        """Test running Arena and RedTeam validation in parallel."""
        with (
            patch("aragora.nomic.testfixer.validators.arena_validator.create_agent") as arena_mock,
            patch(
                "aragora.nomic.testfixer.validators.redteam_validator.create_agent"
            ) as redteam_mock,
        ):
            arena_agent = create_mock_agent("arena", "VERDICT: APPROVE\nCONFIDENCE: 0.8")
            arena_mock.return_value = arena_agent

            attacker = create_mock_agent(
                "attacker",
                "---\nATTACK_TYPE: EDGE_CASE\nDESCRIPTION: Test\nSEVERITY: LOW\nEXPLOIT: Test\n---",
            )
            redteam_mock.return_value = attacker

            arena_validator = ArenaValidator(ArenaValidatorConfig(agent_types=["test"]))
            redteam_validator = RedTeamValidator(
                RedTeamValidatorConfig(
                    attacker_types=["test"],
                    allow_defense=False,
                    attack_rounds=1,
                )
            )

            # Run in parallel
            arena_result, redteam_result = await asyncio.gather(
                arena_validator.validate(sample_proposal, sample_analysis),
                redteam_validator.validate(sample_proposal, sample_analysis),
            )

            assert arena_result is not None
            assert redteam_result is not None

    @pytest.mark.asyncio
    async def test_timeout_does_not_block_other_components(self, sample_proposal, sample_analysis):
        """Test that one component timeout doesn't block others."""
        with (
            patch("aragora.nomic.testfixer.validators.arena_validator.create_agent") as arena_mock,
            patch(
                "aragora.nomic.testfixer.validators.redteam_validator.create_agent"
            ) as redteam_mock,
        ):
            # Arena times out
            async def slow_arena(prompt):
                await asyncio.sleep(10)
                return "response"

            slow_agent = MagicMock()
            slow_agent.name = "slow"
            slow_agent.generate = AsyncMock(side_effect=slow_arena)
            arena_mock.return_value = slow_agent

            # RedTeam is fast
            fast_agent = create_mock_agent("fast", "No issues found")
            redteam_mock.return_value = fast_agent

            arena_validator = ArenaValidator(
                ArenaValidatorConfig(agent_types=["test"], agent_timeout=0.1)
            )
            redteam_validator = RedTeamValidator(
                RedTeamValidatorConfig(
                    attacker_types=["test"],
                    allow_defense=False,
                    attack_rounds=1,
                )
            )

            # Run with individual timeouts
            arena_task = asyncio.create_task(
                arena_validator.validate(sample_proposal, sample_analysis)
            )
            redteam_task = asyncio.create_task(
                redteam_validator.validate(sample_proposal, sample_analysis)
            )

            # Wait for both with overall timeout
            results = await asyncio.wait_for(
                asyncio.gather(arena_task, redteam_task),
                timeout=5.0,
            )

            # Both should complete (arena with error, redteam successfully)
            assert len(results) == 2


# ---------------------------------------------------------------------------
# Configuration Compatibility Tests
# ---------------------------------------------------------------------------


class TestConfigurationCompatibility:
    """Test configuration compatibility between components."""

    def test_consistent_agent_types(self):
        """Test that agent types are consistent across configs."""
        analyzer_config = LLMAnalyzerConfig()
        arena_config = ArenaValidatorConfig()
        redteam_config = RedTeamValidatorConfig()

        # All should support common agent types
        common_types = {"anthropic-api", "openai-api"}

        assert common_types.issubset(set(analyzer_config.agent_types))
        assert common_types.issubset(set(arena_config.agent_types))
        assert common_types.issubset(set(redteam_config.attacker_types))

    def test_timeout_configurations(self):
        """Test timeout configurations are reasonable."""
        analyzer_config = LLMAnalyzerConfig()
        arena_config = ArenaValidatorConfig()
        redteam_config = RedTeamValidatorConfig()

        # All should have reasonable timeouts (30-120 seconds for agents)
        assert 30 <= analyzer_config.agent_timeout <= 180
        assert 30 <= arena_config.agent_timeout <= 180
        assert 30 <= redteam_config.agent_timeout <= 180
