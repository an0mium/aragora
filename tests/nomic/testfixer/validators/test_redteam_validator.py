"""
Comprehensive tests for RedTeamValidator.

Tests the adversarial red team validation including:
- Configuration defaults and custom settings
- Attack prompt building
- Defense prompt building
- Attack parsing from agent responses
- Defense parsing
- Attack rounds execution
- Defense rounds execution
- Full validation flow
- Robustness scoring
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from aragora.nomic.testfixer.validators.redteam_validator import (
    RedTeamValidator,
    RedTeamValidatorConfig,
    RedTeamResult,
    CodeAttack,
    CodeDefense,
    CodeAttackType,
)
from aragora.nomic.testfixer.runner import TestFailure
from aragora.nomic.testfixer.analyzer import FailureCategory


# ---------------------------------------------------------------------------
# Mock Objects for Testing
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


@dataclass
class MockAnalysis:
    """Mock analysis for testing."""

    failure: TestFailure = None
    root_cause: str = "Missing authentication flag"
    fix_target: FailureCategory = FailureCategory.IMPL_BUG
    category: FailureCategory = FailureCategory.IMPL_BUG
    root_cause_file: str = "src/auth.py"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_test_failure() -> TestFailure:
    """Create a sample test failure."""
    return TestFailure(
        test_name="test_auth",
        test_file="tests/test_auth.py",
        error_type="AssertionError",
        error_message="Expected True but got False",
        stack_trace="File test_auth.py line 10\nAssertionError",
        line_number=10,
    )


@pytest.fixture
def sample_proposal() -> MockProposal:
    """Create a sample proposal."""
    return MockProposal(
        id="prop-001",
        description="Add is_authenticated = True after password check",
        post_debate_confidence=0.85,
        patches=[
            MockPatch(
                file_path="src/auth.py",
                original_content="def authenticate(self):\n    return True",
                patched_content="def authenticate(self):\n    self.is_authenticated = True\n    return True",
            )
        ],
    )


@pytest.fixture
def sample_analysis(sample_test_failure) -> MockAnalysis:
    """Create a sample analysis."""
    return MockAnalysis(
        failure=sample_test_failure,
        root_cause="authenticate() doesn't set is_authenticated flag",
        fix_target=FailureCategory.IMPL_BUG,
    )


@pytest.fixture
def mock_attacker() -> MagicMock:
    """Create a mock attacker agent."""
    agent = MagicMock()
    agent.name = "mock_attacker"
    agent.generate = AsyncMock()
    return agent


@pytest.fixture
def mock_defender() -> MagicMock:
    """Create a mock defender agent."""
    agent = MagicMock()
    agent.name = "mock_defender"
    agent.generate = AsyncMock()
    return agent


# ---------------------------------------------------------------------------
# CodeAttackType Tests
# ---------------------------------------------------------------------------


class TestCodeAttackType:
    """Tests for CodeAttackType enum."""

    def test_all_attack_types(self):
        """Test all attack types exist."""
        assert CodeAttackType.REGRESSION.value == "regression"
        assert CodeAttackType.EDGE_CASE.value == "edge_case"
        assert CodeAttackType.INCOMPLETE_FIX.value == "incomplete_fix"
        assert CodeAttackType.NEW_BUG.value == "new_bug"
        assert CodeAttackType.SIDE_EFFECT.value == "side_effect"
        assert CodeAttackType.TYPE_ERROR.value == "type_error"
        assert CodeAttackType.CONCURRENCY.value == "concurrency"
        assert CodeAttackType.SECURITY.value == "security"
        assert CodeAttackType.PERFORMANCE.value == "performance"
        assert CodeAttackType.API_BREAK.value == "api_break"


# ---------------------------------------------------------------------------
# CodeAttack and CodeDefense Tests
# ---------------------------------------------------------------------------


class TestCodeAttack:
    """Tests for CodeAttack dataclass."""

    def test_basic_creation(self):
        """Test creating a CodeAttack."""
        attack = CodeAttack(
            id="attack-001",
            attack_type=CodeAttackType.REGRESSION,
            attacker="agent1",
            description="This fix breaks existing tests",
            severity=0.8,
            exploit="Call authenticate with invalid password",
        )
        assert attack.id == "attack-001"
        assert attack.attack_type == CodeAttackType.REGRESSION
        assert attack.severity == 0.8
        assert attack.evidence == ""  # Default

    def test_with_evidence(self):
        """Test attack with evidence."""
        attack = CodeAttack(
            id="attack-002",
            attack_type=CodeAttackType.EDGE_CASE,
            attacker="agent2",
            description="Empty password not handled",
            severity=0.6,
            exploit="authenticate('')",
            evidence="if password == '': return False  # Missing",
        )
        assert attack.evidence != ""


class TestCodeDefense:
    """Tests for CodeDefense dataclass."""

    def test_refute_defense(self):
        """Test refute defense."""
        defense = CodeDefense(
            attack_id="attack-001",
            defender="defender1",
            defense_type="refute",
            explanation="The test already covers this case",
            success=True,
            residual_risk=0.0,
        )
        assert defense.success is True
        assert defense.residual_risk == 0.0

    def test_mitigate_defense(self):
        """Test mitigate defense."""
        defense = CodeDefense(
            attack_id="attack-002",
            defender="defender1",
            defense_type="mitigate",
            explanation="Added null check",
            success=False,
            residual_risk=0.2,
        )
        assert defense.success is False
        assert defense.residual_risk == 0.2


# ---------------------------------------------------------------------------
# RedTeamResult Tests
# ---------------------------------------------------------------------------


class TestRedTeamResult:
    """Tests for RedTeamResult dataclass."""

    def test_basic_creation(self):
        """Test creating a result."""
        result = RedTeamResult(
            passes=True,
            robustness_score=0.8,
            confidence=0.85,
            total_attacks=5,
            successful_attacks=1,
            mitigated_attacks=2,
            accepted_risks=1,
        )
        assert result.passes is True
        assert result.robustness_score == 0.8
        assert result.total_attacks == 5

    def test_default_fields(self):
        """Test default field values."""
        result = RedTeamResult(
            passes=False,
            robustness_score=0.5,
            confidence=0.6,
            total_attacks=3,
            successful_attacks=2,
            mitigated_attacks=0,
            accepted_risks=0,
        )
        assert result.critical_issues == []
        assert result.warnings == []
        assert result.suggestions == []

    def test_summary_pass(self):
        """Test summary for passing result."""
        result = RedTeamResult(
            passes=True,
            robustness_score=0.9,
            confidence=0.85,
            total_attacks=5,
            successful_attacks=0,
            mitigated_attacks=3,
            accepted_risks=0,
            critical_issues=[],
        )
        summary = result.summary()

        assert "PASS" in summary
        assert "90%" in summary
        assert "0/5" in summary
        assert "0 critical" in summary

    def test_summary_fail(self):
        """Test summary for failing result."""
        result = RedTeamResult(
            passes=False,
            robustness_score=0.3,
            confidence=0.4,
            total_attacks=5,
            successful_attacks=3,
            mitigated_attacks=1,
            accepted_risks=0,
            critical_issues=[
                CodeAttack(
                    id="1",
                    attack_type=CodeAttackType.SECURITY,
                    attacker="a",
                    description="SQL injection",
                    severity=1.0,
                    exploit="x",
                )
            ],
        )
        summary = result.summary()

        assert "FAIL" in summary
        assert "30%" in summary
        assert "3/5" in summary
        assert "1 critical" in summary


# ---------------------------------------------------------------------------
# RedTeamValidatorConfig Tests
# ---------------------------------------------------------------------------


class TestRedTeamValidatorConfig:
    """Tests for configuration."""

    def test_default_values(self):
        """Test default config values."""
        config = RedTeamValidatorConfig()

        assert "anthropic-api" in config.attacker_types
        assert "openai-api" in config.attacker_types
        assert config.defender_type == "anthropic-api"
        assert config.attack_rounds == 2
        assert config.attacks_per_round == 3
        assert config.defend_rounds == 1
        assert config.allow_defense is True
        assert config.max_critical_issues == 0
        assert config.max_successful_attacks == 2
        assert config.min_robustness_score == 0.6

    def test_default_attack_types(self):
        """Test default attack types."""
        config = RedTeamValidatorConfig()

        assert CodeAttackType.REGRESSION in config.attack_types
        assert CodeAttackType.EDGE_CASE in config.attack_types
        assert CodeAttackType.INCOMPLETE_FIX in config.attack_types

    def test_custom_values(self):
        """Test custom configuration."""
        config = RedTeamValidatorConfig(
            attacker_types=["openai-api"],
            defender_type="gemini-api",
            attack_rounds=3,
            max_critical_issues=1,
            allow_defense=False,
        )

        assert len(config.attacker_types) == 1
        assert config.defender_type == "gemini-api"
        assert config.attack_rounds == 3
        assert config.allow_defense is False


# ---------------------------------------------------------------------------
# Attack Parsing Tests
# ---------------------------------------------------------------------------


class TestAttackParsing:
    """Tests for parsing attack responses."""

    def test_parse_single_attack(self):
        """Test parsing a single attack."""
        validator = RedTeamValidator()
        response = """
---
ATTACK_TYPE: REGRESSION
DESCRIPTION: The fix will break the login flow for existing users.
SEVERITY: HIGH
EXPLOIT: Call authenticate() after session timeout, authentication flag persists incorrectly.
EVIDENCE: test_session_timeout() will fail
---
"""
        attacks = validator._parse_attacks(response, "attacker1")

        assert len(attacks) == 1
        assert attacks[0].attack_type == CodeAttackType.REGRESSION
        assert "login flow" in attacks[0].description
        assert attacks[0].severity == 0.8  # HIGH = 0.8
        assert attacks[0].attacker == "attacker1"

    def test_parse_multiple_attacks(self):
        """Test parsing multiple attacks."""
        validator = RedTeamValidator()
        response = """
---
ATTACK_TYPE: EDGE_CASE
DESCRIPTION: Empty password not handled
SEVERITY: MEDIUM
EXPLOIT: authenticate('')
EVIDENCE: No null check
---

---
ATTACK_TYPE: SECURITY
DESCRIPTION: Password logged in plaintext
SEVERITY: CRITICAL
EXPLOIT: Check logs after authentication
EVIDENCE: logger.info(f"Password: {password}")
---
"""
        attacks = validator._parse_attacks(response, "attacker1")

        assert len(attacks) == 2
        assert attacks[0].attack_type == CodeAttackType.EDGE_CASE
        assert attacks[0].severity == 0.5  # MEDIUM
        assert attacks[1].attack_type == CodeAttackType.SECURITY
        assert attacks[1].severity == 1.0  # CRITICAL

    def test_parse_severity_levels(self):
        """Test parsing different severity levels."""
        validator = RedTeamValidator()

        severities = {
            "CRITICAL": 1.0,
            "HIGH": 0.8,
            "MEDIUM": 0.5,
            "LOW": 0.2,
        }

        for level, expected in severities.items():
            response = f"""
---
ATTACK_TYPE: NEW_BUG
DESCRIPTION: Test
SEVERITY: {level}
EXPLOIT: Test
---
"""
            attacks = validator._parse_attacks(response, "test")
            assert attacks[0].severity == expected, f"Failed for {level}"

    def test_parse_unknown_attack_type(self):
        """Test parsing unknown attack type defaults to NEW_BUG."""
        validator = RedTeamValidator()
        response = """
---
ATTACK_TYPE: UNKNOWN_TYPE
DESCRIPTION: Some issue
SEVERITY: HIGH
EXPLOIT: Something
---
"""
        attacks = validator._parse_attacks(response, "attacker")

        assert len(attacks) == 1
        assert attacks[0].attack_type == CodeAttackType.NEW_BUG

    def test_parse_missing_required_fields(self):
        """Test parsing skips attacks with missing required fields."""
        validator = RedTeamValidator()
        response = """
---
ATTACK_TYPE: REGRESSION
---

---
DESCRIPTION: No attack type
---

---
ATTACK_TYPE: EDGE_CASE
DESCRIPTION: Valid attack
SEVERITY: LOW
EXPLOIT: Test
---
"""
        attacks = validator._parse_attacks(response, "attacker")

        # Only the valid attack should be parsed
        assert len(attacks) == 1
        assert attacks[0].attack_type == CodeAttackType.EDGE_CASE


# ---------------------------------------------------------------------------
# Defense Parsing Tests
# ---------------------------------------------------------------------------


class TestDefenseParsing:
    """Tests for parsing defense responses."""

    def test_parse_refute_defense(self):
        """Test parsing REFUTE defense."""
        validator = RedTeamValidator()

        attacks = [
            CodeAttack(
                id="attack-001",
                attack_type=CodeAttackType.REGRESSION,
                attacker="a",
                description="Login breaks",
                severity=0.8,
                exploit="Test",
            )
        ]

        response = """
---
ATTACK_ID: regression
DEFENSE: REFUTE
EXPLANATION: The existing tests cover this case and pass.
RESIDUAL_RISK: 0
---
"""
        defenses = validator._parse_defenses(response, "defender", attacks)

        assert len(defenses) == 1
        assert defenses[0].defense_type == "refute"
        assert defenses[0].success is True
        assert defenses[0].residual_risk == 0.0

    def test_parse_mitigate_defense(self):
        """Test parsing MITIGATE defense."""
        validator = RedTeamValidator()

        attacks = [
            CodeAttack(
                id="attack-001",
                attack_type=CodeAttackType.EDGE_CASE,
                attacker="a",
                description="Empty string",
                severity=0.5,
                exploit="Test",
            )
        ]

        response = """
---
ATTACK_ID: edge_case
DEFENSE: MITIGATE
EXPLANATION: Added validation for empty strings.
RESIDUAL_RISK: 10
---
"""
        defenses = validator._parse_defenses(response, "defender", attacks)

        assert len(defenses) == 1
        assert defenses[0].defense_type == "mitigate"
        assert defenses[0].success is False
        assert defenses[0].residual_risk == 0.1

    def test_parse_accept_defense(self):
        """Test parsing ACCEPT defense."""
        validator = RedTeamValidator()

        attacks = [
            CodeAttack(
                id="attack-001",
                attack_type=CodeAttackType.PERFORMANCE,
                attacker="a",
                description="Slower",
                severity=0.3,
                exploit="Test",
            )
        ]

        response = """
---
ATTACK_ID: performance
DEFENSE: ACCEPT
EXPLANATION: Performance impact is acceptable for correctness.
RESIDUAL_RISK: 30
---
"""
        defenses = validator._parse_defenses(response, "defender", attacks)

        assert len(defenses) == 1
        assert defenses[0].defense_type == "accept"
        assert defenses[0].residual_risk == 0.3


# ---------------------------------------------------------------------------
# Prompt Building Tests
# ---------------------------------------------------------------------------


class TestPromptBuilding:
    """Tests for prompt building."""

    def test_build_attack_prompt(self, sample_proposal, sample_analysis):
        """Test attack prompt building."""
        validator = RedTeamValidator()
        prompt = validator._build_attack_prompt(
            sample_proposal,
            sample_analysis,
            [CodeAttackType.REGRESSION, CodeAttackType.EDGE_CASE],
            round_num=1,
        )

        assert "RED TEAM ATTACK MISSION" in prompt
        assert "test_auth" in prompt
        assert "REGRESSION" in prompt
        assert "EDGE CASE" in prompt
        assert "Round 1" in prompt

    def test_build_defense_prompt(self, sample_proposal):
        """Test defense prompt building."""
        validator = RedTeamValidator()

        attacks = [
            CodeAttack(
                id="1",
                attack_type=CodeAttackType.REGRESSION,
                attacker="a",
                description="Breaks login",
                severity=0.8,
                exploit="Test after timeout",
            )
        ]

        prompt = validator._build_defense_prompt(sample_proposal, attacks, round_num=1)

        assert "DEFENSE MISSION" in prompt
        assert "Breaks login" in prompt
        assert "80%" in prompt
        assert "Round 1" in prompt


# ---------------------------------------------------------------------------
# Full Validation Flow Tests
# ---------------------------------------------------------------------------


class TestValidationFlow:
    """Tests for complete validation workflow."""

    @pytest.mark.asyncio
    async def test_validate_no_attackers_raises(self, sample_proposal, sample_analysis):
        """Test validation raises when no attackers available."""
        with patch(
            "aragora.nomic.testfixer.validators.redteam_validator.create_agent"
        ) as mock_create:
            mock_create.side_effect = Exception("No API key")

            validator = RedTeamValidator()

            with pytest.raises(RuntimeError, match="No attacker agents"):
                await validator.validate(sample_proposal, sample_analysis)

    @pytest.mark.asyncio
    async def test_validate_no_attacks_found(self, sample_proposal, sample_analysis):
        """Test validation with no attacks found."""
        with patch(
            "aragora.nomic.testfixer.validators.redteam_validator.create_agent"
        ) as mock_create:
            mock_agent = MagicMock()
            mock_agent.name = "attacker"
            mock_agent.generate = AsyncMock(return_value="No issues found")
            mock_create.return_value = mock_agent

            config = RedTeamValidatorConfig(
                attacker_types=["test"],
                allow_defense=False,
            )
            validator = RedTeamValidator(config)

            result = await validator.validate(sample_proposal, sample_analysis)

            # No attacks = perfect robustness
            assert result.passes is True
            assert result.robustness_score == 1.0
            assert result.total_attacks == 0

    @pytest.mark.asyncio
    async def test_validate_all_attacks_defended(self, sample_proposal, sample_analysis):
        """Test validation with all attacks successfully defended."""
        with patch(
            "aragora.nomic.testfixer.validators.redteam_validator.create_agent"
        ) as mock_create:
            attacker = MagicMock()
            attacker.name = "attacker"
            attacker.generate = AsyncMock(
                return_value="""
---
ATTACK_TYPE: REGRESSION
DESCRIPTION: Breaks login
SEVERITY: HIGH
EXPLOIT: Test
---
"""
            )

            defender = MagicMock()
            defender.name = "defender"
            defender.generate = AsyncMock(
                return_value="""
---
ATTACK_ID: regression
DEFENSE: REFUTE
EXPLANATION: Tests pass
RESIDUAL_RISK: 0
---
"""
            )

            mock_create.side_effect = [attacker, defender]

            config = RedTeamValidatorConfig(
                attacker_types=["test"],
                defender_type="test",
                attack_rounds=1,
            )
            validator = RedTeamValidator(config)

            result = await validator.validate(sample_proposal, sample_analysis)

            # All attacks defended = full robustness
            assert result.robustness_score == 1.0
            assert result.successful_attacks == 0

    @pytest.mark.asyncio
    async def test_validate_critical_issues_fail(self, sample_proposal, sample_analysis):
        """Test validation fails with critical issues."""
        with patch(
            "aragora.nomic.testfixer.validators.redteam_validator.create_agent"
        ) as mock_create:
            attacker = MagicMock()
            attacker.name = "attacker"
            attacker.generate = AsyncMock(
                return_value="""
---
ATTACK_TYPE: SECURITY
DESCRIPTION: SQL injection vulnerability
SEVERITY: CRITICAL
EXPLOIT: Input: '; DROP TABLE users; --
---
"""
            )

            mock_create.return_value = attacker

            config = RedTeamValidatorConfig(
                attacker_types=["test"],
                allow_defense=False,
                attack_rounds=1,
                max_critical_issues=0,
            )
            validator = RedTeamValidator(config)

            result = await validator.validate(sample_proposal, sample_analysis)

            # Critical issue = fail
            assert result.passes is False
            assert len(result.critical_issues) == 1

    @pytest.mark.asyncio
    async def test_validate_robustness_threshold(self, sample_proposal, sample_analysis):
        """Test validation fails below robustness threshold."""
        with patch(
            "aragora.nomic.testfixer.validators.redteam_validator.create_agent"
        ) as mock_create:
            attacker = MagicMock()
            attacker.name = "attacker"
            # Return multiple attacks
            attacker.generate = AsyncMock(
                return_value="""
---
ATTACK_TYPE: EDGE_CASE
DESCRIPTION: Issue 1
SEVERITY: MEDIUM
EXPLOIT: Test
---

---
ATTACK_TYPE: EDGE_CASE
DESCRIPTION: Issue 2
SEVERITY: MEDIUM
EXPLOIT: Test
---

---
ATTACK_TYPE: EDGE_CASE
DESCRIPTION: Issue 3
SEVERITY: MEDIUM
EXPLOIT: Test
---
"""
            )

            mock_create.return_value = attacker

            config = RedTeamValidatorConfig(
                attacker_types=["test"],
                allow_defense=False,
                attack_rounds=1,
                min_robustness_score=0.6,
                max_successful_attacks=5,  # Allow attacks
            )
            validator = RedTeamValidator(config)

            result = await validator.validate(sample_proposal, sample_analysis)

            # All attacks succeed = 0 robustness, should fail
            assert result.robustness_score == 0.0
            assert result.passes is False

    @pytest.mark.asyncio
    async def test_validate_collects_suggestions(self, sample_proposal, sample_analysis):
        """Test validation collects mitigation suggestions."""
        with patch(
            "aragora.nomic.testfixer.validators.redteam_validator.create_agent"
        ) as mock_create:
            attacker = MagicMock()
            attacker.name = "attacker"
            attacker.generate = AsyncMock(
                return_value="""
---
ATTACK_TYPE: EDGE_CASE
DESCRIPTION: Empty input not handled
SEVERITY: MEDIUM
EXPLOIT: empty input
---
"""
            )

            defender = MagicMock()
            defender.name = "defender"
            defender.generate = AsyncMock(
                return_value="""
---
ATTACK_ID: edge_case
DEFENSE: MITIGATE
EXPLANATION: Add input validation before processing
RESIDUAL_RISK: 10
---
"""
            )

            mock_create.side_effect = [attacker, defender]

            config = RedTeamValidatorConfig(
                attacker_types=["test"],
                attack_rounds=1,
            )
            validator = RedTeamValidator(config)

            result = await validator.validate(sample_proposal, sample_analysis)

            # Should collect suggestion from mitigation
            assert len(result.suggestions) > 0
            assert "input validation" in result.suggestions[0]


# ---------------------------------------------------------------------------
# Attack Round Tests
# ---------------------------------------------------------------------------


class TestAttackRounds:
    """Tests for attack round execution."""

    @pytest.mark.asyncio
    async def test_run_attack_round(self, sample_proposal, sample_analysis):
        """Test running a single attack round."""
        with patch(
            "aragora.nomic.testfixer.validators.redteam_validator.create_agent"
        ) as mock_create:
            attacker = MagicMock()
            attacker.name = "attacker"
            attacker.generate = AsyncMock(
                return_value="""
---
ATTACK_TYPE: REGRESSION
DESCRIPTION: Test
SEVERITY: HIGH
EXPLOIT: Test
---
"""
            )
            mock_create.return_value = attacker

            config = RedTeamValidatorConfig(attacker_types=["test"])
            validator = RedTeamValidator(config)
            validator._ensure_agents()

            attacks = await validator._run_attack_round(
                sample_proposal, sample_analysis, round_num=1
            )

            assert len(attacks) == 1

    @pytest.mark.asyncio
    async def test_attack_round_timeout(self, sample_proposal, sample_analysis):
        """Test attack round handles timeout."""
        with patch(
            "aragora.nomic.testfixer.validators.redteam_validator.create_agent"
        ) as mock_create:

            async def slow_response(prompt):
                await asyncio.sleep(10)
                return "response"

            attacker = MagicMock()
            attacker.name = "attacker"
            attacker.generate = AsyncMock(side_effect=slow_response)
            mock_create.return_value = attacker

            config = RedTeamValidatorConfig(
                attacker_types=["test"],
                agent_timeout=0.1,
            )
            validator = RedTeamValidator(config)
            validator._ensure_agents()

            attacks = await validator._run_attack_round(
                sample_proposal, sample_analysis, round_num=1
            )

            # Should return empty list on timeout
            assert len(attacks) == 0


# ---------------------------------------------------------------------------
# Defense Round Tests
# ---------------------------------------------------------------------------


class TestDefenseRounds:
    """Tests for defense round execution."""

    @pytest.mark.asyncio
    async def test_run_defense_round(self, sample_proposal):
        """Test running a single defense round."""
        with patch(
            "aragora.nomic.testfixer.validators.redteam_validator.create_agent"
        ) as mock_create:
            # Create attacker and defender
            attacker = MagicMock()
            attacker.name = "attacker"
            attacker.generate = AsyncMock(return_value="")

            defender = MagicMock()
            defender.name = "defender"
            defender.generate = AsyncMock(
                return_value="""
---
ATTACK_ID: regression
DEFENSE: REFUTE
EXPLANATION: Already tested
RESIDUAL_RISK: 0
---
"""
            )

            mock_create.side_effect = [attacker, defender]

            config = RedTeamValidatorConfig(
                attacker_types=["test"],
                defender_type="test",
            )
            validator = RedTeamValidator(config)
            validator._ensure_agents()

            attacks = [
                CodeAttack(
                    id="attack-001",
                    attack_type=CodeAttackType.REGRESSION,
                    attacker="attacker",
                    description="Test",
                    severity=0.8,
                    exploit="Test",
                )
            ]

            defenses = await validator._run_defense_round(sample_proposal, attacks, round_num=1)

            assert len(defenses) == 1
            assert defenses[0].success is True

    @pytest.mark.asyncio
    async def test_defense_round_no_defender(self, sample_proposal):
        """Test defense round with no defender."""
        with patch(
            "aragora.nomic.testfixer.validators.redteam_validator.create_agent"
        ) as mock_create:
            attacker = MagicMock()
            attacker.name = "attacker"
            mock_create.return_value = attacker

            config = RedTeamValidatorConfig(
                attacker_types=["test"],
                allow_defense=False,
            )
            validator = RedTeamValidator(config)
            validator._ensure_agents()

            attacks = [
                CodeAttack(
                    id="1",
                    attack_type=CodeAttackType.REGRESSION,
                    attacker="a",
                    description="Test",
                    severity=0.8,
                    exploit="Test",
                )
            ]

            defenses = await validator._run_defense_round(sample_proposal, attacks, round_num=1)

            assert len(defenses) == 0


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases."""

    def test_ensure_agents_only_once(self):
        """Test agents only initialized once."""
        with patch(
            "aragora.nomic.testfixer.validators.redteam_validator.create_agent"
        ) as mock_create:
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            config = RedTeamValidatorConfig(
                attacker_types=["test"],
                allow_defense=False,
            )
            validator = RedTeamValidator(config)

            validator._ensure_agents()
            validator._ensure_agents()

            assert mock_create.call_count == 1

    def test_truncate_utility(self):
        """Test truncate utility function."""
        validator = RedTeamValidator()

        short = "short text"
        assert validator._truncate(short, 100) == short

        long_text = "X" * 1000
        truncated = validator._truncate(long_text, 500)
        assert len(truncated) < len(long_text)
        assert "truncated" in truncated

    @pytest.mark.asyncio
    async def test_attack_agent_exception(self, sample_proposal, sample_analysis):
        """Test handling exception from attack agent."""
        with patch(
            "aragora.nomic.testfixer.validators.redteam_validator.create_agent"
        ) as mock_create:
            attacker = MagicMock()
            attacker.name = "attacker"
            attacker.generate = AsyncMock(side_effect=Exception("API error"))
            mock_create.return_value = attacker

            config = RedTeamValidatorConfig(
                attacker_types=["test"],
                allow_defense=False,
            )
            validator = RedTeamValidator(config)

            # Should complete without raising, with empty attacks
            result = await validator.validate(sample_proposal, sample_analysis)

            assert result.total_attacks == 0
