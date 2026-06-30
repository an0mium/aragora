"""
Tests for Red Team adversarial mode.

Tests the components for adversarial red-team debates including:
- Attack types and attack/defense dataclasses
- Protocol for generating attack/defend prompts
- Red team mode orchestration
- Report generation
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from aragora.modes.redteam import (
    AttackType,
    Attack,
    Defense,
    RedTeamRound,
    RedTeamResult,
    RedTeamProtocol,
    RedTeamMode,
    redteam_code_review,
    redteam_policy,
)


class TestAttackType:
    """Tests for AttackType enum."""

    def test_all_attack_types_defined(self):
        """Test all 10 attack types are defined."""
        expected_types = [
            "logical_fallacy",
            "edge_case",
            "unstated_assumption",
            "counterexample",
            "scalability",
            "security",
            "adversarial_input",
            "resource_exhaustion",
            "race_condition",
            "dependency_failure",
        ]
        actual_types = [at.value for at in AttackType]
        assert sorted(actual_types) == sorted(expected_types)

    def test_attack_type_values(self):
        """Test specific attack type values."""
        assert AttackType.SECURITY.value == "security"
        assert AttackType.LOGICAL_FALLACY.value == "logical_fallacy"
        assert AttackType.RACE_CONDITION.value == "race_condition"


class TestAttack:
    """Tests for Attack dataclass."""

    def test_attack_creation(self):
        """Test creating an Attack object."""
        attack = Attack(
            attack_id="attack-001",
            attack_type=AttackType.SECURITY,
            attacker="red_agent_1",
            target_agent="proposer",
            target_claim="The API is secure",
            attack_description="SQL injection vulnerability in user input",
            severity=0.9,
            exploitability=0.8,
        )

        assert attack.attack_id == "attack-001"
        assert attack.attack_type == AttackType.SECURITY
        assert attack.attacker == "red_agent_1"
        assert attack.severity == 0.9

    def test_attack_risk_score(self):
        """Test risk score calculation."""
        attack = Attack(
            attack_id="attack-002",
            attack_type=AttackType.EDGE_CASE,
            attacker="agent",
            target_agent="target",
            target_claim="claim",
            attack_description="description",
            severity=0.8,
            exploitability=0.5,
        )

        assert attack.risk_score == 0.4  # 0.8 * 0.5

    def test_attack_risk_score_max(self):
        """Test max risk score."""
        attack = Attack(
            attack_id="attack-003",
            attack_type=AttackType.SECURITY,
            attacker="agent",
            target_agent="target",
            target_claim="claim",
            attack_description="description",
            severity=1.0,
            exploitability=1.0,
        )

        assert attack.risk_score == 1.0

    def test_attack_risk_score_zero(self):
        """Test zero risk score."""
        attack = Attack(
            attack_id="attack-004",
            attack_type=AttackType.LOGICAL_FALLACY,
            attacker="agent",
            target_agent="target",
            target_claim="claim",
            attack_description="description",
            severity=0.0,
            exploitability=0.5,
        )

        assert attack.risk_score == 0.0

    def test_attack_default_values(self):
        """Test attack default values."""
        attack = Attack(
            attack_id="attack-005",
            attack_type=AttackType.EDGE_CASE,
            attacker="agent",
            target_agent="target",
            target_claim="claim",
            attack_description="description",
            severity=0.5,
            exploitability=0.5,
        )

        assert attack.evidence == ""
        assert attack.mitigation is None
        assert attack.created_at is not None

    def test_attack_with_mitigation(self):
        """Test attack with mitigation provided."""
        attack = Attack(
            attack_id="attack-006",
            attack_type=AttackType.SECURITY,
            attacker="agent",
            target_agent="target",
            target_claim="claim",
            attack_description="description",
            severity=0.7,
            exploitability=0.6,
            mitigation="Use parameterized queries",
        )

        assert attack.mitigation == "Use parameterized queries"


class TestDefense:
    """Tests for Defense dataclass."""

    def test_defense_creation(self):
        """Test creating a Defense object."""
        defense = Defense(
            defense_id="defense-001",
            attack_id="attack-001",
            defender="proposer",
            defense_type="refute",
            explanation="The claim is false because...",
            success=True,
        )

        assert defense.defense_id == "defense-001"
        assert defense.attack_id == "attack-001"
        assert defense.success is True

    def test_defense_types(self):
        """Test different defense types."""
        for defense_type in ["refute", "acknowledge", "mitigate", "accept"]:
            defense = Defense(
                defense_id="d-1",
                attack_id="a-1",
                defender="defender",
                defense_type=defense_type,
                explanation="explanation",
                success=True,
            )
            assert defense.defense_type == defense_type

    def test_defense_residual_risk(self):
        """Test defense with residual risk."""
        defense = Defense(
            defense_id="defense-002",
            attack_id="attack-002",
            defender="proposer",
            defense_type="mitigate",
            explanation="Reduced but not eliminated",
            success=True,
            residual_risk=0.2,
        )

        assert defense.residual_risk == 0.2


class TestRedTeamRound:
    """Tests for RedTeamRound dataclass."""

    def test_round_creation(self):
        """Test creating a RedTeamRound."""
        round_obj = RedTeamRound(round_num=1, phase="attack")

        assert round_obj.round_num == 1
        assert round_obj.phase == "attack"
        assert round_obj.attacks == []
        assert round_obj.defenses == []
        assert round_obj.escalations == []

    def test_round_with_attacks(self):
        """Test round with attacks."""
        attack = Attack(
            attack_id="a-1",
            attack_type=AttackType.SECURITY,
            attacker="agent",
            target_agent="target",
            target_claim="claim",
            attack_description="desc",
            severity=0.5,
            exploitability=0.5,
        )

        round_obj = RedTeamRound(
            round_num=1,
            phase="attack",
            attacks=[attack],
        )

        assert len(round_obj.attacks) == 1


class TestRedTeamResult:
    """Tests for RedTeamResult dataclass."""

    def test_result_creation(self):
        """Test creating a RedTeamResult."""
        result = RedTeamResult(
            session_id="session-001",
            target="Test proposal",
            rounds=[],
            total_attacks=10,
            successful_attacks=3,
            critical_issues=[],
            mitigated_issues=[],
            accepted_risks=[],
            robustness_score=0.7,
            coverage_score=0.5,
        )

        assert result.session_id == "session-001"
        assert result.total_attacks == 10
        assert result.robustness_score == 0.7

    def test_vulnerability_ratio(self):
        """Test vulnerability ratio calculation."""
        result = RedTeamResult(
            session_id="session-002",
            target="Test",
            rounds=[],
            total_attacks=10,
            successful_attacks=4,
            critical_issues=[],
            mitigated_issues=[],
            accepted_risks=[],
            robustness_score=0.6,
            coverage_score=0.5,
        )

        assert result.vulnerability_ratio == 0.4  # 4/10

    def test_vulnerability_ratio_no_attacks(self):
        """Test vulnerability ratio with no attacks."""
        result = RedTeamResult(
            session_id="session-003",
            target="Test",
            rounds=[],
            total_attacks=0,
            successful_attacks=0,
            critical_issues=[],
            mitigated_issues=[],
            accepted_risks=[],
            robustness_score=1.0,
            coverage_score=0.0,
        )

        assert result.vulnerability_ratio == 0.0


class TestRedTeamProtocol:
    """Tests for RedTeamProtocol class."""

    def test_default_protocol(self):
        """Test default protocol settings."""
        protocol = RedTeamProtocol()

        assert protocol.attack_rounds == 2
        assert protocol.defend_rounds == 2
        assert protocol.include_steelman is True
        assert protocol.include_strawman is True

    def test_custom_protocol(self):
        """Test custom protocol settings."""
        protocol = RedTeamProtocol(
            attack_rounds=5,
            defend_rounds=3,
            include_steelman=False,
            include_strawman=False,
        )

        assert protocol.attack_rounds == 5
        assert protocol.defend_rounds == 3
        assert protocol.include_steelman is False

    def test_attack_categories(self):
        """Test attack categories are defined."""
        protocol = RedTeamProtocol()
        assert len(protocol.ATTACK_CATEGORIES) == 5
        assert AttackType.SECURITY in protocol.ATTACK_CATEGORIES

    def test_generate_attack_prompt(self):
        """Test attack prompt generation."""
        protocol = RedTeamProtocol()

        prompt = protocol.generate_attack_prompt(
            target_proposal="Use MD5 for password hashing",
            attack_types=[AttackType.SECURITY, AttackType.EDGE_CASE],
            round_num=1,
        )

        assert "RED TEAM" in prompt
        assert "MD5" in prompt
        assert "security" in prompt
        assert "edge case" in prompt
        assert "Round 1" in prompt

    def test_generate_defend_prompt(self):
        """Test defense prompt generation."""
        protocol = RedTeamProtocol()

        attacks = [
            Attack(
                attack_id="a-1",
                attack_type=AttackType.SECURITY,
                attacker="agent",
                target_agent="target",
                target_claim="claim",
                attack_description="MD5 is cryptographically broken",
                severity=0.9,
                exploitability=0.8,
            )
        ]

        prompt = protocol.generate_defend_prompt(
            original_proposal="Use MD5",
            attacks=attacks,
            round_num=1,
        )

        assert "DEFENDING" in prompt
        assert "MD5" in prompt
        assert "cryptographically broken" in prompt
        assert "REFUTE" in prompt
        assert "MITIGATE" in prompt

    def test_generate_steelman_prompt(self):
        """Test steelman prompt generation."""
        protocol = RedTeamProtocol()

        prompt = protocol.generate_steelman_prompt(
            proposal="Use NoSQL for everything",
            agent="developer",
        )

        assert "STEELMAN" in prompt
        assert "STRONGEST" in prompt
        assert "NoSQL" in prompt
        assert "developer" in prompt

    def test_generate_strawman_prompt(self):
        """Test strawman prompt generation."""
        protocol = RedTeamProtocol()

        prompt = protocol.generate_strawman_prompt(
            proposal="Use microservices",
            agent="architect",
        )

        assert "STRAWMANNING" in prompt
        assert "microservices" in prompt
        assert "architect" in prompt
        assert "ACTUAL claim" in prompt


class TestRedTeamMode:
    """Tests for RedTeamMode class."""

    def test_mode_default_protocol(self):
        """Test mode with default protocol."""
        mode = RedTeamMode()
        assert mode.protocol is not None
        assert mode._attack_counter == 0

    def test_mode_custom_protocol(self):
        """Test mode with custom protocol."""
        protocol = RedTeamProtocol(attack_rounds=5)
        mode = RedTeamMode(protocol=protocol)
        assert mode.protocol.attack_rounds == 5

    def test_parse_attacks_basic(self):
        """Test parsing attacks from response."""
        mode = RedTeamMode()

        response = """
        I found a critical vulnerability in the code.
        The issue is that user input is not sanitized.
        This could lead to SQL injection attacks.
        """

        attacks = mode._parse_attacks(response, "red_agent", "target")

        # Should find at least one attack
        assert len(attacks) >= 1
        assert all(a.attacker == "red_agent" for a in attacks)
        assert all(a.target_agent == "target" for a in attacks)

    def test_parse_attacks_severity_detection(self):
        """Test severity detection in parsing."""
        mode = RedTeamMode()

        # Test critical severity
        response = "This is a critical vulnerability that could be exploited"
        attacks = mode._parse_attacks(response, "agent", "target")
        if attacks:
            assert attacks[0].severity == 0.9

    def test_parse_attacks_high_severity(self):
        """Test high severity detection."""
        mode = RedTeamMode()

        response = "This is a high severity issue with the code"
        attacks = mode._parse_attacks(response, "agent", "target")
        if attacks:
            assert attacks[0].severity == 0.7

    def test_parse_attacks_medium_severity(self):
        """Test medium severity detection."""
        mode = RedTeamMode()

        response = "This is a medium severity problem in the implementation"
        attacks = mode._parse_attacks(response, "agent", "target")
        if attacks:
            assert attacks[0].severity == 0.5

    def test_parse_attacks_low_severity(self):
        """Test low severity detection."""
        mode = RedTeamMode()

        response = "This is a low severity weakness that might cause issues"
        attacks = mode._parse_attacks(response, "agent", "target")
        if attacks:
            assert attacks[0].severity == 0.3

    def test_parse_attacks_type_detection(self):
        """Test attack type detection."""
        mode = RedTeamMode()

        response = "There is a race condition vulnerability in this code"
        attacks = mode._parse_attacks(response, "agent", "target")
        if attacks:
            assert attacks[0].attack_type == AttackType.RACE_CONDITION

    def test_parse_attacks_security_type(self):
        """Test security attack type detection."""
        mode = RedTeamMode()

        response = "Security vulnerability detected in authentication"
        attacks = mode._parse_attacks(response, "agent", "target")
        if attacks:
            assert attacks[0].attack_type == AttackType.SECURITY

    def test_parse_attacks_empty_response(self):
        """Test parsing empty response."""
        mode = RedTeamMode()

        attacks = mode._parse_attacks("", "agent", "target")
        assert attacks == []

    def test_parse_attacks_no_keywords(self):
        """Test parsing response without attack keywords."""
        mode = RedTeamMode()

        response = "This code looks fine to me."
        attacks = mode._parse_attacks(response, "agent", "target")
        assert attacks == []

    def test_attack_id_increments(self):
        """Test attack IDs increment correctly."""
        mode = RedTeamMode()
        mode._attack_counter = 0

        response1 = "Critical vulnerability found in the system"
        response2 = "Another issue that could cause problems"

        attacks1 = mode._parse_attacks(response1, "agent", "target")
        attacks2 = mode._parse_attacks(response2, "agent", "target")

        if attacks1 and attacks2:
            # IDs should be different
            assert attacks1[0].attack_id != attacks2[0].attack_id

    @pytest.mark.asyncio
    async def test_run_redteam_basic(self):
        """Test basic red team session."""
        mode = RedTeamMode()

        # Mock agents
        agent1 = MagicMock()
        agent1.name = "red_agent_1"

        # Mock run function
        async def mock_run(agent, prompt):
            return "Found a vulnerability that could be exploited"

        result = await mode.run_redteam(
            target_proposal="Use password123 as default password",
            proposer="developer",
            red_team_agents=[agent1],
            run_agent_fn=mock_run,
            max_rounds=1,
        )

        assert result.session_id.startswith("redteam-")
        assert result.target is not None
        assert len(result.rounds) >= 1

    @pytest.mark.asyncio
    async def test_run_redteam_multiple_agents(self):
        """Test red team with multiple agents."""
        mode = RedTeamMode()

        agents = [MagicMock(name=f"agent_{i}") for i in range(3)]
        for i, a in enumerate(agents):
            a.name = f"agent_{i}"

        async def mock_run(agent, prompt):
            return f"Issue found by {agent.name}"

        result = await mode.run_redteam(
            target_proposal="Test proposal",
            proposer="proposer",
            red_team_agents=agents,
            run_agent_fn=mock_run,
            max_rounds=1,
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_run_redteam_steelman_round(self):
        """Test steelman round is included."""
        protocol = RedTeamProtocol(include_steelman=True)
        mode = RedTeamMode(protocol=protocol)

        agent = MagicMock()
        agent.name = "agent"

        async def mock_run(agent, prompt):
            return "Found issue"

        result = await mode.run_redteam(
            target_proposal="Test",
            proposer="proposer",
            red_team_agents=[agent],
            run_agent_fn=mock_run,
            max_rounds=1,
        )

        # Check for steelman round
        steelman_rounds = [r for r in result.rounds if r.phase == "steelman"]
        assert len(steelman_rounds) == 1


class TestRedTeamReport:
    """Tests for report generation."""

    def test_generate_report_basic(self):
        """Test basic report generation."""
        mode = RedTeamMode()

        result = RedTeamResult(
            session_id="session-001",
            target="Test proposal for review",
            rounds=[],
            total_attacks=5,
            successful_attacks=2,
            critical_issues=[],
            mitigated_issues=[],
            accepted_risks=[],
            robustness_score=0.6,
            coverage_score=0.4,
        )

        report = mode.generate_report(result)

        assert "# Red Team Report" in report
        assert "session-001" in report
        assert "5" in report  # total attacks
        assert "60%" in report  # robustness

    def test_generate_report_with_critical_issues(self):
        """Test report with critical issues."""
        mode = RedTeamMode()

        critical_attack = Attack(
            attack_id="a-1",
            attack_type=AttackType.SECURITY,
            attacker="red_agent",
            target_agent="target",
            target_claim="claim",
            attack_description="Critical SQL injection vulnerability",
            severity=0.95,
            exploitability=0.9,
        )

        result = RedTeamResult(
            session_id="session-002",
            target="Test",
            rounds=[],
            total_attacks=1,
            successful_attacks=1,
            critical_issues=[critical_attack],
            mitigated_issues=[],
            accepted_risks=[],
            robustness_score=0.0,
            coverage_score=0.1,
        )

        report = mode.generate_report(result)

        assert "Critical Issues" in report
        assert "security" in report
        assert "SQL injection" in report

    def test_generate_report_with_rounds(self):
        """Test report includes round info."""
        mode = RedTeamMode()

        rounds = [
            RedTeamRound(round_num=1, phase="attack"),
            RedTeamRound(round_num=1, phase="defend"),
            RedTeamRound(round_num=2, phase="attack"),
        ]

        result = RedTeamResult(
            session_id="session-003",
            target="Test",
            rounds=rounds,
            total_attacks=0,
            successful_attacks=0,
            critical_issues=[],
            mitigated_issues=[],
            accepted_risks=[],
            robustness_score=1.0,
            coverage_score=0.0,
        )

        report = mode.generate_report(result)

        assert "Round 1" in report
        assert "Round 2" in report
        assert "attack" in report
        assert "defend" in report


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_redteam_code_review(self):
        """Test code review convenience function."""
        agent = MagicMock()
        agent.name = "security_agent"

        async def mock_run(agent, prompt):
            return "Found security vulnerability in code"

        result = await redteam_code_review(
            code="def login(password): return password == 'admin'",
            agents=[agent],
            run_agent_fn=mock_run,
        )

        assert result is not None
        assert result.session_id.startswith("redteam-")

    @pytest.mark.asyncio
    async def test_redteam_code_review_attack_categories(self):
        """Test code review uses code-specific attack categories."""
        agent = MagicMock()
        agent.name = "agent"

        prompts_seen = []

        async def capture_prompt(agent, prompt):
            prompts_seen.append(prompt)
            return "Issue found"

        await redteam_code_review(
            code="test code",
            agents=[agent],
            run_agent_fn=capture_prompt,
        )

        # Should focus on code-relevant categories
        if prompts_seen:
            prompt = prompts_seen[0]
            # At least one of these should be mentioned
            assert any(
                cat in prompt.lower()
                for cat in ["security", "edge case", "race condition", "resource exhaustion"]
            )

    @pytest.mark.asyncio
    async def test_redteam_policy(self):
        """Test policy review convenience function."""
        agent = MagicMock()
        agent.name = "policy_analyst"

        async def mock_run(agent, prompt):
            return "Found logical fallacy in policy"

        result = await redteam_policy(
            policy="All users must use 2FA unless they don't want to",
            agents=[agent],
            run_agent_fn=mock_run,
        )

        assert result is not None
        assert result.session_id.startswith("redteam-")

    @pytest.mark.asyncio
    async def test_redteam_policy_includes_steelman(self):
        """Test policy review includes steelman/strawman."""
        agent = MagicMock()
        agent.name = "agent"

        async def mock_run(agent, prompt):
            return "Issue"

        result = await redteam_policy(
            policy="Test policy",
            agents=[agent],
            run_agent_fn=mock_run,
        )

        # Check steelman round exists
        steelman_rounds = [r for r in result.rounds if r.phase == "steelman"]
        assert len(steelman_rounds) > 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_attack_description_truncation(self):
        """Test attack description is truncated at 500 chars."""
        mode = RedTeamMode()

        long_response = "This is a vulnerability " + "x" * 600 + " that could be exploited"
        attacks = mode._parse_attacks(long_response, "agent", "target")

        if attacks:
            assert len(attacks[0].attack_description) <= 500

    @pytest.mark.asyncio
    async def test_run_redteam_empty_agents(self):
        """Test run with empty agent list."""
        mode = RedTeamMode()

        async def mock_run(agent, prompt):
            return "Issue"

        result = await mode.run_redteam(
            target_proposal="Test",
            proposer="proposer",
            red_team_agents=[],
            run_agent_fn=mock_run,
            max_rounds=1,
        )

        assert result.total_attacks == 0
        assert result.robustness_score == 1.0

    @pytest.mark.asyncio
    async def test_run_redteam_zero_rounds(self):
        """Test run with zero max_rounds."""
        mode = RedTeamMode()
        agent = MagicMock()
        agent.name = "agent"

        async def mock_run(agent, prompt):
            return "Issue"

        result = await mode.run_redteam(
            target_proposal="Test",
            proposer="proposer",
            red_team_agents=[agent],
            run_agent_fn=mock_run,
            max_rounds=0,
        )

        assert len(result.rounds) == 0
        assert result.robustness_score == 1.0

    def test_result_created_at_timestamp(self):
        """Test result has valid created_at timestamp."""
        result = RedTeamResult(
            session_id="test",
            target="test",
            rounds=[],
            total_attacks=0,
            successful_attacks=0,
            critical_issues=[],
            mitigated_issues=[],
            accepted_risks=[],
            robustness_score=1.0,
            coverage_score=0.0,
        )

        # Should be valid ISO format
        datetime.fromisoformat(result.created_at)

    def test_attack_created_at_timestamp(self):
        """Test attack has valid created_at timestamp."""
        attack = Attack(
            attack_id="a-1",
            attack_type=AttackType.SECURITY,
            attacker="agent",
            target_agent="target",
            target_claim="claim",
            attack_description="desc",
            severity=0.5,
            exploitability=0.5,
        )

        # Should be valid ISO format
        datetime.fromisoformat(attack.created_at)
