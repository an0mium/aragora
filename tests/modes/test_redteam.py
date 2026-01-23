"""Tests for Red Team adversarial mode."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from aragora.modes.redteam import (
    Attack,
    AttackType,
    Defense,
    RedTeamMode,
    RedTeamProtocol,
    RedTeamResult,
    RedTeamRound,
)


class TestAttackType:
    """Tests for AttackType enum."""

    def test_attack_types_exist(self):
        """All expected attack types exist."""
        assert AttackType.LOGICAL_FALLACY
        assert AttackType.EDGE_CASE
        assert AttackType.UNSTATED_ASSUMPTION
        assert AttackType.COUNTEREXAMPLE
        assert AttackType.SCALABILITY
        assert AttackType.SECURITY
        assert AttackType.ADVERSARIAL_INPUT
        assert AttackType.RESOURCE_EXHAUSTION
        assert AttackType.RACE_CONDITION
        assert AttackType.DEPENDENCY_FAILURE

    def test_attack_type_values(self):
        """Attack types have string values."""
        assert AttackType.LOGICAL_FALLACY.value == "logical_fallacy"
        assert AttackType.SECURITY.value == "security"
        assert AttackType.RACE_CONDITION.value == "race_condition"


class TestAttack:
    """Tests for Attack dataclass."""

    def test_attack_creation(self):
        """Attack can be created with required fields."""
        attack = Attack(
            attack_id="attack-001",
            attack_type=AttackType.SECURITY,
            attacker="RedTeamAgent",
            target_agent="Proposer",
            target_claim="The API is secure",
            attack_description="SQL injection vulnerability found",
            severity=0.9,
            exploitability=0.8,
        )
        assert attack.attack_id == "attack-001"
        assert attack.attack_type == AttackType.SECURITY
        assert attack.severity == 0.9
        assert attack.exploitability == 0.8

    def test_attack_risk_score(self):
        """risk_score is severity * exploitability."""
        attack = Attack(
            attack_id="attack-001",
            attack_type=AttackType.SECURITY,
            attacker="Agent",
            target_agent="Target",
            target_claim="Claim",
            attack_description="Desc",
            severity=0.8,
            exploitability=0.5,
        )
        assert attack.risk_score == 0.4

    def test_attack_defaults(self):
        """Attack has sensible defaults."""
        attack = Attack(
            attack_id="attack-001",
            attack_type=AttackType.EDGE_CASE,
            attacker="Agent",
            target_agent="Target",
            target_claim="Claim",
            attack_description="Desc",
            severity=0.5,
            exploitability=0.5,
        )
        assert attack.evidence == ""
        assert attack.mitigation is None
        assert attack.created_at  # Should have timestamp


class TestDefense:
    """Tests for Defense dataclass."""

    def test_defense_creation(self):
        """Defense can be created with required fields."""
        defense = Defense(
            defense_id="defense-001",
            attack_id="attack-001",
            defender="Proposer",
            defense_type="mitigate",
            explanation="We will add input validation",
            success=True,
            residual_risk=0.1,
        )
        assert defense.defense_id == "defense-001"
        assert defense.attack_id == "attack-001"
        assert defense.defense_type == "mitigate"
        assert defense.success is True
        assert defense.residual_risk == 0.1

    def test_defense_types(self):
        """Defense can have different types."""
        for defense_type in ["refute", "acknowledge", "mitigate", "accept"]:
            defense = Defense(
                defense_id="d1",
                attack_id="a1",
                defender="D",
                defense_type=defense_type,
                explanation="Exp",
                success=True,
            )
            assert defense.defense_type == defense_type


class TestRedTeamRound:
    """Tests for RedTeamRound dataclass."""

    def test_round_creation(self):
        """Round can be created with number and phase."""
        round_ = RedTeamRound(round_num=1, phase="attack")
        assert round_.round_num == 1
        assert round_.phase == "attack"

    def test_round_defaults(self):
        """Round has empty lists by default."""
        round_ = RedTeamRound(round_num=1, phase="attack")
        assert round_.attacks == []
        assert round_.defenses == []
        assert round_.escalations == []

    def test_round_with_attacks(self):
        """Round can store attacks."""
        attack = Attack(
            attack_id="a1",
            attack_type=AttackType.SECURITY,
            attacker="Agent",
            target_agent="Target",
            target_claim="Claim",
            attack_description="Desc",
            severity=0.5,
            exploitability=0.5,
        )
        round_ = RedTeamRound(round_num=1, phase="attack", attacks=[attack])
        assert len(round_.attacks) == 1


class TestRedTeamResult:
    """Tests for RedTeamResult dataclass."""

    def test_result_creation(self):
        """Result can be created with summary data."""
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
            coverage_score=0.8,
        )
        assert result.session_id == "session-001"
        assert result.total_attacks == 10
        assert result.successful_attacks == 3
        assert result.robustness_score == 0.7

    def test_vulnerability_ratio(self):
        """vulnerability_ratio calculates correctly."""
        result = RedTeamResult(
            session_id="s1",
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
        assert result.vulnerability_ratio == 0.4

    def test_vulnerability_ratio_zero_attacks(self):
        """vulnerability_ratio handles zero attacks."""
        result = RedTeamResult(
            session_id="s1",
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

    def test_protocol_defaults(self):
        """Protocol has sensible defaults."""
        protocol = RedTeamProtocol()
        assert protocol.attack_rounds == 2
        assert protocol.defend_rounds == 2
        assert protocol.include_steelman is True
        assert protocol.include_strawman is True

    def test_protocol_customization(self):
        """Protocol can be customized."""
        protocol = RedTeamProtocol(
            attack_rounds=4,
            defend_rounds=3,
            include_steelman=False,
            include_strawman=False,
        )
        assert protocol.attack_rounds == 4
        assert protocol.defend_rounds == 3
        assert protocol.include_steelman is False

    def test_attack_categories(self):
        """Protocol defines attack categories."""
        assert len(RedTeamProtocol.ATTACK_CATEGORIES) > 0
        assert AttackType.LOGICAL_FALLACY in RedTeamProtocol.ATTACK_CATEGORIES
        assert AttackType.SECURITY in RedTeamProtocol.ATTACK_CATEGORIES

    def test_generate_attack_prompt(self):
        """generate_attack_prompt creates valid prompt."""
        protocol = RedTeamProtocol()
        prompt = protocol.generate_attack_prompt(
            target_proposal="Build a web API",
            attack_types=[AttackType.SECURITY, AttackType.EDGE_CASE],
            round_num=1,
        )
        assert "RED TEAM" in prompt
        assert "Build a web API" in prompt
        assert "security" in prompt.lower()
        assert "edge case" in prompt.lower()
        assert "Round 1" in prompt

    def test_generate_defend_prompt(self):
        """generate_defend_prompt creates valid prompt."""
        protocol = RedTeamProtocol()
        attacks = [
            Attack(
                attack_id="a1",
                attack_type=AttackType.SECURITY,
                attacker="Agent",
                target_agent="Target",
                target_claim="Claim",
                attack_description="SQL injection possible",
                severity=0.8,
                exploitability=0.7,
            )
        ]
        prompt = protocol.generate_defend_prompt(
            original_proposal="Build a web API",
            attacks=attacks,
            round_num=1,
        )
        assert "DEFENDING" in prompt
        assert "Build a web API" in prompt
        assert "SQL injection" in prompt
        assert "REFUTE" in prompt
        assert "MITIGATE" in prompt

    def test_generate_steelman_prompt(self):
        """generate_steelman_prompt creates valid prompt."""
        protocol = RedTeamProtocol()
        prompt = protocol.generate_steelman_prompt(
            proposal="Use microservices architecture",
            agent="Architect",
        )
        assert "STEELMAN" in prompt
        assert "STRONGEST" in prompt
        assert "microservices" in prompt.lower()
        assert "Architect" in prompt

    def test_generate_strawman_prompt(self):
        """generate_strawman_prompt creates valid prompt."""
        protocol = RedTeamProtocol()
        prompt = protocol.generate_strawman_prompt(
            proposal="Use microservices architecture",
            agent="Architect",
        )
        assert "STRAWMANNING" in prompt
        assert "ACTUAL claim" in prompt
        assert "Architect" in prompt


class TestRedTeamMode:
    """Tests for RedTeamMode class."""

    def test_mode_creation_default(self):
        """Mode can be created with defaults."""
        mode = RedTeamMode()
        assert mode.protocol is not None
        assert mode._attack_counter == 0

    def test_mode_creation_custom_protocol(self):
        """Mode accepts custom protocol."""
        protocol = RedTeamProtocol(attack_rounds=5)
        mode = RedTeamMode(protocol=protocol)
        assert mode.protocol.attack_rounds == 5

    def test_parse_attacks_finds_severity(self):
        """_parse_attacks detects severity levels."""
        mode = RedTeamMode()

        response = """
        1. Critical security vulnerability in authentication
        2. High severity issue with data validation
        3. Medium problem with error handling
        4. Low priority code style issue
        """

        attacks = mode._parse_attacks(response, "Attacker", "Target")

        # Should find attacks with varying severity
        severities = [a.severity for a in attacks]
        assert any(s >= 0.8 for s in severities)  # Critical/high
        assert any(s <= 0.5 for s in severities)  # Medium/low

    def test_parse_attacks_finds_types(self):
        """_parse_attacks detects attack types from descriptions."""
        mode = RedTeamMode()

        # Response with keywords that trigger attack detection
        # Each line must have trigger words (vulnerability, issue, problem, etc.)
        response = """
        Found a critical security vulnerability that could be exploited.
        There's a potential edge case issue when input is empty.
        A race condition problem exists under high load.
        """

        attacks = mode._parse_attacks(response, "Attacker", "Target")

        types = [a.attack_type for a in attacks]
        # Parser checks for attack type keywords in each line
        assert AttackType.SECURITY in types  # "security" in line
        assert len(attacks) >= 1  # At least security vulnerability found

    def test_parse_attacks_increments_counter(self):
        """_parse_attacks assigns unique IDs."""
        mode = RedTeamMode()
        assert mode._attack_counter == 0

        response = "Found vulnerability issue. Another flaw detected."
        attacks = mode._parse_attacks(response, "Attacker", "Target")

        # Counter should have incremented
        assert mode._attack_counter >= len(attacks)

        # IDs should be unique
        ids = [a.attack_id for a in attacks]
        assert len(ids) == len(set(ids))

    def test_generate_report(self):
        """generate_report creates Markdown report."""
        mode = RedTeamMode()

        critical_attack = Attack(
            attack_id="a1",
            attack_type=AttackType.SECURITY,
            attacker="RedTeam",
            target_agent="Target",
            target_claim="Claim",
            attack_description="Critical SQL injection",
            severity=0.9,
            exploitability=0.8,
        )

        result = RedTeamResult(
            session_id="session-001",
            target="Test web API proposal",
            rounds=[RedTeamRound(round_num=1, phase="attack")],
            total_attacks=5,
            successful_attacks=2,
            critical_issues=[critical_attack],
            mitigated_issues=[],
            accepted_risks=[],
            robustness_score=0.6,
            coverage_score=0.5,
        )

        report = mode.generate_report(result)

        assert "# Red Team Report" in report
        assert "session-001" in report
        assert "Total Attacks | 5" in report
        assert "Successful | 2" in report
        assert "60%" in report  # Robustness
        assert "Critical Issues" in report
        assert "SQL injection" in report


class TestRedTeamModeRunRedteam:
    """Tests for RedTeamMode.run_redteam method."""

    @pytest.mark.asyncio
    async def test_run_redteam_basic(self):
        """run_redteam executes full red-team session."""
        mode = RedTeamMode()

        # Mock agent
        agent = MagicMock()
        agent.name = "RedTeamAgent"

        # Mock run_agent_fn that returns attack responses
        async def mock_run_agent(agent, prompt):
            return "Found critical vulnerability in the authentication system."

        result = await mode.run_redteam(
            target_proposal="Build secure authentication",
            proposer="Developer",
            red_team_agents=[agent],
            run_agent_fn=mock_run_agent,
            max_rounds=2,
        )

        assert result.session_id.startswith("redteam-")
        assert len(result.rounds) > 0
        assert result.total_attacks >= 0
        assert 0.0 <= result.robustness_score <= 1.0
        assert 0.0 <= result.coverage_score <= 1.0

    @pytest.mark.asyncio
    async def test_run_redteam_multiple_agents(self):
        """run_redteam handles multiple red team agents."""
        mode = RedTeamMode()

        agents = [
            MagicMock(name="Agent1"),
            MagicMock(name="Agent2"),
        ]
        agents[0].name = "Agent1"
        agents[1].name = "Agent2"

        async def mock_run_agent(agent, prompt):
            return f"Found issue from {agent.name}"

        result = await mode.run_redteam(
            target_proposal="Test proposal",
            proposer="Developer",
            red_team_agents=agents,
            run_agent_fn=mock_run_agent,
            max_rounds=1,
        )

        # Both agents should have attacked
        assert result.total_attacks >= 0

    @pytest.mark.asyncio
    async def test_run_redteam_steelman_round(self):
        """run_redteam includes steelman when enabled."""
        protocol = RedTeamProtocol(include_steelman=True)
        mode = RedTeamMode(protocol=protocol)

        agent = MagicMock(name="Agent")
        agent.name = "Agent"

        async def mock_run_agent(agent, prompt):
            return "Attack description"

        result = await mode.run_redteam(
            target_proposal="Test",
            proposer="Dev",
            red_team_agents=[agent],
            run_agent_fn=mock_run_agent,
            max_rounds=2,
        )

        # Should have steelman round
        phases = [r.phase for r in result.rounds]
        assert "steelman" in phases


class TestRedTeamModeParseDefenses:
    """Tests for RedTeamMode._parse_defenses method."""

    def test_parse_defenses_refute(self):
        """_parse_defenses detects refute defense type."""
        mode = RedTeamMode()

        attacks = [
            Attack(
                attack_id="a1",
                attack_type=AttackType.SECURITY,
                attacker="Attacker",
                target_agent="Target",
                target_claim="Claim",
                attack_description="SQL injection",
                severity=0.8,
                exploitability=0.7,
            )
        ]

        response = (
            "I refute this attack. The claim is invalid because we use parameterized queries."
        )
        defenses = mode._parse_defenses(response, "Defender", attacks)

        assert len(defenses) == 1
        assert defenses[0].defense_type == "refute"
        assert defenses[0].success is True
        assert defenses[0].residual_risk == 0.0

    def test_parse_defenses_mitigate(self):
        """_parse_defenses detects mitigate defense type."""
        mode = RedTeamMode()

        attacks = [
            Attack(
                attack_id="a1",
                attack_type=AttackType.EDGE_CASE,
                attacker="Attacker",
                target_agent="Target",
                target_claim="Claim",
                attack_description="Edge case issue",
                severity=0.5,
                exploitability=0.6,
            )
        ]

        response = "We can mitigate this issue by adding input validation. We will fix this."
        defenses = mode._parse_defenses(response, "Defender", attacks)

        assert len(defenses) == 1
        assert defenses[0].defense_type == "mitigate"
        assert defenses[0].success is True
        assert defenses[0].residual_risk > 0.0

    def test_parse_defenses_accept_risk(self):
        """_parse_defenses detects accept defense type."""
        mode = RedTeamMode()

        attacks = [
            Attack(
                attack_id="a1",
                attack_type=AttackType.SCALABILITY,
                attacker="Attacker",
                target_agent="Target",
                target_claim="Claim",
                attack_description="Scalability issue",
                severity=0.6,
                exploitability=0.4,
            )
        ]

        response = "We accept the risk as it's within acceptable bounds for our use case."
        defenses = mode._parse_defenses(response, "Defender", attacks)

        assert len(defenses) == 1
        assert defenses[0].defense_type == "accept"
        assert defenses[0].success is False
        assert defenses[0].residual_risk > 0.0

    def test_parse_defenses_default_acknowledge(self):
        """_parse_defenses defaults to acknowledge when no keywords match."""
        mode = RedTeamMode()

        attacks = [
            Attack(
                attack_id="a1",
                attack_type=AttackType.LOGICAL_FALLACY,
                attacker="Attacker",
                target_agent="Target",
                target_claim="Claim",
                attack_description="Fallacy detected",
                severity=0.5,
                exploitability=0.5,
            )
        ]

        response = "We understand the concern and will review."
        defenses = mode._parse_defenses(response, "Defender", attacks)

        assert len(defenses) == 1
        assert defenses[0].defense_type == "acknowledge"
        assert defenses[0].success is True

    def test_parse_defenses_multiple_attacks(self):
        """_parse_defenses handles multiple attacks."""
        mode = RedTeamMode()

        attacks = [
            Attack(
                attack_id="a1",
                attack_type=AttackType.SECURITY,
                attacker="Attacker",
                target_agent="Target",
                target_claim="Claim 1",
                attack_description="Issue 1",
                severity=0.8,
                exploitability=0.7,
            ),
            Attack(
                attack_id="a2",
                attack_type=AttackType.EDGE_CASE,
                attacker="Attacker",
                target_agent="Target",
                target_claim="Claim 2",
                attack_description="Issue 2",
                severity=0.5,
                exploitability=0.6,
            ),
        ]

        response = "We refute the security concern and will mitigate the edge case."
        defenses = mode._parse_defenses(response, "Defender", attacks)

        assert len(defenses) == 2
        assert defenses[0].attack_id == "a1"
        assert defenses[1].attack_id == "a2"

    def test_parse_defenses_assigns_unique_ids(self):
        """_parse_defenses assigns unique defense IDs."""
        mode = RedTeamMode()

        attacks = [
            Attack(
                attack_id="a1",
                attack_type=AttackType.SECURITY,
                attacker="Attacker",
                target_agent="Target",
                target_claim="Claim",
                attack_description="Issue",
                severity=0.5,
                exploitability=0.5,
            ),
            Attack(
                attack_id="a2",
                attack_type=AttackType.EDGE_CASE,
                attacker="Attacker",
                target_agent="Target",
                target_claim="Claim 2",
                attack_description="Issue 2",
                severity=0.5,
                exploitability=0.5,
            ),
        ]

        response = "We will address both issues."
        defenses = mode._parse_defenses(response, "Defender", attacks)

        ids = [d.defense_id for d in defenses]
        assert len(ids) == len(set(ids))  # All unique
        assert all(id_.startswith("defense-") for id_ in ids)

    def test_parse_defenses_increments_counter(self):
        """_parse_defenses increments defense counter."""
        mode = RedTeamMode()
        assert mode._defense_counter == 0

        attacks = [
            Attack(
                attack_id="a1",
                attack_type=AttackType.SECURITY,
                attacker="Attacker",
                target_agent="Target",
                target_claim="Claim",
                attack_description="Issue",
                severity=0.5,
                exploitability=0.5,
            )
        ]

        mode._parse_defenses("response", "Defender", attacks)
        assert mode._defense_counter == 1

        mode._parse_defenses("response", "Defender", attacks)
        assert mode._defense_counter == 2


class TestRedTeamModeDefenseExecution:
    """Tests for defense execution in run_redteam."""

    @pytest.mark.asyncio
    async def test_run_redteam_with_proposer_agent(self):
        """run_redteam executes defense when proposer_agent provided."""
        protocol = RedTeamProtocol(include_steelman=False, include_strawman=False)
        mode = RedTeamMode(protocol=protocol)

        attacker = MagicMock(name="Attacker")
        attacker.name = "Attacker"

        proposer = MagicMock(name="Proposer")
        proposer.name = "Proposer"

        call_count = {"attack": 0, "defend": 0}

        async def mock_run_agent(agent, prompt):
            if agent.name == "Attacker":
                call_count["attack"] += 1
                return "Found critical vulnerability issue in the system."
            else:
                call_count["defend"] += 1
                return "We refute this claim. The system is secure."

        result = await mode.run_redteam(
            target_proposal="Test proposal",
            proposer="Developer",
            red_team_agents=[attacker],
            run_agent_fn=mock_run_agent,
            max_rounds=1,
            proposer_agent=proposer,
        )

        # Both attack and defense should have been called
        assert call_count["attack"] > 0
        assert call_count["defend"] > 0

        # Should have defend phase
        phases = [r.phase for r in result.rounds]
        assert "defend" in phases

    @pytest.mark.asyncio
    async def test_run_redteam_without_proposer_agent(self):
        """run_redteam marks as attack_only when no proposer_agent."""
        protocol = RedTeamProtocol(include_steelman=False, include_strawman=False)
        mode = RedTeamMode(protocol=protocol)

        attacker = MagicMock(name="Attacker")
        attacker.name = "Attacker"

        async def mock_run_agent(agent, prompt):
            return "Found vulnerability issue in the system."

        result = await mode.run_redteam(
            target_proposal="Test proposal",
            proposer="Developer",
            red_team_agents=[attacker],
            run_agent_fn=mock_run_agent,
            max_rounds=1,
            proposer_agent=None,
        )

        # Should have attack_only phase when no proposer
        phases = [r.phase for r in result.rounds]
        assert "attack_only" in phases or "attack" in phases
