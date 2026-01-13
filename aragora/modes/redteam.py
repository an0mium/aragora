"""
Adversarial Red-Team Mode.

Makes agents actively try to break each other's arguments through:
- Steelman + Strawman rounds
- Exploit detection (logical fallacies, edge cases)
- Devil's advocate protocol
- Security/policy stress-testing

This is a key differentiator vs AutoGen/CrewAI which focus on cooperation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional


class AttackType(Enum):
    """Types of adversarial attacks."""

    LOGICAL_FALLACY = "logical_fallacy"  # Find reasoning errors
    EDGE_CASE = "edge_case"  # Find boundary conditions that break
    UNSTATED_ASSUMPTION = "unstated_assumption"  # Expose hidden premises
    COUNTEREXAMPLE = "counterexample"  # Find cases that disprove
    SCALABILITY = "scalability"  # Stress at scale
    SECURITY = "security"  # Find vulnerabilities
    ADVERSARIAL_INPUT = "adversarial_input"  # Malicious inputs
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # DoS conditions
    RACE_CONDITION = "race_condition"  # Concurrency issues
    DEPENDENCY_FAILURE = "dependency_failure"  # What if X fails?


@dataclass
class Attack:
    """An adversarial attack on a proposal."""

    attack_id: str
    attack_type: AttackType
    attacker: str
    target_agent: str
    target_claim: str
    attack_description: str
    severity: float  # 0-1, how severe if exploited
    exploitability: float  # 0-1, how easy to exploit
    evidence: str = ""  # Proof of concept
    mitigation: Optional[str] = None  # How to fix
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def risk_score(self) -> float:
        """Calculate overall risk score."""
        return self.severity * self.exploitability


@dataclass
class Defense:
    """A defense against an attack."""

    defense_id: str
    attack_id: str
    defender: str
    defense_type: str  # "refute", "acknowledge", "mitigate", "accept"
    explanation: str
    success: bool  # Did the defense hold?
    residual_risk: float = 0.0  # Remaining risk after defense


@dataclass
class RedTeamRound:
    """A round of red-team testing."""

    round_num: int
    phase: str  # "attack", "defend", "steelman", "strawman"
    attacks: list[Attack] = field(default_factory=list)
    defenses: list[Defense] = field(default_factory=list)
    escalations: list[str] = field(default_factory=list)  # Unresolved issues


@dataclass
class RedTeamResult:
    """Result of a red-team session."""

    session_id: str
    target: str  # What was being tested
    rounds: list[RedTeamRound]

    # Findings
    total_attacks: int
    successful_attacks: int
    critical_issues: list[Attack]
    mitigated_issues: list[Attack]
    accepted_risks: list[Attack]

    # Scores
    robustness_score: float  # 0-1, how well did target hold up
    coverage_score: float  # 0-1, how thoroughly was it tested

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def vulnerability_ratio(self) -> float:
        """Ratio of successful attacks."""
        return self.successful_attacks / self.total_attacks if self.total_attacks > 0 else 0.0


class RedTeamProtocol:
    """
    Protocol for adversarial red-team debates.

    Structures the debate into attack/defend phases with specific objectives.
    """

    # Attack categories to cover
    ATTACK_CATEGORIES = [
        AttackType.LOGICAL_FALLACY,
        AttackType.EDGE_CASE,
        AttackType.UNSTATED_ASSUMPTION,
        AttackType.SECURITY,
        AttackType.SCALABILITY,
    ]

    def __init__(
        self,
        attack_rounds: int = 2,
        defend_rounds: int = 2,
        include_steelman: bool = True,
        include_strawman: bool = True,
    ):
        self.attack_rounds = attack_rounds
        self.defend_rounds = defend_rounds
        self.include_steelman = include_steelman
        self.include_strawman = include_strawman

    def generate_attack_prompt(
        self,
        target_proposal: str,
        attack_types: list[AttackType],
        round_num: int,
    ) -> str:
        """Generate prompt for attack phase."""
        types_list = ", ".join(t.value.replace("_", " ") for t in attack_types)

        return f"""You are a RED TEAM agent. Your goal is to find weaknesses in this proposal.

## Target Proposal
{target_proposal}

## Your Objectives
Find vulnerabilities in these categories: {types_list}

For EACH vulnerability you find:
1. **Type**: Which category
2. **Description**: What is the weakness
3. **Exploit**: How could this be exploited (be specific)
4. **Severity**: Low/Medium/High/Critical
5. **Evidence**: Proof of concept or example

Be adversarial but fair. Focus on real issues, not nitpicks.
Round {round_num} of {self.attack_rounds} - find the most critical issues first.
"""

    def generate_defend_prompt(
        self,
        original_proposal: str,
        attacks: list[Attack],
        round_num: int,
    ) -> str:
        """Generate prompt for defense phase."""
        attacks_text = "\n".join(
            [
                f"- [{a.attack_type.value}] {a.attack_description} (Severity: {a.severity:.0%})"
                for a in attacks
            ]
        )

        return f"""You are DEFENDING this proposal against attacks.

## Your Proposal
{original_proposal}

## Attacks Against You
{attacks_text}

For EACH attack, respond with ONE of:
1. **REFUTE**: Explain why the attack is invalid
2. **ACKNOWLEDGE + MITIGATE**: Accept the issue and explain how to fix it
3. **ACCEPT RISK**: Explain why the risk is acceptable (with tradeoffs)

You MUST address every attack. Be honest - don't dismiss valid concerns.
Round {round_num} of {self.defend_rounds}.
"""

    def generate_steelman_prompt(self, proposal: str, agent: str) -> str:
        """Generate prompt for steelman phase (best version of opponent's argument)."""
        return f"""You must STEELMAN this proposal - present the STRONGEST possible version.

## Original Proposal (by {agent})
{proposal}

## Your Task
1. Identify the core insight or goal
2. Remove any weak arguments
3. Add the strongest supporting evidence
4. Address obvious objections preemptively
5. Present it as compellingly as possible

You are not agreeing - you are showing you understand the best version of this argument.
This demonstrates intellectual honesty and helps find real disagreements vs misunderstandings.
"""

    def generate_strawman_prompt(self, proposal: str, agent: str) -> str:
        """Generate prompt for strawman phase (identify if others are misrepresenting)."""
        return f"""Analyze whether opponents are STRAWMANNING this proposal.

## Original Proposal (by {agent})
{proposal}

## Your Task
1. What is the ACTUAL claim being made?
2. Are critics attacking the real claim or a distorted version?
3. Identify any misrepresentations
4. Clarify what {agent} is NOT claiming

This helps ensure the debate addresses real disagreements.
"""


class RedTeamMode:
    """
    Runs adversarial red-team debates.

    Features:
    - Systematic attack/defend cycles
    - Coverage of multiple vulnerability types
    - Steelman/strawman for intellectual honesty
    - Quantified risk assessment
    """

    def __init__(
        self,
        protocol: Optional[RedTeamProtocol] = None,
    ):
        self.protocol = protocol or RedTeamProtocol()
        self._attack_counter = 0
        self._defense_counter = 0

    async def run_redteam(
        self,
        target_proposal: str,
        proposer: str,
        red_team_agents: list[Any],
        run_agent_fn: Callable,
        max_rounds: int = 4,
    ) -> RedTeamResult:
        """
        Run a complete red-team session.

        Args:
            target_proposal: The proposal to test
            proposer: Name of the proposing agent
            red_team_agents: Agents that will attack
            run_agent_fn: Function to run an agent with a prompt
            max_rounds: Maximum attack/defend cycles
        """
        session_id = f"redteam-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        rounds = []
        all_attacks = []
        all_defenses = []

        current_proposal = target_proposal

        for round_num in range(1, max_rounds + 1):
            round_result = RedTeamRound(round_num=round_num, phase="attack")

            # Attack phase - each red team agent attacks
            for i, agent in enumerate(red_team_agents):
                # Rotate attack types across rounds
                attack_types = self.protocol.ATTACK_CATEGORIES[
                    i % len(self.protocol.ATTACK_CATEGORIES) :
                ]
                if not isinstance(attack_types, list):
                    attack_types = [attack_types]

                prompt = self.protocol.generate_attack_prompt(
                    current_proposal,
                    attack_types,
                    round_num,
                )

                response = await run_agent_fn(agent, prompt)
                attacks = self._parse_attacks(response, agent.name, proposer)
                round_result.attacks.extend(attacks)
                all_attacks.extend(attacks)

            # Defense phase - proposer defends
            if round_result.attacks:
                defend_prompt = self.protocol.generate_defend_prompt(
                    current_proposal,
                    round_result.attacks,
                    round_num,
                )

                # Would call proposer agent here
                # For now, mark as requiring defense
                round_result.phase = "defend"

            # Steelman round (if enabled and first round)
            if self.protocol.include_steelman and round_num == 1:
                steelman_round = RedTeamRound(round_num=round_num, phase="steelman")
                rounds.append(steelman_round)

            rounds.append(round_result)

        # Calculate results
        successful = [a for a in all_attacks if a.risk_score > 0.5]
        critical = [a for a in all_attacks if a.severity > 0.8]
        mitigated = [
            a
            for a in all_attacks
            if any(d.attack_id == a.attack_id and d.success for d in all_defenses)
        ]

        robustness = 1.0 - (len(successful) / len(all_attacks)) if all_attacks else 1.0

        return RedTeamResult(
            session_id=session_id,
            target=target_proposal[:200],
            rounds=rounds,
            total_attacks=len(all_attacks),
            successful_attacks=len(successful),
            critical_issues=critical,
            mitigated_issues=mitigated,
            accepted_risks=[],
            robustness_score=robustness,
            coverage_score=(
                len(set(a.attack_type for a in all_attacks)) / len(AttackType)
                if len(AttackType) > 0
                else 0.0
            ),
        )

    def _parse_attacks(
        self,
        response: str,
        attacker: str,
        target: str,
    ) -> list[Attack]:
        """Parse attacks from agent response."""
        attacks = []

        # Simple parsing - would use structured output in production
        lines = response.split("\n")
        current_attack = None

        for line in lines:
            line = line.strip()

            # Look for severity indicators
            severity = 0.5
            if "critical" in line.lower():
                severity = 0.9
            elif "high" in line.lower():
                severity = 0.7
            elif "medium" in line.lower():
                severity = 0.5
            elif "low" in line.lower():
                severity = 0.3

            # Look for attack type indicators
            attack_type = AttackType.LOGICAL_FALLACY
            for at in AttackType:
                if at.value.replace("_", " ") in line.lower():
                    attack_type = at
                    break

            # If line looks like an attack description
            if len(line) > 20 and any(
                word in line.lower()
                for word in [
                    "vulnerability",
                    "issue",
                    "problem",
                    "weakness",
                    "flaw",
                    "attack",
                    "exploit",
                    "could",
                    "might",
                    "fails",
                ]
            ):
                self._attack_counter += 1
                attack = Attack(
                    attack_id=f"attack-{self._attack_counter:04d}",
                    attack_type=attack_type,
                    attacker=attacker,
                    target_agent=target,
                    target_claim="",
                    attack_description=line[:500],
                    severity=severity,
                    exploitability=0.5,
                )
                attacks.append(attack)

        return attacks

    def generate_report(self, result: RedTeamResult) -> str:
        """Generate a Markdown report of red-team findings."""
        lines = [
            "# Red Team Report",
            "",
            f"**Session:** {result.session_id}",
            f"**Target:** {result.target[:100]}...",
            "",
            "---",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Attacks | {result.total_attacks} |",
            f"| Successful | {result.successful_attacks} |",
            f"| Robustness Score | {result.robustness_score:.0%} |",
            f"| Coverage Score | {result.coverage_score:.0%} |",
            "",
        ]

        if result.critical_issues:
            lines.append("## Critical Issues")
            lines.append("")
            for issue in result.critical_issues:
                lines.append(f"### {issue.attack_type.value}")
                lines.append(f"**Severity:** {issue.severity:.0%}")
                lines.append(f"**Attacker:** {issue.attacker}")
                lines.append("")
                lines.append(issue.attack_description)
                lines.append("")

        lines.append("## Rounds")
        lines.append("")
        for round_result in result.rounds:
            lines.append(f"### Round {round_result.round_num} ({round_result.phase})")
            lines.append(f"Attacks: {len(round_result.attacks)}")
            lines.append("")

        return "\n".join(lines)


# Convenience functions for common red-team scenarios
async def redteam_code_review(
    code: str,
    agents: list[Any],
    run_agent_fn: Callable,
) -> RedTeamResult:
    """Red-team a code review."""
    mode = RedTeamMode(
        RedTeamProtocol(
            attack_rounds=2,
            include_steelman=False,
        )
    )

    # Override attack categories for code
    mode.protocol.ATTACK_CATEGORIES = [
        AttackType.SECURITY,
        AttackType.EDGE_CASE,
        AttackType.RACE_CONDITION,
        AttackType.RESOURCE_EXHAUSTION,
    ]

    return await mode.run_redteam(
        target_proposal=f"Code to review:\n```\n{code}\n```",
        proposer="code_author",
        red_team_agents=agents,
        run_agent_fn=run_agent_fn,
    )


async def redteam_policy(
    policy: str,
    agents: list[Any],
    run_agent_fn: Callable,
) -> RedTeamResult:
    """Red-team a policy document."""
    mode = RedTeamMode(
        RedTeamProtocol(
            attack_rounds=3,
            include_steelman=True,
            include_strawman=True,
        )
    )

    mode.protocol.ATTACK_CATEGORIES = [
        AttackType.LOGICAL_FALLACY,
        AttackType.UNSTATED_ASSUMPTION,
        AttackType.EDGE_CASE,
        AttackType.COUNTEREXAMPLE,
    ]

    return await mode.run_redteam(
        target_proposal=policy,
        proposer="policy_author",
        red_team_agents=agents,
        run_agent_fn=run_agent_fn,
    )
