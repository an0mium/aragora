"""
Disagreement analysis for multi-agent debates.

Extracts and reports on areas of agreement and disagreement from debate
votes and critiques, surfacing unanimous critiques (high confidence issues)
and split opinions (risk areas requiring attention).
"""

__all__ = [
    "DisagreementReporter",
]

from collections import Counter
from typing import Optional

from aragora.core import Critique, DisagreementReport, Vote


class DisagreementReporter:
    """
    Generate disagreement reports from debate votes and critiques.

    Inspired by Heavy3.ai: surfaces unanimous critiques (high confidence issues)
    and split opinions (risk areas requiring attention).
    """

    def __init__(
        self,
        low_confidence_threshold: float = 0.6,
        high_severity_threshold: float = 0.7,
        max_risk_areas: int = 5,
        max_severe_critiques: int = 3,
    ):
        """
        Initialize the disagreement reporter.

        Args:
            low_confidence_threshold: Votes below this confidence are flagged
            high_severity_threshold: Critiques above this severity are flagged
            max_risk_areas: Maximum number of low-confidence risk areas to report
            max_severe_critiques: Maximum number of severe critiques to report
        """
        self.low_confidence_threshold = low_confidence_threshold
        self.high_severity_threshold = high_severity_threshold
        self.max_risk_areas = max_risk_areas
        self.max_severe_critiques = max_severe_critiques

    def generate_report(
        self,
        votes: list[Vote],
        critiques: list[Critique],
        winner: Optional[str] = None,
    ) -> DisagreementReport:
        """
        Generate a DisagreementReport from debate votes and critiques.

        Args:
            votes: List of agent votes from the debate
            critiques: List of critiques raised during the debate
            winner: The winning choice/agent, if determined

        Returns:
            DisagreementReport with agreement scores, unanimous critiques,
            split opinions, and risk areas
        """
        report = DisagreementReport()

        if not votes and not critiques:
            return report

        # 1. Analyze vote alignment
        vote_choices = {v.agent: v.choice for v in votes}

        # Calculate agreement score
        if len(vote_choices) > 1:
            choice_counts = Counter(vote_choices.values())
            most_common_list = choice_counts.most_common(1) if choice_counts else []
            most_common_count = most_common_list[0][1] if most_common_list else 0
            report.agreement_score = most_common_count / len(vote_choices)

        # Calculate per-agent alignment with winner
        if winner:
            for agent, choice in vote_choices.items():
                report.agent_alignment[agent] = 1.0 if choice == winner else 0.0

        # 2. Find unanimous critiques (all critics agree on an issue)
        report.unanimous_critiques = self._find_unanimous_critiques(critiques)

        # 3. Find split opinions from votes
        report.split_opinions = self._find_split_opinions(vote_choices)

        # 4. Identify risk areas from low-confidence votes
        report.risk_areas = self._find_risk_areas(votes, critiques, winner)

        return report

    def _find_unanimous_critiques(self, critiques: list[Critique]) -> list[str]:
        """
        Find critiques that all critics agree on.

        Args:
            critiques: List of critiques from the debate

        Returns:
            List of issue texts that all critics raised
        """
        unanimous = []

        # Map normalized issue text to set of agents who raised it
        issue_agents: dict[str, set[str]] = {}
        for critique in critiques:
            for issue in critique.issues:
                issue_key = issue.lower().strip()[:100]  # Normalize for matching
                if issue_key not in issue_agents:
                    issue_agents[issue_key] = set()
                issue_agents[issue_key].add(critique.agent)

        # Unanimous = raised by all critics
        critic_agents = set(c.agent for c in critiques)
        if len(critic_agents) > 1:
            for issue_key, agents in issue_agents.items():
                if agents == critic_agents:
                    # Find the original full issue text
                    for critique in critiques:
                        for orig_issue in critique.issues:
                            if orig_issue.lower().strip()[:100] == issue_key:
                                unanimous.append(orig_issue)
                                break
                        if issue_key in [uc.lower().strip()[:100] for uc in unanimous]:
                            break

        return unanimous

    def _find_split_opinions(
        self, vote_choices: dict[str, str]
    ) -> list[tuple[str, list[str], list[str]]]:
        """
        Find split opinions where agents voted differently.

        Args:
            vote_choices: Mapping of agent name to their vote choice

        Returns:
            List of (description, majority_agents, minority_agents) tuples
        """
        split_opinions: list[tuple[str, list[str], list[str]]] = []

        if len(vote_choices) <= 1:
            return split_opinions

        unique_choices = set(vote_choices.values())
        if len(unique_choices) <= 1:
            return split_opinions

        # Group agents by their choice
        choice_to_agents: dict[str, list[str]] = {}
        for agent, choice in vote_choices.items():
            if choice not in choice_to_agents:
                choice_to_agents[choice] = []
            choice_to_agents[choice].append(agent)

        # Create split opinion entries
        sorted_choices = sorted(choice_to_agents.items(), key=lambda x: -len(x[1]))
        if len(sorted_choices) >= 2:
            majority_choice, majority_agents = sorted_choices[0]
            for minority_choice, minority_agents in sorted_choices[1:]:
                split_opinions.append(
                    (
                        f"Vote split: '{majority_choice[:50]}...' vs '{minority_choice[:50]}...'",
                        majority_agents,
                        minority_agents,
                    )
                )

        return split_opinions

    def _find_risk_areas(
        self,
        votes: list[Vote],
        critiques: list[Critique],
        winner: Optional[str],
    ) -> list[str]:
        """
        Identify risk areas from low-confidence votes and unaddressed critiques.

        Args:
            votes: List of agent votes
            critiques: List of critiques from the debate
            winner: The winning choice/agent, if determined

        Returns:
            List of risk area descriptions
        """
        risk_areas = []

        # Low-confidence votes
        for vote in votes:
            if vote.confidence < self.low_confidence_threshold:
                risk_areas.append(
                    f"{vote.agent} has low confidence ({vote.confidence:.0%}) "
                    f"in '{vote.choice[:50]}...'"
                )
        risk_areas = risk_areas[: self.max_risk_areas]

        # High-severity critiques of the winner (issues may remain)
        severe_unaddressed = []
        for critique in critiques:
            if critique.severity >= self.high_severity_threshold:
                # Check if the critique target won (meaning issues may remain)
                if winner and critique.target_agent == winner:
                    severe_unaddressed.append(
                        f"High-severity ({critique.severity:.0%}) critique of winner "
                        f"{critique.target_agent}: "
                        f"{critique.issues[0][:100] if critique.issues else 'various issues'}"
                    )
        risk_areas.extend(severe_unaddressed[: self.max_severe_critiques])

        return risk_areas
