"""
Epistemic Hygiene enforcement for debate agents.

When ``enable_epistemic_hygiene`` is set on a DebateProtocol, this module:

1. **Prompt injection**: Adds structured requirements to proposal and revision
   prompts so agents include alternatives, falsifiers, confidence intervals,
   and explicit unknowns.

2. **Response scoring**: Parses agent responses and scores them on four
   epistemic dimensions (alternatives, falsifiers, confidence, unknowns).

3. **Consensus penalty**: Computes a penalty multiplier (0-1) that
   VoteBonusCalculator applies to vote weights for proposals lacking
   epistemic rigour.

Scoring rules
-------------
Each of the four dimensions is scored 0 or 1 (present / absent).  The
composite *epistemic score* is the mean of the four binary scores.
Proposals with a score < 1.0 receive a consensus weight penalty of::

    penalty = protocol.epistemic_hygiene_penalty * (1.0 - epistemic_score)

so a proposal missing all four elements is penalized by the full
``epistemic_hygiene_penalty`` (default 0.15).

Thread-safety
-------------
All functions are pure or operate on per-call state; the module carries no
mutable module-level state.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt text injected when epistemic hygiene is enabled
# ---------------------------------------------------------------------------

_EPISTEMIC_PROPOSAL_REQUIREMENTS = """
## EPISTEMIC HYGIENE REQUIREMENTS
You MUST include ALL of the following sections in your response:

### ALTERNATIVES CONSIDERED
List at least {min_alternatives} alternative approach(es) you considered and
explain why you rejected each one.  Format:
- **Alternative:** <description> | **Rejected because:** <reason>

### FALSIFIABILITY
For each major claim, state what evidence would disprove it.  Format:
- **Claim:** <your claim> | **Falsified if:** <condition>

### CONFIDENCE LEVELS
Assign a confidence level (0.0-1.0) to each major claim.  Format:
- **Claim:** <your claim> | **Confidence:** <0.0-1.0>

### EXPLICIT UNKNOWNS
List what you do NOT know or are uncertain about that is relevant to this task.
- <unknown or uncertainty>

Failure to include these sections will reduce the weight of your proposal
in the consensus decision.
"""

_EPISTEMIC_REVISION_REQUIREMENTS = """
## EPISTEMIC HYGIENE REQUIREMENTS (Revision)
Your revised proposal MUST retain or update the following sections:

### ALTERNATIVES CONSIDERED
Update with any new alternatives surfaced by critiques.
- **Alternative:** <description> | **Rejected because:** <reason>

### FALSIFIABILITY
Update falsifiers in light of critique feedback.
- **Claim:** <your claim> | **Falsified if:** <condition>

### CONFIDENCE LEVELS
Re-calibrate your confidence levels after considering critiques.
- **Claim:** <your claim> | **Confidence:** <0.0-1.0>

### EXPLICIT UNKNOWNS
Update unknowns: what has been resolved, and what remains uncertain?
- <unknown or uncertainty>
"""


def get_epistemic_proposal_prompt(protocol: Any) -> str:
    """Return the epistemic hygiene prompt section for proposals.

    Args:
        protocol: DebateProtocol with epistemic hygiene configuration.

    Returns:
        Formatted prompt string, or empty string if disabled.
    """
    if not getattr(protocol, "enable_epistemic_hygiene", False):
        return ""
    min_alt = getattr(protocol, "epistemic_min_alternatives", 1)
    return _EPISTEMIC_PROPOSAL_REQUIREMENTS.format(min_alternatives=min_alt).strip()


def get_epistemic_revision_prompt(protocol: Any) -> str:
    """Return the epistemic hygiene prompt section for revisions.

    Args:
        protocol: DebateProtocol with epistemic hygiene configuration.

    Returns:
        Formatted prompt string, or empty string if disabled.
    """
    if not getattr(protocol, "enable_epistemic_hygiene", False):
        return ""
    return _EPISTEMIC_REVISION_REQUIREMENTS.strip()


# ---------------------------------------------------------------------------
# Response scoring
# ---------------------------------------------------------------------------

# Regex patterns for detecting epistemic elements in agent responses.
# These are intentionally lenient to match natural-language variations.

_ALTERNATIVES_PATTERNS = [
    re.compile(r"(?i)\balternative\s*:", re.MULTILINE),
    re.compile(r"(?i)rejected\s+because", re.MULTILINE),
    re.compile(r"(?i)instead\s+of\s+this", re.MULTILINE),
    re.compile(r"(?i)another\s+option", re.MULTILINE),
    re.compile(r"(?i)we?\s+could\s+also", re.MULTILINE),
]

_FALSIFIER_PATTERNS = [
    re.compile(r"(?i)falsified\s+if", re.MULTILINE),
    re.compile(r"(?i)falsifier\s*:", re.MULTILINE),
    re.compile(r"(?i)disprove[dns]?\s+(?:if|by|when)", re.MULTILINE),
    re.compile(r"(?i)would\s+be\s+(?:wrong|false|invalidated)\s+if", re.MULTILINE),
    re.compile(r"(?i)refuted\s+(?:if|by|when)", re.MULTILINE),
    re.compile(r"(?i)evidence\s+against", re.MULTILINE),
]

_CONFIDENCE_PATTERNS = [
    re.compile(r"(?i)confidence[:\s]+(?:0\.\d+|1\.0|[01])", re.MULTILINE),
    re.compile(r"(?i)\b(?:high|medium|low)\s+confidence\b", re.MULTILINE),
    re.compile(r"\b(?:0\.[0-9]{1,2}|1\.0)\b"),  # bare float 0.xx or 1.0
]

_UNKNOWNS_PATTERNS = [
    re.compile(r"(?i)\bunknown[s]?\s*:", re.MULTILINE),
    re.compile(r"(?i)\buncertain\b", re.MULTILINE),
    re.compile(r"(?i)uncertain(?:ty|ties)", re.MULTILINE),
    re.compile(r"(?i)(?:do|does)\s+not\s+know", re.MULTILINE),
    re.compile(r"(?i)unclear|unresolved|open\s+question", re.MULTILINE),
    re.compile(r"(?i)limitation[s]?\s+(?:of|in)", re.MULTILINE),
]


def _has_section(text: str, patterns: list[re.Pattern]) -> bool:  # type: ignore[type-arg]
    """Return True if *text* matches at least one pattern from *patterns*."""
    for pat in patterns:
        if pat.search(text):
            return True
    return False


@dataclass
class EpistemicScore:
    """Per-agent, per-round epistemic hygiene score.

    Each dimension is a boolean indicating presence (True) or absence (False).
    The composite ``score`` is the mean of all four dimensions (0.0-1.0).
    """

    has_alternatives: bool = False
    has_falsifiers: bool = False
    has_confidence: bool = False
    has_unknowns: bool = False
    agent: str = ""
    round_number: int = 0

    @property
    def score(self) -> float:
        """Composite epistemic score (0.0 = none present, 1.0 = all present)."""
        total = sum(
            [
                self.has_alternatives,
                self.has_falsifiers,
                self.has_confidence,
                self.has_unknowns,
            ]
        )
        return total / 4.0

    @property
    def missing(self) -> list[str]:
        """Return names of missing epistemic elements."""
        names = []
        if not self.has_alternatives:
            names.append("alternatives")
        if not self.has_falsifiers:
            names.append("falsifiers")
        if not self.has_confidence:
            names.append("confidence_levels")
        if not self.has_unknowns:
            names.append("explicit_unknowns")
        return names

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for audit/analytics."""
        return {
            "agent": self.agent,
            "round_number": self.round_number,
            "has_alternatives": self.has_alternatives,
            "has_falsifiers": self.has_falsifiers,
            "has_confidence": self.has_confidence,
            "has_unknowns": self.has_unknowns,
            "score": self.score,
            "missing": self.missing,
        }


def score_response(text: str, agent: str = "", round_number: int = 0) -> EpistemicScore:
    """Score an agent response on epistemic hygiene dimensions.

    Args:
        text: The agent's full response text.
        agent: Agent name (for tracking).
        round_number: Current debate round (for tracking).

    Returns:
        EpistemicScore with per-dimension booleans and composite score.
    """
    return EpistemicScore(
        has_alternatives=_has_section(text, _ALTERNATIVES_PATTERNS),
        has_falsifiers=_has_section(text, _FALSIFIER_PATTERNS),
        has_confidence=_has_section(text, _CONFIDENCE_PATTERNS),
        has_unknowns=_has_section(text, _UNKNOWNS_PATTERNS),
        agent=agent,
        round_number=round_number,
    )


# ---------------------------------------------------------------------------
# Consensus penalty
# ---------------------------------------------------------------------------


def compute_epistemic_penalty(
    score: EpistemicScore,
    protocol: Any,
) -> float:
    """Compute the consensus weight penalty for an epistemic score.

    The penalty is subtracted from the proposal's vote weight.  A fully
    compliant proposal (score=1.0) receives zero penalty.

    Args:
        score: EpistemicScore for the proposal.
        protocol: DebateProtocol with ``epistemic_hygiene_penalty``.

    Returns:
        Penalty value in [0.0, ``epistemic_hygiene_penalty``].
    """
    if not getattr(protocol, "enable_epistemic_hygiene", False):
        return 0.0

    base_penalty = getattr(protocol, "epistemic_hygiene_penalty", 0.15)

    # Only penalize dimensions that are actually required
    required_count = 0
    missing_count = 0

    if getattr(protocol, "epistemic_min_alternatives", 1) > 0:
        required_count += 1
        if not score.has_alternatives:
            missing_count += 1

    if getattr(protocol, "epistemic_require_falsifiers", True):
        required_count += 1
        if not score.has_falsifiers:
            missing_count += 1

    if getattr(protocol, "epistemic_require_confidence", True):
        required_count += 1
        if not score.has_confidence:
            missing_count += 1

    if getattr(protocol, "epistemic_require_unknowns", True):
        required_count += 1
        if not score.has_unknowns:
            missing_count += 1

    if required_count == 0:
        return 0.0

    fraction_missing = missing_count / required_count
    return base_penalty * fraction_missing


# ---------------------------------------------------------------------------
# Per-debate tracker
# ---------------------------------------------------------------------------


@dataclass
class EpistemicHygieneTracker:
    """Tracks epistemic hygiene scores across rounds for a single debate.

    Instantiated per-debate when ``enable_epistemic_hygiene`` is True.
    Records scores for every agent response and provides aggregate statistics.
    """

    debate_id: str = ""
    scores: list[EpistemicScore] = field(default_factory=list)

    def record(self, score: EpistemicScore) -> None:
        """Record an epistemic score."""
        self.scores.append(score)
        if score.missing:
            logger.debug(
                "[epistemic] Agent %s round %d missing: %s (score=%.2f)",
                score.agent,
                score.round_number,
                ", ".join(score.missing),
                score.score,
            )

    def get_agent_scores(self, agent: str) -> list[EpistemicScore]:
        """Get all scores for a specific agent."""
        return [s for s in self.scores if s.agent == agent]

    def get_agent_average(self, agent: str) -> float:
        """Get average epistemic score for an agent across all rounds."""
        agent_scores = self.get_agent_scores(agent)
        if not agent_scores:
            return 0.0
        return sum(s.score for s in agent_scores) / len(agent_scores)

    def get_round_scores(self, round_number: int) -> list[EpistemicScore]:
        """Get all scores for a specific round."""
        return [s for s in self.scores if s.round_number == round_number]

    def get_debate_average(self) -> float:
        """Get overall average epistemic score for the debate."""
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)

    def summary(self) -> dict[str, Any]:
        """Generate summary statistics for the debate.

        Returns:
            Dictionary with aggregate stats suitable for audit trails.
        """
        agents: dict[str, list[EpistemicScore]] = {}
        for s in self.scores:
            agents.setdefault(s.agent, []).append(s)

        agent_summaries = {}
        for agent, scores in agents.items():
            avg = sum(s.score for s in scores) / len(scores) if scores else 0.0
            agent_summaries[agent] = {
                "average_score": round(avg, 3),
                "rounds_scored": len(scores),
                "fully_compliant": sum(1 for s in scores if s.score == 1.0),
            }

        return {
            "debate_id": self.debate_id,
            "total_scores": len(self.scores),
            "debate_average": round(self.get_debate_average(), 3),
            "agents": agent_summaries,
        }
