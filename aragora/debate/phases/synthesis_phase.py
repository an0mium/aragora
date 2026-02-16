"""
Dialectical synthesis phase for debate orchestration.

This module implements an explicit synthesis phase where opposing debate
positions are reconciled into a higher-order combined position. Instead of
merely voting for one side, it identifies thesis/antithesis from proposals
and critiques, then asks a synthesis agent to generate a novel position that
incorporates the strongest elements from each side.

The dialectical process follows the Hegelian pattern:
1. Thesis: The leading proposal (most votes or first proposed)
2. Antithesis: The strongest opposing position
3. Synthesis: A novel combined position that addresses concerns from both

Key classes:
- SynthesisConfig: Configuration for the dialectical synthesis phase
- DialecticalPosition: A distinct position identified from proposals
- SynthesisResult: The output of synthesis including provenance
- DialecticalSynthesizer: Main orchestrator for synthesis generation
"""

from __future__ import annotations

__all__ = [
    "SynthesisConfig",
    "DialecticalPosition",
    "SynthesisResult",
    "DialecticalSynthesizer",
]

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.core import Agent
    from aragora.debate.context import DebateContext

logger = logging.getLogger(__name__)


# =========================================================================
# Configuration
# =========================================================================


@dataclass
class SynthesisConfig:
    """Configuration for the dialectical synthesis phase.

    Attributes:
        enable_synthesis: Master switch for the synthesis phase.
        max_synthesis_attempts: Maximum retries for synthesis generation.
        min_opposing_positions: Minimum distinct positions required to
            perform synthesis. With fewer than this, synthesis is skipped.
        synthesis_confidence_threshold: Minimum confidence score (0-1)
            for the synthesis to be considered successful.
        prefer_agent: Specific agent name to use for synthesis generation.
            When None, the synthesizer selects the best-calibrated agent.
    """

    enable_synthesis: bool = True
    max_synthesis_attempts: int = 3
    min_opposing_positions: int = 2
    synthesis_confidence_threshold: float = 0.5
    prefer_agent: str | None = None


# =========================================================================
# Data structures
# =========================================================================


@dataclass
class DialecticalPosition:
    """A distinct position identified from debate proposals.

    Represents either the thesis or antithesis in the dialectical
    framework, with metadata about which agents support it.

    Attributes:
        agent: Name of the primary agent who proposed this position.
        content: The full text of the position/proposal.
        key_claims: Extracted claims or key arguments from the position.
        supporting_agents: Names of other agents who voted for or
            expressed support for this position.
    """

    agent: str
    content: str
    key_claims: list[str] = field(default_factory=list)
    supporting_agents: list[str] = field(default_factory=list)


@dataclass
class SynthesisResult:
    """The output of a dialectical synthesis.

    Contains the combined position along with provenance information
    showing which elements were drawn from each side and what novel
    ideas emerged during synthesis.

    Attributes:
        synthesis: The combined position text.
        confidence: Score (0-1) reflecting how well the synthesis
            addresses concerns from all positions.
        thesis: The primary position (typically the leading proposal).
        antithesis: The opposing position.
        elements_from_thesis: Key points retained from the thesis.
        elements_from_antithesis: Key points retained from the antithesis.
        novel_elements: New ideas that emerged in the synthesis.
        synthesizer_agent: Name of the agent that generated the synthesis.
    """

    synthesis: str
    confidence: float
    thesis: DialecticalPosition
    antithesis: DialecticalPosition
    elements_from_thesis: list[str] = field(default_factory=list)
    elements_from_antithesis: list[str] = field(default_factory=list)
    novel_elements: list[str] = field(default_factory=list)
    synthesizer_agent: str = ""


# =========================================================================
# Claim extraction helpers
# =========================================================================


def _extract_claims(text: str) -> list[str]:
    """Extract key claims from a position text.

    Uses simple sentence-level heuristics to identify claims.
    Sentences containing assertion markers (should, must, is, are,
    will, can, need) are treated as claims.

    Args:
        text: The proposal or position text.

    Returns:
        List of extracted claim strings (up to 5).
    """
    if not text:
        return []

    # Split into sentences on common terminators
    sentences: list[str] = []
    for raw_sentence in text.replace("\n", " ").split("."):
        stripped = raw_sentence.strip()
        if len(stripped) > 15:
            sentences.append(stripped)

    # Filter to sentences that look like claims (contain assertion words)
    assertion_markers = {"should", "must", "is", "are", "will", "can", "need", "require"}
    claims: list[str] = []
    for sentence in sentences:
        words = set(sentence.lower().split())
        if words & assertion_markers:
            claims.append(sentence)

    return claims[:5]


# =========================================================================
# Synthesis prompt builder
# =========================================================================


def _build_synthesis_prompt(
    thesis: DialecticalPosition,
    antithesis: DialecticalPosition,
    task: str,
) -> str:
    """Build the prompt for a synthesis agent.

    Args:
        thesis: The primary position.
        antithesis: The opposing position.
        task: The original debate task/question.

    Returns:
        Formatted prompt string.
    """
    thesis_claims = "\n".join(f"- {c}" for c in thesis.key_claims) if thesis.key_claims else "(none extracted)"
    antithesis_claims = "\n".join(f"- {c}" for c in antithesis.key_claims) if antithesis.key_claims else "(none extracted)"

    return f"""You are tasked with generating a dialectical synthesis from two opposing debate positions.

## Original Question
{task}

## Thesis ({thesis.agent})
{thesis.content}

Key claims:
{thesis_claims}

Supporters: {', '.join(thesis.supporting_agents) if thesis.supporting_agents else 'none'}

## Antithesis ({antithesis.agent})
{antithesis.content}

Key claims:
{antithesis_claims}

Supporters: {', '.join(antithesis.supporting_agents) if antithesis.supporting_agents else 'none'}

## Your Task
Generate a SYNTHESIS that:
1. Identifies the strongest elements from BOTH positions
2. Addresses the concerns and critiques raised by each side
3. Produces a novel combined position that is stronger than either alone
4. Explicitly states which elements come from each position
5. Notes any genuinely new ideas that emerge from the combination

Structure your response as:
### Synthesis
[Your combined position]

### Elements from Thesis
- [bullet points of what was kept from the thesis]

### Elements from Antithesis
- [bullet points of what was kept from the antithesis]

### Novel Elements
- [bullet points of new ideas that emerged]

Write clearly and authoritatively. The synthesis should resolve the tension between the two positions."""


# =========================================================================
# Synthesis output parser
# =========================================================================


def _parse_synthesis_output(raw: str) -> tuple[str, list[str], list[str], list[str]]:
    """Parse structured synthesis output from an agent.

    Extracts the synthesis text and element lists from the agent's
    structured response. Falls back gracefully when sections are missing.

    Args:
        raw: Raw text output from the synthesis agent.

    Returns:
        Tuple of (synthesis_text, elements_from_thesis,
                  elements_from_antithesis, novel_elements).
    """
    synthesis_text = raw
    elements_thesis: list[str] = []
    elements_antithesis: list[str] = []
    novel: list[str] = []

    sections = raw.split("###")
    for section in sections:
        stripped = section.strip()
        lower = stripped.lower()
        if lower.startswith("synthesis"):
            # Everything after the heading
            lines = stripped.split("\n", 1)
            synthesis_text = lines[1].strip() if len(lines) > 1 else stripped
        elif lower.startswith("elements from thesis"):
            elements_thesis = _extract_bullet_points(stripped)
        elif lower.startswith("elements from antithesis"):
            elements_antithesis = _extract_bullet_points(stripped)
        elif lower.startswith("novel"):
            novel = _extract_bullet_points(stripped)

    return synthesis_text, elements_thesis, elements_antithesis, novel


def _extract_bullet_points(section: str) -> list[str]:
    """Extract bullet points from a section of text.

    Args:
        section: A section that may contain bullet points (lines starting
            with '-' or '*').

    Returns:
        List of extracted bullet point strings.
    """
    items: list[str] = []
    for line in section.split("\n"):
        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            items.append(stripped[2:].strip())
    return items


# =========================================================================
# Main synthesizer
# =========================================================================


class DialecticalSynthesizer:
    """Generates novel synthesis from opposing debate positions.

    Instead of just voting for one side, identifies the strongest elements
    from each position and constructs a combined position that addresses
    the concerns raised by all sides.

    Usage:
        config = SynthesisConfig(max_synthesis_attempts=3)
        synthesizer = DialecticalSynthesizer(config)
        result = await synthesizer.synthesize(ctx)

    The synthesizer requires at least two distinct positions to operate.
    If only one position exists, it returns None (synthesis not applicable).
    """

    def __init__(self, config: SynthesisConfig | None = None) -> None:
        """Initialize the dialectical synthesizer.

        Args:
            config: Configuration for synthesis behavior. Uses defaults
                if not provided.
        """
        self.config = config or SynthesisConfig()

    async def synthesize(self, ctx: DebateContext) -> SynthesisResult | None:
        """Run dialectical synthesis on the debate.

        Steps:
        1. Identify thesis/antithesis from proposals + votes
        2. Extract key claims from each position
        3. Ask a synthesis agent to generate a combined position
        4. Validate synthesis addresses the critiques

        Args:
            ctx: The DebateContext with proposals, votes, and agents.

        Returns:
            SynthesisResult with the combined position, or None if
            synthesis is not applicable (e.g., too few distinct positions
            or synthesis is disabled).
        """
        if not self.config.enable_synthesis:
            logger.info("dialectical_synthesis_disabled")
            return None

        proposals = ctx.proposals
        if not proposals:
            logger.info("dialectical_synthesis_skip reason=no_proposals")
            return None

        # Step 1: Identify distinct positions
        votes = ctx.result.votes if ctx.result else []
        positions = self._identify_positions(proposals, votes)

        if len(positions) < self.config.min_opposing_positions:
            logger.info(
                "dialectical_synthesis_skip reason=insufficient_positions "
                "found=%d required=%d",
                len(positions),
                self.config.min_opposing_positions,
            )
            return None

        thesis = positions[0]
        antithesis = positions[1]

        logger.info(
            "dialectical_synthesis_start thesis=%s antithesis=%s",
            thesis.agent,
            antithesis.agent,
        )

        # Step 2: Select synthesis agent
        agent = self._select_synthesis_agent(ctx)
        if agent is None:
            logger.warning("dialectical_synthesis_skip reason=no_agent_available")
            return None

        # Step 3: Generate synthesis with retries
        task = ctx.env.task if ctx.env else ""
        synthesis_text: str | None = None

        for attempt in range(self.config.max_synthesis_attempts):
            try:
                synthesis_text = await self._generate_synthesis(
                    agent, thesis, antithesis, task
                )
                if synthesis_text:
                    break
            except (RuntimeError, OSError, ConnectionError, TimeoutError) as exc:
                logger.warning(
                    "dialectical_synthesis_attempt_failed attempt=%d error=%s: %s",
                    attempt + 1,
                    type(exc).__name__,
                    exc,
                )

        if not synthesis_text:
            logger.warning("dialectical_synthesis_failed all attempts exhausted")
            return None

        # Step 4: Parse structured output
        parsed_synthesis, elements_thesis, elements_antithesis, novel = (
            _parse_synthesis_output(synthesis_text)
        )

        # Step 5: Validate synthesis
        confidence = self._validate_synthesis(parsed_synthesis, positions)

        if confidence < self.config.synthesis_confidence_threshold:
            logger.info(
                "dialectical_synthesis_low_confidence confidence=%.2f threshold=%.2f",
                confidence,
                self.config.synthesis_confidence_threshold,
            )

        result = SynthesisResult(
            synthesis=parsed_synthesis,
            confidence=confidence,
            thesis=thesis,
            antithesis=antithesis,
            elements_from_thesis=elements_thesis,
            elements_from_antithesis=elements_antithesis,
            novel_elements=novel,
            synthesizer_agent=agent.name,
        )

        logger.info(
            "dialectical_synthesis_complete synthesizer=%s confidence=%.2f "
            "thesis_elements=%d antithesis_elements=%d novel=%d",
            agent.name,
            confidence,
            len(elements_thesis),
            len(elements_antithesis),
            len(novel),
        )

        return result

    def _identify_positions(
        self,
        proposals: dict[str, str],
        votes: list[Any],
    ) -> list[DialecticalPosition]:
        """Identify distinct positions (thesis/antithesis) from proposals.

        Positions are ranked by vote support. The position with the most
        votes becomes the thesis, and the runner-up becomes the antithesis.
        If no votes are available, the first two proposals are used by
        insertion order.

        Args:
            proposals: Agent name to proposal text mapping.
            votes: List of Vote objects from the voting phase.

        Returns:
            List of DialecticalPosition objects, sorted by support
            (most supported first).
        """
        if not proposals:
            return []

        # Count votes per agent
        vote_counts: dict[str, int] = {}
        for vote in votes:
            choice = getattr(vote, "choice", None)
            if choice:
                vote_counts[choice] = vote_counts.get(choice, 0) + 1

        # Build positions for each proposal
        positions: list[DialecticalPosition] = []
        for agent_name, content in proposals.items():
            # Find supporters: other agents who voted for this one
            supporters: list[str] = []
            for vote in votes:
                voter = getattr(vote, "agent", None)
                choice = getattr(vote, "choice", None)
                if choice == agent_name and voter != agent_name:
                    supporters.append(voter)

            positions.append(
                DialecticalPosition(
                    agent=agent_name,
                    content=content,
                    key_claims=_extract_claims(content),
                    supporting_agents=supporters,
                )
            )

        # Sort by vote count descending, breaking ties by proposal order
        agent_order = list(proposals.keys())
        positions.sort(
            key=lambda p: (-vote_counts.get(p.agent, 0), agent_order.index(p.agent)),
        )

        return positions

    def _select_synthesis_agent(self, ctx: DebateContext) -> Agent | None:
        """Select the best agent for synthesis generation.

        Selection priority:
        1. If prefer_agent is set in config, use that agent.
        2. Otherwise, select the agent with role "synthesizer" if any.
        3. Fall back to the first available agent.

        Args:
            ctx: The DebateContext with available agents.

        Returns:
            The selected Agent, or None if no agents are available.
        """
        if not ctx.agents:
            return None

        # Priority 1: Preferred agent from config
        if self.config.prefer_agent:
            for agent in ctx.agents:
                if agent.name == self.config.prefer_agent:
                    logger.debug(
                        "synthesis_agent_selected agent=%s reason=prefer_agent",
                        agent.name,
                    )
                    return agent
            logger.debug(
                "synthesis_preferred_agent_not_found preferred=%s",
                self.config.prefer_agent,
            )

        # Priority 2: Agent with synthesizer role
        for agent in ctx.agents:
            if getattr(agent, "role", None) == "synthesizer":
                logger.debug(
                    "synthesis_agent_selected agent=%s reason=synthesizer_role",
                    agent.name,
                )
                return agent

        # Priority 3: First available agent
        agent = ctx.agents[0]
        logger.debug(
            "synthesis_agent_selected agent=%s reason=first_available",
            agent.name,
        )
        return agent

    async def _generate_synthesis(
        self,
        agent: Agent,
        thesis: DialecticalPosition,
        antithesis: DialecticalPosition,
        task: str,
    ) -> str | None:
        """Ask an agent to generate a synthesis position.

        Args:
            agent: The agent to use for generation.
            thesis: The primary position.
            antithesis: The opposing position.
            task: The original debate task/question.

        Returns:
            The generated synthesis text, or None on failure.
        """
        prompt = _build_synthesis_prompt(thesis, antithesis, task)

        logger.debug(
            "synthesis_generation_start agent=%s prompt_length=%d",
            agent.name,
            len(prompt),
        )

        result = await agent.generate(prompt, context=None)

        if not result or not result.strip():
            logger.warning(
                "synthesis_generation_empty agent=%s",
                agent.name,
            )
            return None

        logger.debug(
            "synthesis_generation_complete agent=%s length=%d",
            agent.name,
            len(result),
        )

        return result

    def _validate_synthesis(
        self,
        synthesis: str,
        positions: list[DialecticalPosition],
    ) -> float:
        """Score how well the synthesis addresses concerns from all positions.

        Uses a heuristic scoring approach:
        - Base score: 0.3 (for any non-empty synthesis)
        - +0.1 for each position whose key claims are referenced
        - +0.1 for sufficient length (>100 chars)
        - +0.1 for mentioning multiple agents by name
        - +0.2 for containing structured sections

        The maximum score is 1.0.

        Args:
            synthesis: The generated synthesis text.
            positions: All identified positions in the debate.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if not synthesis:
            return 0.0

        score = 0.3  # Base score for non-empty synthesis
        synthesis_lower = synthesis.lower()

        # Check if key claims from each position are addressed
        positions_addressed = 0
        for position in positions[:2]:  # Only check thesis/antithesis
            claims_found = 0
            for claim in position.key_claims:
                # Check if any significant words from the claim appear
                claim_words = set(claim.lower().split())
                # Remove common words
                significant_words = claim_words - {
                    "the", "a", "an", "is", "are", "was", "were", "be",
                    "to", "of", "in", "for", "on", "and", "or", "it",
                    "that", "this", "with", "as", "at", "by", "from",
                }
                if significant_words:
                    matches = sum(1 for w in significant_words if w in synthesis_lower)
                    if matches >= len(significant_words) * 0.3:
                        claims_found += 1

            if claims_found > 0:
                positions_addressed += 1

        score += min(0.2, positions_addressed * 0.1)

        # Length check
        if len(synthesis) > 100:
            score += 0.1

        # Agent name references
        agents_mentioned = sum(
            1 for p in positions if p.agent.lower() in synthesis_lower
        )
        if agents_mentioned >= 2:
            score += 0.1

        # Structured output check
        structured_markers = ["###", "**", "elements from", "novel"]
        structure_count = sum(1 for m in structured_markers if m in synthesis_lower)
        if structure_count >= 2:
            score += 0.2

        return min(1.0, score)
