"""Live Explainability Stream — real-time factor decomposition during debates.

Fuses the spectator event stream with incremental explanation building,
producing live factor updates as evidence accumulates, votes shift,
and beliefs change during an active debate.

Usage:
    from aragora.explainability.live_stream import LiveExplainabilityStream

    stream = LiveExplainabilityStream(event_emitter=emitter)
    stream.on_proposal("claude", "We should use rate limiting", round_num=1)
    stream.on_critique("gpt4", "Rate limiting alone is insufficient", round_num=1)
    stream.on_vote("claude", "rate_limiting", confidence=0.85, round_num=2)
    snapshot = stream.get_snapshot()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LiveEvidence:
    """A piece of evidence accumulated during the debate."""

    content: str
    source: str  # Agent name
    evidence_type: str  # "proposal", "critique", "refinement"
    round_num: int
    timestamp: float = field(default_factory=time.time)
    relevance: float = 1.0
    cited_by: list[str] = field(default_factory=list)


@dataclass
class LiveVote:
    """A vote recorded during the debate."""

    agent: str
    choice: str
    confidence: float
    round_num: int
    reasoning: str = ""
    weight: float = 1.0
    flipped: bool = False  # Did this agent change their vote?


@dataclass
class LiveBeliefShift:
    """A tracked belief change for an agent."""

    agent: str
    round_num: int
    topic: str
    prior_confidence: float
    posterior_confidence: float
    trigger: str = ""  # What caused the shift


@dataclass
class LiveFactor:
    """A real-time confidence attribution factor."""

    name: str
    contribution: float  # 0.0 to 1.0
    explanation: str
    trend: str = "stable"  # "rising", "falling", "stable"
    raw_value: float = 0.0


@dataclass
class ExplanationSnapshot:
    """Point-in-time snapshot of the evolving explanation."""

    round_num: int
    timestamp: float
    evidence_count: int
    vote_count: int
    belief_shifts: int
    top_factors: list[dict[str, Any]]
    leading_position: str | None
    position_confidence: float
    agent_agreement: float
    evidence_quality: float
    narrative: str


class LiveExplainabilityStream:
    """Incrementally builds explanations as debate events flow in.

    Listens to spectator-style events and maintains a running
    explanation state that can be snapshotted at any point.
    """

    def __init__(
        self,
        debate_id: str = "",
        event_emitter: Any | None = None,
    ):
        self.debate_id = debate_id
        self.event_emitter = event_emitter

        # Accumulated state
        self._evidence: list[LiveEvidence] = []
        self._votes: list[LiveVote] = []
        self._belief_shifts: list[LiveBeliefShift] = []
        self._agent_positions: dict[str, str] = {}  # agent -> latest position
        self._agent_confidences: dict[str, float] = {}  # agent -> latest confidence
        self._round_num = 0
        self._snapshots: list[ExplanationSnapshot] = []

        # Position tracking for vote flip detection
        self._previous_votes: dict[str, str] = {}  # agent -> previous choice

    def on_round_start(self, round_num: int) -> None:
        """Signal start of a new debate round."""
        self._round_num = round_num

    def on_proposal(
        self,
        agent: str,
        content: str,
        round_num: int | None = None,
        confidence: float = 0.0,
    ) -> None:
        """Record a proposal from an agent."""
        rn = round_num if round_num is not None else self._round_num
        self._evidence.append(LiveEvidence(
            content=content,
            source=agent,
            evidence_type="proposal",
            round_num=rn,
        ))
        self._agent_positions[agent] = content[:200]
        if confidence > 0:
            self._agent_confidences[agent] = confidence
        self._emit_update("evidence_added", {
            "agent": agent,
            "type": "proposal",
            "round": rn,
            "evidence_count": len(self._evidence),
        })

    def on_critique(
        self,
        agent: str,
        content: str,
        target_agent: str = "",
        round_num: int | None = None,
    ) -> None:
        """Record a critique from an agent."""
        rn = round_num if round_num is not None else self._round_num
        self._evidence.append(LiveEvidence(
            content=content,
            source=agent,
            evidence_type="critique",
            round_num=rn,
        ))
        self._emit_update("critique_recorded", {
            "agent": agent,
            "target": target_agent,
            "round": rn,
            "evidence_count": len(self._evidence),
        })

    def on_refinement(
        self,
        agent: str,
        content: str,
        round_num: int | None = None,
    ) -> None:
        """Record a refinement/revision from an agent."""
        rn = round_num if round_num is not None else self._round_num
        old_position = self._agent_positions.get(agent, "")
        self._evidence.append(LiveEvidence(
            content=content,
            source=agent,
            evidence_type="refinement",
            round_num=rn,
        ))
        self._agent_positions[agent] = content[:200]

        # Track as belief shift if position changed
        if old_position and old_position != content[:200]:
            old_conf = self._agent_confidences.get(agent, 0.5)
            self._belief_shifts.append(LiveBeliefShift(
                agent=agent,
                round_num=rn,
                topic="position",
                prior_confidence=old_conf,
                posterior_confidence=old_conf * 0.9,  # Revision implies reduced confidence
                trigger="critique_response",
            ))

    def on_vote(
        self,
        agent: str,
        choice: str,
        confidence: float = 0.5,
        round_num: int | None = None,
        reasoning: str = "",
        weight: float = 1.0,
    ) -> None:
        """Record a vote from an agent."""
        rn = round_num if round_num is not None else self._round_num

        # Detect flip
        previous = self._previous_votes.get(agent)
        flipped = previous is not None and previous != choice

        self._votes.append(LiveVote(
            agent=agent,
            choice=choice,
            confidence=confidence,
            round_num=rn,
            reasoning=reasoning,
            weight=weight,
            flipped=flipped,
        ))
        self._previous_votes[agent] = choice
        self._agent_confidences[agent] = confidence

        if flipped:
            self._belief_shifts.append(LiveBeliefShift(
                agent=agent,
                round_num=rn,
                topic="vote_position",
                prior_confidence=self._agent_confidences.get(agent, 0.5),
                posterior_confidence=confidence,
                trigger="vote_flip",
            ))

        self._emit_update("vote_recorded", {
            "agent": agent,
            "choice": choice[:50],
            "confidence": confidence,
            "flipped": flipped,
            "round": rn,
            "vote_count": len(self._votes),
        })

    def on_consensus(
        self,
        conclusion: str,
        confidence: float,
        consensus_type: str = "majority",
    ) -> None:
        """Record consensus reached."""
        self._emit_update("consensus_reached", {
            "conclusion": conclusion[:200],
            "confidence": confidence,
            "type": consensus_type,
            "total_evidence": len(self._evidence),
            "total_votes": len(self._votes),
            "total_shifts": len(self._belief_shifts),
        })

    def on_belief_change(
        self,
        agent: str,
        topic: str,
        prior: float,
        posterior: float,
        trigger: str = "",
        round_num: int | None = None,
    ) -> None:
        """Explicitly record a belief change."""
        rn = round_num if round_num is not None else self._round_num
        self._belief_shifts.append(LiveBeliefShift(
            agent=agent,
            round_num=rn,
            topic=topic,
            prior_confidence=prior,
            posterior_confidence=posterior,
            trigger=trigger,
        ))
        self._agent_confidences[agent] = posterior

    def compute_factors(self) -> list[LiveFactor]:
        """Compute current confidence attribution factors."""
        factors = []

        # 1. Evidence quality factor
        if self._evidence:
            proposal_count = sum(1 for e in self._evidence if e.evidence_type == "proposal")
            critique_count = sum(1 for e in self._evidence if e.evidence_type == "critique")
            refinement_count = sum(1 for e in self._evidence if e.evidence_type == "refinement")
            total = len(self._evidence)

            # More critiques + refinements = more thorough vetting
            quality = min(1.0, (critique_count + refinement_count) / max(1, total))
            factors.append(LiveFactor(
                name="evidence_quality",
                contribution=quality * 0.3,
                explanation=f"{total} pieces of evidence ({proposal_count} proposals, "
                           f"{critique_count} critiques, {refinement_count} refinements)",
                raw_value=quality,
            ))

        # 2. Agent agreement factor
        if self._votes:
            latest_votes = {}
            for v in self._votes:
                latest_votes[v.agent] = v.choice
            unique_choices = set(latest_votes.values())
            if latest_votes:
                most_common = max(set(latest_votes.values()),
                                  key=list(latest_votes.values()).count)
                agreement = sum(1 for c in latest_votes.values() if c == most_common) / len(latest_votes)
            else:
                agreement = 0.0

            trend = "stable"
            if len(self._snapshots) >= 1:
                prev_agreement = self._snapshots[-1].agent_agreement
                if agreement > prev_agreement + 0.05:
                    trend = "rising"
                elif agreement < prev_agreement - 0.05:
                    trend = "falling"

            factors.append(LiveFactor(
                name="agent_agreement",
                contribution=agreement * 0.3,
                explanation=f"{len(latest_votes)} agents, {len(unique_choices)} distinct positions, "
                           f"{agreement:.0%} agreement",
                trend=trend,
                raw_value=agreement,
            ))

        # 3. Confidence-weighted consensus factor
        if self._votes:
            weighted_sum = sum(v.confidence * v.weight for v in self._votes)
            weight_total = sum(v.weight for v in self._votes)
            weighted_confidence = weighted_sum / max(weight_total, 0.001)
            factors.append(LiveFactor(
                name="confidence_weighted_consensus",
                contribution=weighted_confidence * 0.2,
                explanation=f"Weighted average confidence: {weighted_confidence:.2f}",
                raw_value=weighted_confidence,
            ))

        # 4. Belief stability factor
        if self._belief_shifts:
            # Fewer shifts in later rounds = more stable
            late_shifts = [s for s in self._belief_shifts if s.round_num > 1]
            early_shifts = [s for s in self._belief_shifts if s.round_num <= 1]
            stability = 1.0 - min(1.0, len(late_shifts) / max(len(early_shifts), 1))
            factors.append(LiveFactor(
                name="belief_stability",
                contribution=stability * 0.2,
                explanation=f"{len(self._belief_shifts)} total belief shifts "
                           f"({len(late_shifts)} in later rounds)",
                raw_value=stability,
            ))

        return factors

    def get_leading_position(self) -> tuple[str | None, float]:
        """Get the current leading position and its confidence."""
        if not self._votes:
            # Fall back to most recent proposal
            if self._evidence:
                proposals = [e for e in self._evidence if e.evidence_type == "proposal"]
                if proposals:
                    return proposals[-1].content[:200], 0.5
            return None, 0.0

        # Count weighted votes per choice
        choice_weights: dict[str, float] = {}
        for v in self._votes:
            choice_weights[v.choice] = choice_weights.get(v.choice, 0) + v.confidence * v.weight

        if not choice_weights:
            return None, 0.0

        best = max(choice_weights, key=choice_weights.get)  # type: ignore[arg-type]
        total = sum(choice_weights.values())
        confidence = choice_weights[best] / max(total, 0.001)
        return best[:200], confidence

    def generate_narrative(self) -> str:
        """Generate a natural language narrative of the current state."""
        parts = []

        leading, confidence = self.get_leading_position()
        if leading:
            parts.append(f"The leading position ({confidence:.0%} weighted support) is: "
                        f"\"{leading[:100]}\"")

        if self._evidence:
            parts.append(f"So far, {len(self._evidence)} pieces of evidence have been presented.")

        if self._votes:
            latest_round_votes = [v for v in self._votes if v.round_num == self._round_num]
            if latest_round_votes:
                parts.append(f"In round {self._round_num}, {len(latest_round_votes)} votes were cast.")

        flips = [v for v in self._votes if v.flipped]
        if flips:
            agents = ", ".join(set(v.agent for v in flips))
            parts.append(f"Notable: {agents} changed their position during the debate.")

        if self._belief_shifts:
            significant = [s for s in self._belief_shifts
                          if abs(s.posterior_confidence - s.prior_confidence) > 0.1]
            if significant:
                parts.append(f"{len(significant)} significant belief shifts have occurred.")

        return " ".join(parts) if parts else "Debate is in progress."

    def get_snapshot(self) -> ExplanationSnapshot:
        """Capture current state as an ExplanationSnapshot."""
        factors = self.compute_factors()
        leading, confidence = self.get_leading_position()

        # Compute aggregate scores
        evidence_quality = 0.0
        agent_agreement = 0.0
        for f in factors:
            if f.name == "evidence_quality":
                evidence_quality = f.raw_value
            elif f.name == "agent_agreement":
                agent_agreement = f.raw_value

        snapshot = ExplanationSnapshot(
            round_num=self._round_num,
            timestamp=time.time(),
            evidence_count=len(self._evidence),
            vote_count=len(self._votes),
            belief_shifts=len(self._belief_shifts),
            top_factors=[{
                "name": f.name,
                "contribution": round(f.contribution, 3),
                "explanation": f.explanation,
                "trend": f.trend,
            } for f in sorted(factors, key=lambda x: x.contribution, reverse=True)],
            leading_position=leading,
            position_confidence=round(confidence, 3),
            agent_agreement=round(agent_agreement, 3),
            evidence_quality=round(evidence_quality, 3),
            narrative=self.generate_narrative(),
        )
        self._snapshots.append(snapshot)

        self._emit_update("explanation_snapshot", {
            "round": self._round_num,
            "evidence_count": snapshot.evidence_count,
            "vote_count": snapshot.vote_count,
            "position_confidence": snapshot.position_confidence,
            "agent_agreement": snapshot.agent_agreement,
            "narrative": snapshot.narrative[:300],
            "factors": snapshot.top_factors[:4],
        })

        return snapshot

    @property
    def evidence(self) -> list[LiveEvidence]:
        """All accumulated evidence."""
        return list(self._evidence)

    @property
    def votes(self) -> list[LiveVote]:
        """All recorded votes."""
        return list(self._votes)

    @property
    def belief_shifts(self) -> list[LiveBeliefShift]:
        """All tracked belief shifts."""
        return list(self._belief_shifts)

    @property
    def snapshots(self) -> list[ExplanationSnapshot]:
        """All captured snapshots."""
        return list(self._snapshots)

    def reset(self) -> None:
        """Reset all accumulated state."""
        self._evidence.clear()
        self._votes.clear()
        self._belief_shifts.clear()
        self._agent_positions.clear()
        self._agent_confidences.clear()
        self._previous_votes.clear()
        self._snapshots.clear()
        self._round_num = 0

    def _emit_update(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit a live explanation event."""
        if self.event_emitter is None:
            return
        try:
            data["debate_id"] = self.debate_id
            data["event"] = f"live_explain_{event_type}"
            self.event_emitter.emit(f"live_explain_{event_type}", data)
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug("Live explainability emit failed (non-fatal): %s", e)
