#!/usr/bin/env python3
"""Example: Hiring decision debate — Senior engineer candidate assessment.

Three AI agents with different evaluation perspectives debate a hiring decision.
The debate produces a decision receipt that documents the reasoning, dissent,
and confidence level — useful for demonstrating fair and auditable hiring
processes.

Usage:
    python examples/hiring_decision.py
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aragora_debate import (
    Agent,
    Arena,
    Critique,
    DebateConfig,
    Message,
    ReceiptBuilder,
    Vote,
)
from aragora_debate.types import ConsensusMethod


# ---------------------------------------------------------------------------
# Candidate profile
# ---------------------------------------------------------------------------

CANDIDATE_PROFILE = """\
Position: Senior Backend Engineer
Candidate: Alex Chen (anonymized)

Technical Assessment (coding exercise):
- Completed distributed rate limiter in Go (45 min)
- Clean code, good error handling, wrote tests
- Missed edge case: clock skew in distributed nodes
- Score: 7.5/10

System Design Interview:
- Designed URL shortener scaling to 1B URLs
- Strong on storage tier (consistent hashing, read replicas)
- Weak on analytics pipeline (proposed batch when real-time needed)
- Score: 7/10

Behavioral Interview:
- 6 years experience (3 at startup, 3 at large company)
- Led migration from monolith to microservices (team of 5)
- Conflict example: disagreed with manager on deadline, escalated constructively
- Communication: clear and structured
- Score: 8/10

Reference Checks (2 of 2 completed):
- Former manager: "Strong technically, occasionally over-engineers solutions"
- Former peer: "Great collaborator, sometimes slow to make decisions under pressure"

Compensation: Asking $195K base + equity. Our band: $175K-$210K.
Team need: Backfill for departing senior engineer. Timeline: urgent (2 weeks).
"""


# ---------------------------------------------------------------------------
# Evaluator agents
# ---------------------------------------------------------------------------


class HiringAgent(Agent):
    """Mock hiring evaluator with a specific perspective."""

    def __init__(self, name: str, perspective: str, analysis: dict[str, str]) -> None:
        super().__init__(name, stance="neutral")
        self.perspective = perspective
        self.analysis = analysis

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        round_num = 1 + (max((m.round for m in context), default=0) if context else 0)
        return self.analysis.get(f"round{round_num}", self.analysis.get("round1", ""))

    async def critique(self, proposal: str, task: str, **kw) -> Critique:
        target = kw.get("target_agent", "unknown")
        return Critique(
            agent=self.name,
            target_agent=target,
            target_content=proposal,
            issues=self.analysis.get("critique_issues", ["Could provide more specifics"]),
            suggestions=self.analysis.get("critique_suggestions", ["Add concrete examples"]),
            severity=float(self.analysis.get("critique_severity", "4.0")),
        )

    async def vote(self, proposals: dict[str, str], task: str) -> Vote:
        return Vote(
            agent=self.name,
            choice=self.analysis.get("vote_choice", self.name),
            reasoning=self.analysis.get("vote_reasoning", "Based on my analysis"),
            confidence=float(self.analysis.get("vote_confidence", "0.7")),
        )


technical_evaluator = HiringAgent(
    name="technical-eval",
    perspective="Technical skill assessment",
    analysis={
        "round1": (
            "## Technical Evaluation\n\n"
            "**Recommendation: HIRE (with reservations)**\n\n"
            "**Strengths:**\n"
            "- 7.5/10 on coding is solid for senior level. The rate limiter "
            "solution demonstrates distributed systems thinking.\n"
            "- Writing tests unprompted is a strong signal — many candidates skip them.\n"
            "- Monolith-to-microservices experience directly relevant to our roadmap.\n\n"
            "**Concerns:**\n"
            "- Missing the clock skew edge case is notable for a *senior* distributed "
            "systems role. This is a known failure mode that seniors should catch.\n"
            "- System design weakness on analytics (batch vs real-time) suggests gaps "
            "in streaming architecture knowledge.\n\n"
            "**Assessment:** Solid senior engineer, but not a strong senior. The "
            "reference about over-engineering aligns with what I'd expect from someone "
            "who writes clean code but misses distributed edge cases — they focus on "
            "elegance over robustness.\n\n"
            "Given our urgent timeline, I'd hire at the lower end of our band ($185K) "
            "with clear expectations about distributed systems edge cases."
        ),
        "round2": (
            "After considering the culture evaluator's points, I still lean hire:\n\n"
            "The 'slow under pressure' concern is real but manageable. Pairing with "
            "our existing senior who's fast-but-sometimes-sloppy could be complementary.\n\n"
            "However, I want to strengthen my position: the clock skew miss is a "
            "**coaching opportunity, not a disqualifier**. Many excellent seniors learn "
            "distributed edge cases on the job — what matters is whether they learn "
            "quickly when the issue arises. The reference saying 'strong technically' "
            "suggests they do."
        ),
        "critique_issues": [
            "Behavioral interview scores shouldn't outweigh technical gaps for an engineering role",
            "Need to separate 'nice to work with' from 'can handle production incidents'",
        ],
        "critique_suggestions": [
            "Weight the technical assessment more heavily for a senior backend role",
        ],
        "critique_severity": "5.0",
        "vote_choice": "technical-eval",
        "vote_reasoning": "Hire at $185K with clear distributed systems expectations",
        "vote_confidence": "0.72",
    },
)

culture_evaluator = HiringAgent(
    name="culture-eval",
    perspective="Team fit and growth potential",
    analysis={
        "round1": (
            "## Culture & Team Fit Evaluation\n\n"
            "**Recommendation: HIRE**\n\n"
            "**Strengths:**\n"
            "- 8/10 behavioral score is our highest this quarter. The conflict "
            "resolution example (escalating constructively) shows maturity.\n"
            "- Led a team of 5 through a major migration — this is exactly the "
            "kind of person who can mentor our 2 junior engineers.\n"
            "- 'Clear and structured' communication is critical for our remote-first team.\n\n"
            "**Concerns:**\n"
            "- 'Slow to make decisions under pressure' is worth monitoring. Our "
            "on-call rotation requires quick incident triage.\n"
            "- Over-engineering tendency could clash with our 'ship fast' culture.\n\n"
            "**Assessment:** Strong cultural fit. The mentorship capability alone "
            "is worth the hire — we've been struggling to level up our juniors. "
            "The 'slow under pressure' risk is mitigated by our existing on-call "
            "buddy system.\n\n"
            "Recommend hiring at asking price ($195K) to close quickly given "
            "our 2-week timeline."
        ),
        "critique_issues": [
            "Technical gaps are coachable; cultural fit is much harder to change",
            "The urgency of our backfill means we should weigh culture more heavily",
        ],
        "critique_suggestions": [
            "Consider the cost of continuing to search vs hiring now",
        ],
        "critique_severity": "3.0",
        "vote_choice": "culture-eval",
        "vote_reasoning": "Strong culture fit + mentorship outweighs technical gaps",
        "vote_confidence": "0.8",
    },
)

devil_advocate = HiringAgent(
    name="devils-advocate",
    perspective="Risk assessment and counter-arguments",
    analysis={
        "round1": (
            "## Risk Assessment: Case Against Hiring\n\n"
            "**Recommendation: DO NOT HIRE (or wait)**\n\n"
            "I'm deliberately taking the contrarian position to stress-test:\n\n"
            "**Red flags the other evaluators are minimizing:**\n\n"
            "1. **Two independent sources flag decision-making speed.** The "
            "reference ('slow under pressure') and the system design interview "
            "(batch when real-time was needed) both point to the same issue: "
            "this candidate defaults to deliberation over action.\n\n"
            "2. **Senior means owning production.** Missing clock skew in a "
            "distributed systems exercise isn't a minor gap — it's the kind "
            "of oversight that causes production incidents. We're not hiring "
            "for a mid-level role.\n\n"
            "3. **Urgency bias.** We're at risk of lowering our bar because "
            "we need to fill the seat in 2 weeks. The departing engineer gave "
            "4 weeks notice. Why are we only now at the 'urgent' stage?\n\n"
            "4. **$195K for a 7/10 system design score.** That's top-of-band "
            "compensation for mid-band performance.\n\n"
            "**Alternative:** Hire a contractor for 3 months while we continue "
            "the search. The cost of a mis-hire at senior level ($300K+ when "
            "you factor in ramp time, team disruption, and re-hiring) far "
            "exceeds 3 months of contractor spend."
        ),
        "round2": (
            "The other evaluators make fair points about coachability and "
            "mentorship value. I'll adjust my position:\n\n"
            "I can support a **conditional hire** if:\n"
            "1. Comp is $185K (lower band), not $195K asking\n"
            "2. 90-day checkpoint with explicit distributed systems milestones\n"
            "3. Shadow on-call for first month before joining rotation\n\n"
            "The contractor alternative remains my preferred path, but I "
            "recognize the team is leaning hire. The conditions above mitigate "
            "my top concerns."
        ),
        "critique_issues": [
            "Urgency bias is causing the team to rationalize weaknesses",
            "Two independent sources flagging the same issue (decision speed) should carry more weight",
        ],
        "critique_suggestions": [
            "Set clear 90-day milestones if hiring",
            "Compare against our pipeline — do we have other candidates?",
        ],
        "critique_severity": "7.0",
        "vote_choice": "technical-eval",
        "vote_reasoning": "Hire with conditions is acceptable — but at $185K with milestones, not $195K",
        "vote_confidence": "0.55",
    },
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    agents = [technical_evaluator, culture_evaluator, devil_advocate]

    arena = Arena(
        question=(
            "Should we extend an offer to this candidate for the Senior Backend "
            "Engineer position? If yes, at what compensation level and with what "
            "conditions?"
        ),
        agents=agents,
        config=DebateConfig(
            rounds=2,
            consensus_method=ConsensusMethod.MAJORITY,
            early_stopping=False,  # Always run all rounds for thorough eval
            require_reasoning=True,
        ),
        context=CANDIDATE_PROFILE,
    )

    print("=" * 60)
    print("HIRING DECISION: Senior Backend Engineer")
    print("=" * 60)
    print()

    result = await arena.run()

    # Summary
    print(f"\nDecision: {result.status}")
    print(f"Confidence: {result.confidence:.0%}")
    if result.consensus and result.consensus.reached:
        print(f"Consensus: Reached ({result.consensus.method.value})")
        print(f"Agreement: {result.consensus.agreement_ratio:.0%}")
    print()

    # Each evaluator's final position
    print("EVALUATOR POSITIONS:")
    for name, proposal in result.proposals.items():
        print(f"\n  [{name}]")
        # Show first line (recommendation)
        first_line = proposal.split("\n")[0] if proposal else ""
        print(f"  {first_line}")
    print()

    # Vote breakdown
    print("VOTE BREAKDOWN:")
    for v in result.votes[-len(result.participants) :]:  # last round's votes
        print(f"  {v.agent}: voted for '{v.choice}' (confidence: {v.confidence:.0%})")
        print(f"    Reasoning: {v.reasoning}")
    print()

    # Dissenting views
    if result.dissenting_views:
        print("DISSENTING VIEWS:")
        for dv in result.dissenting_views:
            print(f"  - {dv}")
        print()

    # Decision receipt
    print("=" * 60)
    print("DECISION RECEIPT")
    print("=" * 60)
    assert result.receipt is not None
    print(result.receipt.to_markdown())

    # HMAC signing demo
    signed = ReceiptBuilder.sign_hmac(result.receipt, key="demo-signing-key-2026")
    verified = ReceiptBuilder.verify_hmac(result.receipt, key="demo-signing-key-2026")
    print(f"\nHMAC-SHA256 signed: {signed.signature[:32]}...")
    print(f"Signature verified: {verified}")

    tampered = ReceiptBuilder.verify_hmac(result.receipt, key="wrong-key")
    print(f"Tamper check (wrong key): {tampered}")


if __name__ == "__main__":
    asyncio.run(main())
