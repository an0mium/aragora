"""
Audit-Consensus Adapter.

Connects the document audit system to the debate consensus verification
system. Enables multi-agent verification of audit findings.

Usage:
    from aragora.audit.consensus_adapter import FindingVerifier

    verifier = FindingVerifier(agents=["anthropic-api", "openai-api"])
    proof = await verifier.verify_finding(finding)

    if proof.consensus_reached:
        print(f"Finding verified with {proof.confidence:.0%} confidence")
    else:
        print(f"Finding disputed: {proof.get_dissent_summary()}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence
from uuid import uuid4

from aragora.debate.consensus import (
    Claim,
    ConsensusProof,
    ConsensusVote,
    DissentRecord,
    Evidence,
    VoteType,
)

logger = logging.getLogger(__name__)


@dataclass
class VerificationConfig:
    """Configuration for finding verification."""

    # Agents to use for verification
    agents: list[str] = field(default_factory=lambda: ["anthropic-api", "openai-api"])

    # Verification parameters
    consensus_threshold: float = 0.8  # Require 80% agreement
    confidence_threshold: float = 0.7  # Minimum confidence to pass
    max_rounds: int = 2  # Max discussion rounds

    # What to verify
    verify_severity: bool = True  # Verify severity classification
    verify_validity: bool = True  # Verify finding is valid
    verify_recommendation: bool = False  # Verify recommendation is appropriate

    # Context
    include_evidence: bool = True
    include_document_context: bool = True
    max_context_tokens: int = 4000


@dataclass
class VerificationResult:
    """Result of finding verification."""

    finding_id: str
    verified: bool
    consensus_proof: ConsensusProof
    original_severity: str
    verified_severity: Optional[str] = None
    severity_changed: bool = False
    verification_notes: list[str] = field(default_factory=list)
    duration_ms: int = 0


class FindingVerifier:
    """
    Verifies audit findings through multi-agent consensus.

    This adapter converts audit findings into claims and runs them
    through the debate consensus system for verification. This provides:

    1. Independent verification by multiple agents
    2. Severity validation
    3. False positive detection
    4. Confidence scoring with evidence trails
    """

    def __init__(
        self,
        config: Optional[VerificationConfig] = None,
        agents: Optional[list[str]] = None,
    ):
        """
        Initialize the finding verifier.

        Args:
            config: Full configuration (takes precedence)
            agents: Shorthand for specifying verification agents
        """
        if config:
            self.config = config
        else:
            self.config = VerificationConfig(agents=agents or ["anthropic-api", "openai-api"])

    def _finding_to_claim(
        self,
        finding: Any,  # AuditFinding
        claim_id: str,
    ) -> Claim:
        """Convert an audit finding to a claim for verification."""
        # Build claim statement
        statement = f"[{finding.severity.value.upper()}] {finding.title}: " f"{finding.description}"

        # Create supporting evidence from finding
        evidence = []
        if finding.evidence_text:
            evidence.append(
                Evidence(
                    evidence_id=f"ev_{claim_id}_1",
                    source=finding.found_by or "audit_system",
                    content=finding.evidence_text[:1000],
                    evidence_type="data",
                    supports_claim=True,
                    strength=finding.confidence,
                    metadata={
                        "location": finding.evidence_location,
                        "document_id": finding.document_id,
                    },
                )
            )

        return Claim(
            claim_id=claim_id,
            statement=statement,
            author="audit_system",
            confidence=finding.confidence,
            supporting_evidence=evidence,
            refuting_evidence=[],
            round_introduced=0,
            status="active",
        )

    async def verify_finding(
        self,
        finding: Any,  # AuditFinding
        document_context: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verify a single audit finding through multi-agent consensus.

        Args:
            finding: The AuditFinding to verify
            document_context: Optional context from the document

        Returns:
            VerificationResult with consensus proof
        """
        import time

        start_time = time.time()

        # Generate IDs
        verification_id = f"verify_{uuid4().hex[:8]}"
        claim_id = (
            f"claim_{finding.id[:8] if hasattr(finding, 'id') and finding.id else uuid4().hex[:8]}"
        )

        # Convert finding to claim
        claim = self._finding_to_claim(finding, claim_id)

        # Build verification prompt
        prompt = self._build_verification_prompt(finding, document_context)

        # Run verification debate
        votes, dissents = await self._run_verification_debate(
            prompt=prompt,
            claim=claim,
            finding=finding,
        )

        # Analyze results
        supporting = [v.agent for v in votes if v.vote == VoteType.AGREE]
        dissenting = [v.agent for v in votes if v.vote == VoteType.DISAGREE]
        avg_confidence = sum(v.confidence for v in votes) / len(votes) if votes else 0

        consensus_reached = (
            len(supporting) / len(votes) >= self.config.consensus_threshold if votes else False
        )

        # Check for severity changes
        severity_changed = False
        verified_severity = None
        notes = []

        for dissent in dissents:
            if "severity" in dissent.reasons[0].lower():
                severity_changed = True
                if dissent.alternative_view:
                    # Try to extract suggested severity
                    for sev in ["critical", "high", "medium", "low", "info"]:
                        if sev in dissent.alternative_view.lower():
                            verified_severity = sev
                            notes.append(f"Agent {dissent.agent} suggests severity: {sev}")
                            break

        # Build consensus proof
        proof = ConsensusProof(
            proof_id=verification_id,
            debate_id=f"audit_verify_{finding.id if hasattr(finding, 'id') else 'unknown'}",
            task=f"Verify audit finding: {finding.title}",
            final_claim=claim.statement,
            confidence=avg_confidence,
            consensus_reached=consensus_reached,
            votes=votes,
            supporting_agents=supporting,
            dissenting_agents=dissenting,
            claims=[claim],
            dissents=dissents,
            unresolved_tensions=[],
            evidence_chain=claim.supporting_evidence,
            reasoning_summary=self._build_reasoning_summary(votes, dissents),
            rounds_to_consensus=1,
            metadata={
                "finding_id": getattr(finding, "id", "unknown"),
                "original_severity": finding.severity.value,
                "audit_type": finding.audit_type.value,
                "verified": consensus_reached
                and avg_confidence >= self.config.confidence_threshold,
            },
        )

        duration_ms = int((time.time() - start_time) * 1000)

        return VerificationResult(
            finding_id=getattr(finding, "id", "unknown"),
            verified=consensus_reached and avg_confidence >= self.config.confidence_threshold,
            consensus_proof=proof,
            original_severity=finding.severity.value,
            verified_severity=verified_severity,
            severity_changed=severity_changed,
            verification_notes=notes,
            duration_ms=duration_ms,
        )

    async def verify_findings_batch(
        self,
        findings: Sequence[Any],  # Sequence[AuditFinding]
        document_contexts: Optional[dict[str, str]] = None,
        max_concurrent: int = 5,
    ) -> list[VerificationResult]:
        """
        Verify multiple findings in parallel.

        Args:
            findings: List of findings to verify
            document_contexts: Optional dict mapping document IDs to context
            max_concurrent: Maximum concurrent verifications

        Returns:
            List of VerificationResults
        """
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)

        async def verify_with_limit(finding: Any) -> VerificationResult:
            async with semaphore:
                context = None
                if document_contexts and hasattr(finding, "document_id"):
                    context = document_contexts.get(finding.document_id)
                return await self.verify_finding(finding, context)

        return await asyncio.gather(*[verify_with_limit(f) for f in findings])

    def _build_verification_prompt(
        self,
        finding: Any,
        document_context: Optional[str] = None,
    ) -> str:
        """Build the prompt for verification debate."""
        prompt_parts = [
            "# Audit Finding Verification",
            "",
            "Please verify the following audit finding. Evaluate whether:",
            "1. The finding is valid and represents a real issue",
            "2. The severity classification is appropriate",
            "3. The evidence supports the conclusion",
            "",
            "## Finding",
            f"**Title:** {finding.title}",
            f"**Severity:** {finding.severity.value}",
            f"**Category:** {finding.category}",
            f"**Confidence:** {finding.confidence:.0%}",
            "",
            "**Description:**",
            finding.description,
            "",
        ]

        if finding.evidence_text:
            prompt_parts.extend(
                [
                    "**Evidence:**",
                    "```",
                    finding.evidence_text[:2000],
                    "```",
                    "",
                ]
            )

        if finding.recommendation:
            prompt_parts.extend(
                [
                    "**Recommendation:**",
                    finding.recommendation,
                    "",
                ]
            )

        if document_context and self.config.include_document_context:
            prompt_parts.extend(
                [
                    "## Document Context",
                    document_context[: self.config.max_context_tokens],
                    "",
                ]
            )

        prompt_parts.extend(
            [
                "## Your Task",
                "Respond with:",
                "1. VALID or INVALID (is this a real issue?)",
                "2. AGREE or DISAGREE with the severity",
                "3. Your confidence (0-100%)",
                "4. Brief reasoning",
                "",
                "If you disagree with severity, suggest the appropriate level.",
            ]
        )

        return "\n".join(prompt_parts)

    async def _run_verification_debate(
        self,
        prompt: str,
        claim: Claim,
        finding: Any,
    ) -> tuple[list[ConsensusVote], list[DissentRecord]]:
        """Run the actual verification debate with agents."""
        votes: list[ConsensusVote] = []
        dissents: list[DissentRecord] = []

        try:
            from aragora.agents import get_agent
        except ImportError:
            logger.warning("Could not import agents, using mock verification")
            return self._mock_verification(finding)

        for agent_name in self.config.agents:
            try:
                agent = get_agent(agent_name)
                if not agent:
                    logger.warning(f"Agent not available: {agent_name}")
                    continue

                # Get agent's verification
                response = await agent.generate(prompt)
                vote, confidence, reasoning = self._parse_verification_response(
                    response, agent_name
                )

                votes.append(
                    ConsensusVote(
                        agent=agent_name,
                        vote=vote,
                        confidence=confidence,
                        reasoning=reasoning,
                    )
                )

                # Record dissent if disagreeing
                if vote == VoteType.DISAGREE:
                    dissents.append(
                        DissentRecord(
                            agent=agent_name,
                            claim_id=claim.claim_id,
                            dissent_type="full" if confidence > 0.8 else "partial",
                            reasons=[reasoning],
                            alternative_view=self._extract_alternative(response),
                            severity=confidence,
                        )
                    )

            except Exception as e:
                logger.error(f"Error getting verification from {agent_name}: {e}")
                # Add abstain vote on error
                votes.append(
                    ConsensusVote(
                        agent=agent_name,
                        vote=VoteType.ABSTAIN,
                        confidence=0.0,
                        reasoning=f"Error during verification: {str(e)[:100]}",
                    )
                )

        return votes, dissents

    def _parse_verification_response(
        self,
        response: str,
        agent_name: str,
    ) -> tuple[VoteType, float, str]:
        """Parse agent's verification response."""
        response_lower = response.lower()

        # Determine vote
        if "invalid" in response_lower or "false positive" in response_lower:
            vote = VoteType.DISAGREE
        elif "valid" in response_lower and "agree" in response_lower:
            vote = VoteType.AGREE
        elif "disagree" in response_lower:
            vote = VoteType.DISAGREE
        elif "conditional" in response_lower or "partially" in response_lower:
            vote = VoteType.CONDITIONAL
        else:
            vote = VoteType.AGREE  # Default to agree if valid

        # Extract confidence
        confidence = 0.7  # Default
        import re

        conf_match = re.search(r"(\d{1,3})%", response)
        if conf_match:
            confidence = int(conf_match.group(1)) / 100.0

        # Extract reasoning (first sentence or two)
        sentences = response.split(".")
        reasoning = ". ".join(sentences[:2]).strip()
        if len(reasoning) > 200:
            reasoning = reasoning[:200] + "..."

        return vote, confidence, reasoning

    def _extract_alternative(self, response: str) -> Optional[str]:
        """Extract alternative severity suggestion from response."""
        response_lower = response.lower()

        # Look for severity suggestions
        for keyword in ["should be", "suggest", "recommend", "appropriate"]:
            if keyword in response_lower:
                idx = response_lower.index(keyword)
                snippet = response[idx : idx + 100]
                for sev in ["critical", "high", "medium", "low", "info"]:
                    if sev in snippet.lower():
                        return f"Suggested severity: {sev}"

        return None

    def _mock_verification(
        self,
        finding: Any,
    ) -> tuple[list[ConsensusVote], list[DissentRecord]]:
        """Provide mock verification when agents unavailable."""
        # Simple heuristic-based mock
        confidence = finding.confidence

        votes = [
            ConsensusVote(
                agent="mock_agent_1",
                vote=VoteType.AGREE if confidence > 0.6 else VoteType.CONDITIONAL,
                confidence=confidence,
                reasoning="Mock verification based on finding confidence.",
            ),
            ConsensusVote(
                agent="mock_agent_2",
                vote=VoteType.AGREE if confidence > 0.5 else VoteType.DISAGREE,
                confidence=confidence * 0.9,
                reasoning="Mock verification (secondary agent).",
            ),
        ]

        dissents = []
        if confidence < 0.5:
            dissents.append(
                DissentRecord(
                    agent="mock_agent_2",
                    claim_id="mock_claim",
                    dissent_type="partial",
                    reasons=["Low confidence finding requires additional review"],
                    severity=0.5,
                )
            )

        return votes, dissents

    def _build_reasoning_summary(
        self,
        votes: list[ConsensusVote],
        dissents: list[DissentRecord],
    ) -> str:
        """Build a summary of the verification reasoning."""
        parts = []

        # Summarize votes
        agree_count = sum(1 for v in votes if v.vote == VoteType.AGREE)
        disagree_count = sum(1 for v in votes if v.vote == VoteType.DISAGREE)
        parts.append(f"Verification: {agree_count} agents agreed, {disagree_count} disagreed.")

        # Add key reasoning
        for vote in votes:
            if vote.reasoning:
                parts.append(f"- {vote.agent}: {vote.reasoning[:100]}")

        # Note dissents
        if dissents:
            parts.append("\nDissenting views:")
            for dissent in dissents:
                parts.append(f"- {dissent.agent}: {dissent.reasons[0][:100]}")

        return "\n".join(parts)


# Convenience function for quick verification
async def verify_finding(
    finding: Any,
    agents: Optional[list[str]] = None,
    document_context: Optional[str] = None,
) -> VerificationResult:
    """
    Quick verification of a single finding.

    Args:
        finding: The AuditFinding to verify
        agents: List of agent names to use
        document_context: Optional document context

    Returns:
        VerificationResult with consensus proof
    """
    verifier = FindingVerifier(agents=agents)
    return await verifier.verify_finding(finding, document_context)


__all__ = [
    "FindingVerifier",
    "VerificationConfig",
    "VerificationResult",
    "verify_finding",
]
