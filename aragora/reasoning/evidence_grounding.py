"""
Evidence Grounding - Link evidence to claims and create grounded verdicts.

Extracts citation-worthy claims from debate results and links them to
collected evidence, producing GroundedVerdict objects with scholarly citations.

Inspired by Heavy3.ai's Deep Audit which delivers "verdicts with scholarly references".

Usage:
    grounder = EvidenceGrounder(evidence_pack, citation_extractor)
    verdict = grounder.create_grounded_verdict(final_answer, confidence)

    # Optional formal verification
    await grounder.verify_claims_formally(verdict)
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:
    from aragora.evidence.collector import EvidencePack
    from aragora.reasoning.citations import (
        CitationExtractor,
        GroundedVerdict,
        CitedClaim,
        ScholarlyEvidence,
    )

logger = logging.getLogger(__name__)


@dataclass
class GroundingResult:
    """Result of linking evidence to a claim."""

    citations: list["ScholarlyEvidence"]
    grounding_score: float


class EvidenceGrounder:
    """
    Links evidence snippets to claims and creates grounded verdicts.

    Converts EvidenceSnippet objects to ScholarlyEvidence and calculates
    grounding scores based on keyword overlap and source quality.

    Usage:
        grounder = EvidenceGrounder(evidence_pack, citation_extractor)

        # Create grounded verdict for a final answer
        verdict = grounder.create_grounded_verdict(
            final_answer="The debate concluded...",
            confidence=0.85,
        )

        # Optional: verify claims formally with Z3
        await grounder.verify_claims_formally(verdict)
    """

    # Quality score mapping for grounding calculation
    QUALITY_SCORES = {
        "peer_reviewed": 1.0,
        "authoritative": 0.9,
        "reputable": 0.7,
        "mixed": 0.5,
        "unverified": 0.3,
        "questionable": 0.1,
    }

    def __init__(
        self,
        evidence_pack: Optional["EvidencePack"] = None,
        citation_extractor: Optional["CitationExtractor"] = None,
    ):
        """
        Initialize the evidence grounder.

        Args:
            evidence_pack: Collection of evidence snippets from research
            citation_extractor: Extractor for identifying claims needing citations
        """
        self.evidence_pack = evidence_pack
        self.citation_extractor = citation_extractor

    def set_evidence_pack(self, evidence_pack: "EvidencePack") -> None:
        """Update the evidence pack."""
        self.evidence_pack = evidence_pack

    def link_evidence_to_claim(self, claim_text: str) -> GroundingResult:
        """
        Link evidence snippets to a claim based on keyword matching.

        Args:
            claim_text: The claim text to find evidence for

        Returns:
            GroundingResult with list of ScholarlyEvidence and grounding score
        """
        from aragora.reasoning.citations import (
            ScholarlyEvidence,
            CitationType,
            CitationQuality,
        )

        if not self.evidence_pack or not self.evidence_pack.snippets:
            return GroundingResult(citations=[], grounding_score=0.0)

        # Extract keywords from claim
        claim_lower = claim_text.lower()
        claim_words = set(claim_lower.split())

        matched_citations = []
        for snippet in self.evidence_pack.snippets:
            # Calculate relevance based on keyword overlap
            snippet_words = set(snippet.snippet.lower().split())
            snippet_words.update(set(snippet.title.lower().split()))

            # Check for keyword overlap
            overlap = claim_words.intersection(snippet_words)
            if len(overlap) >= 2:  # At least 2 matching keywords
                relevance = len(overlap) / max(len(claim_words), 1)

                # Determine citation type based on source
                source = snippet.source.lower()
                if "github" in source:
                    citation_type = CitationType.CODE_REPOSITORY
                elif "doc" in source or "local" in source:
                    citation_type = CitationType.DOCUMENTATION
                else:
                    citation_type = CitationType.WEB_PAGE

                # Map reliability score to quality
                if snippet.reliability_score >= 0.8:
                    quality = CitationQuality.AUTHORITATIVE
                elif snippet.reliability_score >= 0.6:
                    quality = CitationQuality.REPUTABLE
                elif snippet.reliability_score >= 0.4:
                    quality = CitationQuality.MIXED
                else:
                    quality = CitationQuality.UNVERIFIED

                evidence = ScholarlyEvidence(
                    id=snippet.id,
                    citation_type=citation_type,
                    title=snippet.title,
                    url=snippet.url,
                    excerpt=snippet.snippet[:500],
                    relevance_score=relevance,
                    quality=quality,
                    claim_id=claim_text[:50],  # Truncated claim as ID
                    metadata=snippet.metadata,
                )
                matched_citations.append(evidence)

        # Sort by relevance and take top 3
        matched_citations.sort(key=lambda e: e.relevance_score, reverse=True)
        top_citations = matched_citations[:3]

        # Calculate grounding score based on evidence quality
        if not top_citations:
            return GroundingResult(citations=[], grounding_score=0.0)

        # Average quality score weighted by relevance
        total_weight = sum(e.relevance_score for e in top_citations)
        if total_weight == 0:
            return GroundingResult(citations=top_citations, grounding_score=0.3)

        weighted_score = (
            sum(
                self.QUALITY_SCORES.get(e.quality.value, 0.3) * e.relevance_score
                for e in top_citations
            )
            / total_weight
        )

        return GroundingResult(citations=top_citations, grounding_score=weighted_score)

    def create_grounded_verdict(
        self,
        final_answer: str,
        confidence: float,
    ) -> Optional["GroundedVerdict"]:
        """
        Create a GroundedVerdict for a final answer.

        Heavy3-inspired: Wrap final answers with evidence grounding analysis.
        Identifies claims that should be backed by evidence and links available citations.

        Args:
            final_answer: The final answer text to ground
            confidence: Confidence score for the answer

        Returns:
            GroundedVerdict with cited claims and grounding score, or None on error
        """
        if not self.citation_extractor or not final_answer:
            return None

        try:
            from aragora.reasoning.citations import GroundedVerdict, CitedClaim

            # Extract claims from the final answer
            claims_text = self.citation_extractor.extract_claims(final_answer)

            if not claims_text:
                # No claims needing citations - return minimal grounded verdict
                return GroundedVerdict(
                    verdict=final_answer,
                    confidence=confidence,
                    grounding_score=1.0,  # No claims = fully grounded (nothing to cite)
                )

            # Create CitedClaim objects with linked evidence
            cited_claims = []
            all_citations = []
            total_grounding = 0.0

            for claim_text in claims_text:
                # Link evidence to this claim
                result = self.link_evidence_to_claim(claim_text)
                all_citations.extend(result.citations)

                claim = CitedClaim(
                    claim_text=claim_text,
                    confidence=confidence,
                    grounding_score=result.grounding_score,
                    citations=result.citations,
                )
                cited_claims.append(claim)
                total_grounding += result.grounding_score

            # Calculate overall grounding score
            if cited_claims:
                avg_grounding = total_grounding / len(cited_claims)
            else:
                # Fallback: penalize high claim density
                answer_words = len(final_answer.split())
                claim_density = len(claims_text) / max(answer_words / 100, 1)
                avg_grounding = max(0.0, 1.0 - (claim_density * 0.2))

            # Deduplicate citations by ID
            seen_ids: set[str] = set()
            unique_citations = []
            for citation in all_citations:
                if citation.id not in seen_ids:
                    seen_ids.add(citation.id)
                    unique_citations.append(citation)

            return GroundedVerdict(
                verdict=final_answer,
                confidence=confidence,
                claims=cited_claims,
                all_citations=unique_citations,
                grounding_score=avg_grounding,
            )

        except Exception as e:
            logger.warning(f"Error creating grounded verdict: {e}")
            return None

    async def verify_claims_formally(
        self,
        verdict: "GroundedVerdict",
        max_claims: int = 5,
        timeout_seconds: float = 5.0,
    ) -> tuple[int, int]:
        """
        Verify decidable claims using Z3 SMT solver.

        For arithmetic, logic, and constraint claims, attempts formal verification
        to provide machine-verified evidence. Updates the verdict in-place.

        Args:
            verdict: GroundedVerdict to verify
            max_claims: Maximum number of claims to verify
            timeout_seconds: Timeout per claim

        Returns:
            Tuple of (verified_count, disproven_count)
        """
        if not verdict or not verdict.claims:
            return 0, 0

        try:
            from aragora.verification.formal import (
                get_formal_verification_manager,
                FormalProofStatus,
            )
        except ImportError:
            return 0, 0  # Formal verification not available

        try:
            manager = get_formal_verification_manager()
            status = manager.status_report()

            if not status.get("any_available"):
                return 0, 0  # No backends available

            verified_count = 0
            disproven_count = 0

            for claim in verdict.claims[:max_claims]:
                try:
                    # Attempt formal verification
                    proof_result = await manager.attempt_formal_verification(
                        claim=claim.claim_text,
                        claim_type="decidable",
                        context=verdict.verdict[:500] if verdict.verdict else "",
                        timeout_seconds=timeout_seconds,
                    )

                    if proof_result and proof_result.status == FormalProofStatus.PROOF_FOUND:
                        claim.grounding_score = 1.0  # Formally verified
                        claim.citations.append(
                            {  # type: ignore[arg-type]
                                "type": "formal_proof",
                                "prover": proof_result.language.value,
                                "verified": True,
                            }
                        )
                        verified_count += 1
                    elif proof_result and proof_result.status == FormalProofStatus.PROOF_FAILED:
                        claim.grounding_score = 0.0  # Disproven
                        claim.citations.append(
                            {  # type: ignore[arg-type]
                                "type": "formal_proof",
                                "prover": proof_result.language.value,
                                "verified": False,
                                "counterexample": proof_result.proof_text,
                            }
                        )
                        disproven_count += 1

                except Exception as e:
                    logger.debug(f"Verification skipped for claim: {e}")

            if verified_count > 0 or disproven_count > 0:
                logger.info(
                    f"Formal verification: {verified_count} verified, {disproven_count} disproven"
                )

            return verified_count, disproven_count

        except Exception as e:
            logger.warning(f"Formal verification system error: {e}")
            return 0, 0


def create_evidence_grounder(
    evidence_pack: Optional["EvidencePack"] = None,
    citation_extractor: Optional["CitationExtractor"] = None,
) -> EvidenceGrounder:
    """
    Factory function to create an EvidenceGrounder.

    Args:
        evidence_pack: Collection of evidence snippets
        citation_extractor: Extractor for identifying claims

    Returns:
        Configured EvidenceGrounder instance
    """
    return EvidenceGrounder(
        evidence_pack=evidence_pack,
        citation_extractor=citation_extractor,
    )
