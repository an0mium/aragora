"""Tests for ClaimCheck and evidence grounding integration."""

from __future__ import annotations

from aragora.evidence.collector import EvidencePack, EvidenceSnippet
from aragora.reasoning.claim_check import ClaimCheck
from aragora.reasoning.evidence_grounding import EvidenceGrounder


class DummyExtractor:
    """Simple citation extractor stub."""

    def extract_claims(self, _: str) -> list[str]:
        return ["Alpha is true and Beta is true"]


def test_extract_atomic_claims_from_bullets():
    checker = ClaimCheck()
    text = "- First claim\n- Second claim\n"
    assert checker.extract_atomic_claims(text) == ["First claim", "Second claim"]


def test_match_evidence_uses_overlap_and_numbers():
    checker = ClaimCheck()
    snippet = EvidenceSnippet(
        id="1",
        source="docs",
        title="Agent 42 Results",
        snippet="Agent 42 achieved 95% accuracy in tests.",
        reliability_score=0.9,
    )
    pack = EvidencePack(topic_keywords=["agent"], snippets=[snippet])

    matches = checker.match_evidence(pack, "Agent 42 achieved 95% accuracy")
    assert matches
    assert matches[0].snippet.id == "1"
    assert matches[0].number_matches >= 1


def test_evidence_grounder_expands_atomic_claims():
    snippet = EvidenceSnippet(
        id="alpha",
        source="docs",
        title="Alpha claim",
        snippet="Alpha is true according to tests.",
        reliability_score=0.8,
    )
    pack = EvidencePack(topic_keywords=["alpha"], snippets=[snippet])
    grounder = EvidenceGrounder(evidence_pack=pack, citation_extractor=DummyExtractor())

    verdict = grounder.create_grounded_verdict("ignored", confidence=0.7)
    assert verdict is not None
    assert len(verdict.claims) == 2

    alpha_claim = next(c for c in verdict.claims if "Alpha" in c.claim_text)
    beta_claim = next(c for c in verdict.claims if "Beta" in c.claim_text)

    assert alpha_claim.citations
    assert beta_claim.citations == []
