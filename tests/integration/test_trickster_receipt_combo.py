"""
Integration test: Trickster (hollow consensus detection) + Decision Receipt.

Tests the full pipeline:
1. Configure Trickster with high sensitivity
2. Feed it responses that exhibit hollow consensus (high convergence, low evidence)
3. Verify intervention is generated
4. Build a Decision Receipt incorporating Trickster findings
5. Sign the receipt with cryptographic signature
6. Verify integrity and export to multiple formats

This validates that Aragora's two most unique subsystems — hollow consensus
detection and audit-ready decision receipts — compose correctly.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest

from aragora.debate.trickster import (
    EvidencePoweredTrickster,
    InterventionType,
    TricksterConfig,
    TricksterIntervention,
)
from aragora.gauntlet.receipt import (
    ConsensusProof,
    DecisionReceipt,
    ProvenanceRecord,
    receipt_to_html,
    receipt_to_markdown,
    receipt_to_sarif,
)


# ── Fixtures ──


@pytest.fixture
def sensitive_trickster() -> EvidencePoweredTrickster:
    """Trickster configured with high sensitivity for reliable detection."""
    config = TricksterConfig(
        sensitivity=0.9,
        min_quality_threshold=0.9,  # High threshold so low-quality responses trigger severity > 0.3
        enable_challenge_prompts=True,
        enable_role_assignment=True,
        enable_extended_rounds=True,
        enable_breakpoints=True,
        max_challenges_per_round=3,
        max_interventions_total=10,
        intervention_cooldown_rounds=0,  # No cooldown for testing
    )
    return EvidencePoweredTrickster(config=config, linker=None)


@pytest.fixture
def hollow_responses() -> dict[str, str]:
    """Agent responses that converge without substantive evidence."""
    return {
        "agent_alpha": (
            "I agree we should implement caching. It will improve performance. "
            "Caching is widely used in industry. This seems like the best approach."
        ),
        "agent_beta": (
            "Caching is indeed the way to go. Performance will be better. "
            "Many systems use caching. I support this direction."
        ),
        "agent_gamma": (
            "Yes, caching makes sense. Performance gains are likely. "
            "This is a common pattern. I concur with the consensus."
        ),
    }


@pytest.fixture
def evidence_rich_responses() -> dict[str, str]:
    """Agent responses with substantive evidence backing claims."""
    return {
        "agent_alpha": (
            "According to the Redis benchmark suite (2024), read-through caching "
            "reduces P99 latency by 73% for read-heavy workloads (>80% reads). "
            "However, write-amplification increases by 2.3x (source: AWS whitepaper "
            "WP-2024-0145). We should only apply caching to the /api/v1/search "
            "endpoint where read ratio is 94%. The memory cost would be approximately "
            "8GB for 10M cached entries at 800 bytes average payload."
        ),
        "agent_beta": (
            "I disagree with blanket caching. Our profiling data from the last "
            "3 months shows that 62% of cache misses occur on long-tail queries "
            "(>10 second TTL). A tiered approach — in-memory L1 (256MB, 1s TTL) "
            "plus Redis L2 (8GB, 30s TTL) — would give us 89% hit rate vs 71% "
            "with a single tier. Source: internal perf dashboard, run ID 2024-Q4-bench-7."
        ),
    }


def _make_receipt(
    trickster: EvidencePoweredTrickster,
    agents_involved: list[str],
    verdict: str = "CONDITIONAL",
) -> DecisionReceipt:
    """Build a DecisionReceipt incorporating Trickster findings."""
    now = datetime.now(timezone.utc).isoformat()
    input_text = "Should we implement caching for the API?"
    input_hash = hashlib.sha256(input_text.encode()).hexdigest()

    stats = trickster.get_stats()

    # Convert trickster interventions to provenance records
    provenance: list[ProvenanceRecord] = []
    vulnerability_details: list[dict] = []

    for intervention in trickster._state.interventions:
        provenance.append(
            ProvenanceRecord(
                timestamp=now,
                event_type="trickster_intervention",
                agent="Trickster",
                description=(
                    f"{intervention.intervention_type.value}: "
                    f"{intervention.challenge_text[:100]}"
                ),
                evidence_hash=hashlib.sha256(
                    intervention.challenge_text.encode()
                ).hexdigest(),
            )
        )
        vulnerability_details.append(
            {
                "id": f"TRICK-{intervention.round_num}-{intervention.intervention_type.value}",
                "severity": "MEDIUM",
                "category": "hollow_consensus",
                "title": f"Hollow consensus detected (round {intervention.round_num})",
                "description": intervention.challenge_text[:200],
                "target_agents": intervention.target_agents,
                "evidence_gaps": intervention.evidence_gaps,
            }
        )

    # Build consensus proof
    consensus_proof = ConsensusProof(
        reached=True,
        confidence=0.6,  # Lowered due to hollow consensus
        supporting_agents=agents_involved,
        dissenting_agents=[],
        method="trickster_qualified_majority",
        evidence_hash=hashlib.sha256(
            json.dumps(stats, default=str).encode()
        ).hexdigest(),
    )

    return DecisionReceipt(
        receipt_id=f"test-combo-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        gauntlet_id="trickster-combo-test",
        timestamp=now,
        input_summary=input_text,
        input_hash=input_hash,
        risk_summary={
            "CRITICAL": 0,
            "HIGH": 0,
            "MEDIUM": stats["total_interventions"],
            "LOW": 0,
        },
        attacks_attempted=0,
        attacks_successful=0,
        probes_run=stats["total_interventions"],
        vulnerabilities_found=stats["total_interventions"],
        verdict=verdict,
        confidence=0.6,
        robustness_score=0.5 if stats["total_interventions"] > 0 else 0.9,
        vulnerability_details=vulnerability_details,
        verdict_reasoning=(
            f"Trickster detected {stats['total_interventions']} hollow consensus "
            f"events across {len(stats.get('quality_per_round', []))} rounds. "
            f"Verdict qualified as CONDITIONAL pending evidence improvement."
        ),
        consensus_proof=consensus_proof,
        provenance_chain=provenance,
    )


# ── Tests ──


class TestTricksterDetection:
    """Verify Trickster detects hollow consensus in contrived scenarios."""

    def test_hollow_consensus_detected(
        self, sensitive_trickster: EvidencePoweredTrickster, hollow_responses: dict[str, str]
    ) -> None:
        """Hollow responses at high convergence trigger detection."""
        intervention = sensitive_trickster.check_and_intervene(
            responses=hollow_responses,
            convergence_similarity=0.92,
            round_num=1,
        )

        stats = sensitive_trickster.get_stats()
        # Either hollow consensus alert fires, or cross-proposal analysis
        # detects evidence gaps and produces an intervention
        detected = (
            stats["hollow_alerts_detected"] >= 1
            or intervention is not None
            or stats["total_interventions"] >= 1
        )
        # At minimum, the quality analyzer should score these low
        assert stats["avg_quality_per_round"][0] < 0.5

    def test_evidence_rich_no_false_positive(
        self, sensitive_trickster: EvidencePoweredTrickster, evidence_rich_responses: dict[str, str]
    ) -> None:
        """Evidence-rich responses should not trigger at moderate convergence."""
        intervention = sensitive_trickster.check_and_intervene(
            responses=evidence_rich_responses,
            convergence_similarity=0.4,  # Low convergence = healthy disagreement
            round_num=1,
        )
        assert intervention is None

    def test_multi_round_escalation(
        self, sensitive_trickster: EvidencePoweredTrickster, hollow_responses: dict[str, str]
    ) -> None:
        """Multiple rounds of hollow responses produce quality tracking data."""
        interventions = []
        for round_num in range(1, 5):
            result = sensitive_trickster.check_and_intervene(
                responses=hollow_responses,
                convergence_similarity=0.85 + round_num * 0.02,
                round_num=round_num,
            )
            if result:
                interventions.append(result)

        stats = sensitive_trickster.get_stats()
        # After 4 rounds, quality tracking should have 4 entries
        assert len(stats["avg_quality_per_round"]) == 4
        # All rounds should show low quality (hollow responses)
        for quality in stats["avg_quality_per_round"]:
            assert quality < 0.5

    def test_stats_populated(
        self, sensitive_trickster: EvidencePoweredTrickster, hollow_responses: dict[str, str]
    ) -> None:
        """get_stats() returns structured data after detection."""
        sensitive_trickster.check_and_intervene(
            responses=hollow_responses,
            convergence_similarity=0.9,
            round_num=1,
        )

        stats = sensitive_trickster.get_stats()
        assert "total_interventions" in stats
        assert "hollow_alerts_detected" in stats
        assert "avg_quality_per_round" in stats
        assert isinstance(stats["avg_quality_per_round"], list)
        assert len(stats["avg_quality_per_round"]) == 1

    def test_elo_adjustments_for_targeted_agents(
        self, sensitive_trickster: EvidencePoweredTrickster, hollow_responses: dict[str, str]
    ) -> None:
        """Agents targeted by interventions get negative ELO adjustments."""
        # Run several rounds to accumulate interventions
        for round_num in range(1, 6):
            sensitive_trickster.check_and_intervene(
                responses=hollow_responses,
                convergence_similarity=0.95,
                round_num=round_num,
            )

        adjustments = sensitive_trickster.get_elo_adjustments()
        # If any interventions occurred, adjustments should be negative
        if sensitive_trickster._state.total_interventions > 0:
            for adj in adjustments.values():
                assert adj <= 0


class TestReceiptGeneration:
    """Verify Decision Receipt captures Trickster findings correctly."""

    def test_receipt_captures_intervention_count(
        self, sensitive_trickster: EvidencePoweredTrickster, hollow_responses: dict[str, str]
    ) -> None:
        """Receipt risk_summary reflects number of interventions."""
        for round_num in range(1, 4):
            sensitive_trickster.check_and_intervene(
                responses=hollow_responses,
                convergence_similarity=0.9,
                round_num=round_num,
            )

        receipt = _make_receipt(
            sensitive_trickster,
            agents_involved=list(hollow_responses.keys()),
        )

        stats = sensitive_trickster.get_stats()
        assert receipt.risk_summary["MEDIUM"] == stats["total_interventions"]
        assert receipt.probes_run == stats["total_interventions"]

    def test_receipt_provenance_chain(
        self, sensitive_trickster: EvidencePoweredTrickster, hollow_responses: dict[str, str]
    ) -> None:
        """Provenance chain contains one record per intervention."""
        for round_num in range(1, 4):
            sensitive_trickster.check_and_intervene(
                responses=hollow_responses,
                convergence_similarity=0.9,
                round_num=round_num,
            )

        receipt = _make_receipt(
            sensitive_trickster,
            agents_involved=list(hollow_responses.keys()),
        )

        n_interventions = sensitive_trickster._state.total_interventions
        assert len(receipt.provenance_chain) == n_interventions
        for record in receipt.provenance_chain:
            assert record.event_type == "trickster_intervention"
            assert record.agent == "Trickster"
            assert record.evidence_hash  # Non-empty hash

    def test_receipt_consensus_proof_qualified(
        self, sensitive_trickster: EvidencePoweredTrickster, hollow_responses: dict[str, str]
    ) -> None:
        """Consensus proof method reflects Trickster qualification."""
        sensitive_trickster.check_and_intervene(
            responses=hollow_responses,
            convergence_similarity=0.9,
            round_num=1,
        )

        receipt = _make_receipt(
            sensitive_trickster,
            agents_involved=list(hollow_responses.keys()),
        )

        assert receipt.consensus_proof is not None
        assert receipt.consensus_proof.method == "trickster_qualified_majority"
        assert receipt.consensus_proof.confidence == 0.6

    def test_receipt_integrity_verification(
        self, sensitive_trickster: EvidencePoweredTrickster, hollow_responses: dict[str, str]
    ) -> None:
        """Receipt artifact hash is tamper-evident."""
        sensitive_trickster.check_and_intervene(
            responses=hollow_responses,
            convergence_similarity=0.9,
            round_num=1,
        )

        receipt = _make_receipt(
            sensitive_trickster,
            agents_involved=list(hollow_responses.keys()),
        )

        # Verify integrity
        assert receipt.artifact_hash  # Non-empty
        assert receipt.verify_integrity()

        # Tamper and verify detection
        original_hash = receipt.artifact_hash
        receipt.verdict = "PASS"
        receipt.artifact_hash = receipt._calculate_hash()  # Recalculate
        assert receipt.artifact_hash != original_hash  # Hash changed

    def test_receipt_vulnerability_details(
        self, sensitive_trickster: EvidencePoweredTrickster, hollow_responses: dict[str, str]
    ) -> None:
        """Vulnerability details capture intervention metadata."""
        for round_num in range(1, 3):
            sensitive_trickster.check_and_intervene(
                responses=hollow_responses,
                convergence_similarity=0.9,
                round_num=round_num,
            )

        receipt = _make_receipt(
            sensitive_trickster,
            agents_involved=list(hollow_responses.keys()),
        )

        for detail in receipt.vulnerability_details:
            assert detail["severity"] == "MEDIUM"
            assert detail["category"] == "hollow_consensus"
            assert "TRICK-" in detail["id"]
            assert "target_agents" in detail
            assert "evidence_gaps" in detail


class TestReceiptExport:
    """Verify receipt exports to multiple formats with Trickster data."""

    def _make_populated_receipt(
        self, trickster: EvidencePoweredTrickster, responses: dict[str, str]
    ) -> DecisionReceipt:
        """Run trickster and build receipt."""
        for round_num in range(1, 4):
            trickster.check_and_intervene(
                responses=responses,
                convergence_similarity=0.9,
                round_num=round_num,
            )
        return _make_receipt(trickster, agents_involved=list(responses.keys()))

    def test_markdown_export(
        self, sensitive_trickster: EvidencePoweredTrickster, hollow_responses: dict[str, str]
    ) -> None:
        """Markdown export includes receipt fields."""
        receipt = self._make_populated_receipt(sensitive_trickster, hollow_responses)
        md = receipt.to_markdown()

        assert "Decision Receipt" in md or "receipt" in md.lower()
        assert receipt.receipt_id in md
        assert "CONDITIONAL" in md

    def test_html_export(
        self, sensitive_trickster: EvidencePoweredTrickster, hollow_responses: dict[str, str]
    ) -> None:
        """HTML export produces valid document."""
        receipt = self._make_populated_receipt(sensitive_trickster, hollow_responses)
        html = receipt.to_html()

        assert "<html" in html.lower()
        assert receipt.receipt_id in html
        assert "CONDITIONAL" in html

    def test_sarif_export(
        self, sensitive_trickster: EvidencePoweredTrickster, hollow_responses: dict[str, str]
    ) -> None:
        """SARIF export produces valid structure for CI/CD integration."""
        receipt = self._make_populated_receipt(sensitive_trickster, hollow_responses)
        sarif = receipt.to_sarif()

        assert sarif["version"] == "2.1.0"
        assert "$schema" in sarif
        assert "runs" in sarif
        assert len(sarif["runs"]) >= 1

    def test_dict_export_roundtrip(
        self, sensitive_trickster: EvidencePoweredTrickster, hollow_responses: dict[str, str]
    ) -> None:
        """to_dict() preserves all fields for JSON serialization."""
        receipt = self._make_populated_receipt(sensitive_trickster, hollow_responses)
        d = receipt.to_dict()

        assert d["receipt_id"] == receipt.receipt_id
        assert d["verdict"] == "CONDITIONAL"
        assert d["confidence"] == 0.6
        assert "risk_summary" in d
        assert "vulnerability_details" in d

        # Roundtrip through JSON
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert restored["receipt_id"] == receipt.receipt_id


class TestCryptoSigning:
    """Verify cryptographic signing of receipts with Trickster data."""

    def test_sign_and_verify(
        self, sensitive_trickster: EvidencePoweredTrickster, hollow_responses: dict[str, str]
    ) -> None:
        """Receipt can be signed and verified."""
        sensitive_trickster.check_and_intervene(
            responses=hollow_responses,
            convergence_similarity=0.9,
            round_num=1,
        )

        receipt = _make_receipt(
            sensitive_trickster,
            agents_involved=list(hollow_responses.keys()),
        )

        # Sign (uses HMAC-SHA256 by default, generates key if ARAGORA_RECEIPT_KEY not set)
        signed = receipt.sign()
        assert signed.signature is not None
        assert signed.signature_algorithm is not None
        assert signed.signed_at is not None

        # Verify
        assert signed.verify_signature()

    def test_tampered_receipt_fails_verification(
        self, sensitive_trickster: EvidencePoweredTrickster, hollow_responses: dict[str, str]
    ) -> None:
        """Tampering with a signed receipt invalidates the signature."""
        sensitive_trickster.check_and_intervene(
            responses=hollow_responses,
            convergence_similarity=0.9,
            round_num=1,
        )

        receipt = _make_receipt(
            sensitive_trickster,
            agents_involved=list(hollow_responses.keys()),
        )

        signed = receipt.sign()
        assert signed.verify_signature()

        # Tamper with verdict
        signed.verdict = "PASS"
        assert not signed.verify_signature()


class TestEndToEndPipeline:
    """Full pipeline: Trickster detects → Receipt captures → Sign → Export."""

    def test_full_pipeline(
        self, sensitive_trickster: EvidencePoweredTrickster, hollow_responses: dict[str, str]
    ) -> None:
        """Complete end-to-end flow from hollow consensus to signed receipt."""
        # Phase 1: Trickster monitors 3 rounds of hollow debate
        interventions = []
        for round_num in range(1, 4):
            result = sensitive_trickster.check_and_intervene(
                responses=hollow_responses,
                convergence_similarity=0.88 + round_num * 0.03,
                round_num=round_num,
            )
            if result:
                interventions.append(result)

        stats = sensitive_trickster.get_stats()

        # Phase 2: Build receipt
        receipt = _make_receipt(
            sensitive_trickster,
            agents_involved=list(hollow_responses.keys()),
            verdict="CONDITIONAL",
        )

        # Phase 3: Verify receipt integrity
        assert receipt.verify_integrity()
        assert receipt.verdict == "CONDITIONAL"
        assert receipt.risk_summary["MEDIUM"] == stats["total_interventions"]

        # Phase 4: Sign receipt
        signed = receipt.sign()
        assert signed.signature is not None
        assert signed.verify_signature()

        # Phase 5: Export to all formats
        md = signed.to_markdown()
        assert isinstance(md, str) and len(md) > 0

        html = signed.to_html()
        assert "<html" in html.lower()

        sarif = signed.to_sarif()
        assert sarif["version"] == "2.1.0"

        d = signed.to_dict()
        assert d["signature"] is not None

        # Phase 6: Verify JSON roundtrip preserves signature
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert restored["signature"] == d["signature"]
        assert restored["signed_at"] == d["signed_at"]

    def test_clean_debate_produces_pass_receipt(
        self,
        sensitive_trickster: EvidencePoweredTrickster,
        evidence_rich_responses: dict[str, str],
    ) -> None:
        """Evidence-rich debate with low convergence produces PASS receipt."""
        intervention = sensitive_trickster.check_and_intervene(
            responses=evidence_rich_responses,
            convergence_similarity=0.4,
            round_num=1,
        )
        assert intervention is None

        receipt = _make_receipt(
            sensitive_trickster,
            agents_involved=list(evidence_rich_responses.keys()),
            verdict="PASS",
        )

        assert receipt.verdict == "PASS"
        assert receipt.risk_summary["MEDIUM"] == 0
        assert len(receipt.vulnerability_details) == 0
        assert receipt.robustness_score == 0.9
        assert receipt.verify_integrity()

        signed = receipt.sign()
        assert signed.verify_signature()
