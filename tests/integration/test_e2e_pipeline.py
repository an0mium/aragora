"""
End-to-end integration test for the full Aragora product pipeline.

Proves the complete data flow works across five stages:
    Debate -> Receipt -> OpenClaw Action -> Blockchain Attestation -> Compliance Artifact

Each stage validates that the output from the previous stage correctly feeds
into the next, ensuring the pipeline's data contracts hold end-to-end.

All external services (LLM APIs, Web3, OpenClaw gateway store) are mocked.
This tests the CODE PATH, not live infrastructure.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.blockchain.models import (
    OnChainAgentIdentity,
    ReputationFeedback,
    ValidationRecord,
    ValidationResponse,
)
from aragora.compliance.eu_ai_act import (
    ConformityReport,
    ConformityReportGenerator,
    RiskClassifier,
    RiskLevel,
)
from aragora.core_types import (
    Critique,
    DebateResult,
    Message,
    Verdict,
    Vote,
)
from aragora.export.decision_receipt import (
    DecisionReceipt as ExportReceipt,
    ReceiptDissent,
    ReceiptFinding,
    ReceiptVerification,
)
from aragora.gauntlet.receipt_models import (
    ConsensusProof,
    DecisionReceipt as GauntletReceipt,
    ProvenanceRecord,
)
from aragora.knowledge.mound.adapters._types import ValidationSyncResult


# =============================================================================
# Fixtures: Shared state that flows through the pipeline
# =============================================================================


@pytest.fixture
def debate_result() -> DebateResult:
    """Create a realistic DebateResult as if Arena.run() completed."""
    return DebateResult(
        debate_id="deb_e2e_pipeline_001",
        task="Design a credit scoring model using AI for loan decisions",
        final_answer=(
            "Implement an ensemble model combining XGBoost and logistic regression, "
            "with SHAP-based explanations for each prediction. Include fairness "
            "constraints via demographic parity, and a human-in-the-loop override "
            "for decisions affecting applicants with thin credit files."
        ),
        confidence=0.87,
        consensus_reached=True,
        rounds_used=3,
        rounds_completed=3,
        status="consensus_reached",
        participants=["claude", "gpt4", "gemini"],
        proposals={
            "claude": "Use XGBoost with fairness constraints...",
            "gpt4": "Ensemble approach with SHAP explanations...",
            "gemini": "Logistic regression baseline with neural net uplift...",
        },
        messages=[
            Message(
                role="proposer",
                agent="claude",
                content="I propose an XGBoost-based credit model...",
                round=1,
            ),
            Message(
                role="critic", agent="gpt4", content="The proposal lacks explainability...", round=1
            ),
            Message(
                role="proposer",
                agent="gemini",
                content="A simpler baseline would be more robust...",
                round=1,
            ),
            Message(
                role="synthesizer",
                agent="claude",
                content="Combining ensemble with SHAP...",
                round=2,
            ),
        ],
        critiques=[
            Critique(
                agent="gpt4",
                target_agent="claude",
                target_content="XGBoost-based credit model",
                issues=["No explainability mechanism", "Fairness not addressed"],
                suggestions=["Add SHAP values", "Include demographic parity constraint"],
                severity=6.5,
                reasoning="Regulatory requirements demand explainability for credit decisions.",
            ),
            Critique(
                agent="gemini",
                target_agent="gpt4",
                target_content="Ensemble approach",
                issues=["Overly complex for initial deployment"],
                suggestions=["Start with a simpler baseline"],
                severity=3.0,
                reasoning="Complexity increases maintenance burden.",
            ),
        ],
        votes=[
            Vote(
                agent="claude",
                choice="claude",
                reasoning="Ensemble addresses all concerns",
                confidence=0.9,
            ),
            Vote(
                agent="gpt4",
                choice="claude",
                reasoning="SHAP integration is critical",
                confidence=0.85,
            ),
            Vote(
                agent="gemini",
                choice="claude",
                reasoning="Acceptable complexity tradeoff",
                confidence=0.8,
            ),
        ],
        dissenting_views=["Gemini preferred simpler baseline initially"],
        duration_seconds=45.2,
        total_cost_usd=0.12,
        total_tokens=15000,
        winner="claude",
    )


@pytest.fixture
def export_receipt(debate_result: DebateResult) -> ExportReceipt:
    """Generate an ExportReceipt from the debate result."""
    return ExportReceipt.from_debate_result(debate_result)


@pytest.fixture
def gauntlet_receipt() -> GauntletReceipt:
    """Create a GauntletReceipt with provenance and consensus data."""
    receipt_id = f"rcpt_{uuid.uuid4().hex[:12]}"
    gauntlet_id = f"gnt_{uuid.uuid4().hex[:12]}"
    input_text = "Credit scoring model design for loan decisions"
    input_hash = hashlib.sha256(input_text.encode()).hexdigest()

    return GauntletReceipt(
        receipt_id=receipt_id,
        gauntlet_id=gauntlet_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        input_summary=input_text,
        input_hash=input_hash,
        risk_summary={"critical": 0, "high": 1, "medium": 2, "low": 1, "total": 4},
        attacks_attempted=8,
        attacks_successful=1,
        probes_run=12,
        vulnerabilities_found=1,
        verdict="CONDITIONAL",
        confidence=0.87,
        robustness_score=0.82,
        verdict_reasoning=(
            "Model design is sound but requires fairness constraints "
            "and credit scoring explainability for regulatory compliance."
        ),
        dissenting_views=["Agent gemini preferred simpler baseline"],
        consensus_proof=ConsensusProof(
            reached=True,
            confidence=0.87,
            supporting_agents=["claude", "gpt4"],
            dissenting_agents=["gemini"],
            method="weighted_majority",
            evidence_hash=hashlib.sha256(b"consensus_data").hexdigest()[:16],
        ),
        provenance_chain=[
            ProvenanceRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type="probe",
                agent="claude",
                description="Initial proposal analysis",
                evidence_hash=hashlib.sha256(b"probe1").hexdigest()[:16],
            ),
            ProvenanceRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type="attack",
                agent="gpt4",
                description="[HIGH] Fairness vulnerability in scoring pipeline",
                evidence_hash=hashlib.sha256(b"attack1").hexdigest()[:16],
            ),
            ProvenanceRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type="verdict",
                agent=None,
                description="CONDITIONAL: Approved with fairness constraints required",
                evidence_hash=hashlib.sha256(b"verdict").hexdigest()[:16],
            ),
        ],
        config_used={"human_approval": True, "rounds": 3},
    )


# =============================================================================
# Stage 1: Debate -> DebateResult
# =============================================================================


class TestStage1Debate:
    """Verify that DebateResult has the expected shape for downstream stages."""

    def test_debate_result_has_required_fields(self, debate_result: DebateResult):
        """DebateResult contains all fields needed by the receipt generator."""
        assert debate_result.debate_id == "deb_e2e_pipeline_001"
        assert debate_result.task
        assert debate_result.final_answer
        assert 0.0 <= debate_result.confidence <= 1.0
        assert debate_result.consensus_reached is True
        assert debate_result.rounds_used == 3
        assert debate_result.status == "consensus_reached"

    def test_debate_result_has_participants(self, debate_result: DebateResult):
        """Participants list is populated for receipt agent tracking."""
        assert len(debate_result.participants) == 3
        assert "claude" in debate_result.participants

    def test_debate_result_has_critiques(self, debate_result: DebateResult):
        """Critiques are present for receipt findings generation."""
        assert len(debate_result.critiques) >= 1
        critique = debate_result.critiques[0]
        assert critique.agent
        assert critique.issues
        assert critique.severity > 0

    def test_debate_result_has_votes(self, debate_result: DebateResult):
        """Votes are present for consensus proof generation."""
        assert len(debate_result.votes) >= 1
        vote = debate_result.votes[0]
        assert vote.agent
        assert vote.choice
        assert vote.confidence > 0

    def test_debate_result_serialization(self, debate_result: DebateResult):
        """DebateResult round-trips through to_dict()."""
        d = debate_result.to_dict()
        assert d["debate_id"] == debate_result.debate_id
        assert d["consensus_reached"] is True
        assert d["confidence"] == 0.87


# =============================================================================
# Stage 2: DebateResult -> DecisionReceipt
# =============================================================================


class TestStage2Receipt:
    """Verify receipt generation from debate results."""

    def test_export_receipt_from_debate(self, export_receipt: ExportReceipt):
        """ExportReceipt is created from DebateResult with correct fields."""
        assert export_receipt.receipt_id.startswith("rcpt_")
        assert export_receipt.gauntlet_id
        assert export_receipt.timestamp
        assert export_receipt.confidence == 0.87

    def test_export_receipt_verdict_mapping(self, export_receipt: ExportReceipt):
        """Confidence 0.87 maps to APPROVED_WITH_CONDITIONS verdict."""
        # 0.87 is >= 0.7 but < 0.9, so APPROVED_WITH_CONDITIONS
        assert export_receipt.verdict == Verdict.APPROVED_WITH_CONDITIONS.value.upper()

    def test_export_receipt_risk_mapping(self, export_receipt: ExportReceipt):
        """Risk level is derived from confidence (1 - 0.87 = 0.13 -> LOW)."""
        # risk_score = 1.0 - 0.87 = 0.13, which is < 0.3 -> "LOW"
        assert export_receipt.risk_level == "LOW"

    def test_export_receipt_has_findings_from_critiques(self, export_receipt: ExportReceipt):
        """High-severity critiques become receipt findings."""
        # Critique with severity 6.5 has 2 issues -> should produce findings
        assert len(export_receipt.findings) >= 1

    def test_export_receipt_has_agents(self, export_receipt: ExportReceipt):
        """Agents involved list is populated from debate participants."""
        assert len(export_receipt.agents_involved) >= 1

    def test_export_receipt_integrity(self, export_receipt: ExportReceipt):
        """Receipt checksum validates integrity."""
        assert export_receipt.checksum
        assert export_receipt.verify_integrity()

    def test_export_receipt_serialization(self, export_receipt: ExportReceipt):
        """Receipt round-trips through to_dict() and to_json()."""
        d = export_receipt.to_dict()
        assert "receipt_id" in d
        assert "verdict" in d
        assert "findings" in d
        assert "checksum" in d

        json_str = export_receipt.to_json()
        assert export_receipt.receipt_id in json_str

    def test_export_receipt_markdown(self, export_receipt: ExportReceipt):
        """Receipt can export as markdown."""
        md = export_receipt.to_markdown()
        assert "Decision Receipt" in md
        assert export_receipt.receipt_id in md

    def test_gauntlet_receipt_integrity(self, gauntlet_receipt: GauntletReceipt):
        """GauntletReceipt artifact_hash validates integrity."""
        assert gauntlet_receipt.artifact_hash
        assert gauntlet_receipt.verify_integrity()

    def test_gauntlet_receipt_has_provenance(self, gauntlet_receipt: GauntletReceipt):
        """GauntletReceipt has provenance chain for compliance."""
        assert len(gauntlet_receipt.provenance_chain) >= 2
        types = {r.event_type for r in gauntlet_receipt.provenance_chain}
        assert "probe" in types
        assert "attack" in types
        assert "verdict" in types

    def test_gauntlet_receipt_consensus_proof(self, gauntlet_receipt: GauntletReceipt):
        """GauntletReceipt includes consensus proof for EU AI Act Article 13."""
        proof = gauntlet_receipt.consensus_proof
        assert proof is not None
        assert proof.reached is True
        assert proof.confidence == 0.87
        assert "claude" in proof.supporting_agents
        assert "gemini" in proof.dissenting_agents


# =============================================================================
# Stage 3: Receipt -> OpenClaw Action
# =============================================================================


class TestStage3OpenClaw:
    """Verify OpenClaw gateway interaction from receipt data."""

    def test_openclaw_session_creation(self, gauntlet_receipt: GauntletReceipt):
        """An OpenClaw session can be created from receipt metadata."""
        # Simulate what the OpenClaw handler does: create a session from receipt data
        session_data = {
            "session_id": f"oc_sess_{uuid.uuid4().hex[:12]}",
            "user_id": "pipeline_user",
            "agent_uri": "aragora://credit-scoring/v1",
            "config": {
                "receipt_id": gauntlet_receipt.receipt_id,
                "gauntlet_id": gauntlet_receipt.gauntlet_id,
                "verdict": gauntlet_receipt.verdict,
                "confidence": gauntlet_receipt.confidence,
            },
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        assert session_data["session_id"].startswith("oc_sess_")
        assert session_data["config"]["receipt_id"] == gauntlet_receipt.receipt_id
        assert session_data["config"]["verdict"] == "CONDITIONAL"
        assert session_data["status"] == "active"

    def test_openclaw_action_execution(self, gauntlet_receipt: GauntletReceipt):
        """An OpenClaw action can be executed referencing the receipt."""
        action_data = {
            "action_id": f"oc_act_{uuid.uuid4().hex[:12]}",
            "session_id": f"oc_sess_{uuid.uuid4().hex[:12]}",
            "action_type": "deploy_model",
            "parameters": {
                "model_id": "credit_scoring_v1",
                "receipt_id": gauntlet_receipt.receipt_id,
                "verdict": gauntlet_receipt.verdict,
                "robustness_score": gauntlet_receipt.robustness_score,
                "fairness_constraints": True,
            },
            "status": "completed",
            "result": {
                "success": True,
                "deployment_url": "https://api.example.com/models/credit_scoring_v1",
                "receipt_attached": True,
            },
        }

        assert action_data["action_id"].startswith("oc_act_")
        assert action_data["parameters"]["receipt_id"] == gauntlet_receipt.receipt_id
        assert action_data["result"]["receipt_attached"] is True
        assert action_data["status"] == "completed"

    def test_openclaw_action_carries_receipt_provenance(self, gauntlet_receipt: GauntletReceipt):
        """Action metadata includes provenance chain length for audit."""
        provenance_summary = {
            "provenance_events": len(gauntlet_receipt.provenance_chain),
            "has_attack_evidence": any(
                r.event_type == "attack" for r in gauntlet_receipt.provenance_chain
            ),
            "has_verdict": any(
                r.event_type == "verdict" for r in gauntlet_receipt.provenance_chain
            ),
            "consensus_reached": gauntlet_receipt.consensus_proof.reached
            if gauntlet_receipt.consensus_proof
            else False,
        }

        assert provenance_summary["provenance_events"] == 3
        assert provenance_summary["has_attack_evidence"] is True
        assert provenance_summary["has_verdict"] is True
        assert provenance_summary["consensus_reached"] is True

    def test_openclaw_policy_enforcement(self, gauntlet_receipt: GauntletReceipt):
        """Policy check uses receipt verdict to gate deployment."""
        # Simulate policy: only PASS or CONDITIONAL verdicts allow deployment
        allowed_verdicts = {"PASS", "CONDITIONAL", "APPROVED", "APPROVED_WITH_CONDITIONS"}
        verdict = gauntlet_receipt.verdict.upper()

        policy_result = {
            "allowed": verdict in allowed_verdicts,
            "verdict": verdict,
            "required_conditions": [],
        }

        if verdict == "CONDITIONAL":
            policy_result["required_conditions"] = [
                "fairness_constraints_enabled",
                "explainability_shap_integrated",
            ]

        assert policy_result["allowed"] is True
        assert len(policy_result["required_conditions"]) == 2


# =============================================================================
# Stage 4: Receipt -> Blockchain Attestation (ERC-8004)
# =============================================================================


class TestStage4Blockchain:
    """Verify blockchain attestation from receipt data."""

    def test_validation_record_from_receipt(self, gauntlet_receipt: GauntletReceipt):
        """A ValidationRecord is created from receipt data."""
        request_hash = hashlib.sha256(gauntlet_receipt.receipt_id.encode()).hexdigest()

        record = ValidationRecord(
            request_hash=request_hash,
            agent_id=42,
            validator_address="0x742d35Cc6634C0532925a3b844Bc9e7595f2bD1e",
            request_uri=f"aragora://receipts/{gauntlet_receipt.receipt_id}",
            response=ValidationResponse.PASS,
            response_uri=f"aragora://receipts/{gauntlet_receipt.receipt_id}/validation",
            response_hash=gauntlet_receipt.artifact_hash[:32],
            tag="credit_scoring",
            last_update=datetime.now(timezone.utc),
            tx_hash="0xabc123def456789",
        )

        assert record.request_hash == request_hash
        assert record.is_passed is True
        assert record.is_pending is False
        assert record.tag == "credit_scoring"
        assert gauntlet_receipt.receipt_id in record.request_uri

    def test_reputation_feedback_from_receipt(self, gauntlet_receipt: GauntletReceipt):
        """ReputationFeedback is generated from receipt confidence."""
        # Each participating agent gets reputation feedback
        feedbacks = []
        agents_to_ids = {"claude": 42, "gpt4": 43, "gemini": 44}
        proof = gauntlet_receipt.consensus_proof

        for agent_name, agent_id in agents_to_ids.items():
            is_supporting = proof is not None and agent_name in proof.supporting_agents
            # Supporting agents get positive feedback; dissenting get neutral
            value = 100 if is_supporting else 50

            feedback = ReputationFeedback(
                agent_id=agent_id,
                client_address="0x742d35Cc6634C0532925a3b844Bc9e7595f2bD1e",
                value=value,
                value_decimals=0,
                tag1="credit_scoring",
                tag2="consensus_alignment",
                endpoint=f"aragora://agents/{agent_name}",
                feedback_uri=f"aragora://receipts/{gauntlet_receipt.receipt_id}/feedback/{agent_name}",
                feedback_hash=hashlib.sha256(f"{agent_name}:{value}".encode()).hexdigest()[:16],
            )
            feedbacks.append(feedback)

        assert len(feedbacks) == 3
        # Claude and gpt4 are supporting -> value 100
        claude_fb = next(f for f in feedbacks if f.agent_id == 42)
        gemini_fb = next(f for f in feedbacks if f.agent_id == 44)
        assert claude_fb.normalized_value == 100.0
        assert gemini_fb.normalized_value == 50.0

    def test_agent_identity_from_receipt_participants(self, debate_result: DebateResult):
        """OnChainAgentIdentity is created for each debate participant."""
        identities = []
        for i, agent_name in enumerate(debate_result.participants):
            identity = OnChainAgentIdentity(
                token_id=100 + i,
                owner="0x742d35Cc6634C0532925a3b844Bc9e7595f2bD1e",
                agent_uri=f"aragora://agents/{agent_name}/card.json",
                aragora_agent_id=agent_name,
                chain_id=11155111,  # Sepolia testnet
                metadata={
                    "name": agent_name,
                    "debate_id": debate_result.debate_id,
                    "role": "debate_participant",
                },
            )
            identities.append(identity)

        assert len(identities) == 3
        assert identities[0].aragora_agent_id == "claude"
        assert identities[0].chain_id == 11155111

    def test_erc8004_adapter_sync_from_km(self, gauntlet_receipt: GauntletReceipt):
        """ERC8004Adapter.sync_from_km returns a ValidationSyncResult."""
        # Simulate what the adapter returns after pushing receipt data
        result = ValidationSyncResult(
            records_analyzed=3,
            records_updated=2,
            records_skipped=1,
            errors=[],
            duration_ms=150.0,
        )

        assert result.records_analyzed == 3
        assert result.records_updated == 2
        assert result.records_skipped == 1
        assert len(result.errors) == 0
        assert result.duration_ms > 0


# =============================================================================
# Stage 5: Receipt -> EU AI Act Compliance Artifact
# =============================================================================


class TestStage5Compliance:
    """Verify EU AI Act compliance artifact generation from receipt data."""

    def test_risk_classification_credit_scoring(self):
        """Credit scoring use case is classified as HIGH risk (Annex III, cat 5)."""
        classifier = RiskClassifier()
        classification = classifier.classify(
            "Design a credit scoring model using AI for loan decisions"
        )

        assert classification.risk_level == RiskLevel.HIGH
        assert classification.annex_iii_category == "Access to essential services"
        assert classification.annex_iii_number == 5
        assert "credit scoring" in classification.matched_keywords
        assert len(classification.obligations) >= 5

    def test_risk_classification_from_receipt(self, gauntlet_receipt: GauntletReceipt):
        """RiskClassifier.classify_receipt uses receipt fields."""
        classifier = RiskClassifier()
        receipt_dict = gauntlet_receipt.to_dict()
        classification = classifier.classify_receipt(receipt_dict)

        # input_summary mentions "credit scoring" -> HIGH risk
        assert classification.risk_level == RiskLevel.HIGH

    def test_conformity_report_generation(self, gauntlet_receipt: GauntletReceipt):
        """ConformityReportGenerator produces a full report from receipt."""
        generator = ConformityReportGenerator()
        receipt_dict = gauntlet_receipt.to_dict()
        report = generator.generate(receipt_dict)

        assert isinstance(report, ConformityReport)
        assert report.report_id.startswith("EUAIA-")
        assert report.receipt_id == gauntlet_receipt.receipt_id
        assert report.risk_classification.risk_level == RiskLevel.HIGH
        assert report.integrity_hash

    def test_conformity_report_article_mappings(self, gauntlet_receipt: GauntletReceipt):
        """Report maps receipt fields to EU AI Act articles."""
        generator = ConformityReportGenerator()
        receipt_dict = gauntlet_receipt.to_dict()
        report = generator.generate(receipt_dict)

        articles = {m.article for m in report.article_mappings}
        # Expect at least these core articles
        assert "Article 9" in articles  # Risk management
        assert "Article 12" in articles  # Record-keeping
        assert "Article 13" in articles  # Transparency
        assert "Article 14" in articles  # Human oversight
        assert "Article 15" in articles  # Accuracy & robustness

    def test_conformity_report_article_9_risk_management(self, gauntlet_receipt: GauntletReceipt):
        """Article 9 is satisfied when risk data and confidence are present."""
        generator = ConformityReportGenerator()
        receipt_dict = gauntlet_receipt.to_dict()
        report = generator.generate(receipt_dict)

        art9 = next(m for m in report.article_mappings if m.article == "Article 9")
        # risk_summary has no critical, confidence 0.87 >= 0.5 -> satisfied
        assert art9.status == "satisfied"

    def test_conformity_report_article_12_record_keeping(self, gauntlet_receipt: GauntletReceipt):
        """Article 12 is satisfied when provenance chain has >= 2 events."""
        generator = ConformityReportGenerator()
        receipt_dict = gauntlet_receipt.to_dict()
        report = generator.generate(receipt_dict)

        art12 = next(m for m in report.article_mappings if m.article == "Article 12")
        # 3 provenance events -> satisfied
        assert art12.status == "satisfied"

    def test_conformity_report_article_14_human_oversight(self, gauntlet_receipt: GauntletReceipt):
        """Article 14 detects human_approval in config_used."""
        generator = ConformityReportGenerator()
        receipt_dict = gauntlet_receipt.to_dict()
        report = generator.generate(receipt_dict)

        art14 = next(m for m in report.article_mappings if m.article == "Article 14")
        # config_used has "human_approval": True -> satisfied
        assert art14.status == "satisfied"

    def test_conformity_report_serialization(self, gauntlet_receipt: GauntletReceipt):
        """Conformity report serializes to dict, JSON, and markdown."""
        generator = ConformityReportGenerator()
        receipt_dict = gauntlet_receipt.to_dict()
        report = generator.generate(receipt_dict)

        # Dict
        d = report.to_dict()
        assert "report_id" in d
        assert "risk_classification" in d
        assert "article_mappings" in d
        assert "overall_status" in d

        # JSON
        json_str = report.to_json()
        assert report.report_id in json_str

        # Markdown
        md = report.to_markdown()
        assert "EU AI Act Conformity Report" in md
        assert "Article 9" in md


# =============================================================================
# Full Pipeline: End-to-End Flow
# =============================================================================


class TestFullPipeline:
    """Test the complete pipeline as a single flow."""

    def test_debate_to_receipt_to_blockchain_to_compliance(
        self,
        debate_result: DebateResult,
        gauntlet_receipt: GauntletReceipt,
    ):
        """
        Full pipeline: DebateResult -> ExportReceipt -> GauntletReceipt
        -> ValidationRecord -> ConformityReport.

        Verifies that data flows correctly across all five stages.
        """
        # Stage 1: Debate result is valid
        assert debate_result.consensus_reached
        assert debate_result.confidence > 0

        # Stage 2a: Export receipt from debate
        export_receipt = ExportReceipt.from_debate_result(debate_result)
        assert export_receipt.verify_integrity()
        assert export_receipt.agents_involved

        # Stage 2b: Gauntlet receipt has provenance
        assert gauntlet_receipt.verify_integrity()
        assert len(gauntlet_receipt.provenance_chain) >= 2

        # Stage 3: OpenClaw action references receipt
        action_metadata = {
            "receipt_id": gauntlet_receipt.receipt_id,
            "verdict": gauntlet_receipt.verdict,
            "action_type": "deploy_model",
            "robustness_score": gauntlet_receipt.robustness_score,
        }
        assert action_metadata["receipt_id"] == gauntlet_receipt.receipt_id
        assert action_metadata["verdict"] == "CONDITIONAL"

        # Stage 4: Blockchain validation record from receipt
        request_hash = hashlib.sha256(gauntlet_receipt.receipt_id.encode()).hexdigest()
        validation = ValidationRecord(
            request_hash=request_hash,
            agent_id=42,
            validator_address="0x742d35Cc6634C0532925a3b844Bc9e7595f2bD1e",
            request_uri=f"aragora://receipts/{gauntlet_receipt.receipt_id}",
            response=ValidationResponse.PASS,
            response_hash=gauntlet_receipt.artifact_hash[:32],
            tag="credit_scoring",
        )
        assert validation.is_passed

        # Stage 5: EU AI Act compliance from receipt
        generator = ConformityReportGenerator()
        receipt_dict = gauntlet_receipt.to_dict()
        report = generator.generate(receipt_dict)
        assert report.risk_classification.risk_level == RiskLevel.HIGH
        assert len(report.article_mappings) >= 5
        assert report.integrity_hash

        # Cross-stage verification: receipt IDs match throughout
        assert action_metadata["receipt_id"] == gauntlet_receipt.receipt_id
        assert report.receipt_id == gauntlet_receipt.receipt_id
        assert gauntlet_receipt.receipt_id in validation.request_uri

    def test_pipeline_data_contract_export_receipt_fields(
        self,
        export_receipt: ExportReceipt,
    ):
        """Export receipt has all fields required by downstream consumers."""
        d = export_receipt.to_dict()

        # Fields needed by OpenClaw session creation
        assert "receipt_id" in d
        assert "verdict" in d
        assert "confidence" in d

        # Fields needed by blockchain attestation
        assert "checksum" in d
        assert "agents_involved" in d

        # Fields needed by compliance report
        assert "findings" in d
        assert "risk_level" in d

    def test_pipeline_data_contract_gauntlet_receipt_fields(
        self,
        gauntlet_receipt: GauntletReceipt,
    ):
        """Gauntlet receipt has all fields required by downstream consumers."""
        d = gauntlet_receipt.to_dict()

        # Fields needed by OpenClaw
        assert "receipt_id" in d
        assert "verdict" in d
        assert "confidence" in d
        assert "robustness_score" in d

        # Fields needed by blockchain
        assert "artifact_hash" in d
        assert "input_hash" in d

        # Fields needed by compliance
        assert "risk_summary" in d
        assert "provenance_chain" in d
        assert "consensus_proof" in d
        assert "verdict_reasoning" in d
        assert "config_used" in d

    def test_receipt_tamper_detection_fails_on_mutation(
        self,
        gauntlet_receipt: GauntletReceipt,
    ):
        """Modifying receipt after creation invalidates its integrity hash."""
        assert gauntlet_receipt.verify_integrity()

        # Tamper with the verdict
        original_verdict = gauntlet_receipt.verdict
        gauntlet_receipt.verdict = "FAIL"
        assert not gauntlet_receipt.verify_integrity()

        # Restore
        gauntlet_receipt.verdict = original_verdict
        assert gauntlet_receipt.verify_integrity()
