"""
Tests for EU AI Act compliance module.

Covers:
- RiskClassifier: all 4 risk levels + all 8 Annex III categories
- ConformityReportGenerator: article mapping, report generation
- ConformityReport: serialization, markdown export
- CLI commands: audit and classify
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from aragora.compliance.eu_ai_act import (
    ANNEX_III_CATEGORIES,
    ArticleMapping,
    ConformityReport,
    ConformityReportGenerator,
    RiskClassification,
    RiskClassifier,
    RiskLevel,
    _detect_human_oversight,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classifier():
    return RiskClassifier()


@pytest.fixture
def generator():
    return ConformityReportGenerator()


@pytest.fixture
def sample_receipt() -> dict:
    """A well-formed receipt dict with all fields."""
    return {
        "receipt_id": "test-receipt-001",
        "gauntlet_id": "gauntlet-001",
        "timestamp": "2026-02-12T00:00:00Z",
        "input_summary": "Evaluate hiring algorithm for recruitment decisions",
        "input_hash": "abc123",
        "risk_summary": {
            "critical": 0,
            "high": 1,
            "medium": 2,
            "low": 0,
            "total": 3,
        },
        "attacks_attempted": 5,
        "attacks_successful": 1,
        "probes_run": 10,
        "vulnerabilities_found": 3,
        "verdict": "CONDITIONAL",
        "confidence": 0.75,
        "robustness_score": 0.8,
        "verdict_reasoning": "Recruitment system shows bias risk in CV screening",
        "dissenting_views": ["Agent-B: potential gender bias not fully mitigated"],
        "consensus_proof": {
            "reached": True,
            "confidence": 0.75,
            "supporting_agents": ["Agent-A", "Agent-C"],
            "dissenting_agents": ["Agent-B"],
            "method": "majority",
            "evidence_hash": "deadbeef",
        },
        "provenance_chain": [
            {
                "timestamp": "2026-02-12T00:00:01Z",
                "event_type": "attack",
                "agent": "Agent-A",
                "description": "[HIGH] Bias test",
                "evidence_hash": "1234",
            },
            {
                "timestamp": "2026-02-12T00:00:02Z",
                "event_type": "verdict",
                "agent": None,
                "description": "Verdict: CONDITIONAL",
                "evidence_hash": "5678",
            },
        ],
        "schema_version": "1.0",
        "artifact_hash": "abcdef1234567890",
        "config_used": {"require_approval": True},
    }


@pytest.fixture
def minimal_receipt() -> dict:
    """A receipt with minimal fields."""
    return {
        "receipt_id": "minimal-001",
        "gauntlet_id": "g-001",
        "timestamp": "2026-02-12T00:00:00Z",
        "input_summary": "Simple chatbot for customer FAQ",
        "input_hash": "xyz",
        "risk_summary": {},
        "verdict": "PASS",
        "confidence": 0.0,
        "robustness_score": 0.0,
        "verdict_reasoning": "",
        "provenance_chain": [],
        "config_used": {},
    }


# ---------------------------------------------------------------------------
# RiskClassifier Tests
# ---------------------------------------------------------------------------

class TestRiskClassifier:
    """Tests for EU AI Act risk classification."""

    def test_unacceptable_social_scoring(self, classifier):
        result = classifier.classify("AI system for social scoring of citizens")
        assert result.risk_level == RiskLevel.UNACCEPTABLE
        assert "social scoring" in result.matched_keywords
        assert "Article 5" in result.applicable_articles[0]

    def test_unacceptable_subliminal_manipulation(self, classifier):
        result = classifier.classify("System using subliminal manipulation techniques")
        assert result.risk_level == RiskLevel.UNACCEPTABLE

    def test_high_risk_biometrics(self, classifier):
        result = classifier.classify("Real-time facial recognition for public surveillance")
        assert result.risk_level == RiskLevel.HIGH
        assert result.annex_iii_category == "Biometrics"
        assert result.annex_iii_number == 1

    def test_high_risk_critical_infrastructure(self, classifier):
        result = classifier.classify("AI managing water supply distribution systems")
        assert result.risk_level == RiskLevel.HIGH
        assert result.annex_iii_category == "Critical infrastructure"
        assert result.annex_iii_number == 2

    def test_high_risk_education(self, classifier):
        result = classifier.classify("Automated student assessment and grading system")
        assert result.risk_level == RiskLevel.HIGH
        assert result.annex_iii_category == "Education and vocational training"
        assert result.annex_iii_number == 3

    def test_high_risk_employment(self, classifier):
        result = classifier.classify("AI for recruitment and CV screening of job applicants")
        assert result.risk_level == RiskLevel.HIGH
        assert result.annex_iii_category == "Employment and worker management"
        assert result.annex_iii_number == 4

    def test_high_risk_essential_services(self, classifier):
        result = classifier.classify("Credit scoring system for loan decision making")
        assert result.risk_level == RiskLevel.HIGH
        assert result.annex_iii_category == "Access to essential services"
        assert result.annex_iii_number == 5

    def test_high_risk_law_enforcement(self, classifier):
        result = classifier.classify("Predictive policing crime prediction system")
        assert result.risk_level == RiskLevel.HIGH
        assert result.annex_iii_category == "Law enforcement"
        assert result.annex_iii_number == 6

    def test_high_risk_migration(self, classifier):
        result = classifier.classify("AI for visa application processing at border control")
        assert result.risk_level == RiskLevel.HIGH
        assert result.annex_iii_category == "Migration, asylum and border control"
        assert result.annex_iii_number == 7

    def test_high_risk_justice(self, classifier):
        result = classifier.classify("AI-assisted judicial sentencing recommendation system")
        assert result.risk_level == RiskLevel.HIGH
        assert result.annex_iii_category == "Administration of justice and democratic processes"
        assert result.annex_iii_number == 8

    def test_limited_risk_chatbot(self, classifier):
        result = classifier.classify("Customer-facing chatbot for product FAQ")
        assert result.risk_level == RiskLevel.LIMITED
        assert "chatbot" in result.matched_keywords
        assert any("Article 50" in a for a in result.applicable_articles)

    def test_limited_risk_deepfake(self, classifier):
        result = classifier.classify("System generating deepfake videos for entertainment")
        assert result.risk_level == RiskLevel.LIMITED

    def test_minimal_risk(self, classifier):
        result = classifier.classify("AI spam filter for internal email")
        assert result.risk_level == RiskLevel.MINIMAL

    def test_minimal_risk_no_keywords(self, classifier):
        result = classifier.classify("Simple data aggregation tool")
        assert result.risk_level == RiskLevel.MINIMAL
        assert not result.matched_keywords

    def test_high_risk_has_obligations(self, classifier):
        result = classifier.classify("Employee performance evaluation system")
        assert result.risk_level == RiskLevel.HIGH
        assert len(result.obligations) > 0
        assert any("risk management" in o.lower() for o in result.obligations)

    def test_high_risk_has_applicable_articles(self, classifier):
        result = classifier.classify("AI for credit scoring decisions")
        assert "Article 9 (Risk management)" in result.applicable_articles
        assert "Article 13 (Transparency)" in result.applicable_articles
        assert "Article 14 (Human oversight)" in result.applicable_articles

    def test_classification_to_dict(self, classifier):
        result = classifier.classify("Biometric identification system")
        d = result.to_dict()
        assert d["risk_level"] == "high"
        assert d["annex_iii_category"] == "Biometrics"
        assert isinstance(d["obligations"], list)

    def test_classify_receipt(self, classifier, sample_receipt):
        result = classifier.classify_receipt(sample_receipt)
        # sample_receipt mentions "recruitment" and "CV screening"
        assert result.risk_level == RiskLevel.HIGH
        assert result.annex_iii_number == 4


# ---------------------------------------------------------------------------
# ConformityReportGenerator Tests
# ---------------------------------------------------------------------------

class TestConformityReportGenerator:
    """Tests for EU AI Act conformity report generation."""

    def test_generate_report_from_receipt(self, generator, sample_receipt):
        report = generator.generate(sample_receipt)
        assert report.receipt_id == "test-receipt-001"
        assert report.report_id.startswith("EUAIA-")
        assert report.risk_classification.risk_level == RiskLevel.HIGH
        assert len(report.article_mappings) > 0

    def test_article_9_risk_management_mapping(self, generator, sample_receipt):
        report = generator.generate(sample_receipt)
        art9 = [m for m in report.article_mappings if m.article == "Article 9"]
        assert len(art9) == 1
        assert "risk" in art9[0].evidence.lower()

    def test_article_12_record_keeping(self, generator, sample_receipt):
        report = generator.generate(sample_receipt)
        art12 = [m for m in report.article_mappings if m.article == "Article 12"]
        assert len(art12) == 1
        # sample_receipt has 2 provenance events -> satisfied
        assert art12[0].status == "satisfied"

    def test_article_13_transparency(self, generator, sample_receipt):
        report = generator.generate(sample_receipt)
        art13 = [m for m in report.article_mappings if m.article == "Article 13"]
        assert len(art13) == 1
        assert "Agent-A" in art13[0].evidence or "3 agents" in art13[0].evidence

    def test_article_14_human_oversight_present(self, generator, sample_receipt):
        report = generator.generate(sample_receipt)
        art14 = [m for m in report.article_mappings if m.article == "Article 14"]
        assert len(art14) == 1
        # sample_receipt has require_approval in config
        assert art14[0].status == "satisfied"

    def test_article_14_human_oversight_absent(self, generator, minimal_receipt):
        report = generator.generate(minimal_receipt)
        art14 = [m for m in report.article_mappings if m.article == "Article 14"]
        assert len(art14) == 1
        assert art14[0].status == "partial"

    def test_article_15_accuracy_robustness(self, generator, sample_receipt):
        report = generator.generate(sample_receipt)
        art15 = [m for m in report.article_mappings if m.article == "Article 15"]
        assert len(art15) == 1
        # robustness_score 0.8 -> satisfied
        assert art15[0].status == "satisfied"

    def test_low_robustness_partial(self, generator, sample_receipt):
        sample_receipt["robustness_score"] = 0.3
        report = generator.generate(sample_receipt)
        art15 = [m for m in report.article_mappings if m.article == "Article 15"]
        assert art15[0].status == "partial"

    def test_very_low_robustness_not_satisfied(self, generator, sample_receipt):
        sample_receipt["robustness_score"] = 0.1
        report = generator.generate(sample_receipt)
        art15 = [m for m in report.article_mappings if m.article == "Article 15"]
        assert art15[0].status == "not_satisfied"

    def test_overall_status_conformant(self, generator, sample_receipt):
        # Make everything pass
        sample_receipt["risk_summary"]["critical"] = 0
        sample_receipt["risk_summary"]["high"] = 0
        sample_receipt["robustness_score"] = 0.9
        report = generator.generate(sample_receipt)
        assert report.overall_status == "conformant"

    def test_overall_status_non_conformant(self, generator):
        receipt = {
            "receipt_id": "bad-001",
            "input_summary": "Credit scoring AI",
            "risk_summary": {"critical": 5, "high": 3, "medium": 0, "low": 0, "total": 8},
            "confidence": 0.2,
            "robustness_score": 0.1,
            "verdict_reasoning": "High bias in credit scoring",
            "provenance_chain": [],
            "config_used": {},
        }
        report = generator.generate(receipt)
        assert report.overall_status == "non_conformant"

    def test_recommendations_for_failing_articles(self, generator):
        receipt = {
            "receipt_id": "rec-002",
            "input_summary": "Simple tool",
            "risk_summary": {},
            "confidence": 0.0,
            "robustness_score": 0.0,
            "verdict_reasoning": "",
            "provenance_chain": [],
            "config_used": {},
        }
        report = generator.generate(receipt)
        assert len(report.recommendations) > 0

    def test_report_integrity_hash(self, generator, sample_receipt):
        report = generator.generate(sample_receipt)
        assert report.integrity_hash
        assert len(report.integrity_hash) == 64  # SHA-256


# ---------------------------------------------------------------------------
# ConformityReport serialization tests
# ---------------------------------------------------------------------------

class TestConformityReport:
    """Tests for ConformityReport serialization."""

    def test_to_dict(self, generator, sample_receipt):
        report = generator.generate(sample_receipt)
        d = report.to_dict()
        assert d["report_id"].startswith("EUAIA-")
        assert d["receipt_id"] == "test-receipt-001"
        assert "risk_classification" in d
        assert "article_mappings" in d
        assert isinstance(d["article_mappings"], list)

    def test_to_json(self, generator, sample_receipt):
        report = generator.generate(sample_receipt)
        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["receipt_id"] == "test-receipt-001"

    def test_to_markdown(self, generator, sample_receipt):
        report = generator.generate(sample_receipt)
        md = report.to_markdown()
        assert "# EU AI Act Conformity Report" in md
        assert "Article 9" in md
        assert "Article 13" in md
        assert "Article 14" in md
        assert "Risk Level:" in md

    def test_markdown_includes_recommendations(self, generator, minimal_receipt):
        report = generator.generate(minimal_receipt)
        md = report.to_markdown()
        if report.recommendations:
            assert "## Recommendations" in md


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers:
    """Tests for helper functions."""

    def test_detect_human_oversight_from_config(self):
        config = {"require_approval": True}
        assert _detect_human_oversight(config, {"provenance_chain": []})

    def test_detect_human_oversight_from_provenance(self):
        receipt = {
            "provenance_chain": [
                {"event_type": "plan_approved", "description": "Approved by admin"},
            ],
        }
        assert _detect_human_oversight({}, receipt)

    def test_no_human_oversight(self):
        assert not _detect_human_oversight({}, {"provenance_chain": []})

    def test_annex_iii_has_8_categories(self):
        assert len(ANNEX_III_CATEGORIES) == 8

    def test_annex_iii_categories_numbered_1_to_8(self):
        numbers = [c["number"] for c in ANNEX_III_CATEGORIES]
        assert numbers == list(range(1, 9))


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

class TestComplianceCLI:
    """Tests for the compliance CLI commands."""

    def test_audit_command_json_output(self, sample_receipt, tmp_path):
        from aragora.cli.commands.compliance import _cmd_audit
        import argparse

        receipt_file = tmp_path / "receipt.json"
        receipt_file.write_text(json.dumps(sample_receipt))

        output_file = tmp_path / "report.json"
        args = argparse.Namespace(
            receipt_file=str(receipt_file),
            output_format="json",
            output=str(output_file),
        )
        _cmd_audit(args)

        report = json.loads(output_file.read_text())
        assert report["receipt_id"] == "test-receipt-001"
        assert "article_mappings" in report

    def test_audit_command_markdown_output(self, sample_receipt, tmp_path):
        from aragora.cli.commands.compliance import _cmd_audit
        import argparse

        receipt_file = tmp_path / "receipt.json"
        receipt_file.write_text(json.dumps(sample_receipt))

        output_file = tmp_path / "report.md"
        args = argparse.Namespace(
            receipt_file=str(receipt_file),
            output_format="markdown",
            output=str(output_file),
        )
        _cmd_audit(args)

        md = output_file.read_text()
        assert "# EU AI Act Conformity Report" in md

    def test_audit_command_missing_file(self, tmp_path):
        from aragora.cli.commands.compliance import _cmd_audit
        import argparse

        args = argparse.Namespace(
            receipt_file=str(tmp_path / "nonexistent.json"),
            output_format="json",
            output=None,
        )
        with pytest.raises(SystemExit):
            _cmd_audit(args)

    def test_classify_command(self, capsys):
        from aragora.cli.commands.compliance import _cmd_classify
        import argparse

        args = argparse.Namespace(description=["AI", "for", "credit", "scoring"])
        _cmd_classify(args)
        captured = capsys.readouterr()
        assert "HIGH" in captured.out
        assert "Annex III" in captured.out

    def test_classify_command_minimal(self, capsys):
        from aragora.cli.commands.compliance import _cmd_classify
        import argparse

        args = argparse.Namespace(description=["simple", "data", "tool"])
        _cmd_classify(args)
        captured = capsys.readouterr()
        assert "MINIMAL" in captured.out
