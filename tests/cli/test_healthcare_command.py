"""Tests for the healthcare CLI command.

Tests the healthcare vertical integration including:
- FHIR bundle to clinical summary conversion
- PHI stripping from receipt metadata
- Healthcare receipt generation
- CLI argument parsing
- Demo mode
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.cli.commands.healthcare import (
    DEMO_CLINICAL_QUESTION,
    DEMO_FHIR_BUNDLE,
    HEALTHCARE_AGENTS,
    HEALTHCARE_PROFILE,
    HEALTHCARE_ROUNDS,
    PHI_RECEIPT_FIELDS,
    _build_healthcare_receipt,
    add_healthcare_parser,
    cmd_healthcare,
    fhir_bundle_to_clinical_summary,
    run_healthcare_review,
    strip_phi_from_metadata,
)


# ---------------------------------------------------------------------------
# FHIR Bundle -> Clinical Summary Tests
# ---------------------------------------------------------------------------


class TestFhirBundleToClinicalSummary:
    """Tests for converting FHIR bundles to clinical text."""

    def test_extracts_patient_demographics(self):
        """Patient demographics are extracted from FHIR Patient resource."""
        bundle = {
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "gender": "female",
                        "birthDate": "1970",
                    }
                }
            ]
        }
        summary = fhir_bundle_to_clinical_summary(bundle)
        assert "Gender: female" in summary
        assert "Birth Year: 1970" in summary

    def test_extracts_conditions(self):
        """Active conditions are extracted with status and onset."""
        bundle = {
            "entry": [
                {
                    "resource": {
                        "resourceType": "Condition",
                        "code": {"text": "Type 2 Diabetes"},
                        "clinicalStatus": {"coding": [{"code": "active"}]},
                        "onsetDateTime": "2020-01-01",
                    }
                }
            ]
        }
        summary = fhir_bundle_to_clinical_summary(bundle)
        assert "Type 2 Diabetes" in summary
        assert "active" in summary
        assert "2020-01-01" in summary

    def test_extracts_medications(self):
        """Medication requests are extracted with dosage."""
        bundle = {
            "entry": [
                {
                    "resource": {
                        "resourceType": "MedicationRequest",
                        "status": "active",
                        "medicationCodeableConcept": {"text": "Metformin 500mg"},
                        "dosageInstruction": [{"text": "twice daily"}],
                    }
                }
            ]
        }
        summary = fhir_bundle_to_clinical_summary(bundle)
        assert "Metformin 500mg" in summary
        assert "twice daily" in summary

    def test_extracts_observations_with_value(self):
        """Observations with valueQuantity are extracted."""
        bundle = {
            "entry": [
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"text": "HbA1c"},
                        "valueQuantity": {"value": 8.2, "unit": "%"},
                        "effectiveDateTime": "2026-01-10",
                    }
                }
            ]
        }
        summary = fhir_bundle_to_clinical_summary(bundle)
        assert "HbA1c" in summary
        assert "8.2" in summary
        assert "%" in summary

    def test_extracts_observations_with_components(self):
        """Observations with components (e.g. blood pressure) are extracted."""
        bundle = {
            "entry": [
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"text": "Blood Pressure"},
                        "component": [
                            {
                                "code": {"text": "Systolic"},
                                "valueQuantity": {"value": 145, "unit": "mmHg"},
                            },
                            {
                                "code": {"text": "Diastolic"},
                                "valueQuantity": {"value": 92, "unit": "mmHg"},
                            },
                        ],
                    }
                }
            ]
        }
        summary = fhir_bundle_to_clinical_summary(bundle)
        assert "Blood Pressure" in summary
        assert "145" in summary
        assert "92" in summary

    def test_extracts_observation_interpretation(self):
        """Observation interpretations are included."""
        bundle = {
            "entry": [
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"text": "HbA1c"},
                        "valueQuantity": {"value": 8.2, "unit": "%"},
                        "interpretation": [
                            {"coding": [{"code": "H", "display": "High"}]}
                        ],
                    }
                }
            ]
        }
        summary = fhir_bundle_to_clinical_summary(bundle)
        assert "[High]" in summary

    def test_demo_bundle_produces_summary(self):
        """The demo FHIR bundle produces a complete summary."""
        summary = fhir_bundle_to_clinical_summary(DEMO_FHIR_BUNDLE)
        assert "PATIENT DEMOGRAPHICS" in summary
        assert "ACTIVE CONDITIONS" in summary
        assert "CURRENT MEDICATIONS" in summary
        assert "RECENT OBSERVATIONS" in summary
        assert "Type 2 Diabetes" in summary
        assert "Metformin" in summary
        assert "HbA1c" in summary

    def test_empty_bundle(self):
        """Empty bundle returns empty string."""
        summary = fhir_bundle_to_clinical_summary({"entry": []})
        assert summary == ""

    def test_unknown_resource_types_are_skipped(self):
        """Unknown resource types are silently skipped."""
        bundle = {
            "entry": [
                {
                    "resource": {
                        "resourceType": "UnknownType",
                        "data": "something",
                    }
                }
            ]
        }
        summary = fhir_bundle_to_clinical_summary(bundle)
        assert summary == ""

    def test_missing_fields_handled_gracefully(self):
        """Missing optional fields don't cause errors."""
        bundle = {
            "entry": [
                {
                    "resource": {
                        "resourceType": "Condition",
                        "code": {},
                    }
                },
                {
                    "resource": {
                        "resourceType": "MedicationRequest",
                        "status": "active",
                    }
                },
            ]
        }
        summary = fhir_bundle_to_clinical_summary(bundle)
        # Should not raise, may have "Unknown" defaults
        assert isinstance(summary, str)


# ---------------------------------------------------------------------------
# PHI Stripping Tests
# ---------------------------------------------------------------------------


class TestStripPhiFromMetadata:
    """Tests for HIPAA Safe Harbor PHI removal."""

    def test_strips_patient_name(self):
        """Patient name is removed from metadata."""
        metadata = {"patient_name": "John Doe", "condition": "Diabetes"}
        result = strip_phi_from_metadata(metadata)
        assert "patient_name" not in result
        assert result["condition"] == "Diabetes"

    def test_strips_all_phi_fields(self):
        """All 18 HIPAA identifier fields are stripped."""
        metadata = {field: f"value_{field}" for field in PHI_RECEIPT_FIELDS}
        metadata["clinical_data"] = "preserved"
        result = strip_phi_from_metadata(metadata)
        for field in PHI_RECEIPT_FIELDS:
            assert field not in result
        assert result["clinical_data"] == "preserved"

    def test_strips_nested_phi(self):
        """PHI in nested dicts is also stripped."""
        metadata = {
            "outer": {
                "patient_name": "Jane Doe",
                "diagnosis": "Hypertension",
            }
        }
        result = strip_phi_from_metadata(metadata)
        assert "patient_name" not in result["outer"]
        assert result["outer"]["diagnosis"] == "Hypertension"

    def test_strips_phi_in_lists(self):
        """PHI in list items (dicts) is stripped."""
        metadata = {
            "records": [
                {"patient_name": "John", "condition": "Diabetes"},
                {"patient_name": "Jane", "condition": "HTN"},
            ]
        }
        result = strip_phi_from_metadata(metadata)
        for record in result["records"]:
            assert "patient_name" not in record
            assert "condition" in record

    def test_case_insensitive_field_matching(self):
        """Field matching is case-insensitive."""
        metadata = {"Patient_Name": "John", "MRN": "12345", "data": "ok"}
        result = strip_phi_from_metadata(metadata)
        assert "Patient_Name" not in result
        assert "MRN" not in result
        assert result["data"] == "ok"

    def test_hyphenated_field_matching(self):
        """Hyphenated field names are normalized."""
        metadata = {"patient-name": "John", "patient-id": "123", "test": "ok"}
        result = strip_phi_from_metadata(metadata)
        assert "patient-name" not in result
        assert "patient-id" not in result
        assert result["test"] == "ok"

    def test_preserves_non_phi_fields(self):
        """Non-PHI clinical data is preserved."""
        metadata = {
            "condition": "Diabetes",
            "hba1c": 8.2,
            "medications": ["Metformin"],
            "confidence": 0.85,
        }
        result = strip_phi_from_metadata(metadata)
        assert result == metadata

    def test_empty_metadata(self):
        """Empty metadata returns empty dict."""
        assert strip_phi_from_metadata({}) == {}


# ---------------------------------------------------------------------------
# Healthcare Receipt Generation Tests
# ---------------------------------------------------------------------------


class TestBuildHealthcareReceipt:
    """Tests for HIPAA-compliant receipt generation."""

    def _make_debate_result(self, consensus=True, confidence=0.85):
        """Create a mock debate result."""
        msg1 = SimpleNamespace(
            agent="anthropic-api",
            role="proposer",
            content="I recommend option A",
            round=1,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        msg2 = SimpleNamespace(
            agent="openai-api",
            role="critic",
            content="I disagree with the risk assessment",
            round=2,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        vote1 = SimpleNamespace(agent="anthropic-api", choice="A")
        return SimpleNamespace(
            consensus_reached=consensus,
            confidence=confidence,
            final_answer="Option A is recommended with caveats",
            messages=[msg1, msg2],
            votes=[vote1],
            debate_id="test-debate-001",
        )

    def test_receipt_has_required_fields(self):
        """Receipt contains all required compliance fields."""
        result = self._make_debate_result()
        receipt = _build_healthcare_receipt(result, "abc123hash")

        assert "receipt_id" in receipt
        assert "debate_id" in receipt
        assert "timestamp" in receipt
        assert "profile" in receipt
        assert receipt["profile"] == HEALTHCARE_PROFILE
        assert "compliance" in receipt
        assert "verdict" in receipt
        assert "audit_trail" in receipt
        assert "integrity" in receipt

    def test_receipt_marks_hipaa_compliant(self):
        """Receipt marks itself as HIPAA-compliant."""
        result = self._make_debate_result()
        receipt = _build_healthcare_receipt(result, "abc123hash")

        assert receipt["compliance"]["hipaa_compliant"] is True
        assert receipt["compliance"]["phi_redacted"] is True
        assert receipt["compliance"]["safe_harbor_method"] is True

    def test_receipt_captures_consensus(self):
        """Receipt captures consensus verdict."""
        result = self._make_debate_result(consensus=True, confidence=0.92)
        receipt = _build_healthcare_receipt(result, "hash1")

        assert receipt["verdict"]["consensus_reached"] is True
        assert receipt["verdict"]["confidence"] == 0.92

    def test_receipt_captures_no_consensus(self):
        """Receipt captures when no consensus is reached."""
        result = self._make_debate_result(consensus=False, confidence=0.45)
        receipt = _build_healthcare_receipt(result, "hash2")

        assert receipt["verdict"]["consensus_reached"] is False
        assert receipt["verdict"]["confidence"] == 0.45

    def test_receipt_has_artifact_hash(self):
        """Receipt contains integrity hash."""
        result = self._make_debate_result()
        receipt = _build_healthcare_receipt(result, "abc123hash")

        assert receipt["integrity"]["artifact_hash"]
        assert len(receipt["integrity"]["artifact_hash"]) == 64  # SHA-256

    def test_receipt_includes_fhir_metadata(self):
        """Receipt includes FHIR bundle metadata when provided."""
        result = self._make_debate_result()
        receipt = _build_healthcare_receipt(result, "hash3", DEMO_FHIR_BUNDLE)

        assert receipt["input"]["has_fhir_data"] is True
        assert receipt["input"]["resource_count"] == len(DEMO_FHIR_BUNDLE["entry"])

    def test_receipt_without_fhir(self):
        """Receipt works without FHIR bundle."""
        result = self._make_debate_result()
        receipt = _build_healthcare_receipt(result, "hash4")

        assert receipt["input"]["has_fhir_data"] is False
        assert receipt["input"]["resource_count"] == 0

    def test_receipt_audit_trail(self):
        """Receipt contains audit trail with agent info."""
        result = self._make_debate_result()
        receipt = _build_healthcare_receipt(result, "hash5")

        audit = receipt["audit_trail"]
        assert audit["agents_consulted"] >= 1
        assert audit["votes_cast"] >= 1
        assert isinstance(audit["agent_summaries"], list)

    def test_receipt_input_hash_preserved(self):
        """Input hash is preserved in the receipt."""
        result = self._make_debate_result()
        input_hash = "a1b2c3d4e5f6"
        receipt = _build_healthcare_receipt(result, input_hash)

        assert receipt["input"]["input_hash"] == input_hash


# ---------------------------------------------------------------------------
# Run Healthcare Review Tests
# ---------------------------------------------------------------------------


class TestRunHealthcareReview:
    """Tests for the async healthcare review runner."""

    @pytest.mark.asyncio
    async def test_review_with_demo_bundle(self):
        """Review with demo bundle produces valid receipt."""
        mock_result = SimpleNamespace(
            consensus_reached=True,
            confidence=0.88,
            final_answer="GLP-1 RA recommended",
            messages=[],
            votes=[],
            debate_id="demo-1",
        )
        with patch(
            "aragora.cli.commands.debate.run_debate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await run_healthcare_review(
                clinical_input=DEMO_CLINICAL_QUESTION,
                fhir_bundle=DEMO_FHIR_BUNDLE,
                verbose=False,
            )

        assert "receipt" in result
        assert "debate_result" in result
        assert result["profile"] == HEALTHCARE_PROFILE
        assert result["receipt"]["compliance"]["hipaa_compliant"] is True

    @pytest.mark.asyncio
    async def test_review_without_fhir(self):
        """Review works without FHIR bundle."""
        mock_result = SimpleNamespace(
            consensus_reached=False,
            confidence=0.5,
            final_answer="Insufficient data",
            messages=[],
            votes=[],
            debate_id="no-fhir-1",
        )
        with patch(
            "aragora.cli.commands.debate.run_debate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await run_healthcare_review(
                clinical_input="Should we prescribe drug X?",
            )

        assert result["receipt"]["input"]["has_fhir_data"] is False

    @pytest.mark.asyncio
    async def test_review_computes_input_hash(self):
        """Review computes SHA-256 hash of input."""
        mock_result = SimpleNamespace(
            consensus_reached=True,
            confidence=0.9,
            final_answer="OK",
            messages=[],
            votes=[],
            debate_id="hash-test-1",
        )
        clinical_input = "test question"
        expected_hash = hashlib.sha256(clinical_input.encode()).hexdigest()

        with patch(
            "aragora.cli.commands.debate.run_debate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await run_healthcare_review(
                clinical_input=clinical_input,
            )

        assert result["input_hash"] == expected_hash

    @pytest.mark.asyncio
    async def test_review_passes_vertical_to_debate(self):
        """Review enables healthcare vertical in debate."""
        mock_result = SimpleNamespace(
            consensus_reached=True,
            confidence=0.9,
            final_answer="OK",
            messages=[],
            votes=[],
            debate_id="vertical-test",
        )
        with patch(
            "aragora.cli.commands.debate.run_debate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_debate:
            await run_healthcare_review(clinical_input="test")

        mock_debate.assert_called_once()
        call_kwargs = mock_debate.call_args[1]
        assert call_kwargs["enable_verticals"] is True
        assert call_kwargs["vertical_id"] == "healthcare"

    @pytest.mark.asyncio
    async def test_review_writes_receipt_to_output_dir(self, tmp_path):
        """Receipt is written to output directory when specified."""
        mock_result = SimpleNamespace(
            consensus_reached=True,
            confidence=0.9,
            final_answer="OK",
            messages=[],
            votes=[],
            debate_id="output-test",
        )
        with patch(
            "aragora.cli.commands.debate.run_debate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await run_healthcare_review(
                clinical_input="test",
                output_dir=str(tmp_path),
            )

        assert "receipt_path" in result
        receipt_file = tmp_path / result["receipt_path"].split("/")[-1]
        assert receipt_file.exists()
        data = json.loads(receipt_file.read_text())
        assert data["compliance"]["hipaa_compliant"] is True


# ---------------------------------------------------------------------------
# CLI Argument Parsing Tests
# ---------------------------------------------------------------------------


class TestHealthcareParser:
    """Tests for CLI argument parsing."""

    def _build_parser(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_healthcare_parser(subparsers)
        return parser

    def test_healthcare_subcommand_registered(self):
        """Healthcare subcommand is registered."""
        parser = self._build_parser()
        args = parser.parse_args(["healthcare"])
        assert args.command == "healthcare"

    def test_review_subcommand(self):
        """healthcare review subcommand parses input."""
        parser = self._build_parser()
        args = parser.parse_args(["healthcare", "review", "Should we prescribe X?"])
        assert args.healthcare_command == "review"
        assert args.input == "Should we prescribe X?"

    def test_review_demo_flag(self):
        """--demo flag is parsed."""
        parser = self._build_parser()
        args = parser.parse_args(["healthcare", "review", "--demo"])
        assert args.demo is True

    def test_review_fhir_flag(self):
        """--fhir flag is parsed with path."""
        parser = self._build_parser()
        args = parser.parse_args(["healthcare", "review", "--fhir", "/tmp/bundle.json", "question"])
        assert args.fhir == "/tmp/bundle.json"
        assert args.input == "question"

    def test_review_agents_override(self):
        """--agents flag overrides default."""
        parser = self._build_parser()
        args = parser.parse_args(["healthcare", "review", "--agents", "grok,gemini", "q"])
        assert args.agents == "grok,gemini"

    def test_review_rounds_override(self):
        """--rounds flag overrides default."""
        parser = self._build_parser()
        args = parser.parse_args(["healthcare", "review", "--rounds", "3", "q"])
        assert args.rounds == 3

    def test_review_json_flag(self):
        """--json flag is parsed."""
        parser = self._build_parser()
        args = parser.parse_args(["healthcare", "review", "--json", "q"])
        assert getattr(args, "json", False) is True

    def test_review_output_dir_flag(self):
        """--output-dir flag is parsed."""
        parser = self._build_parser()
        args = parser.parse_args(["healthcare", "review", "--output-dir", "/tmp/out", "q"])
        assert args.output_dir == "/tmp/out"

    def test_review_verbose_flag(self):
        """--verbose flag is parsed."""
        parser = self._build_parser()
        args = parser.parse_args(["healthcare", "review", "--verbose", "q"])
        assert args.verbose is True

    def test_default_agents(self):
        """Default agents are the healthcare team."""
        parser = self._build_parser()
        args = parser.parse_args(["healthcare", "review", "q"])
        assert args.agents == HEALTHCARE_AGENTS

    def test_default_rounds(self):
        """Default rounds are healthcare-specific."""
        parser = self._build_parser()
        args = parser.parse_args(["healthcare", "review", "q"])
        assert args.rounds == HEALTHCARE_ROUNDS


# ---------------------------------------------------------------------------
# cmd_healthcare dispatch tests
# ---------------------------------------------------------------------------


class TestCmdHealthcareDispatch:
    """Tests for the cmd_healthcare dispatcher."""

    def test_no_subcommand_shows_help(self, capsys):
        """No subcommand shows usage help."""
        args = argparse.Namespace(healthcare_command=None)
        cmd_healthcare(args)
        captured = capsys.readouterr()
        assert "Usage:" in captured.out or "aragora healthcare" in captured.out

    def test_review_subcommand_dispatches(self):
        """review subcommand dispatches to _cmd_healthcare_review."""
        args = argparse.Namespace(
            healthcare_command="review",
            demo=True,
            fhir=None,
            json=False,
            output_dir=None,
            verbose=False,
            agents=HEALTHCARE_AGENTS,
            rounds=HEALTHCARE_ROUNDS,
            input="",
        )
        with patch(
            "aragora.cli.commands.healthcare.run_healthcare_review",
            new_callable=AsyncMock,
        ) as mock_review:
            mock_review.return_value = {
                "receipt": {
                    "receipt_id": "test-id",
                    "profile": HEALTHCARE_PROFILE,
                    "timestamp": "2026-01-01T00:00:00Z",
                    "compliance": {
                        "hipaa_compliant": True,
                        "phi_redacted": True,
                        "safe_harbor_method": True,
                    },
                    "input": {"input_hash": "abc", "has_fhir_data": True, "resource_count": 7},
                    "verdict": {
                        "consensus_reached": True,
                        "confidence": 0.9,
                        "final_answer": "Recommendation",
                    },
                    "audit_trail": {
                        "agents_consulted": 3,
                        "rounds_completed": 5,
                        "votes_cast": 3,
                        "dissenting_views_count": 0,
                        "agent_summaries": [],
                    },
                    "integrity": {"artifact_hash": "a" * 64},
                },
                "input_hash": "abc",
                "profile": HEALTHCARE_PROFILE,
                "debate_result": None,
            }
            cmd_healthcare(args)
            mock_review.assert_called_once()


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------


class TestHealthcareConstants:
    """Tests for healthcare module constants."""

    def test_demo_fhir_bundle_is_valid(self):
        """Demo FHIR bundle has valid structure."""
        assert DEMO_FHIR_BUNDLE["resourceType"] == "Bundle"
        assert DEMO_FHIR_BUNDLE["type"] == "collection"
        assert len(DEMO_FHIR_BUNDLE["entry"]) >= 5

    def test_demo_bundle_has_patient(self):
        """Demo bundle contains a Patient resource."""
        types = [e["resource"]["resourceType"] for e in DEMO_FHIR_BUNDLE["entry"]]
        assert "Patient" in types

    def test_demo_bundle_has_conditions(self):
        """Demo bundle contains Condition resources."""
        types = [e["resource"]["resourceType"] for e in DEMO_FHIR_BUNDLE["entry"]]
        assert "Condition" in types

    def test_demo_bundle_has_medications(self):
        """Demo bundle contains MedicationRequest resources."""
        types = [e["resource"]["resourceType"] for e in DEMO_FHIR_BUNDLE["entry"]]
        assert "MedicationRequest" in types

    def test_demo_bundle_has_observations(self):
        """Demo bundle contains Observation resources."""
        types = [e["resource"]["resourceType"] for e in DEMO_FHIR_BUNDLE["entry"]]
        assert "Observation" in types

    def test_demo_question_references_clinical_terms(self):
        """Demo question mentions clinical terms for realistic testing."""
        assert "HbA1c" in DEMO_CLINICAL_QUESTION
        assert "Metformin" in DEMO_CLINICAL_QUESTION

    def test_phi_receipt_fields_cover_hipaa_identifiers(self):
        """PHI fields cover key HIPAA Safe Harbor identifiers."""
        assert "patient_name" in PHI_RECEIPT_FIELDS
        assert "ssn" in PHI_RECEIPT_FIELDS
        assert "mrn" in PHI_RECEIPT_FIELDS
        assert "email" in PHI_RECEIPT_FIELDS
        assert "phone" in PHI_RECEIPT_FIELDS
        assert "address" in PHI_RECEIPT_FIELDS
        assert "date_of_birth" in PHI_RECEIPT_FIELDS

    def test_healthcare_profile_name(self):
        """Healthcare profile constant is correct."""
        assert HEALTHCARE_PROFILE == "healthcare_hipaa"

    def test_healthcare_rounds_reasonable(self):
        """Healthcare rounds are a reasonable number."""
        assert 3 <= HEALTHCARE_ROUNDS <= 10
