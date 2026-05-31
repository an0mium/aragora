"""Tests for structured PHI identifier detection in the compliance framework.

Covers the deterministic, validated format detectors integrated into the HIPAA
framework: SSN, NPI (Luhn-checked), ICD-10, MRN, DOB, email, and phone numbers.

These detectors close GitHub issue #7394, where ``aragora compliance check
--frameworks hipaa`` against synthetic PHI only flagged the literal keyword
"ssn"/"diagnosis" and missed the actual structured identifiers in the content.
"""

from __future__ import annotations

import pytest

from aragora.compliance.framework import (
    ComplianceFrameworkManager,
    ComplianceSeverity,
)

# Synthetic PHI string from issue #7394.
SYNTHETIC_PHI = (
    "Patient John Doe, SSN 123-45-6789, DOB 1980-03-15, was treated for diabetes "
    "at Springfield General Hospital. Contact: john.doe@example.com, "
    "phone 555-123-4567"
)


def _rule_ids(result) -> set[str]:
    return {issue.rule_id for issue in result.issues}


class TestStructuredPHIDetection:
    """Structured identifiers are detected as HIPAA findings."""

    @pytest.fixture
    def result(self):
        manager = ComplianceFrameworkManager()
        return manager.check(SYNTHETIC_PHI, frameworks=["hipaa"])

    def test_detects_ssn_value(self, result):
        """The actual SSN value (not just the word 'ssn') is detected."""
        ssn_issues = [i for i in result.issues if i.rule_id == "hipaa-phi-ssn"]
        assert ssn_issues, f"SSN not detected. Rules: {_rule_ids(result)}"
        assert "123-45-6789" in ssn_issues[0].matched_text
        assert ssn_issues[0].severity == ComplianceSeverity.CRITICAL

    def test_detects_date_of_birth(self, result):
        dob_issues = [i for i in result.issues if i.rule_id == "hipaa-phi-dob"]
        assert dob_issues, f"DOB not detected. Rules: {_rule_ids(result)}"
        assert "1980-03-15" in dob_issues[0].matched_text

    def test_detects_email(self, result):
        email_issues = [i for i in result.issues if i.rule_id == "hipaa-phi-email"]
        assert email_issues, f"Email not detected. Rules: {_rule_ids(result)}"
        assert "john.doe@example.com" in email_issues[0].matched_text

    def test_detects_phone(self, result):
        phone_issues = [i for i in result.issues if i.rule_id == "hipaa-phi-phone"]
        assert phone_issues, f"Phone not detected. Rules: {_rule_ids(result)}"
        assert "555-123-4567" in phone_issues[0].matched_text

    def test_result_is_non_compliant_with_multiple_findings(self, result):
        """Repro should now surface several findings, not just one keyword."""
        assert not result.compliant
        assert len(result.issues) >= 4


class TestSSNDetector:
    def test_dashed_ssn(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("SSN: 078-05-1120", frameworks=["hipaa"])
        assert any(i.rule_id == "hipaa-phi-ssn" for i in result.issues)

    def test_rejects_invalid_area_000(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("Code 000-12-3456 here", frameworks=["hipaa"])
        assert not any(i.rule_id == "hipaa-phi-ssn" for i in result.issues)

    def test_rejects_invalid_group_00(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("Value 123-00-6789", frameworks=["hipaa"])
        assert not any(i.rule_id == "hipaa-phi-ssn" for i in result.issues)

    def test_rejects_900_series(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("Number 900-12-3456", frameworks=["hipaa"])
        assert not any(i.rule_id == "hipaa-phi-ssn" for i in result.issues)


class TestNPIDetector:
    """NPI must pass the Luhn check (with the 80840 prefix)."""

    def test_valid_npi(self):
        # 1234567893 is a well-known valid example NPI (Luhn-valid).
        manager = ComplianceFrameworkManager()
        result = manager.check("Provider NPI 1234567893", frameworks=["hipaa"])
        assert any(i.rule_id == "hipaa-phi-npi" for i in result.issues)

    def test_invalid_npi_fails_luhn(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("Provider NPI 1234567890", frameworks=["hipaa"])
        assert not any(i.rule_id == "hipaa-phi-npi" for i in result.issues)

    def test_luhn_passing_number_without_npi_context_is_not_npi(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("Customer account 1234567893 was updated", frameworks=["hipaa"])
        assert not any(i.rule_id == "hipaa-phi-npi" for i in result.issues)


class TestICD10Detector:
    def test_valid_icd10_code(self):
        # E11.9 = Type 2 diabetes mellitus without complications.
        manager = ComplianceFrameworkManager()
        result = manager.check("Diagnosed with E11.9", frameworks=["hipaa"])
        assert any(i.rule_id == "hipaa-phi-icd10" for i in result.issues)

    def test_icd10_without_decimal(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("Code I10 hypertension", frameworks=["hipaa"])
        assert any(i.rule_id == "hipaa-phi-icd10" for i in result.issues)

    def test_rejects_unlabeled_short_technical_tokens(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("Built on 2024-01-15; see B12 in module A10.", frameworks=["hipaa"])
        assert not any(i.rule_id == "hipaa-phi-icd10" for i in result.issues)


class TestMRNDetector:
    def test_labeled_mrn(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("MRN: 00123456", frameworks=["hipaa"])
        assert any(i.rule_id == "hipaa-phi-mrn" for i in result.issues)

    def test_medical_record_number_label(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("Medical Record Number 7654321", frameworks=["hipaa"])
        assert any(i.rule_id == "hipaa-phi-mrn" for i in result.issues)


class TestNoFalsePositivesOnCleanContent:
    def test_clean_text_has_no_phi_findings(self):
        manager = ComplianceFrameworkManager()
        result = manager.check(
            "The deployment pipeline uses TLS 1.2 and rotates keys monthly.",
            frameworks=["hipaa"],
        )
        phi_rule_ids = {
            "hipaa-phi-ssn",
            "hipaa-phi-npi",
            "hipaa-phi-icd10",
            "hipaa-phi-mrn",
            "hipaa-phi-dob",
            "hipaa-phi-email",
            "hipaa-phi-phone",
        }
        assert not (phi_rule_ids & _rule_ids(result))

    def test_clean_operational_dates_are_not_dates_of_birth(self):
        manager = ComplianceFrameworkManager()
        result = manager.check(
            "Deployed on 2026-05-31 and reviewed 01/15/2025.",
            frameworks=["hipaa"],
        )
        assert not any(i.rule_id == "hipaa-phi-dob" for i in result.issues)

    def test_common_parenthesized_phone_without_space_is_detected(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("Call patient at (555)123-4567", frameworks=["hipaa"])
        assert any(i.rule_id == "hipaa-phi-phone" for i in result.issues)
