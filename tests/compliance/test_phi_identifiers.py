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

    def test_labeled_compact_ssn(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("SSN: 078051120", frameworks=["hipaa"])
        assert any(i.rule_id == "hipaa-phi-ssn" for i in result.issues)

    def test_unlabeled_compact_ssn_is_not_detected(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("Ticket 078051120 was processed", frameworks=["hipaa"])
        assert not any(i.rule_id == "hipaa-phi-ssn" for i in result.issues)

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

    def test_trailing_npi_label_is_detected(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("Provider 1234567893 (NPI)", frameworks=["hipaa"])
        assert any(i.rule_id == "hipaa-phi-npi" for i in result.issues)


class TestICD10Detector:
    def test_valid_icd10_code(self):
        # E11.9 = Type 2 diabetes mellitus without complications.
        manager = ComplianceFrameworkManager()
        result = manager.check("Diagnosed with E11.9", frameworks=["hipaa"])
        assert any(i.rule_id == "hipaa-phi-icd10" for i in result.issues)

    def test_icd10_without_decimal(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("Diagnosis I10 hypertension", frameworks=["hipaa"])
        assert any(i.rule_id == "hipaa-phi-icd10" for i in result.issues)

    def test_rejects_unlabeled_short_technical_tokens(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("Built on 2024-01-15; see B12 in module A10.", frameworks=["hipaa"])
        assert not any(i.rule_id == "hipaa-phi-icd10" for i in result.issues)

    def test_rejects_unlabeled_decimal_technical_tokens(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("module A10.2 and version B12.3 shipped", frameworks=["hipaa"])
        assert not any(i.rule_id == "hipaa-phi-icd10" for i in result.issues)

    def test_rejects_technical_status_code_context(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("status code I10 returned by module R51", frameworks=["hipaa"])
        assert not any(i.rule_id == "hipaa-phi-icd10" for i in result.issues)

    def test_u_prefixed_icd10_code_with_context(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("Diagnosis U07.1 confirmed", frameworks=["hipaa"])
        assert any(i.rule_id == "hipaa-phi-icd10" for i in result.issues)

    def test_trailing_diagnosis_context_detects_icd10(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("U07.1 was the diagnosis", frameworks=["hipaa"])
        assert any(i.rule_id == "hipaa-phi-icd10" for i in result.issues)


class TestMRNDetector:
    def test_labeled_mrn(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("MRN: 00123456", frameworks=["hipaa"])
        assert any(i.rule_id == "hipaa-phi-mrn" for i in result.issues)

    def test_medical_record_number_label(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("Medical Record Number 7654321", frameworks=["hipaa"])
        assert any(i.rule_id == "hipaa-phi-mrn" for i in result.issues)

    def test_mrn_matched_text_is_identifier_value(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("MRN: ABCD-12345", frameworks=["hipaa"])
        mrn_issues = [i for i in result.issues if i.rule_id == "hipaa-phi-mrn"]
        assert mrn_issues[0].matched_text == "ABCD-12345"

    def test_medical_record_hash_label(self):
        """Regression: "Medical Record # 12345" was missed because the trailing
        ``\\b`` after the label could not match between ``#`` and whitespace."""
        manager = ComplianceFrameworkManager()
        result = manager.check("Medical Record # 12345", frameworks=["hipaa"])
        assert any(i.rule_id == "hipaa-phi-mrn" for i in result.issues)

    def test_medical_record_no_dot_label(self):
        """Regression: "Medical Record no. 12345" was missed because the trailing
        ``\\b`` after the label could not match between ``.`` and whitespace."""
        manager = ComplianceFrameworkManager()
        result = manager.check("Medical Record no. 12345", frameworks=["hipaa"])
        assert any(i.rule_id == "hipaa-phi-mrn" for i in result.issues)

    @pytest.mark.parametrize(
        "content",
        [
            "MRN status PENDING for the patient",
            "medical record number REDACTED in the chart",
            "Medical Record no. attached to file",
            "MRN: UNKNOWN",
            "medical record # PENDING",
            "medical record number SUMMARY",
        ],
    )
    def test_label_followed_by_plain_word_is_not_mrn(self, content):
        """Regression: requiring a digit in the captured identifier prevents the
        detector from flagging an ordinary word that follows the label text.

        The cycle-2 change that widened the trailing label boundary to support
        ``#``/``no.`` variants also let the ``5..12``-char token greedily capture
        any following word as an "MRN". A genuine MRN is numeric/alphanumeric, so
        the identifier group now demands at least one digit."""
        manager = ComplianceFrameworkManager()
        result = manager.check(content, frameworks=["hipaa"])
        assert not any(i.rule_id == "hipaa-phi-mrn" for i in result.issues)


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

    def test_invalid_calendar_dates_are_not_dates_of_birth(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("Patient DOB 1980-02-31", frameworks=["hipaa"])
        assert not any(i.rule_id == "hipaa-phi-dob" for i in result.issues)

    def test_trailing_birth_context_detects_dob(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("1980-03-15 is the patient's date of birth", frameworks=["hipaa"])
        assert any(i.rule_id == "hipaa-phi-dob" for i in result.issues)

    def test_common_parenthesized_phone_without_space_is_detected(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("Call patient at (555)123-4567", frameworks=["hipaa"])
        assert any(i.rule_id == "hipaa-phi-phone" for i in result.issues)

    def test_plus_one_phone_is_detected(self):
        manager = ComplianceFrameworkManager()
        result = manager.check("Call patient at +1 555-123-4567", frameworks=["hipaa"])
        assert any(i.rule_id == "hipaa-phi-phone" for i in result.issues)


class TestEmailDetectorReDoS:
    """Regression tests for the email detector ReDoS (CWE-1333).

    The original pattern ``[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}``
    had an unbounded middle class that overlapped the literal ``.`` separator,
    causing catastrophic (quadratic) backtracking. Crafted input such as
    ``x@`` + ``a.`` * N or ``x`` + ``.x`` * N pinned a CPU for many seconds at
    only tens of KB -- an authenticated DoS reachable via
    ``POST /api/v1/compliance/check`` and ``compliance check --file``.

    The hardened pattern uses non-overlapping, RFC-bounded labels so matching
    is linear; these tests assert the adversarial inputs complete near-instantly
    while normal emails are still detected and benign strings are not.
    """

    # Comfortably above linear cost, far below the multi-second pathological
    # blowup of the original pattern (which took >5s at 64 KB).
    _BUDGET_SECONDS = 1.0

    @pytest.mark.parametrize(
        "make_payload",
        [
            pytest.param(lambda n: "x@" + "a." * n + "!", id="domain-label-dots"),
            pytest.param(lambda n: "x" + ".x" * n + "@!", id="local-part-dots"),
            pytest.param(lambda n: ("a." * n) + "@" + ("b." * n) + "!", id="both-sides-dots"),
        ],
    )
    def test_adversarial_email_input_is_linear(self, make_payload):
        import time

        from aragora.compliance.phi_detectors import detect_email

        # ~64 KB crafted string: instant with the linear pattern, multi-second
        # (and super-linear) with the original backtracking pattern.
        payload = make_payload(16_000)
        assert len(payload) >= 32_000  # ensure the input is genuinely large

        start = time.perf_counter()
        detect_email(payload)
        elapsed = time.perf_counter() - start

        assert elapsed < self._BUDGET_SECONDS, (
            f"detect_email took {elapsed:.3f}s on a {len(payload)}-char adversarial "
            f"input; expected < {self._BUDGET_SECONDS}s (ReDoS regression)"
        )

    def test_normal_emails_still_detected(self):
        from aragora.compliance.phi_detectors import detect_email

        cases = {
            "Contact jane.doe@example.com please": "jane.doe@example.com",
            "user+tag@sub.domain.co.uk": "user+tag@sub.domain.co.uk",
            "a_b%c@host-name.io": "a_b%c@host-name.io",
            "x@y.z.example.org": "x@y.z.example.org",
            "first.middle.last@dept.company.co": "first.middle.last@dept.company.co",
        }
        for content, expected in cases.items():
            found = [m.text for m in detect_email(content)]
            assert expected in found, f"{expected!r} not detected in {content!r}: {found}"

    def test_email_match_offsets_are_correct(self):
        from aragora.compliance.phi_detectors import detect_email

        content = "hello jane.doe@example.com and user+tag@sub.domain.co.uk!"
        for match in detect_email(content):
            assert content[match.start : match.start + len(match.text)] == match.text

    def test_benign_near_email_strings_not_matched_pathologically(self):
        from aragora.compliance.phi_detectors import detect_email

        for content in ["x@", "a@b.", "foo@bar..com", "@nope.com", "user@localhost"]:
            assert detect_email(content) == [], f"unexpected email match in {content!r}"

    def test_adversarial_input_via_compliance_check_is_linear(self):
        import time

        # Exercise the full reachable path: ComplianceFrameworkManager.check is
        # what POST /api/v1/compliance/check and `compliance check --file` call.
        payload = "x@" + "a." * 16_000 + "!"
        manager = ComplianceFrameworkManager()

        start = time.perf_counter()
        manager.check(payload, frameworks=["hipaa"])
        elapsed = time.perf_counter() - start

        assert elapsed < self._BUDGET_SECONDS, (
            f"compliance check took {elapsed:.3f}s on adversarial email input; "
            f"expected < {self._BUDGET_SECONDS}s (ReDoS regression)"
        )
