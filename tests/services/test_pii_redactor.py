"""
Tests for PII Redaction Service.

Tests cover:
- PIIType enum values
- PIIMatch dataclass
- RedactionResult dataclass
- PIIRedactor initialization
- PII pattern detection (email, phone, SSN, credit card, IP, address, etc.)
- Domain preservation for emails
- Email message redaction
- Dictionary field redaction
- Convenience functions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest


# =============================================================================
# PIIType Tests
# =============================================================================


class TestPIIType:
    """Tests for PIIType enum."""

    def test_pii_type_values(self):
        """Test all PII types have expected values."""
        from aragora.services.pii_redactor import PIIType

        assert PIIType.EMAIL.value == "email"
        assert PIIType.PHONE.value == "phone"
        assert PIIType.SSN.value == "ssn"
        assert PIIType.CREDIT_CARD.value == "credit_card"
        assert PIIType.IP_ADDRESS.value == "ip_address"
        assert PIIType.ADDRESS.value == "address"
        assert PIIType.DATE_OF_BIRTH.value == "dob"
        assert PIIType.PASSPORT.value == "passport"
        assert PIIType.DRIVERS_LICENSE.value == "drivers_license"
        assert PIIType.BANK_ACCOUNT.value == "bank_account"
        assert PIIType.MEDICAL_ID.value == "medical_id"

    def test_pii_type_is_string_enum(self):
        """Test PIIType is a string enum."""
        from aragora.services.pii_redactor import PIIType

        assert isinstance(PIIType.EMAIL, str)


# =============================================================================
# PIIMatch Tests
# =============================================================================


class TestPIIMatch:
    """Tests for PIIMatch dataclass."""

    def test_pii_match_creation(self):
        """Test PIIMatch creation."""
        from aragora.services.pii_redactor import PIIMatch, PIIType

        match = PIIMatch(
            pii_type=PIIType.EMAIL,
            original="test@example.com",
            redacted="[EMAIL_REDACTED]",
            start=10,
            end=26,
            confidence=1.0,
        )

        assert match.pii_type == PIIType.EMAIL
        assert match.original == "test@example.com"
        assert match.redacted == "[EMAIL_REDACTED]"
        assert match.start == 10
        assert match.end == 26
        assert match.confidence == 1.0

    def test_pii_match_default_confidence(self):
        """Test PIIMatch default confidence."""
        from aragora.services.pii_redactor import PIIMatch, PIIType

        match = PIIMatch(
            pii_type=PIIType.SSN,
            original="123-45-6789",
            redacted="[SSN_REDACTED]",
            start=0,
            end=11,
        )

        assert match.confidence == 1.0


# =============================================================================
# RedactionResult Tests
# =============================================================================


class TestRedactionResult:
    """Tests for RedactionResult dataclass."""

    def test_redaction_result_empty(self):
        """Test RedactionResult with no matches."""
        from aragora.services.pii_redactor import RedactionResult

        result = RedactionResult(
            original_text="Hello world",
            redacted_text="Hello world",
        )

        assert result.has_pii is False
        assert result.match_count == 0
        assert result.pii_types_found == []

    def test_redaction_result_with_matches(self):
        """Test RedactionResult with matches."""
        from aragora.services.pii_redactor import PIIMatch, PIIType, RedactionResult

        match = PIIMatch(
            pii_type=PIIType.EMAIL,
            original="test@test.com",
            redacted="[EMAIL_REDACTED]",
            start=0,
            end=13,
        )

        result = RedactionResult(
            original_text="test@test.com",
            redacted_text="[EMAIL_REDACTED]",
            matches=[match],
            pii_types_found=[PIIType.EMAIL],
        )

        assert result.has_pii is True
        assert result.match_count == 1
        assert PIIType.EMAIL in result.pii_types_found

    def test_redaction_result_to_dict(self):
        """Test RedactionResult.to_dict()."""
        from aragora.services.pii_redactor import PIIMatch, PIIType, RedactionResult

        match = PIIMatch(
            pii_type=PIIType.SSN,
            original="123-45-6789",
            redacted="[SSN_REDACTED]",
            start=0,
            end=11,
            confidence=0.95,
        )

        result = RedactionResult(
            original_text="SSN: 123-45-6789",
            redacted_text="SSN: [SSN_REDACTED]",
            matches=[match],
            pii_types_found=[PIIType.SSN],
        )

        d = result.to_dict()
        assert d["has_pii"] is True
        assert d["match_count"] == 1
        assert "ssn" in d["pii_types"]
        assert d["matches"][0]["type"] == "ssn"
        assert d["matches"][0]["confidence"] == 0.95


# =============================================================================
# PIIRedactor Initialization Tests
# =============================================================================


class TestPIIRedactorInit:
    """Tests for PIIRedactor initialization."""

    def test_default_initialization(self):
        """Test default PIIRedactor initialization."""
        from aragora.services.pii_redactor import PIIRedactor, PIIType

        redactor = PIIRedactor()

        assert len(redactor.enabled_types) == len(PIIType)
        assert redactor.preserve_domains == []
        assert redactor.log_redactions is True

    def test_custom_enabled_types(self):
        """Test PIIRedactor with custom enabled types."""
        from aragora.services.pii_redactor import PIIRedactor, PIIType

        redactor = PIIRedactor(enabled_types=[PIIType.EMAIL, PIIType.PHONE])

        assert len(redactor.enabled_types) == 2
        assert PIIType.EMAIL in redactor.enabled_types
        assert PIIType.PHONE in redactor.enabled_types
        assert PIIType.SSN not in redactor.enabled_types

    def test_preserve_domains(self):
        """Test PIIRedactor with preserve domains."""
        from aragora.services.pii_redactor import PIIRedactor

        redactor = PIIRedactor(preserve_domains=["company.com", "INTERNAL.org"])

        assert "company.com" in redactor.preserve_domains
        assert "internal.org" in redactor.preserve_domains  # Lowercased

    def test_disable_logging(self):
        """Test PIIRedactor with logging disabled."""
        from aragora.services.pii_redactor import PIIRedactor

        redactor = PIIRedactor(log_redactions=False)

        assert redactor.log_redactions is False


# =============================================================================
# Email Detection Tests
# =============================================================================


class TestEmailDetection:
    """Tests for email address detection."""

    @pytest.fixture
    def redactor(self):
        """Create PIIRedactor for email tests."""
        from aragora.services.pii_redactor import PIIRedactor, PIIType

        return PIIRedactor(enabled_types=[PIIType.EMAIL], log_redactions=False)

    def test_detect_simple_email(self, redactor):
        """Test detection of simple email."""
        result = redactor.redact("Contact: john.doe@example.com")

        assert result.has_pii is True
        assert "[EMAIL_REDACTED]" in result.redacted_text
        assert "john.doe@example.com" not in result.redacted_text

    def test_detect_multiple_emails(self, redactor):
        """Test detection of multiple emails."""
        text = "From: alice@test.com To: bob@test.com"
        result = redactor.redact(text)

        assert result.match_count == 2
        assert result.redacted_text.count("[EMAIL_REDACTED]") == 2

    def test_detect_email_with_subdomain(self, redactor):
        """Test detection of email with subdomain."""
        result = redactor.redact("Email: admin@mail.example.co.uk")

        assert result.has_pii is True
        assert "[EMAIL_REDACTED]" in result.redacted_text

    def test_preserve_domain_emails(self):
        """Test that emails with preserved domains are not redacted."""
        from aragora.services.pii_redactor import PIIRedactor, PIIType

        redactor = PIIRedactor(
            enabled_types=[PIIType.EMAIL],
            preserve_domains=["company.com"],
            log_redactions=False,
        )

        result = redactor.redact("Contact: internal@company.com external@other.com")

        # company.com email should be preserved
        assert "internal@company.com" in result.redacted_text
        # other.com email should be redacted
        assert "[EMAIL_REDACTED]" in result.redacted_text
        assert "external@other.com" not in result.redacted_text


# =============================================================================
# Phone Detection Tests
# =============================================================================


class TestPhoneDetection:
    """Tests for phone number detection."""

    @pytest.fixture
    def redactor(self):
        """Create PIIRedactor for phone tests."""
        from aragora.services.pii_redactor import PIIRedactor, PIIType

        return PIIRedactor(enabled_types=[PIIType.PHONE], log_redactions=False)

    def test_detect_us_phone_parentheses(self, redactor):
        """Test detection of US phone with parentheses."""
        result = redactor.redact("Call me: (555) 123-4567")

        assert result.has_pii is True
        assert "[PHONE_REDACTED]" in result.redacted_text

    def test_detect_us_phone_dashes(self, redactor):
        """Test detection of US phone with dashes."""
        result = redactor.redact("Phone: 555-123-4567")

        assert result.has_pii is True
        assert "[PHONE_REDACTED]" in result.redacted_text

    def test_detect_us_phone_dots(self, redactor):
        """Test detection of US phone with dots."""
        result = redactor.redact("Phone: 555.123.4567")

        assert result.has_pii is True
        assert "[PHONE_REDACTED]" in result.redacted_text

    def test_detect_international_phone(self, redactor):
        """Test detection of international phone."""
        result = redactor.redact("International: +1-555-123-4567")

        assert result.has_pii is True
        assert "[PHONE_REDACTED]" in result.redacted_text


# =============================================================================
# SSN Detection Tests
# =============================================================================


class TestSSNDetection:
    """Tests for SSN detection."""

    @pytest.fixture
    def redactor(self):
        """Create PIIRedactor for SSN tests."""
        from aragora.services.pii_redactor import PIIRedactor, PIIType

        return PIIRedactor(enabled_types=[PIIType.SSN], log_redactions=False)

    def test_detect_ssn_with_dashes(self, redactor):
        """Test detection of SSN with dashes."""
        result = redactor.redact("SSN: 123-45-6789")

        assert result.has_pii is True
        assert "[SSN_REDACTED]" in result.redacted_text
        assert "123-45-6789" not in result.redacted_text

    def test_detect_ssn_without_dashes(self, redactor):
        """Test detection of SSN without dashes (lower confidence)."""
        result = redactor.redact("SSN number: 123456789")

        assert result.has_pii is True
        assert "[SSN_REDACTED]" in result.redacted_text
        # Check confidence is lower for no-dash format
        assert any(m.confidence < 1.0 for m in result.matches)


# =============================================================================
# Credit Card Detection Tests
# =============================================================================


class TestCreditCardDetection:
    """Tests for credit card detection."""

    @pytest.fixture
    def redactor(self):
        """Create PIIRedactor for credit card tests."""
        from aragora.services.pii_redactor import PIIRedactor, PIIType

        return PIIRedactor(enabled_types=[PIIType.CREDIT_CARD], log_redactions=False)

    def test_detect_card_with_spaces(self, redactor):
        """Test detection of card number with spaces."""
        result = redactor.redact("Card: 4111 1111 1111 1111")

        assert result.has_pii is True
        assert "[CARD_REDACTED]" in result.redacted_text

    def test_detect_card_with_dashes(self, redactor):
        """Test detection of card number with dashes."""
        result = redactor.redact("Card: 4111-1111-1111-1111")

        assert result.has_pii is True
        assert "[CARD_REDACTED]" in result.redacted_text

    def test_detect_visa_card(self, redactor):
        """Test detection of Visa card (starts with 4)."""
        result = redactor.redact("Visa: 4242424242424242")

        assert result.has_pii is True
        assert "[CARD_REDACTED]" in result.redacted_text

    def test_detect_mastercard(self, redactor):
        """Test detection of Mastercard (starts with 51-55)."""
        result = redactor.redact("MC: 5555555555554444")

        assert result.has_pii is True
        assert "[CARD_REDACTED]" in result.redacted_text


# =============================================================================
# IP Address Detection Tests
# =============================================================================


class TestIPAddressDetection:
    """Tests for IP address detection."""

    @pytest.fixture
    def redactor(self):
        """Create PIIRedactor for IP tests."""
        from aragora.services.pii_redactor import PIIRedactor, PIIType

        return PIIRedactor(enabled_types=[PIIType.IP_ADDRESS], log_redactions=False)

    def test_detect_ipv4(self, redactor):
        """Test detection of IPv4 address."""
        result = redactor.redact("Server IP: 192.168.1.100")

        assert result.has_pii is True
        assert "[IP_REDACTED]" in result.redacted_text
        assert "192.168.1.100" not in result.redacted_text

    def test_detect_localhost(self, redactor):
        """Test detection of localhost IP."""
        result = redactor.redact("Connect to 127.0.0.1")

        assert result.has_pii is True
        assert "[IP_REDACTED]" in result.redacted_text

    def test_detect_public_ip(self, redactor):
        """Test detection of public IP."""
        result = redactor.redact("External: 8.8.8.8")

        assert result.has_pii is True
        assert "[IP_REDACTED]" in result.redacted_text


# =============================================================================
# Address Detection Tests
# =============================================================================


class TestAddressDetection:
    """Tests for physical address detection."""

    @pytest.fixture
    def redactor(self):
        """Create PIIRedactor for address tests."""
        from aragora.services.pii_redactor import PIIRedactor, PIIType

        return PIIRedactor(enabled_types=[PIIType.ADDRESS], log_redactions=False)

    def test_detect_street_address(self, redactor):
        """Test detection of street address."""
        result = redactor.redact("Address: 123 Main Street")

        assert result.has_pii is True
        assert "[ADDRESS_REDACTED]" in result.redacted_text

    def test_detect_avenue_address(self, redactor):
        """Test detection of avenue address."""
        result = redactor.redact("Located at 456 Park Avenue")

        assert result.has_pii is True
        assert "[ADDRESS_REDACTED]" in result.redacted_text

    def test_detect_abbreviated_street(self, redactor):
        """Test detection of abbreviated street names."""
        result = redactor.redact("Office: 789 Oak Dr")

        assert result.has_pii is True
        assert "[ADDRESS_REDACTED]" in result.redacted_text


# =============================================================================
# Date of Birth Detection Tests
# =============================================================================


class TestDOBDetection:
    """Tests for date of birth detection."""

    @pytest.fixture
    def redactor(self):
        """Create PIIRedactor for DOB tests."""
        from aragora.services.pii_redactor import PIIRedactor, PIIType

        return PIIRedactor(enabled_types=[PIIType.DATE_OF_BIRTH], log_redactions=False)

    def test_detect_dob_with_keyword(self, redactor):
        """Test detection of DOB with keyword."""
        result = redactor.redact("DOB: 01/15/1990")

        assert result.has_pii is True
        assert "[DOB_REDACTED]" in result.redacted_text

    def test_detect_date_of_birth_full(self, redactor):
        """Test detection with full 'date of birth' phrase."""
        result = redactor.redact("Date of Birth: 03-25-1985")

        assert result.has_pii is True
        assert "[DOB_REDACTED]" in result.redacted_text

    def test_detect_birthday_keyword(self, redactor):
        """Test detection with birthday keyword."""
        result = redactor.redact("birthday: 12/31/2000")

        assert result.has_pii is True
        assert "[DOB_REDACTED]" in result.redacted_text


# =============================================================================
# Bank Account Detection Tests
# =============================================================================


class TestBankAccountDetection:
    """Tests for bank account number detection."""

    @pytest.fixture
    def redactor(self):
        """Create PIIRedactor for bank account tests."""
        from aragora.services.pii_redactor import PIIRedactor, PIIType

        return PIIRedactor(enabled_types=[PIIType.BANK_ACCOUNT], log_redactions=False)

    def test_detect_account_number(self, redactor):
        """Test detection of account number with keyword."""
        result = redactor.redact("Account: 12345678901")

        assert result.has_pii is True
        assert "[ACCOUNT_REDACTED]" in result.redacted_text

    def test_detect_routing_number(self, redactor):
        """Test detection of routing number with keyword."""
        result = redactor.redact("Routing#: 123456789")

        assert result.has_pii is True
        assert "[ACCOUNT_REDACTED]" in result.redacted_text


# =============================================================================
# Medical ID Detection Tests
# =============================================================================


class TestMedicalIDDetection:
    """Tests for medical ID detection."""

    @pytest.fixture
    def redactor(self):
        """Create PIIRedactor for medical ID tests."""
        from aragora.services.pii_redactor import PIIRedactor, PIIType

        return PIIRedactor(enabled_types=[PIIType.MEDICAL_ID], log_redactions=False)

    def test_detect_mrn(self, redactor):
        """Test detection of MRN."""
        result = redactor.redact("MRN: ABC123456")

        assert result.has_pii is True
        assert "[MEDICAL_ID_REDACTED]" in result.redacted_text

    def test_detect_patient_id(self, redactor):
        """Test detection of patient ID."""
        result = redactor.redact("Patient ID: P12345678")

        assert result.has_pii is True
        assert "[MEDICAL_ID_REDACTED]" in result.redacted_text


# =============================================================================
# Passport Detection Tests
# =============================================================================


class TestPassportDetection:
    """Tests for passport number detection."""

    @pytest.fixture
    def redactor(self):
        """Create PIIRedactor for passport tests."""
        from aragora.services.pii_redactor import PIIRedactor, PIIType

        return PIIRedactor(enabled_types=[PIIType.PASSPORT], log_redactions=False)

    def test_detect_passport(self, redactor):
        """Test detection of passport number."""
        result = redactor.redact("Passport: AB123456")

        assert result.has_pii is True
        assert "[PASSPORT_REDACTED]" in result.redacted_text


# =============================================================================
# Driver's License Detection Tests
# =============================================================================


class TestDriversLicenseDetection:
    """Tests for driver's license detection."""

    @pytest.fixture
    def redactor(self):
        """Create PIIRedactor for DL tests."""
        from aragora.services.pii_redactor import PIIRedactor, PIIType

        return PIIRedactor(enabled_types=[PIIType.DRIVERS_LICENSE], log_redactions=False)

    def test_detect_drivers_license(self, redactor):
        """Test detection of driver's license."""
        result = redactor.redact("Driver's License: D12345678")

        assert result.has_pii is True
        assert "[LICENSE_REDACTED]" in result.redacted_text

    def test_detect_dl_abbreviation(self, redactor):
        """Test detection with DL abbreviation."""
        result = redactor.redact("DL#: ABC12345")

        assert result.has_pii is True
        assert "[LICENSE_REDACTED]" in result.redacted_text


# =============================================================================
# redact_email Method Tests
# =============================================================================


class TestRedactEmail:
    """Tests for redact_email method."""

    @dataclass
    class MockEmailMessage:
        """Mock email message for testing."""

        subject: Optional[str] = None
        body_text: Optional[str] = None
        from_address: Optional[str] = None

    @pytest.fixture
    def redactor(self):
        """Create PIIRedactor for email message tests."""
        from aragora.services.pii_redactor import PIIRedactor

        return PIIRedactor(log_redactions=False)

    def test_redact_email_subject(self, redactor):
        """Test redacting PII from email subject."""
        email = self.MockEmailMessage(
            subject="Meeting with john@example.com",
            body_text="No PII here",
        )

        modified, results = redactor.redact_email(email)

        assert "[EMAIL_REDACTED]" in modified.subject
        assert results["subject"].has_pii is True

    def test_redact_email_body(self, redactor):
        """Test redacting PII from email body."""
        email = self.MockEmailMessage(
            subject="Hello",
            body_text="Call me at (555) 123-4567",
        )

        modified, results = redactor.redact_email(email)

        assert "[PHONE_REDACTED]" in modified.body_text
        assert results["body"].has_pii is True

    def test_redact_email_sender(self, redactor):
        """Test redacting sender email."""
        email = self.MockEmailMessage(
            subject="Test",
            body_text="Test",
            from_address="sender@example.com",
        )

        modified, results = redactor.redact_email(email, redact_sender=True)

        assert "[EMAIL_REDACTED]" in modified.from_address
        assert results["sender"].has_pii is True

    def test_redact_email_skip_subject(self, redactor):
        """Test skipping subject redaction."""
        email = self.MockEmailMessage(
            subject="Email: test@test.com",
            body_text="Phone: 555-123-4567",
        )

        modified, results = redactor.redact_email(email, redact_subject=False)

        assert "test@test.com" in modified.subject
        assert "[PHONE_REDACTED]" in modified.body_text


# =============================================================================
# redact_dict Method Tests
# =============================================================================


class TestRedactDict:
    """Tests for redact_dict method."""

    @pytest.fixture
    def redactor(self):
        """Create PIIRedactor for dict tests."""
        from aragora.services.pii_redactor import PIIRedactor

        return PIIRedactor(log_redactions=False)

    def test_redact_dict_all_fields(self, redactor):
        """Test redacting all string fields."""
        data = {
            "name": "John",
            "email": "john@example.com",
            "phone": "(555) 123-4567",
            "count": 42,  # Not a string
        }

        modified, results = redactor.redact_dict(data)

        assert "[EMAIL_REDACTED]" in modified["email"]
        assert "[PHONE_REDACTED]" in modified["phone"]
        assert modified["count"] == 42  # Unchanged

    def test_redact_dict_specific_fields(self, redactor):
        """Test redacting specific fields only."""
        data = {
            "email": "test@test.com",
            "notes": "Contact: other@test.com",
        }

        modified, results = redactor.redact_dict(data, fields_to_redact=["email"])

        assert "[EMAIL_REDACTED]" in modified["email"]
        # notes field should not be redacted
        assert "other@test.com" in modified["notes"]

    def test_redact_dict_no_pii(self, redactor):
        """Test dict with no PII."""
        data = {"message": "Hello world"}

        modified, results = redactor.redact_dict(data)

        assert modified["message"] == "Hello world"
        assert results["message"].has_pii is False


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_pii_redactor(self):
        """Test get_pii_redactor returns singleton."""
        from aragora.services.pii_redactor import get_pii_redactor

        redactor1 = get_pii_redactor()
        redactor2 = get_pii_redactor()

        assert redactor1 is redactor2

    def test_redact_text(self):
        """Test redact_text convenience function."""
        from aragora.services.pii_redactor import redact_text

        result = redact_text("Email: test@example.com")

        assert "[EMAIL_REDACTED]" in result
        assert "test@example.com" not in result


# =============================================================================
# Edge Cases and Complex Scenarios
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and complex scenarios."""

    @pytest.fixture
    def redactor(self):
        """Create PIIRedactor with all types enabled."""
        from aragora.services.pii_redactor import PIIRedactor

        return PIIRedactor(log_redactions=False)

    def test_empty_text(self, redactor):
        """Test handling of empty text."""
        result = redactor.redact("")

        assert result.has_pii is False
        assert result.redacted_text == ""
        assert result.original_text == ""

    def test_no_pii_in_text(self, redactor):
        """Test text with no PII."""
        result = redactor.redact("This is a normal sentence with no sensitive data.")

        assert result.has_pii is False
        assert result.redacted_text == result.original_text

    def test_multiple_pii_types(self, redactor):
        """Test text with multiple PII types."""
        text = (
            "Contact John at john@example.com or (555) 123-4567. "
            "His SSN is 123-45-6789 and IP is 192.168.1.1."
        )

        result = redactor.redact(text)

        assert result.match_count >= 4
        assert "[EMAIL_REDACTED]" in result.redacted_text
        assert "[PHONE_REDACTED]" in result.redacted_text
        assert "[SSN_REDACTED]" in result.redacted_text
        assert "[IP_REDACTED]" in result.redacted_text

    def test_overlapping_matches(self, redactor):
        """Test handling when matches might overlap."""
        # 10-digit phone could also match as SSN without dashes
        from aragora.services.pii_redactor import PIIRedactor, PIIType

        redactor = PIIRedactor(
            enabled_types=[PIIType.PHONE, PIIType.SSN],
            log_redactions=False,
        )

        result = redactor.redact("Number: 5551234567")

        # Should still redact (one pattern will match)
        assert result.has_pii is True

    def test_redaction_preserves_surrounding_text(self, redactor):
        """Test that surrounding text is preserved."""
        result = redactor.redact("START test@example.com END")

        assert result.redacted_text.startswith("START")
        assert result.redacted_text.endswith("END")
        assert "[EMAIL_REDACTED]" in result.redacted_text
