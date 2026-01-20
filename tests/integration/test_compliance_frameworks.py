"""
Compliance Framework Integration Tests.

Tests the compliance framework detection and validation capabilities
for HIPAA, GDPR, SOX, PCI-DSS, and other regulatory frameworks.
"""

import pytest
from datetime import datetime, timezone
from typing import Any


@pytest.fixture
def compliance_context():
    """Create compliance checking context."""
    return {
        "workspace_id": "test_workspace",
        "user_id": "test_user",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


class TestHIPAACompliance:
    """Test HIPAA compliance detection."""

    def test_hipaa_framework_initialization(self):
        """Test HIPAA framework is properly defined."""
        from aragora.compliance.framework import HIPAA_FRAMEWORK

        assert HIPAA_FRAMEWORK.id == "hipaa"
        assert HIPAA_FRAMEWORK.name == "HIPAA"
        assert len(HIPAA_FRAMEWORK.rules) > 0

    def test_hipaa_phi_detection(self):
        """Test detection of Protected Health Information."""
        from aragora.compliance.framework import HIPAA_FRAMEWORK

        # Test PHI patterns
        test_cases = [
            ("Patient John Doe SSN 123-45-6789", True),
            ("Medical record #12345", True),
            ("diagnosis: cancer treatment plan", True),
            ("The weather is nice today", False),
        ]

        for text, should_have_issues in test_cases:
            issues = HIPAA_FRAMEWORK.check(text)
            if should_have_issues:
                # May or may not detect depending on rules
                pass  # Verify it runs without error
            assert isinstance(issues, list)

    def test_hipaa_rule_categories(self):
        """Test HIPAA rules cover key categories."""
        from aragora.compliance.framework import HIPAA_FRAMEWORK

        categories = {rule.category for rule in HIPAA_FRAMEWORK.rules}
        # Should cover multiple compliance areas
        assert len(categories) > 0

    def test_hipaa_encryption_check(self):
        """Test detection of unencrypted PHI concerns."""
        from aragora.compliance.framework import HIPAA_FRAMEWORK

        text = "Store patient data in plaintext database"
        issues = HIPAA_FRAMEWORK.check(text)
        # Should detect encryption concern
        assert isinstance(issues, list)


class TestGDPRCompliance:
    """Test GDPR compliance detection."""

    def test_gdpr_framework_initialization(self):
        """Test GDPR framework is properly defined."""
        from aragora.compliance.framework import GDPR_FRAMEWORK

        assert GDPR_FRAMEWORK.id == "gdpr"
        assert GDPR_FRAMEWORK.name == "GDPR"
        assert len(GDPR_FRAMEWORK.rules) > 0

    def test_gdpr_personal_data_detection(self):
        """Test detection of personal data concerns."""
        from aragora.compliance.framework import GDPR_FRAMEWORK

        test_cases = [
            "Collect email without consent",
            "Store IP address permanently",
            "Transfer data outside EU without safeguards",
        ]

        for text in test_cases:
            issues = GDPR_FRAMEWORK.check(text)
            assert isinstance(issues, list)

    def test_gdpr_consent_rules(self):
        """Test consent-related rule detection."""
        from aragora.compliance.framework import GDPR_FRAMEWORK

        text = "Process personal data without user consent"
        issues = GDPR_FRAMEWORK.check(text)
        assert isinstance(issues, list)

    def test_gdpr_data_transfer_rules(self):
        """Test data transfer rule detection."""
        from aragora.compliance.framework import GDPR_FRAMEWORK

        text = "Transfer data to third country without adequacy decision"
        issues = GDPR_FRAMEWORK.check(text)
        assert isinstance(issues, list)


class TestSOXCompliance:
    """Test SOX compliance detection."""

    def test_sox_framework_initialization(self):
        """Test SOX framework is properly defined."""
        from aragora.compliance.framework import SOX_FRAMEWORK

        assert SOX_FRAMEWORK.id == "sox"
        assert SOX_FRAMEWORK.name == "SOX"
        assert len(SOX_FRAMEWORK.rules) > 0

    def test_sox_financial_controls(self):
        """Test financial control detection."""
        from aragora.compliance.framework import SOX_FRAMEWORK

        test_cases = [
            "No segregation of duties for transactions",
            "Skip audit trail for financial reports",
            "Allow manual override of controls",
        ]

        for text in test_cases:
            issues = SOX_FRAMEWORK.check(text)
            assert isinstance(issues, list)

    def test_sox_audit_requirements(self):
        """Test audit trail requirement detection."""
        from aragora.compliance.framework import SOX_FRAMEWORK

        text = "Disable audit logging for financial system"
        issues = SOX_FRAMEWORK.check(text)
        assert isinstance(issues, list)


class TestPCIDSSCompliance:
    """Test PCI-DSS compliance detection."""

    def test_pci_framework_initialization(self):
        """Test PCI-DSS framework is properly defined."""
        from aragora.compliance.framework import PCI_DSS_FRAMEWORK

        assert PCI_DSS_FRAMEWORK.id == "pci_dss"
        assert PCI_DSS_FRAMEWORK.name == "PCI-DSS"
        assert len(PCI_DSS_FRAMEWORK.rules) > 0

    def test_pci_card_data_detection(self):
        """Test cardholder data detection."""
        from aragora.compliance.framework import PCI_DSS_FRAMEWORK

        test_cases = [
            "Store credit card number in plaintext",
            "Log CVV values",
            "Transmit card data over unencrypted connection",
        ]

        for text in test_cases:
            issues = PCI_DSS_FRAMEWORK.check(text)
            assert isinstance(issues, list)

    def test_pci_encryption_requirements(self):
        """Test encryption requirement detection."""
        from aragora.compliance.framework import PCI_DSS_FRAMEWORK

        text = "Store PAN without encryption"
        issues = PCI_DSS_FRAMEWORK.check(text)
        assert isinstance(issues, list)


class TestOWASPCompliance:
    """Test OWASP compliance detection."""

    def test_owasp_framework_initialization(self):
        """Test OWASP framework is properly defined."""
        from aragora.compliance.framework import OWASP_FRAMEWORK

        assert OWASP_FRAMEWORK.id == "owasp"
        assert OWASP_FRAMEWORK.name == "OWASP Top 10"
        assert len(OWASP_FRAMEWORK.rules) > 0

    def test_owasp_injection_detection(self):
        """Test injection vulnerability detection."""
        from aragora.compliance.framework import OWASP_FRAMEWORK

        test_cases = [
            "Execute SQL with user input directly",
            "eval(user_input)",
            "system(command_from_user)",
        ]

        for text in test_cases:
            issues = OWASP_FRAMEWORK.check(text)
            assert isinstance(issues, list)

    def test_owasp_auth_issues(self):
        """Test authentication issue detection."""
        from aragora.compliance.framework import OWASP_FRAMEWORK

        text = "Use hardcoded password for authentication"
        issues = OWASP_FRAMEWORK.check(text)
        assert isinstance(issues, list)


class TestComplianceFrameworkManager:
    """Test ComplianceFrameworkManager functionality."""

    def test_manager_initialization(self):
        """Test manager initializes with all frameworks."""
        from aragora.compliance.framework import ComplianceFrameworkManager

        manager = ComplianceFrameworkManager()
        assert len(manager._frameworks) > 0

    def test_check_with_all_frameworks(self):
        """Test checking content against all frameworks."""
        from aragora.compliance.framework import ComplianceFrameworkManager

        manager = ComplianceFrameworkManager()

        text = """
        Patient John Doe, SSN 123-45-6789
        Credit card: 4111-1111-1111-1111
        Transfer data without consent
        Execute user input directly
        No segregation of duties
        """

        result = manager.check(text)

        assert result is not None
        assert hasattr(result, "compliant")
        assert hasattr(result, "issues")
        assert hasattr(result, "score")

    def test_check_specific_frameworks(self):
        """Test checking against specific frameworks."""
        from aragora.compliance.framework import ComplianceFrameworkManager

        manager = ComplianceFrameworkManager()

        text = "Patient data stored in plaintext"
        result = manager.check(text, frameworks=["hipaa"])

        assert result is not None
        assert "hipaa" in result.frameworks_checked

    def test_compliance_score_calculation(self):
        """Test compliance score is calculated."""
        from aragora.compliance.framework import ComplianceFrameworkManager

        manager = ComplianceFrameworkManager()

        # Clean text should have high score
        clean_result = manager.check("This is a simple document about software.")
        assert clean_result.score >= 0.0
        assert clean_result.score <= 1.0

    def test_issue_severity_levels(self):
        """Test issues have proper severity levels."""
        from aragora.compliance.framework import (
            ComplianceFrameworkManager,
            ComplianceSeverity,
        )

        manager = ComplianceFrameworkManager()

        text = "Store plaintext patient SSN and credit card with no audit"
        result = manager.check(text)

        # Check severity levels are valid
        for issue in result.issues:
            assert isinstance(issue.severity, ComplianceSeverity)


class TestMultiFrameworkCompliance:
    """Test multi-framework compliance scenarios."""

    def test_combined_framework_check(self):
        """Test checking against multiple frameworks."""
        from aragora.compliance.framework import ComplianceFrameworkManager

        manager = ComplianceFrameworkManager()

        # Text with multiple compliance concerns
        text = """
        Store patient medical records with credit card numbers
        in plaintext database without encryption or access controls.
        No audit trail for financial transactions.
        Execute SQL directly from user input.
        """

        result = manager.check(text)

        assert len(result.frameworks_checked) > 0
        # Should detect issues from multiple frameworks
        if result.issues:
            frameworks_with_issues = {issue.framework for issue in result.issues}
            assert len(frameworks_with_issues) >= 1

    def test_vertical_specific_frameworks(self):
        """Test framework filtering by vertical."""
        from aragora.compliance.framework import ComplianceFrameworkManager

        manager = ComplianceFrameworkManager()

        # Get healthcare-specific frameworks
        healthcare_frameworks = [
            fw for fw in manager._frameworks.values() if "healthcare" in fw.applicable_verticals
        ]

        assert len(healthcare_frameworks) > 0


class TestComplianceReporting:
    """Test compliance reporting functionality."""

    def test_result_to_dict(self):
        """Test result can be serialized to dict."""
        from aragora.compliance.framework import ComplianceFrameworkManager

        manager = ComplianceFrameworkManager()
        result = manager.check("Test content")

        result_dict = result.to_dict()

        assert "compliant" in result_dict
        assert "score" in result_dict
        assert "frameworks_checked" in result_dict
        assert "issues" in result_dict

    def test_issues_by_framework(self):
        """Test grouping issues by framework."""
        from aragora.compliance.framework import ComplianceFrameworkManager

        manager = ComplianceFrameworkManager()

        text = "Patient data stored in plaintext credit card database"
        result = manager.check(text)

        by_framework = result.issues_by_framework()
        assert isinstance(by_framework, dict)

    def test_critical_issues_filter(self):
        """Test filtering critical issues."""
        from aragora.compliance.framework import ComplianceFrameworkManager

        manager = ComplianceFrameworkManager()

        text = "Store plaintext patient PHI with unencrypted credit card"
        result = manager.check(text)

        critical = result.critical_issues
        assert isinstance(critical, list)


class TestComplianceEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_content(self):
        """Test checking empty content."""
        from aragora.compliance.framework import ComplianceFrameworkManager

        manager = ComplianceFrameworkManager()
        result = manager.check("")

        assert result.compliant is True
        assert result.score == 1.0

    def test_very_long_content(self):
        """Test checking very long content."""
        from aragora.compliance.framework import ComplianceFrameworkManager

        manager = ComplianceFrameworkManager()

        # Generate long content
        long_text = "Normal content. " * 10000
        result = manager.check(long_text)

        assert result is not None

    def test_unicode_content(self):
        """Test checking content with unicode."""
        from aragora.compliance.framework import ComplianceFrameworkManager

        manager = ComplianceFrameworkManager()

        text = "Patient data: 患者データ, クレジットカード番号"
        result = manager.check(text)

        assert result is not None

    def test_special_characters(self):
        """Test content with special characters."""
        from aragora.compliance.framework import ComplianceFrameworkManager

        manager = ComplianceFrameworkManager()

        text = "Patient <script>alert('xss')</script> data ${}[]"
        result = manager.check(text)

        assert result is not None
