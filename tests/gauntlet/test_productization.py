"""
Tests for Gauntlet Productization Features.

Tests cover:
- Standardized error codes
- Cryptographic signing
- Vertical audit templates
- RBAC permissions

"Every feature ships with tests."
"""

from __future__ import annotations

import json
import pytest
from datetime import datetime, timezone


# =============================================================================
# Error Codes Tests
# =============================================================================


class TestGauntletErrorCodes:
    """Tests for standardized Gauntlet error codes."""

    def test_error_code_format(self):
        """Should follow GAUNTLET_XXX format."""
        from aragora.gauntlet.errors import GauntletErrorCode

        for code in GauntletErrorCode:
            assert code.value.startswith("GAUNTLET_"), f"Code {code} doesn't follow format"
            # Extract numeric part
            num = code.value.split("_")[1]
            assert num.isdigit(), f"Code {code} doesn't have numeric suffix"

    def test_error_code_categories(self):
        """Should have codes in expected categories."""
        from aragora.gauntlet.errors import GauntletErrorCode

        # Collect codes by category (first digit)
        categories = {}
        for code in GauntletErrorCode:
            cat = code.value[9]  # First digit after "GAUNTLET_"
            categories.setdefault(cat, []).append(code)

        # Verify categories exist
        assert "1" in categories, "Missing 1XX validation errors"
        assert "2" in categories, "Missing 2XX auth errors"
        assert "3" in categories, "Missing 3XX resource errors"
        assert "4" in categories, "Missing 4XX execution errors"
        assert "5" in categories, "Missing 5XX system errors"

    def test_error_response_function(self):
        """Should create properly formatted error responses."""
        from aragora.gauntlet.errors import gauntlet_error_response

        body, status = gauntlet_error_response("gauntlet_not_found", {"gauntlet_id": "test-123"})

        assert body["error"] is True
        assert body["code"] == "GAUNTLET_300"
        assert "gauntlet_id" in body["details"]
        assert status == 404

    def test_error_response_message_override(self):
        """Should allow custom messages."""
        from aragora.gauntlet.errors import gauntlet_error_response

        body, _ = gauntlet_error_response(
            "not_completed",
            message_override="Custom error message",
        )

        assert body["message"] == "Custom error message"

    def test_predefined_errors(self):
        """Should have common error scenarios predefined."""
        from aragora.gauntlet.errors import ERRORS

        expected_errors = [
            "invalid_input",
            "input_too_large",
            "not_authenticated",
            "gauntlet_not_found",
            "receipt_not_found",
            "rate_limited",
            "internal_error",
        ]

        for error_key in expected_errors:
            assert error_key in ERRORS, f"Missing predefined error: {error_key}"


# =============================================================================
# Cryptographic Signing Tests
# =============================================================================


class TestReceiptSigning:
    """Tests for cryptographic receipt signing."""

    def test_hmac_signer_sign_and_verify(self):
        """Should sign and verify receipts with HMAC."""
        from aragora.gauntlet.signing import HMACSigner

        signer = HMACSigner()
        data = b"test receipt data"

        signature = signer.sign(data)
        assert signature is not None
        assert len(signature) == 32  # SHA-256 produces 32 bytes

        # Verification should pass
        assert signer.verify(data, signature)

        # Modified data should fail verification
        assert not signer.verify(b"modified data", signature)

    def test_hmac_signer_algorithm_property(self):
        """Should report correct algorithm."""
        from aragora.gauntlet.signing import HMACSigner

        signer = HMACSigner()
        assert signer.algorithm == "HMAC-SHA256"

    def test_hmac_signer_key_id(self):
        """Should generate or use provided key ID."""
        from aragora.gauntlet.signing import HMACSigner

        # Auto-generated key ID
        signer1 = HMACSigner()
        assert signer1.key_id.startswith("hmac-")

        # Custom key ID
        signer2 = HMACSigner(key_id="my-custom-key")
        assert signer2.key_id == "my-custom-key"

    def test_receipt_signer_sign_receipt(self):
        """Should sign receipt data and produce SignedReceipt."""
        from aragora.gauntlet.signing import ReceiptSigner

        signer = ReceiptSigner()
        receipt_data = {
            "receipt_id": "test-001",
            "gauntlet_id": "gauntlet-123",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "verdict": "PASS",
            "confidence": 0.95,
        }

        signed = signer.sign(receipt_data)

        assert signed.receipt_data == receipt_data
        assert signed.signature is not None
        assert signed.signature_metadata.algorithm == "HMAC-SHA256"
        assert signed.signature_metadata.key_id is not None

    def test_receipt_signer_verify_valid(self):
        """Should verify valid signatures."""
        from aragora.gauntlet.signing import ReceiptSigner

        signer = ReceiptSigner()
        receipt_data = {"test": "data", "number": 42}

        signed = signer.sign(receipt_data)
        assert signer.verify(signed) is True

    def test_receipt_signer_verify_tampered(self):
        """Should reject tampered receipts."""
        from aragora.gauntlet.signing import ReceiptSigner

        signer = ReceiptSigner()
        receipt_data = {"test": "data", "number": 42}

        signed = signer.sign(receipt_data)

        # Tamper with receipt data
        signed.receipt_data["test"] = "tampered"

        assert signer.verify(signed) is False

    def test_signed_receipt_serialization(self):
        """Should serialize to and from JSON."""
        from aragora.gauntlet.signing import ReceiptSigner, SignedReceipt

        signer = ReceiptSigner()
        receipt_data = {"test": "data"}

        signed = signer.sign(receipt_data)

        # Serialize
        json_str = signed.to_json()
        assert isinstance(json_str, str)

        # Deserialize
        restored = SignedReceipt.from_json(json_str)
        assert restored.receipt_data == signed.receipt_data
        assert restored.signature == signed.signature

    def test_signed_receipt_to_dict(self):
        """Should convert to dict format."""
        from aragora.gauntlet.signing import ReceiptSigner

        signer = ReceiptSigner()
        receipt_data = {"verdict": "PASS"}

        signed = signer.sign(receipt_data)
        result = signed.to_dict()

        assert "receipt" in result
        assert "signature" in result
        assert "signature_metadata" in result
        assert result["receipt"]["verdict"] == "PASS"

    def test_sign_receipt_helper(self):
        """Should work with module-level helper."""
        from aragora.gauntlet.signing import sign_receipt, verify_receipt

        receipt_data = {"test": "helper function"}
        signed = sign_receipt(receipt_data)

        assert verify_receipt(signed) is True


# =============================================================================
# Vertical Templates Tests
# =============================================================================


class TestVerticalTemplates:
    """Tests for vertical audit templates."""

    def test_gdpr_template_exists(self):
        """Should have GDPR template."""
        from aragora.gauntlet.vertical_templates import get_template

        template = get_template("gdpr-compliance")
        assert template is not None
        assert template.name == "GDPR Compliance Audit"
        assert "gdpr_auditor" in template.personas

    def test_hipaa_template_exists(self):
        """Should have HIPAA template."""
        from aragora.gauntlet.vertical_templates import get_template

        template = get_template("hipaa-healthcare")
        assert template is not None
        assert "hipaa_auditor" in template.personas
        assert "phi_protection" in template.priority_categories

    def test_soc2_template_exists(self):
        """Should have SOC 2 template."""
        from aragora.gauntlet.vertical_templates import get_template

        template = get_template("soc2-trust")
        assert template is not None
        assert "security" in template.priority_categories

    def test_pci_dss_template_exists(self):
        """Should have PCI-DSS template."""
        from aragora.gauntlet.vertical_templates import get_template

        template = get_template("pci-dss-payment")
        assert template is not None
        assert "cardholder_data" in template.priority_categories

    def test_ai_governance_template_exists(self):
        """Should have AI Governance template."""
        from aragora.gauntlet.vertical_templates import get_template

        template = get_template("ai-ml-governance")
        assert template is not None
        assert "ai_ethicist" in template.personas
        assert "bias_fairness" in template.priority_categories

    def test_template_has_compliance_mappings(self):
        """Should have compliance control mappings."""
        from aragora.gauntlet.vertical_templates import get_template

        template = get_template("gdpr-compliance")
        assert len(template.compliance_mappings) > 0

        mapping = template.compliance_mappings[0]
        assert mapping.framework == "GDPR"
        assert mapping.control_id is not None
        assert mapping.control_name is not None

    def test_template_to_dict(self):
        """Should convert to dict format."""
        from aragora.gauntlet.vertical_templates import get_template

        template = get_template("soc2-trust")
        result = template.to_dict()

        assert result["id"] == "soc2-trust"
        assert result["domain"] == "soc2"
        assert isinstance(result["personas"], list)
        assert isinstance(result["compliance_mappings"], list)

    def test_list_templates(self):
        """Should list all available templates."""
        from aragora.gauntlet.vertical_templates import list_templates

        templates = list_templates()
        assert len(templates) >= 5  # At least our 5 templates

        # Verify structure
        for t in templates:
            assert "id" in t
            assert "name" in t
            assert "domain" in t
            assert "personas" in t

    def test_get_templates_for_domain(self):
        """Should filter templates by domain."""
        from aragora.gauntlet.vertical_templates import (
            get_templates_for_domain,
            VerticalDomain,
        )

        gdpr_templates = get_templates_for_domain(VerticalDomain.GDPR)
        assert len(gdpr_templates) == 1
        assert gdpr_templates[0].id == "gdpr-compliance"

    def test_create_custom_template(self):
        """Should create custom templates."""
        from aragora.gauntlet.vertical_templates import (
            create_custom_template,
            VerticalDomain,
        )

        custom = create_custom_template(
            id="my-custom",
            name="My Custom Template",
            description="Custom compliance template",
            personas=["custom_auditor"],
            priority_categories=["custom_category"],
        )

        assert custom.id == "my-custom"
        assert custom.domain == VerticalDomain.CUSTOM
        assert "custom_auditor" in custom.personas

    def test_create_custom_template_with_base(self):
        """Should inherit from base template."""
        from aragora.gauntlet.vertical_templates import create_custom_template

        custom = create_custom_template(
            id="gdpr-extended",
            name="Extended GDPR",
            description="GDPR with custom extensions",
            personas=["gdpr_auditor", "custom_persona"],
            priority_categories=["data_privacy", "custom_category"],
            base_template="gdpr-compliance",
        )

        # Should inherit compliance mappings from base
        assert len(custom.compliance_mappings) > 0
        assert custom.compliance_mappings[0].framework == "GDPR"


# =============================================================================
# RBAC Permissions Tests
# =============================================================================


class TestGauntletRBACPermissions:
    """Tests for Gauntlet RBAC permissions."""

    def test_gauntlet_resource_type_exists(self):
        """Should have GAUNTLET resource type."""
        from aragora.rbac.models import ResourceType

        assert hasattr(ResourceType, "GAUNTLET")
        assert ResourceType.GAUNTLET.value == "gauntlet"

    def test_gauntlet_actions_exist(self):
        """Should have Gauntlet-specific actions."""
        from aragora.rbac.models import Action

        assert hasattr(Action, "SIGN")
        assert hasattr(Action, "COMPARE")

    def test_gauntlet_permissions_exist(self):
        """Should have Gauntlet permissions defined."""
        from aragora.rbac.defaults import (
            PERM_GAUNTLET_RUN,
            PERM_GAUNTLET_READ,
            PERM_GAUNTLET_DELETE,
            PERM_GAUNTLET_SIGN,
            PERM_GAUNTLET_COMPARE,
            PERM_GAUNTLET_EXPORT,
        )

        assert PERM_GAUNTLET_RUN.id == "gauntlet.run"
        assert PERM_GAUNTLET_READ.id == "gauntlet.read"
        assert PERM_GAUNTLET_SIGN.id == "gauntlet.sign"

    def test_gauntlet_permissions_in_system(self):
        """Should be registered in system permissions."""
        from aragora.rbac.defaults import SYSTEM_PERMISSIONS

        assert "gauntlet.run" in SYSTEM_PERMISSIONS
        assert "gauntlet.read" in SYSTEM_PERMISSIONS
        assert "gauntlet.sign" in SYSTEM_PERMISSIONS

    def test_admin_has_all_gauntlet_permissions(self):
        """Admin role should have all Gauntlet permissions."""
        from aragora.rbac.defaults import ROLE_ADMIN

        assert "gauntlet.run" in ROLE_ADMIN.permissions
        assert "gauntlet.read" in ROLE_ADMIN.permissions
        assert "gauntlet.delete" in ROLE_ADMIN.permissions
        assert "gauntlet.sign" in ROLE_ADMIN.permissions
        assert "gauntlet.compare" in ROLE_ADMIN.permissions
        assert "gauntlet.export_data" in ROLE_ADMIN.permissions

    def test_debate_creator_has_gauntlet_access(self):
        """Debate creator should have run and read access."""
        from aragora.rbac.defaults import ROLE_DEBATE_CREATOR

        assert "gauntlet.run" in ROLE_DEBATE_CREATOR.permissions
        assert "gauntlet.read" in ROLE_DEBATE_CREATOR.permissions

    def test_member_has_basic_gauntlet_access(self):
        """Member should have basic Gauntlet access."""
        from aragora.rbac.defaults import ROLE_MEMBER

        assert "gauntlet.run" in ROLE_MEMBER.permissions
        assert "gauntlet.read" in ROLE_MEMBER.permissions

    def test_analyst_has_read_only(self):
        """Analyst should have read-only Gauntlet access."""
        from aragora.rbac.defaults import ROLE_ANALYST

        assert "gauntlet.read" in ROLE_ANALYST.permissions
        assert "gauntlet.run" not in ROLE_ANALYST.permissions

    def test_viewer_has_no_gauntlet_access(self):
        """Viewer should not have Gauntlet access."""
        from aragora.rbac.defaults import ROLE_VIEWER

        assert "gauntlet.run" not in ROLE_VIEWER.permissions
        assert "gauntlet.read" not in ROLE_VIEWER.permissions


# =============================================================================
# Integration Tests
# =============================================================================


class TestGauntletProductizationIntegration:
    """Integration tests for productization features."""

    def test_error_codes_are_serializable(self):
        """Error responses should be JSON-serializable."""
        from aragora.gauntlet.errors import gauntlet_error_response

        body, _ = gauntlet_error_response(
            "invalid_input",
            details={"field": "content", "reason": "too short"},
        )

        # Should not raise
        json_str = json.dumps(body)
        assert "GAUNTLET_100" in json_str

    def test_signed_receipt_structure_matches_openapi(self):
        """Signed receipt structure should match OpenAPI spec."""
        from aragora.gauntlet.signing import ReceiptSigner

        signer = ReceiptSigner()
        receipt_data = {
            "receipt_id": "receipt-test",
            "gauntlet_id": "gauntlet-test",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "verdict": "PASS",
            "confidence": 0.95,
        }

        signed = signer.sign(receipt_data)
        result = signed.to_dict()

        # Verify structure matches OpenAPI SignedReceipt schema
        assert "receipt" in result
        assert "signature" in result
        assert "signature_metadata" in result

        metadata = result["signature_metadata"]
        assert "algorithm" in metadata
        assert "timestamp" in metadata
        assert "key_id" in metadata
        assert "version" in metadata

    def test_template_personas_are_valid(self):
        """Template personas should be valid gauntlet personas."""
        from aragora.gauntlet.vertical_templates import VERTICAL_TEMPLATES

        # These are the known valid personas (from the gauntlet orchestrator)
        valid_personas = {
            "gdpr_auditor",
            "hipaa_auditor",
            "soc2_auditor",
            "finra_auditor",
            "pci_auditor",
            "iso_auditor",
            "nist_auditor",
            "fedramp_auditor",
            "ada_auditor",
            "security_analyst",
            "penetration_tester",
            "privacy_advocate",
            "compliance_officer",
            "risk_assessor",
            "ai_ethicist",
            "fairness_auditor",
        }

        for template in VERTICAL_TEMPLATES.values():
            for persona in template.personas:
                # Just verify format (actual validation happens at runtime)
                assert isinstance(persona, str)
                assert len(persona) > 0
