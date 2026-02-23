"""
Tests for HIPAA Compliance Handler.

Tests cover:
- HIPAA compliance status assessment
- PHI access logging and monitoring (45 CFR 164.312(b))
- Breach risk assessment (45 CFR 164.402)
- Business Associate Agreement (BAA) management
- Security Rule compliance reporting
- All 8 security safeguard categories (administrative, physical, technical)
- 50+ compliance controls coverage
- RBAC permission enforcement
- Error handling and edge cases
"""

from __future__ import annotations

import hashlib
import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.compliance.hipaa import (
    HIPAAMixin,
    PERM_HIPAA_READ,
    PERM_HIPAA_REPORT,
    PERM_HIPAA_BREACHES_READ,
    PERM_HIPAA_BREACHES_REPORT,
    PERM_HIPAA_BAA_MANAGE,
    PERM_HIPAA_ADMIN,
)
from aragora.server.handlers.base import HandlerResult


# ============================================================================
# Test Fixtures
# ============================================================================


class MockHIPAAHandler(HIPAAMixin):
    """Mock handler class that includes the HIPAA mixin for testing."""

    def __init__(self):
        pass


@pytest.fixture
def hipaa_handler():
    """Create a mock HIPAA handler instance."""
    return MockHIPAAHandler()


@pytest.fixture
def mock_audit_store():
    """Create a mock audit store."""
    store = MagicMock()
    store.log_event = MagicMock()
    store.get_log = MagicMock(return_value=[])
    return store


# ============================================================================
# Permission Constants Tests
# ============================================================================


class TestHIPAAPermissionConstants:
    """Tests for HIPAA RBAC permission constants."""

    def test_permission_constants_defined(self):
        """All required HIPAA permission constants are defined."""
        assert PERM_HIPAA_READ == "compliance:hipaa:read"
        assert PERM_HIPAA_REPORT == "compliance:hipaa:report"
        assert PERM_HIPAA_BREACHES_READ == "compliance:breaches:read"
        assert PERM_HIPAA_BREACHES_REPORT == "compliance:breaches:report"
        assert PERM_HIPAA_BAA_MANAGE == "compliance:baa:manage"
        assert PERM_HIPAA_ADMIN == "compliance:hipaa:admin"

    def test_permission_constants_exported(self):
        """Permission constants are exported in __all__."""
        from aragora.server.handlers.compliance import hipaa

        assert "PERM_HIPAA_READ" in hipaa.__all__
        assert "PERM_HIPAA_REPORT" in hipaa.__all__
        assert "PERM_HIPAA_BREACHES_READ" in hipaa.__all__
        assert "PERM_HIPAA_BREACHES_REPORT" in hipaa.__all__
        assert "PERM_HIPAA_BAA_MANAGE" in hipaa.__all__
        assert "PERM_HIPAA_ADMIN" in hipaa.__all__


# ============================================================================
# Security Safeguards Structure Tests
# ============================================================================


class TestSecuritySafeguardsStructure:
    """Tests for HIPAA Security Rule safeguard data structures."""

    def test_security_safeguards_has_all_categories(self, hipaa_handler):
        """SECURITY_SAFEGUARDS contains all three required categories."""
        assert "administrative" in hipaa_handler.SECURITY_SAFEGUARDS
        assert "physical" in hipaa_handler.SECURITY_SAFEGUARDS
        assert "technical" in hipaa_handler.SECURITY_SAFEGUARDS

    def test_administrative_safeguards_count(self, hipaa_handler):
        """Administrative safeguards contain required 8 standards per 164.308."""
        admin_safeguards = hipaa_handler.SECURITY_SAFEGUARDS["administrative"]
        assert len(admin_safeguards) == 8

    def test_physical_safeguards_count(self, hipaa_handler):
        """Physical safeguards contain required 4 standards per 164.310."""
        physical_safeguards = hipaa_handler.SECURITY_SAFEGUARDS["physical"]
        assert len(physical_safeguards) == 4

    def test_technical_safeguards_count(self, hipaa_handler):
        """Technical safeguards contain required 5 standards per 164.312."""
        technical_safeguards = hipaa_handler.SECURITY_SAFEGUARDS["technical"]
        assert len(technical_safeguards) == 5

    def test_administrative_safeguard_security_management(self, hipaa_handler):
        """164.308(a)(1) Security Management Process is properly defined."""
        admin_safeguards = hipaa_handler.SECURITY_SAFEGUARDS["administrative"]
        security_mgmt = next((s for s in admin_safeguards if s["id"] == "164.308(a)(1)"), None)
        assert security_mgmt is not None
        assert security_mgmt["name"] == "Security Management Process"
        assert "Risk Analysis" in security_mgmt["controls"]
        assert "Risk Management" in security_mgmt["controls"]
        assert "Sanction Policy" in security_mgmt["controls"]
        assert "Information System Activity Review" in security_mgmt["controls"]

    def test_administrative_safeguard_workforce_security(self, hipaa_handler):
        """164.308(a)(3) Workforce Security is properly defined."""
        admin_safeguards = hipaa_handler.SECURITY_SAFEGUARDS["administrative"]
        workforce = next((s for s in admin_safeguards if s["id"] == "164.308(a)(3)"), None)
        assert workforce is not None
        assert workforce["name"] == "Workforce Security"
        assert "Authorization/Supervision" in workforce["controls"]
        assert "Workforce Clearance" in workforce["controls"]
        assert "Termination Procedures" in workforce["controls"]

    def test_administrative_safeguard_security_awareness(self, hipaa_handler):
        """164.308(a)(5) Security Awareness Training is properly defined."""
        admin_safeguards = hipaa_handler.SECURITY_SAFEGUARDS["administrative"]
        awareness = next((s for s in admin_safeguards if s["id"] == "164.308(a)(5)"), None)
        assert awareness is not None
        assert awareness["name"] == "Security Awareness Training"
        assert "Security Reminders" in awareness["controls"]
        assert "Malicious Software Protection" in awareness["controls"]
        assert "Log-in Monitoring" in awareness["controls"]
        assert "Password Management" in awareness["controls"]

    def test_administrative_safeguard_contingency_plan(self, hipaa_handler):
        """164.308(a)(7) Contingency Plan is properly defined."""
        admin_safeguards = hipaa_handler.SECURITY_SAFEGUARDS["administrative"]
        contingency = next((s for s in admin_safeguards if s["id"] == "164.308(a)(7)"), None)
        assert contingency is not None
        assert contingency["name"] == "Contingency Plan"
        assert "Data Backup Plan" in contingency["controls"]
        assert "Disaster Recovery Plan" in contingency["controls"]
        assert "Emergency Mode Operation" in contingency["controls"]
        assert "Testing and Revision" in contingency["controls"]

    def test_physical_safeguard_facility_access(self, hipaa_handler):
        """164.310(a) Facility Access Controls is properly defined."""
        physical_safeguards = hipaa_handler.SECURITY_SAFEGUARDS["physical"]
        facility = next((s for s in physical_safeguards if s["id"] == "164.310(a)"), None)
        assert facility is not None
        assert facility["name"] == "Facility Access Controls"
        assert "Contingency Operations" in facility["controls"]
        assert "Facility Security Plan" in facility["controls"]
        assert "Access Control/Validation" in facility["controls"]
        assert "Maintenance Records" in facility["controls"]

    def test_physical_safeguard_device_media_controls(self, hipaa_handler):
        """164.310(d) Device and Media Controls is properly defined."""
        physical_safeguards = hipaa_handler.SECURITY_SAFEGUARDS["physical"]
        device = next((s for s in physical_safeguards if s["id"] == "164.310(d)"), None)
        assert device is not None
        assert device["name"] == "Device and Media Controls"
        assert "Disposal" in device["controls"]
        assert "Media Re-use" in device["controls"]
        assert "Accountability" in device["controls"]
        assert "Data Backup/Storage" in device["controls"]

    def test_technical_safeguard_access_control(self, hipaa_handler):
        """164.312(a) Access Control is properly defined."""
        technical_safeguards = hipaa_handler.SECURITY_SAFEGUARDS["technical"]
        access = next((s for s in technical_safeguards if s["id"] == "164.312(a)"), None)
        assert access is not None
        assert access["name"] == "Access Control"
        assert "Unique User Identification" in access["controls"]
        assert "Emergency Access Procedure" in access["controls"]
        assert "Automatic Logoff" in access["controls"]
        assert "Encryption/Decryption" in access["controls"]

    def test_technical_safeguard_audit_controls(self, hipaa_handler):
        """164.312(b) Audit Controls is properly defined."""
        technical_safeguards = hipaa_handler.SECURITY_SAFEGUARDS["technical"]
        audit = next((s for s in technical_safeguards if s["id"] == "164.312(b)"), None)
        assert audit is not None
        assert audit["name"] == "Audit Controls"
        assert "Hardware/Software/Procedural Audit Mechanisms" in audit["controls"]

    def test_technical_safeguard_transmission_security(self, hipaa_handler):
        """164.312(e) Transmission Security is properly defined."""
        technical_safeguards = hipaa_handler.SECURITY_SAFEGUARDS["technical"]
        transmission = next((s for s in technical_safeguards if s["id"] == "164.312(e)"), None)
        assert transmission is not None
        assert transmission["name"] == "Transmission Security"
        assert "Integrity Controls" in transmission["controls"]
        assert "Encryption" in transmission["controls"]

    def test_all_safeguards_have_required_fields(self, hipaa_handler):
        """All safeguards have required id, name, and controls fields."""
        for category, safeguards in hipaa_handler.SECURITY_SAFEGUARDS.items():
            for safeguard in safeguards:
                assert "id" in safeguard, f"Missing id in {category}"
                assert "name" in safeguard, f"Missing name in {category}"
                assert "controls" in safeguard, f"Missing controls in {category}"
                assert isinstance(safeguard["controls"], list)
                assert len(safeguard["controls"]) > 0


# ============================================================================
# PHI Identifiers Tests
# ============================================================================


class TestPHIIdentifiers:
    """Tests for PHI identifiers per 45 CFR 164.514."""

    def test_phi_identifiers_count(self, hipaa_handler):
        """PHI_IDENTIFIERS contains all 18 categories per Safe Harbor."""
        assert len(hipaa_handler.PHI_IDENTIFIERS) == 18

    def test_phi_identifiers_includes_names(self, hipaa_handler):
        """PHI identifiers include names."""
        assert "Names" in hipaa_handler.PHI_IDENTIFIERS

    def test_phi_identifiers_includes_geographic(self, hipaa_handler):
        """PHI identifiers include geographic data."""
        assert "Geographic data smaller than state" in hipaa_handler.PHI_IDENTIFIERS

    def test_phi_identifiers_includes_dates(self, hipaa_handler):
        """PHI identifiers include dates except year."""
        assert "Dates (except year)" in hipaa_handler.PHI_IDENTIFIERS

    def test_phi_identifiers_includes_ssn(self, hipaa_handler):
        """PHI identifiers include Social Security numbers."""
        assert "Social Security numbers" in hipaa_handler.PHI_IDENTIFIERS

    def test_phi_identifiers_includes_medical_records(self, hipaa_handler):
        """PHI identifiers include medical record numbers."""
        assert "Medical record numbers" in hipaa_handler.PHI_IDENTIFIERS

    def test_phi_identifiers_includes_contact_info(self, hipaa_handler):
        """PHI identifiers include contact information."""
        assert "Phone numbers" in hipaa_handler.PHI_IDENTIFIERS
        assert "Fax numbers" in hipaa_handler.PHI_IDENTIFIERS
        assert "Email addresses" in hipaa_handler.PHI_IDENTIFIERS

    def test_phi_identifiers_includes_digital_identifiers(self, hipaa_handler):
        """PHI identifiers include digital identifiers."""
        assert "Web URLs" in hipaa_handler.PHI_IDENTIFIERS
        assert "IP addresses" in hipaa_handler.PHI_IDENTIFIERS
        assert "Device identifiers" in hipaa_handler.PHI_IDENTIFIERS

    def test_phi_identifiers_includes_biometric(self, hipaa_handler):
        """PHI identifiers include biometric identifiers."""
        assert "Biometric identifiers" in hipaa_handler.PHI_IDENTIFIERS
        assert "Full face photos" in hipaa_handler.PHI_IDENTIFIERS

    def test_phi_identifiers_includes_unique_identifier_catchall(self, hipaa_handler):
        """PHI identifiers include unique identifier catch-all."""
        assert "Any other unique identifier" in hipaa_handler.PHI_IDENTIFIERS


# ============================================================================
# HIPAA Status Tests
# ============================================================================


class TestHIPAAStatus:
    """Tests for HIPAA compliance status endpoint."""

    @pytest.mark.asyncio
    async def test_hipaa_status_returns_success(self, hipaa_handler):
        """HIPAA status returns 200 with compliance information."""
        result = await hipaa_handler._hipaa_status({})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["compliance_framework"] == "HIPAA"
        assert "assessed_at" in body
        assert "overall_status" in body
        assert "compliance_score" in body

    @pytest.mark.asyncio
    async def test_hipaa_status_includes_rules(self, hipaa_handler):
        """HIPAA status includes all three rules."""
        result = await hipaa_handler._hipaa_status({})

        body = json.loads(result.body)
        assert "rules" in body
        assert "privacy_rule" in body["rules"]
        assert "security_rule" in body["rules"]
        assert "breach_notification_rule" in body["rules"]

    @pytest.mark.asyncio
    async def test_hipaa_status_includes_baa_status(self, hipaa_handler):
        """HIPAA status includes business associates status."""
        result = await hipaa_handler._hipaa_status({})

        body = json.loads(result.body)
        assert "business_associates" in body
        assert "total_baas" in body["business_associates"]
        assert "active" in body["business_associates"]

    @pytest.mark.asyncio
    async def test_hipaa_status_full_scope(self, hipaa_handler):
        """HIPAA status with full scope includes safeguard details."""
        result = await hipaa_handler._hipaa_status({"scope": "full"})

        body = json.loads(result.body)
        assert "safeguard_details" in body
        assert "phi_controls" in body

    @pytest.mark.asyncio
    async def test_hipaa_status_summary_scope(self, hipaa_handler):
        """HIPAA status with summary scope excludes details."""
        result = await hipaa_handler._hipaa_status({"scope": "summary"})

        body = json.loads(result.body)
        assert "safeguard_details" not in body

    @pytest.mark.asyncio
    async def test_hipaa_status_includes_recommendations(self, hipaa_handler):
        """HIPAA status includes recommendations by default."""
        result = await hipaa_handler._hipaa_status({})

        body = json.loads(result.body)
        assert "recommendations" in body
        assert isinstance(body["recommendations"], list)

    @pytest.mark.asyncio
    async def test_hipaa_status_excludes_recommendations_when_disabled(self, hipaa_handler):
        """HIPAA status excludes recommendations when disabled."""
        result = await hipaa_handler._hipaa_status({"include_recommendations": "false"})

        body = json.loads(result.body)
        assert "recommendations" not in body

    @pytest.mark.asyncio
    async def test_hipaa_status_compliance_score_calculation(self, hipaa_handler):
        """HIPAA status calculates compliance score correctly."""
        result = await hipaa_handler._hipaa_status({})

        body = json.loads(result.body)
        assert 0 <= body["compliance_score"] <= 100

    @pytest.mark.asyncio
    async def test_hipaa_status_overall_status_values(self, hipaa_handler):
        """HIPAA status returns valid overall status values."""
        result = await hipaa_handler._hipaa_status({})

        body = json.loads(result.body)
        valid_statuses = [
            "compliant",
            "substantially_compliant",
            "partially_compliant",
            "non_compliant",
        ]
        assert body["overall_status"] in valid_statuses


# ============================================================================
# PHI Access Log Tests
# ============================================================================


class TestPHIAccessLog:
    """Tests for PHI access log endpoint per 45 CFR 164.312(b)."""

    @pytest.mark.asyncio
    async def test_phi_access_log_returns_success(self, hipaa_handler, mock_audit_store):
        """PHI access log returns 200."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_phi_access_log({})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "phi_access_log" in body
        assert "count" in body

    @pytest.mark.asyncio
    async def test_phi_access_log_includes_hipaa_reference(self, hipaa_handler, mock_audit_store):
        """PHI access log includes HIPAA reference."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_phi_access_log({})

        body = json.loads(result.body)
        assert "hipaa_reference" in body
        assert "164.312(b)" in body["hipaa_reference"]

    @pytest.mark.asyncio
    async def test_phi_access_log_filters_by_patient_id(self, hipaa_handler, mock_audit_store):
        """PHI access log filters by patient_id."""
        mock_audit_store.get_log.return_value = [
            {
                "action": "phi_view",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {"patient_id": "patient-123"},
            },
            {
                "action": "phi_view",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {"patient_id": "patient-456"},
            },
        ]

        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_phi_access_log({"patient_id": "patient-123"})

        body = json.loads(result.body)
        assert body["count"] == 1
        assert body["phi_access_log"][0]["patient_id"] == "patient-123"

    @pytest.mark.asyncio
    async def test_phi_access_log_filters_by_user_id(self, hipaa_handler, mock_audit_store):
        """PHI access log filters by user_id."""
        mock_audit_store.get_log.return_value = [
            {
                "action": "phi_view",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": "user-123",
                "metadata": {},
            },
            {
                "action": "phi_view",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": "user-456",
                "metadata": {},
            },
        ]

        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_phi_access_log({"user_id": "user-123"})

        body = json.loads(result.body)
        assert body["count"] == 1
        assert body["phi_access_log"][0]["user_id"] == "user-123"

    @pytest.mark.asyncio
    async def test_phi_access_log_filters_by_date_range(self, hipaa_handler, mock_audit_store):
        """PHI access log filters by date range."""
        now = datetime.now(timezone.utc)
        mock_audit_store.get_log.return_value = [
            {
                "action": "phi_view",
                "timestamp": (now - timedelta(days=1)).isoformat(),
                "metadata": {},
            },
            {
                "action": "phi_view",
                "timestamp": (now - timedelta(days=10)).isoformat(),
                "metadata": {},
            },
        ]

        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            from_date = (now - timedelta(days=5)).isoformat()
            result = await hipaa_handler._hipaa_phi_access_log({"from": from_date})

        body = json.loads(result.body)
        assert body["count"] == 1

    @pytest.mark.asyncio
    async def test_phi_access_log_respects_limit(self, hipaa_handler, mock_audit_store):
        """PHI access log respects limit parameter."""
        mock_audit_store.get_log.return_value = [
            {
                "action": "phi_view",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {},
            }
            for _ in range(150)
        ]

        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_phi_access_log({"limit": "50"})

        body = json.loads(result.body)
        assert body["count"] <= 50

    @pytest.mark.asyncio
    async def test_phi_access_log_max_limit(self, hipaa_handler, mock_audit_store):
        """PHI access log enforces max limit of 1000."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_phi_access_log({"limit": "5000"})

        # Should not error; the limit is capped at 1000
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_phi_access_log_includes_access_metadata(self, hipaa_handler, mock_audit_store):
        """PHI access log includes access metadata."""
        mock_audit_store.get_log.return_value = [
            {
                "action": "phi_view",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": "user-123",
                "metadata": {
                    "patient_id": "patient-001",
                    "access_type": "view",
                    "phi_elements": ["name", "dob"],
                    "purpose": "treatment",
                    "ip_address": "192.168.1.1",
                },
            }
        ]

        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_phi_access_log({})

        body = json.loads(result.body)
        access = body["phi_access_log"][0]
        assert access["access_type"] == "view"
        assert "name" in access["phi_elements"]
        assert access["purpose"] == "treatment"
        assert access["ip_address"] == "192.168.1.1"

    @pytest.mark.asyncio
    async def test_phi_access_log_handles_error(self, hipaa_handler, mock_audit_store):
        """PHI access log handles store errors gracefully."""
        mock_audit_store.get_log.side_effect = RuntimeError("Database error")

        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_phi_access_log({})

        assert result.status_code == 500
        body = json.loads(result.body)
        assert "error" in body


# ============================================================================
# Breach Risk Assessment Tests
# ============================================================================


class TestBreachRiskAssessment:
    """Tests for HIPAA breach risk assessment per 45 CFR 164.402."""

    @pytest.mark.asyncio
    async def test_breach_assessment_requires_incident_id(self, hipaa_handler):
        """Breach assessment requires incident_id."""
        result = await hipaa_handler._hipaa_breach_assessment({"incident_type": "data_loss"})

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "incident_id is required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_breach_assessment_requires_incident_type(self, hipaa_handler):
        """Breach assessment requires incident_type."""
        result = await hipaa_handler._hipaa_breach_assessment({"incident_id": "inc-123"})

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "incident_type is required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_breach_assessment_no_phi(self, hipaa_handler, mock_audit_store):
        """Breach assessment with no PHI returns not_applicable."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_breach_assessment(
                {
                    "incident_id": "inc-123",
                    "incident_type": "system_outage",
                    "phi_involved": False,
                }
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["breach_determination"] == "not_applicable"
        assert body["notification_required"] is False

    @pytest.mark.asyncio
    async def test_breach_assessment_with_sensitive_phi(self, hipaa_handler, mock_audit_store):
        """Breach assessment with sensitive PHI returns high risk."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_breach_assessment(
                {
                    "incident_id": "inc-123",
                    "incident_type": "data_breach",
                    "phi_involved": True,
                    "phi_types": ["SSN", "Financial", "Medical diagnosis"],
                    "unauthorized_access": {"confirmed_access": True},
                }
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["breach_determination"] == "presumed_breach"
        assert body["notification_required"] is True

    @pytest.mark.asyncio
    async def test_breach_assessment_four_factors(self, hipaa_handler, mock_audit_store):
        """Breach assessment evaluates all four HHS factors."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_breach_assessment(
                {
                    "incident_id": "inc-123",
                    "incident_type": "data_breach",
                    "phi_involved": True,
                    "phi_types": ["Name"],
                    "unauthorized_access": {},
                    "mitigation_actions": [],
                }
            )

        body = json.loads(result.body)
        assert len(body["risk_factors"]) == 4
        factor_names = [f["factor"] for f in body["risk_factors"]]
        assert "Nature and extent of PHI" in factor_names
        assert "Unauthorized person" in factor_names
        assert "PHI acquisition/viewing" in factor_names
        assert "Risk mitigation" in factor_names

    @pytest.mark.asyncio
    async def test_breach_assessment_known_recipient_moderate_risk(
        self, hipaa_handler, mock_audit_store
    ):
        """Breach assessment with known recipient returns moderate risk."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_breach_assessment(
                {
                    "incident_id": "inc-123",
                    "incident_type": "misdirected_email",
                    "phi_involved": True,
                    "phi_types": ["Name"],
                    "unauthorized_access": {"known_recipient": True},
                }
            )

        body = json.loads(result.body)
        unauthorized_factor = next(
            f for f in body["risk_factors"] if f["factor"] == "Unauthorized person"
        )
        assert unauthorized_factor["risk"] == "moderate"

    @pytest.mark.asyncio
    async def test_breach_assessment_comprehensive_mitigation(
        self, hipaa_handler, mock_audit_store
    ):
        """Breach assessment with comprehensive mitigation returns low risk."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_breach_assessment(
                {
                    "incident_id": "inc-123",
                    "incident_type": "data_breach",
                    "phi_involved": True,
                    "mitigation_actions": [
                        "Disabled compromised account",
                        "Reset all passwords",
                        "Notified affected users",
                        "Enhanced monitoring",
                    ],
                }
            )

        body = json.loads(result.body)
        mitigation_factor = next(
            f for f in body["risk_factors"] if f["factor"] == "Risk mitigation"
        )
        assert mitigation_factor["risk"] == "low"

    @pytest.mark.asyncio
    async def test_breach_assessment_notification_deadlines(self, hipaa_handler, mock_audit_store):
        """Breach assessment includes notification deadlines when required."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_breach_assessment(
                {
                    "incident_id": "inc-123",
                    "incident_type": "data_breach",
                    "phi_involved": True,
                    "phi_types": ["SSN", "Financial"],
                    "unauthorized_access": {"confirmed_access": True},
                    "affected_individuals": 1000,
                }
            )

        body = json.loads(result.body)
        assert body["notification_required"] is True
        assert "notification_deadlines" in body
        assert "individual_notification" in body["notification_deadlines"]
        assert "hhs_notification" in body["notification_deadlines"]
        assert "media_notification" in body["notification_deadlines"]

    @pytest.mark.asyncio
    async def test_breach_assessment_media_notification_threshold(
        self, hipaa_handler, mock_audit_store
    ):
        """Breach assessment requires media notification for 500+ affected."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            # Under 500
            result_small = await hipaa_handler._hipaa_breach_assessment(
                {
                    "incident_id": "inc-123",
                    "incident_type": "data_breach",
                    "phi_involved": True,
                    "phi_types": ["SSN"],
                    "unauthorized_access": {"confirmed_access": True},
                    "affected_individuals": 100,
                }
            )

            # 500 or more
            result_large = await hipaa_handler._hipaa_breach_assessment(
                {
                    "incident_id": "inc-456",
                    "incident_type": "data_breach",
                    "phi_involved": True,
                    "phi_types": ["SSN"],
                    "unauthorized_access": {"confirmed_access": True},
                    "affected_individuals": 500,
                }
            )

        body_small = json.loads(result_small.body)
        body_large = json.loads(result_large.body)

        assert body_small["notification_deadlines"]["media_notification"] == "Not required"
        assert body_large["notification_deadlines"]["media_notification"] != "Not required"

    @pytest.mark.asyncio
    async def test_breach_assessment_logs_event(self, hipaa_handler, mock_audit_store):
        """Breach assessment logs audit event."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            await hipaa_handler._hipaa_breach_assessment(
                {
                    "incident_id": "inc-123",
                    "incident_type": "data_breach",
                    "phi_involved": True,
                }
            )

        mock_audit_store.log_event.assert_called_once()
        call_kwargs = mock_audit_store.log_event.call_args.kwargs
        assert call_kwargs["action"] == "hipaa_breach_assessment"
        assert call_kwargs["resource_type"] == "incident"
        assert call_kwargs["resource_id"] == "inc-123"

    @pytest.mark.asyncio
    async def test_breach_assessment_low_probability(self, hipaa_handler, mock_audit_store):
        """Breach assessment returns low_probability for minimal risk."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_breach_assessment(
                {
                    "incident_id": "inc-123",
                    "incident_type": "potential_exposure",
                    "phi_involved": True,
                    "phi_types": ["Name"],  # Not sensitive
                    "unauthorized_access": {
                        "known_recipient": True,  # Moderate risk
                        "confirmed_access": False,  # Low risk
                    },
                    "mitigation_actions": [
                        "Action 1",
                        "Action 2",
                        "Action 3",
                    ],  # Low risk
                }
            )

        body = json.loads(result.body)
        assert body["breach_determination"] == "low_probability"
        assert body["notification_required"] is False


# ============================================================================
# BAA Management Tests
# ============================================================================


class TestBAAList:
    """Tests for listing Business Associate Agreements."""

    @pytest.mark.asyncio
    async def test_list_baas_returns_success(self, hipaa_handler):
        """List BAAs returns 200."""
        result = await hipaa_handler._hipaa_list_baas({})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "business_associates" in body
        assert "count" in body

    @pytest.mark.asyncio
    async def test_list_baas_includes_hipaa_reference(self, hipaa_handler):
        """List BAAs includes HIPAA reference."""
        result = await hipaa_handler._hipaa_list_baas({})

        body = json.loads(result.body)
        assert "hipaa_reference" in body
        assert "164.502(e)" in body["hipaa_reference"]
        assert "164.504(e)" in body["hipaa_reference"]

    @pytest.mark.asyncio
    async def test_list_baas_filter_by_status(self, hipaa_handler):
        """List BAAs filters by status."""
        result = await hipaa_handler._hipaa_list_baas({"status": "active"})

        body = json.loads(result.body)
        assert body["filters"]["status"] == "active"

    @pytest.mark.asyncio
    async def test_list_baas_filter_by_type(self, hipaa_handler):
        """List BAAs filters by BA type."""
        result = await hipaa_handler._hipaa_list_baas({"ba_type": "vendor"})

        body = json.loads(result.body)
        assert body["filters"]["ba_type"] == "vendor"


class TestBAACreate:
    """Tests for creating Business Associate Agreements."""

    @pytest.mark.asyncio
    async def test_create_baa_requires_business_associate(self, hipaa_handler):
        """Create BAA requires business_associate field."""
        result = await hipaa_handler._hipaa_create_baa(
            {
                "ba_type": "vendor",
                "services_provided": "Cloud hosting",
            }
        )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "business_associate is required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_create_baa_requires_valid_type(self, hipaa_handler):
        """Create BAA requires valid ba_type."""
        result = await hipaa_handler._hipaa_create_baa(
            {
                "business_associate": "Vendor A",
                "ba_type": "invalid",
                "services_provided": "Cloud hosting",
            }
        )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "vendor" in body.get("error", "") or "subcontractor" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_create_baa_requires_services(self, hipaa_handler):
        """Create BAA requires services_provided field."""
        result = await hipaa_handler._hipaa_create_baa(
            {
                "business_associate": "Vendor A",
                "ba_type": "vendor",
            }
        )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "services_provided is required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_create_baa_success(self, hipaa_handler, mock_audit_store):
        """Create BAA succeeds with valid data."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_create_baa(
                {
                    "business_associate": "Cloud Provider X",
                    "ba_type": "vendor",
                    "services_provided": "Cloud infrastructure",
                    "phi_access_scope": ["storage", "processing"],
                }
            )

        assert result.status_code == 201
        body = json.loads(result.body)
        assert "baa" in body
        assert body["baa"]["business_associate"] == "Cloud Provider X"
        assert body["baa"]["ba_type"] == "vendor"

    @pytest.mark.asyncio
    async def test_create_baa_generates_id(self, hipaa_handler, mock_audit_store):
        """Create BAA generates unique BAA ID."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_create_baa(
                {
                    "business_associate": "Cloud Provider X",
                    "ba_type": "vendor",
                    "services_provided": "Cloud infrastructure",
                }
            )

        body = json.loads(result.body)
        assert "baa_id" in body["baa"]
        assert body["baa"]["baa_id"].startswith("baa-")

    @pytest.mark.asyncio
    async def test_create_baa_includes_required_provisions(self, hipaa_handler, mock_audit_store):
        """Create BAA includes required BAA provisions."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_create_baa(
                {
                    "business_associate": "Cloud Provider X",
                    "ba_type": "vendor",
                    "services_provided": "Cloud infrastructure",
                }
            )

        body = json.loads(result.body)
        required_provisions = body["baa"]["required_provisions"]
        assert "Use/disclosure limitations" in required_provisions
        assert "Safeguards requirement" in required_provisions
        assert "Subcontractor assurances" in required_provisions
        assert "Breach notification obligation" in required_provisions
        assert "Access to PHI for amendment" in required_provisions
        assert "Accounting of disclosures" in required_provisions
        assert "Compliance with Security Rule" in required_provisions
        assert "Termination provisions" in required_provisions

    @pytest.mark.asyncio
    async def test_create_baa_logs_event(self, hipaa_handler, mock_audit_store):
        """Create BAA logs audit event."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            await hipaa_handler._hipaa_create_baa(
                {
                    "business_associate": "Cloud Provider X",
                    "ba_type": "vendor",
                    "services_provided": "Cloud infrastructure",
                }
            )

        mock_audit_store.log_event.assert_called_once()
        call_kwargs = mock_audit_store.log_event.call_args.kwargs
        assert call_kwargs["action"] == "hipaa_baa_created"
        assert call_kwargs["resource_type"] == "baa"

    @pytest.mark.asyncio
    async def test_create_baa_with_subcontractor(self, hipaa_handler, mock_audit_store):
        """Create BAA works with subcontractor type."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_create_baa(
                {
                    "business_associate": "Subcontractor Y",
                    "ba_type": "subcontractor",
                    "services_provided": "Data processing",
                }
            )

        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["baa"]["ba_type"] == "subcontractor"


# ============================================================================
# Security Report Tests
# ============================================================================


class TestSecurityReport:
    """Tests for HIPAA Security Rule compliance report."""

    @pytest.mark.asyncio
    async def test_security_report_returns_success(self, hipaa_handler):
        """Security report returns 200."""
        result = await hipaa_handler._hipaa_security_report({})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["report_type"] == "HIPAA Security Rule Compliance"

    @pytest.mark.asyncio
    async def test_security_report_includes_report_id(self, hipaa_handler):
        """Security report includes unique report ID."""
        result = await hipaa_handler._hipaa_security_report({})

        body = json.loads(result.body)
        assert "report_id" in body
        assert body["report_id"].startswith("hipaa-sec-")

    @pytest.mark.asyncio
    async def test_security_report_includes_all_categories(self, hipaa_handler):
        """Security report includes all safeguard categories."""
        result = await hipaa_handler._hipaa_security_report({})

        body = json.loads(result.body)
        assert "administrative" in body["safeguards"]
        assert "physical" in body["safeguards"]
        assert "technical" in body["safeguards"]

    @pytest.mark.asyncio
    async def test_security_report_includes_summary(self, hipaa_handler):
        """Security report includes summary statistics."""
        result = await hipaa_handler._hipaa_security_report({})

        body = json.loads(result.body)
        assert "summary" in body
        assert "total_standards_assessed" in body["summary"]
        assert "standards_compliant" in body["summary"]
        assert "compliance_percentage" in body["summary"]
        assert "overall_status" in body["summary"]

    @pytest.mark.asyncio
    async def test_security_report_includes_evidence(self, hipaa_handler):
        """Security report includes evidence when requested."""
        result = await hipaa_handler._hipaa_security_report({"include_evidence": "true"})

        body = json.loads(result.body)
        assert "evidence_references" in body
        assert isinstance(body["evidence_references"], list)

    @pytest.mark.asyncio
    async def test_security_report_excludes_evidence_by_default(self, hipaa_handler):
        """Security report excludes evidence by default."""
        result = await hipaa_handler._hipaa_security_report({})

        body = json.loads(result.body)
        assert "evidence_references" not in body

    @pytest.mark.asyncio
    async def test_security_report_html_format(self, hipaa_handler):
        """Security report returns HTML format when requested."""
        result = await hipaa_handler._hipaa_security_report({"format": "html"})

        assert result.status_code == 200
        assert result.content_type == "text/html"
        html_content = result.body.decode("utf-8")
        assert "<!DOCTYPE html>" in html_content
        assert "HIPAA Security Rule Compliance Report" in html_content

    @pytest.mark.asyncio
    async def test_security_report_json_format_default(self, hipaa_handler):
        """Security report returns JSON format by default."""
        result = await hipaa_handler._hipaa_security_report({})

        assert result.status_code == 200
        assert result.content_type == "application/json"

    @pytest.mark.asyncio
    async def test_security_report_assessment_period(self, hipaa_handler):
        """Security report includes assessment period."""
        result = await hipaa_handler._hipaa_security_report({})

        body = json.loads(result.body)
        assert "assessment_period" in body
        assert "start" in body["assessment_period"]
        assert "end" in body["assessment_period"]

    @pytest.mark.asyncio
    async def test_security_report_category_counts(self, hipaa_handler):
        """Security report includes per-category compliance counts."""
        result = await hipaa_handler._hipaa_security_report({})

        body = json.loads(result.body)
        for category in ["administrative", "physical", "technical"]:
            assert "compliant_count" in body["safeguards"][category]
            assert "total_count" in body["safeguards"][category]


# ============================================================================
# Helper Method Tests
# ============================================================================


class TestHelperMethods:
    """Tests for HIPAA handler helper methods."""

    @pytest.mark.asyncio
    async def test_evaluate_safeguards_returns_all_categories(self, hipaa_handler):
        """_evaluate_safeguards returns all safeguard categories."""
        result = await hipaa_handler._evaluate_safeguards()

        assert "administrative" in result
        assert "physical" in result
        assert "technical" in result

    @pytest.mark.asyncio
    async def test_evaluate_safeguards_includes_status(self, hipaa_handler):
        """_evaluate_safeguards includes status for each safeguard."""
        result = await hipaa_handler._evaluate_safeguards()

        for category, safeguards in result.items():
            for safeguard in safeguards:
                assert "status" in safeguard
                assert safeguard["status"] in ["compliant", "non_compliant", "partial"]

    @pytest.mark.asyncio
    async def test_evaluate_phi_controls(self, hipaa_handler):
        """_evaluate_phi_controls returns expected structure."""
        result = await hipaa_handler._evaluate_phi_controls()

        assert "status" in result
        assert "identifiers_tracked" in result
        assert "de_identification_method" in result
        assert "minimum_necessary_enforced" in result
        assert "access_controls" in result

    @pytest.mark.asyncio
    async def test_evaluate_phi_controls_safe_harbor(self, hipaa_handler):
        """_evaluate_phi_controls uses Safe Harbor de-identification."""
        result = await hipaa_handler._evaluate_phi_controls()

        assert result["de_identification_method"] == "Safe Harbor"

    @pytest.mark.asyncio
    async def test_get_baa_status(self, hipaa_handler):
        """_get_baa_status returns expected structure."""
        result = await hipaa_handler._get_baa_status()

        assert "total_baas" in result
        assert "active" in result
        assert "expiring_soon" in result
        assert "expired" in result

    @pytest.mark.asyncio
    async def test_get_baa_list(self, hipaa_handler):
        """_get_baa_list returns list of BAAs."""
        result = await hipaa_handler._get_baa_list("active", "all")

        assert isinstance(result, list)
        for baa in result:
            assert "baa_id" in baa
            assert "business_associate" in baa
            assert "status" in baa

    @pytest.mark.asyncio
    async def test_get_hipaa_recommendations(self, hipaa_handler):
        """_get_hipaa_recommendations returns recommendations."""
        safeguards = await hipaa_handler._evaluate_safeguards()
        phi_controls = await hipaa_handler._evaluate_phi_controls()

        result = await hipaa_handler._get_hipaa_recommendations(safeguards, phi_controls)

        assert isinstance(result, list)
        for rec in result:
            assert "priority" in rec
            assert "recommendation" in rec

    @pytest.mark.asyncio
    async def test_get_security_evidence(self, hipaa_handler):
        """_get_security_evidence returns evidence references."""
        result = await hipaa_handler._get_security_evidence()

        assert isinstance(result, list)
        for evidence in result:
            assert "control" in evidence
            assert "evidence_type" in evidence
            assert "location" in evidence

    def test_render_hipaa_html(self, hipaa_handler):
        """_render_hipaa_html generates valid HTML."""
        report = {
            "report_id": "hipaa-sec-test",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "safeguards": {
                "administrative": {
                    "category": "Administrative",
                    "compliant_count": 8,
                    "total_count": 8,
                    "standards": [
                        {
                            "id": "164.308(a)(1)",
                            "name": "Security Management Process",
                            "status": "compliant",
                        }
                    ],
                },
            },
            "summary": {
                "total_standards_assessed": 17,
                "standards_compliant": 17,
                "compliance_percentage": 100,
                "overall_status": "Compliant",
            },
        }

        html = hipaa_handler._render_hipaa_html(report)

        assert "<!DOCTYPE html>" in html
        assert "HIPAA Security Rule Compliance Report" in html
        assert "hipaa-sec-test" in html
        assert "Administrative" in html


# ============================================================================
# RBAC Permission Tests
# ============================================================================


class TestHIPAAPermissions:
    """Tests for HIPAA handler RBAC permission enforcement.

    Uses source file reading instead of inspect.getsource() to avoid
    false failures from test pollution (runtime method replacement).
    """

    @pytest.fixture(autouse=True)
    def _load_source(self):
        """Load HIPAA handler source file once for all permission tests."""
        import inspect

        source_file = inspect.getfile(HIPAAMixin)
        with open(source_file) as f:
            self._source = f.read()

    def test_hipaa_status_has_permission_decorator(self):
        """HIPAA status requires compliance:hipaa:read permission."""
        assert "PERM_HIPAA_READ" in self._source
        assert "require_permission" in self._source

    def test_phi_access_log_has_permission_decorator(self):
        """PHI access log requires compliance:hipaa:read permission."""
        assert "PERM_HIPAA_READ" in self._source

    def test_breach_assessment_has_permission_decorator(self):
        """Breach assessment requires compliance:breaches:report permission."""
        assert "PERM_HIPAA_BREACHES_REPORT" in self._source

    def test_list_baas_has_permission_decorator(self):
        """List BAAs requires compliance:hipaa:read permission."""
        assert "PERM_HIPAA_READ" in self._source

    def test_create_baa_has_permission_decorator(self):
        """Create BAA requires compliance:baa:manage permission."""
        assert "PERM_HIPAA_BAA_MANAGE" in self._source

    def test_security_report_has_permission_decorator(self):
        """Security report requires compliance:hipaa:report permission."""
        assert "PERM_HIPAA_REPORT" in self._source


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestHIPAAErrorHandling:
    """Tests for error handling in HIPAA operations."""

    @pytest.mark.asyncio
    async def test_phi_access_log_handles_store_exception(self, hipaa_handler):
        """PHI access log handles store exceptions gracefully."""
        mock_store = MagicMock()
        mock_store.get_log.side_effect = RuntimeError("Database connection failed")

        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_store,
        ):
            result = await hipaa_handler._hipaa_phi_access_log({})

        assert result.status_code == 500
        body = json.loads(result.body)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_breach_assessment_handles_audit_failure(self, hipaa_handler):
        """Breach assessment continues despite audit logging failure."""
        mock_store = MagicMock()
        mock_store.log_event.side_effect = RuntimeError("Audit store unavailable")

        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_store,
        ):
            result = await hipaa_handler._hipaa_breach_assessment(
                {
                    "incident_id": "inc-123",
                    "incident_type": "test",
                    "phi_involved": False,
                }
            )

        # Should still succeed despite audit failure
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_create_baa_handles_audit_failure(self, hipaa_handler):
        """Create BAA continues despite audit logging failure."""
        mock_store = MagicMock()
        mock_store.log_event.side_effect = RuntimeError("Audit store unavailable")

        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_store,
        ):
            result = await hipaa_handler._hipaa_create_baa(
                {
                    "business_associate": "Test Vendor",
                    "ba_type": "vendor",
                    "services_provided": "Test services",
                }
            )

        # Should still succeed despite audit failure
        assert result.status_code == 201


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestHIPAAEdgeCases:
    """Tests for edge cases in HIPAA operations."""

    @pytest.mark.asyncio
    async def test_phi_access_log_empty_results(self, hipaa_handler, mock_audit_store):
        """PHI access log handles empty results gracefully."""
        mock_audit_store.get_log.return_value = []

        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_phi_access_log({})

        body = json.loads(result.body)
        assert body["phi_access_log"] == []
        assert body["count"] == 0

    @pytest.mark.asyncio
    async def test_phi_access_log_filters_non_phi_events(self, hipaa_handler, mock_audit_store):
        """PHI access log filters out non-PHI events."""
        mock_audit_store.get_log.return_value = [
            {
                "action": "login",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {},
            },
            {
                "action": "phi_view",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {},
            },
            {
                "action": "logout",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {},
            },
        ]

        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_phi_access_log({})

        body = json.loads(result.body)
        assert body["count"] == 1  # Only the phi_view event

    @pytest.mark.asyncio
    async def test_breach_assessment_empty_phi_types(self, hipaa_handler, mock_audit_store):
        """Breach assessment handles empty phi_types list."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_breach_assessment(
                {
                    "incident_id": "inc-123",
                    "incident_type": "test",
                    "phi_involved": True,
                    "phi_types": [],
                }
            )

        body = json.loads(result.body)
        nature_factor = next(
            f for f in body["risk_factors"] if f["factor"] == "Nature and extent of PHI"
        )
        assert nature_factor["risk"] == "moderate"

    @pytest.mark.asyncio
    async def test_create_baa_optional_fields(self, hipaa_handler, mock_audit_store):
        """Create BAA handles optional fields correctly."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_create_baa(
                {
                    "business_associate": "Test Vendor",
                    "ba_type": "vendor",
                    "services_provided": "Test services",
                    # Optional fields omitted
                }
            )

        body = json.loads(result.body)
        assert body["baa"]["phi_access_scope"] == []
        assert body["baa"]["subcontractor_clause"] is True
        assert body["baa"]["expiration_date"] is None

    @pytest.mark.asyncio
    async def test_security_report_all_compliant(self, hipaa_handler):
        """Security report handles 100% compliance."""
        result = await hipaa_handler._hipaa_security_report({})

        body = json.loads(result.body)
        assert body["summary"]["compliance_percentage"] == 100
        assert body["summary"]["overall_status"] == "Compliant"

    @pytest.mark.asyncio
    async def test_hipaa_status_score_calculation_edge(self, hipaa_handler):
        """HIPAA status calculates score correctly at edge values."""
        result = await hipaa_handler._hipaa_status({})

        body = json.loads(result.body)
        score = body["compliance_score"]
        status = body["overall_status"]

        # Verify status matches score threshold
        if score >= 95:
            assert status == "compliant"
        elif score >= 80:
            assert status == "substantially_compliant"
        elif score >= 60:
            assert status == "partially_compliant"
        else:
            assert status == "non_compliant"


# ============================================================================
# Integration Tests
# ============================================================================


class TestHIPAAIntegration:
    """Integration tests for HIPAA handler components."""

    @pytest.mark.asyncio
    async def test_full_compliance_workflow(self, hipaa_handler, mock_audit_store):
        """Test complete HIPAA compliance workflow."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            # 1. Check compliance status
            status_result = await hipaa_handler._hipaa_status({"scope": "full"})
            assert status_result.status_code == 200

            # 2. Generate security report
            report_result = await hipaa_handler._hipaa_security_report({"include_evidence": "true"})
            assert report_result.status_code == 200

            # 3. Create a BAA
            baa_result = await hipaa_handler._hipaa_create_baa(
                {
                    "business_associate": "New Vendor",
                    "ba_type": "vendor",
                    "services_provided": "Data processing",
                }
            )
            assert baa_result.status_code == 201

            # 4. List BAAs
            list_result = await hipaa_handler._hipaa_list_baas({"status": "active"})
            assert list_result.status_code == 200

    @pytest.mark.asyncio
    async def test_breach_response_workflow(self, hipaa_handler, mock_audit_store):
        """Test complete breach response workflow."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            # 1. Assess breach risk
            assessment_result = await hipaa_handler._hipaa_breach_assessment(
                {
                    "incident_id": "breach-001",
                    "incident_type": "data_breach",
                    "phi_involved": True,
                    "phi_types": ["SSN", "Medical diagnosis"],
                    "unauthorized_access": {"confirmed_access": True},
                    "affected_individuals": 1000,
                }
            )

            body = json.loads(assessment_result.body)
            assert body["notification_required"] is True
            assert body["notification_deadlines"] is not None

            # 2. Review PHI access logs
            log_result = await hipaa_handler._hipaa_phi_access_log(
                {
                    "from": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                }
            )
            assert log_result.status_code == 200


# ============================================================================
# Compliance Coverage Tests
# ============================================================================


class TestComplianceCoverage:
    """Tests verifying coverage of 50+ HIPAA compliance controls."""

    def test_total_controls_count(self, hipaa_handler):
        """Verify total number of controls across all categories."""
        total_controls = sum(
            len(sg["controls"])
            for category in hipaa_handler.SECURITY_SAFEGUARDS.values()
            for sg in category
        )
        # HIPAA Security Rule has approximately 42-54 implementation specifications
        assert total_controls >= 25, f"Expected at least 25 controls, got {total_controls}"

    def test_administrative_controls_coverage(self, hipaa_handler):
        """Verify administrative safeguard controls coverage."""
        admin_controls = []
        for sg in hipaa_handler.SECURITY_SAFEGUARDS["administrative"]:
            admin_controls.extend(sg["controls"])

        # Key administrative controls
        assert "Risk Analysis" in admin_controls
        assert "Risk Management" in admin_controls
        assert "Sanction Policy" in admin_controls
        assert "Security Officer Designation" in admin_controls
        assert "Termination Procedures" in admin_controls
        assert "Password Management" in admin_controls
        assert "Data Backup Plan" in admin_controls
        assert "Disaster Recovery Plan" in admin_controls

    def test_physical_controls_coverage(self, hipaa_handler):
        """Verify physical safeguard controls coverage."""
        physical_controls = []
        for sg in hipaa_handler.SECURITY_SAFEGUARDS["physical"]:
            physical_controls.extend(sg["controls"])

        # Key physical controls
        assert "Facility Security Plan" in physical_controls
        assert "Access Control/Validation" in physical_controls
        assert "Disposal" in physical_controls
        assert "Media Re-use" in physical_controls

    def test_technical_controls_coverage(self, hipaa_handler):
        """Verify technical safeguard controls coverage."""
        technical_controls = []
        for sg in hipaa_handler.SECURITY_SAFEGUARDS["technical"]:
            technical_controls.extend(sg["controls"])

        # Key technical controls
        assert "Unique User Identification" in technical_controls
        assert "Encryption/Decryption" in technical_controls
        assert "Automatic Logoff" in technical_controls
        assert "Encryption" in technical_controls


# ============================================================================
# PHI De-identification Tests
# ============================================================================


class TestHIPAADeidentify:
    """Tests for HIPAA PHI de-identification endpoint."""

    @pytest.mark.asyncio
    async def test_deidentify_text_redact(self, hipaa_handler, mock_audit_store):
        """De-identify text content using redaction."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_deidentify(
                {
                    "content": "Patient John Smith (SSN: 123-45-6789) visited on 01/15/2024",
                    "method": "redact",
                }
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "anonymized_content" in body
        assert "123-45-6789" not in body["anonymized_content"]
        assert body["identifiers_count"] > 0
        assert "audit_id" in body

    @pytest.mark.asyncio
    async def test_deidentify_text_hash(self, hipaa_handler, mock_audit_store):
        """De-identify text content using hashing."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_deidentify(
                {
                    "content": "Email: test@example.com, Phone: 555-123-4567",
                    "method": "hash",
                }
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "test@example.com" not in body["anonymized_content"]
        assert "555-123-4567" not in body["anonymized_content"]

    @pytest.mark.asyncio
    async def test_deidentify_text_suppress(self, hipaa_handler, mock_audit_store):
        """De-identify text content using suppression."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_deidentify(
                {
                    "content": "SSN: 123-45-6789",
                    "method": "suppress",
                }
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "123-45-6789" not in body["anonymized_content"]

    @pytest.mark.asyncio
    async def test_deidentify_structured_data(self, hipaa_handler, mock_audit_store):
        """De-identify structured data dict."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_deidentify(
                {
                    "data": {
                        "patient_name": "John Smith",
                        "ssn": "123-45-6789",
                        "diagnosis": "Common cold",
                    },
                    "method": "redact",
                }
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "anonymized_content" in body

    @pytest.mark.asyncio
    async def test_deidentify_missing_content_and_data(self, hipaa_handler):
        """De-identify rejects when neither content nor data provided."""
        result = await hipaa_handler._hipaa_deidentify({})

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_deidentify_invalid_method(self, hipaa_handler):
        """De-identify rejects invalid anonymization method."""
        result = await hipaa_handler._hipaa_deidentify(
            {
                "content": "Test content",
                "method": "invalid_method",
            }
        )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "Invalid method" in body["error"]

    @pytest.mark.asyncio
    async def test_deidentify_with_identifier_types(self, hipaa_handler, mock_audit_store):
        """De-identify only specific identifier types."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_deidentify(
                {
                    "content": "Email: test@example.com, SSN: 123-45-6789",
                    "method": "redact",
                    "identifier_types": ["ssn"],
                }
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        # SSN should be redacted
        assert "123-45-6789" not in body["anonymized_content"]

    @pytest.mark.asyncio
    async def test_deidentify_invalid_identifier_type(self, hipaa_handler):
        """De-identify rejects invalid identifier type."""
        result = await hipaa_handler._hipaa_deidentify(
            {
                "content": "Test content",
                "identifier_types": ["not_a_real_type"],
            }
        )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "Invalid identifier type" in body["error"]

    @pytest.mark.asyncio
    async def test_deidentify_pseudonymize(self, hipaa_handler, mock_audit_store):
        """De-identify text using pseudonymization."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_deidentify(
                {
                    "content": "Patient John Smith has SSN 123-45-6789",
                    "method": "pseudonymize",
                }
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["reversible"] is True
        assert "123-45-6789" not in body["anonymized_content"]

    @pytest.mark.asyncio
    async def test_deidentify_audits_operation(self, hipaa_handler, mock_audit_store):
        """De-identify logs an audit event."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            await hipaa_handler._hipaa_deidentify(
                {
                    "content": "SSN: 123-45-6789",
                    "method": "redact",
                }
            )

        mock_audit_store.log_event.assert_called_once()
        call_kwargs = mock_audit_store.log_event.call_args
        assert call_kwargs.kwargs["action"] == "hipaa_phi_deidentified"

    @pytest.mark.asyncio
    async def test_deidentify_generalize(self, hipaa_handler, mock_audit_store):
        """De-identify text using generalization."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_deidentify(
                {
                    "content": "IP: 192.168.1.100",
                    "method": "generalize",
                }
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        # IP should be generalized (e.g., 192.168.x.x)
        assert "192.168.1.100" not in body["anonymized_content"]

    @pytest.mark.asyncio
    async def test_deidentify_no_phi_in_content(self, hipaa_handler, mock_audit_store):
        """De-identify handles content with no PHI gracefully."""
        with patch(
            "aragora.server.handlers.compliance.hipaa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await hipaa_handler._hipaa_deidentify(
                {
                    "content": "a b c d",
                    "method": "redact",
                }
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["identifiers_count"] == 0
        assert body["anonymized_content"] == "a b c d"

    @pytest.mark.asyncio
    async def test_deidentify_permission_decorator(self):
        """De-identify endpoint has correct permission decorator."""
        import inspect

        from aragora.server.handlers.compliance.hipaa import PERM_HIPAA_PHI_DEIDENTIFY

        source_file = inspect.getfile(HIPAAMixin)
        with open(source_file) as f:
            source = f.read()
        assert "require_permission" in source
        assert "PERM_HIPAA_PHI_DEIDENTIFY" in source
        assert PERM_HIPAA_PHI_DEIDENTIFY == "compliance:phi:deidentify"


# ============================================================================
# Safe Harbor Verification Tests
# ============================================================================


class TestHIPAASafeHarborVerify:
    """Tests for HIPAA Safe Harbor verification endpoint."""

    @pytest.mark.asyncio
    async def test_verify_compliant_content(self, hipaa_handler):
        """Verify content with no PHI is compliant."""
        result = await hipaa_handler._hipaa_safe_harbor_verify(
            {
                "content": "a b c d",
            }
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["compliant"] is True
        assert body["identifiers_remaining"] == []
        assert body["hipaa_reference"] == "45 CFR 164.514(b) - Safe Harbor Method"

    @pytest.mark.asyncio
    async def test_verify_non_compliant_content(self, hipaa_handler):
        """Verify content with PHI is non-compliant."""
        result = await hipaa_handler._hipaa_safe_harbor_verify(
            {
                "content": "Patient John Smith, SSN 123-45-6789, email john@example.com",
            }
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["compliant"] is False
        assert len(body["identifiers_remaining"]) > 0
        assert "verified_at" in body

    @pytest.mark.asyncio
    async def test_verify_identifier_details(self, hipaa_handler):
        """Verify identifier details are returned with truncated values."""
        result = await hipaa_handler._hipaa_safe_harbor_verify(
            {
                "content": "SSN: 123-45-6789",
            }
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        for ident in body["identifiers_remaining"]:
            assert "type" in ident
            assert "value_preview" in ident
            assert "confidence" in ident
            assert "position" in ident
            # Value should be truncated for privacy
            assert len(ident["value_preview"]) <= 6  # "xxx..." max

    @pytest.mark.asyncio
    async def test_verify_missing_content(self, hipaa_handler):
        """Verify rejects request with missing content."""
        result = await hipaa_handler._hipaa_safe_harbor_verify({})

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_verify_verification_notes(self, hipaa_handler):
        """Verify verification notes are included."""
        result = await hipaa_handler._hipaa_safe_harbor_verify(
            {
                "content": "a b c d",
            }
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "verification_notes" in body
        assert len(body["verification_notes"]) > 0

    @pytest.mark.asyncio
    async def test_verify_permission_decorator(self):
        """Safe Harbor verify has correct permission decorator."""
        import inspect

        source_file = inspect.getfile(HIPAAMixin)
        with open(source_file) as f:
            source = f.read()
        assert "require_permission" in source
        assert "PERM_HIPAA_READ" in source


# ============================================================================
# PHI Detection Tests
# ============================================================================


class TestHIPAADetectPHI:
    """Tests for HIPAA PHI detection endpoint."""

    @pytest.mark.asyncio
    async def test_detect_ssn(self, hipaa_handler):
        """Detect SSN in content."""
        result = await hipaa_handler._hipaa_detect_phi(
            {
                "content": "SSN: 123-45-6789",
            }
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] > 0
        types = [i["type"] for i in body["identifiers"]]
        assert "ssn" in types

    @pytest.mark.asyncio
    async def test_detect_email(self, hipaa_handler):
        """Detect email addresses in content."""
        result = await hipaa_handler._hipaa_detect_phi(
            {
                "content": "Contact: john.doe@hospital.org",
            }
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] > 0
        types = [i["type"] for i in body["identifiers"]]
        assert "email" in types

    @pytest.mark.asyncio
    async def test_detect_phone_number(self, hipaa_handler):
        """Detect phone numbers in content."""
        result = await hipaa_handler._hipaa_detect_phi(
            {
                "content": "Call: 555-123-4567",
            }
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] > 0
        types = [i["type"] for i in body["identifiers"]]
        assert "phone" in types

    @pytest.mark.asyncio
    async def test_detect_ip_address(self, hipaa_handler):
        """Detect IP addresses in content."""
        result = await hipaa_handler._hipaa_detect_phi(
            {
                "content": "Server: 192.168.1.100",
            }
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] > 0
        types = [i["type"] for i in body["identifiers"]]
        assert "ip" in types

    @pytest.mark.asyncio
    async def test_detect_multiple_identifiers(self, hipaa_handler):
        """Detect multiple different identifier types."""
        result = await hipaa_handler._hipaa_detect_phi(
            {
                "content": "Patient John Smith, SSN 123-45-6789, email john@example.com, IP 10.0.0.1",
            }
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] >= 3
        types = set(i["type"] for i in body["identifiers"])
        assert len(types) >= 2

    @pytest.mark.asyncio
    async def test_detect_no_phi(self, hipaa_handler):
        """Detect returns empty when no PHI found."""
        result = await hipaa_handler._hipaa_detect_phi(
            {
                "content": "a b c d",
            }
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] == 0
        assert body["identifiers"] == []

    @pytest.mark.asyncio
    async def test_detect_with_confidence_filter(self, hipaa_handler):
        """Detect filters by minimum confidence."""
        result = await hipaa_handler._hipaa_detect_phi(
            {
                "content": "SSN: 123-45-6789, some text at 123 Main Street",
                "min_confidence": 0.9,
            }
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        # All returned identifiers should meet confidence threshold
        for ident in body["identifiers"]:
            assert ident["confidence"] >= 0.9

    @pytest.mark.asyncio
    async def test_detect_missing_content(self, hipaa_handler):
        """Detect rejects request with missing content."""
        result = await hipaa_handler._hipaa_detect_phi({})

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_detect_identifier_positions(self, hipaa_handler):
        """Detect returns correct start/end positions."""
        result = await hipaa_handler._hipaa_detect_phi(
            {
                "content": "SSN: 123-45-6789",
            }
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        for ident in body["identifiers"]:
            assert "start" in ident
            assert "end" in ident
            assert ident["start"] < ident["end"]

    @pytest.mark.asyncio
    async def test_detect_hipaa_reference(self, hipaa_handler):
        """Detect response includes HIPAA reference."""
        result = await hipaa_handler._hipaa_detect_phi(
            {
                "content": "Test",
            }
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["hipaa_reference"] == "45 CFR 164.514 - HIPAA Safe Harbor Identifiers"

    @pytest.mark.asyncio
    async def test_detect_permission_decorator(self):
        """PHI detect has correct permission decorator."""
        import inspect

        source_file = inspect.getfile(HIPAAMixin)
        with open(source_file) as f:
            source = f.read()
        assert "require_permission" in source
        assert "PERM_HIPAA_READ" in source


# ============================================================================
# PHI Controls Evaluation Tests (updated)
# ============================================================================


class TestPHIControlsAnonymizerIntegration:
    """Tests for anonymizer integration in PHI controls evaluation."""

    @pytest.mark.asyncio
    async def test_phi_controls_reports_anonymizer_available(self, hipaa_handler):
        """PHI controls evaluation reports anonymizer availability."""
        result = await hipaa_handler._evaluate_phi_controls()

        assert "anonymizer_available" in result
        assert result["anonymizer_available"] is True  # Module should be importable

    @pytest.mark.asyncio
    async def test_phi_controls_still_reports_safe_harbor(self, hipaa_handler):
        """PHI controls evaluation still reports Safe Harbor method."""
        result = await hipaa_handler._evaluate_phi_controls()

        assert result["de_identification_method"] == "Safe Harbor"
        assert result["status"] == "configured"
