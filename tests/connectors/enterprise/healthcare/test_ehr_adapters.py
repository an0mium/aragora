"""
Tests for EHR (Epic/Cerner) Adapters.

Tests cover:
- EHR vendor detection
- Configuration and connection setup
- SMART on FHIR authentication helpers
- Epic-specific features
- Cerner-specific features
"""

from datetime import datetime, timezone, timedelta

import pytest


# =============================================================================
# EHR Vendor Tests
# =============================================================================


class TestEHRVendor:
    """Tests for EHR vendor enum."""

    def test_vendor_values(self):
        """Vendor enum has expected values."""
        from aragora.connectors.enterprise.healthcare.ehr import EHRVendor

        assert EHRVendor.EPIC.value == "epic"
        assert EHRVendor.CERNER.value == "cerner"
        assert EHRVendor.ALLSCRIPTS.value == "allscripts"
        assert EHRVendor.MEDITECH.value == "meditech"
        assert EHRVendor.ATHENAHEALTH.value == "athenahealth"
        assert EHRVendor.UNKNOWN.value == "unknown"


class TestEHRCapability:
    """Tests for EHR capability enum."""

    def test_capability_values(self):
        """Capability enum has expected values."""
        from aragora.connectors.enterprise.healthcare.ehr import EHRCapability

        # Authentication
        assert EHRCapability.SMART_ON_FHIR.value == "smart_on_fhir"
        assert EHRCapability.BACKEND_SERVICES.value == "backend_services"

        # FHIR versions
        assert EHRCapability.FHIR_R4.value == "fhir_r4"
        assert EHRCapability.FHIR_STU3.value == "fhir_stu3"

        # Vendor-specific
        assert EHRCapability.EPIC_MYCHART.value == "epic_mychart"
        assert EHRCapability.CERNER_MILLENNIUM.value == "cerner_millennium"


# =============================================================================
# Vendor Detection Tests
# =============================================================================


class TestVendorDetection:
    """Tests for automatic vendor detection."""

    def test_detect_epic_from_url(self):
        """Detect Epic from URL patterns."""
        from aragora.connectors.enterprise.healthcare.ehr.base import detect_vendor, EHRVendor

        assert (
            detect_vendor("https://fhir.epic.com/interconnect-fhir-oauth/api/FHIR/R4")
            == EHRVendor.EPIC
        )
        assert detect_vendor("https://apporchard.epic.com/fhir") == EHRVendor.EPIC
        assert detect_vendor("https://mychart.hospital.org/api") == EHRVendor.EPIC

    def test_detect_cerner_from_url(self):
        """Detect Cerner from URL patterns."""
        from aragora.connectors.enterprise.healthcare.ehr.base import detect_vendor, EHRVendor

        assert detect_vendor("https://fhir-ehr.cerner.com/r4/organization") == EHRVendor.CERNER
        assert detect_vendor("https://cernercentral.com/fhir") == EHRVendor.CERNER
        assert detect_vendor("https://millennium.hospital.org/fhir") == EHRVendor.CERNER

    def test_detect_unknown_from_url(self):
        """Return UNKNOWN for unrecognized URLs."""
        from aragora.connectors.enterprise.healthcare.ehr.base import detect_vendor, EHRVendor

        assert detect_vendor("https://generic-fhir-server.com/api") == EHRVendor.UNKNOWN

    def test_detect_from_metadata(self):
        """Detect vendor from FHIR metadata."""
        from aragora.connectors.enterprise.healthcare.ehr.base import detect_vendor, EHRVendor

        epic_metadata = {"software": {"name": "Epic Systems FHIR Server", "version": "1.0"}}
        assert detect_vendor("https://example.com", epic_metadata) == EHRVendor.EPIC

        cerner_metadata = {"software": {"name": "Cerner Millennium FHIR R4", "version": "2.0"}}
        assert detect_vendor("https://example.com", cerner_metadata) == EHRVendor.CERNER


# =============================================================================
# Configuration Tests
# =============================================================================


class TestEHRConnectionConfig:
    """Tests for EHR connection configuration."""

    def test_basic_config(self):
        """Create basic config."""
        from aragora.connectors.enterprise.healthcare.ehr import EHRConnectionConfig, EHRVendor

        config = EHRConnectionConfig(
            vendor=EHRVendor.EPIC,
            base_url="https://fhir.epic.com/api/FHIR/R4",
            organization_id="org123",
            client_id="client123",
        )

        assert config.vendor == EHRVendor.EPIC
        assert config.base_url == "https://fhir.epic.com/api/FHIR/R4"
        assert config.organization_id == "org123"
        assert config.client_id == "client123"
        assert config.enable_phi_redaction is True
        assert config.audit_all_access is True

    def test_config_with_backend_services(self):
        """Create config for backend services auth."""
        from aragora.connectors.enterprise.healthcare.ehr import EHRConnectionConfig, EHRVendor

        config = EHRConnectionConfig(
            vendor=EHRVendor.CERNER,
            base_url="https://fhir-ehr.cerner.com/r4",
            organization_id="org456",
            client_id="client456",
            private_key="-----BEGIN RSA PRIVATE KEY-----\n...",
            key_id="key123",
        )

        assert config.private_key is not None
        assert config.key_id == "key123"

    def test_default_scopes(self):
        """Config has default scopes."""
        from aragora.connectors.enterprise.healthcare.ehr import EHRConnectionConfig, EHRVendor

        config = EHRConnectionConfig(
            vendor=EHRVendor.EPIC,
            base_url="https://example.com",
            organization_id="org",
            client_id="client",
        )

        assert len(config.scopes) > 0
        assert "patient/*.read" in config.scopes


# =============================================================================
# Token Response Tests
# =============================================================================


class TestTokenResponse:
    """Tests for token response handling."""

    def test_token_from_dict(self):
        """Parse token from OAuth response."""
        from aragora.connectors.enterprise.healthcare.ehr.base import TokenResponse

        data = {
            "access_token": "abc123",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "refresh456",
            "scope": "patient/*.read",
            "patient": "patient789",
        }

        token = TokenResponse.from_dict(data)

        assert token.access_token == "abc123"
        assert token.token_type == "Bearer"
        assert token.expires_in == 3600
        assert token.refresh_token == "refresh456"
        assert token.patient == "patient789"

    def test_token_is_expired(self):
        """Check token expiration."""
        from aragora.connectors.enterprise.healthcare.ehr.base import TokenResponse

        # Create token that expires in 30 seconds (within 60-second buffer)
        token = TokenResponse(
            access_token="test",
            expires_in=30,
            obtained_at=datetime.now(timezone.utc),
        )
        assert token.is_expired is True

        # Create token that expires in 2 hours
        token = TokenResponse(
            access_token="test",
            expires_in=7200,
            obtained_at=datetime.now(timezone.utc),
        )
        assert token.is_expired is False


# =============================================================================
# SMART Configuration Tests
# =============================================================================


class TestSMARTConfiguration:
    """Tests for SMART on FHIR configuration."""

    def test_parse_smart_config(self):
        """Parse SMART discovery document."""
        from aragora.connectors.enterprise.healthcare.ehr.base import SMARTConfiguration

        data = {
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
            "introspection_endpoint": "https://auth.example.com/introspect",
            "scopes_supported": ["patient/*.read", "launch/patient"],
            "capabilities": ["launch-ehr", "client-confidential-asymmetric"],
            "code_challenge_methods_supported": ["S256"],
        }

        config = SMARTConfiguration.from_dict(data)

        assert config.authorization_endpoint == "https://auth.example.com/authorize"
        assert config.token_endpoint == "https://auth.example.com/token"
        assert "patient/*.read" in config.scopes_supported
        assert "S256" in config.code_challenge_methods_supported


# =============================================================================
# Epic Adapter Tests
# =============================================================================


class TestEpicAdapter:
    """Tests for Epic adapter."""

    def test_epic_adapter_vendor(self):
        """Epic adapter has correct vendor."""
        from aragora.connectors.enterprise.healthcare.ehr import (
            EpicAdapter,
            EHRConnectionConfig,
            EHRVendor,
            EHRCapability,
        )

        config = EHRConnectionConfig(
            vendor=EHRVendor.EPIC,
            base_url="https://fhir.epic.com/api/FHIR/R4",
            organization_id="org123",
            client_id="client123",
        )

        adapter = EpicAdapter(config)

        assert adapter.vendor == EHRVendor.EPIC
        assert EHRCapability.EPIC_MYCHART in adapter.capabilities
        assert EHRCapability.EPIC_CARE_EVERYWHERE in adapter.capabilities
        assert EHRCapability.SMART_ON_FHIR in adapter.capabilities
        assert EHRCapability.FHIR_R4 in adapter.capabilities

    def test_epic_default_scopes(self):
        """Epic adapter includes Epic-specific scopes."""
        from aragora.connectors.enterprise.healthcare.ehr import (
            EpicAdapter,
            EHRConnectionConfig,
            EHRVendor,
        )

        config = EHRConnectionConfig(
            vendor=EHRVendor.EPIC,
            base_url="https://fhir.epic.com/api/FHIR/R4",
            organization_id="org123",
            client_id="client123",
            scopes=[],  # Empty to use defaults
        )

        adapter = EpicAdapter(config)

        assert "launch/patient" in adapter.config.scopes
        assert "online_access" in adapter.config.scopes

    def test_epic_stats(self):
        """Epic adapter provides stats."""
        from aragora.connectors.enterprise.healthcare.ehr import (
            EpicAdapter,
            EHRConnectionConfig,
            EHRVendor,
        )

        config = EHRConnectionConfig(
            vendor=EHRVendor.EPIC,
            base_url="https://fhir.epic.com/api/FHIR/R4",
            organization_id="org123",
            client_id="client123",
        )

        adapter = EpicAdapter(config)
        stats = adapter.get_stats()

        assert stats["vendor"] == "epic"
        assert stats["requests_made"] == 0
        assert stats["is_authenticated"] is False


# =============================================================================
# Cerner Adapter Tests
# =============================================================================


class TestCernerAdapter:
    """Tests for Cerner adapter."""

    def test_cerner_adapter_vendor(self):
        """Cerner adapter has correct vendor."""
        from aragora.connectors.enterprise.healthcare.ehr import (
            CernerAdapter,
            EHRConnectionConfig,
            EHRVendor,
            EHRCapability,
        )

        config = EHRConnectionConfig(
            vendor=EHRVendor.CERNER,
            base_url="https://fhir-ehr.cerner.com/r4",
            organization_id="org456",
            client_id="client456",
        )

        adapter = CernerAdapter(config)

        assert adapter.vendor == EHRVendor.CERNER
        assert EHRCapability.CERNER_MILLENNIUM in adapter.capabilities
        assert EHRCapability.CERNER_POWERCHART in adapter.capabilities
        assert EHRCapability.SUBSCRIPTIONS in adapter.capabilities
        assert EHRCapability.SMART_ON_FHIR in adapter.capabilities

    def test_cerner_default_scopes(self):
        """Cerner adapter includes Cerner-specific scopes."""
        from aragora.connectors.enterprise.healthcare.ehr import (
            CernerAdapter,
            EHRConnectionConfig,
            EHRVendor,
        )

        config = EHRConnectionConfig(
            vendor=EHRVendor.CERNER,
            base_url="https://fhir-ehr.cerner.com/r4",
            organization_id="org456",
            client_id="client456",
            scopes=[],  # Empty to use defaults
        )

        adapter = CernerAdapter(config)

        assert "fhirUser" in adapter.config.scopes
        assert "system/Patient.read" in adapter.config.scopes
        assert "patient/CarePlan.read" in adapter.config.scopes

    def test_cerner_identifier_systems(self):
        """Cerner adapter knows identifier systems."""
        from aragora.connectors.enterprise.healthcare.ehr.cerner import CERNER_IDENTIFIER_SYSTEMS

        assert "federated_person_principal" in CERNER_IDENTIFIER_SYSTEMS
        assert "mrn" in CERNER_IDENTIFIER_SYSTEMS
        assert "npi" in CERNER_IDENTIFIER_SYSTEMS

    def test_cerner_stats(self):
        """Cerner adapter provides stats."""
        from aragora.connectors.enterprise.healthcare.ehr import (
            CernerAdapter,
            EHRConnectionConfig,
            EHRVendor,
        )

        config = EHRConnectionConfig(
            vendor=EHRVendor.CERNER,
            base_url="https://fhir-ehr.cerner.com/r4",
            organization_id="org456",
            client_id="client456",
        )

        adapter = CernerAdapter(config)
        stats = adapter.get_stats()

        assert stats["vendor"] == "cerner"
        assert stats["requests_made"] == 0
        assert "cerner_millennium" in stats["capabilities"]


# =============================================================================
# Adapter Factory Tests
# =============================================================================


class TestAdapterFactory:
    """Tests for adapter factory function."""

    def test_create_epic_adapter(self):
        """Factory creates Epic adapter."""
        from aragora.connectors.enterprise.healthcare.ehr.base import create_adapter
        from aragora.connectors.enterprise.healthcare.ehr import (
            EpicAdapter,
            EHRConnectionConfig,
            EHRVendor,
        )

        config = EHRConnectionConfig(
            vendor=EHRVendor.EPIC,
            base_url="https://fhir.epic.com/api/FHIR/R4",
            organization_id="org123",
            client_id="client123",
        )

        adapter = create_adapter(config)

        assert isinstance(adapter, EpicAdapter)

    def test_create_cerner_adapter(self):
        """Factory creates Cerner adapter."""
        from aragora.connectors.enterprise.healthcare.ehr.base import create_adapter
        from aragora.connectors.enterprise.healthcare.ehr import (
            CernerAdapter,
            EHRConnectionConfig,
            EHRVendor,
        )

        config = EHRConnectionConfig(
            vendor=EHRVendor.CERNER,
            base_url="https://fhir-ehr.cerner.com/r4",
            organization_id="org456",
            client_id="client456",
        )

        adapter = create_adapter(config)

        assert isinstance(adapter, CernerAdapter)

    def test_create_unsupported_vendor(self):
        """Factory raises for unsupported vendor."""
        from aragora.connectors.enterprise.healthcare.ehr.base import create_adapter
        from aragora.connectors.enterprise.healthcare.ehr import (
            EHRConnectionConfig,
            EHRVendor,
        )

        config = EHRConnectionConfig(
            vendor=EHRVendor.MEDITECH,  # Not implemented
            base_url="https://meditech.example.com",
            organization_id="org789",
            client_id="client789",
        )

        with pytest.raises(ValueError, match="Unsupported EHR vendor"):
            create_adapter(config)


# =============================================================================
# Integration Scenarios (Mocked)
# =============================================================================


class TestEHRScenarios:
    """Integration scenarios for EHR adapters (no actual network calls)."""

    def test_epic_patient_context(self):
        """Epic patient context dataclass."""
        from aragora.connectors.enterprise.healthcare.ehr.epic import EpicPatientContext

        ctx = EpicPatientContext(
            patient_id="pat123",
            fhir_id="fhir456",
            mychart_status="active",
            mrn="MRN789",
            encounter_id="enc001",
        )

        assert ctx.patient_id == "pat123"
        assert ctx.mychart_status == "active"
        assert ctx.mrn == "MRN789"

    def test_cerner_patient_context(self):
        """Cerner patient context dataclass."""
        from aragora.connectors.enterprise.healthcare.ehr.cerner import CernerPatientContext

        ctx = CernerPatientContext(
            patient_id="pat456",
            fhir_id="fhir789",
            federated_id="fed001",
            mrn="CMRN123",
            organization_id="org001",
        )

        assert ctx.patient_id == "pat456"
        assert ctx.federated_id == "fed001"
        assert ctx.organization_id == "org001"
