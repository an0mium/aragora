"""
Tests for FHIR (Fast Healthcare Interoperability Resources) Enterprise Connector.
Tests the healthcare data integration including:
- SMART on FHIR OAuth2 authentication
- Resource type sync (Patient, Observation, Condition, etc.)
- PHI redaction and de-identification
- Incremental sync via _lastUpdated
- FHIR search parameters
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, AsyncIterator

from aragora.connectors.enterprise.base import SyncState, SyncStatus


# =============================================================================
# Mock FHIR Connector (mirrors actual implementation)
# =============================================================================


@dataclass
class FHIRResource:
    """Represents a FHIR resource."""

    id: str
    resource_type: str
    data: Dict[str, Any]
    last_updated: Optional[datetime] = None
    version_id: Optional[str] = None

    @property
    def reference(self) -> str:
        return f"{self.resource_type}/{self.id}"


class MockFHIRConnector:
    """Mock FHIR connector for testing."""

    ROUTES = ["/api/connectors/fhir/sync", "/api/connectors/fhir/search"]

    # PHI fields that should be redacted
    PHI_FIELDS = [
        "name",
        "address",
        "telecom",
        "birthDate",
        "identifier",
        "photo",
        "contact",
    ]

    def __init__(
        self,
        fhir_base_url: str,
        client_id: str = "",
        client_secret: str = "",
        resource_types: List[str] = None,
        redact_phi: bool = True,
        **kwargs,
    ):
        self.connector_id = f"fhir_{fhir_base_url.replace('https://', '').replace('/', '_')}"
        self.fhir_base_url = fhir_base_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.resource_types = resource_types or ["Patient", "Observation", "Condition"]
        self.redact_phi = redact_phi

        self._access_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None
        self._resources_cache: Dict[str, FHIRResource] = {}

    @property
    def name(self) -> str:
        return f"FHIR: {self.fhir_base_url}"

    @property
    def source_type(self) -> str:
        return "healthcare"

    async def _authenticate(self) -> str:
        """Get SMART on FHIR access token."""
        now = datetime.now(timezone.utc)

        if self._access_token and self._token_expires and now < self._token_expires:
            return self._access_token

        # Simulate token fetch
        self._access_token = "fhir_access_token"
        self._token_expires = now + timedelta(hours=1)
        return self._access_token

    async def _fhir_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Make authenticated FHIR API request."""
        await self._authenticate()
        # This would be mocked in tests
        raise NotImplementedError("Mock this method")

    def _redact_phi(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Redact PHI from resource."""
        if not self.redact_phi:
            return resource

        redacted = resource.copy()

        for phi_field in self.PHI_FIELDS:
            if phi_field in redacted:
                redacted[phi_field] = "[REDACTED]"

        return redacted

    def _extract_resource_text(self, resource: Dict[str, Any]) -> str:
        """Extract human-readable text from resource."""
        parts = []

        # Get narrative text if available
        if "text" in resource and "div" in resource["text"]:
            parts.append(resource["text"]["div"])

        # Extract key fields based on resource type
        resource_type = resource.get("resourceType", "")

        if resource_type == "Patient":
            if "name" in resource and not self.redact_phi:
                names = resource["name"]
                if names:
                    name = names[0]
                    full_name = f"{name.get('given', [''])[0]} {name.get('family', '')}"
                    parts.append(f"Patient: {full_name}")

        elif resource_type == "Observation":
            if "code" in resource:
                code = resource["code"]
                if "text" in code:
                    parts.append(f"Observation: {code['text']}")
            if "valueQuantity" in resource:
                vq = resource["valueQuantity"]
                parts.append(f"Value: {vq.get('value')} {vq.get('unit', '')}")

        elif resource_type == "Condition":
            if "code" in resource:
                code = resource["code"]
                if "text" in code:
                    parts.append(f"Condition: {code['text']}")

        return "\n".join(parts)

    async def _search_resources(
        self,
        resource_type: str,
        params: Dict[str, Any] = None,
    ) -> tuple[List[FHIRResource], Optional[str]]:
        """Search for resources of a given type."""
        params = params or {}
        endpoint = f"/{resource_type}"

        response = await self._fhir_request(endpoint, params=params)

        resources = []
        for entry in response.get("entry", []):
            resource_data = entry.get("resource", {})
            resource = FHIRResource(
                id=resource_data.get("id", ""),
                resource_type=resource_data.get("resourceType", resource_type),
                data=resource_data,
                last_updated=self._parse_datetime(resource_data.get("meta", {}).get("lastUpdated")),
                version_id=resource_data.get("meta", {}).get("versionId"),
            )
            resources.append(resource)
            self._resources_cache[resource.reference] = resource

        # Get next page link
        next_link = None
        for link in response.get("link", []):
            if link.get("relation") == "next":
                next_link = link.get("url")
                break

        return resources, next_link

    def _parse_datetime(self, dt_str: Optional[str]) -> Optional[datetime]:
        """Parse FHIR datetime string."""
        if not dt_str:
            return None
        try:
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except ValueError:
            return None

    async def sync_items(
        self,
        state: SyncState,
        batch_size: int = 100,
    ) -> AsyncIterator[Any]:
        """Sync FHIR resources."""
        # Parse last sync time from cursor
        last_updated = None
        if state.cursor:
            try:
                last_updated = datetime.fromisoformat(state.cursor)
            except ValueError:
                pass

        items_yielded = 0

        for resource_type in self.resource_types:
            params = {"_count": str(batch_size)}

            if last_updated:
                params["_lastUpdated"] = f"gt{last_updated.isoformat()}"

            resources, _ = await self._search_resources(resource_type, params)

            for resource in resources:
                # Redact PHI
                redacted_data = self._redact_phi(resource.data)

                # Extract text content
                content = self._extract_resource_text(redacted_data)

                yield MagicMock(
                    id=f"fhir-{resource.reference}",
                    content=content[:50000],
                    source_type="healthcare",
                    source_id=f"fhir://{self.fhir_base_url}/{resource.reference}",
                    title=f"{resource.resource_type} {resource.id}",
                    url=f"{self.fhir_base_url}/{resource.reference}",
                    created_at=resource.last_updated,
                    updated_at=resource.last_updated,
                    metadata={
                        "resource_type": resource.resource_type,
                        "resource_id": resource.id,
                        "version_id": resource.version_id,
                        "phi_redacted": self.redact_phi,
                    },
                )

                items_yielded += 1

                # Update cursor
                if resource.last_updated:
                    ts = resource.last_updated.isoformat()
                    if not state.cursor or ts > state.cursor:
                        state.cursor = ts

                if items_yielded >= batch_size:
                    return

    async def fetch(self, item_id: str) -> Optional[Any]:
        """Fetch a specific resource."""
        # Parse item ID
        if item_id.startswith("fhir-"):
            reference = item_id[5:]
        else:
            reference = item_id

        parts = reference.split("/")
        if len(parts) != 2:
            return None

        resource_type, resource_id = parts

        try:
            response = await self._fhir_request(f"/{resource_type}/{resource_id}")
        except Exception:
            return None

        resource = FHIRResource(
            id=response.get("id", resource_id),
            resource_type=response.get("resourceType", resource_type),
            data=response,
            last_updated=self._parse_datetime(response.get("meta", {}).get("lastUpdated")),
        )

        redacted_data = self._redact_phi(resource.data)
        content = self._extract_resource_text(redacted_data)

        return MagicMock(
            id=f"fhir-{resource.reference}",
            content=content,
            title=f"{resource.resource_type} {resource.id}",
            metadata={
                "resource_type": resource.resource_type,
                "phi_redacted": self.redact_phi,
            },
        )

    async def search(self, query: str, limit: int = 10) -> List[Any]:
        """Search across resources."""
        results = []

        for resource_type in self.resource_types:
            params = {"_content": query, "_count": str(limit)}
            resources, _ = await self._search_resources(resource_type, params)

            for resource in resources:
                results.append(
                    MagicMock(
                        id=f"fhir-{resource.reference}",
                        title=f"{resource.resource_type} {resource.id}",
                        url=f"{self.fhir_base_url}/{resource.reference}",
                        score=1.0,
                    )
                )

            if len(results) >= limit:
                break

        return results[:limit]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fhir_connector():
    """Create FHIR connector for testing."""
    return MockFHIRConnector(
        fhir_base_url="https://fhir.example.com/r4",
        client_id="client123",
        client_secret="secret",
        resource_types=["Patient", "Observation", "Condition"],
        redact_phi=True,
    )


@pytest.fixture
def sample_patient():
    """Sample Patient resource."""
    return {
        "resourceType": "Patient",
        "id": "patient-001",
        "meta": {
            "lastUpdated": "2024-01-15T10:00:00Z",
            "versionId": "1",
        },
        "name": [{"family": "Smith", "given": ["John"]}],
        "birthDate": "1980-01-15",
        "address": [{"city": "Boston", "state": "MA"}],
        "identifier": [{"system": "mrn", "value": "12345"}],
    }


@pytest.fixture
def sample_observation():
    """Sample Observation resource."""
    return {
        "resourceType": "Observation",
        "id": "obs-001",
        "meta": {
            "lastUpdated": "2024-01-16T14:00:00Z",
            "versionId": "1",
        },
        "status": "final",
        "code": {
            "coding": [{"system": "http://loinc.org", "code": "8867-4"}],
            "text": "Heart rate",
        },
        "subject": {"reference": "Patient/patient-001"},
        "valueQuantity": {
            "value": 72,
            "unit": "beats/minute",
        },
    }


@pytest.fixture
def sample_condition():
    """Sample Condition resource."""
    return {
        "resourceType": "Condition",
        "id": "cond-001",
        "meta": {
            "lastUpdated": "2024-01-17T09:00:00Z",
            "versionId": "1",
        },
        "clinicalStatus": {
            "coding": [{"code": "active"}],
        },
        "code": {
            "coding": [{"system": "http://snomed.info/sct", "code": "38341003"}],
            "text": "Hypertension",
        },
        "subject": {"reference": "Patient/patient-001"},
    }


# =============================================================================
# Test Classes
# =============================================================================


class TestFHIRInit:
    """Test FHIR connector initialization."""

    def test_connector_initialization(self, fhir_connector):
        """Test basic initialization."""
        assert fhir_connector.fhir_base_url == "https://fhir.example.com/r4"
        assert fhir_connector.client_id == "client123"
        assert fhir_connector.redact_phi is True
        assert "Patient" in fhir_connector.resource_types

    def test_name_property(self, fhir_connector):
        """Test name property."""
        assert "fhir.example.com" in fhir_connector.name

    def test_source_type(self, fhir_connector):
        """Test source type."""
        assert fhir_connector.source_type == "healthcare"

    def test_default_resource_types(self):
        """Test default resource types."""
        connector = MockFHIRConnector(fhir_base_url="https://fhir.test.com")
        assert "Patient" in connector.resource_types
        assert "Observation" in connector.resource_types


class TestAuthentication:
    """Test SMART on FHIR authentication."""

    @pytest.mark.asyncio
    async def test_authenticate_gets_token(self, fhir_connector):
        """Test token acquisition."""
        token = await fhir_connector._authenticate()

        assert token == "fhir_access_token"
        assert fhir_connector._token_expires is not None

    @pytest.mark.asyncio
    async def test_authenticate_reuses_valid_token(self, fhir_connector):
        """Test token reuse when still valid."""
        # First call
        token1 = await fhir_connector._authenticate()

        # Second call should reuse token
        token2 = await fhir_connector._authenticate()

        assert token1 == token2

    @pytest.mark.asyncio
    async def test_authenticate_refreshes_expired_token(self, fhir_connector):
        """Test token refresh when expired."""
        # Set expired token
        fhir_connector._access_token = "old_token"
        fhir_connector._token_expires = datetime.now(timezone.utc) - timedelta(hours=1)

        token = await fhir_connector._authenticate()

        assert token == "fhir_access_token"


class TestPHIRedaction:
    """Test PHI redaction functionality."""

    def test_redact_phi_enabled(self, fhir_connector, sample_patient):
        """Test PHI redaction when enabled."""
        redacted = fhir_connector._redact_phi(sample_patient)

        assert redacted["name"] == "[REDACTED]"
        assert redacted["birthDate"] == "[REDACTED]"
        assert redacted["address"] == "[REDACTED]"
        assert redacted["identifier"] == "[REDACTED]"

    def test_redact_phi_disabled(self, sample_patient):
        """Test PHI not redacted when disabled."""
        connector = MockFHIRConnector(
            fhir_base_url="https://fhir.test.com",
            redact_phi=False,
        )

        redacted = connector._redact_phi(sample_patient)

        assert redacted["name"] == sample_patient["name"]
        assert redacted["birthDate"] == sample_patient["birthDate"]

    def test_redact_phi_preserves_non_phi(self, fhir_connector, sample_patient):
        """Test that non-PHI fields are preserved."""
        redacted = fhir_connector._redact_phi(sample_patient)

        assert redacted["resourceType"] == "Patient"
        assert redacted["id"] == "patient-001"
        assert redacted["meta"] == sample_patient["meta"]


class TestResourceTextExtraction:
    """Test resource text extraction."""

    def test_extract_observation_text(self, fhir_connector, sample_observation):
        """Test extracting observation text."""
        text = fhir_connector._extract_resource_text(sample_observation)

        assert "Heart rate" in text
        assert "72" in text

    def test_extract_condition_text(self, fhir_connector, sample_condition):
        """Test extracting condition text."""
        text = fhir_connector._extract_resource_text(sample_condition)

        assert "Hypertension" in text

    def test_extract_patient_text_redacted(self, fhir_connector, sample_patient):
        """Test patient text with PHI redacted."""
        redacted = fhir_connector._redact_phi(sample_patient)
        text = fhir_connector._extract_resource_text(redacted)

        # Should not contain PHI
        assert "John" not in text
        assert "Smith" not in text


class TestResourceSearch:
    """Test FHIR resource search."""

    @pytest.mark.asyncio
    async def test_search_resources(self, fhir_connector, sample_patient):
        """Test searching resources."""

        async def mock_request(endpoint, params=None):
            return {
                "entry": [{"resource": sample_patient}],
                "link": [],
            }

        fhir_connector._fhir_request = mock_request

        resources, next_link = await fhir_connector._search_resources("Patient")

        assert len(resources) == 1
        assert resources[0].id == "patient-001"
        assert resources[0].resource_type == "Patient"

    @pytest.mark.asyncio
    async def test_search_resources_pagination(self, fhir_connector, sample_patient):
        """Test pagination with next link."""

        async def mock_request(endpoint, params=None):
            return {
                "entry": [{"resource": sample_patient}],
                "link": [{"relation": "next", "url": "https://fhir.example.com/r4/Patient?page=2"}],
            }

        fhir_connector._fhir_request = mock_request

        resources, next_link = await fhir_connector._search_resources("Patient")

        assert next_link is not None
        assert "page=2" in next_link

    @pytest.mark.asyncio
    async def test_search_resources_empty(self, fhir_connector):
        """Test empty search results."""

        async def mock_request(endpoint, params=None):
            return {"entry": [], "link": []}

        fhir_connector._fhir_request = mock_request

        resources, next_link = await fhir_connector._search_resources("Patient")

        assert len(resources) == 0


class TestSyncItems:
    """Test sync_items functionality."""

    @pytest.mark.asyncio
    async def test_sync_items_multiple_types(
        self, fhir_connector, sample_patient, sample_observation, sample_condition
    ):
        """Test syncing multiple resource types."""
        call_count = {"Patient": 0, "Observation": 0, "Condition": 0}

        async def mock_request(endpoint, params=None):
            if "Patient" in endpoint:
                call_count["Patient"] += 1
                return {"entry": [{"resource": sample_patient}], "link": []}
            elif "Observation" in endpoint:
                call_count["Observation"] += 1
                return {"entry": [{"resource": sample_observation}], "link": []}
            elif "Condition" in endpoint:
                call_count["Condition"] += 1
                return {"entry": [{"resource": sample_condition}], "link": []}
            return {"entry": [], "link": []}

        fhir_connector._fhir_request = mock_request
        state = SyncState(connector_id="fhir", status=SyncStatus.IDLE)

        items = []
        async for item in fhir_connector.sync_items(state, batch_size=100):
            items.append(item)

        assert len(items) == 3
        assert call_count["Patient"] == 1
        assert call_count["Observation"] == 1
        assert call_count["Condition"] == 1

    @pytest.mark.asyncio
    async def test_sync_incremental(self, fhir_connector, sample_observation):
        """Test incremental sync with _lastUpdated."""
        cursor = "2024-01-16T00:00:00+00:00"
        state = SyncState(connector_id="fhir", status=SyncStatus.IDLE, cursor=cursor)

        request_params = {}

        async def mock_request(endpoint, params=None):
            request_params.update(params or {})
            return {"entry": [{"resource": sample_observation}], "link": []}

        fhir_connector._fhir_request = mock_request

        items = []
        async for item in fhir_connector.sync_items(state, batch_size=100):
            items.append(item)

        assert "_lastUpdated" in request_params

    @pytest.mark.asyncio
    async def test_sync_updates_cursor(self, fhir_connector, sample_observation):
        """Test that sync updates cursor."""
        state = SyncState(connector_id="fhir", status=SyncStatus.IDLE)

        async def mock_request(endpoint, params=None):
            return {"entry": [{"resource": sample_observation}], "link": []}

        fhir_connector._fhir_request = mock_request

        async for _ in fhir_connector.sync_items(state, batch_size=100):
            pass

        assert state.cursor is not None

    @pytest.mark.asyncio
    async def test_sync_respects_batch_size(self, fhir_connector, sample_patient):
        """Test batch size limit."""
        state = SyncState(connector_id="fhir", status=SyncStatus.IDLE)

        # Return many resources
        many_patients = [{**sample_patient, "id": f"patient-{i}"} for i in range(10)]

        async def mock_request(endpoint, params=None):
            return {
                "entry": [{"resource": p} for p in many_patients],
                "link": [],
            }

        fhir_connector._fhir_request = mock_request

        items = []
        async for item in fhir_connector.sync_items(state, batch_size=5):
            items.append(item)

        assert len(items) == 5


class TestFetch:
    """Test fetch functionality."""

    @pytest.mark.asyncio
    async def test_fetch_resource(self, fhir_connector, sample_patient):
        """Test fetching a resource."""

        async def mock_request(endpoint, params=None):
            return sample_patient

        fhir_connector._fhir_request = mock_request

        result = await fhir_connector.fetch("Patient/patient-001")

        assert result is not None
        assert "patient-001" in result.id

    @pytest.mark.asyncio
    async def test_fetch_with_fhir_prefix(self, fhir_connector, sample_observation):
        """Test fetching with fhir- prefix."""

        async def mock_request(endpoint, params=None):
            return sample_observation

        fhir_connector._fhir_request = mock_request

        result = await fhir_connector.fetch("fhir-Observation/obs-001")

        assert result is not None

    @pytest.mark.asyncio
    async def test_fetch_with_redaction(self, fhir_connector, sample_patient):
        """Test fetch applies PHI redaction."""

        async def mock_request(endpoint, params=None):
            return sample_patient

        fhir_connector._fhir_request = mock_request

        result = await fhir_connector.fetch("Patient/patient-001")

        assert result.metadata["phi_redacted"] is True

    @pytest.mark.asyncio
    async def test_fetch_invalid_reference(self, fhir_connector):
        """Test fetching with invalid reference."""
        result = await fhir_connector.fetch("invalid")

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_not_found(self, fhir_connector):
        """Test fetching non-existent resource."""

        async def mock_request(endpoint, params=None):
            raise Exception("Resource not found")

        fhir_connector._fhir_request = mock_request

        result = await fhir_connector.fetch("Patient/nonexistent")

        assert result is None


class TestSearch:
    """Test search functionality."""

    @pytest.mark.asyncio
    async def test_search_across_types(self, fhir_connector, sample_patient, sample_observation):
        """Test searching across resource types."""

        async def mock_request(endpoint, params=None):
            if "Patient" in endpoint:
                return {"entry": [{"resource": sample_patient}], "link": []}
            elif "Observation" in endpoint:
                return {"entry": [{"resource": sample_observation}], "link": []}
            return {"entry": [], "link": []}

        fhir_connector._fhir_request = mock_request

        results = await fhir_connector.search("heart")

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_with_limit(self, fhir_connector, sample_patient):
        """Test search with limit."""

        async def mock_request(endpoint, params=None):
            return {
                "entry": [{"resource": sample_patient}] * 10,
                "link": [],
            }

        fhir_connector._fhir_request = mock_request

        results = await fhir_connector.search("patient", limit=3)

        assert len(results) == 3


class TestDatetimeParsing:
    """Test FHIR datetime parsing."""

    def test_parse_valid_datetime(self, fhir_connector):
        """Test parsing valid datetime."""
        dt = fhir_connector._parse_datetime("2024-01-15T10:00:00Z")

        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15

    def test_parse_datetime_with_timezone(self, fhir_connector):
        """Test parsing datetime with timezone."""
        dt = fhir_connector._parse_datetime("2024-01-15T10:00:00+05:00")

        assert dt is not None

    def test_parse_none_datetime(self, fhir_connector):
        """Test parsing None."""
        dt = fhir_connector._parse_datetime(None)

        assert dt is None

    def test_parse_invalid_datetime(self, fhir_connector):
        """Test parsing invalid datetime."""
        dt = fhir_connector._parse_datetime("not-a-date")

        assert dt is None
