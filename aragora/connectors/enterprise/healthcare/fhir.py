"""
HL7 FHIR R4 Healthcare Connector.

HIPAA-compliant integration with healthcare systems:
- FHIR R4 resource retrieval
- SMART on FHIR OAuth2 authentication
- PHI redaction using Safe Harbor method
- Comprehensive audit logging for compliance
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

from aragora.connectors.enterprise.base import (
    EnterpriseConnector,
    SyncItem,
    SyncState,
)
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)


# =============================================================================
# FHIR Resource Types
# =============================================================================

class FHIRResourceType(str, Enum):
    """Supported FHIR R4 resource types."""

    # Clinical
    PATIENT = "Patient"
    CONDITION = "Condition"
    OBSERVATION = "Observation"
    PROCEDURE = "Procedure"
    MEDICATION_REQUEST = "MedicationRequest"
    MEDICATION_STATEMENT = "MedicationStatement"
    ALLERGY_INTOLERANCE = "AllergyIntolerance"
    IMMUNIZATION = "Immunization"
    DIAGNOSTIC_REPORT = "DiagnosticReport"
    CARE_PLAN = "CarePlan"

    # Administrative
    PRACTITIONER = "Practitioner"
    ORGANIZATION = "Organization"
    LOCATION = "Location"
    ENCOUNTER = "Encounter"
    APPOINTMENT = "Appointment"

    # Documents
    DOCUMENT_REFERENCE = "DocumentReference"
    COMPOSITION = "Composition"


# =============================================================================
# PHI Safe Harbor Identifiers (18 HIPAA identifiers)
# =============================================================================

PHI_IDENTIFIERS = {
    "names",
    "geographic_data",  # Smaller than state
    "dates",  # Except year for ages > 89
    "phone_numbers",
    "fax_numbers",
    "email_addresses",
    "ssn",
    "mrn",  # Medical record numbers
    "health_plan_beneficiary",
    "account_numbers",
    "certificate_numbers",
    "vehicle_identifiers",
    "device_identifiers",
    "urls",
    "ip_addresses",
    "biometric_identifiers",
    "photographs",
    "unique_identifiers",
}


# =============================================================================
# PHI Redaction
# =============================================================================

@dataclass
class RedactionResult:
    """Result of PHI redaction."""

    original_hash: str  # SHA-256 of original for audit
    redacted_text: str
    redactions_count: int
    redaction_types: List[str]


class PHIRedactor:
    """
    PHI redactor using Safe Harbor method.

    Removes or masks the 18 HIPAA identifiers while preserving
    clinical utility of the data.
    """

    # Regex patterns for PHI detection
    PATTERNS = {
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "phone": re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "mrn": re.compile(r"\b(?:MRN|Medical Record|Patient ID)[:\s#]*[\w-]+\b", re.IGNORECASE),
        "date_full": re.compile(r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b"),
        "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "zip_full": re.compile(r"\b\d{5}(?:-\d{4})?\b"),
        "account": re.compile(r"\b(?:Account|Acct)[:\s#]*[\w-]+\b", re.IGNORECASE),
    }

    # FHIR paths containing PHI
    PHI_PATHS = {
        "Patient": [
            "name", "telecom", "address", "birthDate",
            "identifier", "photo", "contact",
        ],
        "Practitioner": [
            "name", "telecom", "address", "identifier", "photo",
        ],
        "Organization": [
            "telecom", "address", "identifier",
        ],
    }

    def __init__(
        self,
        redact_names: bool = True,
        redact_dates: bool = True,
        preserve_year: bool = True,  # Keep year for clinical relevance
        preserve_age_over_89: bool = False,
    ):
        self.redact_names = redact_names
        self.redact_dates = redact_dates
        self.preserve_year = preserve_year
        self.preserve_age_over_89 = preserve_age_over_89

    def redact_text(self, text: str) -> RedactionResult:
        """Redact PHI from free text."""
        original_hash = hashlib.sha256(text.encode()).hexdigest()
        redacted = text
        redaction_types = []
        count = 0

        for phi_type, pattern in self.PATTERNS.items():
            matches = pattern.findall(redacted)
            if matches:
                redaction_types.append(phi_type)
                count += len(matches)
                redacted = pattern.sub(f"[REDACTED-{phi_type.upper()}]", redacted)

        return RedactionResult(
            original_hash=original_hash,
            redacted_text=redacted,
            redactions_count=count,
            redaction_types=redaction_types,
        )

    def redact_fhir_resource(
        self,
        resource: Dict[str, Any],
        resource_type: str,
    ) -> Dict[str, Any]:
        """
        Redact PHI from a FHIR resource.

        Applies Safe Harbor method to remove/mask identifiers.
        """
        redacted = json.loads(json.dumps(resource))  # Deep copy

        # Get PHI paths for this resource type
        phi_paths = self.PHI_PATHS.get(resource_type, [])

        for path in phi_paths:
            if path in redacted:
                redacted[path] = self._redact_field(redacted[path], path)

        # Redact any text fields
        redacted = self._redact_text_fields(redacted)

        return redacted

    def _redact_field(self, value: Any, field_name: str) -> Any:
        """Redact a specific field value."""
        if value is None:
            return None

        if isinstance(value, list):
            return [self._redact_field(v, field_name) for v in value]

        if isinstance(value, dict):
            if field_name == "name":
                return self._redact_name(value)
            elif field_name == "telecom":
                return self._redact_telecom(value)
            elif field_name == "address":
                return self._redact_address(value)
            elif field_name == "identifier":
                return self._redact_identifier(value)
            # birthDate is always a string in FHIR, not a dict
            # If we encounter a dict, recursively process its fields
            return {k: self._redact_field(v, k) for k, v in value.items()}

        if isinstance(value, str):
            if field_name == "birthDate":
                return self._redact_date(value)
            return "[REDACTED]"

        return value

    def _redact_name(self, name: Dict[str, Any]) -> Dict[str, Any]:
        """Redact a FHIR HumanName."""
        return {
            "use": name.get("use"),
            "family": "[REDACTED]",
            "given": ["[REDACTED]"] if name.get("given") else None,
            "prefix": name.get("prefix"),  # Keep titles
            "suffix": name.get("suffix"),  # Keep credentials
        }

    def _redact_telecom(self, telecom: Dict[str, Any]) -> Dict[str, Any]:
        """Redact a FHIR ContactPoint."""
        return {
            "system": telecom.get("system"),
            "use": telecom.get("use"),
            "value": "[REDACTED]",
        }

    def _redact_address(self, address: Dict[str, Any]) -> Dict[str, Any]:
        """Redact a FHIR Address (keep state for geographic analysis)."""
        return {
            "use": address.get("use"),
            "type": address.get("type"),
            "line": ["[REDACTED]"],
            "city": "[REDACTED]",
            "state": address.get("state"),  # Keep state (Safe Harbor allows)
            "postalCode": address.get("postalCode", "")[:3] + "XX" if address.get("postalCode") else None,
            "country": address.get("country"),
        }

    def _redact_identifier(self, identifier: Dict[str, Any]) -> Dict[str, Any]:
        """Redact a FHIR Identifier."""
        return {
            "system": identifier.get("system"),
            "type": identifier.get("type"),
            "value": hashlib.sha256(
                str(identifier.get("value", "")).encode()
            ).hexdigest()[:16],  # One-way hash
        }

    def _redact_date(self, date_str: str) -> str:
        """Redact a date, optionally preserving year."""
        if not date_str or not self.redact_dates:
            return date_str

        if self.preserve_year and len(date_str) >= 4:
            return date_str[:4]  # Keep year only

        return "[REDACTED-DATE]"

    def _redact_text_fields(self, obj: Any) -> Any:
        """Recursively redact text fields that might contain PHI."""
        if isinstance(obj, dict):
            result: Dict[str, Any] = {}
            for key, value in obj.items():
                if key in {"text", "div", "narrative", "note", "comment"}:
                    if isinstance(value, str):
                        result[key] = self.redact_text(value).redacted_text
                    elif isinstance(value, dict) and "div" in value:
                        result[key] = {
                            **value,
                            "div": self.redact_text(value["div"]).redacted_text,
                        }
                    else:
                        result[key] = value
                else:
                    result[key] = self._redact_text_fields(value)
            return result
        elif isinstance(obj, list):
            return [self._redact_text_fields(item) for item in obj]
        return obj


# =============================================================================
# FHIR Audit Logger
# =============================================================================

@dataclass
class AuditEvent:
    """HIPAA-compliant audit event."""

    id: str
    timestamp: datetime
    action: str  # C, R, U, D (Create, Read, Update, Delete)
    resource_type: str
    resource_id: str
    user_id: str
    user_role: str
    organization_id: str
    outcome: str  # 0=success, 4=minor failure, 8=serious failure, 12=major failure
    reason: Optional[str] = None
    query_params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "resourceType": self.resource_type,
            "resourceId": self.resource_id,
            "userId": self.user_id,
            "userRole": self.user_role,
            "organizationId": self.organization_id,
            "outcome": self.outcome,
            "reason": self.reason,
            "queryParams": self.query_params,
        }


class FHIRAuditLogger:
    """
    HIPAA-compliant audit logger for FHIR operations.

    Logs all access to PHI with required audit trail information.
    """

    def __init__(
        self,
        organization_id: str,
        user_id: str = "system",
        user_role: str = "service",
    ):
        self.organization_id = organization_id
        self.user_id = user_id
        self.user_role = user_role
        self._events: List[AuditEvent] = []

    def log_read(
        self,
        resource_type: str,
        resource_id: str,
        reason: str = "clinical_care",
        query_params: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log a read operation."""
        event = AuditEvent(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            action="R",
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=self.user_id,
            user_role=self.user_role,
            organization_id=self.organization_id,
            outcome="0",
            reason=reason,
            query_params=query_params,
        )
        self._events.append(event)
        logger.info(
            f"[AUDIT] READ {resource_type}/{resource_id} by {self.user_id} "
            f"reason={reason}"
        )
        return event

    def log_search(
        self,
        resource_type: str,
        query_params: Dict[str, Any],
        results_count: int,
        reason: str = "clinical_care",
    ) -> AuditEvent:
        """Log a search operation."""
        event = AuditEvent(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            action="R",
            resource_type=resource_type,
            resource_id=f"search:{results_count}",
            user_id=self.user_id,
            user_role=self.user_role,
            organization_id=self.organization_id,
            outcome="0",
            reason=reason,
            query_params=query_params,
        )
        self._events.append(event)
        logger.info(
            f"[AUDIT] SEARCH {resource_type} by {self.user_id} "
            f"results={results_count} reason={reason}"
        )
        return event

    def log_export(
        self,
        resource_types: List[str],
        record_count: int,
        reason: str = "data_sync",
    ) -> AuditEvent:
        """Log a bulk export operation."""
        event = AuditEvent(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            action="R",
            resource_type="BulkExport",
            resource_id=f"export:{record_count}",
            user_id=self.user_id,
            user_role=self.user_role,
            organization_id=self.organization_id,
            outcome="0",
            reason=reason,
            query_params={"resourceTypes": resource_types},
        )
        self._events.append(event)
        logger.info(
            f"[AUDIT] EXPORT {resource_types} by {self.user_id} "
            f"records={record_count} reason={reason}"
        )
        return event

    def get_events(
        self,
        since: Optional[datetime] = None,
        resource_type: Optional[str] = None,
    ) -> List[AuditEvent]:
        """Get audit events with optional filtering."""
        events = self._events

        if since:
            events = [e for e in events if e.timestamp >= since]

        if resource_type:
            events = [e for e in events if e.resource_type == resource_type]

        return events


# =============================================================================
# FHIR Connector
# =============================================================================

class FHIRConnector(EnterpriseConnector):
    """
    HIPAA-compliant FHIR R4 connector.

    Features:
    - SMART on FHIR OAuth2 authentication
    - Incremental sync using _lastUpdated
    - PHI redaction (Safe Harbor method)
    - Comprehensive audit logging
    - Resource type filtering
    """

    def __init__(
        self,
        base_url: str,
        organization_id: str,
        resource_types: Optional[List[FHIRResourceType]] = None,
        client_id: Optional[str] = None,
        enable_phi_redaction: bool = True,
        preserve_year_in_dates: bool = True,
        audit_reason: str = "clinical_decision_support",
        **kwargs,
    ):
        connector_id = f"fhir_{hashlib.sha256(base_url.encode()).hexdigest()[:12]}"
        super().__init__(connector_id=connector_id, **kwargs)

        self.base_url = base_url.rstrip("/")
        self.organization_id = organization_id
        self.resource_types = resource_types or [
            FHIRResourceType.PATIENT,
            FHIRResourceType.CONDITION,
            FHIRResourceType.OBSERVATION,
            FHIRResourceType.MEDICATION_REQUEST,
        ]
        self.client_id = client_id
        self.enable_phi_redaction = enable_phi_redaction
        self.audit_reason = audit_reason

        # Initialize components
        self._redactor = PHIRedactor(
            redact_names=True,
            redact_dates=True,
            preserve_year=preserve_year_in_dates,
        )
        self._audit_logger = FHIRAuditLogger(
            organization_id=organization_id,
            user_id="aragora_sync",
            user_role="service",
        )

        self._client = None
        self._access_token = None
        self._token_expires_at: Optional[datetime] = None

    @property
    def source_type(self) -> SourceType:
        return SourceType.DATABASE

    @property
    def name(self) -> str:
        return f"FHIR ({self.base_url})"

    async def _get_client(self):
        """Get HTTP client with authentication."""
        try:
            import httpx
        except ImportError:
            logger.error("httpx not installed. Run: pip install httpx")
            raise

        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={"Accept": "application/fhir+json"},
            )

        # Check token expiration
        if self._access_token and self._token_expires_at:
            if datetime.now(timezone.utc) >= self._token_expires_at:
                self._access_token = None

        # Get new token if needed
        if not self._access_token:
            await self._authenticate()

        return self._client

    async def _authenticate(self):
        """Authenticate using SMART on FHIR OAuth2."""
        # Get credentials
        client_id = self.client_id or await self.credentials.get_credential("FHIR_CLIENT_ID")
        client_secret = await self.credentials.get_credential("FHIR_CLIENT_SECRET")

        if not client_id or not client_secret:
            logger.warning("FHIR credentials not configured, using unauthenticated access")
            return

        try:
            import httpx

            # Discover token endpoint from capability statement
            async with httpx.AsyncClient() as client:
                # Get SMART configuration
                smart_url = f"{self.base_url}/.well-known/smart-configuration"
                try:
                    response = await client.get(smart_url)
                    if response.status_code == 200:
                        smart_config = response.json()
                        token_url = smart_config.get("token_endpoint")
                    else:
                        # Fallback to metadata
                        token_url = f"{self.base_url}/oauth2/token"
                except Exception as e:
                    logger.debug(f"[{self.name}] SMART config discovery failed, using default: {e}")
                    token_url = f"{self.base_url}/oauth2/token"

                # Request token (client credentials flow)
                response = await client.post(
                    token_url,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "scope": "system/*.read",
                    },
                )

                if response.status_code == 200:
                    token_data = response.json()
                    self._access_token = token_data["access_token"]
                    expires_in = token_data.get("expires_in", 3600)
                    self._token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in - 60)
                    logger.info(f"[{self.name}] Authenticated successfully")
                else:
                    logger.warning(f"[{self.name}] Authentication failed: {response.status_code}")

        except Exception as e:
            logger.warning(f"[{self.name}] Authentication error: {e}")

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        headers = {
            "Accept": "application/fhir+json",
            "Content-Type": "application/fhir+json",
        }

        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"

        return headers

    def _resource_to_content(self, resource: Dict[str, Any]) -> str:
        """Convert FHIR resource to searchable text content."""
        resource_type = resource.get("resourceType", "Unknown")

        parts = [f"Resource Type: {resource_type}"]

        # Extract key clinical information based on resource type
        if resource_type == "Patient":
            if resource.get("name"):
                parts.append(f"Name: {self._format_name(resource['name'][0])}")
            if resource.get("birthDate"):
                parts.append(f"Birth Year: {resource['birthDate']}")
            if resource.get("gender"):
                parts.append(f"Gender: {resource['gender']}")

        elif resource_type == "Condition":
            if resource.get("code", {}).get("text"):
                parts.append(f"Condition: {resource['code']['text']}")
            if resource.get("clinicalStatus", {}).get("coding"):
                status = resource["clinicalStatus"]["coding"][0].get("code", "")
                parts.append(f"Status: {status}")
            if resource.get("onsetDateTime"):
                parts.append(f"Onset: {resource['onsetDateTime']}")

        elif resource_type == "Observation":
            if resource.get("code", {}).get("text"):
                parts.append(f"Observation: {resource['code']['text']}")
            if resource.get("valueQuantity"):
                vq = resource["valueQuantity"]
                parts.append(f"Value: {vq.get('value')} {vq.get('unit', '')}")
            elif resource.get("valueCodeableConcept", {}).get("text"):
                parts.append(f"Value: {resource['valueCodeableConcept']['text']}")
            if resource.get("effectiveDateTime"):
                parts.append(f"Effective: {resource['effectiveDateTime']}")

        elif resource_type == "MedicationRequest":
            if resource.get("medicationCodeableConcept", {}).get("text"):
                parts.append(f"Medication: {resource['medicationCodeableConcept']['text']}")
            if resource.get("status"):
                parts.append(f"Status: {resource['status']}")
            if resource.get("intent"):
                parts.append(f"Intent: {resource['intent']}")

        # Add narrative if present
        if resource.get("text", {}).get("div"):
            # Strip HTML tags
            narrative = re.sub(r"<[^>]+>", " ", resource["text"]["div"])
            narrative = re.sub(r"\s+", " ", narrative).strip()
            if narrative:
                parts.append(f"Narrative: {narrative[:500]}")

        return "\n".join(parts)

    def _format_name(self, name: Dict[str, Any]) -> str:
        """Format a FHIR HumanName."""
        parts = []
        if name.get("prefix"):
            parts.extend(name["prefix"])
        if name.get("given"):
            parts.extend(name["given"])
        if name.get("family"):
            parts.append(name["family"])
        if name.get("suffix"):
            parts.extend(name["suffix"])
        return " ".join(parts)

    def _infer_domain(self, resource_type: str) -> str:
        """Infer domain from resource type."""
        clinical_types = {
            "Condition", "Observation", "Procedure", "DiagnosticReport",
            "MedicationRequest", "MedicationStatement", "Immunization",
            "AllergyIntolerance", "CarePlan",
        }

        admin_types = {
            "Patient", "Practitioner", "Organization", "Location",
            "Encounter", "Appointment",
        }

        if resource_type in clinical_types:
            return "healthcare/clinical"
        elif resource_type in admin_types:
            return "healthcare/administrative"
        elif resource_type in {"DocumentReference", "Composition"}:
            return "healthcare/documents"

        return "healthcare/general"

    async def sync_items(
        self,
        state: SyncState,
        batch_size: int = 100,
    ) -> AsyncIterator[SyncItem]:
        """
        Yield items from FHIR server for sync.

        Uses _lastUpdated for incremental sync.
        """
        client = await self._get_client()

        for resource_type in self.resource_types:
            resource_name = resource_type.value

            try:
                # Build search parameters
                params = {
                    "_count": str(batch_size),
                    "_sort": "_lastUpdated",
                }

                # Add incremental sync filter
                if state.last_item_timestamp:
                    params["_lastUpdated"] = f"gt{state.last_item_timestamp.isoformat()}"

                # Use cursor if available
                url = f"{self.base_url}/{resource_name}"
                if state.cursor and state.cursor.startswith(f"{resource_name}:"):
                    url = state.cursor.split(":", 1)[1]

                while url:
                    response = await client.get(
                        url,
                        params=params if not state.cursor else None,
                        headers=self._get_headers(),
                    )

                    if response.status_code != 200:
                        logger.warning(
                            f"[{self.name}] Failed to fetch {resource_name}: "
                            f"{response.status_code}"
                        )
                        break

                    bundle = response.json()
                    entries = bundle.get("entry", [])

                    # Log search for audit
                    self._audit_logger.log_search(
                        resource_type=resource_name,
                        query_params=params,
                        results_count=len(entries),
                        reason=self.audit_reason,
                    )

                    for entry in entries:
                        resource = entry.get("resource", {})
                        resource_id = resource.get("id", "")

                        # Log read for audit
                        self._audit_logger.log_read(
                            resource_type=resource_name,
                            resource_id=resource_id,
                            reason=self.audit_reason,
                        )

                        # Apply PHI redaction if enabled
                        if self.enable_phi_redaction:
                            resource = self._redactor.redact_fhir_resource(
                                resource, resource_name
                            )

                        # Convert to content
                        content = self._resource_to_content(resource)

                        # Get timestamp
                        last_updated = resource.get("meta", {}).get("lastUpdated")
                        updated_at = datetime.now(timezone.utc)
                        if last_updated:
                            try:
                                updated_at = datetime.fromisoformat(
                                    last_updated.replace("Z", "+00:00")
                                )
                            except Exception as e:
                                logger.debug(f"Invalid FHIR lastUpdated format: {e}")

                        # Create sync item
                        item_id = f"fhir:{self.organization_id}:{resource_name}:{resource_id}"

                        yield SyncItem(
                            id=item_id,
                            content=content[:100000],
                            source_type="healthcare",
                            source_id=f"{self.base_url}/{resource_name}/{resource_id}",
                            title=f"{resource_name} {resource_id}",
                            url=f"{self.base_url}/{resource_name}/{resource_id}",
                            updated_at=updated_at,
                            domain=self._infer_domain(resource_name),
                            confidence=0.9,  # High confidence for structured data
                            metadata={
                                "organization_id": self.organization_id,
                                "resource_type": resource_name,
                                "resource_id": resource_id,
                                "phi_redacted": self.enable_phi_redaction,
                                "fhir_version": "R4",
                            },
                        )

                    # Check for next page
                    url = None
                    for link in bundle.get("link", []):
                        if link.get("relation") == "next":
                            url = link.get("url")
                            state.cursor = f"{resource_name}:{url}"
                            break

                    # Clear params for pagination URL
                    params = {}

            except Exception as e:
                logger.error(f"[{self.name}] Error syncing {resource_name}: {e}")
                state.errors.append(f"{resource_name}: {str(e)}")

    async def search(
        self,
        query: str,
        limit: int = 10,
        resource_type: Optional[str] = None,
        **kwargs,
    ) -> list:
        """
        Search FHIR resources.

        Uses FHIR search parameters with content-based matching.
        """
        client = await self._get_client()
        results = []

        resource_types = (
            [resource_type] if resource_type
            else [rt.value for rt in self.resource_types]
        )

        for rt in resource_types[:3]:  # Limit resource types
            try:
                # Build search URL with _content parameter
                params = {
                    "_content": query,
                    "_count": str(limit),
                }

                response = await client.get(
                    f"{self.base_url}/{rt}",
                    params=params,
                    headers=self._get_headers(),
                )

                if response.status_code == 200:
                    bundle = response.json()
                    entries = bundle.get("entry", [])

                    # Log for audit
                    self._audit_logger.log_search(
                        resource_type=rt,
                        query_params=params,
                        results_count=len(entries),
                        reason="search",
                    )

                    for entry in entries:
                        resource = entry.get("resource", {})

                        if self.enable_phi_redaction:
                            resource = self._redactor.redact_fhir_resource(resource, rt)

                        results.append({
                            "resource_type": rt,
                            "resource_id": resource.get("id"),
                            "content": self._resource_to_content(resource),
                            "score": entry.get("search", {}).get("score", 0.5),
                        })

            except Exception as e:
                logger.debug(f"Search failed for {rt}: {e}")

        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:limit]

    async def fetch(self, evidence_id: str):
        """Fetch a specific FHIR resource."""
        if not evidence_id.startswith("fhir:"):
            return None

        parts = evidence_id.split(":")
        if len(parts) < 4:
            return None

        org_id, resource_type, resource_id = parts[1], parts[2], parts[3]

        if org_id != self.organization_id:
            return None

        try:
            client = await self._get_client()

            response = await client.get(
                f"{self.base_url}/{resource_type}/{resource_id}",
                headers=self._get_headers(),
            )

            if response.status_code == 200:
                resource = response.json()

                # Log for audit
                self._audit_logger.log_read(
                    resource_type=resource_type,
                    resource_id=resource_id,
                    reason="fetch",
                )

                if self.enable_phi_redaction:
                    resource = self._redactor.redact_fhir_resource(
                        resource, resource_type
                    )

                return resource

        except Exception as e:
            logger.error(f"[{self.name}] Fetch failed: {e}")

        return None

    def get_audit_events(
        self,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get audit events for compliance reporting."""
        events = self._audit_logger.get_events(since=since)
        return [e.to_dict() for e in events]

    async def handle_webhook(self, payload: Dict[str, Any]) -> bool:
        """Handle FHIR subscription notification."""
        # FHIR subscriptions send notifications for resource changes
        notification_type = payload.get("type")

        if notification_type == "subscription-notification":
            entries = payload.get("entry", [])
            if entries:
                logger.info(f"[{self.name}] Webhook: {len(entries)} resource updates")
                asyncio.create_task(self.sync(max_items=len(entries) * 2))
                return True

        return False
