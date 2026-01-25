"""
PagerDuty Connector.

Integration for incident management and on-call scheduling.
Supports:
- Incident creation from critical findings
- Investigation notes
- Incident resolution
- On-call schedule lookup
- Service health status
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class IncidentUrgency(str, Enum):
    """Incident urgency levels."""

    HIGH = "high"
    LOW = "low"


class IncidentStatus(str, Enum):
    """Incident status values."""

    TRIGGERED = "triggered"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class IncidentPriority(str, Enum):
    """Incident priority levels (P1-P5)."""

    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"
    P5 = "P5"


class ServiceStatus(str, Enum):
    """Service health status."""

    ACTIVE = "active"
    WARNING = "warning"
    CRITICAL = "critical"
    MAINTENANCE = "maintenance"
    DISABLED = "disabled"


@dataclass
class PagerDutyCredentials:
    """PagerDuty API credentials."""

    api_key: str
    email: str  # Email of the user making requests (for From header)
    webhook_secret: str | None = None  # For verifying webhooks


@dataclass
class Service:
    """PagerDuty service representation."""

    id: str
    name: str
    description: str | None = None
    status: ServiceStatus = ServiceStatus.ACTIVE
    escalation_policy_id: str | None = None
    html_url: str | None = None
    created_at: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Service:
        """Create Service from API response."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            status=ServiceStatus(data.get("status", "active")),
            escalation_policy_id=data.get("escalation_policy", {}).get("id"),
            html_url=data.get("html_url"),
            created_at=(
                datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
                if data.get("created_at")
                else None
            ),
        )


@dataclass
class User:
    """PagerDuty user representation."""

    id: str
    name: str
    email: str
    role: str | None = None
    html_url: str | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> User:
        """Create User from API response."""
        return cls(
            id=data["id"],
            name=data["name"],
            email=data["email"],
            role=data.get("role"),
            html_url=data.get("html_url"),
        )


@dataclass
class OnCallSchedule:
    """On-call schedule entry."""

    user: User
    schedule_id: str
    schedule_name: str
    start: datetime
    end: datetime
    escalation_level: int = 1

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> OnCallSchedule:
        """Create OnCallSchedule from API response."""
        user_data = data.get("user", {})
        return cls(
            user=User.from_api(user_data)
            if user_data
            else User(id="unknown", name="Unknown", email="unknown@example.com"),
            schedule_id=data.get("schedule", {}).get("id", ""),
            schedule_name=data.get("schedule", {}).get("summary", ""),
            start=datetime.fromisoformat(data["start"].replace("Z", "+00:00")),
            end=datetime.fromisoformat(data["end"].replace("Z", "+00:00")),
            escalation_level=data.get("escalation_level", 1),
        )


@dataclass
class IncidentNote:
    """Note attached to an incident."""

    id: str
    content: str
    user: User | None = None
    created_at: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> IncidentNote:
        """Create IncidentNote from API response."""
        user_data = data.get("user")
        return cls(
            id=data["id"],
            content=data["content"],
            user=User.from_api(user_data) if user_data else None,
            created_at=(
                datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
                if data.get("created_at")
                else None
            ),
        )


@dataclass
class Incident:
    """PagerDuty incident representation."""

    id: str
    incident_number: int
    title: str
    status: IncidentStatus
    urgency: IncidentUrgency
    service: Service | None = None
    assigned_to: list[User] = field(default_factory=list)
    priority: IncidentPriority | None = None
    html_url: str | None = None
    description: str | None = None
    created_at: datetime | None = None
    resolved_at: datetime | None = None
    last_status_change_at: datetime | None = None
    notes: list[IncidentNote] = field(default_factory=list)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Incident:
        """Create Incident from API response."""
        service_data = data.get("service")
        assigned = data.get("assignments", [])
        priority_data = data.get("priority")

        return cls(
            id=data["id"],
            incident_number=data.get("incident_number", 0),
            title=data["title"],
            status=IncidentStatus(data["status"]),
            urgency=IncidentUrgency(data["urgency"]),
            service=Service.from_api(service_data) if service_data else None,
            assigned_to=[User.from_api(a["assignee"]) for a in assigned if a.get("assignee")],
            priority=(
                IncidentPriority(priority_data["summary"])
                if priority_data and priority_data.get("summary")
                else None
            ),
            html_url=data.get("html_url"),
            description=data.get("description"),
            created_at=(
                datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
                if data.get("created_at")
                else None
            ),
            resolved_at=(
                datetime.fromisoformat(data["resolved_at"].replace("Z", "+00:00"))
                if data.get("resolved_at")
                else None
            ),
            last_status_change_at=(
                datetime.fromisoformat(data["last_status_change_at"].replace("Z", "+00:00"))
                if data.get("last_status_change_at")
                else None
            ),
        )


@dataclass
class IncidentCreateRequest:
    """Request to create a new incident."""

    title: str
    service_id: str
    urgency: IncidentUrgency = IncidentUrgency.HIGH
    description: str | None = None
    priority_id: str | None = None
    escalation_policy_id: str | None = None
    incident_key: str | None = None  # Deduplication key
    assignments: list[str] | None = None  # User IDs to assign


@dataclass
class WebhookPayload:
    """Parsed PagerDuty webhook payload."""

    event_type: str
    incident: Incident | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)


class PagerDutyError(Exception):
    """PagerDuty API error."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_code: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


class PagerDutyConnector:
    """
    PagerDuty API connector.

    Provides incident management, on-call scheduling, and service health
    monitoring for IT/DevOps workflows.

    Example:
        ```python
        credentials = PagerDutyCredentials(
            api_key="your-api-key",
            email="user@example.com"
        )

        async with PagerDutyConnector(credentials) as pd:
            # Create incident from critical finding
            incident = await pd.create_incident(
                IncidentCreateRequest(
                    title="Critical security vulnerability detected",
                    service_id="SERVICE_ID",
                    urgency=IncidentUrgency.HIGH,
                    description="SAST scanner found SQL injection in auth.py"
                )
            )

            # Add investigation notes
            await pd.add_note(incident.id, "Investigating root cause...")

            # Resolve when fixed
            await pd.resolve_incident(
                incident.id,
                resolution="Patched SQL injection vulnerability"
            )
        ```
    """

    BASE_URL = "https://api.pagerduty.com"

    def __init__(self, credentials: PagerDutyCredentials):
        """Initialize the connector."""
        self.credentials = credentials
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> PagerDutyConnector:
        """Enter async context."""
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Token token={self.credentials.api_key}",
                "Content-Type": "application/json",
                "From": self.credentials.email,
            },
            timeout=30.0,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get the HTTP client."""
        if not self._client:
            raise PagerDutyError("Connector not initialized. Use async context manager.")
        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an API request."""
        try:
            response = await self.client.request(
                method,
                path,
                json=json,
                params=params,
            )

            if response.status_code == 204:
                return {}

            data = response.json()

            if response.status_code >= 400:
                error = data.get("error", {})
                raise PagerDutyError(
                    message=error.get("message", "Unknown error"),
                    status_code=response.status_code,
                    error_code=error.get("code"),
                )

            return data

        except httpx.HTTPError as e:
            raise PagerDutyError(f"HTTP error: {e}") from e

    # -------------------------------------------------------------------------
    # Incident Management
    # -------------------------------------------------------------------------

    async def create_incident(self, request: IncidentCreateRequest) -> Incident:
        """
        Create a new incident.

        Args:
            request: Incident creation parameters

        Returns:
            Created incident
        """
        body: dict[str, Any] = {
            "type": "incident",
            "title": request.title,
            "service": {
                "id": request.service_id,
                "type": "service_reference",
            },
            "urgency": request.urgency.value,
        }

        if request.description:
            body["body"] = {
                "type": "incident_body",
                "details": request.description,
            }

        if request.priority_id:
            body["priority"] = {
                "id": request.priority_id,
                "type": "priority_reference",
            }

        if request.escalation_policy_id:
            body["escalation_policy"] = {
                "id": request.escalation_policy_id,
                "type": "escalation_policy_reference",
            }

        if request.incident_key:
            body["incident_key"] = request.incident_key

        if request.assignments:
            body["assignments"] = [
                {"assignee": {"id": uid, "type": "user_reference"}} for uid in request.assignments
            ]

        data = await self._request("POST", "/incidents", json={"incident": body})
        incident = Incident.from_api(data["incident"])

        logger.info(
            "Created PagerDuty incident",
            extra={
                "incident_id": incident.id,
                "incident_number": incident.incident_number,
                "title": incident.title,
            },
        )

        return incident

    async def get_incident(self, incident_id: str) -> Incident:
        """
        Get incident by ID.

        Args:
            incident_id: The incident ID

        Returns:
            Incident details
        """
        data = await self._request("GET", f"/incidents/{incident_id}")
        return Incident.from_api(data["incident"])

    async def list_incidents(
        self,
        statuses: list[IncidentStatus] | None = None,
        service_ids: list[str] | None = None,
        urgencies: list[IncidentUrgency] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> list[Incident]:
        """
        List incidents with filters.

        Args:
            statuses: Filter by status
            service_ids: Filter by service IDs
            urgencies: Filter by urgency
            since: Filter incidents created after this time
            until: Filter incidents created before this time
            limit: Maximum results (default 25, max 100)
            offset: Pagination offset

        Returns:
            List of matching incidents
        """
        params: dict[str, Any] = {
            "limit": min(limit, 100),
            "offset": offset,
        }

        if statuses:
            params["statuses[]"] = [s.value for s in statuses]

        if service_ids:
            params["service_ids[]"] = service_ids

        if urgencies:
            params["urgencies[]"] = [u.value for u in urgencies]

        if since:
            params["since"] = since.isoformat()

        if until:
            params["until"] = until.isoformat()

        data = await self._request("GET", "/incidents", params=params)
        return [Incident.from_api(i) for i in data.get("incidents", [])]

    async def acknowledge_incident(self, incident_id: str) -> Incident:
        """
        Acknowledge an incident.

        Args:
            incident_id: The incident ID

        Returns:
            Updated incident
        """
        data = await self._request(
            "PUT",
            f"/incidents/{incident_id}",
            json={
                "incident": {
                    "type": "incident_reference",
                    "status": IncidentStatus.ACKNOWLEDGED.value,
                }
            },
        )

        logger.info("Acknowledged incident", extra={"incident_id": incident_id})
        return Incident.from_api(data["incident"])

    async def resolve_incident(
        self,
        incident_id: str,
        resolution: str | None = None,
    ) -> Incident:
        """
        Resolve an incident.

        Args:
            incident_id: The incident ID
            resolution: Resolution note to add

        Returns:
            Updated incident
        """
        # First add resolution note if provided
        if resolution:
            await self.add_note(incident_id, f"Resolution: {resolution}")

        # Then resolve the incident
        data = await self._request(
            "PUT",
            f"/incidents/{incident_id}",
            json={
                "incident": {
                    "type": "incident_reference",
                    "status": IncidentStatus.RESOLVED.value,
                }
            },
        )

        logger.info(
            "Resolved incident",
            extra={"incident_id": incident_id, "resolution": resolution},
        )
        return Incident.from_api(data["incident"])

    async def reassign_incident(
        self,
        incident_id: str,
        user_ids: list[str],
    ) -> Incident:
        """
        Reassign an incident to different users.

        Args:
            incident_id: The incident ID
            user_ids: List of user IDs to assign

        Returns:
            Updated incident
        """
        data = await self._request(
            "PUT",
            f"/incidents/{incident_id}",
            json={
                "incident": {
                    "type": "incident_reference",
                    "assignments": [
                        {"assignee": {"id": uid, "type": "user_reference"}} for uid in user_ids
                    ],
                }
            },
        )

        logger.info(
            "Reassigned incident",
            extra={"incident_id": incident_id, "assigned_to": user_ids},
        )
        return Incident.from_api(data["incident"])

    async def merge_incidents(
        self,
        target_incident_id: str,
        source_incident_ids: list[str],
    ) -> Incident:
        """
        Merge multiple incidents into one.

        Args:
            target_incident_id: The incident to merge into
            source_incident_ids: Incidents to merge from

        Returns:
            Merged incident
        """
        data = await self._request(
            "PUT",
            f"/incidents/{target_incident_id}/merge",
            json={
                "source_incidents": [
                    {"id": iid, "type": "incident_reference"} for iid in source_incident_ids
                ]
            },
        )

        logger.info(
            "Merged incidents",
            extra={
                "target_id": target_incident_id,
                "source_ids": source_incident_ids,
            },
        )
        return Incident.from_api(data["incident"])

    # -------------------------------------------------------------------------
    # Incident Notes
    # -------------------------------------------------------------------------

    async def add_note(self, incident_id: str, content: str) -> IncidentNote:
        """
        Add a note to an incident.

        Args:
            incident_id: The incident ID
            content: Note content

        Returns:
            Created note
        """
        data = await self._request(
            "POST",
            f"/incidents/{incident_id}/notes",
            json={"note": {"content": content}},
        )

        logger.info(
            "Added note to incident",
            extra={"incident_id": incident_id, "note_length": len(content)},
        )
        return IncidentNote.from_api(data["note"])

    async def list_notes(self, incident_id: str) -> list[IncidentNote]:
        """
        List notes for an incident.

        Args:
            incident_id: The incident ID

        Returns:
            List of notes
        """
        data = await self._request("GET", f"/incidents/{incident_id}/notes")
        return [IncidentNote.from_api(n) for n in data.get("notes", [])]

    # -------------------------------------------------------------------------
    # Services
    # -------------------------------------------------------------------------

    async def list_services(
        self,
        query: str | None = None,
        include_disabled: bool = False,
        limit: int = 25,
        offset: int = 0,
    ) -> list[Service]:
        """
        List services.

        Args:
            query: Search query
            include_disabled: Include disabled services
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of services
        """
        params: dict[str, Any] = {
            "limit": min(limit, 100),
            "offset": offset,
        }

        if query:
            params["query"] = query

        if include_disabled:
            params["include[]"] = "disabled"

        data = await self._request("GET", "/services", params=params)
        return [Service.from_api(s) for s in data.get("services", [])]

    async def get_service(self, service_id: str) -> Service:
        """
        Get service by ID.

        Args:
            service_id: The service ID

        Returns:
            Service details
        """
        data = await self._request("GET", f"/services/{service_id}")
        return Service.from_api(data["service"])

    # -------------------------------------------------------------------------
    # On-Call Schedules
    # -------------------------------------------------------------------------

    async def get_on_call(
        self,
        schedule_ids: list[str] | None = None,
        escalation_policy_ids: list[str] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[OnCallSchedule]:
        """
        Get current on-call users.

        Args:
            schedule_ids: Filter by schedule IDs
            escalation_policy_ids: Filter by escalation policy IDs
            since: Start of time range
            until: End of time range

        Returns:
            List of on-call entries
        """
        params: dict[str, Any] = {}

        if schedule_ids:
            params["schedule_ids[]"] = schedule_ids

        if escalation_policy_ids:
            params["escalation_policy_ids[]"] = escalation_policy_ids

        if since:
            params["since"] = since.isoformat()

        if until:
            params["until"] = until.isoformat()

        data = await self._request("GET", "/oncalls", params=params)
        return [OnCallSchedule.from_api(o) for o in data.get("oncalls", [])]

    async def get_current_on_call_for_service(
        self,
        service_id: str,
    ) -> list[User]:
        """
        Get users currently on-call for a service.

        Args:
            service_id: The service ID

        Returns:
            List of on-call users
        """
        # First get the service to find its escalation policy
        service = await self.get_service(service_id)

        if not service.escalation_policy_id:
            return []

        # Get on-call for the escalation policy
        on_calls = await self.get_on_call(escalation_policy_ids=[service.escalation_policy_id])

        # Return unique users
        seen_ids: set[str] = set()
        users: list[User] = []

        for oc in on_calls:
            if oc.user.id not in seen_ids:
                seen_ids.add(oc.user.id)
                users.append(oc.user)

        return users

    # -------------------------------------------------------------------------
    # Users
    # -------------------------------------------------------------------------

    async def list_users(
        self,
        query: str | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> list[User]:
        """
        List users.

        Args:
            query: Search query
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of users
        """
        params: dict[str, Any] = {
            "limit": min(limit, 100),
            "offset": offset,
        }

        if query:
            params["query"] = query

        data = await self._request("GET", "/users", params=params)
        return [User.from_api(u) for u in data.get("users", [])]

    async def get_user(self, user_id: str) -> User:
        """
        Get user by ID.

        Args:
            user_id: The user ID

        Returns:
            User details
        """
        data = await self._request("GET", f"/users/{user_id}")
        return User.from_api(data["user"])

    # -------------------------------------------------------------------------
    # Webhooks
    # -------------------------------------------------------------------------

    def verify_webhook_signature(
        self,
        payload: bytes,
        signature: str,
    ) -> bool:
        """
        Verify a webhook signature.

        Args:
            payload: Raw webhook payload bytes
            signature: X-PagerDuty-Signature header value

        Returns:
            True if signature is valid
        """
        if not self.credentials.webhook_secret:
            logger.warning("No webhook secret configured, skipping verification")
            return True

        # PagerDuty uses HMAC-SHA256
        expected = hmac.new(
            self.credentials.webhook_secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        # Signature format: v1=<hex>
        if signature.startswith("v1="):
            signature = signature[3:]

        return hmac.compare_digest(expected, signature)

    def parse_webhook(self, data: dict[str, Any]) -> WebhookPayload:
        """
        Parse a webhook payload.

        Args:
            data: Webhook JSON data

        Returns:
            Parsed webhook payload
        """
        event = data.get("event", {})
        event_type = event.get("event_type", "unknown")

        incident_data = event.get("data")
        incident = None

        if incident_data and incident_data.get("type") == "incident":
            incident = Incident.from_api(incident_data)

        return WebhookPayload(
            event_type=event_type,
            incident=incident,
            raw_data=data,
        )

    # -------------------------------------------------------------------------
    # Integration Helpers
    # -------------------------------------------------------------------------

    async def create_incident_from_finding(
        self,
        title: str,
        service_id: str,
        severity: str,
        description: str,
        source: str,
        finding_id: str | None = None,
        file_path: str | None = None,
        line_number: int | None = None,
    ) -> Incident:
        """
        Create an incident from a security or bug finding.

        This is a convenience method for creating incidents from SAST,
        bug detector, or other analysis findings.

        Args:
            title: Finding title
            service_id: PagerDuty service ID to create incident for
            severity: Severity level (critical, high, medium, low)
            description: Finding description
            source: Source of finding (e.g., "sast_scanner", "bug_detector")
            finding_id: Optional unique finding ID for deduplication
            file_path: Optional file path where finding was detected
            line_number: Optional line number

        Returns:
            Created incident
        """
        # Map severity to urgency
        urgency = (
            IncidentUrgency.HIGH
            if severity.lower() in ("critical", "high")
            else IncidentUrgency.LOW
        )

        # Build detailed description
        details = [description]

        if file_path:
            location = f"File: {file_path}"
            if line_number:
                location += f":{line_number}"
            details.append(location)

        details.append(f"Source: {source}")
        details.append(f"Severity: {severity}")

        full_description = "\n\n".join(details)

        # Use finding_id as incident key for deduplication
        incident_key = None
        if finding_id:
            incident_key = f"{source}:{finding_id}"

        return await self.create_incident(
            IncidentCreateRequest(
                title=f"[{severity.upper()}] {title}",
                service_id=service_id,
                urgency=urgency,
                description=full_description,
                incident_key=incident_key,
            )
        )

    async def add_investigation_update(
        self,
        incident_id: str,
        update: str,
        investigator: str | None = None,
    ) -> IncidentNote:
        """
        Add an investigation update to an incident.

        Args:
            incident_id: The incident ID
            update: Investigation update text
            investigator: Optional investigator name

        Returns:
            Created note
        """
        content = f"Investigation Update:\n{update}"

        if investigator:
            content = f"[{investigator}] {content}"

        return await self.add_note(incident_id, content)

    async def resolve_with_runbook(
        self,
        incident_id: str,
        runbook_steps: list[str],
        resolution_summary: str,
    ) -> Incident:
        """
        Resolve an incident with runbook documentation.

        Args:
            incident_id: The incident ID
            runbook_steps: Steps taken from runbook
            resolution_summary: Brief resolution summary

        Returns:
            Resolved incident
        """
        # Document the runbook steps taken
        steps_text = "\n".join(f"  {i + 1}. {step}" for i, step in enumerate(runbook_steps))
        await self.add_note(
            incident_id,
            f"Runbook Steps Executed:\n{steps_text}",
        )

        # Resolve with summary
        return await self.resolve_incident(incident_id, resolution_summary)


# -----------------------------------------------------------------------------
# Mock Data for Testing
# -----------------------------------------------------------------------------


def get_mock_service() -> Service:
    """Get a mock service for testing."""
    return Service(
        id="PSERVICE01",
        name="Production API",
        description="Main production API service",
        status=ServiceStatus.ACTIVE,
        escalation_policy_id="PESCPOL01",
        html_url="https://example.pagerduty.com/services/PSERVICE01",
        created_at=datetime(2024, 1, 1, 0, 0, 0),
    )


def get_mock_user() -> User:
    """Get a mock user for testing."""
    return User(
        id="PUSER01",
        name="John Doe",
        email="john.doe@example.com",
        role="admin",
        html_url="https://example.pagerduty.com/users/PUSER01",
    )


def get_mock_incident() -> Incident:
    """Get a mock incident for testing."""
    return Incident(
        id="PINC001",
        incident_number=1234,
        title="Critical: Database connection pool exhausted",
        status=IncidentStatus.TRIGGERED,
        urgency=IncidentUrgency.HIGH,
        service=get_mock_service(),
        assigned_to=[get_mock_user()],
        priority=IncidentPriority.P1,
        html_url="https://example.pagerduty.com/incidents/PINC001",
        description="Connection pool at 100% capacity, requests failing",
        created_at=datetime(2024, 6, 15, 10, 30, 0),
    )


def get_mock_on_call() -> OnCallSchedule:
    """Get a mock on-call schedule for testing."""
    return OnCallSchedule(
        user=get_mock_user(),
        schedule_id="PSCHED01",
        schedule_name="Primary On-Call",
        start=datetime(2024, 6, 15, 0, 0, 0),
        end=datetime(2024, 6, 22, 0, 0, 0),
        escalation_level=1,
    )
