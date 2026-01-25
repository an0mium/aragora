"""
Pipedrive CRM Connector.

Full integration with Pipedrive CRM API:
- Deals (pipeline management)
- Persons (contacts)
- Organizations (companies)
- Activities (calls, meetings, tasks)
- Products
- Notes
- Webhooks for real-time updates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class DealStatus(str, Enum):
    """Pipedrive deal status."""

    OPEN = "open"
    WON = "won"
    LOST = "lost"
    DELETED = "deleted"


class ActivityType(str, Enum):
    """Pipedrive activity types."""

    CALL = "call"
    MEETING = "meeting"
    TASK = "task"
    DEADLINE = "deadline"
    EMAIL = "email"
    LUNCH = "lunch"


class PersonVisibility(str, Enum):
    """Visibility levels."""

    OWNER = "1"  # Owner only
    OWNER_FOLLOWERS = "3"  # Owner and followers
    ENTIRE_COMPANY = "5"  # Entire company


# =============================================================================
# Credentials
# =============================================================================


@dataclass
class PipedriveCredentials:
    """Pipedrive API credentials."""

    api_token: str
    base_url: str = "https://api.pipedrive.com/v1"

    @classmethod
    def from_env(cls, prefix: str = "PIPEDRIVE_") -> "PipedriveCredentials":
        """Load credentials from environment variables."""
        import os

        api_token = os.environ.get(f"{prefix}API_TOKEN", "")
        base_url = os.environ.get(f"{prefix}BASE_URL", "https://api.pipedrive.com/v1")

        if not api_token:
            raise ValueError(f"Missing {prefix}API_TOKEN environment variable")

        return cls(api_token=api_token, base_url=base_url)


# =============================================================================
# Error Handling
# =============================================================================


class PipedriveError(Exception):
    """Pipedrive API error."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class Person:
    """Pipedrive person (contact)."""

    id: int
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    org_id: Optional[int] = None
    org_name: Optional[str] = None
    owner_id: Optional[int] = None
    visible_to: str = PersonVisibility.ENTIRE_COMPANY.value
    add_time: Optional[datetime] = None
    update_time: Optional[datetime] = None
    active_flag: bool = True
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "Person":
        """Parse from API response."""
        # Extract primary email
        emails = data.get("email", [])
        primary_email = None
        if emails and isinstance(emails, list):
            for e in emails:
                if isinstance(e, dict) and e.get("primary"):
                    primary_email = e.get("value")
                    break
            if not primary_email and emails:
                primary_email = emails[0].get("value") if isinstance(emails[0], dict) else emails[0]

        # Extract primary phone
        phones = data.get("phone", [])
        primary_phone = None
        if phones and isinstance(phones, list):
            for p in phones:
                if isinstance(p, dict) and p.get("primary"):
                    primary_phone = p.get("value")
                    break
            if not primary_phone and phones:
                primary_phone = phones[0].get("value") if isinstance(phones[0], dict) else phones[0]

        return cls(
            id=data["id"],
            name=data.get("name", ""),
            email=primary_email,
            phone=primary_phone,
            org_id=data.get("org_id"),
            org_name=data.get("org_name"),
            owner_id=data.get("owner_id"),
            visible_to=str(data.get("visible_to", "5")),
            add_time=_parse_datetime(data.get("add_time")),
            update_time=_parse_datetime(data.get("update_time")),
            active_flag=data.get("active_flag", True),
        )

    def to_api(self) -> Dict[str, Any]:
        """Convert to API request format."""
        result: Dict[str, Any] = {"name": self.name}

        if self.email:
            result["email"] = [{"value": self.email, "primary": True}]
        if self.phone:
            result["phone"] = [{"value": self.phone, "primary": True}]
        if self.org_id:
            result["org_id"] = self.org_id
        if self.owner_id:
            result["owner_id"] = self.owner_id
        if self.visible_to:
            result["visible_to"] = self.visible_to

        return result


@dataclass
class Organization:
    """Pipedrive organization (company)."""

    id: int
    name: str
    address: Optional[str] = None
    owner_id: Optional[int] = None
    visible_to: str = PersonVisibility.ENTIRE_COMPANY.value
    add_time: Optional[datetime] = None
    update_time: Optional[datetime] = None
    active_flag: bool = True
    people_count: int = 0
    open_deals_count: int = 0
    won_deals_count: int = 0
    lost_deals_count: int = 0
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "Organization":
        """Parse from API response."""
        return cls(
            id=data["id"],
            name=data.get("name", ""),
            address=data.get("address"),
            owner_id=data.get("owner_id"),
            visible_to=str(data.get("visible_to", "5")),
            add_time=_parse_datetime(data.get("add_time")),
            update_time=_parse_datetime(data.get("update_time")),
            active_flag=data.get("active_flag", True),
            people_count=data.get("people_count", 0),
            open_deals_count=data.get("open_deals_count", 0),
            won_deals_count=data.get("won_deals_count", 0),
            lost_deals_count=data.get("lost_deals_count", 0),
        )

    def to_api(self) -> Dict[str, Any]:
        """Convert to API request format."""
        result: Dict[str, Any] = {"name": self.name}

        if self.address:
            result["address"] = self.address
        if self.owner_id:
            result["owner_id"] = self.owner_id
        if self.visible_to:
            result["visible_to"] = self.visible_to

        return result


@dataclass
class Pipeline:
    """Pipedrive pipeline."""

    id: int
    name: str
    url_title: str = ""
    order_nr: int = 0
    active: bool = True
    deal_probability: bool = False
    add_time: Optional[datetime] = None
    update_time: Optional[datetime] = None

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "Pipeline":
        """Parse from API response."""
        return cls(
            id=data["id"],
            name=data.get("name", ""),
            url_title=data.get("url_title", ""),
            order_nr=data.get("order_nr", 0),
            active=data.get("active", True),
            deal_probability=data.get("deal_probability", False),
            add_time=_parse_datetime(data.get("add_time")),
            update_time=_parse_datetime(data.get("update_time")),
        )


@dataclass
class Stage:
    """Pipedrive pipeline stage."""

    id: int
    name: str
    pipeline_id: int
    order_nr: int = 0
    active_flag: bool = True
    deal_probability: int = 100
    rotten_flag: bool = False
    rotten_days: Optional[int] = None
    add_time: Optional[datetime] = None
    update_time: Optional[datetime] = None

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "Stage":
        """Parse from API response."""
        return cls(
            id=data["id"],
            name=data.get("name", ""),
            pipeline_id=data.get("pipeline_id", 0),
            order_nr=data.get("order_nr", 0),
            active_flag=data.get("active_flag", True),
            deal_probability=data.get("deal_probability", 100),
            rotten_flag=data.get("rotten_flag", False),
            rotten_days=data.get("rotten_days"),
            add_time=_parse_datetime(data.get("add_time")),
            update_time=_parse_datetime(data.get("update_time")),
        )


@dataclass
class Deal:
    """Pipedrive deal."""

    id: int
    title: str
    value: float = 0.0
    currency: str = "USD"
    status: DealStatus = DealStatus.OPEN
    stage_id: Optional[int] = None
    pipeline_id: Optional[int] = None
    person_id: Optional[int] = None
    person_name: Optional[str] = None
    org_id: Optional[int] = None
    org_name: Optional[str] = None
    owner_id: Optional[int] = None
    expected_close_date: Optional[datetime] = None
    won_time: Optional[datetime] = None
    lost_time: Optional[datetime] = None
    close_time: Optional[datetime] = None
    add_time: Optional[datetime] = None
    update_time: Optional[datetime] = None
    probability: Optional[float] = None
    lost_reason: Optional[str] = None
    visible_to: str = PersonVisibility.ENTIRE_COMPANY.value
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "Deal":
        """Parse from API response."""
        return cls(
            id=data["id"],
            title=data.get("title", ""),
            value=float(data.get("value", 0)),
            currency=data.get("currency", "USD"),
            status=DealStatus(data.get("status", "open")),
            stage_id=data.get("stage_id"),
            pipeline_id=data.get("pipeline_id"),
            person_id=data.get("person_id"),
            person_name=data.get("person_name"),
            org_id=data.get("org_id"),
            org_name=data.get("org_name"),
            owner_id=data.get("owner_id"),
            expected_close_date=_parse_date(data.get("expected_close_date")),
            won_time=_parse_datetime(data.get("won_time")),
            lost_time=_parse_datetime(data.get("lost_time")),
            close_time=_parse_datetime(data.get("close_time")),
            add_time=_parse_datetime(data.get("add_time")),
            update_time=_parse_datetime(data.get("update_time")),
            probability=data.get("probability"),
            lost_reason=data.get("lost_reason"),
            visible_to=str(data.get("visible_to", "5")),
        )

    def to_api(self) -> Dict[str, Any]:
        """Convert to API request format."""
        result: Dict[str, Any] = {"title": self.title}

        if self.value:
            result["value"] = self.value
        if self.currency:
            result["currency"] = self.currency
        if self.stage_id:
            result["stage_id"] = self.stage_id
        if self.pipeline_id:
            result["pipeline_id"] = self.pipeline_id
        if self.person_id:
            result["person_id"] = self.person_id
        if self.org_id:
            result["org_id"] = self.org_id
        if self.owner_id:
            result["owner_id"] = self.owner_id
        if self.expected_close_date:
            result["expected_close_date"] = self.expected_close_date.strftime("%Y-%m-%d")
        if self.probability is not None:
            result["probability"] = self.probability
        if self.visible_to:
            result["visible_to"] = self.visible_to

        return result


@dataclass
class Activity:
    """Pipedrive activity (call, meeting, task, etc.)."""

    id: int
    type: str
    subject: str
    done: bool = False
    due_date: Optional[datetime] = None
    due_time: Optional[str] = None
    duration: Optional[str] = None  # HH:MM format
    deal_id: Optional[int] = None
    person_id: Optional[int] = None
    org_id: Optional[int] = None
    owner_id: Optional[int] = None
    note: Optional[str] = None
    location: Optional[str] = None
    public_description: Optional[str] = None
    add_time: Optional[datetime] = None
    update_time: Optional[datetime] = None
    marked_as_done_time: Optional[datetime] = None

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "Activity":
        """Parse from API response."""
        return cls(
            id=data["id"],
            type=data.get("type", ""),
            subject=data.get("subject", ""),
            done=data.get("done", False),
            due_date=_parse_date(data.get("due_date")),
            due_time=data.get("due_time"),
            duration=data.get("duration"),
            deal_id=data.get("deal_id"),
            person_id=data.get("person_id"),
            org_id=data.get("org_id"),
            owner_id=data.get("user_id") or data.get("owner_id"),
            note=data.get("note"),
            location=data.get("location"),
            public_description=data.get("public_description"),
            add_time=_parse_datetime(data.get("add_time")),
            update_time=_parse_datetime(data.get("update_time")),
            marked_as_done_time=_parse_datetime(data.get("marked_as_done_time")),
        )

    def to_api(self) -> Dict[str, Any]:
        """Convert to API request format."""
        result: Dict[str, Any] = {
            "type": self.type,
            "subject": self.subject,
            "done": 1 if self.done else 0,
        }

        if self.due_date:
            result["due_date"] = self.due_date.strftime("%Y-%m-%d")
        if self.due_time:
            result["due_time"] = self.due_time
        if self.duration:
            result["duration"] = self.duration
        if self.deal_id:
            result["deal_id"] = self.deal_id
        if self.person_id:
            result["person_id"] = self.person_id
        if self.org_id:
            result["org_id"] = self.org_id
        if self.owner_id:
            result["user_id"] = self.owner_id
        if self.note:
            result["note"] = self.note
        if self.location:
            result["location"] = self.location

        return result


@dataclass
class Note:
    """Pipedrive note."""

    id: int
    content: str
    deal_id: Optional[int] = None
    person_id: Optional[int] = None
    org_id: Optional[int] = None
    user_id: Optional[int] = None
    add_time: Optional[datetime] = None
    update_time: Optional[datetime] = None
    pinned_to_deal_flag: bool = False
    pinned_to_person_flag: bool = False
    pinned_to_organization_flag: bool = False

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "Note":
        """Parse from API response."""
        return cls(
            id=data["id"],
            content=data.get("content", ""),
            deal_id=data.get("deal_id"),
            person_id=data.get("person_id"),
            org_id=data.get("org_id"),
            user_id=data.get("user_id"),
            add_time=_parse_datetime(data.get("add_time")),
            update_time=_parse_datetime(data.get("update_time")),
            pinned_to_deal_flag=data.get("pinned_to_deal_flag", False),
            pinned_to_person_flag=data.get("pinned_to_person_flag", False),
            pinned_to_organization_flag=data.get("pinned_to_organization_flag", False),
        )

    def to_api(self) -> Dict[str, Any]:
        """Convert to API request format."""
        result: Dict[str, Any] = {"content": self.content}

        if self.deal_id:
            result["deal_id"] = self.deal_id
        if self.person_id:
            result["person_id"] = self.person_id
        if self.org_id:
            result["org_id"] = self.org_id
        if self.pinned_to_deal_flag:
            result["pinned_to_deal_flag"] = 1
        if self.pinned_to_person_flag:
            result["pinned_to_person_flag"] = 1
        if self.pinned_to_organization_flag:
            result["pinned_to_organization_flag"] = 1

        return result


@dataclass
class Product:
    """Pipedrive product."""

    id: int
    name: str
    code: Optional[str] = None
    description: Optional[str] = None
    unit: Optional[str] = None
    tax: float = 0.0
    active_flag: bool = True
    selectable: bool = True
    first_char: Optional[str] = None
    visible_to: str = PersonVisibility.ENTIRE_COMPANY.value
    owner_id: Optional[int] = None
    prices: List[Dict[str, Any]] = field(default_factory=list)
    add_time: Optional[datetime] = None
    update_time: Optional[datetime] = None

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "Product":
        """Parse from API response."""
        return cls(
            id=data["id"],
            name=data.get("name", ""),
            code=data.get("code"),
            description=data.get("description"),
            unit=data.get("unit"),
            tax=float(data.get("tax", 0)),
            active_flag=data.get("active_flag", True),
            selectable=data.get("selectable", True),
            first_char=data.get("first_char"),
            visible_to=str(data.get("visible_to", "5")),
            owner_id=data.get("owner_id"),
            prices=data.get("prices", []),
            add_time=_parse_datetime(data.get("add_time")),
            update_time=_parse_datetime(data.get("update_time")),
        )

    def to_api(self) -> Dict[str, Any]:
        """Convert to API request format."""
        result: Dict[str, Any] = {"name": self.name}

        if self.code:
            result["code"] = self.code
        if self.description:
            result["description"] = self.description
        if self.unit:
            result["unit"] = self.unit
        if self.tax:
            result["tax"] = self.tax
        if self.visible_to:
            result["visible_to"] = self.visible_to
        if self.owner_id:
            result["owner_id"] = self.owner_id
        if self.prices:
            result["prices"] = self.prices

        return result


@dataclass
class User:
    """Pipedrive user (owner)."""

    id: int
    name: str
    email: str
    active_flag: bool = True
    is_admin: bool = False
    is_you: bool = False
    role_id: Optional[int] = None
    timezone_name: Optional[str] = None
    icon_url: Optional[str] = None
    created: Optional[datetime] = None
    modified: Optional[datetime] = None

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "User":
        """Parse from API response."""
        return cls(
            id=data["id"],
            name=data.get("name", ""),
            email=data.get("email", ""),
            active_flag=data.get("active_flag", True),
            is_admin=data.get("is_admin", False),
            is_you=data.get("is_you", False),
            role_id=data.get("role_id"),
            timezone_name=data.get("timezone_name"),
            icon_url=data.get("icon_url"),
            created=_parse_datetime(data.get("created")),
            modified=_parse_datetime(data.get("modified")),
        )


# =============================================================================
# Helper Functions
# =============================================================================


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse datetime from API response."""
    if not value:
        return None
    try:
        # Pipedrive uses format: 2023-06-15 10:30:00
        if "T" in value:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    """Parse date from API response."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


# =============================================================================
# Pipedrive Client
# =============================================================================


class PipedriveClient:
    """
    Pipedrive CRM API client.

    Provides full access to Pipedrive CRM including:
    - Deals (pipeline management)
    - Persons (contacts)
    - Organizations (companies)
    - Activities (calls, meetings, tasks)
    - Products
    - Notes
    - Pipelines and stages
    - Users

    Example:
        async with PipedriveClient(credentials) as client:
            # Create a contact
            person = await client.create_person(name="John Doe", email="john@example.com")

            # Create a deal
            deal = await client.create_deal(
                title="New Sale",
                value=10000,
                person_id=person.id,
            )

            # Log an activity
            activity = await client.create_activity(
                type="call",
                subject="Follow-up call",
                deal_id=deal.id,
            )
    """

    def __init__(self, credentials: PipedriveCredentials):
        self.credentials = credentials
        self._client: Optional["httpx.AsyncClient"] = None

    async def __aenter__(self) -> "PipedriveClient":
        import httpx

        self._client = httpx.AsyncClient(
            base_url=self.credentials.base_url,
            timeout=httpx.Timeout(30.0),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make authenticated API request."""
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")

        # Add API token to params
        params = params or {}
        params["api_token"] = self.credentials.api_token

        response = await self._client.request(
            method,
            endpoint,
            params=params,
            json=json,
        )

        if response.status_code >= 400:
            error_data = response.json() if response.content else {}
            raise PipedriveError(
                message=error_data.get("error", f"HTTP {response.status_code}"),
                status_code=response.status_code,
                error_code=error_data.get("error_code"),
            )

        data = response.json()
        if not data.get("success", True):
            raise PipedriveError(
                message=data.get("error", "Unknown error"),
                error_code=data.get("error_code"),
            )

        return data

    # -------------------------------------------------------------------------
    # Persons (Contacts)
    # -------------------------------------------------------------------------

    async def get_persons(
        self,
        start: int = 0,
        limit: int = 100,
        filter_id: Optional[int] = None,
        sort: Optional[str] = None,
    ) -> List[Person]:
        """Get all persons with pagination."""
        params: Dict[str, Any] = {"start": start, "limit": limit}

        if filter_id:
            params["filter_id"] = filter_id
        if sort:
            params["sort"] = sort

        data = await self._request("GET", "/persons", params=params)
        persons = data.get("data") or []

        return [Person.from_api(p) for p in persons]

    async def get_person(self, person_id: int) -> Person:
        """Get a single person by ID."""
        data = await self._request("GET", f"/persons/{person_id}")
        return Person.from_api(data["data"])

    async def create_person(
        self,
        name: str,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        org_id: Optional[int] = None,
        owner_id: Optional[int] = None,
        visible_to: Optional[str] = None,
        **custom_fields,
    ) -> Person:
        """Create a new person."""
        body: Dict[str, Any] = {"name": name}

        if email:
            body["email"] = [{"value": email, "primary": True}]
        if phone:
            body["phone"] = [{"value": phone, "primary": True}]
        if org_id:
            body["org_id"] = org_id
        if owner_id:
            body["owner_id"] = owner_id
        if visible_to:
            body["visible_to"] = visible_to

        body.update(custom_fields)

        data = await self._request("POST", "/persons", json=body)
        return Person.from_api(data["data"])

    async def update_person(
        self,
        person_id: int,
        **properties,
    ) -> Person:
        """Update a person's properties."""
        data = await self._request("PUT", f"/persons/{person_id}", json=properties)
        return Person.from_api(data["data"])

    async def delete_person(self, person_id: int) -> bool:
        """Delete a person."""
        await self._request("DELETE", f"/persons/{person_id}")
        return True

    async def search_persons(
        self,
        term: str,
        fields: Optional[str] = None,
        limit: int = 100,
    ) -> List[Person]:
        """Search for persons."""
        params: Dict[str, Any] = {"term": term, "limit": limit}
        if fields:
            params["fields"] = fields

        data = await self._request("GET", "/persons/search", params=params)
        items = data.get("data", {}).get("items", [])

        return [Person.from_api(item["item"]) for item in items]

    # -------------------------------------------------------------------------
    # Organizations (Companies)
    # -------------------------------------------------------------------------

    async def get_organizations(
        self,
        start: int = 0,
        limit: int = 100,
        filter_id: Optional[int] = None,
        sort: Optional[str] = None,
    ) -> List[Organization]:
        """Get all organizations with pagination."""
        params: Dict[str, Any] = {"start": start, "limit": limit}

        if filter_id:
            params["filter_id"] = filter_id
        if sort:
            params["sort"] = sort

        data = await self._request("GET", "/organizations", params=params)
        orgs = data.get("data") or []

        return [Organization.from_api(o) for o in orgs]

    async def get_organization(self, org_id: int) -> Organization:
        """Get a single organization by ID."""
        data = await self._request("GET", f"/organizations/{org_id}")
        return Organization.from_api(data["data"])

    async def create_organization(
        self,
        name: str,
        address: Optional[str] = None,
        owner_id: Optional[int] = None,
        visible_to: Optional[str] = None,
        **custom_fields,
    ) -> Organization:
        """Create a new organization."""
        body: Dict[str, Any] = {"name": name}

        if address:
            body["address"] = address
        if owner_id:
            body["owner_id"] = owner_id
        if visible_to:
            body["visible_to"] = visible_to

        body.update(custom_fields)

        data = await self._request("POST", "/organizations", json=body)
        return Organization.from_api(data["data"])

    async def update_organization(
        self,
        org_id: int,
        **properties,
    ) -> Organization:
        """Update an organization's properties."""
        data = await self._request("PUT", f"/organizations/{org_id}", json=properties)
        return Organization.from_api(data["data"])

    async def delete_organization(self, org_id: int) -> bool:
        """Delete an organization."""
        await self._request("DELETE", f"/organizations/{org_id}")
        return True

    async def search_organizations(
        self,
        term: str,
        fields: Optional[str] = None,
        limit: int = 100,
    ) -> List[Organization]:
        """Search for organizations."""
        params: Dict[str, Any] = {"term": term, "limit": limit}
        if fields:
            params["fields"] = fields

        data = await self._request("GET", "/organizations/search", params=params)
        items = data.get("data", {}).get("items", [])

        return [Organization.from_api(item["item"]) for item in items]

    # -------------------------------------------------------------------------
    # Deals
    # -------------------------------------------------------------------------

    async def get_deals(
        self,
        start: int = 0,
        limit: int = 100,
        filter_id: Optional[int] = None,
        stage_id: Optional[int] = None,
        status: Optional[DealStatus] = None,
        sort: Optional[str] = None,
    ) -> List[Deal]:
        """Get all deals with pagination and filtering."""
        params: Dict[str, Any] = {"start": start, "limit": limit}

        if filter_id:
            params["filter_id"] = filter_id
        if stage_id:
            params["stage_id"] = stage_id
        if status:
            params["status"] = status.value
        if sort:
            params["sort"] = sort

        data = await self._request("GET", "/deals", params=params)
        deals = data.get("data") or []

        return [Deal.from_api(d) for d in deals]

    async def get_deal(self, deal_id: int) -> Deal:
        """Get a single deal by ID."""
        data = await self._request("GET", f"/deals/{deal_id}")
        return Deal.from_api(data["data"])

    async def create_deal(
        self,
        title: str,
        value: Optional[float] = None,
        currency: str = "USD",
        person_id: Optional[int] = None,
        org_id: Optional[int] = None,
        stage_id: Optional[int] = None,
        pipeline_id: Optional[int] = None,
        owner_id: Optional[int] = None,
        expected_close_date: Optional[datetime] = None,
        probability: Optional[float] = None,
        visible_to: Optional[str] = None,
        **custom_fields,
    ) -> Deal:
        """Create a new deal."""
        body: Dict[str, Any] = {"title": title, "currency": currency}

        if value is not None:
            body["value"] = value
        if person_id:
            body["person_id"] = person_id
        if org_id:
            body["org_id"] = org_id
        if stage_id:
            body["stage_id"] = stage_id
        if pipeline_id:
            body["pipeline_id"] = pipeline_id
        if owner_id:
            body["owner_id"] = owner_id
        if expected_close_date:
            body["expected_close_date"] = expected_close_date.strftime("%Y-%m-%d")
        if probability is not None:
            body["probability"] = probability
        if visible_to:
            body["visible_to"] = visible_to

        body.update(custom_fields)

        data = await self._request("POST", "/deals", json=body)
        return Deal.from_api(data["data"])

    async def update_deal(
        self,
        deal_id: int,
        **properties,
    ) -> Deal:
        """Update a deal's properties."""
        data = await self._request("PUT", f"/deals/{deal_id}", json=properties)
        return Deal.from_api(data["data"])

    async def delete_deal(self, deal_id: int) -> bool:
        """Delete a deal."""
        await self._request("DELETE", f"/deals/{deal_id}")
        return True

    async def move_deal_to_stage(self, deal_id: int, stage_id: int) -> Deal:
        """Move a deal to a different stage."""
        return await self.update_deal(deal_id, stage_id=stage_id)

    async def mark_deal_won(self, deal_id: int) -> Deal:
        """Mark a deal as won."""
        return await self.update_deal(deal_id, status=DealStatus.WON.value)

    async def mark_deal_lost(self, deal_id: int, lost_reason: Optional[str] = None) -> Deal:
        """Mark a deal as lost."""
        properties: Dict[str, Any] = {"status": DealStatus.LOST.value}
        if lost_reason:
            properties["lost_reason"] = lost_reason
        return await self.update_deal(deal_id, **properties)

    async def search_deals(
        self,
        term: str,
        fields: Optional[str] = None,
        limit: int = 100,
    ) -> List[Deal]:
        """Search for deals."""
        params: Dict[str, Any] = {"term": term, "limit": limit}
        if fields:
            params["fields"] = fields

        data = await self._request("GET", "/deals/search", params=params)
        items = data.get("data", {}).get("items", [])

        return [Deal.from_api(item["item"]) for item in items]

    # -------------------------------------------------------------------------
    # Pipelines & Stages
    # -------------------------------------------------------------------------

    async def get_pipelines(self) -> List[Pipeline]:
        """Get all pipelines."""
        data = await self._request("GET", "/pipelines")
        pipelines = data.get("data") or []
        return [Pipeline.from_api(p) for p in pipelines]

    async def get_pipeline(self, pipeline_id: int) -> Pipeline:
        """Get a single pipeline by ID."""
        data = await self._request("GET", f"/pipelines/{pipeline_id}")
        return Pipeline.from_api(data["data"])

    async def get_stages(self, pipeline_id: Optional[int] = None) -> List[Stage]:
        """Get all stages, optionally filtered by pipeline."""
        params: Dict[str, Any] = {}
        if pipeline_id:
            params["pipeline_id"] = pipeline_id

        data = await self._request("GET", "/stages", params=params)
        stages = data.get("data") or []
        return [Stage.from_api(s) for s in stages]

    async def get_stage(self, stage_id: int) -> Stage:
        """Get a single stage by ID."""
        data = await self._request("GET", f"/stages/{stage_id}")
        return Stage.from_api(data["data"])

    # -------------------------------------------------------------------------
    # Activities
    # -------------------------------------------------------------------------

    async def get_activities(
        self,
        start: int = 0,
        limit: int = 100,
        type: Optional[str] = None,
        user_id: Optional[int] = None,
        done: Optional[bool] = None,
    ) -> List[Activity]:
        """Get all activities with filtering."""
        params: Dict[str, Any] = {"start": start, "limit": limit}

        if type:
            params["type"] = type
        if user_id:
            params["user_id"] = user_id
        if done is not None:
            params["done"] = 1 if done else 0

        data = await self._request("GET", "/activities", params=params)
        activities = data.get("data") or []

        return [Activity.from_api(a) for a in activities]

    async def get_activity(self, activity_id: int) -> Activity:
        """Get a single activity by ID."""
        data = await self._request("GET", f"/activities/{activity_id}")
        return Activity.from_api(data["data"])

    async def create_activity(
        self,
        type: str,
        subject: str,
        done: bool = False,
        due_date: Optional[datetime] = None,
        due_time: Optional[str] = None,
        duration: Optional[str] = None,
        deal_id: Optional[int] = None,
        person_id: Optional[int] = None,
        org_id: Optional[int] = None,
        user_id: Optional[int] = None,
        note: Optional[str] = None,
        location: Optional[str] = None,
    ) -> Activity:
        """Create a new activity."""
        body: Dict[str, Any] = {
            "type": type,
            "subject": subject,
            "done": 1 if done else 0,
        }

        if due_date:
            body["due_date"] = due_date.strftime("%Y-%m-%d")
        if due_time:
            body["due_time"] = due_time
        if duration:
            body["duration"] = duration
        if deal_id:
            body["deal_id"] = deal_id
        if person_id:
            body["person_id"] = person_id
        if org_id:
            body["org_id"] = org_id
        if user_id:
            body["user_id"] = user_id
        if note:
            body["note"] = note
        if location:
            body["location"] = location

        data = await self._request("POST", "/activities", json=body)
        return Activity.from_api(data["data"])

    async def update_activity(
        self,
        activity_id: int,
        **properties,
    ) -> Activity:
        """Update an activity's properties."""
        data = await self._request("PUT", f"/activities/{activity_id}", json=properties)
        return Activity.from_api(data["data"])

    async def mark_activity_done(self, activity_id: int) -> Activity:
        """Mark an activity as done."""
        return await self.update_activity(activity_id, done=1)

    async def delete_activity(self, activity_id: int) -> bool:
        """Delete an activity."""
        await self._request("DELETE", f"/activities/{activity_id}")
        return True

    # -------------------------------------------------------------------------
    # Notes
    # -------------------------------------------------------------------------

    async def get_notes(
        self,
        deal_id: Optional[int] = None,
        person_id: Optional[int] = None,
        org_id: Optional[int] = None,
        start: int = 0,
        limit: int = 100,
    ) -> List[Note]:
        """Get notes, optionally filtered by deal/person/org."""
        params: Dict[str, Any] = {"start": start, "limit": limit}

        if deal_id:
            params["deal_id"] = deal_id
        if person_id:
            params["person_id"] = person_id
        if org_id:
            params["org_id"] = org_id

        data = await self._request("GET", "/notes", params=params)
        notes = data.get("data") or []

        return [Note.from_api(n) for n in notes]

    async def get_note(self, note_id: int) -> Note:
        """Get a single note by ID."""
        data = await self._request("GET", f"/notes/{note_id}")
        return Note.from_api(data["data"])

    async def create_note(
        self,
        content: str,
        deal_id: Optional[int] = None,
        person_id: Optional[int] = None,
        org_id: Optional[int] = None,
        pinned_to_deal: bool = False,
        pinned_to_person: bool = False,
        pinned_to_organization: bool = False,
    ) -> Note:
        """Create a new note."""
        body: Dict[str, Any] = {"content": content}

        if deal_id:
            body["deal_id"] = deal_id
        if person_id:
            body["person_id"] = person_id
        if org_id:
            body["org_id"] = org_id
        if pinned_to_deal:
            body["pinned_to_deal_flag"] = 1
        if pinned_to_person:
            body["pinned_to_person_flag"] = 1
        if pinned_to_organization:
            body["pinned_to_organization_flag"] = 1

        data = await self._request("POST", "/notes", json=body)
        return Note.from_api(data["data"])

    async def update_note(self, note_id: int, content: str) -> Note:
        """Update a note's content."""
        data = await self._request("PUT", f"/notes/{note_id}", json={"content": content})
        return Note.from_api(data["data"])

    async def delete_note(self, note_id: int) -> bool:
        """Delete a note."""
        await self._request("DELETE", f"/notes/{note_id}")
        return True

    # -------------------------------------------------------------------------
    # Products
    # -------------------------------------------------------------------------

    async def get_products(
        self,
        start: int = 0,
        limit: int = 100,
    ) -> List[Product]:
        """Get all products."""
        params: Dict[str, Any] = {"start": start, "limit": limit}

        data = await self._request("GET", "/products", params=params)
        products = data.get("data") or []

        return [Product.from_api(p) for p in products]

    async def get_product(self, product_id: int) -> Product:
        """Get a single product by ID."""
        data = await self._request("GET", f"/products/{product_id}")
        return Product.from_api(data["data"])

    async def create_product(
        self,
        name: str,
        code: Optional[str] = None,
        description: Optional[str] = None,
        unit: Optional[str] = None,
        tax: float = 0.0,
        prices: Optional[List[Dict[str, Any]]] = None,
        owner_id: Optional[int] = None,
        visible_to: Optional[str] = None,
    ) -> Product:
        """Create a new product."""
        body: Dict[str, Any] = {"name": name}

        if code:
            body["code"] = code
        if description:
            body["description"] = description
        if unit:
            body["unit"] = unit
        if tax:
            body["tax"] = tax
        if prices:
            body["prices"] = prices
        if owner_id:
            body["owner_id"] = owner_id
        if visible_to:
            body["visible_to"] = visible_to

        data = await self._request("POST", "/products", json=body)
        return Product.from_api(data["data"])

    async def update_product(
        self,
        product_id: int,
        **properties,
    ) -> Product:
        """Update a product's properties."""
        data = await self._request("PUT", f"/products/{product_id}", json=properties)
        return Product.from_api(data["data"])

    async def delete_product(self, product_id: int) -> bool:
        """Delete a product."""
        await self._request("DELETE", f"/products/{product_id}")
        return True

    async def search_products(
        self,
        term: str,
        limit: int = 100,
    ) -> List[Product]:
        """Search for products."""
        params: Dict[str, Any] = {"term": term, "limit": limit}

        data = await self._request("GET", "/products/search", params=params)
        items = data.get("data", {}).get("items", [])

        return [Product.from_api(item["item"]) for item in items]

    # -------------------------------------------------------------------------
    # Users
    # -------------------------------------------------------------------------

    async def get_users(self) -> List[User]:
        """Get all users."""
        data = await self._request("GET", "/users")
        users = data.get("data") or []
        return [User.from_api(u) for u in users]

    async def get_user(self, user_id: int) -> User:
        """Get a single user by ID."""
        data = await self._request("GET", f"/users/{user_id}")
        return User.from_api(data["data"])

    async def get_current_user(self) -> User:
        """Get the current authenticated user."""
        data = await self._request("GET", "/users/me")
        return User.from_api(data["data"])

    # -------------------------------------------------------------------------
    # Deal-Person/Organization Associations
    # -------------------------------------------------------------------------

    async def get_deal_persons(self, deal_id: int) -> List[Person]:
        """Get all persons associated with a deal."""
        data = await self._request("GET", f"/deals/{deal_id}/persons")
        persons = data.get("data") or []
        return [Person.from_api(p) for p in persons]

    async def get_person_deals(self, person_id: int) -> List[Deal]:
        """Get all deals associated with a person."""
        data = await self._request("GET", f"/persons/{person_id}/deals")
        deals = data.get("data") or []
        return [Deal.from_api(d) for d in deals]

    async def get_organization_deals(self, org_id: int) -> List[Deal]:
        """Get all deals associated with an organization."""
        data = await self._request("GET", f"/organizations/{org_id}/deals")
        deals = data.get("data") or []
        return [Deal.from_api(d) for d in deals]

    async def get_organization_persons(self, org_id: int) -> List[Person]:
        """Get all persons associated with an organization."""
        data = await self._request("GET", f"/organizations/{org_id}/persons")
        persons = data.get("data") or []
        return [Person.from_api(p) for p in persons]


# =============================================================================
# Mock Data Generators
# =============================================================================


def get_mock_person() -> Person:
    """Get a mock person for testing."""
    return Person(
        id=1,
        name="John Doe",
        email="john.doe@example.com",
        phone="+1-555-123-4567",
        org_id=1,
        org_name="Acme Corp",
        owner_id=1,
        add_time=datetime.now(timezone.utc),
        active_flag=True,
    )


def get_mock_organization() -> Organization:
    """Get a mock organization for testing."""
    return Organization(
        id=1,
        name="Acme Corporation",
        address="123 Main St, San Francisco, CA 94102",
        owner_id=1,
        people_count=5,
        open_deals_count=3,
        won_deals_count=10,
        add_time=datetime.now(timezone.utc),
        active_flag=True,
    )


def get_mock_deal() -> Deal:
    """Get a mock deal for testing."""
    return Deal(
        id=1,
        title="Enterprise License",
        value=50000.0,
        currency="USD",
        status=DealStatus.OPEN,
        stage_id=1,
        pipeline_id=1,
        person_id=1,
        person_name="John Doe",
        org_id=1,
        org_name="Acme Corp",
        owner_id=1,
        probability=75.0,
        add_time=datetime.now(timezone.utc),
    )


def get_mock_activity() -> Activity:
    """Get a mock activity for testing."""
    return Activity(
        id=1,
        type=ActivityType.CALL.value,
        subject="Discovery call",
        done=False,
        due_date=datetime.now(timezone.utc),
        due_time="14:00",
        duration="00:30",
        deal_id=1,
        person_id=1,
        owner_id=1,
        add_time=datetime.now(timezone.utc),
    )
