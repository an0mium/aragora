"""
Intercom Connector.

Integration with Intercom API:
- Conversations (inbox, messaging)
- Contacts (leads and users)
- Companies
- Tags and segments
- Articles (help center)
- Teams and admins

Requires Intercom access token.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class ConversationState(str, Enum):
    """Conversation state."""

    OPEN = "open"
    CLOSED = "closed"
    SNOOZED = "snoozed"


class ContactRole(str, Enum):
    """Contact type."""

    USER = "user"
    LEAD = "lead"


class MessageType(str, Enum):
    """Message type."""

    COMMENT = "comment"
    NOTE = "note"
    ASSIGNMENT = "assignment"


@dataclass
class IntercomCredentials:
    """Intercom API credentials."""

    access_token: str
    base_url: str = "https://api.intercom.io"


@dataclass
class IntercomContact:
    """Intercom contact (user or lead)."""

    id: str
    external_id: str | None
    email: str | None
    name: str | None
    role: ContactRole
    phone: str | None = None
    avatar: str | None = None
    owner_id: int | None = None
    location_data: dict[str, Any] = field(default_factory=dict)
    custom_attributes: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    signed_up_at: datetime | None = None
    last_seen_at: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> IntercomContact:
        """Create from API response."""
        return cls(
            id=data.get("id", ""),
            external_id=data.get("external_id"),
            email=data.get("email"),
            name=data.get("name"),
            role=ContactRole(data.get("role", "user")),
            phone=data.get("phone"),
            avatar=data.get("avatar"),
            owner_id=data.get("owner_id"),
            location_data=data.get("location_data", {}),
            custom_attributes=data.get("custom_attributes", {}),
            tags=[t.get("name", "") for t in data.get("tags", {}).get("data", [])],
            created_at=_from_timestamp(data.get("created_at")),
            updated_at=_from_timestamp(data.get("updated_at")),
            signed_up_at=_from_timestamp(data.get("signed_up_at")),
            last_seen_at=_from_timestamp(data.get("last_seen_at")),
        )


@dataclass
class IntercomCompany:
    """Intercom company."""

    id: str
    company_id: str
    name: str
    plan: str | None = None
    size: int | None = None
    website: str | None = None
    industry: str | None = None
    monthly_spend: float | None = None
    session_count: int = 0
    user_count: int = 0
    custom_attributes: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> IntercomCompany:
        """Create from API response."""
        return cls(
            id=data.get("id", ""),
            company_id=data.get("company_id", ""),
            name=data.get("name", ""),
            plan=data.get("plan", {}).get("name") if data.get("plan") else None,
            size=data.get("size"),
            website=data.get("website"),
            industry=data.get("industry"),
            monthly_spend=data.get("monthly_spend"),
            session_count=data.get("session_count", 0),
            user_count=data.get("user_count", 0),
            custom_attributes=data.get("custom_attributes", {}),
            tags=[t.get("name", "") for t in data.get("tags", {}).get("data", [])],
            created_at=_from_timestamp(data.get("created_at")),
            updated_at=_from_timestamp(data.get("updated_at")),
        )


@dataclass
class ConversationPart:
    """Part of a conversation (message, note, etc.)."""

    id: str
    part_type: MessageType
    body: str | None
    author_type: str
    author_id: str
    author_name: str | None = None
    created_at: datetime | None = None
    attachments: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> ConversationPart:
        """Create from API response."""
        author = data.get("author", {})
        return cls(
            id=data.get("id", ""),
            part_type=MessageType(data.get("part_type", "comment")),
            body=data.get("body"),
            author_type=author.get("type", ""),
            author_id=author.get("id", ""),
            author_name=author.get("name"),
            created_at=_from_timestamp(data.get("created_at")),
            attachments=data.get("attachments", []),
        )


@dataclass
class Conversation:
    """Intercom conversation."""

    id: str
    title: str | None
    state: ConversationState
    open: bool
    read: bool
    priority: str
    source_type: str
    source_author_type: str | None = None
    source_author_id: str | None = None
    assignee_type: str | None = None
    assignee_id: str | None = None
    contacts: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    waiting_since: datetime | None = None
    snoozed_until: datetime | None = None
    first_contact_reply_at: datetime | None = None
    conversation_parts: list[ConversationPart] = field(default_factory=list)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Conversation:
        """Create from API response."""
        source = data.get("source", {})
        assignee = data.get("assignee", {})
        contacts = data.get("contacts", {}).get("contacts", [])

        return cls(
            id=data.get("id", ""),
            title=data.get("title"),
            state=ConversationState(data.get("state", "open")),
            open=data.get("open", True),
            read=data.get("read", False),
            priority=data.get("priority", "not_priority"),
            source_type=source.get("type", ""),
            source_author_type=source.get("author", {}).get("type"),
            source_author_id=source.get("author", {}).get("id"),
            assignee_type=assignee.get("type") if assignee else None,
            assignee_id=assignee.get("id") if assignee else None,
            contacts=[c.get("id", "") for c in contacts],
            tags=[t.get("name", "") for t in data.get("tags", {}).get("tags", [])],
            created_at=_from_timestamp(data.get("created_at")),
            updated_at=_from_timestamp(data.get("updated_at")),
            waiting_since=_from_timestamp(data.get("waiting_since")),
            snoozed_until=_from_timestamp(data.get("snoozed_until")),
            first_contact_reply_at=_from_timestamp(data.get("first_contact_reply_at")),
        )


@dataclass
class Admin:
    """Intercom admin/team member."""

    id: str
    name: str
    email: str
    type: str = "admin"
    away_mode_enabled: bool = False
    away_mode_reassign: bool = False
    has_inbox_seat: bool = True
    team_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Admin:
        """Create from API response."""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            email=data.get("email", ""),
            type=data.get("type", "admin"),
            away_mode_enabled=data.get("away_mode_enabled", False),
            away_mode_reassign=data.get("away_mode_reassign", False),
            has_inbox_seat=data.get("has_inbox_seat", True),
            team_ids=[t.get("id", "") for t in data.get("team_ids", [])],
        )


@dataclass
class Article:
    """Help center article."""

    id: str
    title: str
    body: str
    description: str | None = None
    author_id: str | None = None
    state: str = "draft"
    url: str | None = None
    parent_id: str | None = None
    parent_type: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Article:
        """Create from API response."""
        return cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            body=data.get("body", ""),
            description=data.get("description"),
            author_id=data.get("author_id"),
            state=data.get("state", "draft"),
            url=data.get("url"),
            parent_id=data.get("parent_id"),
            parent_type=data.get("parent_type"),
            created_at=_from_timestamp(data.get("created_at")),
            updated_at=_from_timestamp(data.get("updated_at")),
        )


class IntercomError(Exception):
    """Intercom API error."""

    def __init__(self, message: str, status_code: int | None = None, errors: list | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.errors = errors or []


class IntercomConnector:
    """
    Intercom API connector.

    Provides integration with Intercom for:
    - Conversation management (inbox)
    - Contact and company management
    - Help center articles
    - Admin and team management
    """

    def __init__(self, credentials: IntercomCredentials):
        self.credentials = credentials
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.credentials.base_url,
                headers={
                    "Authorization": f"Bearer {self.credentials.access_token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Intercom-Version": "2.10",
                },
                timeout=30.0,
            )
        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        json_data: dict | None = None,
    ) -> dict[str, Any]:
        """Make API request."""
        client = await self._get_client()
        response = await client.request(method, path, params=params, json=json_data)

        if response.status_code >= 400:
            try:
                error_data = response.json()
                raise IntercomError(
                    message=error_data.get("message", response.text),
                    status_code=response.status_code,
                    errors=error_data.get("errors", []),
                )
            except ValueError:
                raise IntercomError(
                    f"HTTP {response.status_code}: {response.text}",
                    status_code=response.status_code,
                )

        if response.status_code == 204:
            return {}
        return response.json()

    # =========================================================================
    # Conversations
    # =========================================================================

    async def get_conversations(
        self,
        state: ConversationState | None = None,
        assignee_id: str | None = None,
        per_page: int = 20,
        starting_after: str | None = None,
    ) -> tuple[list[Conversation], str | None]:
        """Get conversations. Returns (conversations, next_cursor)."""
        params: dict[str, Any] = {"per_page": min(per_page, 150)}
        if state:
            params["state"] = state.value
        if assignee_id:
            params["assignee_id"] = assignee_id
        if starting_after:
            params["starting_after"] = starting_after

        data = await self._request("GET", "/conversations", params=params)
        conversations = [Conversation.from_api(c) for c in data.get("conversations", [])]
        next_cursor = data.get("pages", {}).get("next", {}).get("starting_after")
        return conversations, next_cursor

    async def get_conversation(
        self, conversation_id: str, display_as: str = "plaintext"
    ) -> Conversation:
        """Get a single conversation with all parts."""
        data = await self._request(
            "GET",
            f"/conversations/{conversation_id}",
            params={"display_as": display_as},
        )
        conversation = Conversation.from_api(data)
        parts_data = data.get("conversation_parts", {}).get("conversation_parts", [])
        conversation.conversation_parts = [ConversationPart.from_api(p) for p in parts_data]
        return conversation

    async def reply_to_conversation(
        self,
        conversation_id: str,
        body: str,
        message_type: MessageType = MessageType.COMMENT,
        admin_id: str | None = None,
        attachment_urls: list[str] | None = None,
    ) -> Conversation:
        """Reply to a conversation."""
        reply_data: dict[str, Any] = {
            "message_type": message_type.value,
            "body": body,
            "type": "admin",
        }

        if admin_id:
            reply_data["admin_id"] = admin_id
        if attachment_urls:
            reply_data["attachment_urls"] = attachment_urls

        data = await self._request(
            "POST",
            f"/conversations/{conversation_id}/reply",
            json_data=reply_data,
        )
        return Conversation.from_api(data)

    async def close_conversation(
        self, conversation_id: str, admin_id: str, body: str | None = None
    ) -> Conversation:
        """Close a conversation."""
        close_data: dict[str, Any] = {
            "message_type": "close",
            "type": "admin",
            "admin_id": admin_id,
        }
        if body:
            close_data["body"] = body

        data = await self._request(
            "POST",
            f"/conversations/{conversation_id}/reply",
            json_data=close_data,
        )
        return Conversation.from_api(data)

    async def assign_conversation(
        self,
        conversation_id: str,
        admin_id: str,
        assignee_id: str,
        body: str | None = None,
    ) -> Conversation:
        """Assign conversation to an admin."""
        assign_data: dict[str, Any] = {
            "message_type": "assignment",
            "type": "admin",
            "admin_id": admin_id,
            "assignee_id": assignee_id,
        }
        if body:
            assign_data["body"] = body

        data = await self._request(
            "POST",
            f"/conversations/{conversation_id}/reply",
            json_data=assign_data,
        )
        return Conversation.from_api(data)

    async def snooze_conversation(
        self,
        conversation_id: str,
        admin_id: str,
        snoozed_until: datetime,
    ) -> Conversation:
        """Snooze a conversation until a specific time."""
        snooze_data: dict[str, Any] = {
            "message_type": "snoozed",
            "admin_id": admin_id,
            "snoozed_until": int(snoozed_until.timestamp()),
        }

        data = await self._request(
            "POST",
            f"/conversations/{conversation_id}/reply",
            json_data=snooze_data,
        )
        return Conversation.from_api(data)

    # =========================================================================
    # Contacts
    # =========================================================================

    async def get_contacts(
        self,
        per_page: int = 50,
        starting_after: str | None = None,
    ) -> tuple[list[IntercomContact], str | None]:
        """Get contacts. Returns (contacts, next_cursor)."""
        params: dict[str, Any] = {"per_page": min(per_page, 150)}
        if starting_after:
            params["starting_after"] = starting_after

        data = await self._request("GET", "/contacts", params=params)
        contacts = [IntercomContact.from_api(c) for c in data.get("data", [])]
        next_cursor = data.get("pages", {}).get("next", {}).get("starting_after")
        return contacts, next_cursor

    async def get_contact(self, contact_id: str) -> IntercomContact:
        """Get a single contact."""
        data = await self._request("GET", f"/contacts/{contact_id}")
        return IntercomContact.from_api(data)

    async def search_contacts(
        self,
        query: dict[str, Any],
        per_page: int = 50,
    ) -> list[IntercomContact]:
        """
        Search contacts using Intercom query syntax.

        Example query:
        {
            "field": "email",
            "operator": "=",
            "value": "john@example.com"
        }
        """
        data = await self._request(
            "POST",
            "/contacts/search",
            json_data={"query": query, "pagination": {"per_page": per_page}},
        )
        return [IntercomContact.from_api(c) for c in data.get("data", [])]

    async def create_contact(
        self,
        role: ContactRole = ContactRole.USER,
        email: str | None = None,
        external_id: str | None = None,
        name: str | None = None,
        phone: str | None = None,
        custom_attributes: dict[str, Any] | None = None,
    ) -> IntercomContact:
        """Create a new contact."""
        contact_data: dict[str, Any] = {"role": role.value}
        if email:
            contact_data["email"] = email
        if external_id:
            contact_data["external_id"] = external_id
        if name:
            contact_data["name"] = name
        if phone:
            contact_data["phone"] = phone
        if custom_attributes:
            contact_data["custom_attributes"] = custom_attributes

        data = await self._request("POST", "/contacts", json_data=contact_data)
        return IntercomContact.from_api(data)

    async def update_contact(
        self,
        contact_id: str,
        name: str | None = None,
        email: str | None = None,
        phone: str | None = None,
        custom_attributes: dict[str, Any] | None = None,
    ) -> IntercomContact:
        """Update a contact."""
        contact_data: dict[str, Any] = {}
        if name:
            contact_data["name"] = name
        if email:
            contact_data["email"] = email
        if phone:
            contact_data["phone"] = phone
        if custom_attributes:
            contact_data["custom_attributes"] = custom_attributes

        data = await self._request("PUT", f"/contacts/{contact_id}", json_data=contact_data)
        return IntercomContact.from_api(data)

    # =========================================================================
    # Companies
    # =========================================================================

    async def get_companies(
        self,
        per_page: int = 50,
        starting_after: str | None = None,
    ) -> tuple[list[IntercomCompany], str | None]:
        """Get companies. Returns (companies, next_cursor)."""
        params: dict[str, Any] = {"per_page": min(per_page, 150)}
        if starting_after:
            params["starting_after"] = starting_after

        data = await self._request("GET", "/companies", params=params)
        companies = [IntercomCompany.from_api(c) for c in data.get("data", [])]
        next_cursor = data.get("pages", {}).get("next", {}).get("starting_after")
        return companies, next_cursor

    async def get_company(self, company_id: str) -> IntercomCompany:
        """Get a single company."""
        data = await self._request("GET", f"/companies/{company_id}")
        return IntercomCompany.from_api(data)

    async def create_or_update_company(
        self,
        company_id: str,
        name: str | None = None,
        plan: str | None = None,
        size: int | None = None,
        website: str | None = None,
        industry: str | None = None,
        custom_attributes: dict[str, Any] | None = None,
    ) -> IntercomCompany:
        """Create or update a company."""
        company_data: dict[str, Any] = {"company_id": company_id}
        if name:
            company_data["name"] = name
        if plan:
            company_data["plan"] = plan
        if size:
            company_data["size"] = size
        if website:
            company_data["website"] = website
        if industry:
            company_data["industry"] = industry
        if custom_attributes:
            company_data["custom_attributes"] = custom_attributes

        data = await self._request("POST", "/companies", json_data=company_data)
        return IntercomCompany.from_api(data)

    # =========================================================================
    # Admins
    # =========================================================================

    async def get_admins(self) -> list[Admin]:
        """Get all admins."""
        data = await self._request("GET", "/admins")
        return [Admin.from_api(a) for a in data.get("admins", [])]

    async def get_admin(self, admin_id: str) -> Admin:
        """Get a single admin."""
        data = await self._request("GET", f"/admins/{admin_id}")
        return Admin.from_api(data)

    # =========================================================================
    # Articles
    # =========================================================================

    async def get_articles(
        self,
        per_page: int = 25,
        starting_after: str | None = None,
    ) -> tuple[list[Article], str | None]:
        """Get help center articles. Returns (articles, next_cursor)."""
        params: dict[str, Any] = {"per_page": min(per_page, 50)}
        if starting_after:
            params["starting_after"] = starting_after

        data = await self._request("GET", "/articles", params=params)
        articles = [Article.from_api(a) for a in data.get("data", [])]
        next_cursor = data.get("pages", {}).get("next", {}).get("starting_after")
        return articles, next_cursor

    async def get_article(self, article_id: str) -> Article:
        """Get a single article."""
        data = await self._request("GET", f"/articles/{article_id}")
        return Article.from_api(data)

    async def create_article(
        self,
        title: str,
        body: str,
        author_id: str,
        description: str | None = None,
        state: str = "draft",
        parent_id: str | None = None,
        parent_type: str | None = None,
    ) -> Article:
        """Create a help center article."""
        article_data: dict[str, Any] = {
            "title": title,
            "body": body,
            "author_id": author_id,
            "state": state,
        }
        if description:
            article_data["description"] = description
        if parent_id:
            article_data["parent_id"] = parent_id
        if parent_type:
            article_data["parent_type"] = parent_type

        data = await self._request("POST", "/articles", json_data=article_data)
        return Article.from_api(data)

    # =========================================================================
    # Tags
    # =========================================================================

    async def tag_contact(self, contact_id: str, tag_name: str) -> bool:
        """Add a tag to a contact."""
        await self._request(
            "POST",
            f"/contacts/{contact_id}/tags",
            json_data={"name": tag_name},
        )
        return True

    async def untag_contact(self, contact_id: str, tag_id: str) -> bool:
        """Remove a tag from a contact."""
        await self._request("DELETE", f"/contacts/{contact_id}/tags/{tag_id}")
        return True

    async def tag_conversation(self, conversation_id: str, tag_name: str, admin_id: str) -> bool:
        """Add a tag to a conversation."""
        await self._request(
            "POST",
            f"/conversations/{conversation_id}/tags",
            json_data={"name": tag_name, "admin_id": admin_id},
        )
        return True

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> IntercomConnector:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


def _from_timestamp(value: int | None) -> datetime | None:
    """Convert Unix timestamp to datetime."""
    if not value:
        return None
    try:
        return datetime.fromtimestamp(value)
    except (ValueError, OSError):
        return None


def get_mock_conversation() -> Conversation:
    """Get a mock conversation for testing."""
    return Conversation(
        id="12345",
        title="Help with billing",
        state=ConversationState.OPEN,
        open=True,
        read=False,
        priority="priority",
        source_type="email",
        contacts=["contact_123"],
        created_at=datetime.now(),
    )


def get_mock_contact() -> IntercomContact:
    """Get a mock contact for testing."""
    return IntercomContact(
        id="contact_123",
        external_id="user_456",
        email="user@example.com",
        name="Test User",
        role=ContactRole.USER,
    )
