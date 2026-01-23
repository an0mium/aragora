"""
Help Scout Connector.

Integration with Help Scout API:
- Conversations (threads, notes, assignments)
- Customers
- Mailboxes and folders
- Users and teams
- Tags and custom fields
- Workflows and automation

Requires Help Scout OAuth2 app credentials.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class ConversationStatus(str, Enum):
    """Conversation status."""

    ACTIVE = "active"
    PENDING = "pending"
    CLOSED = "closed"
    SPAM = "spam"


class ConversationType(str, Enum):
    """Conversation type."""

    EMAIL = "email"
    CHAT = "chat"
    PHONE = "phone"


class ThreadType(str, Enum):
    """Thread (message) type."""

    CUSTOMER = "customer"
    MESSAGE = "message"  # Agent reply
    NOTE = "note"
    FORWARD = "forwardparent"
    REPLY_FORWARD = "reply"


class ThreadStatus(str, Enum):
    """Thread status."""

    ACTIVE = "active"
    CLOSED = "closed"
    PENDING = "pending"
    NO_CHANGE = "nochange"


@dataclass
class HelpScoutCredentials:
    """Help Scout API credentials."""

    client_id: str
    client_secret: str
    base_url: str = "https://api.helpscout.net/v2"


@dataclass
class HelpScoutCustomer:
    """Help Scout customer."""

    id: int
    first_name: str
    last_name: str
    emails: list[str] = field(default_factory=list)
    phones: list[dict[str, str]] = field(default_factory=list)
    organization: str | None = None
    job_title: str | None = None
    location: str | None = None
    photo_url: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> HelpScoutCustomer:
        """Create from API response."""
        emails = data.get("emails", [])
        email_list = [e.get("value", "") for e in emails] if emails else []

        return cls(
            id=data.get("id", 0),
            first_name=data.get("firstName", ""),
            last_name=data.get("lastName", ""),
            emails=email_list,
            phones=data.get("phones", []),
            organization=data.get("organization"),
            job_title=data.get("jobTitle"),
            location=data.get("location"),
            photo_url=data.get("photoUrl"),
            created_at=_parse_datetime(data.get("createdAt")),
            updated_at=_parse_datetime(data.get("updatedAt")),
        )


@dataclass
class User:
    """Help Scout user (agent)."""

    id: int
    first_name: str
    last_name: str
    email: str
    role: str = "user"
    timezone: str | None = None
    photo_url: str | None = None
    type: str = "user"
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> User:
        """Create from API response."""
        return cls(
            id=data.get("id", 0),
            first_name=data.get("firstName", ""),
            last_name=data.get("lastName", ""),
            email=data.get("email", ""),
            role=data.get("role", "user"),
            timezone=data.get("timezone"),
            photo_url=data.get("photoUrl"),
            type=data.get("type", "user"),
            created_at=_parse_datetime(data.get("createdAt")),
            updated_at=_parse_datetime(data.get("updatedAt")),
        )


@dataclass
class Mailbox:
    """Help Scout mailbox."""

    id: int
    name: str
    slug: str
    email: str
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Mailbox:
        """Create from API response."""
        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            slug=data.get("slug", ""),
            email=data.get("email", ""),
            created_at=_parse_datetime(data.get("createdAt")),
            updated_at=_parse_datetime(data.get("updatedAt")),
        )


@dataclass
class Thread:
    """Conversation thread (message/note)."""

    id: int
    type: ThreadType
    body: str
    status: ThreadStatus
    state: str
    source_type: str | None = None
    created_by_customer: bool = False
    customer_id: int | None = None
    assigned_to_id: int | None = None
    created_at: datetime | None = None
    attachments: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Thread:
        """Create from API response."""
        created_by = data.get("createdBy", {})
        return cls(
            id=data.get("id", 0),
            type=ThreadType(data.get("type", "message")),
            body=data.get("body", ""),
            status=ThreadStatus(data.get("status", "active")),
            state=data.get("state", "published"),
            source_type=data.get("source", {}).get("type"),
            created_by_customer=created_by.get("type") == "customer",
            customer_id=created_by.get("id") if created_by.get("type") == "customer" else None,
            assigned_to_id=data.get("assignedTo", {}).get("id"),
            created_at=_parse_datetime(data.get("createdAt")),
            attachments=data.get("_embedded", {}).get("attachments", []),
        )


@dataclass
class Conversation:
    """Help Scout conversation."""

    id: int
    number: int
    subject: str
    status: ConversationStatus
    type: ConversationType
    mailbox_id: int
    assignee_id: int | None = None
    customer_id: int | None = None
    customer_email: str | None = None
    tags: list[str] = field(default_factory=list)
    cc: list[str] = field(default_factory=list)
    bcc: list[str] = field(default_factory=list)
    preview: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    closed_at: datetime | None = None
    threads: list[Thread] = field(default_factory=list)
    custom_fields: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Conversation:
        """Create from API response."""
        primary_customer = data.get("primaryCustomer", {})
        assignee = data.get("assignee", {})

        return cls(
            id=data.get("id", 0),
            number=data.get("number", 0),
            subject=data.get("subject", ""),
            status=ConversationStatus(data.get("status", "active")),
            type=ConversationType(data.get("type", "email")),
            mailbox_id=data.get("mailboxId", 0),
            assignee_id=assignee.get("id") if assignee else None,
            customer_id=primary_customer.get("id"),
            customer_email=primary_customer.get("email"),
            tags=[t.get("tag", "") for t in data.get("tags", [])],
            cc=data.get("cc", []),
            bcc=data.get("bcc", []),
            preview=data.get("preview"),
            created_at=_parse_datetime(data.get("createdAt")),
            updated_at=_parse_datetime(data.get("updatedAt")),
            closed_at=_parse_datetime(data.get("closedAt")),
            custom_fields=data.get("customFields", []),
        )


@dataclass
class Folder:
    """Mailbox folder."""

    id: int
    name: str
    type: str
    user_id: int | None = None
    total_count: int = 0
    active_count: int = 0
    updated_at: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Folder:
        """Create from API response."""
        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            type=data.get("type", ""),
            user_id=data.get("userId"),
            total_count=data.get("totalCount", 0),
            active_count=data.get("activeCount", 0),
            updated_at=_parse_datetime(data.get("updatedAt")),
        )


class HelpScoutError(Exception):
    """Help Scout API error."""

    def __init__(self, message: str, status_code: int | None = None, details: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details or {}


class HelpScoutConnector:
    """
    Help Scout API connector.

    Provides integration with Help Scout for:
    - Conversation management
    - Customer management
    - Mailbox and folder management
    - User management
    """

    def __init__(self, credentials: HelpScoutCredentials):
        self.credentials = credentials
        self._client: httpx.AsyncClient | None = None
        self._access_token: str | None = None
        self._token_expires_at: datetime | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.credentials.base_url,
                timeout=30.0,
            )
        return self._client

    async def _ensure_token(self) -> str:
        """Ensure we have a valid access token."""
        if (
            self._access_token
            and self._token_expires_at
            and datetime.now() < self._token_expires_at
        ):
            return self._access_token

        client = await self._get_client()
        response = await client.post(
            "https://api.helpscout.net/v2/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self.credentials.client_id,
                "client_secret": self.credentials.client_secret,
            },
        )
        response.raise_for_status()
        data = response.json()

        self._access_token = data["access_token"]
        expires_in = int(data.get("expires_in", 7200))
        from datetime import timedelta

        self._token_expires_at = datetime.now() + timedelta(seconds=expires_in - 60)

        return self._access_token

    async def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        json_data: dict | None = None,
    ) -> dict[str, Any]:
        """Make authenticated API request."""
        token = await self._ensure_token()
        client = await self._get_client()

        response = await client.request(
            method,
            path,
            params=params,
            json=json_data,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )

        if response.status_code >= 400:
            try:
                error_data = response.json()
                raise HelpScoutError(
                    message=error_data.get("message", response.text),
                    status_code=response.status_code,
                    details=error_data,
                )
            except ValueError:
                raise HelpScoutError(
                    f"HTTP {response.status_code}: {response.text}",
                    status_code=response.status_code,
                )

        if response.status_code == 204 or response.status_code == 201:
            # Get resource URL from Location header if present
            location = response.headers.get("Resource-ID")
            return {"id": location} if location else {}

        return response.json()

    # =========================================================================
    # Conversations
    # =========================================================================

    async def get_conversations(
        self,
        mailbox_id: int | None = None,
        status: ConversationStatus | None = None,
        folder_id: int | None = None,
        assigned_to: int | None = None,
        tag: str | None = None,
        page: int = 1,
    ) -> tuple[list[Conversation], int]:
        """Get conversations. Returns (conversations, total_pages)."""
        params: dict[str, Any] = {"page": page}
        if mailbox_id:
            params["mailbox"] = mailbox_id
        if status:
            params["status"] = status.value
        if folder_id:
            params["folder"] = folder_id
        if assigned_to:
            params["assignedTo"] = assigned_to
        if tag:
            params["tag"] = tag

        data = await self._request("GET", "/conversations", params=params)
        conversations = [
            Conversation.from_api(c) for c in data.get("_embedded", {}).get("conversations", [])
        ]
        total_pages = data.get("page", {}).get("totalPages", 1)
        return conversations, total_pages

    async def get_conversation(self, conversation_id: int) -> Conversation:
        """Get a single conversation."""
        data = await self._request("GET", f"/conversations/{conversation_id}")
        return Conversation.from_api(data)

    async def get_conversation_threads(self, conversation_id: int) -> list[Thread]:
        """Get all threads for a conversation."""
        data = await self._request("GET", f"/conversations/{conversation_id}/threads")
        return [Thread.from_api(t) for t in data.get("_embedded", {}).get("threads", [])]

    async def create_conversation(
        self,
        mailbox_id: int,
        customer_email: str,
        subject: str,
        text: str,
        type: ConversationType = ConversationType.EMAIL,
        status: ConversationStatus = ConversationStatus.ACTIVE,
        tags: list[str] | None = None,
        assigned_to: int | None = None,
        cc: list[str] | None = None,
    ) -> int:
        """Create a new conversation. Returns the conversation ID."""
        conv_data: dict[str, Any] = {
            "type": type.value,
            "mailboxId": mailbox_id,
            "status": status.value,
            "subject": subject,
            "customer": {"email": customer_email},
            "threads": [
                {
                    "type": "customer",
                    "customer": {"email": customer_email},
                    "text": text,
                }
            ],
        }

        if tags:
            conv_data["tags"] = tags
        if assigned_to:
            conv_data["assignTo"] = assigned_to
        if cc:
            conv_data["cc"] = cc

        data = await self._request("POST", "/conversations", json_data=conv_data)
        return int(data.get("id", 0))

    async def reply_to_conversation(
        self,
        conversation_id: int,
        text: str,
        user_id: int,
        status: ThreadStatus = ThreadStatus.ACTIVE,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
    ) -> bool:
        """Reply to a conversation as an agent."""
        thread_data: dict[str, Any] = {
            "type": "reply",
            "text": text,
            "user": user_id,
            "status": status.value,
        }

        if cc:
            thread_data["cc"] = cc
        if bcc:
            thread_data["bcc"] = bcc

        await self._request(
            "POST",
            f"/conversations/{conversation_id}/reply",
            json_data=thread_data,
        )
        return True

    async def add_note(
        self,
        conversation_id: int,
        text: str,
        user_id: int,
    ) -> bool:
        """Add a private note to a conversation."""
        note_data: dict[str, Any] = {
            "type": "note",
            "text": text,
            "user": user_id,
        }

        await self._request(
            "POST",
            f"/conversations/{conversation_id}/notes",
            json_data=note_data,
        )
        return True

    async def update_conversation(
        self,
        conversation_id: int,
        op: str,
        path: str,
        value: Any,
    ) -> bool:
        """
        Update a conversation using JSON Patch.

        Example: update_conversation(123, "replace", "/status", "closed")
        """
        await self._request(
            "PATCH",
            f"/conversations/{conversation_id}",
            json_data={"op": op, "path": path, "value": value},
        )
        return True

    async def assign_conversation(self, conversation_id: int, user_id: int) -> bool:
        """Assign a conversation to a user."""
        return await self.update_conversation(conversation_id, "replace", "/assignTo", user_id)

    async def close_conversation(self, conversation_id: int) -> bool:
        """Close a conversation."""
        return await self.update_conversation(conversation_id, "replace", "/status", "closed")

    async def add_tags(self, conversation_id: int, tags: list[str]) -> bool:
        """Add tags to a conversation."""
        return await self.update_conversation(conversation_id, "replace", "/tags", tags)

    # =========================================================================
    # Customers
    # =========================================================================

    async def get_customers(
        self,
        email: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
        page: int = 1,
    ) -> tuple[list[HelpScoutCustomer], int]:
        """Get customers. Returns (customers, total_pages)."""
        params: dict[str, Any] = {"page": page}
        if email:
            params["email"] = email
        if first_name:
            params["firstName"] = first_name
        if last_name:
            params["lastName"] = last_name

        data = await self._request("GET", "/customers", params=params)
        customers = [
            HelpScoutCustomer.from_api(c) for c in data.get("_embedded", {}).get("customers", [])
        ]
        total_pages = data.get("page", {}).get("totalPages", 1)
        return customers, total_pages

    async def get_customer(self, customer_id: int) -> HelpScoutCustomer:
        """Get a single customer."""
        data = await self._request("GET", f"/customers/{customer_id}")
        return HelpScoutCustomer.from_api(data)

    async def create_customer(
        self,
        email: str,
        first_name: str,
        last_name: str,
        phone: str | None = None,
        organization: str | None = None,
        job_title: str | None = None,
    ) -> int:
        """Create a new customer. Returns the customer ID."""
        customer_data: dict[str, Any] = {
            "firstName": first_name,
            "lastName": last_name,
            "emails": [{"type": "work", "value": email}],
        }

        if phone:
            customer_data["phones"] = [{"type": "work", "value": phone}]
        if organization:
            customer_data["organization"] = organization
        if job_title:
            customer_data["jobTitle"] = job_title

        data = await self._request("POST", "/customers", json_data=customer_data)
        return int(data.get("id", 0))

    # =========================================================================
    # Mailboxes
    # =========================================================================

    async def get_mailboxes(self) -> list[Mailbox]:
        """Get all mailboxes."""
        data = await self._request("GET", "/mailboxes")
        return [Mailbox.from_api(m) for m in data.get("_embedded", {}).get("mailboxes", [])]

    async def get_mailbox(self, mailbox_id: int) -> Mailbox:
        """Get a single mailbox."""
        data = await self._request("GET", f"/mailboxes/{mailbox_id}")
        return Mailbox.from_api(data)

    async def get_mailbox_folders(self, mailbox_id: int) -> list[Folder]:
        """Get folders for a mailbox."""
        data = await self._request("GET", f"/mailboxes/{mailbox_id}/folders")
        return [Folder.from_api(f) for f in data.get("_embedded", {}).get("folders", [])]

    # =========================================================================
    # Users
    # =========================================================================

    async def get_users(self, page: int = 1) -> tuple[list[User], int]:
        """Get all users. Returns (users, total_pages)."""
        data = await self._request("GET", "/users", params={"page": page})
        users = [User.from_api(u) for u in data.get("_embedded", {}).get("users", [])]
        total_pages = data.get("page", {}).get("totalPages", 1)
        return users, total_pages

    async def get_user(self, user_id: int) -> User:
        """Get a single user."""
        data = await self._request("GET", f"/users/{user_id}")
        return User.from_api(data)

    async def get_current_user(self) -> User:
        """Get the authenticated user."""
        data = await self._request("GET", "/users/me")
        return User.from_api(data)

    # =========================================================================
    # Search
    # =========================================================================

    async def search_conversations(
        self,
        query: str,
        page: int = 1,
    ) -> tuple[list[Conversation], int]:
        """
        Search conversations.

        Query syntax: field:value AND/OR field:value
        Example: "status:active AND tag:urgent"
        """
        data = await self._request(
            "GET",
            "/conversations",
            params={"query": f"({query})", "page": page},
        )
        conversations = [
            Conversation.from_api(c) for c in data.get("_embedded", {}).get("conversations", [])
        ]
        total_pages = data.get("page", {}).get("totalPages", 1)
        return conversations, total_pages

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> HelpScoutConnector:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


def _parse_datetime(value: str | None) -> datetime | None:
    """Parse ISO datetime string."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def get_mock_conversation() -> Conversation:
    """Get a mock conversation for testing."""
    return Conversation(
        id=12345,
        number=1001,
        subject="Order not received",
        status=ConversationStatus.ACTIVE,
        type=ConversationType.EMAIL,
        mailbox_id=100,
        customer_email="customer@example.com",
        created_at=datetime.now(),
    )


def get_mock_customer() -> HelpScoutCustomer:
    """Get a mock customer for testing."""
    return HelpScoutCustomer(
        id=67890,
        first_name="Alice",
        last_name="Johnson",
        emails=["alice.johnson@example.com"],
    )
