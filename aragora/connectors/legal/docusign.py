"""
DocuSign E-Signature Connector.

Provides e-signature integration via DocuSign:
- OAuth 2.0 authentication (JWT or Authorization Code)
- Create and send envelopes for signature
- Track signing status
- Download signed documents
- Template-based envelope creation

Dependencies:
    pip install docusign-esign

Environment Variables:
    DOCUSIGN_INTEGRATION_KEY - DocuSign integration key (client ID)
    DOCUSIGN_USER_ID - DocuSign user ID for JWT auth
    DOCUSIGN_ACCOUNT_ID - DocuSign account ID
    DOCUSIGN_PRIVATE_KEY - Path to RSA private key for JWT auth
    DOCUSIGN_ENVIRONMENT - 'demo' or 'production'
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DocuSignEnvironment(str, Enum):
    """DocuSign environment."""

    DEMO = "demo"
    PRODUCTION = "production"


class EnvelopeStatus(str, Enum):
    """Envelope status values."""

    CREATED = "created"
    SENT = "sent"
    DELIVERED = "delivered"
    SIGNED = "signed"
    COMPLETED = "completed"
    DECLINED = "declined"
    VOIDED = "voided"


class RecipientType(str, Enum):
    """Types of envelope recipients."""

    SIGNER = "signer"
    CARBON_COPY = "carbon_copy"
    CERTIFIED_DELIVERY = "certified_delivery"
    IN_PERSON_SIGNER = "in_person_signer"
    EDITOR = "editor"
    AGENT = "agent"


@dataclass
class DocuSignCredentials:
    """OAuth credentials for DocuSign."""

    access_token: str
    account_id: str
    base_uri: str
    token_type: str = "Bearer"
    expires_at: Optional[datetime] = None
    refresh_token: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        """Check if access token is expired."""
        if not self.expires_at:
            return True
        return datetime.now(timezone.utc) >= self.expires_at


@dataclass
class Recipient:
    """A recipient of an envelope."""

    email: str
    name: str
    recipient_type: RecipientType = RecipientType.SIGNER
    routing_order: int = 1
    recipient_id: str = ""

    # For in-person signing
    host_email: Optional[str] = None
    host_name: Optional[str] = None

    # Signing options
    signing_group_id: Optional[str] = None
    access_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "email": self.email,
            "name": self.name,
            "recipient_type": self.recipient_type.value,
            "routing_order": self.routing_order,
            "recipient_id": self.recipient_id or str(self.routing_order),
        }


@dataclass
class SignatureTab:
    """A signature or other tab on a document."""

    tab_type: str = "signature"  # signature, initial, date_signed, text, etc.
    page_number: int = 1
    x_position: int = 100
    y_position: int = 100
    recipient_id: str = "1"
    optional: bool = False

    # For text tabs
    tab_label: str = ""
    value: str = ""
    width: int = 100
    height: int = 20

    def to_dict(self) -> Dict[str, Any]:
        base = {
            "pageNumber": str(self.page_number),
            "xPosition": str(self.x_position),
            "yPosition": str(self.y_position),
            "recipientId": self.recipient_id,
            "optional": str(self.optional).lower(),
        }
        if self.tab_type == "text":
            base.update(
                {
                    "tabLabel": self.tab_label,
                    "value": self.value,
                    "width": str(self.width),
                    "height": str(self.height),
                }
            )
        return base


@dataclass
class Document:
    """A document to be signed."""

    document_id: str
    name: str
    content: bytes  # PDF or other supported format
    file_extension: str = "pdf"
    order: int = 1

    def to_base64(self) -> str:
        """Convert content to base64."""
        return base64.b64encode(self.content).decode("utf-8")


@dataclass
class Envelope:
    """A DocuSign envelope (signature request)."""

    envelope_id: str
    status: EnvelopeStatus
    email_subject: str
    email_body: str = ""

    # Recipients
    signers: List[Dict[str, Any]] = field(default_factory=list)
    carbon_copies: List[Dict[str, Any]] = field(default_factory=list)

    # Documents
    documents: List[Dict[str, Any]] = field(default_factory=list)

    # Timestamps
    created_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    voided_at: Optional[datetime] = None

    # Metadata
    sender_name: str = ""
    sender_email: str = ""
    status_changed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "envelope_id": self.envelope_id,
            "status": self.status.value,
            "email_subject": self.email_subject,
            "email_body": self.email_body,
            "signers": self.signers,
            "carbon_copies": self.carbon_copies,
            "documents": self.documents,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "sender_name": self.sender_name,
            "sender_email": self.sender_email,
        }


@dataclass
class EnvelopeCreateRequest:
    """Request to create an envelope."""

    email_subject: str
    recipients: List[Recipient]
    documents: List[Document]
    email_body: str = ""
    status: str = "sent"  # 'created' for draft, 'sent' to send immediately
    tabs: Optional[List[SignatureTab]] = None

    # Options
    enforce_signer_visibility: bool = True
    brand_id: Optional[str] = None
    expire_enabled: bool = True
    expire_days: int = 30
    expire_warn: int = 5


class DocuSignConnector:
    """
    DocuSign e-signature integration connector.

    Supports both JWT and Authorization Code OAuth flows.
    """

    # API URLs
    DEMO_AUTH_URL = "https://account-d.docusign.com/oauth"
    PRODUCTION_AUTH_URL = "https://account.docusign.com/oauth"
    DEMO_API_URL = "https://demo.docusign.net/restapi"
    PRODUCTION_API_URL = "https://www.docusign.net/restapi"

    def __init__(
        self,
        integration_key: Optional[str] = None,
        user_id: Optional[str] = None,
        account_id: Optional[str] = None,
        private_key_path: Optional[str] = None,
        environment: Optional[DocuSignEnvironment] = None,
    ):
        """
        Initialize DocuSign connector.

        Args:
            integration_key: DocuSign integration key (client ID)
            user_id: DocuSign user ID for JWT auth
            account_id: DocuSign account ID
            private_key_path: Path to RSA private key for JWT auth
            environment: Demo or production environment
        """
        self.integration_key = integration_key or os.getenv("DOCUSIGN_INTEGRATION_KEY")
        self.user_id = user_id or os.getenv("DOCUSIGN_USER_ID")
        self.account_id = account_id or os.getenv("DOCUSIGN_ACCOUNT_ID")
        self.private_key_path = private_key_path or os.getenv("DOCUSIGN_PRIVATE_KEY")

        env_str = environment or os.getenv("DOCUSIGN_ENVIRONMENT", "demo")
        if isinstance(env_str, str):
            self.environment = (
                DocuSignEnvironment.PRODUCTION
                if env_str.lower() == "production"
                else DocuSignEnvironment.DEMO
            )
        else:
            self.environment = env_str

        # Set URLs based on environment
        if self.environment == DocuSignEnvironment.PRODUCTION:
            self.auth_url = self.PRODUCTION_AUTH_URL
            self.api_url = self.PRODUCTION_API_URL
        else:
            self.auth_url = self.DEMO_AUTH_URL
            self.api_url = self.DEMO_API_URL

        self._credentials: Optional[DocuSignCredentials] = None

    @property
    def is_configured(self) -> bool:
        """Check if connector is configured."""
        return bool(self.integration_key and self.user_id and self.account_id)

    @property
    def is_authenticated(self) -> bool:
        """Check if connector has valid credentials."""
        return self._credentials is not None and not self._credentials.is_expired

    async def authenticate_jwt(self) -> DocuSignCredentials:
        """
        Authenticate using JWT grant.

        Requires private key file.
        """
        import aiohttp
        import jwt
        import time

        if not self.private_key_path:
            raise ValueError("Private key path required for JWT authentication")

        # Read private key
        with open(self.private_key_path) as f:
            private_key = f.read()

        # Create JWT
        now = int(time.time())
        payload = {
            "iss": self.integration_key,
            "sub": self.user_id,
            "aud": self.auth_url.replace("https://", "").replace("/oauth", ""),
            "iat": now,
            "exp": now + 3600,
            "scope": "signature impersonation",
        }

        token = jwt.encode(payload, private_key, algorithm="RS256")

        # Exchange for access token
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.auth_url}/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "assertion": token,
                },
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"JWT auth failed: {error_text}")

                data = await response.json()

                self._credentials = DocuSignCredentials(
                    access_token=data["access_token"],
                    account_id=self.account_id or "",
                    base_uri=self.api_url,
                    token_type=data.get("token_type", "Bearer"),
                    expires_at=datetime.now(timezone.utc)
                    + timedelta(seconds=data.get("expires_in", 3600)),
                )

                # Get user info to confirm account
                if not self.account_id:
                    user_info = await self._get_user_info()
                    if user_info.get("accounts"):
                        account = user_info["accounts"][0]
                        self._credentials.account_id = account["account_id"]
                        self._credentials.base_uri = account["base_uri"]

                return self._credentials

    async def _get_user_info(self) -> Dict[str, Any]:
        """Get user info including accounts."""
        import aiohttp

        if not self._credentials:
            raise Exception("Not authenticated")

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.auth_url}/userinfo",
                headers={"Authorization": f"Bearer {self._credentials.access_token}"},
            ) as response:
                return await response.json()

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        raw_response: bool = False,
    ) -> Any:
        """Make authenticated API request."""
        if not self._credentials:
            raise Exception("Not authenticated")

        if self._credentials.is_expired:
            await self.authenticate_jwt()

        import aiohttp

        url = (
            f"{self._credentials.base_uri}/v2.1/accounts/{self._credentials.account_id}/{endpoint}"
        )

        headers = {
            "Authorization": f"Bearer {self._credentials.access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.request(
                method,
                url,
                headers=headers,
                json=data,
            ) as response:
                if raw_response:
                    return await response.read()

                response_data = await response.json()

                if response.status >= 400:
                    error = response_data.get("message", "Unknown error")
                    raise Exception(f"DocuSign API error: {error}")

                return response_data

    # =========================================================================
    # Envelope Operations
    # =========================================================================

    async def create_envelope(
        self,
        request: EnvelopeCreateRequest,
    ) -> Envelope:
        """
        Create and optionally send an envelope.

        Args:
            request: Envelope creation request

        Returns:
            Created Envelope
        """
        # Build signers
        signers = []
        carbon_copies = []

        for recipient in request.recipients:
            recipient_data = {
                "email": recipient.email,
                "name": recipient.name,
                "recipientId": recipient.recipient_id or str(recipient.routing_order),
                "routingOrder": str(recipient.routing_order),
            }

            # Add tabs if specified
            if request.tabs:
                tabs_for_recipient = [
                    t for t in request.tabs if t.recipient_id == recipient_data["recipientId"]
                ]
                if tabs_for_recipient:
                    recipient_data["tabs"] = self._build_tabs(tabs_for_recipient)

            if recipient.recipient_type == RecipientType.SIGNER:
                signers.append(recipient_data)
            elif recipient.recipient_type == RecipientType.CARBON_COPY:
                carbon_copies.append(recipient_data)

        # Build documents
        documents = []
        for doc in request.documents:
            documents.append(
                {
                    "documentId": doc.document_id,
                    "name": doc.name,
                    "fileExtension": doc.file_extension,
                    "documentBase64": doc.to_base64(),
                    "order": str(doc.order),
                }
            )

        # Build envelope definition
        envelope_definition = {
            "emailSubject": request.email_subject,
            "emailBlurb": request.email_body,
            "status": request.status,
            "documents": documents,
            "recipients": {
                "signers": signers,
                "carbonCopies": carbon_copies,
            },
        }

        # Add options
        if request.expire_enabled:
            envelope_definition["notification"] = {
                "useAccountDefaults": "false",
                "expirations": {
                    "expireEnabled": "true",
                    "expireAfter": str(request.expire_days),
                    "expireWarn": str(request.expire_warn),
                },
            }

        response = await self._request("POST", "envelopes", envelope_definition)

        envelope_id = response["envelopeId"]
        status_str = response.get("status", "created")

        return Envelope(
            envelope_id=envelope_id,
            status=EnvelopeStatus(status_str)
            if status_str in [e.value for e in EnvelopeStatus]
            else EnvelopeStatus.CREATED,
            email_subject=request.email_subject,
            email_body=request.email_body,
            signers=[s for s in signers],
            carbon_copies=[c for c in carbon_copies],
            documents=[{"id": d["documentId"], "name": d["name"]} for d in documents],
            created_at=datetime.now(timezone.utc),
        )

    def _build_tabs(self, tabs: List[SignatureTab]) -> Dict[str, List[Dict[str, Any]]]:
        """Build tabs structure for DocuSign API."""
        tabs_dict: Dict[str, List[Dict[str, Any]]] = {}

        for tab in tabs:
            tab_key = f"{tab.tab_type}Tabs"
            if tab_key not in tabs_dict:
                tabs_dict[tab_key] = []
            tabs_dict[tab_key].append(tab.to_dict())

        return tabs_dict

    async def get_envelope(self, envelope_id: str) -> Optional[Envelope]:
        """
        Get envelope status and details.

        Args:
            envelope_id: Envelope ID

        Returns:
            Envelope or None if not found
        """
        try:
            response = await self._request("GET", f"envelopes/{envelope_id}")

            status_str = response.get("status", "created")

            return Envelope(
                envelope_id=envelope_id,
                status=EnvelopeStatus(status_str)
                if status_str in [e.value for e in EnvelopeStatus]
                else EnvelopeStatus.CREATED,
                email_subject=response.get("emailSubject", ""),
                email_body=response.get("emailBlurb", ""),
                created_at=datetime.fromisoformat(
                    response["createdDateTime"].replace("Z", "+00:00")
                )
                if response.get("createdDateTime")
                else None,
                sent_at=datetime.fromisoformat(response["sentDateTime"].replace("Z", "+00:00"))
                if response.get("sentDateTime")
                else None,
                completed_at=datetime.fromisoformat(
                    response["completedDateTime"].replace("Z", "+00:00")
                )
                if response.get("completedDateTime")
                else None,
                sender_name=response.get("sender", {}).get("userName", ""),
                sender_email=response.get("sender", {}).get("email", ""),
                status_changed_at=datetime.fromisoformat(
                    response["statusChangedDateTime"].replace("Z", "+00:00")
                )
                if response.get("statusChangedDateTime")
                else None,
            )
        except Exception as e:
            logger.error(f"Failed to get envelope {envelope_id}: {e}")
            return None

    async def list_envelopes(
        self,
        from_date: Optional[datetime] = None,
        status: Optional[EnvelopeStatus] = None,
        limit: int = 100,
    ) -> List[Envelope]:
        """
        List envelopes.

        Args:
            from_date: Filter envelopes from this date
            status: Filter by status
            limit: Maximum envelopes to return

        Returns:
            List of Envelopes
        """
        params = [f"count={limit}"]

        if from_date:
            params.append(f"from_date={from_date.isoformat()}")
        else:
            # Default to last 30 days
            default_from = datetime.now(timezone.utc) - timedelta(days=30)
            params.append(f"from_date={default_from.isoformat()}")

        if status:
            params.append(f"status={status.value}")

        query = "&".join(params)
        response = await self._request("GET", f"envelopes?{query}")

        envelopes = []
        for item in response.get("envelopes", []):
            status_str = item.get("status", "created")
            envelopes.append(
                Envelope(
                    envelope_id=item["envelopeId"],
                    status=EnvelopeStatus(status_str)
                    if status_str in [e.value for e in EnvelopeStatus]
                    else EnvelopeStatus.CREATED,
                    email_subject=item.get("emailSubject", ""),
                    created_at=datetime.fromisoformat(
                        item["createdDateTime"].replace("Z", "+00:00")
                    )
                    if item.get("createdDateTime")
                    else None,
                    status_changed_at=datetime.fromisoformat(
                        item["statusChangedDateTime"].replace("Z", "+00:00")
                    )
                    if item.get("statusChangedDateTime")
                    else None,
                )
            )

        return envelopes

    async def void_envelope(
        self,
        envelope_id: str,
        void_reason: str = "Voided by user",
    ) -> bool:
        """
        Void an envelope.

        Args:
            envelope_id: Envelope to void
            void_reason: Reason for voiding

        Returns:
            True if successful
        """
        try:
            await self._request(
                "PUT",
                f"envelopes/{envelope_id}",
                {"status": "voided", "voidedReason": void_reason},
            )
            return True
        except Exception as e:
            logger.error(f"Failed to void envelope {envelope_id}: {e}")
            return False

    async def resend_envelope(self, envelope_id: str) -> bool:
        """
        Resend envelope notifications.

        Args:
            envelope_id: Envelope to resend

        Returns:
            True if successful
        """
        try:
            await self._request(
                "PUT",
                f"envelopes/{envelope_id}?resend_envelope=true",
                {},
            )
            return True
        except Exception as e:
            logger.error(f"Failed to resend envelope {envelope_id}: {e}")
            return False

    # =========================================================================
    # Document Operations
    # =========================================================================

    async def download_document(
        self,
        envelope_id: str,
        document_id: str = "combined",
    ) -> bytes:
        """
        Download document from envelope.

        Args:
            envelope_id: Envelope ID
            document_id: Document ID or 'combined' for all documents

        Returns:
            Document content as bytes
        """
        return await self._request(
            "GET",
            f"envelopes/{envelope_id}/documents/{document_id}",
            raw_response=True,
        )

    async def download_certificate(self, envelope_id: str) -> bytes:
        """
        Download signing certificate.

        Args:
            envelope_id: Envelope ID

        Returns:
            Certificate PDF as bytes
        """
        return await self._request(
            "GET",
            f"envelopes/{envelope_id}/documents/certificate",
            raw_response=True,
        )


# =============================================================================
# Mock Data for Demo
# =============================================================================


def get_mock_envelope() -> Envelope:
    """Generate mock envelope for demo."""
    return Envelope(
        envelope_id="env_demo_001",
        status=EnvelopeStatus.SENT,
        email_subject="Please sign: Contract Agreement",
        email_body="Please review and sign the attached contract.",
        signers=[
            {
                "email": "signer@example.com",
                "name": "John Signer",
                "recipientId": "1",
                "status": "sent",
            }
        ],
        carbon_copies=[
            {
                "email": "cc@example.com",
                "name": "Jane Copy",
                "recipientId": "2",
            }
        ],
        documents=[
            {"id": "1", "name": "Contract.pdf"},
        ],
        created_at=datetime.now(timezone.utc) - timedelta(hours=2),
        sent_at=datetime.now(timezone.utc) - timedelta(hours=2),
        sender_name="Demo User",
        sender_email="demo@example.com",
    )


__all__ = [
    "DocuSignConnector",
    "DocuSignCredentials",
    "DocuSignEnvironment",
    "Envelope",
    "EnvelopeCreateRequest",
    "EnvelopeStatus",
    "Recipient",
    "RecipientType",
    "Document",
    "SignatureTab",
    "get_mock_envelope",
]
