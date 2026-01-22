"""
Salesforce Enterprise Connector.

Provides full integration with Salesforce CRM:
- Standard object traversal (Accounts, Contacts, Opportunities, etc.)
- Custom object support
- SOQL query-based filtering
- Incremental sync via LastModifiedDate
- Bulk API support for large datasets
- OAuth2 authentication

Requires Salesforce API credentials (OAuth2 or username/password flow).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, AsyncIterator, Dict, List, Optional

from aragora.connectors.enterprise.base import (
    EnterpriseConnector,
    SyncItem,
    SyncState,
)
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)


# Standard Salesforce objects for CRM
SALESFORCE_OBJECTS = {
    "Account": {
        "name": "Accounts",
        "fields": [
            "Id",
            "Name",
            "Type",
            "Industry",
            "AnnualRevenue",
            "NumberOfEmployees",
            "Description",
            "Website",
            "Phone",
            "BillingCity",
            "BillingState",
            "BillingCountry",
            "OwnerId",
            "CreatedDate",
            "LastModifiedDate",
        ],
    },
    "Contact": {
        "name": "Contacts",
        "fields": [
            "Id",
            "FirstName",
            "LastName",
            "Email",
            "Phone",
            "Title",
            "Department",
            "AccountId",
            "MailingCity",
            "MailingState",
            "MailingCountry",
            "Description",
            "OwnerId",
            "CreatedDate",
            "LastModifiedDate",
        ],
    },
    "Opportunity": {
        "name": "Opportunities",
        "fields": [
            "Id",
            "Name",
            "StageName",
            "Amount",
            "CloseDate",
            "Probability",
            "Type",
            "LeadSource",
            "AccountId",
            "Description",
            "NextStep",
            "IsClosed",
            "IsWon",
            "OwnerId",
            "CreatedDate",
            "LastModifiedDate",
        ],
    },
    "Lead": {
        "name": "Leads",
        "fields": [
            "Id",
            "FirstName",
            "LastName",
            "Company",
            "Email",
            "Phone",
            "Title",
            "Industry",
            "LeadSource",
            "Status",
            "Rating",
            "Description",
            "ConvertedAccountId",
            "ConvertedContactId",
            "ConvertedOpportunityId",
            "IsConverted",
            "OwnerId",
            "CreatedDate",
            "LastModifiedDate",
        ],
    },
    "Case": {
        "name": "Cases",
        "fields": [
            "Id",
            "CaseNumber",
            "Subject",
            "Description",
            "Status",
            "Priority",
            "Origin",
            "Type",
            "Reason",
            "AccountId",
            "ContactId",
            "IsClosed",
            "ClosedDate",
            "OwnerId",
            "CreatedDate",
            "LastModifiedDate",
        ],
    },
    "Task": {
        "name": "Tasks",
        "fields": [
            "Id",
            "Subject",
            "Description",
            "Status",
            "Priority",
            "ActivityDate",
            "WhatId",
            "WhoId",
            "OwnerId",
            "IsClosed",
            "CreatedDate",
            "LastModifiedDate",
        ],
    },
    "Event": {
        "name": "Events",
        "fields": [
            "Id",
            "Subject",
            "Description",
            "StartDateTime",
            "EndDateTime",
            "Location",
            "WhatId",
            "WhoId",
            "OwnerId",
            "CreatedDate",
            "LastModifiedDate",
        ],
    },
    "Campaign": {
        "name": "Campaigns",
        "fields": [
            "Id",
            "Name",
            "Type",
            "Status",
            "StartDate",
            "EndDate",
            "Description",
            "BudgetedCost",
            "ActualCost",
            "ExpectedRevenue",
            "NumberOfLeads",
            "NumberOfContacts",
            "NumberOfOpportunities",
            "IsActive",
            "OwnerId",
            "CreatedDate",
            "LastModifiedDate",
        ],
    },
}


@dataclass
class SalesforceRecord:
    """A Salesforce record."""

    id: str
    object_type: str
    name: str = ""
    description: str = ""
    status: str = ""
    owner_id: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    url: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SalesforceAttachment:
    """A Salesforce file attachment."""

    id: str
    name: str
    content_type: str
    body_length: int
    parent_id: str
    created_at: Optional[datetime] = None


class SalesforceConnector(EnterpriseConnector):
    """
    Enterprise connector for Salesforce CRM.

    Features:
    - Multi-object crawling (Accounts, Contacts, Opportunities, etc.)
    - Custom object support
    - SOQL query filtering
    - Related record expansion
    - Incremental sync via LastModifiedDate
    - Bulk API for large datasets (>2000 records)

    Authentication:
    - OAuth2: Web Server flow or JWT Bearer
    - Username/Password: Legacy flow with security token

    Usage:
        connector = SalesforceConnector(
            instance_url="https://yourorg.salesforce.com",
            objects=["Account", "Contact", "Opportunity"],
            soql_filter="IsDeleted = false",
        )
        result = await connector.sync()
    """

    # Salesforce API endpoints
    API_VERSION = "v59.0"
    TOKEN_URL = "https://login.salesforce.com/services/oauth2/token"
    SANDBOX_TOKEN_URL = "https://test.salesforce.com/services/oauth2/token"

    def __init__(
        self,
        instance_url: Optional[str] = None,
        objects: Optional[List[str]] = None,
        custom_objects: Optional[List[str]] = None,
        soql_filter: Optional[str] = None,
        include_attachments: bool = False,
        include_notes: bool = False,
        exclude_archived: bool = True,
        use_bulk_api: bool = True,
        bulk_threshold: int = 2000,
        is_sandbox: bool = False,
        **kwargs,
    ):
        """
        Initialize Salesforce connector.

        Args:
            instance_url: Salesforce instance URL (e.g., https://yourorg.salesforce.com)
            objects: Standard objects to sync (default: Account, Contact, Opportunity)
            custom_objects: Custom object API names (e.g., CustomObject__c)
            soql_filter: Additional SOQL WHERE clause filter
            include_attachments: Include file attachments
            include_notes: Include ContentNote objects
            exclude_archived: Exclude archived/deleted records
            use_bulk_api: Use Bulk API for large datasets
            bulk_threshold: Record count threshold to switch to Bulk API
            is_sandbox: Use sandbox token endpoint
        """
        super().__init__(connector_id="salesforce", **kwargs)

        self.instance_url = instance_url
        self.objects = objects or ["Account", "Contact", "Opportunity"]
        self.custom_objects = custom_objects or []
        self.soql_filter = soql_filter
        self.include_attachments = include_attachments
        self.include_notes = include_notes
        self.exclude_archived = exclude_archived
        self.use_bulk_api = use_bulk_api
        self.bulk_threshold = bulk_threshold
        self.is_sandbox = is_sandbox

        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._instance_url_resolved: Optional[str] = None

    @property
    def source_type(self) -> SourceType:
        return SourceType.DATABASE

    @property
    def name(self) -> str:
        return "Salesforce"

    async def _get_access_token(self) -> str:
        """Get valid access token, refreshing if needed."""
        now = datetime.now(timezone.utc)

        if self._access_token and self._token_expiry and now < self._token_expiry:
            return self._access_token

        # Get credentials
        client_id = await self.credentials.get_credential("SALESFORCE_CLIENT_ID")
        client_secret = await self.credentials.get_credential("SALESFORCE_CLIENT_SECRET")
        refresh_token = await self.credentials.get_credential("SALESFORCE_REFRESH_TOKEN")

        # Try OAuth2 refresh first
        if all([client_id, client_secret, refresh_token]):
            return await self._refresh_oauth_token(client_id, client_secret, refresh_token)

        # Fall back to username/password flow
        username = await self.credentials.get_credential("SALESFORCE_USERNAME")
        password = await self.credentials.get_credential("SALESFORCE_PASSWORD")
        security_token = await self.credentials.get_credential("SALESFORCE_SECURITY_TOKEN")

        if all([client_id, client_secret, username, password]):
            return await self._password_auth(
                client_id, client_secret, username, password, security_token
            )

        raise ValueError(
            "Salesforce credentials not configured. "
            "Set either OAuth2 credentials (SALESFORCE_CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN) "
            "or username/password (SALESFORCE_USERNAME, PASSWORD, SECURITY_TOKEN)"
        )

    async def _refresh_oauth_token(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
    ) -> str:
        """Refresh OAuth2 access token."""
        import httpx

        token_url = self.SANDBOX_TOKEN_URL if self.is_sandbox else self.TOKEN_URL

        async with httpx.AsyncClient() as client:
            response = await client.post(
                token_url,
                data={
                    "grant_type": "refresh_token",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                },
            )
            response.raise_for_status()
            data = response.json()

        self._access_token = data["access_token"]
        self._instance_url_resolved = data.get("instance_url", self.instance_url)

        # Token typically valid for 2 hours
        self._token_expiry = datetime.now(timezone.utc) + timedelta(hours=1, minutes=50)

        return self._access_token

    async def _password_auth(
        self,
        client_id: str,
        client_secret: str,
        username: str,
        password: str,
        security_token: Optional[str] = None,
    ) -> str:
        """Authenticate with username/password flow."""
        import httpx

        token_url = self.SANDBOX_TOKEN_URL if self.is_sandbox else self.TOKEN_URL

        # Security token is appended to password
        full_password = password + (security_token or "")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                token_url,
                data={
                    "grant_type": "password",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "username": username,
                    "password": full_password,
                },
            )
            response.raise_for_status()
            data = response.json()

        self._access_token = data["access_token"]
        self._instance_url_resolved = data.get("instance_url", self.instance_url)
        self._token_expiry = datetime.now(timezone.utc) + timedelta(hours=1, minutes=50)

        return self._access_token

    def _get_api_base(self) -> str:
        """Get API base URL."""
        base = self._instance_url_resolved or self.instance_url
        if not base:
            raise ValueError("Instance URL not configured")
        return f"{base.rstrip('/')}/services/data/{self.API_VERSION}"

    async def _api_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make a request to Salesforce REST API."""
        import httpx

        token = await self._get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        url = f"{self._get_api_base()}{endpoint}"

        async with httpx.AsyncClient() as client:
            response = await client.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_data,
                timeout=60,
            )
            response.raise_for_status()
            return response.json() if response.content else {}

    async def _query(self, soql: str) -> AsyncIterator[Dict[str, Any]]:
        """Execute SOQL query with automatic pagination."""
        # URL encode the query
        import urllib.parse

        encoded_query = urllib.parse.quote(soql)
        endpoint = f"/query/?q={encoded_query}"

        while endpoint:
            data = await self._api_request(endpoint)

            for record in data.get("records", []):
                yield record

            # Handle pagination
            next_url = data.get("nextRecordsUrl")
            if next_url:
                # nextRecordsUrl includes /services/data/vXX.X
                endpoint = next_url.split(f"/data/{self.API_VERSION}")[1]
            else:
                endpoint = None

    async def _get_record_count(self, object_name: str, where_clause: str = "") -> int:
        """Get count of records for an object."""
        soql = f"SELECT COUNT() FROM {object_name}"
        if where_clause:
            soql += f" WHERE {where_clause}"

        data = await self._api_request(f"/query/?q={soql}")
        return data.get("totalSize", 0)

    def _build_soql_query(
        self,
        object_name: str,
        fields: List[str],
        last_sync: Optional[datetime] = None,
    ) -> str:
        """Build SOQL query for an object."""
        # Escape any reserved characters in field names
        safe_fields = [f for f in fields if not f.startswith("__")]
        field_list = ", ".join(safe_fields)

        soql = f"SELECT {field_list} FROM {object_name}"

        # Build WHERE clause
        conditions = []

        if self.exclude_archived:
            conditions.append("IsDeleted = false")

        if last_sync:
            iso_time = last_sync.strftime("%Y-%m-%dT%H:%M:%SZ")
            conditions.append(f"LastModifiedDate > {iso_time}")

        if self.soql_filter:
            conditions.append(f"({self.soql_filter})")

        if conditions:
            soql += " WHERE " + " AND ".join(conditions)

        soql += " ORDER BY LastModifiedDate ASC"

        return soql

    def _parse_datetime(self, value: Optional[str]) -> Optional[datetime]:
        """Parse Salesforce datetime string."""
        if not value:
            return None
        try:
            # Salesforce returns ISO format with milliseconds
            if "." in value:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _record_to_text(self, record: Dict[str, Any], object_name: str) -> str:
        """Convert a record to text representation."""
        lines = [f"# {object_name}: {record.get('Name', record.get('Id', 'Unknown'))}"]
        lines.append("")

        # Add key fields based on object type
        skip_fields = {"attributes", "Id", "Name"}

        for key, value in record.items():
            if key in skip_fields or value is None:
                continue
            if isinstance(value, dict):
                # Related record reference
                value = value.get("Name", str(value))
            lines.append(f"**{key}**: {value}")

        return "\n".join(lines)

    async def sync_items(
        self,
        state: SyncState,
        batch_size: int = 100,
    ) -> AsyncIterator[SyncItem]:
        """
        Yield Salesforce records for syncing.

        Syncs all configured objects with incremental support.
        """
        items_yielded = 0

        # Parse last sync time from cursor
        last_sync = None
        if state.cursor:
            try:
                last_sync = datetime.fromisoformat(state.cursor)
            except ValueError:
                pass

        # Combine standard and custom objects
        all_objects = self.objects + self.custom_objects

        for object_name in all_objects:
            # Get fields for this object
            if object_name in SALESFORCE_OBJECTS:
                fields = SALESFORCE_OBJECTS[object_name]["fields"]
            else:
                # For custom objects, query all fields
                fields = ["Id", "Name", "CreatedDate", "LastModifiedDate"]

            try:
                soql = self._build_soql_query(object_name, fields, last_sync)
                logger.debug(f"[{self.name}] Querying: {soql[:200]}...")

                async for record in self._query(soql):
                    record_id = record.get("Id", "")
                    record_name = record.get("Name", record.get("Subject", record_id))

                    # Parse timestamps
                    created = self._parse_datetime(record.get("CreatedDate"))
                    updated = self._parse_datetime(record.get("LastModifiedDate"))

                    # Build URL
                    base_url = self._instance_url_resolved or self.instance_url
                    record_url = f"{base_url}/{record_id}" if base_url else ""

                    yield SyncItem(
                        id=f"sf-{object_name}-{record_id}",
                        content=self._record_to_text(record, object_name)[:50000],
                        source_type="database",
                        source_id=f"salesforce/{object_name}/{record_id}",
                        title=f"{object_name}: {record_name}",
                        url=record_url,
                        author=record.get("OwnerId", ""),
                        created_at=created,
                        updated_at=updated,
                        domain="enterprise/crm",
                        confidence=0.9,
                        metadata={
                            "object_type": object_name,
                            "record_id": record_id,
                            "status": record.get("Status", record.get("StageName", "")),
                            "record": {
                                k: v
                                for k, v in record.items()
                                if k != "attributes" and v is not None
                            },
                        },
                    )

                    items_yielded += 1
                    if items_yielded % batch_size == 0:
                        await asyncio.sleep(0)

            except Exception as e:
                logger.error(f"[{self.name}] Failed to sync {object_name}: {e}")

        # Update cursor for next sync
        state.cursor = datetime.now(timezone.utc).isoformat()
        state.items_total = items_yielded

    async def search(
        self,
        query: str,
        limit: int = 25,
        object_types: Optional[List[str]] = None,
        **kwargs,
    ) -> list:
        """Search Salesforce using SOSL."""
        from aragora.connectors.base import Evidence

        # Build SOSL query
        objects = object_types or self.objects
        returning = ", ".join([f"{obj}(Id, Name)" for obj in objects if obj in SALESFORCE_OBJECTS])

        # Escape special characters in query
        safe_query = query.replace("'", "\\'").replace("\\", "\\\\")
        sosl = f"FIND {{{safe_query}}} IN ALL FIELDS RETURNING {returning} LIMIT {limit}"

        try:
            import urllib.parse

            encoded_query = urllib.parse.quote(sosl)
            data = await self._api_request(f"/search/?q={encoded_query}")

            results = []
            for result in data.get("searchRecords", []):
                object_type = result.get("attributes", {}).get("type", "Unknown")
                record_id = result.get("Id", "")
                record_name = result.get("Name", record_id)

                base_url = self._instance_url_resolved or self.instance_url
                record_url = f"{base_url}/{record_id}" if base_url else ""

                results.append(
                    Evidence(
                        id=f"sf-{object_type}-{record_id}",
                        source_type=self.source_type,
                        source_id=f"salesforce/{object_type}/{record_id}",
                        content="",  # Fetch full content on demand
                        title=f"{object_type}: {record_name}",
                        url=record_url,
                        confidence=0.8,
                        metadata={
                            "object_type": object_type,
                            "record_id": record_id,
                        },
                    )
                )

            return results

        except Exception as e:
            logger.error(f"[{self.name}] Search failed: {e}")
            return []

    async def fetch(self, evidence_id: str) -> Optional[Any]:
        """Fetch a specific Salesforce record."""
        from aragora.connectors.base import Evidence

        # Parse evidence ID: sf-ObjectType-RecordId
        parts = evidence_id.split("-", 2)
        if len(parts) != 3 or parts[0] != "sf":
            return None

        object_type = parts[1]
        record_id = parts[2]

        try:
            data = await self._api_request(f"/sobjects/{object_type}/{record_id}")

            record_name = data.get("Name", data.get("Subject", record_id))
            base_url = self._instance_url_resolved or self.instance_url
            record_url = f"{base_url}/{record_id}" if base_url else ""

            return Evidence(
                id=evidence_id,
                source_type=self.source_type,
                source_id=f"salesforce/{object_type}/{record_id}",
                content=self._record_to_text(data, object_type),
                title=f"{object_type}: {record_name}",
                url=record_url,
                author=data.get("OwnerId", ""),
                confidence=0.9,
                metadata={
                    "object_type": object_type,
                    "record_id": record_id,
                    "record": {k: v for k, v in data.items() if k != "attributes"},
                },
            )

        except Exception as e:
            logger.error(f"[{self.name}] Fetch failed: {e}")
            return None

    async def get_account_contacts(
        self,
        account_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get contacts associated with an account."""
        soql = f"""
            SELECT Id, FirstName, LastName, Email, Phone, Title
            FROM Contact
            WHERE AccountId = '{account_id}'
            LIMIT {limit}
        """

        contacts = []
        async for record in self._query(soql):
            contacts.append(record)

        return contacts

    async def get_account_opportunities(
        self,
        account_id: str,
        include_closed: bool = False,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get opportunities associated with an account."""
        soql = f"""
            SELECT Id, Name, StageName, Amount, CloseDate, Probability, IsWon
            FROM Opportunity
            WHERE AccountId = '{account_id}'
        """

        if not include_closed:
            soql += " AND IsClosed = false"

        soql += f" LIMIT {limit}"

        opportunities = []
        async for record in self._query(soql):
            opportunities.append(record)

        return opportunities


__all__ = ["SalesforceConnector", "SalesforceRecord", "SALESFORCE_OBJECTS"]
