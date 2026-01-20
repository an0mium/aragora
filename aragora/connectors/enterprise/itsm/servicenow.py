"""
ServiceNow Enterprise Connector.

Provides full integration with ServiceNow ITSM:
- Table traversal (Incidents, Problems, Changes, etc.)
- Record field extraction
- Query-based filtering (sysparm_query)
- Incremental sync via sys_updated_on timestamps
- Webhook support for real-time updates

Requires ServiceNow API credentials.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

from aragora.connectors.enterprise.base import (
    EnterpriseConnector,
    SyncItem,
    SyncState,
)
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)


# Standard ServiceNow tables for ITSM
SERVICENOW_TABLES = {
    "incident": {
        "name": "Incidents",
        "fields": [
            "number", "short_description", "description", "state",
            "priority", "urgency", "impact", "assigned_to",
            "caller_id", "category", "subcategory", "resolution_notes",
            "sys_created_on", "sys_updated_on", "resolved_at", "closed_at",
        ],
    },
    "problem": {
        "name": "Problems",
        "fields": [
            "number", "short_description", "description", "state",
            "priority", "urgency", "impact", "assigned_to",
            "known_error", "workaround", "root_cause",
            "sys_created_on", "sys_updated_on", "resolved_at", "closed_at",
        ],
    },
    "change_request": {
        "name": "Change Requests",
        "fields": [
            "number", "short_description", "description", "state", "type",
            "priority", "risk", "impact", "assigned_to", "requested_by",
            "start_date", "end_date", "justification", "implementation_plan",
            "backout_plan", "test_plan", "sys_created_on", "sys_updated_on",
        ],
    },
    "sc_req_item": {
        "name": "Requested Items",
        "fields": [
            "number", "short_description", "description", "state",
            "priority", "requested_for", "assigned_to", "cat_item",
            "sys_created_on", "sys_updated_on",
        ],
    },
    "kb_knowledge": {
        "name": "Knowledge Articles",
        "fields": [
            "number", "short_description", "text", "article_type",
            "workflow_state", "valid_to", "author", "kb_category",
            "sys_created_on", "sys_updated_on",
        ],
    },
}


@dataclass
class ServiceNowRecord:
    """A ServiceNow record."""

    sys_id: str
    number: str
    table: str
    short_description: str
    description: str = ""
    state: str = ""
    priority: str = ""
    urgency: str = ""
    impact: str = ""
    assigned_to: str = ""
    caller_id: str = ""
    category: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    url: str = ""
    additional_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceNowComment:
    """A ServiceNow work note or comment."""

    sys_id: str
    element_id: str
    value: str
    author: str = ""
    created_at: Optional[datetime] = None


class ServiceNowConnector(EnterpriseConnector):
    """
    Enterprise connector for ServiceNow ITSM.

    Features:
    - Multi-table crawling (Incidents, Problems, Changes, etc.)
    - Field-based filtering with sysparm_query
    - Work notes and comments extraction
    - Incremental sync via sys_updated_on timestamps
    - Webhook support for real-time updates

    Authentication:
    - Basic: Username + Password
    - OAuth2: Client credentials flow

    Usage:
        connector = ServiceNowConnector(
            instance_url="https://your-instance.service-now.com",
            tables=["incident", "problem"],  # Optional: specific tables
            query="active=true",  # Optional: filter
        )
        result = await connector.sync()
    """

    def __init__(
        self,
        instance_url: str,
        tables: Optional[List[str]] = None,
        query: Optional[str] = None,
        include_comments: bool = True,
        include_knowledge: bool = False,
        exclude_states: Optional[List[str]] = None,
        use_oauth: bool = False,
        **kwargs,
    ):
        """
        Initialize ServiceNow connector.

        Args:
            instance_url: ServiceNow instance URL (e.g., https://instance.service-now.com)
            tables: Specific table names to sync (None = default ITSM tables)
            query: Additional sysparm_query filter
            include_comments: Whether to index work notes/comments
            include_knowledge: Whether to include knowledge articles
            exclude_states: State values to exclude from indexing
            use_oauth: Whether to use OAuth2 authentication
        """
        # Normalize URL
        self.instance_url = instance_url.rstrip("/")

        # Extract instance name for connector ID
        match = re.search(r"https?://([^.]+)", self.instance_url)
        instance_name = match.group(1) if match else "servicenow"

        connector_id = f"servicenow_{instance_name}"
        super().__init__(connector_id=connector_id, **kwargs)

        # Configure tables to sync
        self.tables = tables or ["incident", "problem", "change_request"]
        if include_knowledge and "kb_knowledge" not in self.tables:
            self.tables.append("kb_knowledge")

        self.query = query
        self.include_comments = include_comments
        self.exclude_states = set(s.lower() for s in (exclude_states or []))
        self.use_oauth = use_oauth

        # OAuth token cache
        self._oauth_token: Optional[str] = None
        self._oauth_expires: Optional[datetime] = None

    @property
    def source_type(self) -> SourceType:
        return SourceType.EXTERNAL_API

    @property
    def name(self) -> str:
        return f"ServiceNow ({self.instance_url})"

    async def _get_auth_header(self) -> Dict[str, str]:
        """Get authentication header."""
        if self.use_oauth:
            token = await self._get_oauth_token()
            return {"Authorization": f"Bearer {token}"}
        else:
            username = await self.credentials.get_credential("SERVICENOW_USERNAME")
            password = await self.credentials.get_credential("SERVICENOW_PASSWORD")

            if not username or not password:
                raise ValueError(
                    "ServiceNow credentials not configured. "
                    "Set SERVICENOW_USERNAME and SERVICENOW_PASSWORD"
                )

            auth = base64.b64encode(f"{username}:{password}".encode()).decode()
            return {"Authorization": f"Basic {auth}"}

    async def _get_oauth_token(self) -> str:
        """Get OAuth2 access token."""
        import httpx

        # Check if we have a valid cached token
        if self._oauth_token and self._oauth_expires:
            if datetime.utcnow() < self._oauth_expires:
                return self._oauth_token

        client_id = await self.credentials.get_credential("SERVICENOW_CLIENT_ID")
        client_secret = await self.credentials.get_credential("SERVICENOW_CLIENT_SECRET")
        username = await self.credentials.get_credential("SERVICENOW_USERNAME")
        password = await self.credentials.get_credential("SERVICENOW_PASSWORD")

        if not client_id or not client_secret:
            raise ValueError(
                "ServiceNow OAuth credentials not configured. "
                "Set SERVICENOW_CLIENT_ID and SERVICENOW_CLIENT_SECRET"
            )

        url = f"{self.instance_url}/oauth_token.do"
        data = {
            "grant_type": "password",
            "client_id": client_id,
            "client_secret": client_secret,
            "username": username,
            "password": password,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=data)
            response.raise_for_status()
            token_data = response.json()

        self._oauth_token = token_data["access_token"]
        expires_in = int(token_data.get("expires_in", 3600))
        self._oauth_expires = datetime.utcnow().replace(
            second=datetime.utcnow().second + expires_in - 60
        )

        return self._oauth_token

    async def _api_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make a request to ServiceNow REST API."""
        import httpx

        headers = await self._get_auth_header()
        headers["Accept"] = "application/json"
        headers["Content-Type"] = "application/json"

        url = f"{self.instance_url}/api/now{endpoint}"

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

    async def _get_table_records(
        self,
        table: str,
        modified_since: Optional[datetime] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Get records from a ServiceNow table."""
        table_config = SERVICENOW_TABLES.get(table, {})
        fields = table_config.get("fields", ["number", "short_description", "description"])

        params = {
            "sysparm_fields": ",".join(["sys_id"] + fields),
            "sysparm_limit": limit,
            "sysparm_offset": offset,
            "sysparm_display_value": "true",  # Get display values instead of sys_ids
        }

        # Build query
        query_parts = []
        if modified_since:
            ts_str = modified_since.strftime("%Y-%m-%d %H:%M:%S")
            query_parts.append(f"sys_updated_on>={ts_str}")

        if self.query:
            query_parts.append(self.query)

        if query_parts:
            params["sysparm_query"] = "^".join(query_parts) + "^ORDERBYsys_updated_on"
        else:
            params["sysparm_query"] = "ORDERBYsys_updated_on"

        return await self._api_request(f"/table/{table}", params=params)

    async def _get_records(
        self,
        table: str,
        modified_since: Optional[datetime] = None,
    ) -> AsyncIterator[ServiceNowRecord]:
        """Get all records from a table."""
        offset = 0
        limit = 100

        while True:
            data = await self._get_table_records(table, modified_since, offset, limit)
            records = data.get("result", [])

            for record in records:
                # Skip excluded states
                state = record.get("state", "")
                if isinstance(state, dict):
                    state = state.get("display_value", "")
                if state.lower() in self.exclude_states:
                    continue

                # Parse dates
                created_at = self._parse_datetime(record.get("sys_created_on"))
                updated_at = self._parse_datetime(record.get("sys_updated_on"))
                resolved_at = self._parse_datetime(record.get("resolved_at"))
                closed_at = self._parse_datetime(record.get("closed_at"))

                # Extract display values from link fields
                assigned_to = self._extract_display_value(record.get("assigned_to"))
                caller_id = self._extract_display_value(record.get("caller_id"))

                yield ServiceNowRecord(
                    sys_id=record.get("sys_id", ""),
                    number=record.get("number", ""),
                    table=table,
                    short_description=record.get("short_description", ""),
                    description=record.get("description", "") or record.get("text", ""),
                    state=state if isinstance(state, str) else str(state),
                    priority=self._extract_display_value(record.get("priority")),
                    urgency=self._extract_display_value(record.get("urgency")),
                    impact=self._extract_display_value(record.get("impact")),
                    assigned_to=assigned_to,
                    caller_id=caller_id,
                    category=self._extract_display_value(record.get("category")),
                    created_at=created_at,
                    updated_at=updated_at,
                    resolved_at=resolved_at,
                    closed_at=closed_at,
                    url=f"{self.instance_url}/{table}.do?sys_id={record.get('sys_id', '')}",
                    additional_fields={
                        k: v for k, v in record.items()
                        if k not in ["sys_id", "number", "short_description", "description",
                                    "state", "priority", "urgency", "impact", "assigned_to",
                                    "caller_id", "category", "sys_created_on", "sys_updated_on",
                                    "resolved_at", "closed_at", "text"]
                    },
                )

            # Check pagination
            if len(records) < limit:
                break
            offset += limit

    def _parse_datetime(self, value: Optional[str]) -> Optional[datetime]:
        """Parse ServiceNow datetime string."""
        if not value:
            return None
        try:
            # ServiceNow uses format: 2024-01-15 14:30:00
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None

    def _extract_display_value(self, field: Any) -> str:
        """Extract display value from a field (handles link/reference fields)."""
        if field is None:
            return ""
        if isinstance(field, str):
            return field
        if isinstance(field, dict):
            return field.get("display_value", str(field.get("value", "")))
        return str(field)

    async def _get_record_comments(
        self,
        table: str,
        sys_id: str,
    ) -> List[ServiceNowComment]:
        """Get work notes and comments for a record."""
        if not self.include_comments:
            return []

        comments = []

        try:
            # Get journal entries (work notes and comments)
            params = {
                "sysparm_query": f"element_id={sys_id}^element=comments^ORDERelement=work_notes",
                "sysparm_fields": "sys_id,element_id,value,sys_created_on,sys_created_by",
                "sysparm_display_value": "true",
            }

            data = await self._api_request("/table/sys_journal_field", params=params)

            for entry in data.get("result", []):
                created_at = self._parse_datetime(entry.get("sys_created_on"))

                comments.append(
                    ServiceNowComment(
                        sys_id=entry.get("sys_id", ""),
                        element_id=entry.get("element_id", ""),
                        value=entry.get("value", ""),
                        author=self._extract_display_value(entry.get("sys_created_by")),
                        created_at=created_at,
                    )
                )

        except Exception as e:
            logger.warning(f"[{self.name}] Failed to get comments for {sys_id}: {e}")

        return comments

    async def sync_items(
        self,
        state: SyncState,
        batch_size: int = 100,
    ) -> AsyncIterator[SyncItem]:
        """
        Yield ServiceNow records for syncing.

        Crawls configured tables and extracts record content.
        """
        # Parse last sync timestamp from cursor
        modified_since = None
        if state.cursor:
            try:
                modified_since = datetime.fromisoformat(state.cursor)
            except ValueError:
                logger.debug("Invalid cursor timestamp, starting fresh sync")

        state.items_total = len(self.tables)
        items_yielded = 0

        for table in self.tables:
            table_config = SERVICENOW_TABLES.get(table, {"name": table})
            logger.info(f"[{self.name}] Syncing table: {table_config.get('name', table)}")

            async for record in self._get_records(table, modified_since):
                # Get comments
                comments = await self._get_record_comments(table, record.sys_id)
                comments_text = ""
                if comments:
                    comments_text = "\n\nWork Notes/Comments:\n" + "\n".join(
                        f"- {c.author}: {c.value}" for c in comments
                    )

                # Build full content
                content_parts = [
                    f"# [{record.number}] {record.short_description}",
                    f"\nTable: {table_config.get('name', table)}",
                    f"State: {record.state}",
                ]

                if record.priority:
                    content_parts.append(f"Priority: {record.priority}")
                if record.urgency:
                    content_parts.append(f"Urgency: {record.urgency}")
                if record.impact:
                    content_parts.append(f"Impact: {record.impact}")
                if record.assigned_to:
                    content_parts.append(f"Assigned To: {record.assigned_to}")
                if record.caller_id:
                    content_parts.append(f"Caller: {record.caller_id}")
                if record.category:
                    content_parts.append(f"Category: {record.category}")

                content_parts.append(f"\n## Description\n{record.description}")
                content_parts.append(comments_text)

                full_content = "\n".join(content_parts)

                yield SyncItem(
                    id=f"snow-{table}-{record.number}",
                    content=full_content[:50000],
                    source_type=table,
                    source_id=f"servicenow/{table}/{record.number}",
                    title=f"[{record.number}] {record.short_description}",
                    url=record.url,
                    author=record.caller_id or record.assigned_to,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                    domain="enterprise/servicenow",
                    confidence=0.85,
                    metadata={
                        "table": table,
                        "table_name": table_config.get("name", table),
                        "number": record.number,
                        "sys_id": record.sys_id,
                        "state": record.state,
                        "priority": record.priority,
                        "urgency": record.urgency,
                        "impact": record.impact,
                        "assigned_to": record.assigned_to,
                        "category": record.category,
                        "comment_count": len(comments),
                        **{
                            k: self._extract_display_value(v)
                            for k, v in record.additional_fields.items()
                            if v
                        },
                    },
                )

                items_yielded += 1

                # Update cursor to latest modification time
                if record.updated_at:
                    current_cursor = state.cursor
                    if not current_cursor or record.updated_at.isoformat() > current_cursor:
                        state.cursor = record.updated_at.isoformat()

                if items_yielded >= batch_size:
                    await asyncio.sleep(0)

    async def search(
        self,
        query: str,
        limit: int = 10,
        table: Optional[str] = None,
        **kwargs,
    ) -> list:
        """Search ServiceNow records via text search."""
        from aragora.connectors.base import Evidence

        # Default to incident table if not specified
        search_tables = [table] if table else self.tables

        results = []

        for tbl in search_tables:
            if len(results) >= limit:
                break

            try:
                params = {
                    "sysparm_query": f"short_descriptionLIKE{query}^ORdescriptionLIKE{query}",
                    "sysparm_fields": "sys_id,number,short_description,description,state,priority",
                    "sysparm_limit": limit - len(results),
                    "sysparm_display_value": "true",
                }

                data = await self._api_request(f"/table/{tbl}", params=params)

                for record in data.get("result", []):
                    results.append(
                        Evidence(
                            id=f"snow-{tbl}-{record.get('number', '')}",
                            source_type=self.source_type,
                            source_id=record.get("number", ""),
                            content=f"{record.get('short_description', '')}\n\n{record.get('description', '')[:1500]}",
                            title=f"[{record.get('number', '')}] {record.get('short_description', '')}",
                            url=f"{self.instance_url}/{tbl}.do?sys_id={record.get('sys_id', '')}",
                            confidence=0.8,
                            metadata={
                                "table": tbl,
                                "state": self._extract_display_value(record.get("state")),
                                "priority": self._extract_display_value(record.get("priority")),
                            },
                        )
                    )

            except Exception as e:
                logger.error(f"[{self.name}] Search in {tbl} failed: {e}")

        return results[:limit]

    async def fetch(self, evidence_id: str) -> Optional[Any]:
        """Fetch a specific ServiceNow record."""
        from aragora.connectors.base import Evidence

        # Parse evidence_id format: snow-{table}-{number}
        parts = evidence_id.split("-", 2)
        if len(parts) < 3 or parts[0] != "snow":
            logger.error(f"[{self.name}] Invalid evidence ID format: {evidence_id}")
            return None

        table = parts[1]
        number = parts[2]

        try:
            params = {
                "sysparm_query": f"number={number}",
                "sysparm_limit": 1,
                "sysparm_display_value": "true",
            }

            data = await self._api_request(f"/table/{table}", params=params)
            records = data.get("result", [])

            if not records:
                return None

            record = records[0]

            return Evidence(
                id=evidence_id,
                source_type=self.source_type,
                source_id=number,
                content=f"{record.get('short_description', '')}\n\n{record.get('description', '')}",
                title=f"[{number}] {record.get('short_description', '')}",
                url=f"{self.instance_url}/{table}.do?sys_id={record.get('sys_id', '')}",
                author=self._extract_display_value(record.get("caller_id")),
                created_at=record.get("sys_created_on"),
                confidence=0.85,
                metadata={
                    "table": table,
                    "state": self._extract_display_value(record.get("state")),
                    "priority": self._extract_display_value(record.get("priority")),
                    "assigned_to": self._extract_display_value(record.get("assigned_to")),
                },
            )

        except Exception as e:
            logger.error(f"[{self.name}] Fetch failed: {e}")
            return None

    async def handle_webhook(
        self,
        payload: Dict[str, Any],
        signature: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> bool:
        """
        Handle ServiceNow webhook notification (Business Rule).

        Args:
            payload: Webhook payload from ServiceNow
            signature: HMAC-SHA256 signature header (X-ServiceNow-Signature)
            timestamp: Request timestamp for replay protection (X-ServiceNow-Timestamp)

        Returns:
            True if webhook was processed successfully
        """
        # Verify signature if secret is configured
        secret = self.get_webhook_secret()
        if secret:
            if not signature:
                logger.warning(f"[{self.name}] Webhook missing signature")
                return False

            if not self.verify_webhook_signature(payload, signature, secret):
                logger.warning(f"[{self.name}] Webhook signature verification failed")
                return False

            # Replay protection: reject requests older than 5 minutes
            if timestamp:
                try:
                    import time
                    request_time = float(timestamp)
                    if abs(time.time() - request_time) > 300:
                        logger.warning(f"[{self.name}] Webhook timestamp too old")
                        return False
                except (ValueError, TypeError):
                    pass

        table = payload.get("table_name", "")
        sys_id = payload.get("sys_id", "")
        operation = payload.get("operation", "")

        logger.info(f"[{self.name}] Webhook: {operation} on {table}/{sys_id}")

        if table in self.tables:
            # Trigger incremental sync
            asyncio.create_task(self.sync(max_items=10))
            return True

        return False

    def verify_webhook_signature(
        self,
        payload: Dict[str, Any],
        signature: str,
        secret: str,
    ) -> bool:
        """
        Verify HMAC-SHA256 signature from ServiceNow webhook.

        ServiceNow Business Rules can be configured to sign payloads
        using a shared secret.

        Args:
            payload: Webhook payload
            signature: Base64-encoded HMAC-SHA256 signature
            secret: Shared secret key

        Returns:
            True if signature is valid
        """
        import hashlib
        import hmac
        import json

        try:
            # Serialize payload consistently
            payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            expected = hmac.new(
                secret.encode(),
                payload_bytes,
                hashlib.sha256,
            ).digest()

            # Compare with provided signature (base64 or hex)
            try:
                provided = base64.b64decode(signature)
            except Exception:
                provided = bytes.fromhex(signature)

            return hmac.compare_digest(expected, provided)

        except Exception as e:
            logger.error(f"[{self.name}] Signature verification error: {e}")
            return False

    def get_webhook_secret(self) -> Optional[str]:
        """Get webhook secret for signature verification."""
        import os
        return os.environ.get("SERVICENOW_WEBHOOK_SECRET")

    async def resolve_reference(
        self,
        table: str,
        sys_id: str,
        fields: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve a reference field to its full record.

        Args:
            table: Target table name
            sys_id: sys_id of the referenced record
            fields: Fields to retrieve (None = all)

        Returns:
            Record data or None if not found
        """
        if not sys_id:
            return None

        params: Dict[str, Any] = {
            "sysparm_query": f"sys_id={sys_id}",
            "sysparm_limit": 1,
        }

        if fields:
            params["sysparm_fields"] = ",".join(fields)
            params["sysparm_display_value"] = "all"

        try:
            async with self._client.get(
                f"{self.instance_url}/api/now/table/{table}",
                headers=self._get_headers(),
                params=params,
            ) as resp:
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("result", [])
                    return results[0] if results else None
        except Exception as e:
            logger.error(f"[{self.name}] Reference resolution failed: {e}")

        return None

    async def get_user_details(self, user_sys_id: str) -> Optional[Dict[str, str]]:
        """
        Get user details from sys_user table.

        Args:
            user_sys_id: User's sys_id

        Returns:
            Dict with user_id, name, email, department
        """
        record = await self.resolve_reference(
            "sys_user",
            user_sys_id,
            ["user_name", "name", "email", "department"],
        )
        if record:
            return {
                "user_id": record.get("user_name", {}).get("value", ""),
                "name": record.get("name", {}).get("display_value", ""),
                "email": record.get("email", {}).get("value", ""),
                "department": record.get("department", {}).get("display_value", ""),
            }
        return None


__all__ = ["ServiceNowConnector", "ServiceNowRecord", "ServiceNowComment", "SERVICENOW_TABLES"]
