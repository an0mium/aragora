"""
Gmail API resource for the Aragora client.

Provides methods for Gmail connector integration:
- Gmail account connection
- Email debate triggers
- Email triage configuration
- Message processing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..client import AragoraClient

logger = logging.getLogger(__name__)


@dataclass
class GmailConnection:
    """A Gmail connection."""

    id: str
    email: str
    status: str  # connected, disconnected, error
    scopes: List[str] = field(default_factory=list)
    connected_at: Optional[datetime] = None
    last_synced_at: Optional[datetime] = None


@dataclass
class EmailTriageRule:
    """A rule for email triage."""

    id: str
    name: str
    enabled: bool = True
    conditions: Dict[str, Any] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)
    priority: int = 0


@dataclass
class EmailDebateConfig:
    """Configuration for email-triggered debates."""

    id: str
    name: str
    enabled: bool = True
    trigger_conditions: Dict[str, Any] = field(default_factory=dict)
    debate_template: Optional[str] = None
    agents: List[str] = field(default_factory=list)
    auto_reply: bool = False


@dataclass
class ProcessedEmail:
    """A processed email record."""

    id: str
    message_id: str
    subject: str
    sender: str
    status: str  # pending, processing, completed, failed
    debate_id: Optional[str] = None
    processed_at: Optional[datetime] = None
    summary: Optional[str] = None


@dataclass
class GmailStats:
    """Gmail processing statistics."""

    total_processed: int = 0
    debates_triggered: int = 0
    auto_replies_sent: int = 0
    errors: int = 0
    avg_processing_time_ms: float = 0.0


class GmailAPI:
    """API interface for Gmail integration."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    # =========================================================================
    # Connection Management
    # =========================================================================

    def get_connection(self) -> Optional[GmailConnection]:
        """
        Get Gmail connection status.

        Returns:
            GmailConnection object if connected, None otherwise.
        """
        try:
            response = self._client._get("/api/v1/connectors/gmail/status")
            if response.get("connected"):
                return self._parse_connection(response)
            return None
        except Exception:
            return None

    async def get_connection_async(self) -> Optional[GmailConnection]:
        """Async version of get_connection()."""
        try:
            response = await self._client._get_async("/api/v1/connectors/gmail/status")
            if response.get("connected"):
                return self._parse_connection(response)
            return None
        except Exception:
            return None

    def initiate_connection(self, redirect_uri: Optional[str] = None) -> Dict[str, Any]:
        """
        Initiate Gmail OAuth connection.

        Args:
            redirect_uri: Optional redirect URI after OAuth.

        Returns:
            OAuth authorization URL and state.
        """
        body: Dict[str, Any] = {}
        if redirect_uri:
            body["redirect_uri"] = redirect_uri

        return self._client._post("/api/v1/connectors/gmail/connect", body)

    async def initiate_connection_async(self, redirect_uri: Optional[str] = None) -> Dict[str, Any]:
        """Async version of initiate_connection()."""
        body: Dict[str, Any] = {}
        if redirect_uri:
            body["redirect_uri"] = redirect_uri

        return await self._client._post_async("/api/v1/connectors/gmail/connect", body)

    def complete_connection(self, code: str, state: str) -> GmailConnection:
        """
        Complete Gmail OAuth connection.

        Args:
            code: OAuth authorization code.
            state: OAuth state parameter.

        Returns:
            GmailConnection object.
        """
        body = {"code": code, "state": state}
        response = self._client._post("/api/v1/connectors/gmail/callback", body)
        return self._parse_connection(response)

    async def complete_connection_async(self, code: str, state: str) -> GmailConnection:
        """Async version of complete_connection()."""
        body = {"code": code, "state": state}
        response = await self._client._post_async("/api/v1/connectors/gmail/callback", body)
        return self._parse_connection(response)

    def disconnect(self) -> bool:
        """
        Disconnect Gmail account.

        Returns:
            True if successful.
        """
        self._client._delete("/api/v1/connectors/gmail")
        return True

    async def disconnect_async(self) -> bool:
        """Async version of disconnect()."""
        await self._client._delete_async("/api/v1/connectors/gmail")
        return True

    def sync(self) -> Dict[str, Any]:
        """
        Trigger manual email sync.

        Returns:
            Sync result.
        """
        return self._client._post("/api/v1/connectors/gmail/sync", {})

    async def sync_async(self) -> Dict[str, Any]:
        """Async version of sync()."""
        return await self._client._post_async("/api/v1/connectors/gmail/sync", {})

    # =========================================================================
    # Triage Rules
    # =========================================================================

    def list_triage_rules(self) -> List[EmailTriageRule]:
        """
        List email triage rules.

        Returns:
            List of EmailTriageRule objects.
        """
        response = self._client._get("/api/v1/connectors/gmail/triage/rules")
        return [self._parse_triage_rule(r) for r in response.get("rules", [])]

    async def list_triage_rules_async(self) -> List[EmailTriageRule]:
        """Async version of list_triage_rules()."""
        response = await self._client._get_async("/api/v1/connectors/gmail/triage/rules")
        return [self._parse_triage_rule(r) for r in response.get("rules", [])]

    def create_triage_rule(
        self,
        name: str,
        conditions: Dict[str, Any],
        actions: List[str],
        priority: int = 0,
    ) -> EmailTriageRule:
        """
        Create an email triage rule.

        Args:
            name: Rule name.
            conditions: Trigger conditions.
            actions: Actions to take.
            priority: Rule priority (higher = first).

        Returns:
            Created EmailTriageRule object.
        """
        body = {
            "name": name,
            "conditions": conditions,
            "actions": actions,
            "priority": priority,
        }
        response = self._client._post("/api/v1/connectors/gmail/triage/rules", body)
        return self._parse_triage_rule(response.get("rule", response))

    async def create_triage_rule_async(
        self,
        name: str,
        conditions: Dict[str, Any],
        actions: List[str],
        priority: int = 0,
    ) -> EmailTriageRule:
        """Async version of create_triage_rule()."""
        body = {
            "name": name,
            "conditions": conditions,
            "actions": actions,
            "priority": priority,
        }
        response = await self._client._post_async("/api/v1/connectors/gmail/triage/rules", body)
        return self._parse_triage_rule(response.get("rule", response))

    def update_triage_rule(
        self,
        rule_id: str,
        name: Optional[str] = None,
        conditions: Optional[Dict[str, Any]] = None,
        actions: Optional[List[str]] = None,
        enabled: Optional[bool] = None,
        priority: Optional[int] = None,
    ) -> EmailTriageRule:
        """
        Update an email triage rule.

        Args:
            rule_id: The rule ID.
            name: New name.
            conditions: New conditions.
            actions: New actions.
            enabled: Enable/disable.
            priority: New priority.

        Returns:
            Updated EmailTriageRule object.
        """
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if conditions is not None:
            body["conditions"] = conditions
        if actions is not None:
            body["actions"] = actions
        if enabled is not None:
            body["enabled"] = enabled
        if priority is not None:
            body["priority"] = priority

        response = self._client._patch(f"/api/v1/connectors/gmail/triage/rules/{rule_id}", body)
        return self._parse_triage_rule(response.get("rule", response))

    async def update_triage_rule_async(
        self,
        rule_id: str,
        name: Optional[str] = None,
        conditions: Optional[Dict[str, Any]] = None,
        actions: Optional[List[str]] = None,
        enabled: Optional[bool] = None,
        priority: Optional[int] = None,
    ) -> EmailTriageRule:
        """Async version of update_triage_rule()."""
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if conditions is not None:
            body["conditions"] = conditions
        if actions is not None:
            body["actions"] = actions
        if enabled is not None:
            body["enabled"] = enabled
        if priority is not None:
            body["priority"] = priority

        response = await self._client._patch_async(
            f"/api/v1/connectors/gmail/triage/rules/{rule_id}", body
        )
        return self._parse_triage_rule(response.get("rule", response))

    def delete_triage_rule(self, rule_id: str) -> bool:
        """
        Delete an email triage rule.

        Args:
            rule_id: The rule ID.

        Returns:
            True if successful.
        """
        self._client._delete(f"/api/v1/connectors/gmail/triage/rules/{rule_id}")
        return True

    async def delete_triage_rule_async(self, rule_id: str) -> bool:
        """Async version of delete_triage_rule()."""
        await self._client._delete_async(f"/api/v1/connectors/gmail/triage/rules/{rule_id}")
        return True

    # =========================================================================
    # Debate Configuration
    # =========================================================================

    def list_debate_configs(self) -> List[EmailDebateConfig]:
        """
        List email debate configurations.

        Returns:
            List of EmailDebateConfig objects.
        """
        response = self._client._get("/api/v1/connectors/gmail/debates")
        return [self._parse_debate_config(c) for c in response.get("configs", [])]

    async def list_debate_configs_async(self) -> List[EmailDebateConfig]:
        """Async version of list_debate_configs()."""
        response = await self._client._get_async("/api/v1/connectors/gmail/debates")
        return [self._parse_debate_config(c) for c in response.get("configs", [])]

    def create_debate_config(
        self,
        name: str,
        trigger_conditions: Dict[str, Any],
        agents: Optional[List[str]] = None,
        debate_template: Optional[str] = None,
        auto_reply: bool = False,
    ) -> EmailDebateConfig:
        """
        Create an email debate configuration.

        Args:
            name: Configuration name.
            trigger_conditions: Conditions that trigger a debate.
            agents: Agents to use for debates.
            debate_template: Template to use.
            auto_reply: Whether to auto-reply with results.

        Returns:
            Created EmailDebateConfig object.
        """
        body: Dict[str, Any] = {
            "name": name,
            "trigger_conditions": trigger_conditions,
            "auto_reply": auto_reply,
        }
        if agents:
            body["agents"] = agents
        if debate_template:
            body["debate_template"] = debate_template

        response = self._client._post("/api/v1/connectors/gmail/debates", body)
        return self._parse_debate_config(response.get("config", response))

    async def create_debate_config_async(
        self,
        name: str,
        trigger_conditions: Dict[str, Any],
        agents: Optional[List[str]] = None,
        debate_template: Optional[str] = None,
        auto_reply: bool = False,
    ) -> EmailDebateConfig:
        """Async version of create_debate_config()."""
        body: Dict[str, Any] = {
            "name": name,
            "trigger_conditions": trigger_conditions,
            "auto_reply": auto_reply,
        }
        if agents:
            body["agents"] = agents
        if debate_template:
            body["debate_template"] = debate_template

        response = await self._client._post_async("/api/v1/connectors/gmail/debates", body)
        return self._parse_debate_config(response.get("config", response))

    # =========================================================================
    # Processed Emails
    # =========================================================================

    def list_processed_emails(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[List[ProcessedEmail], int]:
        """
        List processed emails.

        Args:
            status: Filter by status.
            limit: Maximum number of emails.
            offset: Offset for pagination.

        Returns:
            Tuple of (list of ProcessedEmail objects, total count).
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        response = self._client._get("/api/v1/connectors/gmail/emails", params=params)
        emails = [self._parse_processed_email(e) for e in response.get("emails", [])]
        return emails, response.get("total", len(emails))

    async def list_processed_emails_async(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[List[ProcessedEmail], int]:
        """Async version of list_processed_emails()."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        response = await self._client._get_async("/api/v1/connectors/gmail/emails", params=params)
        emails = [self._parse_processed_email(e) for e in response.get("emails", [])]
        return emails, response.get("total", len(emails))

    def get_stats(self) -> GmailStats:
        """
        Get Gmail processing statistics.

        Returns:
            GmailStats object.
        """
        response = self._client._get("/api/v1/connectors/gmail/stats")
        return self._parse_stats(response)

    async def get_stats_async(self) -> GmailStats:
        """Async version of get_stats()."""
        response = await self._client._get_async("/api/v1/connectors/gmail/stats")
        return self._parse_stats(response)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _parse_connection(self, data: Dict[str, Any]) -> GmailConnection:
        """Parse connection data into GmailConnection object."""
        connected_at = None
        last_synced_at = None

        if data.get("connected_at"):
            try:
                connected_at = datetime.fromisoformat(data["connected_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        if data.get("last_synced_at"):
            try:
                last_synced_at = datetime.fromisoformat(
                    data["last_synced_at"].replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        return GmailConnection(
            id=data.get("id", ""),
            email=data.get("email", ""),
            status=data.get("status", "connected"),
            scopes=data.get("scopes", []),
            connected_at=connected_at,
            last_synced_at=last_synced_at,
        )

    def _parse_triage_rule(self, data: Dict[str, Any]) -> EmailTriageRule:
        """Parse triage rule data into EmailTriageRule object."""
        return EmailTriageRule(
            id=data.get("id", ""),
            name=data.get("name", ""),
            enabled=data.get("enabled", True),
            conditions=data.get("conditions", {}),
            actions=data.get("actions", []),
            priority=data.get("priority", 0),
        )

    def _parse_debate_config(self, data: Dict[str, Any]) -> EmailDebateConfig:
        """Parse debate config data into EmailDebateConfig object."""
        return EmailDebateConfig(
            id=data.get("id", ""),
            name=data.get("name", ""),
            enabled=data.get("enabled", True),
            trigger_conditions=data.get("trigger_conditions", {}),
            debate_template=data.get("debate_template"),
            agents=data.get("agents", []),
            auto_reply=data.get("auto_reply", False),
        )

    def _parse_processed_email(self, data: Dict[str, Any]) -> ProcessedEmail:
        """Parse processed email data into ProcessedEmail object."""
        processed_at = None
        if data.get("processed_at"):
            try:
                processed_at = datetime.fromisoformat(data["processed_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return ProcessedEmail(
            id=data.get("id", ""),
            message_id=data.get("message_id", ""),
            subject=data.get("subject", ""),
            sender=data.get("sender", ""),
            status=data.get("status", "pending"),
            debate_id=data.get("debate_id"),
            processed_at=processed_at,
            summary=data.get("summary"),
        )

    def _parse_stats(self, data: Dict[str, Any]) -> GmailStats:
        """Parse stats data into GmailStats object."""
        return GmailStats(
            total_processed=data.get("total_processed", 0),
            debates_triggered=data.get("debates_triggered", 0),
            auto_replies_sent=data.get("auto_replies_sent", 0),
            errors=data.get("errors", 0),
            avg_processing_time_ms=data.get("avg_processing_time_ms", 0.0),
        )


__all__ = [
    "GmailAPI",
    "GmailConnection",
    "EmailTriageRule",
    "EmailDebateConfig",
    "ProcessedEmail",
    "GmailStats",
]
