"""Tests for GmailAPI client resource."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.client.client import AragoraClient
from aragora.client.resources.gmail import (
    EmailDebateConfig,
    EmailTriageRule,
    GmailAPI,
    GmailConnection,
    GmailStats,
    ProcessedEmail,
)


@pytest.fixture
def mock_client() -> AragoraClient:
    client = MagicMock(spec=AragoraClient)
    return client


@pytest.fixture
def api(mock_client: AragoraClient) -> GmailAPI:
    return GmailAPI(mock_client)


SAMPLE_CONNECTION = {
    "connected": True,
    "id": "conn-1",
    "email": "user@example.com",
    "status": "connected",
    "scopes": ["gmail.readonly", "gmail.send"],
    "connected_at": "2026-01-10T08:00:00Z",
    "last_synced_at": "2026-01-15T12:00:00Z",
}

SAMPLE_TRIAGE_RULE = {
    "id": "rule-1",
    "name": "Urgent filter",
    "enabled": True,
    "conditions": {"subject_contains": "URGENT"},
    "actions": ["flag", "notify"],
    "priority": 10,
}

SAMPLE_DEBATE_CONFIG = {
    "id": "dconf-1",
    "name": "Auto-debate important emails",
    "enabled": True,
    "trigger_conditions": {"label": "important"},
    "debate_template": "quick_review",
    "agents": ["claude", "gpt4"],
    "auto_reply": True,
}

SAMPLE_PROCESSED_EMAIL = {
    "id": "pe-1",
    "message_id": "msg-abc",
    "subject": "Q4 Budget Review",
    "sender": "cfo@example.com",
    "status": "completed",
    "debate_id": "debate-xyz",
    "processed_at": "2026-01-15T14:30:00Z",
    "summary": "Budget approved with minor adjustments.",
}

SAMPLE_STATS = {
    "total_processed": 150,
    "debates_triggered": 42,
    "auto_replies_sent": 30,
    "errors": 3,
    "avg_processing_time_ms": 1234.5,
}


# =========================================================================
# Connection Management
# =========================================================================


class TestGetConnection:
    def test_get_connection_returns_object(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_CONNECTION
        result = api.get_connection()
        assert isinstance(result, GmailConnection)
        assert result.id == "conn-1"
        assert result.email == "user@example.com"
        assert result.status == "connected"
        assert result.scopes == ["gmail.readonly", "gmail.send"]
        mock_client._get.assert_called_once_with("/api/v1/connectors/gmail/status")

    def test_get_connection_not_connected(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"connected": False}
        result = api.get_connection()
        assert result is None

    def test_get_connection_on_exception(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        mock_client._get.side_effect = RuntimeError("network error")
        result = api.get_connection()
        assert result is None

    @pytest.mark.asyncio
    async def test_get_connection_async(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_CONNECTION)
        result = await api.get_connection_async()
        assert isinstance(result, GmailConnection)
        assert result.email == "user@example.com"

    @pytest.mark.asyncio
    async def test_get_connection_async_not_connected(
        self, api: GmailAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value={"connected": False})
        result = await api.get_connection_async()
        assert result is None

    @pytest.mark.asyncio
    async def test_get_connection_async_on_exception(
        self, api: GmailAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(side_effect=RuntimeError("timeout"))
        result = await api.get_connection_async()
        assert result is None


class TestInitiateConnection:
    def test_initiate_no_redirect(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        expected = {"auth_url": "https://accounts.google.com/o/oauth2/auth?...", "state": "abc"}
        mock_client._post.return_value = expected
        result = api.initiate_connection()
        assert result == expected
        body = mock_client._post.call_args[0][1]
        assert "redirect_uri" not in body

    def test_initiate_with_redirect(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"auth_url": "https://...", "state": "xyz"}
        api.initiate_connection(redirect_uri="https://myapp.com/callback")
        body = mock_client._post.call_args[0][1]
        assert body["redirect_uri"] == "https://myapp.com/callback"

    @pytest.mark.asyncio
    async def test_initiate_connection_async(
        self, api: GmailAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post_async = AsyncMock(return_value={"auth_url": "https://...", "state": "s"})
        result = await api.initiate_connection_async(redirect_uri="https://app.com/cb")
        assert "auth_url" in result


class TestCompleteConnection:
    def test_complete_connection(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_CONNECTION
        result = api.complete_connection(code="auth-code-123", state="state-xyz")
        assert isinstance(result, GmailConnection)
        assert result.id == "conn-1"
        body = mock_client._post.call_args[0][1]
        assert body["code"] == "auth-code-123"
        assert body["state"] == "state-xyz"
        mock_client._post.assert_called_once_with("/api/v1/connectors/gmail/callback", body)

    @pytest.mark.asyncio
    async def test_complete_connection_async(
        self, api: GmailAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post_async = AsyncMock(return_value=SAMPLE_CONNECTION)
        result = await api.complete_connection_async(code="c", state="s")
        assert isinstance(result, GmailConnection)


class TestDisconnect:
    def test_disconnect(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        mock_client._delete.return_value = None
        result = api.disconnect()
        assert result is True
        mock_client._delete.assert_called_once_with("/api/v1/connectors/gmail")

    @pytest.mark.asyncio
    async def test_disconnect_async(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        mock_client._delete_async = AsyncMock(return_value=None)
        result = await api.disconnect_async()
        assert result is True


class TestSync:
    def test_sync(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"synced": 15, "new_messages": 3}
        result = api.sync()
        assert result["synced"] == 15
        mock_client._post.assert_called_once_with("/api/v1/connectors/gmail/sync", {})

    @pytest.mark.asyncio
    async def test_sync_async(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        mock_client._post_async = AsyncMock(return_value={"synced": 10})
        result = await api.sync_async()
        assert result["synced"] == 10


# =========================================================================
# Triage Rules
# =========================================================================


class TestListTriageRules:
    def test_list_triage_rules(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"rules": [SAMPLE_TRIAGE_RULE]}
        rules = api.list_triage_rules()
        assert len(rules) == 1
        assert isinstance(rules[0], EmailTriageRule)
        assert rules[0].name == "Urgent filter"
        assert rules[0].priority == 10
        mock_client._get.assert_called_once_with("/api/v1/connectors/gmail/triage/rules")

    def test_list_triage_rules_empty(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"rules": []}
        rules = api.list_triage_rules()
        assert rules == []

    @pytest.mark.asyncio
    async def test_list_triage_rules_async(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={"rules": [SAMPLE_TRIAGE_RULE]})
        rules = await api.list_triage_rules_async()
        assert len(rules) == 1
        assert rules[0].id == "rule-1"


class TestCreateTriageRule:
    def test_create_triage_rule(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"rule": SAMPLE_TRIAGE_RULE}
        rule = api.create_triage_rule(
            name="Urgent filter",
            conditions={"subject_contains": "URGENT"},
            actions=["flag", "notify"],
            priority=10,
        )
        assert isinstance(rule, EmailTriageRule)
        assert rule.name == "Urgent filter"
        assert rule.actions == ["flag", "notify"]
        body = mock_client._post.call_args[0][1]
        assert body["name"] == "Urgent filter"
        assert body["priority"] == 10

    def test_create_triage_rule_default_priority(
        self, api: GmailAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = SAMPLE_TRIAGE_RULE
        api.create_triage_rule(name="test", conditions={}, actions=["archive"])
        body = mock_client._post.call_args[0][1]
        assert body["priority"] == 0

    def test_create_triage_rule_response_without_rule_key(
        self, api: GmailAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = SAMPLE_TRIAGE_RULE
        rule = api.create_triage_rule(name="test", conditions={}, actions=[])
        assert rule.id == "rule-1"

    @pytest.mark.asyncio
    async def test_create_triage_rule_async(
        self, api: GmailAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post_async = AsyncMock(return_value={"rule": SAMPLE_TRIAGE_RULE})
        rule = await api.create_triage_rule_async(
            name="Urgent filter",
            conditions={"subject_contains": "URGENT"},
            actions=["flag"],
        )
        assert rule.name == "Urgent filter"


class TestUpdateTriageRule:
    def test_update_triage_rule_all_fields(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        updated = {**SAMPLE_TRIAGE_RULE, "name": "Renamed", "priority": 20}
        mock_client._patch.return_value = {"rule": updated}
        rule = api.update_triage_rule(
            rule_id="rule-1",
            name="Renamed",
            conditions={"from": "boss@example.com"},
            actions=["star"],
            enabled=False,
            priority=20,
        )
        assert isinstance(rule, EmailTriageRule)
        assert rule.name == "Renamed"
        body = mock_client._patch.call_args[0][1]
        assert body["name"] == "Renamed"
        assert body["enabled"] is False
        assert body["priority"] == 20

    def test_update_triage_rule_partial(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        mock_client._patch.return_value = {"rule": SAMPLE_TRIAGE_RULE}
        api.update_triage_rule(rule_id="rule-1", name="Updated")
        body = mock_client._patch.call_args[0][1]
        assert body == {"name": "Updated"}

    def test_update_triage_rule_only_enabled(
        self, api: GmailAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._patch.return_value = {"rule": SAMPLE_TRIAGE_RULE}
        api.update_triage_rule(rule_id="rule-1", enabled=False)
        body = mock_client._patch.call_args[0][1]
        assert body == {"enabled": False}

    def test_update_triage_rule_url(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        mock_client._patch.return_value = {"rule": SAMPLE_TRIAGE_RULE}
        api.update_triage_rule(rule_id="rule-42")
        url = mock_client._patch.call_args[0][0]
        assert url == "/api/v1/connectors/gmail/triage/rules/rule-42"

    @pytest.mark.asyncio
    async def test_update_triage_rule_async(
        self, api: GmailAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._patch_async = AsyncMock(return_value={"rule": SAMPLE_TRIAGE_RULE})
        rule = await api.update_triage_rule_async(rule_id="rule-1", priority=5)
        assert isinstance(rule, EmailTriageRule)


class TestDeleteTriageRule:
    def test_delete_triage_rule(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        mock_client._delete.return_value = None
        result = api.delete_triage_rule("rule-1")
        assert result is True
        mock_client._delete.assert_called_once_with("/api/v1/connectors/gmail/triage/rules/rule-1")

    @pytest.mark.asyncio
    async def test_delete_triage_rule_async(
        self, api: GmailAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._delete_async = AsyncMock(return_value=None)
        result = await api.delete_triage_rule_async("rule-1")
        assert result is True


# =========================================================================
# Debate Configuration
# =========================================================================


class TestListDebateConfigs:
    def test_list_debate_configs(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"configs": [SAMPLE_DEBATE_CONFIG]}
        configs = api.list_debate_configs()
        assert len(configs) == 1
        assert isinstance(configs[0], EmailDebateConfig)
        assert configs[0].name == "Auto-debate important emails"
        assert configs[0].agents == ["claude", "gpt4"]
        assert configs[0].auto_reply is True
        mock_client._get.assert_called_once_with("/api/v1/connectors/gmail/debates")

    def test_list_debate_configs_empty(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"configs": []}
        configs = api.list_debate_configs()
        assert configs == []

    @pytest.mark.asyncio
    async def test_list_debate_configs_async(
        self, api: GmailAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value={"configs": [SAMPLE_DEBATE_CONFIG]})
        configs = await api.list_debate_configs_async()
        assert len(configs) == 1


class TestCreateDebateConfig:
    def test_create_debate_config_full(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"config": SAMPLE_DEBATE_CONFIG}
        config = api.create_debate_config(
            name="Auto-debate important emails",
            trigger_conditions={"label": "important"},
            agents=["claude", "gpt4"],
            debate_template="quick_review",
            auto_reply=True,
        )
        assert isinstance(config, EmailDebateConfig)
        assert config.debate_template == "quick_review"
        body = mock_client._post.call_args[0][1]
        assert body["name"] == "Auto-debate important emails"
        assert body["agents"] == ["claude", "gpt4"]
        assert body["debate_template"] == "quick_review"
        assert body["auto_reply"] is True

    def test_create_debate_config_minimal(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_DEBATE_CONFIG
        api.create_debate_config(
            name="simple",
            trigger_conditions={"from": "boss@example.com"},
        )
        body = mock_client._post.call_args[0][1]
        assert body["name"] == "simple"
        assert body["auto_reply"] is False
        assert "agents" not in body
        assert "debate_template" not in body

    def test_create_debate_config_response_without_config_key(
        self, api: GmailAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = SAMPLE_DEBATE_CONFIG
        config = api.create_debate_config(name="test", trigger_conditions={"label": "inbox"})
        assert config.id == "dconf-1"

    @pytest.mark.asyncio
    async def test_create_debate_config_async(
        self, api: GmailAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post_async = AsyncMock(return_value={"config": SAMPLE_DEBATE_CONFIG})
        config = await api.create_debate_config_async(
            name="async config",
            trigger_conditions={"label": "important"},
            agents=["claude"],
            auto_reply=True,
        )
        assert isinstance(config, EmailDebateConfig)


# =========================================================================
# Processed Emails
# =========================================================================


class TestListProcessedEmails:
    def test_list_default(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {
            "emails": [SAMPLE_PROCESSED_EMAIL],
            "total": 1,
        }
        emails, total = api.list_processed_emails()
        assert len(emails) == 1
        assert total == 1
        assert isinstance(emails[0], ProcessedEmail)
        assert emails[0].subject == "Q4 Budget Review"
        assert emails[0].debate_id == "debate-xyz"
        params = mock_client._get.call_args[1]["params"]
        assert params["limit"] == 50
        assert params["offset"] == 0
        assert "status" not in params

    def test_list_with_status_filter(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"emails": [], "total": 0}
        api.list_processed_emails(status="completed", limit=10, offset=5)
        params = mock_client._get.call_args[1]["params"]
        assert params["status"] == "completed"
        assert params["limit"] == 10
        assert params["offset"] == 5

    def test_list_total_fallback(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"emails": [SAMPLE_PROCESSED_EMAIL]}
        emails, total = api.list_processed_emails()
        assert total == 1

    @pytest.mark.asyncio
    async def test_list_processed_emails_async(
        self, api: GmailAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(
            return_value={"emails": [SAMPLE_PROCESSED_EMAIL], "total": 1}
        )
        emails, total = await api.list_processed_emails_async()
        assert len(emails) == 1
        assert total == 1

    @pytest.mark.asyncio
    async def test_list_processed_emails_async_with_filter(
        self, api: GmailAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value={"emails": [], "total": 0})
        emails, total = await api.list_processed_emails_async(status="failed")
        assert emails == []
        assert total == 0


# =========================================================================
# Stats
# =========================================================================


class TestGetStats:
    def test_get_stats(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_STATS
        stats = api.get_stats()
        assert isinstance(stats, GmailStats)
        assert stats.total_processed == 150
        assert stats.debates_triggered == 42
        assert stats.auto_replies_sent == 30
        assert stats.errors == 3
        assert stats.avg_processing_time_ms == 1234.5
        mock_client._get.assert_called_once_with("/api/v1/connectors/gmail/stats")

    @pytest.mark.asyncio
    async def test_get_stats_async(self, api: GmailAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_STATS)
        stats = await api.get_stats_async()
        assert stats.total_processed == 150


# =========================================================================
# Parse Helpers
# =========================================================================


class TestParseConnection:
    def test_parse_datetimes(self, api: GmailAPI) -> None:
        conn = api._parse_connection(SAMPLE_CONNECTION)
        assert conn.connected_at is not None
        assert conn.connected_at.year == 2026
        assert conn.last_synced_at is not None
        assert conn.last_synced_at.year == 2026

    def test_parse_missing_datetimes(self, api: GmailAPI) -> None:
        data = {"id": "c1", "email": "a@b.com", "status": "connected"}
        conn = api._parse_connection(data)
        assert conn.connected_at is None
        assert conn.last_synced_at is None

    def test_parse_invalid_datetime(self, api: GmailAPI) -> None:
        data = {
            **SAMPLE_CONNECTION,
            "connected_at": "not-a-date",
            "last_synced_at": "also-bad",
        }
        conn = api._parse_connection(data)
        assert conn.connected_at is None
        assert conn.last_synced_at is None

    def test_parse_defaults(self, api: GmailAPI) -> None:
        conn = api._parse_connection({})
        assert conn.id == ""
        assert conn.email == ""
        assert conn.status == "connected"
        assert conn.scopes == []


class TestParseTriageRule:
    def test_parse_full(self, api: GmailAPI) -> None:
        rule = api._parse_triage_rule(SAMPLE_TRIAGE_RULE)
        assert rule.id == "rule-1"
        assert rule.name == "Urgent filter"
        assert rule.enabled is True
        assert rule.conditions == {"subject_contains": "URGENT"}
        assert rule.actions == ["flag", "notify"]
        assert rule.priority == 10

    def test_parse_defaults(self, api: GmailAPI) -> None:
        rule = api._parse_triage_rule({})
        assert rule.id == ""
        assert rule.name == ""
        assert rule.enabled is True
        assert rule.conditions == {}
        assert rule.actions == []
        assert rule.priority == 0


class TestParseDebateConfig:
    def test_parse_full(self, api: GmailAPI) -> None:
        config = api._parse_debate_config(SAMPLE_DEBATE_CONFIG)
        assert config.id == "dconf-1"
        assert config.name == "Auto-debate important emails"
        assert config.enabled is True
        assert config.trigger_conditions == {"label": "important"}
        assert config.debate_template == "quick_review"
        assert config.agents == ["claude", "gpt4"]
        assert config.auto_reply is True

    def test_parse_defaults(self, api: GmailAPI) -> None:
        config = api._parse_debate_config({})
        assert config.id == ""
        assert config.debate_template is None
        assert config.agents == []
        assert config.auto_reply is False


class TestParseProcessedEmail:
    def test_parse_full(self, api: GmailAPI) -> None:
        email = api._parse_processed_email(SAMPLE_PROCESSED_EMAIL)
        assert email.id == "pe-1"
        assert email.message_id == "msg-abc"
        assert email.subject == "Q4 Budget Review"
        assert email.sender == "cfo@example.com"
        assert email.status == "completed"
        assert email.debate_id == "debate-xyz"
        assert email.processed_at is not None
        assert email.processed_at.year == 2026
        assert email.summary == "Budget approved with minor adjustments."

    def test_parse_missing_optional_fields(self, api: GmailAPI) -> None:
        data = {
            "id": "pe-2",
            "message_id": "msg-def",
            "subject": "Hello",
            "sender": "user@test.com",
            "status": "pending",
        }
        email = api._parse_processed_email(data)
        assert email.debate_id is None
        assert email.processed_at is None
        assert email.summary is None

    def test_parse_invalid_datetime(self, api: GmailAPI) -> None:
        data = {**SAMPLE_PROCESSED_EMAIL, "processed_at": "garbage"}
        email = api._parse_processed_email(data)
        assert email.processed_at is None

    def test_parse_defaults(self, api: GmailAPI) -> None:
        email = api._parse_processed_email({})
        assert email.id == ""
        assert email.message_id == ""
        assert email.status == "pending"


class TestParseStats:
    def test_parse_full(self, api: GmailAPI) -> None:
        stats = api._parse_stats(SAMPLE_STATS)
        assert stats.total_processed == 150
        assert stats.debates_triggered == 42
        assert stats.auto_replies_sent == 30
        assert stats.errors == 3
        assert stats.avg_processing_time_ms == 1234.5

    def test_parse_defaults(self, api: GmailAPI) -> None:
        stats = api._parse_stats({})
        assert stats.total_processed == 0
        assert stats.debates_triggered == 0
        assert stats.auto_replies_sent == 0
        assert stats.errors == 0
        assert stats.avg_processing_time_ms == 0.0


# =========================================================================
# Dataclass Construction
# =========================================================================


class TestDataclasses:
    def test_gmail_connection_defaults(self) -> None:
        conn = GmailConnection(id="c1", email="a@b.com", status="connected")
        assert conn.scopes == []
        assert conn.connected_at is None
        assert conn.last_synced_at is None

    def test_email_triage_rule_defaults(self) -> None:
        rule = EmailTriageRule(id="r1", name="test")
        assert rule.enabled is True
        assert rule.conditions == {}
        assert rule.actions == []
        assert rule.priority == 0

    def test_email_debate_config_defaults(self) -> None:
        config = EmailDebateConfig(id="d1", name="test")
        assert config.enabled is True
        assert config.trigger_conditions == {}
        assert config.debate_template is None
        assert config.agents == []
        assert config.auto_reply is False

    def test_processed_email_defaults(self) -> None:
        email = ProcessedEmail(
            id="p1", message_id="m1", subject="sub", sender="s@s.com", status="pending"
        )
        assert email.debate_id is None
        assert email.processed_at is None
        assert email.summary is None

    def test_gmail_stats_defaults(self) -> None:
        stats = GmailStats()
        assert stats.total_processed == 0
        assert stats.debates_triggered == 0
        assert stats.auto_replies_sent == 0
        assert stats.errors == 0
        assert stats.avg_processing_time_ms == 0.0

    def test_gmail_stats_custom_values(self) -> None:
        stats = GmailStats(
            total_processed=100,
            debates_triggered=50,
            auto_replies_sent=25,
            errors=5,
            avg_processing_time_ms=999.9,
        )
        assert stats.total_processed == 100
        assert stats.avg_processing_time_ms == 999.9
