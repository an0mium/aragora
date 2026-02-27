"""Tests for the Decision Plan exporter system.

Covers:
- TicketData model
- ExportReceipt lifecycle
- DecisionExporter ticket extraction and orchestration
- WebhookAdapter (with mocked HTTP)
- JiraAdapter (with mocked HTTP)
- LinearAdapter (with mocked HTTP)
- Error handling and edge cases
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.integrations.exporters.base import (
    ExportAdapter,
    ExportReceipt,
    ExportStatus,
    TicketData,
    TicketPriority,
)
from aragora.integrations.exporters.exporter import (
    DecisionExporter,
    _complexity_to_priority,
)
from aragora.integrations.exporters.jira_adapter import (
    JiraAdapter,
    _PRIORITY_MAP as JIRA_PRIORITY_MAP,
)
from aragora.integrations.exporters.linear_adapter import (
    LinearAdapter,
    _PRIORITY_MAP as LINEAR_PRIORITY_MAP,
)
from aragora.integrations.exporters.webhook_adapter import WebhookAdapter, _sign_payload


# ---------------------------------------------------------------------------
# Helpers: Minimal stubs for DecisionPlan, ImplementTask, etc.
# ---------------------------------------------------------------------------


@dataclass
class _StubTask:
    id: str = "task-1"
    description: str = "Implement rate limiter"
    files: list[str] = field(default_factory=lambda: ["api/limiter.py"])
    complexity: str = "moderate"
    task_type: str | None = "backend"
    dependencies: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)


@dataclass
class _StubImplementPlan:
    tasks: list[_StubTask] = field(default_factory=lambda: [_StubTask()])
    design_hash: str = "abc123"


@dataclass
class _StubRisk:
    id: str = "risk-1"
    title: str = "Performance risk"
    description: str = "May hit rate limits"
    level: Any = None  # Will be set per test
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _StubRiskRegister:
    risks: list[_StubRisk] = field(default_factory=list)


@dataclass
class _StubTestCase:
    name: str = "Verify rate limiter works"
    description: str = ""
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _StubVerificationPlan:
    test_cases: list[_StubTestCase] = field(default_factory=list)


@dataclass
class _StubDecisionPlan:
    id: str = "dp-test123"
    debate_id: str = "debate-456"
    task: str = "Design a rate limiting system"
    implement_plan: _StubImplementPlan | None = None
    risk_register: _StubRiskRegister | None = None
    verification_plan: _StubVerificationPlan | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _mock_aiohttp_session(
    *,
    status: int = 200,
    json_data: dict[str, Any] | None = None,
    text_data: str = "",
) -> MagicMock:
    """Build a mock aiohttp.ClientSession with a mock response.

    Uses MagicMock for ``session.post`` (not AsyncMock) so that
    ``async with session.post(...)`` works correctly (aiohttp returns
    an async context manager, not a coroutine).
    """
    mock_response = AsyncMock()
    mock_response.status = status
    if json_data is not None:
        mock_response.json = AsyncMock(return_value=json_data)
    mock_response.text = AsyncMock(return_value=text_data)
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.post.return_value = mock_response
    return mock_session


def _make_plan(
    *,
    num_tasks: int = 2,
    with_risks: bool = False,
    with_verification: bool = False,
) -> _StubDecisionPlan:
    """Build a stub DecisionPlan for testing."""
    tasks = [
        _StubTask(id=f"task-{i}", description=f"Task {i} description", complexity="moderate")
        for i in range(1, num_tasks + 1)
    ]
    plan = _StubDecisionPlan(implement_plan=_StubImplementPlan(tasks=tasks))

    if with_risks:
        # Use a mock enum-like object for level.value
        risk_level = MagicMock()
        risk_level.value = "high"
        plan.risk_register = _StubRiskRegister(
            risks=[
                _StubRisk(
                    id="risk-1",
                    title="High risk",
                    level=risk_level,
                    task_id="task-1",
                )
            ]
        )

    if with_verification:
        plan.verification_plan = _StubVerificationPlan(
            test_cases=[
                _StubTestCase(name="Check task-1 output", task_id="task-1"),
                _StubTestCase(name="Verify task-2 result", task_id="task-2"),
            ]
        )

    return plan


# ---------------------------------------------------------------------------
# TicketData tests
# ---------------------------------------------------------------------------


class TestTicketData:
    def test_to_dict(self):
        ticket = TicketData(
            title="Test ticket",
            description="Description here",
            priority=TicketPriority.HIGH,
            labels=["aragora"],
            plan_id="dp-1",
            debate_id="d-1",
            task_id="t-1",
        )
        d = ticket.to_dict()
        assert d["title"] == "Test ticket"
        assert d["priority"] == "high"
        assert d["plan_id"] == "dp-1"

    def test_content_hash_deterministic(self):
        ticket = TicketData(title="A", description="B")
        h1 = ticket.content_hash
        h2 = ticket.content_hash
        assert h1 == h2
        assert len(h1) == 16

    def test_content_hash_changes_with_content(self):
        t1 = TicketData(title="A", description="B")
        t2 = TicketData(title="C", description="D")
        assert t1.content_hash != t2.content_hash


# ---------------------------------------------------------------------------
# ExportReceipt tests
# ---------------------------------------------------------------------------


class TestExportReceipt:
    def test_initial_state(self):
        r = ExportReceipt(adapter_name="test")
        assert r.status == ExportStatus.PENDING
        assert r.tickets_exported == 0
        assert r.completed_at is None

    def test_mark_success(self):
        r = ExportReceipt(adapter_name="test")
        r.mark_success()
        assert r.status == ExportStatus.SUCCESS
        assert r.completed_at is not None

    def test_mark_failed(self):
        r = ExportReceipt(adapter_name="test")
        r.mark_failed("connection refused")
        assert r.status == ExportStatus.FAILED
        assert r.error == "connection refused"
        assert r.completed_at is not None

    def test_to_dict_includes_duration(self):
        r = ExportReceipt(adapter_name="test", created_at=100.0)
        r.completed_at = 102.5
        r.status = ExportStatus.SUCCESS
        d = r.to_dict()
        assert d["duration_s"] == 2.5

    def test_to_dict_without_completion(self):
        r = ExportReceipt(adapter_name="test")
        d = r.to_dict()
        assert "completed_at" not in d
        assert "duration_s" not in d


# ---------------------------------------------------------------------------
# DecisionExporter tests
# ---------------------------------------------------------------------------


class TestDecisionExporter:
    def test_register_unregister_adapter(self):
        exporter = DecisionExporter()
        adapter = MagicMock(spec=ExportAdapter)
        adapter.name = "mock"
        exporter.register_adapter(adapter)
        assert len(exporter.adapters) == 1
        assert exporter.unregister_adapter("mock")
        assert len(exporter.adapters) == 0
        assert not exporter.unregister_adapter("nonexistent")

    def test_extract_tickets_from_plan(self):
        plan = _make_plan(num_tasks=3)
        exporter = DecisionExporter()
        tickets = exporter.extract_tickets(plan)
        assert len(tickets) == 3
        for t in tickets:
            assert t.title.startswith("[Aragora]")
            assert t.plan_id == "dp-test123"
            assert t.debate_id == "debate-456"
            assert "aragora" in t.labels

    def test_extract_tickets_empty_plan(self):
        plan = _StubDecisionPlan()
        exporter = DecisionExporter()
        tickets = exporter.extract_tickets(plan)
        assert tickets == []

    def test_extract_tickets_with_risk_priority(self):
        plan = _make_plan(num_tasks=2, with_risks=True)
        exporter = DecisionExporter()
        tickets = exporter.extract_tickets(plan)
        # task-1 has a high risk
        task1_ticket = [t for t in tickets if t.task_id == "task-1"][0]
        assert task1_ticket.priority == TicketPriority.HIGH
        # task-2 has no risk, falls back to complexity
        task2_ticket = [t for t in tickets if t.task_id == "task-2"][0]
        assert task2_ticket.priority == TicketPriority.MEDIUM

    def test_extract_tickets_with_verification_criteria(self):
        plan = _make_plan(num_tasks=2, with_verification=True)
        exporter = DecisionExporter()
        tickets = exporter.extract_tickets(plan)
        task1_ticket = [t for t in tickets if t.task_id == "task-1"][0]
        assert "Check task-1 output" in task1_ticket.acceptance_criteria

    def test_description_includes_context(self):
        plan = _make_plan(num_tasks=1)
        exporter = DecisionExporter()
        tickets = exporter.extract_tickets(plan)
        desc = tickets[0].description
        assert "dp-test123" in desc
        assert "debate-456" in desc
        assert "Decision Context" in desc

    @pytest.mark.asyncio
    async def test_export_no_adapters(self):
        exporter = DecisionExporter()
        plan = _make_plan()
        receipts = await exporter.export(plan)
        assert receipts == []

    @pytest.mark.asyncio
    async def test_export_skipped_when_no_tasks(self):
        exporter = DecisionExporter()
        adapter = AsyncMock(spec=ExportAdapter)
        adapter.name = "mock"
        exporter.register_adapter(adapter)

        plan = _StubDecisionPlan()  # No tasks
        receipts = await exporter.export(plan)
        assert len(receipts) == 1
        assert receipts[0].status == ExportStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_export_dispatches_to_all_adapters(self):
        exporter = DecisionExporter()
        adapter1 = AsyncMock(spec=ExportAdapter)
        adapter1.name = "a1"
        adapter1.export_tickets.return_value = ExportReceipt(
            adapter_name="a1", status=ExportStatus.SUCCESS, tickets_exported=2
        )
        adapter2 = AsyncMock(spec=ExportAdapter)
        adapter2.name = "a2"
        adapter2.export_tickets.return_value = ExportReceipt(
            adapter_name="a2", status=ExportStatus.SUCCESS, tickets_exported=2
        )
        exporter.register_adapter(adapter1)
        exporter.register_adapter(adapter2)

        plan = _make_plan()
        receipts = await exporter.export(plan)
        assert len(receipts) == 2
        assert all(r.status == ExportStatus.SUCCESS for r in receipts)

    @pytest.mark.asyncio
    async def test_export_handles_adapter_error(self):
        exporter = DecisionExporter()
        adapter = AsyncMock(spec=ExportAdapter)
        adapter.name = "broken"
        adapter.export_tickets.side_effect = ConnectionError("timeout")
        exporter.register_adapter(adapter)

        plan = _make_plan()
        receipts = await exporter.export(plan)
        assert len(receipts) == 1
        assert receipts[0].status == ExportStatus.FAILED
        assert "broken" in (receipts[0].error or "")


class TestComplexityToPriority:
    def test_complex(self):
        assert _complexity_to_priority("complex") == TicketPriority.HIGH

    def test_moderate(self):
        assert _complexity_to_priority("moderate") == TicketPriority.MEDIUM

    def test_simple(self):
        assert _complexity_to_priority("simple") == TicketPriority.LOW

    def test_unknown(self):
        assert _complexity_to_priority("unknown") == TicketPriority.MEDIUM


# ---------------------------------------------------------------------------
# WebhookAdapter tests
# ---------------------------------------------------------------------------


class TestWebhookAdapter:
    def test_name(self):
        adapter = WebhookAdapter(url="https://example.com/hook")
        assert adapter.name == "webhook"

    @pytest.mark.asyncio
    async def test_export_ticket_success(self):
        adapter = WebhookAdapter(url="https://example.com/hook")
        adapter._session = _mock_aiohttp_session(status=200)

        ticket = TicketData(title="Test", description="Desc", task_id="t-1")
        result = await adapter.export_ticket(ticket)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_export_ticket_failure(self):
        adapter = WebhookAdapter(url="https://example.com/hook")
        adapter._session = _mock_aiohttp_session(status=500)

        ticket = TicketData(title="Test", description="Desc", task_id="t-1")
        result = await adapter.export_ticket(ticket)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_batch_mode(self):
        adapter = WebhookAdapter(url="https://example.com/hook", batch_mode=True)
        mock_session = _mock_aiohttp_session(status=200)
        adapter._session = mock_session

        tickets = [
            TicketData(title=f"T{i}", description="D", task_id=f"t-{i}", plan_id="p1")
            for i in range(3)
        ]
        receipt = await adapter.export_tickets(tickets)
        assert receipt.tickets_exported == 3
        # In batch mode, only one POST call is made
        assert mock_session.post.call_count == 1

    def test_sign_payload(self):
        sig = _sign_payload("my-secret", b'{"test": true}')
        assert sig.startswith("sha256=")
        assert len(sig) > 10

    @pytest.mark.asyncio
    async def test_signed_request_includes_header(self):
        adapter = WebhookAdapter(url="https://example.com/hook", secret="test-secret")
        mock_session = _mock_aiohttp_session(status=200)
        adapter._session = mock_session

        ticket = TicketData(title="Test", description="Desc")
        await adapter.export_ticket(ticket)

        call_kwargs = mock_session.post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        assert "X-Aragora-Signature" in headers

    @pytest.mark.asyncio
    async def test_custom_headers_included(self):
        adapter = WebhookAdapter(
            url="https://example.com/hook",
            headers={"X-Custom": "value"},
        )
        mock_session = _mock_aiohttp_session(status=200)
        adapter._session = mock_session

        ticket = TicketData(title="Test", description="Desc")
        await adapter.export_ticket(ticket)

        call_kwargs = mock_session.post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        assert headers.get("X-Custom") == "value"


# ---------------------------------------------------------------------------
# JiraAdapter tests
# ---------------------------------------------------------------------------


class TestJiraAdapter:
    def test_name(self):
        adapter = JiraAdapter(base_url="https://jira.example.com", project_key="PROJ")
        assert adapter.name == "jira"

    def test_priority_mapping(self):
        assert JIRA_PRIORITY_MAP[TicketPriority.CRITICAL] == "Highest"
        assert JIRA_PRIORITY_MAP[TicketPriority.HIGH] == "High"
        assert JIRA_PRIORITY_MAP[TicketPriority.MEDIUM] == "Medium"
        assert JIRA_PRIORITY_MAP[TicketPriority.LOW] == "Low"

    def test_format_issue_payload(self):
        adapter = JiraAdapter(
            base_url="https://jira.example.com",
            project_key="PROJ",
            email="bot@example.com",
            api_token="token",
        )
        ticket = TicketData(
            title="[Aragora] Fix bug",
            description="Full description",
            priority=TicketPriority.HIGH,
            labels=["aragora", "complexity:moderate"],
            plan_id="dp-1",
            debate_id="d-1",
            task_id="t-1",
            acceptance_criteria=["Must pass tests"],
        )
        payload = adapter._format_issue(ticket)
        fields = payload["fields"]
        assert fields["project"]["key"] == "PROJ"
        assert fields["summary"] == "[Aragora] Fix bug"
        assert fields["priority"]["name"] == "High"
        assert fields["issuetype"]["name"] == "Task"
        assert "aragora" in fields["labels"]

    def test_adf_description_structure(self):
        ticket = TicketData(
            title="Test",
            description="Desc",
            acceptance_criteria=["Criterion 1", "Criterion 2"],
            plan_id="dp-1",
            debate_id="d-1",
            task_id="t-1",
        )
        adf = JiraAdapter._to_adf(ticket)
        assert adf["type"] == "doc"
        assert adf["version"] == 1
        content_types = [c["type"] for c in adf["content"]]
        assert "paragraph" in content_types
        # Has acceptance criteria heading and bullet list
        assert "heading" in content_types
        assert "bulletList" in content_types
        # Has provenance panel
        assert "panel" in content_types

    @pytest.mark.asyncio
    async def test_export_ticket_success(self):
        adapter = JiraAdapter(
            base_url="https://jira.example.com",
            project_key="PROJ",
            email="bot@example.com",
            api_token="token",
        )
        adapter._session = _mock_aiohttp_session(
            status=201, json_data={"key": "PROJ-42", "id": "10042"}
        )

        ticket = TicketData(title="Test", description="Desc", task_id="t-1")
        result = await adapter.export_ticket(ticket)
        assert result["success"] is True
        assert result["issue_key"] == "PROJ-42"
        assert "PROJ-42" in result["issue_url"]

    @pytest.mark.asyncio
    async def test_export_ticket_api_error(self):
        adapter = JiraAdapter(
            base_url="https://jira.example.com",
            project_key="PROJ",
            email="bot@example.com",
            api_token="token",
        )
        adapter._session = _mock_aiohttp_session(status=400, text_data="Bad Request: missing field")

        ticket = TicketData(title="Test", description="Desc", task_id="t-1")
        result = await adapter.export_ticket(ticket)
        assert result["success"] is False
        assert "400" in result["error"]

    def test_basic_auth_header(self):
        adapter = JiraAdapter(
            base_url="https://jira.example.com",
            project_key="PROJ",
            email="bot@example.com",
            api_token="token",
        )
        headers = adapter._auth_headers()
        assert headers["Authorization"].startswith("Basic ")

    def test_bearer_auth_header(self):
        adapter = JiraAdapter(
            base_url="https://jira.example.com",
            project_key="PROJ",
            bearer_token="my-pat",
        )
        headers = adapter._auth_headers()
        assert headers["Authorization"] == "Bearer my-pat"

    def test_component_and_labels(self):
        adapter = JiraAdapter(
            base_url="https://jira.example.com",
            project_key="PROJ",
            component="Backend",
            labels=["team:platform"],
        )
        ticket = TicketData(
            title="Test",
            description="Desc",
            labels=["aragora"],
        )
        payload = adapter._format_issue(ticket)
        fields = payload["fields"]
        assert fields["components"] == [{"name": "Backend"}]
        assert "team:platform" in fields["labels"]
        assert "aragora" in fields["labels"]


# ---------------------------------------------------------------------------
# LinearAdapter tests
# ---------------------------------------------------------------------------


class TestLinearAdapter:
    def test_name(self):
        adapter = LinearAdapter(api_key="lin_api_xxx", team_id="team-uuid")
        assert adapter.name == "linear"

    def test_priority_mapping(self):
        assert LINEAR_PRIORITY_MAP[TicketPriority.CRITICAL] == 1
        assert LINEAR_PRIORITY_MAP[TicketPriority.HIGH] == 2
        assert LINEAR_PRIORITY_MAP[TicketPriority.MEDIUM] == 3
        assert LINEAR_PRIORITY_MAP[TicketPriority.LOW] == 4

    def test_format_variables(self):
        adapter = LinearAdapter(
            api_key="lin_api_xxx",
            team_id="team-uuid",
            project_id="proj-uuid",
        )
        ticket = TicketData(
            title="[Aragora] Fix bug",
            description="Full description",
            priority=TicketPriority.HIGH,
            plan_id="dp-1",
            debate_id="d-1",
            task_id="t-1",
        )
        variables = adapter._format_variables(ticket)
        inp = variables["input"]
        assert inp["teamId"] == "team-uuid"
        assert inp["title"] == "[Aragora] Fix bug"
        assert inp["priority"] == 2
        assert inp["projectId"] == "proj-uuid"

    def test_markdown_description(self):
        ticket = TicketData(
            title="Test",
            description="Main description",
            acceptance_criteria=["Criterion A", "Criterion B"],
            plan_id="dp-1",
            debate_id="d-1",
            task_id="t-1",
        )
        md = LinearAdapter._format_markdown_description(ticket)
        assert "Main description" in md
        assert "- [ ] Criterion A" in md
        assert "dp-1" in md

    @pytest.mark.asyncio
    async def test_export_ticket_success(self):
        adapter = LinearAdapter(api_key="lin_api_xxx", team_id="team-uuid")
        adapter._session = _mock_aiohttp_session(
            status=200,
            json_data={
                "data": {
                    "issueCreate": {
                        "success": True,
                        "issue": {
                            "id": "issue-uuid",
                            "identifier": "TEAM-42",
                            "title": "Test",
                            "url": "https://linear.app/team/issue/TEAM-42",
                        },
                    }
                }
            },
        )

        ticket = TicketData(title="Test", description="Desc", task_id="t-1")
        result = await adapter.export_ticket(ticket)
        assert result["success"] is True
        assert result["identifier"] == "TEAM-42"
        assert "linear.app" in result["issue_url"]

    @pytest.mark.asyncio
    async def test_export_ticket_graphql_error(self):
        adapter = LinearAdapter(api_key="lin_api_xxx", team_id="team-uuid")
        adapter._session = _mock_aiohttp_session(
            status=200,
            json_data={"errors": [{"message": "Team not found"}]},
        )

        ticket = TicketData(title="Test", description="Desc", task_id="t-1")
        result = await adapter.export_ticket(ticket)
        assert result["success"] is False
        assert "Team not found" in result["error"]

    @pytest.mark.asyncio
    async def test_export_ticket_api_failure(self):
        adapter = LinearAdapter(api_key="lin_api_xxx", team_id="team-uuid")
        adapter._session = _mock_aiohttp_session(status=401, text_data="Unauthorized")

        ticket = TicketData(title="Test", description="Desc", task_id="t-1")
        result = await adapter.export_ticket(ticket)
        assert result["success"] is False
        assert "401" in result["error"]

    def test_label_ids_included(self):
        adapter = LinearAdapter(
            api_key="lin_api_xxx",
            team_id="team-uuid",
            label_ids=["lbl-1", "lbl-2"],
        )
        ticket = TicketData(title="Test", description="Desc")
        variables = adapter._format_variables(ticket)
        assert variables["input"]["labelIds"] == ["lbl-1", "lbl-2"]

    def test_auth_header(self):
        adapter = LinearAdapter(api_key="lin_api_xxx", team_id="team-uuid")
        headers = adapter._headers()
        assert headers["Authorization"] == "lin_api_xxx"


# ---------------------------------------------------------------------------
# Integration tests: full export pipeline with mocked adapters
# ---------------------------------------------------------------------------


class TestExportPipelineIntegration:
    @pytest.mark.asyncio
    async def test_full_pipeline_with_multiple_adapters(self):
        """End-to-end test: plan -> tickets -> multiple adapters -> receipts."""
        plan = _make_plan(num_tasks=2, with_risks=True, with_verification=True)
        exporter = DecisionExporter()

        # Mock webhook adapter
        webhook = AsyncMock(spec=ExportAdapter)
        webhook.name = "webhook"
        webhook.export_tickets.return_value = ExportReceipt(
            adapter_name="webhook",
            status=ExportStatus.SUCCESS,
            tickets_exported=2,
            plan_id="dp-test123",
        )
        exporter.register_adapter(webhook)

        # Mock jira adapter
        jira = AsyncMock(spec=ExportAdapter)
        jira.name = "jira"
        jira.export_tickets.return_value = ExportReceipt(
            adapter_name="jira",
            status=ExportStatus.SUCCESS,
            tickets_exported=2,
            plan_id="dp-test123",
        )
        exporter.register_adapter(jira)

        receipts = await exporter.export(plan)
        assert len(receipts) == 2
        assert all(r.status == ExportStatus.SUCCESS for r in receipts)
        assert all(r.tickets_exported == 2 for r in receipts)

        # Verify extract_tickets was called implicitly
        for adapter in [webhook, jira]:
            call_args = adapter.export_tickets.call_args
            tickets = call_args[0][0]
            assert len(tickets) == 2

    @pytest.mark.asyncio
    async def test_partial_failure(self):
        """When one adapter fails, others should still succeed."""
        plan = _make_plan()
        exporter = DecisionExporter()

        good = AsyncMock(spec=ExportAdapter)
        good.name = "good"
        good.export_tickets.return_value = ExportReceipt(
            adapter_name="good",
            status=ExportStatus.SUCCESS,
            tickets_exported=2,
        )
        bad = AsyncMock(spec=ExportAdapter)
        bad.name = "bad"
        bad.export_tickets.side_effect = TimeoutError("timed out")

        exporter.register_adapter(good)
        exporter.register_adapter(bad)

        receipts = await exporter.export(plan)
        assert len(receipts) == 2
        good_receipt = [r for r in receipts if r.adapter_name == "good"][0]
        bad_receipt = [r for r in receipts if r.adapter_name == "bad"][0]
        assert good_receipt.status == ExportStatus.SUCCESS
        assert bad_receipt.status == ExportStatus.FAILED


# ---------------------------------------------------------------------------
# ExportAdapter base class tests
# ---------------------------------------------------------------------------


class TestExportAdapterBase:
    @pytest.mark.asyncio
    async def test_export_tickets_default_implementation(self):
        """Test the default sequential export_tickets from the base class."""

        class _TestAdapter(ExportAdapter):
            @property
            def name(self) -> str:
                return "test"

            async def export_ticket(self, ticket: TicketData) -> dict[str, Any]:
                return {"success": True, "id": ticket.task_id}

        adapter = _TestAdapter()
        tickets = [
            TicketData(
                title=f"T{i}", description="D", task_id=f"t-{i}", plan_id="p1", debate_id="d1"
            )
            for i in range(3)
        ]
        receipt = await adapter.export_tickets(tickets)
        assert receipt.status == ExportStatus.SUCCESS
        assert receipt.tickets_exported == 3
        assert receipt.tickets_failed == 0
        assert len(receipt.ticket_results) == 3
        assert receipt.plan_id == "p1"

    @pytest.mark.asyncio
    async def test_export_tickets_with_partial_failure(self):
        """When some tickets fail, receipt reflects partial success."""

        class _FlakeyAdapter(ExportAdapter):
            @property
            def name(self) -> str:
                return "flakey"

            async def export_ticket(self, ticket: TicketData) -> dict[str, Any]:
                if ticket.task_id == "t-2":
                    return {"success": False, "error": "rate limited"}
                return {"success": True}

        adapter = _FlakeyAdapter()
        tickets = [
            TicketData(title=f"T{i}", description="D", task_id=f"t-{i}") for i in range(1, 4)
        ]
        receipt = await adapter.export_tickets(tickets)
        assert receipt.tickets_exported == 2
        assert receipt.tickets_failed == 1
        assert receipt.status == ExportStatus.SUCCESS  # partial success

    @pytest.mark.asyncio
    async def test_export_tickets_all_fail(self):
        """When all tickets fail, receipt is marked FAILED."""

        class _BrokenAdapter(ExportAdapter):
            @property
            def name(self) -> str:
                return "broken"

            async def export_ticket(self, ticket: TicketData) -> dict[str, Any]:
                raise ConnectionError("down")

        adapter = _BrokenAdapter()
        tickets = [TicketData(title="T", description="D", task_id="t-1")]
        receipt = await adapter.export_tickets(tickets)
        assert receipt.status == ExportStatus.FAILED
        assert receipt.tickets_failed == 1
        assert receipt.tickets_exported == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_ticket_title_truncation(self):
        exporter = DecisionExporter()
        long_desc = "A" * 200
        title = exporter._make_title(long_desc, "plan task")
        assert len(title) <= 130  # [Aragora] prefix + 120 max
        assert title.endswith("...")

    def test_ticket_multiline_description_uses_first_line(self):
        exporter = DecisionExporter()
        desc = "First line\nSecond line\nThird line"
        title = exporter._make_title(desc, "plan")
        assert "[Aragora] First line" == title

    def test_extract_tickets_no_implement_plan(self):
        plan = _StubDecisionPlan(implement_plan=None)
        exporter = DecisionExporter()
        assert exporter.extract_tickets(plan) == []

    def test_extract_tickets_empty_tasks(self):
        plan = _StubDecisionPlan(implement_plan=_StubImplementPlan(tasks=[]))
        exporter = DecisionExporter()
        assert exporter.extract_tickets(plan) == []

    def test_risk_lookup_without_task_id(self):
        """Risks without task_id are not mapped."""
        risk = _StubRisk(id="r1", level=MagicMock(value="high"), task_id=None)
        plan = _StubDecisionPlan(risk_register=_StubRiskRegister(risks=[risk]))
        exporter = DecisionExporter()
        lookup = exporter._build_risk_lookup(plan)
        assert lookup == {}

    def test_criteria_lookup_without_task_id(self):
        """Test cases without task_id are not mapped."""
        tc = _StubTestCase(name="Generic test", task_id=None)
        plan = _StubDecisionPlan(verification_plan=_StubVerificationPlan(test_cases=[tc]))
        exporter = DecisionExporter()
        lookup = exporter._build_criteria_lookup(plan)
        assert lookup == {}
