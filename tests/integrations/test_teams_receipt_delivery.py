"""
Tests for Teams receipt delivery and approval button integration.

Verifies that:
- Receipts are formatted as Adaptive Cards with proper structure
- Approval buttons (Approve, Re-debate, Escalate) are included
- Cost information is displayed when available
- deliver_receipt_to_thread posts to the correct conversation/thread
- Receipt loading from store works and handles missing receipts
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeReceipt:
    """Minimal receipt for testing."""

    receipt_id: str = "rcpt_test123"
    verdict: str = "APPROVED"
    confidence: float = 0.85
    findings: list[Any] = field(default_factory=list)
    key_arguments: list[str] = field(default_factory=lambda: ["Arg 1", "Arg 2"])
    dissenting_views: list[str] = field(default_factory=list)
    dissents: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    tokens_used: int = 0


@dataclass
class FakeFinding:
    """Minimal finding for testing."""

    severity: str = "HIGH"
    description: str = "Test finding"
    level: str = ""


@pytest.fixture
def fake_receipt() -> FakeReceipt:
    return FakeReceipt()


@pytest.fixture
def fake_receipt_with_cost() -> FakeReceipt:
    return FakeReceipt(cost_usd=0.0512, tokens_used=6200)


@pytest.fixture
def lifecycle() -> Any:
    from aragora.integrations.teams_debate import TeamsDebateLifecycle

    lc = TeamsDebateLifecycle(bot_token="test-bot-token", service_url="https://test.bot")
    return lc


# ---------------------------------------------------------------------------
# _build_receipt_with_approval_card tests
# ---------------------------------------------------------------------------


class TestBuildReceiptWithApprovalCard:
    """Tests for the Adaptive Card receipt+approval builder."""

    def test_card_is_adaptive_card(self, fake_receipt: FakeReceipt) -> None:
        from aragora.integrations.teams_debate import _build_receipt_with_approval_card

        card = _build_receipt_with_approval_card(fake_receipt, debate_id="d1")
        assert card["type"] == "AdaptiveCard"
        assert card["version"] == "1.4"

    def test_card_contains_verdict_header(self, fake_receipt: FakeReceipt) -> None:
        from aragora.integrations.teams_debate import _build_receipt_with_approval_card

        card = _build_receipt_with_approval_card(fake_receipt, debate_id="d1")
        body = card.get("body", [])
        header_texts = [
            b.get("text", "") for b in body if b.get("type") == "TextBlock" and b.get("weight") == "Bolder"
        ]
        combined = " ".join(header_texts)
        assert "APPROVED" in combined

    def test_card_contains_confidence(self, fake_receipt: FakeReceipt) -> None:
        from aragora.integrations.teams_debate import _build_receipt_with_approval_card

        card = _build_receipt_with_approval_card(fake_receipt, debate_id="d1")
        body = card.get("body", [])
        fact_values = []
        for b in body:
            if b.get("type") == "FactSet":
                for fact in b.get("facts", []):
                    fact_values.append(fact.get("value", ""))
        combined = " ".join(fact_values)
        assert "85%" in combined

    def test_card_contains_approval_actions(self, fake_receipt: FakeReceipt) -> None:
        from aragora.integrations.teams_debate import _build_receipt_with_approval_card

        card = _build_receipt_with_approval_card(fake_receipt, debate_id="d1")
        actions = card.get("actions", [])
        action_titles = [a.get("title", "") for a in actions]
        assert "Approve Decision" in action_titles
        assert "Request Re-debate" in action_titles
        assert "Escalate" in action_titles

    def test_approve_action_is_positive(self, fake_receipt: FakeReceipt) -> None:
        from aragora.integrations.teams_debate import _build_receipt_with_approval_card

        card = _build_receipt_with_approval_card(fake_receipt, debate_id="d1")
        actions = card.get("actions", [])
        approve = next((a for a in actions if a.get("title") == "Approve Decision"), None)
        assert approve is not None
        assert approve.get("style") == "positive"

    def test_escalate_action_is_destructive(self, fake_receipt: FakeReceipt) -> None:
        from aragora.integrations.teams_debate import _build_receipt_with_approval_card

        card = _build_receipt_with_approval_card(fake_receipt, debate_id="d1")
        actions = card.get("actions", [])
        escalate = next((a for a in actions if a.get("title") == "Escalate"), None)
        assert escalate is not None
        assert escalate.get("style") == "destructive"

    def test_action_data_contains_debate_id(self, fake_receipt: FakeReceipt) -> None:
        from aragora.integrations.teams_debate import _build_receipt_with_approval_card

        card = _build_receipt_with_approval_card(fake_receipt, debate_id="test-debate-42")
        actions = card.get("actions", [])
        submit_actions = [a for a in actions if a.get("type") == "Action.Submit"]
        for action in submit_actions:
            data = action.get("data", {})
            assert data.get("debate_id") == "test-debate-42"

    def test_view_receipt_button_when_url_provided(self, fake_receipt: FakeReceipt) -> None:
        from aragora.integrations.teams_debate import _build_receipt_with_approval_card

        card = _build_receipt_with_approval_card(
            fake_receipt, debate_id="d1", receipt_url="https://example.com/receipt"
        )
        actions = card.get("actions", [])
        open_url = [a for a in actions if a.get("type") == "Action.OpenUrl"]
        assert len(open_url) == 1
        assert open_url[0]["url"] == "https://example.com/receipt"

    def test_no_open_url_without_receipt_url(self, fake_receipt: FakeReceipt) -> None:
        from aragora.integrations.teams_debate import _build_receipt_with_approval_card

        card = _build_receipt_with_approval_card(fake_receipt, debate_id="d1")
        actions = card.get("actions", [])
        open_url = [a for a in actions if a.get("type") == "Action.OpenUrl"]
        assert len(open_url) == 0

    def test_cost_info_included_when_available(self, fake_receipt_with_cost: FakeReceipt) -> None:
        from aragora.integrations.teams_debate import _build_receipt_with_approval_card

        card = _build_receipt_with_approval_card(
            fake_receipt_with_cost, debate_id="d1"
        )
        body = card.get("body", [])
        fact_values = []
        for b in body:
            if b.get("type") == "FactSet":
                for fact in b.get("facts", []):
                    fact_values.append(fact.get("value", ""))
        combined = " ".join(fact_values)
        assert "$0.0512" in combined
        assert "6,200" in combined

    def test_no_cost_factset_when_zero(self, fake_receipt: FakeReceipt) -> None:
        from aragora.integrations.teams_debate import _build_receipt_with_approval_card

        card = _build_receipt_with_approval_card(fake_receipt, debate_id="d1")
        body = card.get("body", [])
        # There should be FactSets for verdict/confidence but NOT cost
        all_fact_values = []
        for b in body:
            if b.get("type") == "FactSet":
                for fact in b.get("facts", []):
                    all_fact_values.append(fact.get("title", ""))
        assert "Cost" not in all_fact_values

    def test_card_schema_present(self, fake_receipt: FakeReceipt) -> None:
        from aragora.integrations.teams_debate import _build_receipt_with_approval_card

        card = _build_receipt_with_approval_card(fake_receipt, debate_id="d1")
        assert "$schema" in card

    def test_key_arguments_in_card(self, fake_receipt: FakeReceipt) -> None:
        from aragora.integrations.teams_debate import _build_receipt_with_approval_card

        card = _build_receipt_with_approval_card(fake_receipt, debate_id="d1")
        all_text = _extract_card_text(card)
        assert "Arg 1" in all_text
        assert "Arg 2" in all_text

    def test_rejected_verdict_in_card(self) -> None:
        from aragora.integrations.teams_debate import _build_receipt_with_approval_card

        receipt = FakeReceipt(verdict="REJECTED")
        card = _build_receipt_with_approval_card(receipt, debate_id="d1")
        all_text = _extract_card_text(card)
        assert "REJECTED" in all_text


# ---------------------------------------------------------------------------
# deliver_receipt_to_thread tests
# ---------------------------------------------------------------------------


class TestDeliverReceiptToThread:
    """Tests for the lifecycle deliver_receipt_to_thread method."""

    @pytest.mark.asyncio
    async def test_deliver_with_provided_receipt(self, lifecycle: Any, fake_receipt: FakeReceipt) -> None:
        lifecycle._send_card_to_thread = AsyncMock(return_value=True)
        result = await lifecycle.deliver_receipt_to_thread(
            debate_id="d1",
            conversation_id="conv-123",
            reply_to_id="msg-456",
            receipt=fake_receipt,
        )
        assert result is True
        lifecycle._send_card_to_thread.assert_called_once()

    @pytest.mark.asyncio
    async def test_deliver_returns_false_when_no_receipt(self, lifecycle: Any) -> None:
        result = await lifecycle.deliver_receipt_to_thread(
            debate_id="nonexistent",
            conversation_id="conv-123",
            reply_to_id="msg-456",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_deliver_builds_receipt_url(
        self, lifecycle: Any, fake_receipt: FakeReceipt
    ) -> None:
        lifecycle._send_card_to_thread = AsyncMock(return_value=True)
        with patch.dict(os.environ, {"ARAGORA_PUBLIC_URL": "https://test.aragora.ai"}):
            await lifecycle.deliver_receipt_to_thread(
                debate_id="d1",
                conversation_id="conv-123",
                reply_to_id="msg-456",
                receipt=fake_receipt,
            )
        assert lifecycle._send_card_to_thread.called

    @pytest.mark.asyncio
    async def test_deliver_uses_custom_receipt_url(
        self, lifecycle: Any, fake_receipt: FakeReceipt
    ) -> None:
        lifecycle._send_card_to_thread = AsyncMock(return_value=True)
        await lifecycle.deliver_receipt_to_thread(
            debate_id="d1",
            conversation_id="conv-123",
            reply_to_id="msg-456",
            receipt=fake_receipt,
            receipt_url="https://custom.url/receipt",
        )
        assert lifecycle._send_card_to_thread.called


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_card_text(card: dict[str, Any]) -> str:
    """Extract all text content from an Adaptive Card."""
    parts: list[str] = []
    for b in card.get("body", []):
        if b.get("type") == "TextBlock":
            parts.append(b.get("text", ""))
        if b.get("type") == "FactSet":
            for fact in b.get("facts", []):
                parts.append(fact.get("title", ""))
                parts.append(fact.get("value", ""))
        if b.get("type") == "Container":
            for item in b.get("items", []):
                if item.get("type") == "TextBlock":
                    parts.append(item.get("text", ""))
    return " ".join(parts)
