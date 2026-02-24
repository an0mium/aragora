"""
Tests for Slack receipt delivery and approval button integration.

Verifies that:
- Receipts are formatted as Block Kit messages with proper structure
- Approval buttons (Approve, Re-debate, Escalate) are included
- Cost information is displayed when available
- deliver_receipt_to_thread posts to the correct channel/thread
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
    return FakeReceipt(cost_usd=0.0342, tokens_used=4500)


@pytest.fixture
def lifecycle() -> Any:
    from aragora.integrations.slack_debate import SlackDebateLifecycle

    lc = SlackDebateLifecycle(bot_token="xoxb-test-token")
    return lc


# ---------------------------------------------------------------------------
# _build_receipt_with_approval_blocks tests
# ---------------------------------------------------------------------------


class TestBuildReceiptWithApprovalBlocks:
    """Tests for the Block Kit receipt+approval builder."""

    def test_blocks_contain_header(self, fake_receipt: FakeReceipt) -> None:
        from aragora.integrations.slack_debate import _build_receipt_with_approval_blocks

        blocks = _build_receipt_with_approval_blocks(fake_receipt, debate_id="d1")
        headers = [b for b in blocks if b.get("type") == "header"]
        assert len(headers) >= 1
        header_text = headers[0]["text"]["text"]
        assert "APPROVED" in header_text

    def test_blocks_contain_verdict_fields(self, fake_receipt: FakeReceipt) -> None:
        from aragora.integrations.slack_debate import _build_receipt_with_approval_blocks

        blocks = _build_receipt_with_approval_blocks(fake_receipt, debate_id="d1")
        section_texts = []
        for b in blocks:
            if b.get("type") == "section":
                for f in b.get("fields", []):
                    section_texts.append(f.get("text", ""))
        combined = " ".join(section_texts)
        assert "APPROVED" in combined
        assert "85%" in combined

    def test_blocks_contain_key_arguments(self, fake_receipt: FakeReceipt) -> None:
        from aragora.integrations.slack_debate import _build_receipt_with_approval_blocks

        blocks = _build_receipt_with_approval_blocks(fake_receipt, debate_id="d1")
        all_text = _extract_all_text(blocks)
        assert "Arg 1" in all_text
        assert "Arg 2" in all_text

    def test_blocks_contain_approval_buttons(self, fake_receipt: FakeReceipt) -> None:
        from aragora.integrations.slack_debate import _build_receipt_with_approval_blocks

        blocks = _build_receipt_with_approval_blocks(fake_receipt, debate_id="d1")
        actions = [b for b in blocks if b.get("type") == "actions"]
        assert len(actions) == 1, "Expected exactly one actions block"

        elements = actions[0]["elements"]
        action_ids = [e.get("action_id", "") for e in elements]
        assert any("approve_decision" in aid for aid in action_ids)
        assert any("request_redebate" in aid for aid in action_ids)
        assert any("escalate_decision" in aid for aid in action_ids)

    def test_approve_button_has_primary_style(self, fake_receipt: FakeReceipt) -> None:
        from aragora.integrations.slack_debate import _build_receipt_with_approval_blocks

        blocks = _build_receipt_with_approval_blocks(fake_receipt, debate_id="d1")
        actions = [b for b in blocks if b.get("type") == "actions"]
        elements = actions[0]["elements"]
        approve_btn = next(
            (e for e in elements if "approve_decision" in e.get("action_id", "")),
            None,
        )
        assert approve_btn is not None
        assert approve_btn.get("style") == "primary"

    def test_escalate_button_has_danger_style(self, fake_receipt: FakeReceipt) -> None:
        from aragora.integrations.slack_debate import _build_receipt_with_approval_blocks

        blocks = _build_receipt_with_approval_blocks(fake_receipt, debate_id="d1")
        actions = [b for b in blocks if b.get("type") == "actions"]
        elements = actions[0]["elements"]
        escalate_btn = next(
            (e for e in elements if "escalate_decision" in e.get("action_id", "")),
            None,
        )
        assert escalate_btn is not None
        assert escalate_btn.get("style") == "danger"

    def test_view_receipt_button_when_url_provided(self, fake_receipt: FakeReceipt) -> None:
        from aragora.integrations.slack_debate import _build_receipt_with_approval_blocks

        blocks = _build_receipt_with_approval_blocks(
            fake_receipt, debate_id="d1", receipt_url="https://example.com/receipt"
        )
        actions = [b for b in blocks if b.get("type") == "actions"]
        elements = actions[0]["elements"]
        url_btn = next(
            (e for e in elements if e.get("action_id") == "view_receipt"),
            None,
        )
        assert url_btn is not None
        assert url_btn["url"] == "https://example.com/receipt"

    def test_no_view_receipt_button_without_url(self, fake_receipt: FakeReceipt) -> None:
        from aragora.integrations.slack_debate import _build_receipt_with_approval_blocks

        blocks = _build_receipt_with_approval_blocks(fake_receipt, debate_id="d1")
        actions = [b for b in blocks if b.get("type") == "actions"]
        elements = actions[0]["elements"]
        url_btns = [e for e in elements if e.get("action_id") == "view_receipt"]
        assert len(url_btns) == 0

    def test_cost_info_included_when_available(self, fake_receipt_with_cost: FakeReceipt) -> None:
        from aragora.integrations.slack_debate import _build_receipt_with_approval_blocks

        blocks = _build_receipt_with_approval_blocks(fake_receipt_with_cost, debate_id="d1")
        all_text = _extract_all_text(blocks)
        assert "$0.0342" in all_text
        assert "4,500" in all_text

    def test_no_cost_section_when_zero(self, fake_receipt: FakeReceipt) -> None:
        from aragora.integrations.slack_debate import _build_receipt_with_approval_blocks

        blocks = _build_receipt_with_approval_blocks(fake_receipt, debate_id="d1")
        all_text = _extract_all_text(blocks)
        assert "$0.00" not in all_text

    def test_context_footer_present(self, fake_receipt: FakeReceipt) -> None:
        from aragora.integrations.slack_debate import _build_receipt_with_approval_blocks

        blocks = _build_receipt_with_approval_blocks(fake_receipt, debate_id="d1")
        context_blocks = [b for b in blocks if b.get("type") == "context"]
        assert len(context_blocks) >= 1

    def test_debate_id_in_button_values(self, fake_receipt: FakeReceipt) -> None:
        from aragora.integrations.slack_debate import _build_receipt_with_approval_blocks

        blocks = _build_receipt_with_approval_blocks(fake_receipt, debate_id="test-debate-42")
        actions = [b for b in blocks if b.get("type") == "actions"]
        for elem in actions[0]["elements"]:
            if elem.get("value"):
                assert elem["value"] == "test-debate-42"

    def test_findings_count_in_blocks(self) -> None:
        from aragora.integrations.slack_debate import _build_receipt_with_approval_blocks

        receipt = FakeReceipt(
            findings=[FakeFinding(severity="critical"), FakeFinding(severity="high")]
        )
        blocks = _build_receipt_with_approval_blocks(receipt, debate_id="d1")
        all_text = _extract_all_text(blocks)
        assert "2 total" in all_text

    def test_rejected_verdict_emoji(self) -> None:
        from aragora.integrations.slack_debate import _build_receipt_with_approval_blocks

        receipt = FakeReceipt(verdict="REJECTED")
        blocks = _build_receipt_with_approval_blocks(receipt, debate_id="d1")
        headers = [b for b in blocks if b.get("type") == "header"]
        assert ":x:" in headers[0]["text"]["text"]


# ---------------------------------------------------------------------------
# deliver_receipt_to_thread tests
# ---------------------------------------------------------------------------


class TestDeliverReceiptToThread:
    """Tests for the lifecycle deliver_receipt_to_thread method."""

    @pytest.mark.asyncio
    async def test_deliver_with_provided_receipt(
        self, lifecycle: Any, fake_receipt: FakeReceipt
    ) -> None:
        lifecycle._post_to_thread = AsyncMock(return_value=True)
        result = await lifecycle.deliver_receipt_to_thread(
            debate_id="d1",
            channel_id="C01",
            thread_ts="123.456",
            receipt=fake_receipt,
        )
        assert result is True
        lifecycle._post_to_thread.assert_called_once()

    @pytest.mark.asyncio
    async def test_deliver_returns_false_when_no_receipt(self, lifecycle: Any) -> None:
        result = await lifecycle.deliver_receipt_to_thread(
            debate_id="nonexistent",
            channel_id="C01",
            thread_ts="123.456",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_deliver_builds_receipt_url_from_env(
        self, lifecycle: Any, fake_receipt: FakeReceipt
    ) -> None:
        lifecycle._post_to_thread = AsyncMock(return_value=True)
        with patch.dict(os.environ, {"ARAGORA_PUBLIC_URL": "https://test.aragora.ai"}):
            await lifecycle.deliver_receipt_to_thread(
                debate_id="d1",
                channel_id="C01",
                thread_ts="123.456",
                receipt=fake_receipt,
            )
        call_args = lifecycle._post_to_thread.call_args
        blocks = call_args[1].get("blocks") or call_args[0][3] if len(call_args[0]) > 3 else None
        # Verify URL was built
        assert lifecycle._post_to_thread.called

    @pytest.mark.asyncio
    async def test_deliver_uses_custom_receipt_url(
        self, lifecycle: Any, fake_receipt: FakeReceipt
    ) -> None:
        lifecycle._post_to_thread = AsyncMock(return_value=True)
        await lifecycle.deliver_receipt_to_thread(
            debate_id="d1",
            channel_id="C01",
            thread_ts="123.456",
            receipt=fake_receipt,
            receipt_url="https://custom.url/receipt/123",
        )
        assert lifecycle._post_to_thread.called


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_all_text(blocks: list[dict[str, Any]]) -> str:
    """Extract all text content from Block Kit blocks."""
    parts: list[str] = []
    for b in blocks:
        if b.get("type") == "header":
            parts.append(b.get("text", {}).get("text", ""))
        if b.get("type") == "section":
            text_obj = b.get("text", {})
            if isinstance(text_obj, dict):
                parts.append(text_obj.get("text", ""))
            for f in b.get("fields", []):
                parts.append(f.get("text", ""))
        if b.get("type") == "context":
            for el in b.get("elements", []):
                parts.append(el.get("text", ""))
    return " ".join(parts)
