"""Smoke tests for the LiftMode daily briefing pipeline.

Validates that briefing.py builds correct Slack Block Kit messages
from inbox, Shopify orders, and Zendesk tickets data — and that
the setup.sh Python JSON builder works correctly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Insert deploy/liftmode onto path so we can import briefing
LIFTMODE_DIR = Path(__file__).resolve().parents[2] / "deploy" / "liftmode"
sys.path.insert(0, str(LIFTMODE_DIR))

import briefing  # noqa: E402


# ── Fixtures ──────────────────────────────────────────────────────────

SAMPLE_INBOX = {
    "emails": [
        {
            "subject": "Urgent: FDA label update",
            "from": "compliance@fda.gov",
            "priority": "high",
            "category": "regulatory",
        },
        {"subject": "PO #4421 confirmation", "from": "vendor@supplier.com", "priority": "medium"},
        {"subject": "Weekly newsletter", "from": "news@industry.com", "priority": "low"},
    ]
}

SAMPLE_ORDERS = {
    "orders": [
        {
            "id": 1001,
            "total_price": "89.99",
            "fulfillment_status": "unfulfilled",
            "financial_status": "paid",
        },
        {
            "id": 1002,
            "total_price": "149.50",
            "fulfillment_status": "fulfilled",
            "financial_status": "paid",
        },
        {
            "id": 1003,
            "total_price": "34.00",
            "fulfillment_status": "unfulfilled",
            "financial_status": "refunded",
        },
    ]
}

SAMPLE_TICKETS = {
    "tickets": [
        {"id": 501, "priority": "urgent", "assignee_id": 42},
        {"id": 502, "priority": "normal", "assignee_id": None},
        {"id": 503, "priority": "high", "assignee_id": None},
    ],
    "avg_response_time_hours": 2.3,
}


# ── build_briefing tests ─────────────────────────────────────────────


class TestBuildBriefing:
    """Test that build_briefing produces valid Slack Block Kit messages."""

    def test_full_briefing_with_all_sources(self):
        blocks = briefing.build_briefing(SAMPLE_INBOX, orders=SAMPLE_ORDERS, tickets=SAMPLE_TICKETS)
        text = json.dumps(blocks)

        # Header present
        assert any(b["type"] == "header" for b in blocks)

        # Email stats
        assert "3" in text and "emails" in text.lower() or "processed" in text.lower()

        # Shopify section
        assert "Shopify" in text
        assert "revenue" in text.lower() or "$" in text
        assert "unfulfilled" in text.lower()

        # Zendesk section
        assert "Zendesk" in text
        assert "urgent" in text.lower()
        assert "unassigned" in text.lower()
        assert "2.3" in text  # avg response time

        # Action button at end
        assert blocks[-1]["type"] == "actions"

    def test_briefing_inbox_only(self):
        blocks = briefing.build_briefing(SAMPLE_INBOX)
        text = json.dumps(blocks)

        assert any(b["type"] == "header" for b in blocks)
        assert "Shopify" not in text
        assert "Zendesk" not in text
        assert blocks[-1]["type"] == "actions"

    def test_briefing_orders_only(self):
        blocks = briefing.build_briefing({}, orders=SAMPLE_ORDERS)
        text = json.dumps(blocks)

        assert any(b["type"] == "header" for b in blocks)
        assert "Shopify" in text
        assert "Zendesk" not in text

    def test_briefing_tickets_only(self):
        blocks = briefing.build_briefing({}, tickets=SAMPLE_TICKETS)
        text = json.dumps(blocks)

        assert any(b["type"] == "header" for b in blocks)
        assert "Zendesk" in text
        assert "Shopify" not in text

    def test_briefing_empty_inbox(self):
        blocks = briefing.build_briefing({})
        text = json.dumps(blocks)

        assert any(b["type"] == "header" for b in blocks)
        assert "0" in text  # 0 emails processed

    def test_orders_revenue_calculation(self):
        blocks = briefing.build_briefing({}, orders=SAMPLE_ORDERS)
        text = json.dumps(blocks)
        # 89.99 + 149.50 + 34.00 = 273.49
        assert "273.49" in text

    def test_orders_returns_shown(self):
        blocks = briefing.build_briefing({}, orders=SAMPLE_ORDERS)
        text = json.dumps(blocks)
        assert "return" in text.lower()

    def test_tickets_avg_response_time(self):
        blocks = briefing.build_briefing({}, tickets=SAMPLE_TICKETS)
        text = json.dumps(blocks)
        assert "2.3" in text

    def test_tickets_without_avg_response_time(self):
        tickets_no_avg = {"tickets": SAMPLE_TICKETS["tickets"]}
        blocks = briefing.build_briefing({}, tickets=tickets_no_avg)
        text = json.dumps(blocks)
        assert "avg response" not in text.lower()

    def test_all_blocks_are_valid_slack_types(self):
        blocks = briefing.build_briefing(SAMPLE_INBOX, orders=SAMPLE_ORDERS, tickets=SAMPLE_TICKETS)
        valid_types = {"header", "section", "divider", "actions", "context", "image"}
        for block in blocks:
            assert block.get("type") in valid_types, f"Invalid block type: {block.get('type')}"


# ── main() dry-run test ──────────────────────────────────────────────


class TestMainDryRun:
    """Test that main() --dry-run prints valid JSON."""

    def test_dry_run_with_all_sources(self, capsys):
        with (
            patch.object(briefing, "fetch_priority_inbox", return_value=SAMPLE_INBOX),
            patch.object(briefing, "fetch_shopify_orders", return_value=SAMPLE_ORDERS),
            patch.object(briefing, "fetch_zendesk_tickets", return_value=SAMPLE_TICKETS),
        ):
            with patch("sys.argv", ["briefing.py", "--dry-run"]):
                result = briefing.main()

        assert result == 0
        output = capsys.readouterr().out
        blocks = json.loads(output)
        assert isinstance(blocks, list)
        assert len(blocks) > 0

    def test_dry_run_no_data(self, capsys):
        with (
            patch.object(briefing, "fetch_priority_inbox", return_value={}),
            patch.object(briefing, "fetch_shopify_orders", return_value={}),
            patch.object(briefing, "fetch_zendesk_tickets", return_value={}),
        ):
            with patch("sys.argv", ["briefing.py", "--dry-run"]):
                result = briefing.main()

        assert result == 0
        output = capsys.readouterr().out
        blocks = json.loads(output)
        assert any("could not fetch" in json.dumps(b).lower() for b in blocks)


# ── setup.sh JSON builder test ───────────────────────────────────────


class TestSetupJsonBuilder:
    """Test the inline Python JSON builder from setup.sh."""

    def test_json_builder_with_all_args(self):
        """Run the Python snippet from setup.sh with test arguments."""
        script = (
            "import json, sys\n"
            "pairs = list(zip(sys.argv[1::2], sys.argv[2::2]))\n"
            "secret = {}\n"
            "for key, val in pairs:\n"
            "    if val:\n"
            "        secret[key] = val\n"
            "if 'GMAIL_CLIENT_ID' in secret:\n"
            "    secret['GOOGLE_CLIENT_ID'] = secret['GMAIL_CLIENT_ID']\n"
            "if 'GMAIL_CLIENT_SECRET' in secret:\n"
            "    secret['GOOGLE_CLIENT_SECRET'] = secret['GMAIL_CLIENT_SECRET']\n"
            "print(json.dumps(secret))\n"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                "ANTHROPIC_API_KEY",
                "sk-ant-test",
                "ARAGORA_API_TOKEN",
                "tok-123",
                "GMAIL_CLIENT_ID",
                "client.apps.google.com",
                "GMAIL_CLIENT_SECRET",
                "secret123",
                "OPENAI_API_KEY",
                "",  # empty = should be skipped
                "SHOPIFY_SHOP_DOMAIN",
                "liftmode.myshopify.com",
                "SHOPIFY_ACCESS_TOKEN",
                "shpat_test",
                "ZENDESK_SUBDOMAIN",
                "",  # empty = should be skipped
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        secret = json.loads(result.stdout)

        # Required keys present
        assert secret["ANTHROPIC_API_KEY"] == "sk-ant-test"
        assert secret["ARAGORA_API_TOKEN"] == "tok-123"

        # Gmail mirrored to Google
        assert secret["GOOGLE_CLIENT_ID"] == "client.apps.google.com"
        assert secret["GOOGLE_CLIENT_SECRET"] == "secret123"

        # Empty values skipped
        assert "OPENAI_API_KEY" not in secret
        assert "ZENDESK_SUBDOMAIN" not in secret

        # Shopify present
        assert secret["SHOPIFY_SHOP_DOMAIN"] == "liftmode.myshopify.com"
        assert secret["SHOPIFY_ACCESS_TOKEN"] == "shpat_test"

    def test_json_builder_all_empty_optional(self):
        """All optional fields empty — only required keys in output."""
        script = (
            "import json, sys\n"
            "pairs = list(zip(sys.argv[1::2], sys.argv[2::2]))\n"
            "secret = {}\n"
            "for key, val in pairs:\n"
            "    if val:\n"
            "        secret[key] = val\n"
            "if 'GMAIL_CLIENT_ID' in secret:\n"
            "    secret['GOOGLE_CLIENT_ID'] = secret['GMAIL_CLIENT_ID']\n"
            "if 'GMAIL_CLIENT_SECRET' in secret:\n"
            "    secret['GOOGLE_CLIENT_SECRET'] = secret['GMAIL_CLIENT_SECRET']\n"
            "print(json.dumps(secret))\n"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                "ANTHROPIC_API_KEY",
                "sk-ant-test",
                "ARAGORA_API_TOKEN",
                "tok-123",
                "GMAIL_CLIENT_ID",
                "",
                "GMAIL_CLIENT_SECRET",
                "",
                "OPENAI_API_KEY",
                "",
                "SHOPIFY_SHOP_DOMAIN",
                "",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        secret = json.loads(result.stdout)
        assert len(secret) == 2  # only the two non-empty keys
        assert "GOOGLE_CLIENT_ID" not in secret  # not mirrored since Gmail was empty
