"""Governance test fixtures — isolate external services."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_notification_tokens(monkeypatch):
    """Remove real Slack tokens to prevent external API calls."""
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
