#!/usr/bin/env python3
"""
LiftMode Daily Briefing — Posts operations summary to Slack.

Queries Aragora's Gmail priority inbox and posts a morning briefing
to Slack with action items, urgent emails, and daily stats.

Usage:
    python briefing.py              # Post daily briefing
    python briefing.py --test       # Post test message
    python briefing.py --dry-run    # Print without posting

Schedule via cron (7 AM daily):
    0 7 * * * cd /path/to/deploy/liftmode && python briefing.py

Or via macOS launchd (see setup.sh for plist generation).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from urllib.error import URLError
from urllib.request import Request, urlopen

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("liftmode.briefing")

ARAGORA_URL = os.environ.get("ARAGORA_URL", "http://localhost:8080")
ARAGORA_TOKEN = os.environ.get("ARAGORA_API_TOKEN", "")
SLACK_WEBHOOK = os.environ.get("SLACK_WEBHOOK_URL", "")
SLACK_CHANNEL = os.environ.get("SLACK_CHANNEL", "#ops")


def aragora_get(path: str) -> dict:
    """GET request to Aragora API."""
    url = f"{ARAGORA_URL}{path}"
    req = Request(url, headers={
        "Authorization": f"Bearer {ARAGORA_TOKEN}",
        "Content-Type": "application/json",
    })
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except (URLError, TimeoutError) as exc:
        logger.warning("Aragora API call failed: %s — %s", path, exc)
        return {}


def fetch_priority_inbox() -> dict:
    """Fetch prioritized inbox from Aragora."""
    return aragora_get("/api/v1/gmail/inbox/priority")


def build_briefing(inbox: dict) -> list[dict]:
    """Build Slack Block Kit message from inbox data."""
    now = datetime.now(timezone.utc).strftime("%A, %B %d %Y")
    emails = inbox.get("emails", inbox.get("messages", []))

    high = [e for e in emails if e.get("priority") == "high"]
    medium = [e for e in emails if e.get("priority") == "medium"]
    low = [e for e in emails if e.get("priority") == "low"]

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"LiftMode Daily Briefing — {now}"},
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*{len(emails)}* emails processed | "
                    f"*{len(high)}* urgent | "
                    f"*{len(medium)}* medium | "
                    f"*{len(low)}* low priority"
                ),
            },
        },
    ]

    if high:
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*Urgent — needs attention:*"},
        })
        for email in high[:10]:
            sender = email.get("from", email.get("sender", "Unknown"))
            subject = email.get("subject", "(no subject)")
            category = email.get("category", "")
            tag = f" `{category}`" if category else ""
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"• *{subject}*\n  From: {sender}{tag}",
                },
            })

    if medium:
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Medium priority:* {len(medium)} emails (vendors, inventory, accounts)",
            },
        })

    if low:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Low priority:* {len(low)} emails (newsletters, notifications)",
            },
        })

    blocks.append({"type": "divider"})
    blocks.append({
        "type": "actions",
        "elements": [
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "Open Aragora"},
                "url": f"{ARAGORA_URL}",
            },
        ],
    })

    return blocks


def post_to_slack(blocks: list[dict], text: str = "LiftMode Daily Briefing") -> bool:
    """Post message to Slack via webhook."""
    if not SLACK_WEBHOOK:
        logger.error("SLACK_WEBHOOK_URL not set")
        return False

    payload = json.dumps({
        "channel": SLACK_CHANNEL,
        "text": text,
        "blocks": blocks,
    }).encode()

    req = Request(SLACK_WEBHOOK, data=payload, headers={
        "Content-Type": "application/json",
    })
    try:
        with urlopen(req, timeout=15) as resp:
            if resp.status == 200:
                logger.info("Briefing posted to Slack")
                return True
            logger.warning("Slack returned status %d", resp.status)
            return False
    except (URLError, TimeoutError) as exc:
        logger.error("Failed to post to Slack: %s", exc)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="LiftMode daily briefing")
    parser.add_argument("--test", action="store_true", help="Post a test message")
    parser.add_argument("--dry-run", action="store_true", help="Print without posting")
    args = parser.parse_args()

    if args.test:
        blocks = [{
            "type": "section",
            "text": {"type": "mrkdwn", "text": "Test briefing from Aragora. Connection OK."},
        }]
        if args.dry_run:
            print(json.dumps(blocks, indent=2))
            return 0
        return 0 if post_to_slack(blocks, "Test briefing") else 1

    inbox = fetch_priority_inbox()
    if not inbox:
        logger.warning("No inbox data — Aragora may not be running or Gmail not synced")
        # Post a minimal briefing anyway so Slack shows something
        blocks = [{
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Could not fetch inbox data. Check that Aragora is running and Gmail is synced.",
            },
        }]
    else:
        blocks = build_briefing(inbox)

    if args.dry_run:
        print(json.dumps(blocks, indent=2))
        return 0

    return 0 if post_to_slack(blocks) else 1


if __name__ == "__main__":
    sys.exit(main())
