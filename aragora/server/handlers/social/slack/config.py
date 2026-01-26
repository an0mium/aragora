"""
Slack handler configuration.

Provides environment variable configuration and constants for Slack integration.
"""

from __future__ import annotations

import os
import re

# Environment variables for Slack integration (fallback for single-workspace mode)
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET", "")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")

# Base URL for internal API calls (configurable for production)
ARAGORA_API_BASE_URL = os.environ.get("ARAGORA_API_BASE_URL", "http://localhost:8080")

# Configurable workspace rate limit (requests per minute)
SLACK_WORKSPACE_RATE_LIMIT_RPM = int(os.environ.get("SLACK_WORKSPACE_RATE_LIMIT_RPM", "30"))

# Patterns for command parsing
COMMAND_PATTERN = re.compile(r"^/aragora\s+(\w+)(?:\s+(.*))?$")
TOPIC_PATTERN = re.compile(r'^["\']?(.+?)["\']?$')

# Routes handled by SlackHandler
SLACK_ROUTES = [
    "/api/v1/integrations/slack/commands",
    "/api/v1/integrations/slack/interactive",
    "/api/v1/integrations/slack/events",
    "/api/v1/integrations/slack/status",
]
