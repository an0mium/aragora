"""Tests for Inbox Command Center namespace API.

Tests cover:
- get_inbox() - fetch prioritized inbox with pagination and filters
- quick_action() - execute quick actions on emails
- bulk_action() - execute bulk actions based on filters
- get_sender_profile() - get sender profile information
- get_daily_digest() - get daily digest statistics
- reprioritize() - trigger AI re-prioritization
"""

from __future__ import annotations
