from __future__ import annotations

import asyncio
import os
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

os.environ.setdefault("ARAGORA_USE_SECRETS_MANAGER", "0")

from aragora.server.fastapi.routes.analytics import (
    DayGranularityEnum,
    get_consensus_rates,
    get_deliberation_by_channel,
    get_deliberation_performance,
    get_deliberation_summary,
)


class _SlowDebateStore:
    def get_deliberation_stats(self, **kwargs):
        time.sleep(0.2)
        return {
            "total": 5,
            "completed": 4,
            "in_progress": 1,
            "failed": 0,
            "consensus_reached": 3,
            "avg_rounds": 2.5,
            "avg_duration_seconds": 12.3,
            "by_template": {"default": 5},
            "by_priority": {"medium": 5},
        }

    def get_deliberation_stats_by_channel(self, **kwargs):
        time.sleep(0.2)
        return [
            {
                "platform": "slack",
                "channel": "alerts",
                "total_deliberations": 3,
                "consensus_reached": 2,
                "total_duration": 36,
            }
        ]

    def get_consensus_stats(self, **kwargs):
        time.sleep(0.2)
        return {
            "overall_consensus_rate": "75%",
            "by_team_size": {"3": "75%"},
            "by_agent": [{"agent": "claude", "consensus_rate": "80%"}],
            "top_teams": [{"team": ["claude", "openai", "gemini"], "rate": "75%"}],
        }

    def get_deliberation_performance(self, **kwargs):
        time.sleep(0.2)
        return {
            "summary": {"avg_duration_seconds": 12.3},
            "by_template": [{"template": "default", "avg_duration_seconds": 12.3}],
            "trends": [{"bucket": "2026-03-26", "avg_duration_seconds": 12.3}],
            "cost_by_agent": {"claude": "1.23"},
        }


async def _ticker(duration: float = 0.35) -> int:
    ticks = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration:
        await asyncio.sleep(0.01)
        ticks += 1
    return ticks


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call", "assert_response"),
    [
        (
            lambda auth: get_deliberation_summary(org_id="org-1", days=30, auth=auth),
            lambda response: response.total_deliberations == 5,
        ),
        (
            lambda auth: get_deliberation_by_channel(org_id="org-1", days=30, auth=auth),
            lambda response: response.channels[0]["platform"] == "slack",
        ),
        (
            lambda auth: get_consensus_rates(org_id="org-1", days=30, auth=auth),
            lambda response: response.overall_consensus_rate == "75%",
        ),
        (
            lambda auth: get_deliberation_performance(
                org_id="org-1",
                days=30,
                granularity=DayGranularityEnum.day,
                auth=auth,
            ),
            lambda response: response.summary["avg_duration_seconds"] == 12.3,
        ),
    ],
)
async def test_deliberation_analytics_routes_do_not_block_event_loop_for_sync_store(
    call,
    assert_response,
) -> None:
    auth = SimpleNamespace(user_id="user-1", email="user@example.com")
    task = asyncio.create_task(_ticker())
    await asyncio.sleep(0)

    with patch("aragora.memory.debate_store.get_debate_store", return_value=_SlowDebateStore()):
        response = await call(auth)

    ticks = await task

    assert assert_response(response)
    assert ticks >= 10
