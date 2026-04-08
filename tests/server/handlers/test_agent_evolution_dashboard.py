from __future__ import annotations

import json
from unittest.mock import patch

from aragora.server.handlers.agent_evolution_dashboard import AgentEvolutionDashboardHandler


def _payload(result) -> dict:
    return json.loads(result.body.decode("utf-8"))["data"]


def test_pending_changes_fail_closed_without_live_store() -> None:
    handler = AgentEvolutionDashboardHandler({})

    result = handler.handle("/api/v1/agent-evolution/pending", {}, None)

    assert result is not None
    assert result.status_code == 200
    assert _payload(result) == {"changes": [], "total_pending": 0}


def test_pending_changes_preserve_live_store_data_when_available() -> None:
    handler = AgentEvolutionDashboardHandler({})
    live_data = {
        "changes": [
            {
                "id": "pc-live-1",
                "agent_name": "claude-3-opus",
                "change_type": "prompt_rewrite",
                "nomic_cycle_id": "nomic-101",
                "proposed_at": "2026-04-07T10:00:00+00:00",
                "proposed_by": "nomic-loop",
                "description": "Tighten synthesis prompt",
                "diff_summary": "Prompt v5 -> v6",
                "old_content": "before",
                "new_content": "after",
                "impact_estimate": "Expected +2% clarity",
                "status": "pending",
            }
        ],
        "total_pending": 1,
    }

    with patch.object(handler, "_fetch_real_pending", return_value=live_data):
        result = handler.handle("/api/v1/agent-evolution/pending", {}, None)

    assert result is not None
    assert result.status_code == 200
    assert _payload(result) == live_data
