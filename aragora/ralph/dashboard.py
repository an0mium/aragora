"""Ralph campaign dashboard data service.

Reads supervisor YAML state files and exposes structured data for the
observability dashboard: campaign timelines, blocker breakdowns, repair
stats, budget burn, and PR gate status.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

from aragora.ralph.classifier import BlockerKind
from aragora.ralph.supervisor import SupervisorState, load_supervisor_state

logger = logging.getLogger(__name__)

_DEFAULT_STATE_DIR = Path.home() / ".aragora" / "ralph"


class RalphDashboard:
    """Read-only dashboard over Ralph supervisor state files."""

    def __init__(self, state_dir: Path | None = None) -> None:
        self.state_dir = state_dir or _DEFAULT_STATE_DIR

    def list_campaigns(self) -> list[dict[str, Any]]:
        """List all campaigns with summary info."""
        campaigns: list[dict[str, Any]] = []
        if not self.state_dir.exists():
            return campaigns
        for path in sorted(self.state_dir.glob("*.yaml")):
            if path.name.endswith(".yaml.tmp"):
                continue
            try:
                state = load_supervisor_state(path)
                campaigns.append(
                    {
                        "campaign_id": state.campaign_id or path.stem,
                        "supervisor_id": state.supervisor_id,
                        "status": state.status,
                        "current_step": state.current_step,
                        "budget_spent_usd": state.budget_spent_usd,
                        "repair_attempts": state.repair_attempts,
                        "blocker_count": len(state.blocker_history),
                        "updated_at": state.updated_at,
                        "state_path": str(path),
                    }
                )
            except Exception:
                logger.debug("Failed to load %s", path, exc_info=True)
        return campaigns

    def get_campaign_detail(self, campaign_id: str) -> dict[str, Any] | None:
        """Get full detail for a campaign."""
        state = self._find_campaign(campaign_id)
        if state is None:
            return None
        return state.to_dict()

    def get_campaign_timeline(self, campaign_id: str) -> list[dict[str, Any]]:
        """Get the step-by-step timeline for a campaign.

        Derives timeline from blocker_history entries and current state.
        """
        state = self._find_campaign(campaign_id)
        if state is None:
            return []
        timeline: list[dict[str, Any]] = []
        for entry in state.blocker_history:
            timeline.append(
                {
                    "step": entry.get("step", 0),
                    "event": "blocker_classified",
                    "kind": entry.get("kind", "unknown"),
                    "stop_reason": entry.get("stop_reason", ""),
                    "detail": entry.get("detail", ""),
                }
            )
        timeline.append(
            {
                "step": state.current_step,
                "event": "current_state",
                "status": state.status,
                "active_blocker": state.active_blocker,
                "budget_spent_usd": state.budget_spent_usd,
            }
        )
        return timeline

    def get_blocker_breakdown(self, campaign_id: str | None = None) -> dict[str, Any]:
        """Aggregate blocker classifications by kind.

        If campaign_id is None, aggregates across all campaigns.
        """
        histories: list[list[dict[str, Any]]] = []
        if campaign_id:
            state = self._find_campaign(campaign_id)
            if state:
                histories.append(state.blocker_history)
        else:
            for campaign in self.list_campaigns():
                state = self._find_campaign(campaign["campaign_id"])
                if state:
                    histories.append(state.blocker_history)

        kind_counts: Counter[str] = Counter()
        deterministic_count = 0
        escalation_count = 0
        for history in histories:
            for entry in history:
                kind_str = entry.get("kind", "unknown")
                kind_counts[kind_str] += 1
                try:
                    bk = BlockerKind(kind_str)
                    if bk.is_deterministic:
                        deterministic_count += 1
                    else:
                        escalation_count += 1
                except ValueError:
                    escalation_count += 1

        return {
            "by_kind": dict(kind_counts.most_common()),
            "deterministic_total": deterministic_count,
            "escalation_total": escalation_count,
            "total": sum(kind_counts.values()),
        }

    def get_repair_stats(self, campaign_id: str | None = None) -> dict[str, Any]:
        """Get repair attempt stats."""
        states = self._collect_states(campaign_id)
        total_attempts = sum(s.repair_attempts for s in states)
        completed = sum(1 for s in states if s.status in ("completed", "waiting_for_merge"))
        escalated = sum(1 for s in states if s.status == "escalated")
        return {
            "total_attempts": total_attempts,
            "campaigns_completed": completed,
            "campaigns_escalated": escalated,
            "campaigns_total": len(states),
        }

    def get_budget_summary(self, campaign_id: str | None = None) -> dict[str, Any]:
        """Get budget burn summary."""
        states = self._collect_states(campaign_id)
        total_spent = sum(s.budget_spent_usd for s in states)
        per_campaign = [
            {
                "campaign_id": s.campaign_id or s.supervisor_id,
                "budget_spent_usd": s.budget_spent_usd,
                "status": s.status,
            }
            for s in states
        ]
        return {
            "total_spent_usd": round(total_spent, 2),
            "per_campaign": per_campaign,
        }

    def get_pr_gate_status(self, campaign_id: str) -> dict[str, Any] | None:
        """Get the current PR merge gate state for a campaign."""
        state = self._find_campaign(campaign_id)
        if state is None:
            return None
        target = state.active_merge_target
        if not isinstance(target, dict):
            return {
                "has_active_pr": False,
                "pr_url": state.active_repair_pr,
                "branch": state.active_repair_branch,
            }
        gate = target.get("last_gate_snapshot", {})
        return {
            "has_active_pr": True,
            "pr_url": target.get("pr_url", state.active_repair_pr),
            "branch": target.get("branch", state.active_repair_branch),
            "disposition": gate.get("disposition", "unknown"),
            "checks_passed": gate.get("checks_passed", []),
            "checks_pending": gate.get("checks_pending", []),
            "checks_failed": gate.get("checks_failed", []),
            "review_decision": gate.get("review_decision"),
        }

    def get_overview(self) -> dict[str, Any]:
        """Aggregate overview across all campaigns."""
        campaigns = self.list_campaigns()
        status_counts: Counter[str] = Counter()
        for c in campaigns:
            status_counts[c["status"]] += 1
        return {
            "total_campaigns": len(campaigns),
            "by_status": dict(status_counts),
            "blockers": self.get_blocker_breakdown(),
            "repairs": self.get_repair_stats(),
            "budget": self.get_budget_summary(),
        }

    # --- internal helpers ---

    def _find_campaign(self, campaign_id: str) -> SupervisorState | None:
        """Find a campaign by ID, searching state files."""
        if not self.state_dir.exists():
            return None
        # Try direct file match first
        direct = self.state_dir / f"{campaign_id}.yaml"
        if direct.exists():
            try:
                return load_supervisor_state(direct)
            except Exception:
                pass
        # Search all state files
        for path in self.state_dir.glob("*.yaml"):
            if path.name.endswith(".yaml.tmp"):
                continue
            try:
                state = load_supervisor_state(path)
                if state.campaign_id == campaign_id or state.supervisor_id == campaign_id:
                    return state
            except Exception:
                continue
        return None

    def _collect_states(self, campaign_id: str | None = None) -> list[SupervisorState]:
        """Load states for one or all campaigns."""
        if campaign_id:
            state = self._find_campaign(campaign_id)
            return [state] if state else []
        states: list[SupervisorState] = []
        if not self.state_dir.exists():
            return states
        for path in sorted(self.state_dir.glob("*.yaml")):
            if path.name.endswith(".yaml.tmp"):
                continue
            try:
                states.append(load_supervisor_state(path))
            except Exception:
                continue
        return states
