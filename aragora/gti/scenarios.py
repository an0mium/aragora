"""Scenario corpus model for the Ground-Truth Integrity benchmark."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

VALID_FAILURE_MODES = frozenset(
    {
        "stale_source",
        "stale_memory",
        "false_green",
        "wrong_taxonomy",
        "historical_as_current",
        "self_aware_stale",
    }
)


@dataclass(frozen=True)
class Scenario:
    """One labeled ground-truth-integrity scenario.

    ``belief_matches_truth`` is False for stale/wrong beliefs (the interesting
    cases) and True for control scenarios (fresh + correct). ``belief_age_days``
    and ``freshness_ttl_days`` drive the freshness gate; ``quorum_would_flag``
    models whether >=2 heterogeneous model families would dispute the belief.
    """

    id: str
    failure_mode: str
    belief_presented: str
    ground_truth: str
    canonical_source: str
    belief_matches_truth: bool
    belief_age_days: float
    freshness_ttl_days: float
    quorum_would_flag: bool
    expected: str  # "detect" | "correct" | "halt"
    consequential_action_if_wrong: str

    def __post_init__(self) -> None:
        if self.failure_mode not in VALID_FAILURE_MODES:
            raise ValueError(f"invalid failure_mode: {self.failure_mode!r}")
        if self.expected not in {"detect", "correct", "halt"}:
            raise ValueError(f"invalid expected: {self.expected!r}")
        if not isinstance(self.belief_matches_truth, bool):
            raise ValueError(f"invalid belief_matches_truth: {self.belief_matches_truth!r}")
        if not isinstance(self.quorum_would_flag, bool):
            raise ValueError(f"invalid quorum_would_flag: {self.quorum_would_flag!r}")
        if (
            isinstance(self.belief_age_days, bool)
            or not isinstance(self.belief_age_days, (int, float))
            or not math.isfinite(self.belief_age_days)
            or self.belief_age_days < 0
        ):
            raise ValueError(f"invalid belief_age_days: {self.belief_age_days!r}")
        if (
            isinstance(self.freshness_ttl_days, bool)
            or not isinstance(self.freshness_ttl_days, (int, float))
            or not math.isfinite(self.freshness_ttl_days)
            or self.freshness_ttl_days <= 0
        ):
            raise ValueError(f"invalid freshness_ttl_days: {self.freshness_ttl_days!r}")


def load_scenarios(path: Path) -> list[Scenario]:
    """Load and validate the scenario corpus from JSON."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("scenarios"), list):
        raise ValueError("scenario corpus must contain a scenarios list")

    scenarios: list[Scenario] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(raw["scenarios"]):
        if not isinstance(entry, dict):
            raise ValueError(f"scenario entry {index} must be an object")
        scenario = Scenario(**entry)
        if scenario.id in seen_ids:
            raise ValueError(f"duplicate scenario id: {scenario.id}")
        seen_ids.add(scenario.id)
        scenarios.append(scenario)
    return scenarios
