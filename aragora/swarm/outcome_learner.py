"""Rolling outcome learner for boss-loop and swarm feedback.

Consumes OutcomeSignal records and emits deterministic snapshots that summarize
success/failure/merge/rescue rates by loop and agent. These summaries are
intended to be stable and machine-readable for downstream dashboards or
automations.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from aragora.swarm.outcome_signals import OutcomeSignal

UTC = timezone.utc
_DEFAULT_OUTCOME_SIGNAL_LOG = Path("~/.aragora/outcome_signals.jsonl").expanduser()


@dataclass
class OutcomeAggregate:
    total: int = 0
    successes: int = 0
    failures: int = 0
    merge_count: int = 0
    rescue_count: int = 0
    failure_reasons: dict[str, int] = field(default_factory=dict)
    blocker_kinds: dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.successes / self.total if self.total else 0.0

    @property
    def merge_rate(self) -> float:
        return self.merge_count / self.total if self.total else 0.0

    @property
    def rescue_rate(self) -> float:
        return self.rescue_count / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["success_rate"] = self.success_rate
        data["merge_rate"] = self.merge_rate
        data["rescue_rate"] = self.rescue_rate
        return data


@dataclass
class OutcomeLearnerSnapshot:
    timestamp: str
    window_size: int
    total_signals: int
    by_loop: dict[str, OutcomeAggregate]
    by_agent: dict[str, OutcomeAggregate]
    failure_taxonomy: dict[str, int]
    blocker_kinds: dict[str, int]
    recommendations: list[str]
    routing_hints: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "window_size": self.window_size,
            "total_signals": self.total_signals,
            "by_loop": {k: v.to_dict() for k, v in self.by_loop.items()},
            "by_agent": {k: v.to_dict() for k, v in self.by_agent.items()},
            "failure_taxonomy": dict(self.failure_taxonomy),
            "blocker_kinds": dict(self.blocker_kinds),
            "recommendations": list(self.recommendations),
            "routing_hints": dict(self.routing_hints),
        }


class OutcomeLearner:
    """Rolling learner that ingests outcome signals and emits snapshots."""

    def __init__(
        self,
        *,
        window_size: int = 500,
        min_samples: int = 3,
        merge_rate_threshold: float = 0.3,
        rescue_rate_threshold: float = 0.5,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self._window_size = int(window_size)
        self._min_samples = int(min_samples)
        self._merge_rate_threshold = float(merge_rate_threshold)
        self._rescue_rate_threshold = float(rescue_rate_threshold)
        self._signals: deque[OutcomeSignal] = deque(maxlen=self._window_size)

    def ingest(self, signal: OutcomeSignal) -> None:
        if not isinstance(signal, OutcomeSignal):
            raise TypeError("OutcomeLearner only accepts OutcomeSignal instances")
        self._signals.append(signal)

    def ingest_many(self, signals: Iterable[OutcomeSignal]) -> None:
        for signal in signals:
            self.ingest(signal)

    def snapshot(self) -> OutcomeLearnerSnapshot:
        signals = list(self._signals)
        by_loop: dict[str, OutcomeAggregate] = {}
        by_agent: dict[str, OutcomeAggregate] = {}
        by_category: dict[str, OutcomeAggregate] = {}
        failure_reasons: Counter[str] = Counter()
        blocker_kinds: Counter[str] = Counter()

        for signal in signals:
            loop_key = signal.source_loop or "unknown"
            agent_key = signal.agent_type or "unknown"
            loop_agg = by_loop.setdefault(loop_key, OutcomeAggregate())
            agent_agg = by_agent.setdefault(agent_key, OutcomeAggregate())
            for agg in (loop_agg, agent_agg):
                agg.total += 1
                if signal.is_success:
                    agg.successes += 1
                if signal.is_failure:
                    agg.failures += 1
                if signal.did_merge:
                    agg.merge_count += 1
                if signal.needed_human_rescue:
                    agg.rescue_count += 1
                if signal.failure_reason:
                    agg.failure_reasons[signal.failure_reason] = (
                        agg.failure_reasons.get(signal.failure_reason, 0) + 1
                    )
                if signal.blocker_kind:
                    agg.blocker_kinds[signal.blocker_kind] = (
                        agg.blocker_kinds.get(signal.blocker_kind, 0) + 1
                    )

            category_key = _infer_signal_category(signal)
            if category_key:
                category_agg = by_category.setdefault(category_key, OutcomeAggregate())
                category_agg.total += 1
                if signal.is_success:
                    category_agg.successes += 1
                if signal.is_failure:
                    category_agg.failures += 1
                if signal.did_merge:
                    category_agg.merge_count += 1
                if signal.needed_human_rescue:
                    category_agg.rescue_count += 1
                if signal.failure_reason:
                    category_agg.failure_reasons[signal.failure_reason] = (
                        category_agg.failure_reasons.get(signal.failure_reason, 0) + 1
                    )
                if signal.blocker_kind:
                    category_agg.blocker_kinds[signal.blocker_kind] = (
                        category_agg.blocker_kinds.get(signal.blocker_kind, 0) + 1
                    )

            if signal.failure_reason:
                failure_reasons[signal.failure_reason] += 1
            if signal.blocker_kind:
                blocker_kinds[signal.blocker_kind] += 1

        recommendations = self._recommendations(by_loop, by_agent, failure_reasons, blocker_kinds)
        routing_hints = self._routing_hints(by_loop, by_agent, by_category)

        return OutcomeLearnerSnapshot(
            timestamp=datetime.now(UTC).isoformat(),
            window_size=self._window_size,
            total_signals=len(signals),
            by_loop=self._sorted_aggregates(by_loop),
            by_agent=self._sorted_aggregates(by_agent),
            failure_taxonomy=dict(sorted(failure_reasons.items())),
            blocker_kinds=dict(sorted(blocker_kinds.items())),
            recommendations=recommendations,
            routing_hints=routing_hints,
        )

    def _sorted_aggregates(
        self, aggregates: dict[str, OutcomeAggregate]
    ) -> dict[str, OutcomeAggregate]:
        return {k: aggregates[k] for k in sorted(aggregates.keys())}

    def _recommendations(
        self,
        by_loop: dict[str, OutcomeAggregate],
        by_agent: dict[str, OutcomeAggregate],
        failure_reasons: Counter[str],
        blocker_kinds: Counter[str],
    ) -> list[str]:
        recs: list[str] = []
        min_samples = self._min_samples

        for loop, agg in sorted(by_loop.items()):
            if agg.total < min_samples:
                continue
            if agg.merge_rate < self._merge_rate_threshold:
                recs.append(
                    f"Loop '{loop}' merge rate is {agg.merge_rate:.0%} "
                    f"({agg.merge_count}/{agg.total}); consider routing fewer tasks or "
                    "hardening the dispatch path."
                )
            if agg.rescue_rate > self._rescue_rate_threshold:
                recs.append(
                    f"Loop '{loop}' rescue rate is {agg.rescue_rate:.0%}; "
                    "tighten feasibility checks or add earlier exits."
                )

        for agent, agg in sorted(by_agent.items()):
            if agg.total < min_samples:
                continue
            if agg.success_rate < self._merge_rate_threshold:
                recs.append(
                    f"Agent '{agent}' success rate is {agg.success_rate:.0%}; "
                    "reduce assignment or refresh prompts."
                )

        for reason, count in sorted(failure_reasons.items()):
            if count >= min_samples:
                recs.append(f"Failure reason '{reason}' observed {count} times; add a guardrail.")

        for kind, count in sorted(blocker_kinds.items()):
            if count >= min_samples:
                recs.append(f"Blocker '{kind}' observed {count} times; add a preventive check.")

        return recs

    def _routing_hints(
        self,
        by_loop: dict[str, OutcomeAggregate],
        by_agent: dict[str, OutcomeAggregate],
        by_category: dict[str, OutcomeAggregate],
    ) -> dict[str, Any]:
        deprioritize_loops: list[str] = []
        deprioritize_agents: list[str] = []
        deprioritize_categories: list[str] = []
        category_success_rates: dict[str, float] = {}
        for loop, agg in sorted(by_loop.items()):
            if agg.total >= self._min_samples and agg.merge_rate < self._merge_rate_threshold:
                deprioritize_loops.append(loop)
        for agent, agg in sorted(by_agent.items()):
            if agg.total >= self._min_samples and agg.success_rate < self._merge_rate_threshold:
                deprioritize_agents.append(agent)
        for category, agg in sorted(by_category.items()):
            if agg.total < self._min_samples:
                continue
            category_success_rates[category] = agg.success_rate
            if agg.success_rate < self._merge_rate_threshold:
                deprioritize_categories.append(category)
        return {
            "deprioritize_loops": deprioritize_loops,
            "deprioritize_agents": deprioritize_agents,
            "deprioritize_categories": deprioritize_categories,
            "category_success_rates": category_success_rates,
        }


def _infer_signal_category(signal: OutcomeSignal) -> str | None:
    title = str(signal.entity_title or "").strip()
    if not title:
        return None
    from aragora.swarm.issue_scanner import infer_issue_category_from_title

    return infer_issue_category_from_title(title)


def _parse_signal_row(data: dict[str, Any]) -> OutcomeSignal | None:
    if not isinstance(data, dict):
        return None
    fields = OutcomeSignal.__dataclass_fields__.keys()
    try:
        return OutcomeSignal(**{key: value for key, value in data.items() if key in fields})
    except (TypeError, ValueError):
        return None


def load_category_success_rates(
    *,
    log_path: Path | None = None,
    window_size: int = 500,
    min_samples: int = 3,
) -> dict[str, float]:
    """Load deterministic per-category success rates from the outcome signal log."""
    path = log_path or _DEFAULT_OUTCOME_SIGNAL_LOG
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")
    if not path.exists():
        return {}

    rows: deque[OutcomeSignal] = deque(maxlen=window_size)
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                signal = _parse_signal_row(payload)
                if signal is not None:
                    rows.append(signal)
    except OSError:
        return {}

    if not rows:
        return {}

    learner = OutcomeLearner(window_size=window_size, min_samples=min_samples)
    learner.ingest_many(rows)
    snapshot = learner.snapshot()
    hints = snapshot.routing_hints.get("category_success_rates", {})
    return {
        str(category): float(rate)
        for category, rate in sorted(hints.items())
        if isinstance(category, str)
    }
